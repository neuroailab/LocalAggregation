from __future__ import division, print_function, absolute_import
import os, sys
import json
import numpy as np
import tensorflow as tf
import copy
import pdb
from collections import OrderedDict

from . import resnet_model
from . import alexnet_model
from . import vggnet_model
from .memory_bank import MemoryBank
from .prep_utils import ColorNormalize

DATA_LEN_IMAGENET_FULL = 1281167


def get_global_step_var():
    global_step_vars = [v for v in tf.global_variables() \
                        if 'global_step' in v.name]
    assert len(global_step_vars) == 1
    return global_step_vars[0]


def assert_shape(t, shape):
    assert t.get_shape().as_list() == shape, \
            "Got shape %r, expected %r" % (t.get_shape().as_list(), shape)


def flatten(layer_out):
    curr_shape = layer_out.get_shape().as_list()
    if len(curr_shape) > 2:
        layer_out = tf.reshape(layer_out, [curr_shape[0], -1])
    return layer_out


def get_alexnet_all_layers(all_layers, get_all_layers):
    if get_all_layers == 'default' or get_all_layers is None:
        keys = ['pool1', 'pool2', 'conv3', 'conv4', 'pool5', 'fc6', 'fc7']
    elif get_all_layers == 'conv_all':
        keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
    elif get_all_layers == 'conv5-avg':
        keys = ['conv5']
    else:
        keys = get_all_layers.split(',')

    output_dict = OrderedDict()
    for each_key in keys:
        for layer_name, layer_out in all_layers.items():
            if each_key in layer_name:
                if get_all_layers == 'conv5-avg':
                    layer_out = tf.nn.avg_pool(
                            layer_out,
                            ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
                layer_out = flatten(layer_out)
                output_dict[each_key] = layer_out
                break
    return output_dict


def get_resnet_all_layers(ending_points, get_all_layers):
    ending_dict = {}
    get_all_layers = get_all_layers.split(',')
    for idx, layer_out in enumerate(ending_points):
        if str(idx) in get_all_layers:
            feat_size = np.prod(layer_out.get_shape().as_list()[1:])
            if feat_size > 200000:
                pool_size = 2
                if feat_size / 4 > 200000:
                    pool_size = 4
                layer_out = tf.transpose(layer_out, [0, 2, 3, 1])
                layer_out = tf.nn.avg_pool(
                        layer_out,
                        ksize=[1,pool_size,pool_size,1],
                        strides=[1,pool_size,pool_size,1],
                        padding='SAME')
            layer_out = flatten(layer_out)
            ending_dict[str(idx)] = layer_out
    return ending_dict


def network_embedding(
        img_batch, 
        model_type,
        dtype=tf.float32,
        data_format=None, train=False,
        resnet_version=resnet_model.DEFAULT_VERSION,
        get_all_layers=None,
        skip_final_dense=False):
    image = tf.cast(img_batch, tf.float32)
    image = tf.div(image, tf.constant(255, dtype=tf.float32))
    image = tf.map_fn(ColorNormalize, image)

    if model_type.startswith('resnet'):
        resnet_size = int(model_type[6:])
        model = resnet_model.ImagenetModel(
            resnet_size, data_format,
            resnet_version=resnet_version,
            dtype=dtype)

        if skip_final_dense and get_all_layers is None:
            return model(image, train, skip_final_dense=True)

        if get_all_layers:
            _, ending_points = model(
                    image, train, get_all_layers=get_all_layers)
            all_layers = get_resnet_all_layers(ending_points, get_all_layers)
            return all_layers

        model_out = model(image, train, skip_final_dense=False)
    elif model_type == 'alexnet_bn_no_drop':
        model_out, all_layers = alexnet_model.alexnet_v2_with_bn_no_drop(
                image, is_training=train,
                num_classes=128)
        if get_all_layers or skip_final_dense:
            all_layers = get_alexnet_all_layers(all_layers, get_all_layers)
            if get_all_layers:
                return all_layers
            else:
                return all_layers['fc6']
    elif model_type == 'vggnet_fx':
        model_out, ending_points = vggnet_model.vgg_16(
                image, is_training=train,
                num_classes=128, with_bn=True,
                fix_bug=True,
                dropout_keep_prob=0)
        if get_all_layers:
            all_layers = get_resnet_all_layers(ending_points, get_all_layers)
            return all_layers
    else:
        raise ValueError('Model type not supported!')

    return tf.nn.l2_normalize(model_out, axis=1) # [bs, out_dim]


def repeat_1d_tensor(t, num_reps):
    ret = tf.tile(tf.expand_dims(t, axis=1), (1, num_reps))
    return ret


class LossBuilder(object):
    def __init__(self,
                 inputs, output,
                 memory_bank,
                 instance_k=4096,
                 instance_t=0.07,
                 instance_m=0.5,
                 **kwargs):
        self.inputs = inputs
        self.embed_output = output
        self.batch_size, self.out_dim = self.embed_output.get_shape().as_list()
        self.memory_bank = memory_bank

        self.data_len = memory_bank.size
        self.instance_k = instance_k
        self.instance_t = instance_t
        self.instance_m = instance_m

    def _softmax(self, dot_prods):
        instance_Z = tf.constant(
            2876934.2 / 1281167 * self.data_len,
            dtype=tf.float32)
        return tf.exp(dot_prods / self.instance_t) / instance_Z

    def compute_data_prob(self):
        logits = self.memory_bank.get_dot_products(
                self.embed_output, 
                self.inputs['index'])
        return self._softmax(logits)

    def compute_noise_prob(self):
        noise_indx = tf.random_uniform(
            shape=(self.batch_size, self.instance_k),
            minval=0,
            maxval=self.data_len,
            dtype=tf.int64)
        noise_probs = self._softmax(
            self.memory_bank.get_dot_products(self.embed_output, noise_indx))
        return noise_probs

    def updated_new_data_memory(self):
        data_indx = self.inputs['index'] # [bs]
        data_memory = self.memory_bank.at_idxs(data_indx)
        new_data_memory = (data_memory * self.instance_m
                           + (1 - self.instance_m) * self.embed_output)
        return tf.nn.l2_normalize(new_data_memory, axis=1)

    def __get_close_nei_in_back(
            self, each_k_idx, cluster_labels, 
            back_nei_idxs, k):
        batch_labels = tf.gather(
                cluster_labels[each_k_idx], 
                self.inputs['index'])

        top_cluster_labels = tf.gather(
                cluster_labels[each_k_idx], back_nei_idxs)
        batch_labels = repeat_1d_tensor(batch_labels, k)
        curr_close_nei = tf.equal(batch_labels, top_cluster_labels)
        return curr_close_nei

    def __get_relative_prob(self, all_close_nei, back_nei_probs):
        relative_probs = tf.reduce_sum(
            tf.where(
                all_close_nei,
                x=back_nei_probs, y=tf.zeros_like(back_nei_probs),
            ), axis=1)
        relative_probs /= tf.reduce_sum(back_nei_probs, axis=1)
        return relative_probs

    def get_LA_loss(
            self, cluster_labels, 
            k=None):
        if not k:
            k = self.instance_k
        # use the top k nearest examples as background neighbors
        all_dps = self.memory_bank.get_all_dot_products(self.embed_output)
        back_nei_dps, back_nei_idxs = tf.nn.top_k(all_dps, k=k, sorted=False)
        back_nei_probs = self._softmax(back_nei_dps)

        no_kmeans = cluster_labels.get_shape().as_list()[0]
        all_close_nei = None
        for each_k_idx in range(no_kmeans):
            curr_close_nei = self.__get_close_nei_in_back(
                    each_k_idx, cluster_labels, back_nei_idxs, k)

            if all_close_nei is None:
                all_close_nei = curr_close_nei
            else:
                all_close_nei = tf.logical_or(all_close_nei, curr_close_nei)
        relative_probs = self.__get_relative_prob(
                all_close_nei, back_nei_probs)

        assert_shape(relative_probs, [self.batch_size])
        loss = -tf.reduce_mean(tf.log(relative_probs + 1e-7))
        return loss

    def get_IR_losses(self):
        data_prob = self.compute_data_prob()
        noise_prob = self.compute_noise_prob()
        assert_shape(data_prob, [self.batch_size])
        assert_shape(noise_prob, [self.batch_size, self.instance_k])

        base_prob = 1.0 / self.data_len
        eps = 1e-7
        ## Pmt
        data_div = data_prob + (self.instance_k*base_prob + eps)
        ln_data = tf.log(data_prob / data_div)
        ## Pon
        noise_div = noise_prob + (self.instance_k*base_prob + eps)
        ln_noise = tf.log((self.instance_k*base_prob) / noise_div)

        curr_loss = -(tf.reduce_sum(ln_data) \
                      + tf.reduce_sum(ln_noise)) / self.batch_size
        return curr_loss, \
            -tf.reduce_sum(ln_data)/self.batch_size, \
            -tf.reduce_sum(ln_noise)/self.batch_size


def build_targets(
        inputs, train, 
        model_type,
        kmeans_k,
        task,
        **kwargs):
    # This will be stored in the db
    logged_cfg = {'kwargs': kwargs}

    data_len = kwargs.get('data_len', DATA_LEN_IMAGENET_FULL)
    with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
        all_labels = tf.get_variable(
            'all_labels',
            initializer=tf.zeros_initializer,
            shape=(data_len,),
            trainable=False,
            dtype=tf.int64,
        )
        # TODO: hard-coded output dimension 128
        memory_bank = MemoryBank(data_len, 128)

        lbl_init_values = tf.range(data_len, dtype=tf.int64)
        no_kmeans_k = len(kmeans_k)
        lbl_init_values = tf.tile(
                tf.expand_dims(lbl_init_values, axis=0),
                [no_kmeans_k, 1])
        cluster_labels = tf.get_variable(
            'cluster_labels',
            initializer=lbl_init_values,
            trainable=False, dtype=tf.int64,
        )

    output = network_embedding(
            inputs['image'], train=train, 
            model_type=model_type)

    if not train:
        all_dist = memory_bank.get_all_dot_products(output)
        return [all_dist, all_labels], logged_cfg

    loss_builder = LossBuilder(
        inputs=inputs, output=output,
        memory_bank=memory_bank,
        **kwargs)

    if task == 'IR':
        loss, loss_model, loss_noise = loss_builder.get_IR_losses()
        clustering = None
        ret_loss = [loss, loss_model, loss_noise]
    elif task == 'LA':
        from .cluster_km import Kmeans
        clustering = Kmeans(kmeans_k, memory_bank, cluster_labels)
        loss = loss_builder.get_LA_loss(cluster_labels)
        ret_loss = [loss]
    else:
        raise NotImplementedError('Task not supported!')

    new_data_memory = loss_builder.updated_new_data_memory()
    return {
        "losses": ret_loss,
        "data_indx": inputs['index'],
        "memory_bank": memory_bank.as_tensor(),
        "new_data_memory": new_data_memory,
        "all_labels": all_labels,
    }, logged_cfg, clustering


def build_transfer_targets(
        inputs, train, 
        model_type='resnet18',
        get_all_layers=None, 
        num_classes=1000,
        **kwargs):
    # This will be stored in the db
    logged_cfg = {'kwargs': kwargs}
    input_image = inputs['image']
    num_crop = None
    curr_image_shape = input_image.get_shape().as_list()
    batch_size = curr_image_shape[0]
    if len(curr_image_shape) > 4:
        num_crop = curr_image_shape[1]
        input_image = tf.reshape(input_image, [-1] + curr_image_shape[2:])

    resnet_output = network_embedding(
        input_image,
        train=False,
        model_type=model_type,
        skip_final_dense=True,
        get_all_layers=get_all_layers)

    with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
        init_builder = tf.contrib.layers.variance_scaling_initializer()
        if not get_all_layers:
            class_output = tf.layers.dense(
                inputs=resnet_output, units=num_classes,
                kernel_initializer=init_builder,
                trainable=True,
                name='transfer_dense')
        else:
            class_output = OrderedDict()
            for key, curr_out in resnet_output.items():
                class_output[key] = tf.layers.dense(
                    inputs=curr_out, units=num_classes,
                    kernel_initializer=init_builder,
                    trainable=True,
                    name='transfer_dense_{name}'.format(name=key))

    def __get_loss_accuracy(curr_output):
        if num_crop:
            curr_output = tf.nn.softmax(curr_output)
            curr_output = tf.reshape(curr_output, [batch_size, num_crop, -1])
            curr_output = tf.reduce_mean(curr_output, axis=1)
        _, pred = tf.nn.top_k(curr_output, k=1)
        pred = tf.cast(tf.squeeze(pred), tf.int64)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(pred, inputs['label']), tf.float32)
        )

        one_hot_labels = tf.one_hot(inputs['label'], num_classes)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, curr_output)
        return loss, accuracy
    if not get_all_layers:
        loss, accuracy = __get_loss_accuracy(class_output)
    else:
        loss = []
        accuracy = OrderedDict()
        for key, curr_out in class_output.items():
            curr_loss, curr_acc = __get_loss_accuracy(curr_out)
            loss.append(curr_loss)
            accuracy[key] = curr_acc
        loss = tf.reduce_sum(loss)

    if not train:
        return accuracy, logged_cfg
    return [loss, accuracy], logged_cfg
