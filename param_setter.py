from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf
import cPickle

import json
import copy
import argparse
import time
import functools
import inspect

from model import preprocessing as prep
from model import instance_model
from model.memory_bank import MemoryBank
from model.dataset_utils import dataset_func
from model.instance_model import get_global_step_var

from utils import DATA_LEN_IMAGENET_FULL, tuple_get_one 
import config
import pdb


def get_config():
    cfg = config.Config()
    cfg.add('exp_id', type=str, required=True,
            help='Name of experiment ID')
    cfg.add('batch_size', type=int, default=128,
            help='Training batch size')
    cfg.add('test_batch_size', type=int, default=64,
            help='Testing batch size')
    cfg.add('init_lr', type=float, default=0.03,
            help='Initial learning rate')
    cfg.add('gpu', type=str, required=True,
            help='Value for CUDA_VISIBLE_DEVICES')
    cfg.add('gpu_offset', type=int, default=0,
            help='GPU offset, useful for KMeans')
    cfg.add('image_dir', type=str, required=True,
            help='Directory containing dataset')
    cfg.add('q_cap', type=int, default=102400,
            help='Shuffle queue capacity of tfr data')
    cfg.add('data_len', type=int, default=DATA_LEN_IMAGENET_FULL,
            help='Total number of images in the input dataset')

    # Training parameters
    cfg.add('weight_decay', type=float, default=1e-4,
            help='Weight decay')
    cfg.add('instance_t', type=float, default=0.07,
            help='Temperature in softmax.')
    cfg.add('instance_k', type=int, default=4096,
            help='Closes neighbors to sample.')
    cfg.add('lr_boundaries', type=str, default=None,
            help='Learning rate boundaries for 10x drops')
    cfg.add('train_num_steps', type=int, default=None,
            help='Number of overall steps for training')

    cfg.add('kmeans_k', type=str, default='10000',
            help='K for Kmeans')
    cfg.add('model_type', type=str, default='resnet18',
            help='Model type, resnet or alexnet')
    cfg.add('task', type=str, default='LA',
            help='IR for instance recognition or LA for local aggregation')

    # Saving parameters
    cfg.add('port', type=int, required=True,
            help='Port number for mongodb')
    cfg.add('db_name', type=str, required=True,
            help='Name of database')
    cfg.add('col_name', type=str, required=True,
            help='Name of collection')
    cfg.add('cache_dir', type=str, required=True,
            help='Prefix of saving directory')
    cfg.add('fre_valid', type=int, default=10009,
            help='Frequency of validation')
    cfg.add('fre_filter', type=int, default=10009,
            help='Frequency of saving filters')
    cfg.add('fre_cache_filter', type=int,
            help='Frequency of caching filters')

    # Loading parameters
    cfg.add('load_exp', type=str, default=None,
            help='The experiment to load from, in the format '
                 '[dbname]/[collname]/[exp_id]')
    cfg.add('load_port', type=int,
            help='Port number of mongodb for loading (defaults to saving port')
    cfg.add('load_step', type=int,
            help='Step number for loading')

    return cfg


def loss_func(output, *args, **kwargs):
    loss_pure = output['losses'][0]
    return loss_pure


def reg_loss(loss, weight_decay):
    # Add weight decay to the loss.
    def exclude_batch_norm_and_other_device(name):
        return 'batch_normalization' not in name
    l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in tf.trainable_variables()
                if exclude_batch_norm_and_other_device(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all


def rep_loss_func(
        inputs,
        output,
        gpu_offset=0,
        **kwargs
        ):
    data_indx = output['data_indx']
    new_data_memory = output['new_data_memory']

    memory_bank_list = output['memory_bank']
    all_labels_list = output['all_labels']
    if isinstance(memory_bank_list, tf.Variable):
        memory_bank_list = [memory_bank_list]
        all_labels_list = [all_labels_list]

    devices = ['/gpu:%i' \
               % (idx + gpu_offset) for idx in range(len(memory_bank_list))]
    update_ops = []
    for device, memory_bank, all_labels \
            in zip(devices, memory_bank_list, all_labels_list):
        with tf.device(device):
            mb_update_op = tf.scatter_update(
                    memory_bank, data_indx, new_data_memory)
            update_ops.append(mb_update_op)
            lb_update_op = tf.scatter_update(
                    all_labels, data_indx,
                    inputs['label'])
            update_ops.append(lb_update_op)

    with tf.control_dependencies(update_ops):
        # Force the updates to happen before the next batch.
        if len(output['losses']) == 3:
            _, loss_model, loss_noise = output['losses']
            loss_model = tf.identity(loss_model)
            loss_noise = tf.identity(loss_noise)
            ret_dict = {
                    'loss_model': loss_model,
                    'loss_noise': loss_noise}
        else:
            loss_pure = output['losses'][0]
            loss_pure = tf.identity(loss_pure)
            ret_dict = {'loss_pure': loss_pure}
    return ret_dict


def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res


def valid_perf_func(
        inputs,
        output,
        ):
    curr_dist, all_labels = output
    all_labels = tuple_get_one(all_labels)
    _, top_indices = tf.nn.top_k(curr_dist, k=1)
    curr_pred = tf.gather(all_labels, tf.squeeze(top_indices, axis=1))
    imagenet_top1 = tf.reduce_mean(
            tf.cast(
                tf.equal(curr_pred, inputs['label']),
                tf.float32))
    return {'top1': imagenet_top1}


def get_model_func_params(args):
    model_params = {
        "data_len": args.data_len,
        "instance_t": args.instance_t,
        "instance_k": args.instance_k,
        "kmeans_k": args.kmeans_k,
        "model_type": args.model_type,
        "task": args.task,
    }
    return model_params


def get_lr_from_boundary(
        global_step, boundaries, 
        init_lr,
        NUM_BATCHES_PER_EPOCH):
    if boundaries is not None:
        boundaries = boundaries.split(',')
        boundaries = [int(each_boundary) for each_boundary in boundaries]

        all_lrs = [
                init_lr * (0.1 ** drop_level) \
                for drop_level in range(len(boundaries) + 1)]
        curr_lr = tf.train.piecewise_constant(
                x=global_step,
                boundaries=boundaries, values=all_lrs)
    else:
        curr_lr = tf.constant(init_lr)
    return curr_lr


def get_params_from_arg(args):
    '''
    This function gets parameters needed for training
    '''
    multi_gpu = len(args.gpu.split(',')) - args.gpu_offset
    data_len = args.data_len
    val_len = 50000
    NUM_BATCHES_PER_EPOCH = data_len // args.batch_size

    # save_params: defining where to save the models
    args.fre_cache_filter = args.fre_cache_filter or args.fre_filter
    cache_dir = os.path.join(
            args.cache_dir, 'models',
            args.db_name, args.col_name, args.exp_id)
    save_params = {
            'host': 'localhost', # used for tfutils
            'port': args.port, # used for tfutils
            'dbname': args.db_name,
            'collname': args.col_name,
            'exp_id': args.exp_id,
            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': 1000,
            'save_valid_freq': args.fre_valid,
            'save_filters_freq': args.fre_filter,
            'cache_filters_freq': args.fre_cache_filter,
            'cache_dir': cache_dir,
            }

    # load_params: defining where to load, if needed
    load_port = args.load_port or args.port
    load_dbname = args.db_name
    load_collname = args.col_name
    load_exp_id = args.exp_id
    load_query = None

    if args.load_exp is not None:
        load_dbname, load_collname, load_exp_id = args.load_exp.split('/')
    if args.load_step:
        load_query = {'exp_id': load_exp_id,
                      'saved_filters': True,
                      'step': args.load_step}
        print('Load query', load_query)

    load_params = {
            'host': 'localhost', # used for tfutils
            'port': load_port, # used for tfutils
            'dbname': load_dbname,
            'collname': load_collname,
            'exp_id': load_exp_id,
            'do_restore': True,
            'query': load_query,
            }


    # XXX: hack to set up training loop properly
    if args.kmeans_k.isdigit():
        args.kmeans_k = [int(args.kmeans_k)]
    else:
        args.kmeans_k = [int(each_k) for each_k in args.kmeans_k.split(',')]
    clusterings = []
    first_step = []
    # model_params: a function that will build the model
    model_func_params = get_model_func_params(args)
    def build_output(inputs, train, **kwargs):
        targets = instance_model.build_targets(
                inputs, train, 
                **model_func_params)
        if not train:
            return targets
        outputs, logged_cfg, clustering = targets
        clusterings.append(clustering)
        return outputs, logged_cfg

    def train_loop(sess, train_targets, **params):
        assert len(clusterings) == multi_gpu

        global_step_var = get_global_step_var()
        global_step = sess.run(global_step_var)

        # TODO: consider making this reclustering frequency a flag
        first_flag = len(first_step) == 0 
        update_fre = NUM_BATCHES_PER_EPOCH
        if (global_step % update_fre == 0 or first_flag) \
                and clusterings[0] is not None:
            if first_flag:
                first_step.append(1)
            print("Recomputing clusters...")
            new_clust_labels = clusterings[0].recompute_clusters(sess)
            for clustering in clusterings:
                clustering.apply_clusters(sess, new_clust_labels)

        return sess.run(train_targets)

    model_params = {'func': build_output}
    if multi_gpu > 1:
        model_params['num_gpus'] = multi_gpu
        model_params['devices'] = ['/gpu:%i' \
                                   % (idx + args.gpu_offset) \
                                   for idx in range(multi_gpu)]

    # train_params: parameters about training data
    train_data_param = {
            'func': dataset_func,
            'image_dir': args.image_dir,
            'process_img_func': prep.resnet_train,
            'is_train': True,
            'q_cap': args.q_cap,
            'batch_size': args.batch_size}
    train_num_steps = args.train_num_steps or float('Inf')
    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'thres_loss': float('Inf'),
            'num_steps': train_num_steps,
            'train_loop': {'func': train_loop},
            }

    ## Add other loss reports (loss_model, loss_noise)
    train_params['targets'] = {
            'func': rep_loss_func,
            'gpu_offset': args.gpu_offset,
            }

    # loss_params: parameters to build the loss
    loss_params = {
        'pred_targets': [],
        'agg_func': reg_loss,
        'agg_func_kwargs': {'weight_decay': args.weight_decay},
        'loss_func': loss_func,
    }

    # learning_rate_params: build the learning rate
    # For now, just stay the same
    learning_rate_params = {
            'func': get_lr_from_boundary,
            'init_lr': args.init_lr,
            'NUM_BATCHES_PER_EPOCH': NUM_BATCHES_PER_EPOCH,
            'boundaries': args.lr_boundaries,
            }

    optimizer_params = {
            'optimizer': tf.train.MomentumOptimizer,
            'momentum': .9,
            }

    # validation_params: control the validation
    topn_val_data_param = {
            'func': dataset_func,
            'image_dir': args.image_dir,
            'process_img_func': prep.resnet_validate,
            'is_train': False,
            'q_cap': args.test_batch_size,
            'batch_size': args.test_batch_size}
    val_step_num = int(val_len/args.test_batch_size)
    val_targets = {'func': valid_perf_func}
    topn_val_param = {
        'data_params': topn_val_data_param,
        'targets': val_targets,
        'num_steps': val_step_num,
        'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        'online_agg_func': online_agg,
        }

    validation_params = {
            'topn': topn_val_param,
            }

    # Put all parameters together
    params = {
            'save_params': save_params,
            'load_params': load_params,
            'model_params': model_params,
            'train_params': train_params,
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'log_device_placement': False,
            'validation_params': validation_params,
            'skip_check': True,
            }
    return params
