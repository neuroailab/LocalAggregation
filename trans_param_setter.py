from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf

import json
import copy
import argparse
import time
import functools
import inspect

from model import preprocessing as prep
from model import instance_model
from model.dataset_utils import dataset_func
from utils import DATA_LEN_IMAGENET_FULL, tuple_get_one
from param_setter import get_lr_from_boundary
import config


def get_config():
    cfg = config.Config()
    cfg.add('batch_size', type=int, default=128,
            help='Training batch size')
    cfg.add('test_batch_size', type=int, default=64,
            help='Testing batch size')
    cfg.add('init_lr', type=float, default=0.01,
            help='Initial learning rate')
    cfg.add('gpu', type=str, required=True,
            help='Value for CUDA_VISIBLE_DEVICES')
    cfg.add('weight_decay', type=float, default=1e-4,
            help='Weight decay')
    cfg.add('image_dir', type=str, required=True,
            help='Directory containing dataset')
    cfg.add('q_cap', type=int, default=102400,
            help='Shuffle queue capacity of tfr data')
    cfg.add('ten_crop', type=bool,
            help='Whether do ten crop validation')

    # Loading parameters
    cfg.add('load_exp', type=str, required=True,
            help='The experiment to load from, in the format '
                 '[dbname]/[collname]/[exp_id]')
    cfg.add('load_step', type=int, default=None,
            help='Step number for loading')
    cfg.add('load_port', type=int,
            help='Port number of mongodb for loading (defaults to saving port')
    cfg.add('resume', type=bool,
            help='Flag for loading from last step of this exp_id, will override'
            ' all other loading options.')

    # Saving parameters
    cfg.add('port', type=int, required=True,
            help='Port number for mongodb')
    cfg.add('host', type=str, default='localhost',
            help='Host for mongodb')
    cfg.add('save_exp', type=str, required=True,
            help='The [dbname]/[collname]/[exp_id] of this experiment.')
    cfg.add('cache_dir', type=str, required=True,
            help='Prefix of saving directory')
    cfg.add('fre_valid', type=int, default=10009,
            help='Frequency of validation')
    cfg.add('fre_metric', type=int, default=1000,
            help='Frequency of saving metrics')
    cfg.add('fre_filter', type=int, default=10009,
            help='Frequency of saving filters')
    cfg.add('fre_cache_filter', type=int,
            help='Frequency of caching filters')

    # Training parameters
    cfg.add('model_type', type=str, default='resnet18',
            help='Model type, resnet or alexnet')
    cfg.add('get_all_layers', type=str, default=None,
            help='Whether get outputs for all layers')
    cfg.add('lr_boundaries', type=str, default=None,
            help='Learning rate boundaries for 10x drops')
    cfg.add('train_crop', type=str, default='default',
            help='Train crop style')
    cfg.add('num_classes', type=int, default=1000,
            help='Number of classes')
    return cfg


def reg_loss(loss, weight_decay):
    # Add weight decay to the loss.
    def exclude_batch_norm_and_other_device(name):
        return 'batch_normalization' not in name \
                and 'instance' in name
    l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in tf.trainable_variables()
                if exclude_batch_norm_and_other_device(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all


def add_training_params(params, args):
    NUM_BATCHES_PER_EPOCH = DATA_LEN_IMAGENET_FULL / args.batch_size

    # model_params: a function that will build the model
    model_params = {
        'func': instance_model.build_transfer_targets,
        'trainable_scopes': ['instance'],
        'get_all_layers': args.get_all_layers,
        "model_type": args.model_type,
        "num_classes": args.num_classes,
    }
    multi_gpu = len(args.gpu.split(','))
    if multi_gpu > 1:
        model_params['num_gpus'] = multi_gpu
        model_params['devices'] = ['/gpu:%i' % idx for idx in range(multi_gpu)]
    params['model_params'] = model_params

    # train_params: parameters about training data
    process_img_func = prep.resnet_train
    if args.train_crop == 'resnet_crop_flip':
        process_img_func = prep.resnet_crop_flip
    elif args.train_crop == 'alexnet_crop_flip':
        process_img_func = prep.alexnet_crop_flip
    elif args.train_crop == 'validate_crop':
        process_img_func = prep.resnet_validate

    train_data_param = {
            'func': dataset_func,
            'image_dir': args.image_dir,
            'process_img_func': process_img_func,
            'is_train': True,
            'q_cap': args.q_cap,
            'batch_size': args.batch_size}

    def _train_target_func(
        inputs,
        output,
        get_all_layers=None,
        *args,
        **kwargs):
        if not get_all_layers:
            return {'accuracy': output[1]}
        else:
            return {'accuracy': tf.reduce_mean(output[1].values())}

    params['train_params'] = {
        'validate_first': False,
        'data_params': train_data_param,
        'queue_params': None,
        'thres_loss': float('Inf'),
        'num_steps': int(2000 * NUM_BATCHES_PER_EPOCH),
        'targets': {
                'func': _train_target_func,
                'get_all_layers': args.get_all_layers,
                },
    }

    # loss_params: parameters to build the loss
    def loss_func(output, *args, **kwargs):
        #print('loss_output', output)
        return output[0]
    params['loss_params'] = {
        'pred_targets': [],
        # we don't want GPUs to calculate l2 loss separately
        'agg_func': reg_loss,
        'agg_func_kwargs': {'weight_decay': args.weight_decay},
        'loss_func': loss_func,
    }


def add_validation_params(params, args):
    # validation_params: control the validation
    val_len = 50000
    valid_prep_func = prep.resnet_validate
    if args.ten_crop:
        valid_prep_func = prep.resnet_10crop_validate

    topn_val_data_param = {
            'func': dataset_func,
            'image_dir': args.image_dir,
            'process_img_func': valid_prep_func,
            'is_train': False,
            'q_cap': args.test_batch_size,
            'batch_size': args.test_batch_size}

    def online_agg(agg_res, res, step):
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            agg_res[k].append(np.mean(v))
        return agg_res
    def valid_perf_func(inputs, output):
        if not args.get_all_layers:
            return {'top1': output}
        else:
            ret_dict = {}
            for key, each_out in output.items():
                ret_dict['top1_{name}'.format(name=key)] = each_out
            return ret_dict

    topn_val_param = {
        'data_params': topn_val_data_param,
        'queue_params': None,
        'targets': {'func': valid_perf_func},
        # TODO: slight rounding error?
        'num_steps': int(val_len/args.test_batch_size),
        'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        'online_agg_func': online_agg,
    }
    params['validation_params'] = {
        'topn': topn_val_param,
    }


def add_save_and_load_params(params, args):
    # save_params: defining where to save the models
    db_name, col_name, exp_id = args.save_exp.split('/')
    cache_dir = os.path.join(
        args.cache_dir, 'models',
        db_name, col_name, exp_id)
    params['save_params'] = {
        'host': 'localhost', # used for tfutils
        'port': args.port, # used for tfutils
        'dbname': db_name,
        'collname': col_name,
        'exp_id': exp_id,
        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': args.fre_metric,
        'save_valid_freq': args.fre_valid,
        'save_filters_freq': args.fre_filter,
        'cache_filters_freq': args.fre_cache_filter or args.fre_filter,
        'cache_dir': cache_dir,
    }

    # load_params: defining where to load, if needed
    if args.resume or args.load_exp is None:
        load_exp = args.save_exp
    else:
        load_exp = args.load_exp
    load_dbname, load_collname, load_exp_id = load_exp.split('/')
    if args.resume or args.load_step is None:
        load_query = None
    else:
        load_query = {
            'exp_id': load_exp_id,
            'saved_filters': True,
            'step': args.load_step
        }
    params['load_params'] = {
        'host': 'localhost', # used for tfutils
        'port': args.load_port or args.port, # used for tfutils
        'dbname': load_dbname,
        'collname': load_collname,
        'exp_id': load_exp_id,
        'do_restore': True,
        'query': load_query,
    }


def add_optimization_params(params, args):
    # learning_rate_params: build the learning rate
    # For now, just stay the same
    NUM_BATCHES_PER_EPOCH = DATA_LEN_IMAGENET_FULL / args.batch_size
    params['learning_rate_params'] = {
            'func': get_lr_from_boundary,
            'init_lr': args.init_lr,
            'NUM_BATCHES_PER_EPOCH': NUM_BATCHES_PER_EPOCH,
            'boundaries': args.lr_boundaries,
            }

    # optimizer_params
    params['optimizer_params'] = {
        'optimizer': tf.train.MomentumOptimizer,
        'momentum': .9,
    }


def get_params_from_args(args):
    params = {
        'skip_check': True,
        'log_device_placement': False
    }

    add_training_params(params, args)
    add_save_and_load_params(params, args)
    add_optimization_params(params, args)
    add_validation_params(params, args)
    return params
