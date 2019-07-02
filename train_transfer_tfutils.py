from __future__ import division, print_function, absolute_import
import os
from trans_param_setter import get_config, get_params_from_args
from tfutils import base
import tensorflow as tf


def reg_loss_in_faster(loss, which_device, weight_decay):
    from tfutils.multi_gpu.easy_variable_mgr import COPY_NAME_SCOPE
    curr_scope_name = '%s%i' % (COPY_NAME_SCOPE, which_device)
    # Add weight decay to the loss.
    def exclude_batch_norm_and_other_device(name):
        return 'batch_normalization' not in name \
                and curr_scope_name in name \
                and 'instance' in name
    l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32))
                for v in tf.trainable_variables()
                if exclude_batch_norm_and_other_device(v.name)])
    loss_all = tf.add(loss, l2_loss)
    return loss_all


def main():
    # Parse arguments
    cfg = get_config()
    args = cfg.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    params = get_params_from_args(args)
    params['loss_params']['agg_func'] = reg_loss_in_faster
    cache_dir = os.path.join(
            args.cache_dir, 'models_tfutils', args.save_exp)
    params['save_params']['cache_dir'] = cache_dir
    base.train_from_params(**params)


if __name__ == "__main__":
    main()
