from __future__ import division, print_function, absolute_import
import os

from param_setter import get_config, get_params_from_arg
from framework import TrainFramework


def main():
    # Parse arguments
    cfg = get_config()
    args = cfg.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Get params needed, start training
    params = get_params_from_arg(args)
    train_framework = TrainFramework(params)
    train_framework.train()


if __name__ == "__main__":
    main()
