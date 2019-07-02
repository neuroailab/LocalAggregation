from __future__ import division, print_function, absolute_import
import os, sys, datetime
import numpy as np
import tensorflow as tf
import copy
import pdb
from model.instance_model import DATA_LEN_IMAGENET_FULL


def tuple_get_one(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return x[0]
    return x


class Logger(object):
    def __init__(self, path_prefix, exp_id):
        self.path_prefix = path_prefix
        self.exp_id = exp_id
        self.exp_path = exp_path = os.path.join(path_prefix, exp_id)
        os.system('mkdir -p %s' % exp_path)

        # Record info about this experiment
        config_path = os.path.join(exp_path, 'config_%s.txt' % exp_id)
        print('Storing experiment configs at %s' % config_path)
        # TODO: prompt if config already exists (indicating experiment has been run previously)
        with open(config_path, 'w') as f:
            f.write('Started running at: %s\n' % str(datetime.datetime.now()))
            # Record command used to run the training
            f.write(" ".join(sys.argv) + '\n')

        self._log_path = os.path.join(exp_path, 'log_performance_%s.txt' % exp_id)
        print('Logging results to %s' % self._log_path)
        self._writer = open(self._log_path, 'a+')

    def reopen(self):
        self._writer.close()
        self._writer = open(self._log_path, 'a+')

    def log(self, s, also_print=True):
        if also_print:
            print(s)
            sys.stdout.flush()
        self._writer.write(s + '\n')

    def close(self):
        self._writer.close()
