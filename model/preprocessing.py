from __future__ import division, print_function
import os, sys
import numpy as np
import tensorflow as tf

from .prep_utils import (
    preprocessing_inst, RandomSizedCrop_from_jpeg,
    ApplyGray, ColorJitter, alexnet_crop_from_jpg, prep_10crops_validate
)

# This file contains various preprocessing ops for images (typically
# used for data augmentation).

def resnet_train(img_str):
    return preprocessing_inst(img_str, 224, 224, is_train=True)


def resnet_validate(img_str):
    return preprocessing_inst(img_str, 224, 224, is_train=False)


def resnet_10crop_validate(img_str):
    return prep_10crops_validate(img_str, 224, 224)


def resnet_crop_flip(img_str):
    img = RandomSizedCrop_from_jpeg(
            img_str, out_height=224, out_width=224, size_minval=0.2)
    img = tf.image.random_flip_left_right(img)
    return img


def alexnet_crop_flip(img_str):
    img = alexnet_crop_from_jpg(img_str)
    img = tf.image.random_flip_left_right(img)
    return img
