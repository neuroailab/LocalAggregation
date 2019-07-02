import tensorflow as tf
import argparse
import os
import numpy as np
import sys
import pdb
from tqdm import tqdm

NUM_TRAIN_TFR = 1024
NUM_VAL_TFR = 128
SEED = 0
TRAIN_PAT = 'train-%05i-%05i'
VAL_PAT = 'validation-%05i-%05i'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to write to tfrecords for places')
    parser.add_argument(
            '--out_dir', 
            default='/data5/chengxuz/Dataset/places/tfrs_205', 
            type=str, action='store', 
            help='Output directory')
    parser.add_argument(
            '--csv_folder', 
            default='/data5/chengxuz/Dataset/places/split/trainvalsplit_places205', 
            type=str, action='store', 
            help='Csv folder')
    parser.add_argument(
            '--base_dir', 
            default='/data5/chengxuz/Dataset/places/images', 
            type=str, action='store', 
            help='Image base folder')
    parser.add_argument(
            '--run', 
            action='store_true', 
            help='Whether actually run')
    return parser


def get_jpg_list(csv_path):
    fin = open(csv_path, 'r')
    all_lines = fin.readlines()

    all_jpg_lbls = []
    for each_line in all_lines:
        try:
            line_splits = each_line.split()
            jpg_path = line_splits[0]
            curr_label = int(line_splits[1])

            all_jpg_lbls.append((jpg_path, curr_label))
        except:
            print(each_line)
    return all_jpg_lbls


def get_train_val_list(args):
    train_csv_path = os.path.join(args.csv_folder, 'train_places205.csv')
    val_csv_path = os.path.join(args.csv_folder, 'val_places205.csv')

    train_jpg_lbls = get_jpg_list(train_csv_path)
    val_jpg_lbls = get_jpg_list(val_csv_path)
    return train_jpg_lbls, val_jpg_lbls


def write_one_rec(writer, idx, img_path, lbl, args):
    img_path = os.path.join(args.base_dir, img_path)
    img_jpg_str = open(img_path, 'rb').read()
    feature_dict = {
            'images': _bytes_feature(img_jpg_str),
            'labels': _int64_feature(lbl)}
    if idx is not None:
        feature_dict['index'] = _int64_feature(idx)
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    writer.write(example.SerializeToString())


def write_tfrs(jpg_lbls, num_files, file_pat, args):
    no_imgs_each_tfr = int(len(jpg_lbls) / num_files)
    curr_tfr_idx = 0
    writer = None
    for idx, (img_path, lbl) in enumerate(tqdm(jpg_lbls)):
        if idx % no_imgs_each_tfr == 0:
            if writer is not None:
                writer.close()
            tfr_path = os.path.join(
                    args.out_dir, file_pat % (curr_tfr_idx, num_files))
            writer = tf.python_io.TFRecordWriter(tfr_path)
            curr_tfr_idx += 1
        write_one_rec(writer, idx, img_path, lbl, args)
    writer.close()


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.base_dir = os.path.join(
            args.base_dir, 
            'data/vision/torralba/deeplearning/images256/')

    os.system('mkdir -p {path}'.format(path=args.out_dir))
    train_jpg_lbls, val_jpg_lbls = get_train_val_list(args)

    np.random.seed(SEED)
    np.random.shuffle(train_jpg_lbls)
    if args.run:
        write_tfrs(train_jpg_lbls, NUM_TRAIN_TFR, TRAIN_PAT, args)
        write_tfrs(val_jpg_lbls, NUM_VAL_TFR, VAL_PAT, args)


if __name__=="__main__":
    main()
