import sys, os
import numpy as np
import argparse
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_parser():
    parser = argparse.ArgumentParser(
            description='Generate tfrecords from jpeg images, ' \
                    + 'default parameters are for node7')
    parser.add_argument(
            '--save_dir', 
            default='/data5/chengxuz/Dataset/test_la_imagenet_tfr', type=str, 
            action='store', help='Directory to save the tfrecords')
    parser.add_argument(
            '--img_folder', 
            default='/data5/chengxuz/Dataset/imagenet_raw', type=str, 
            action='store', help='Directory storing the original images')
    parser.add_argument(
            '--random_seed', 
            default=0, type=int, 
            action='store', help='Random seed for numpy')
    return parser


def get_label_dict(folder):
    all_nouns = os.listdir(folder)
    all_nouns.sort()
    label_dict = {noun:idx for idx, noun in enumerate(all_nouns)}
    return label_dict, all_nouns


def get_imgs_from_dir(synset_dir):
    curr_imgs = os.listdir(synset_dir)
    curr_imgs = [os.path.join(synset_dir, each_img) 
                 for each_img in curr_imgs]
    curr_imgs.sort()
    return curr_imgs


def get_path_and_label(img_folder):
    all_path_labels = []
    label_dict, all_nouns = get_label_dict(img_folder)
    print('Getting all image paths')
    for each_noun in tqdm(all_nouns):
        curr_paths = get_imgs_from_dir(
                os.path.join(img_folder, each_noun))
        curr_path_labels = [(each_path, label_dict[each_noun]) \
                            for each_path in curr_paths]
        all_path_labels.extend(curr_path_labels)
    return all_path_labels


class ImageCoder(object):
  """
  Helper class that provides TensorFlow image coding utilities.
  from https://github.com/tensorflow/models/blob/a156e20367c2a8195ba11da2e1d8589e93afdf40/research/inception/inception/data/build_imagenet_data.py
  """

  def __init__(self):
    # Create a single Session to run all image coding calls.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

coder = ImageCoder()


def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  # File list from:
  # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
  return 'n02105855_2933.JPEG' in filename


def _is_cmyk(filename):
  """Determine if file contains a CMYK JPEG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.
  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  # add one validation file which is also CMYK
  blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
               'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
               'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
               'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
               'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
               'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
               'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
               'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
               'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
               'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
               'n07583066_647.JPEG', 'n13037406_4650.JPEG',
               'ILSVRC2012_val_00019877.JPEG']
  return filename.split('/')[-1] in blacklist


def get_img_raw_str(jpg_path):
    with tf.gfile.FastGFile(jpg_path, 'rb') as f:
        img_raw_str = f.read()
    if _is_png(jpg_path):
        # 1 image is a PNG.
        print('Converting PNG to JPEG for %s' % jpg_path)
        img_raw_str = coder.png_to_jpeg(img_raw_str)
    elif _is_cmyk(jpg_path):
        # 23 JPEG images are in CMYK colorspace.
        print('Converting CMYK to RGB for %s' % jpg_path)
        img_raw_str = coder.cmyk_to_rgb(img_raw_str)
    return img_raw_str


def write_to_tfrs(tfrs_path, curr_file_list):
    # Write each image and label
    writer = tf.python_io.TFRecordWriter(tfrs_path)
    for idx, jpg_path, lbl in curr_file_list:
        img_raw_str = get_img_raw_str(jpg_path)
        example = tf.train.Example(features=tf.train.Features(feature={
            'images': _bytes_feature(img_raw_str),
            'labels': _int64_feature(lbl),
            'index': _int64_feature(idx),
            }))
        writer.write(example.SerializeToString())
    writer.close()


def build_all_tfrs_from_folder(
        folder_path, num_tfrs, tfr_pat, 
        random_seed=None):
    # get all path and labels, shuffle them if needed
    all_path_labels = get_path_and_label(folder_path)
    if random_seed is not None:
        np.random.seed(random_seed)
        all_path_labels = np.random.permutation(all_path_labels)
    overall_num_imgs = len(all_path_labels)
    all_path_lbl_idx = [(idx, path, int(lbl)) \
                        for idx, (path, lbl) in enumerate(all_path_labels)]
    print('%i images in total' % overall_num_imgs)

    # Cut into num_tfr tfrecords and write each of them
    num_img_per = int(np.ceil(overall_num_imgs*1.0/num_tfrs))
    print('Writing into tfrecords')
    for curr_tfr in tqdm(range(num_tfrs)):
        tfrs_path = tfr_pat % (curr_tfr, num_tfrs)
        start_num = curr_tfr * num_img_per
        end_num = min((curr_tfr+1) * num_img_per, overall_num_imgs)
        write_to_tfrs(tfrs_path, all_path_lbl_idx[start_num:end_num])


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.system('mkdir -p %s' % args.save_dir)

    build_all_tfrs_from_folder(
            os.path.join(args.img_folder, 'train'),
            1024,
            os.path.join(args.save_dir, 'train-%05i-of-%05i'), 
            args.random_seed)
    build_all_tfrs_from_folder(
            os.path.join(args.img_folder, 'val'),
            128,
            os.path.join(args.save_dir, 'validation-%05i-of-%05i'))


if __name__=="__main__":
    main()
