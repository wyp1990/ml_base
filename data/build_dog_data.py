import os
import glob
import pandas as pd
import tensorflow as tf

tf.app.flags.DEFINE_string('base_dir', r'F:\machine-learning-dataset\dog idendify',
                           'base directory of data')
tf.app.flags.DEFINE_string('output_dir', r'F:\machine-learning-dataset\dog idendify',
                           'generate TFRecode file, saved in the output directory')

tf.app.flags.DEFINE_string('train_batch_size', 1024, 'Number of shards in training TFRecord files')
tf.app.flags.DEFINE_string('test_batch_size', 128, 'Number of shards in testing TFRecord files')
tf.app.flags.DEFINE_string('num_thread', 8, 'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def _read_file(base_dir, pack_name):
    files = glob.glob(os.path.join(base_dir, pack_name, '*'))
    return files


def _process_one_image():
    pass


def _process_image_batch():
    pass


def _labels_lookup(label_dir):
    label = pd.read_csv(label_dir)
    return dict(zip(label.iloc[:, 0], label.iloc[:, 1]))


def _process_files(base_dir, pack_name, batch_size, num_thread, label_map=None):
    files = _read_file(base_dir, pack_name)


def main(unused_argv):
    assert FLAGS.train_batch_size % FLAGS.num_thread, \
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_batch_size'
    assert FLAGS.test_batch_size % FLAGS.num_thread, \
        'Please make the FLAGS.num_threads commensurate with FLAGS.test_batch_size'
    sub_dirs = glob.glob(os.path.join(FLAGS.base_dir, '*'))
    assert len(sub_dirs) > 0
    for sub in sub_dirs:
        if 'train' in sub and '.' not in sub:
            train_pack = sub
        elif 'test' in sub and '.' not in sub:
            test_pack = sub
        elif 'label' in sub and str(sub).endswith('.csv'):
            label_dir = os.path.join(FLAGS.base_dir, sub)
    assert train_pack in locals().keys(), 'can not find train package under the base directory.'
    assert test_pack in locals().keys(), 'can not find test package under the base directory.'
    assert label_dir in locals().keys(), 'can not find label file under the base directory.'
    label_map = _labels_lookup(label_dir)
    _process_files(FLAGS.base_dir, train_pack, FLAGS.train_batch_size, FLAGS.num_thread, label_map)