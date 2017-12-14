import os
import glob
import threading

import numpy as np
import pandas as pd
import tensorflow as tf

tf.app.flags.DEFINE_string('base_dir', r'F:\machine-learning-dataset\dog idendify',
                           'base directory of data')
tf.app.flags.DEFINE_string('output_dir', r'F:\machine-learning-dataset\dog idendify',
                           'generate TFRecode file, saved in the output directory')

tf.app.flags.DEFINE_string('num_thread', 8, 'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def _read_file(base_dir, pack_name):
    files = glob.glob(os.path.join(base_dir, pack_name, '*'))
    return files


def _process_one_image():
    pass


def _process_image_batch(name, files, out_dir, thread_id, label_map=None):
    count = 0
    for f in files:
        dir_name = os.path.join(out_dir, 'recode', name)
        os.makedirs(dir_name, exist_ok=True)
        out_file_name = '{}_{}'.format(name, thread_id * len(files) + count)
        print(out_file_name, f)
        count += 1


def _labels_lookup(label_dir):
    label = pd.read_csv(label_dir)
    return dict(zip(label.iloc[:, 0], label.iloc[:, 1]))


def _process_files(name, base_dir, pack_name, num_thread, out_dir, label_map=None):
    assert name == 'train' and label_map is not None
    files = _read_file(base_dir, pack_name)
    assert len(files) % FLAGS.num_thread, 'wrong num_thread'
    linspace = np.linspace(0, len(files), num_thread + 1).astype(np.int32)
    rang_list = [files[linspace[i]: linspace[i + 1]] for i in range(len(linspace) - 1)]
    threads = []
    for t in range(num_thread):
        args = (name, rang_list[t], out_dir, t, label_map)
        thread = threading.Thread(target=_process_image_batch, args=args)
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()


def main(unused_argv):
    sub_dirs = glob.glob(os.path.join(FLAGS.base_dir, '*'))
    assert len(sub_dirs) > 0
    for sub in sub_dirs:
        if 'train' in sub and '.' not in sub:
            train_pack = sub
        elif 'test' in sub and '.' not in sub:
            test_pack = sub
        elif 'label' in sub and str(sub).endswith('.csv'):
            label_dir = os.path.join(FLAGS.base_dir, sub)
    assert train_pack not in locals().keys(), 'can not find train package under the base directory.'
    assert test_pack not in locals().keys(), 'can not find test package under the base directory.'
    assert label_dir not in locals().keys(), 'can not find label file under the base directory.'
    label_map = _labels_lookup(label_dir)
    _process_files('train', FLAGS.base_dir, train_pack, FLAGS.num_thread, FLAGS.output_dir, label_map)


if __name__ == '__main__':
    tf.app.run()