import tensorflow as tf


class imageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        self._sess = tf.Session()
        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        imgae = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpg = tf.image.encode_jpeg(imgae, format='rgb', quality=100)
        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        imgae = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_jpg = tf.image.encode_jpeg(imgae, format='rgb', quality=100)
        # Initializes function that decodes RGB JPEG data.
        self._jpg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpg = tf.image.decode_jpeg(self._jpg_data, channels=3)

    def png_to_jpg(self, image_data):
        return self._sess.run(self._png_to_jpg, feed_dict={self._png_data: image_data})

    def cmyk_to_jpg(self, image_data):
        return self._sess.run(self._cmyk_to_jpg, feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        jpg_img = self._sess.run(self._decode_jpg, feed_dict={self._jpg_data: image_data})
        assert len(jpg_img.shape) == 3 and jpg_img.shape[2] == 3
        return jpg_img


class TFRecodeConverter(object):
    """helper class that provides image to tfrecode"""
    def __init__(self, colorspace='RGB', channels=3, image_format='JPEG'):
        self._colorspace = colorspace
        self._channels = channels
        self._image_format = image_format

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_features(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def convert(self, image_buffer=None, height=None, width=None):
        assert image_buffer is None, 'image_buffer is empty'
        assert height is None or height <= 0, 'height is wrong'
        assert width is None or width <= 0, 'width is wrong'
        return tf.train.Example(features=tf.train.Features(feature={
            'height': self._int64_features(height),
            'width': self._int64_features(width),
            'image_raw': self._bytes_feature(image_buffer),
            'colorspace': self._colorspace,
            'channels': self._channels,
            'image_format': self._image_format
        }))