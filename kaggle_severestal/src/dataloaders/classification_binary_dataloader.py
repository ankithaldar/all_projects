#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Ankit Haldar

'''Binary Classification Dataset'''


# imports
import tensorflow as tf
# import pandas as pd
from pathlib import Path
#    script imports
from helpers.dataset_builder import DatasetBuilder
# imports


# constants
AUTOTUNE = tf.data.AUTOTUNE
tf.get_logger().setLevel('INFO')
# constants

# https://github.com/tensorflow/tensorflow/issues/39523
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset
# https://cs230.stanford.edu/blog/datapipeline/
# https://medium.com/analytics-vidhya/using-advanced-tensorflow-api-for-training-nns-subclassing-data-processing-extensions-a56a784f4084



# classes
class BinaryClassificationDataloader:
  '''Dataset for binary Classification'''

  def __init__(self, hparams):
    self.hparams = hparams
    #here
    self._create_tf_data()

  def _get_train_image_tuple(self, train_path: Path) -> list:
    '''
    create a list of tuples (x, y)
    x = image filename
    y = binary label
    '''
    return DatasetBuilder(train_path, self.hparams.csv_path, class_type='binary').create_dataset()


  # @tf.function
  def _load_image_in_tf(self, img_path:Path):
    if img_path is not None:
      return tf.cast(
        self._check_image_type_and_decode(
          tf.io.read_file(img_path),
          img_path.suffixes
        )/255,
        dtype=tf.float32
      )


  def _check_image_type_and_decode(self, img, img_type):
    '''check file type and decode image'''
    # check if file is jpeg
    if img_type in ['jpg', 'jpeg']:
      return tf.image.decode_jpeg(img)

    if img_type in ['png']:
      return tf.image.decode_png(img)


  def _create_tf_data(self):
    self.ds = tf.data.Dataset.from_tensor_slices(
      (self._get_train_image_tuple(self.hparams.train_path))
    )
    self.ds = self.ds.map(
        # lambda x, y: (tf.py_function(self._load_image, [x], [tf.float32]), y)
        lambda x, y: (
          self._load_image_in_tf(Path(self.hparams.train_ds_params['root_dir'] + x)),
          y
        )
    )
    self.ds = self.ds.shuffle(max(self.hparams.train_bs*25, 500))
    self.ds = self.ds.batch(self.hparams.train_bs)
    self.ds = self.ds.prefetch(AUTOTUNE)


# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
