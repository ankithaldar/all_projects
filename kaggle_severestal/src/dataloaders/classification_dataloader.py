#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Create Dataloader for classification'''


# imports
import tensorflow as tf
#    script imports
from helpers.dataset_builder import ClassificationDatasetBuilder
# imports


# constants
# constants


# classes
class ClassificationDataloader(tf.keras.utils.PyDataset):
  '''docstring for ClassificationDataloader'''

  def __init__(self, hparams, is_train=True):
    super().__init__(**hparams.dataloader_params)
    self.dataset= ClassificationDatasetBuilder(hparams, is_train).get_list_of_files()

    self.file_paths = self.dataset['file_path']
    self.labels = self.dataset['labels']

    self.batch_size = hparams.train_batch_size

    self.shuffle = hparams.train_ds_params['shuffle_on_epoch_end']

    self.dim = hparams.image_size
    self.channels = hparams.image_channels

    self.on_epoch_end()

  def __len__(self):
    # Return number of batches.
    return tf.math.ceil(len(self.dataset) / self.batch_size)

  def __getitem__(self, index):
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of file paths
    filepath_labels = [(self.file_paths[k], self.labels[k]) for k in indexes]

    # Generate data
    x, y = self.__data_generation(filepath_labels)

    return x, y

  def __data_generation(self, data):
    img_data = tf.zeros((self.batch_size, *self.dim, self.channels))
    labels = tf.zeros((self.batch_size, 5), dtype=int)

    for i, f_l in enumerate(data):
      # Load & Normalize image data
      img_data[i, ] = self.read_img_data(f_l[0]) / 255.0

      # Convert labels to one-hot encoding
      labels[i] = f_l[1]

      return img_data, labels


  def on_epoch_end(self):
    self.indexes = tf.range(len(self.file_paths))
    if self.shuffle:
      tf.random.shuffle(self.indexes)

  def read_img_data(self, file_path):
    return tf.image.decode_jpeg(
      contents=tf.io.read_file(filename=file_path),
      channels=self.channels
    )

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
