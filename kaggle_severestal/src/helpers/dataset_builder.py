#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Read CSV file for training in Pandas and do basic manupulation'''


# imports
import pandas as pd
from pathlib import Path
import copy
#    script imports
# imports


# constants
# constants


# classes
class CSVReader:
  '''Read and manipulate training CSV'''

  def __init__(self, csv_path:Path):
    self.train_df = self._read_csv(csv_path)

  def _read_csv(self, csv_path:Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

  def create_multiclass_classification_dataset(self):
    # -- create a key value pair in config params for selecting the columns
    classification_set = self.train_df[['ImageId', 'ClassId']]

    return classification_set

  def create_binary_classification_dataset(self):
    train_df = copy.deepcopy(self.train_df)
    train_df['class_id'] = 1

    classification_set = train_df[['ImageId', 'class_id']].drop_duplicates().reset_index()

    return classification_set



class DatasetBuilder:
  '''Build Dataset from file path and CSV files'''

  def __init__(self, folder_path:Path, csv_path:Path=None, class_type:str='binary'):
    self.folder_path = folder_path
    self.csv_path = csv_path
    self.class_type = class_type

  def create_dataset(self):

    # make listof file from given training/test folder
    img_list = [f.name for f in self.folder_path.iterdir() if f.is_file()]

    img_label = pd.DataFrame.from_dict({
      'ImageId': img_list,
      'class_id': [0] * len(img_list)
    })


    if self.csv_path is not None:
      csv_read = CSVReader(self.csv_path)
      if self.class_type == 'binary':
        class_labels = csv_read.create_binary_classification_dataset()

        img_label = pd.merge(img_label, class_labels, on='ImageId', how='left', suffixes=('', '_csv'))
        img_label['class_id'] = img_label['class_id_csv'].fillna(0).map(int)


    # return img_label[['ImageId', 'class_id']] #.to_dict('records')
    return list(img_label[['ImageId', 'class_id']].itertuples(index=False))


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
