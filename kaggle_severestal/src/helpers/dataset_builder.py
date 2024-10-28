#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
import pandas as pd
from pathlib import Path
#    script imports
# imports


# constants
# constants


# classes
class CSVReader:
  '''docstring for CSV Reader'''
  def __init__(self, csv_file_path: Path):
    self.df = self._read_csv_pandas(csv_file_path)

  def _read_csv_pandas(self, csv_file_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_file_path)

  def create_classification_dataset(self, req_cols: list[str]):
    df = self.df[req_cols]
    one_hot = pd.get_dummies(df, columns=['ClassId'], dtype=int)
    return one_hot.groupby('ImageId').sum().reset_index()


class ClassificationDatasetBuilder:
  '''docstring for Dataset construction'''

  def __init__(self, hparams, is_train=True):
    super().__init__()
    self.is_train = is_train

    if self.is_train:
      self.img_folder_path = Path(hparams.train_ds_params['img_dir'])
      self.csv_reader = CSVReader(hparams.train_ds_params['csv_file_path'])
      self.req_cols = hparams.train_ds_params['columns_from_csv']

    if not self.is_train:
      self.img_folder_path = hparams.test_ds_params['img_dir']

  def get_list_of_files(self):
    # create dict of lists to feed
    img_label = pd.DataFrame.from_dict({
      'file_path': [self.img_folder_path / f.name for f in self.img_folder_path.iterdir()],
      'ImageId': [f.name for f in self.img_folder_path.iterdir() if f.is_file()]
    })

    if self.is_train:
      img_label['ClassId_0'] = 0

      csv_data = self.csv_reader.create_classification_dataset(self.req_cols)
      img_label = pd.merge(
        left=img_label,
        right=csv_data,
        how='left',
        left_on='ImageId',
        right_on='ImageId',
        suffixes=('', '_csv')
      )

      # flag ClassID_0 as 1
      img_label.loc[img_label['ImageId'].isnull(), 'ClassId_0'] = 1

      # replace missing in 4 ClassId with 0
      img_label = img_label.fillna(0)

      # drop ImageId column
      img_label = img_label.drop(columns='ImageId')

      # convert ClassId columns to int type
      to_int = [i for i in img_label.columns if 'ClassId' in i ]
      img_label[to_int] = img_label[to_int].astype(int)

    return {
      'split': 'train' if self.is_train else 'test',
      'file_path': img_label['file_path'].values,
      'labels': img_label[to_int].values,
    }

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
