#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''PyTorch Dataset for classification'''


# imports

from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
from augmentations.aug_classification import ClassificationAugmentation
from helpers.rle import rle_decode  # we will implement later
from torch.utils.data import Dataset

# imports


# constants
DATA_ROOT = Path('/content/kaggle/input/severstal-steel-defect-detection')
# constants


# classes
class ClassificationDataset(Dataset):
  '''
  Returns (image_tensor, label)

  label = 1 if any defect is present (unless blackout removes all)
  '''

  def __init__(
    self,
    csv_path=DATA_ROOT / 'train.csv',
    images_dir=DATA_ROOT / 'train_images',
    train=True,
    blackout=True,
    num_classes=4
  ):
    super().__init__()

    self.train = train
    self.images_dir = images_dir
    self.num_classes = num_classes
    self.blackout = blackout

    # fetch all images in dataframe
    df_enc = pd.read_csv(csv_path)

    # merge all files and labels
    df = self.setup_label_and_rle(
      self.merge_annotation_data(
        df=self.list_file_in_df(),
        df_1=df_enc.groupby('ImageId')['EncodedPixels'].apply(list),
        df_2=df_enc.groupby('ImageId')['ClassId'].apply(list)
      )
    )

    self.items = [{
      'image_id': img_id,
      'file_path': df.loc[img_id, 'abs_path'],
      'masks_rle': df.loc[img_id, 'EncodedPixels'],
      'classes': df.loc[img_id, 'ClassId'],
    }
    for img_id in df.index
    ]


  def list_file_in_df(self):
    if not self.images_dir.exists():
      raise FileNotFoundError(f'Path {self.images_dir} does not exist')
    if not self.images_dir.is_dir():
      raise NotADirectoryError(f'Path {self.images_dir} is not a directory')

    # Get all files in the directory (non-recursive)
    files = [f for f in self.images_dir.iterdir() if f.is_file()]

    return pd.DataFrame({
      'ImageId': [f.name for f in files],
      'abs_path': [str(f.resolve()) for f in files],
      'EncodedPixels': [['']] * len(files),
      'ClassId': [[0]] * len(files),
    }).set_index('ImageId')


  def setup_label_and_rle(self, df):
    '''
    Transforms a DataFrame containing segmentation mask annotations in Run-Length Encoding (RLE) format into a structured representation where each image has a fixed-length list per class for both labels and RLE strings.

    Parameters:
      - df: Base DataFrame with ImageId as index, columns: ['absolute file path', 'EncodedPixels', 'ClassId']

    Returns:
    - Merged DataFrame with updated EncodedPixels and ClassId for structured representation where each image has a fixed-length list per class for both labels and RLE strings.

    '''
    result_df = []

    for index, row in df.iterrows():
      labels = [0] * self.num_classes
      rle = [''] * self.num_classes

      for c_id in row['ClassId']:
        try:
          labels[c_id - 1] = 1
          rle[c_id - 1] = row['EncodedPixels'][row['ClassId'].index(c_id)]
        except IndexError as e:
          print(row['ClassId'], row['EncodedPixels'])
          raise

      result_df.append({
        'ImageId': index,
        'abs_path': row['abs_path'],
        'EncodedPixels': rle,
        'ClassId': labels
      })

    return pd.DataFrame(result_df).set_index('ImageId')


  def merge_annotation_data(self, df, df_1=None, df_2=None):
    '''
    Merge base file DataFrame with annotation DataFrames.

    Parameters:
    - df: Base DataFrame with ImageId as index, columns: ['absolute file path', 'EncodedPixels', 'ClassId']
    - df_1: Optional DataFrame with ImageId (index or column) and 'EncodedPixels' (list of strings)
    - df_2: Optional DataFrame with ImageId (index or column) and 'ClassId' (int or list of ints)

    Returns:
    - Merged DataFrame with updated EncodedPixels and ClassId where matches exist.
    '''
    # Make a copy to avoid modifying original
    result_df = df.copy()

    df_1 = df_1.to_frame()
    df_2 = df_2.to_frame()

    # Helper to ensure ImageId is index
    def ensure_index(df_temp, id_col='ImageId'):
      if df_temp.index.name != id_col and id_col in df_temp.columns:
        return df_temp.set_index(id_col)
      return df_temp

    # Merge EncodedPixels from df_1
    if df_1 is not None:
      df_1 = ensure_index(df_1)
      if 'EncodedPixels' in df_1.columns:
        # Only update rows that exist in both
        common_index = result_df.index.intersection(df_1.index)
        result_df.loc[common_index, 'EncodedPixels'] = df_1.loc[common_index, 'EncodedPixels']

    # Merge ClassId from df_2
    if df_2 is not None:
      df_2 = ensure_index(df_2)
      if 'ClassId' in df_2.columns:
        common_index = result_df.index.intersection(df_2.index)
        result_df.loc[common_index, 'ClassId'] = df_2.loc[common_index, 'ClassId']

    return result_df


  def load_image(self, file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


  def decode_masks(self, masks_rle):
    masks = []

    for rle in masks_rle:
      if isinstance(rle, str) and len(rle) > 0:
        masks.append(rle_decode(rle, shape=(256, 1600)))
      else:
        masks.append(np.zeros((256, 1600), dtype=np.uint8))
    return np.array(masks)


  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    item = self.items[idx]

    image = self.load_image(item['file_path'])
    masks = self.decode_masks(item['masks_rle'])
    labels = item['classes']

    self.aug = ClassificationAugmentation(train=self.train, blackout=self.blackout)()

    augmented = self.aug(image=image, masks=np.stack(masks, axis=0))
    image_tensor, masks = augmented['image'], augmented['masks']

    if isinstance(masks, torch.Tensor):
      masks = masks.numpy()

    # label is scalar int
    return image_tensor, np.array([np.any(mask) for mask in masks], dtype=int)

# classes
