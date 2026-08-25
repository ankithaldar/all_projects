#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports

from dataclasses import dataclass

import albumentations as A
from albumentations.pytorch import ToTensorV2
#    script imports
from helpers.module_importer import instantiate_class

# imports


# constants
# constants


# classes
@dataclass
class Augmentaiton:
  class_path: str
  init_args: dict


def train_augmentation(blackout: bool = True):
  return [
    Augmentaiton(
      class_path='augmentations.custom_defect_blackout.CustomDefectBlackout',
      init_args={'p': 0.5}
    )
  ] if blackout else [] + [
    Augmentaiton(
      class_path='albumentations.RandomCrop',
      init_args={'height': 224, 'width': 1568}
    ),
    Augmentaiton(
      class_path='albumentations.HorizontalFlip',
      init_args={'p': 0.5}
    ),
    Augmentaiton(
      class_path='albumentations.VerticalFlip',
      init_args={'p': 0.5}
    ),
    Augmentaiton(
      class_path='albumentations.RandomBrightnessContrast',
      init_args={'p': 0.5}
    ),
    Augmentaiton(
      class_path='albumentations.RandomBrightnessContrast',
      init_args={'p': 0.5}
    ),
    Augmentaiton(
      class_path='albumentations.Normalize',
      init_args={}
    )
  ]


def validation_augmentation():
  return [
    Augmentaiton(
      class_path='albumentations.Normalize',
      init_args={}
    )
  ]

# ------------------------------------------------------------------------------


class ClassificationAugmentation:
  '''Augmentation for classification'''

  def __init__(self, train: bool = True, blackout: bool = True):
    self.aug_list = train_augmentation(blackout) if train else validation_augmentation()

  def __call__(self):
    augs = []
    if isinstance(self.aug_list, list) and len(self.aug_list) > 0:
      augs = [instantiate_class(aug.class_path, aug.init_args) for aug in self.aug_list]

    augs.append(ToTensorV2())

    return A.Compose(augs)

# classes
