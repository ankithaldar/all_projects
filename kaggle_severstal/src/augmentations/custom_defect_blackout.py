#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Custom Blackout for Defects'''


# imports
import random

import numpy as np

#    script imports
# imports


# constants
# constants


# classes
class DefectBlackout:
  '''
  Randomly blacks out defect regions. If ALL defects are blacked out in the crop,
  the label is flipped from 1 → 0.

  Expected input format:
    image: HxWx3 uint8
    masks: list of per-defect binary masks (np arrays)
  '''

  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, image, masks):
    '''
    :param image: RGB image np.ndarray
    :param masks: list of binary masks [H,W] for each defect instance
    :return: modified image, updated masks, new_label
    '''
    if random.random() > self.p:
      # no blackout applied
      has_defect = int(any(mask.sum() > 0 for mask in masks))
      return image, masks, has_defect

    new_masks = []
    blackout_all = True

    for mask in masks:
      if np.random.rand() < 0.5:
        # blackout this defect region
        coords = np.where(mask > 0)
        image[coords[0], coords[1], :] = 0  # set to black
        new_masks.append(np.zeros_like(mask))
      else:
        # keep mask
        new_masks.append(mask)
        if mask.sum() > 0:
          blackout_all = False

    new_label = 0 if blackout_all else 1
    return image, new_masks, new_label
# classes
