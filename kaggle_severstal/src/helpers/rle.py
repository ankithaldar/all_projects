#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Run Length Encoding Utils'''


# imports
import numpy as np

#    script imports
# imports


# constants
# constants


# functions
def rle_decode(mask_rle, shape):
  '''
  Decodes RLE (Run-Length Encoding) string into a binary mask.
  This matches the Severstal competition format (column-major order).

  Parameters
  ----------
  mask_rle : str
      RLE string, e.g. "3 5 10 2"
  shape : tuple
      (height, width) of output mask

  Returns
  -------
  mask : np.ndarray
      (H, W) uint8 mask of 0s and 1s
  '''

  h, w = shape

  if not isinstance(mask_rle, str) or mask_rle == '':
    return np.zeros((h, w), dtype=np.uint8)

  s = list(map(int, mask_rle.strip().split()))
  starts = np.asarray(s[0::2]) - 1  # convert to 0-index
  lengths = np.asarray(s[1::2])

  ends = starts + lengths

  # Prepare flat mask
  mask = np.zeros(h * w, dtype=np.uint8)
  for lo, hi in zip(starts, ends):
    mask[lo:hi] = 1

  # Reshape to 2D (Fortran order → column-major)
  mask = mask.reshape((w, h)).T

  return mask

# functions
