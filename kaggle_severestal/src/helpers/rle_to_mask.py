#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''
#https://www.kaggle.com/code/stainsby/fast-tested-rle

# imports
import numpy as np
#    script imports
# imports


# constants
DEFAULT_IMAGE_SHAPE = (1600, 256)
# constants

# functions
def rle_to_mask(mask_rle, shape=(1600, 256)):
  '''
  mask_rle: run-length as string formated (start length)
  shape: (width,height) of array to return
  Returns numpy array, 1 - mask, 0 - background
  '''
  s = mask_rle.split()
  starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
  mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
  for lo, hi in zip(starts, lengths):
    mask[lo - 1:lo - 1 + hi] = 1
  return mask.reshape(shape).T





# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def mask_to_rle(img):
  """
  img: numpy array, 1 - mask, 0 - background
  Returns run length as string formatted
  """
  pixels = img.flatten()
  pixels = np.concatenate([[0], pixels, [0]])
  runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
  runs[1::2] -= runs[::2]
  return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle: str, label=1, shape=DEFAULT_IMAGE_SHAPE):
  """
  mask_rle: run-length as string formatted (start length)
  shape: (height,width) of array to return
  Returns numpy array, 1 - mask, 0 - background

  """
  s = mask_rle.split()
  starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
  starts -= 1
  ends = starts + lengths
  img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
  for lo, hi in zip(starts, ends):
    img[lo:hi] = label
  return img.reshape(shape)  # Needed to align to RLE direction
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
