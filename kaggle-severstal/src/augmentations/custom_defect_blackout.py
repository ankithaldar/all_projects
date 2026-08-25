#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Custom Blackout for Defects'''


# imports
import random
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np

#    script imports
# imports


# constants
# constants


# classes
class CustomDefectBlackout(A.DualTransform):
  '''
  Randomly black out image regions and corresponding mask channels for each defect class.

  This transform generates four independent random numbers r1..r4 from Uniform(0,1).
  For each class n (1..4) if r_n > 0.5, it:
    - Sets the image pixels to 0 where the mask channel n is active (value == 1).
    - Sets the whole mask channel n to 0 (i.e., removes that defect).

  The transformation is applied only with probability `p` (default 0.5).
  It is fully deterministic when a fixed random_state (seed) is used.

  Targets:
    image, mask

  Args:
    always_apply (bool): If True, the transform is always applied.
    p (float): Probability of applying the transform. Default: 0.5.

  Example:
    >>> import albumentations as A
    >>> import numpy as np
    >>> image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    >>> mask = np.zeros((256, 256, 4), dtype=np.uint8)
    >>> mask[50:100, 50:100, 0] = 1   # defect class 1
    >>> mask[150:200, 150:200, 2] = 1 # defect class 3
    >>> transform = A.Compose([CustomDefectBlackout(p=1.0)])
    >>> transformed = transform(image=image, mask=mask)
    >>> # Now some defect classes may be blacked out.
  '''

  def __init__(self, always_apply: bool = False, p: float = 0.5):
    super().__init__(always_apply=always_apply, p=p)
    # Fixed number of defect classes as per problem statement
    self.num_classes = 4
    self.blackout_threshold = 0.5

  @property
  def targets_as_params(self) -> List[str]:
    '''Mask is needed to compute blackout regions.'''
    return ['mask']

  def get_params_dependent_on_targets(self, params: Dict[str, np.ndarray]) -> Dict:
    '''
    Compute which classes to black out and the combined pixel mask.

    Args:
      params: Dictionary with key 'mask' containing a numpy array of shape (H, W, 4)
          with binary values (0 or 1).

    Returns:
      Dictionary with:
        - 'selected_channels': list of channel indices (0‑based) that will be blacked out.
        - 'blackout_mask': boolean array of shape (H, W) indicating pixels that must be
                  set to zero in the image (union of selected classes).
    '''
    mask = params['mask']
    assert mask.shape[-1] == self.num_classes, (
      f'Mask must have {self.num_classes} channels, got {mask.shape[-1]}'
    )

    # Generate one random number per class
    r = self.random.rand(self.num_classes)
    selected_channels = [i for i in range(self.num_classes) if r[i] > self.blackout_threshold]

    # Combined pixel mask where any selected class is active
    if selected_channels:
      # Using any on the boolean mask (non‑zero entries are active)
      combined = np.any(mask[:, :, selected_channels], axis=-1)
    else:
      combined = np.zeros(mask.shape[:2], dtype=bool)

    return {'selected_channels': selected_channels, 'blackout_mask': combined}

  def apply(self, img: np.ndarray, blackout_mask: np.ndarray = None, **params) -> np.ndarray:
    '''
    Black out image pixels where any selected defect class is active.

    Args:
      img: Input image (H, W, C) or (H, W). Blackout mask is 2D boolean.
      blackout_mask: Boolean mask of shape (H, W) indicating pixels to set to zero.

    Returns:
      Modified image with zeros at selected pixels.
    '''
    if blackout_mask is None or not np.any(blackout_mask):
      return img
    # Set all channels of selected pixels to 0
    img[blackout_mask] = 0
    return img

  def apply_to_mask(self, mask: np.ndarray, selected_channels: List[int] = None, **params) -> np.ndarray:
    '''
    Remove selected defect classes from the mask by setting the whole channels to zero.

    Args:
      mask: Input mask of shape (H, W, 4).
      selected_channels: List of channel indices to be blacked out.

    Returns:
      Modified mask with selected channels set to zero.
    '''
    if selected_channels is None:
      return mask
    for ch in selected_channels:
      mask[:, :, ch] = 0
    return mask

  def get_transform_init_args_names(self) -> Tuple[str, ...]:
    '''Names of custom arguments defined in __init__ (none in this case).'''
    return ()
# classes
