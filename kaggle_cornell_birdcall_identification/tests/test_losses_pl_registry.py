#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the criterion registry.

The names are the ones the configs use, and they are the same constants
the configuration layer validates against.  An unknown name is a clear
error, not a `KeyError`.
'''

import pytest

from birdcall.losses.pl_loss_bce_controlled_standard import (
    BCEControlledStandardLoss,
)
from birdcall.losses.pl_loss_focal_standard import FocalLossStandard
from birdcall.losses.pl_loss_removed_focal import SedRemovedFocalLoss
from birdcall.losses.pl_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss,
)
from birdcall.losses.pl_loss_scaled_pos_neg_focal_augd import (
    SedScaledPosNegFocalLossAugd,
)
from birdcall.losses.pl_registry import CRITERION_REGISTRY
from birdcall.losses.pl_registry import build_criterion


def test_build_criterion_returns_the_requested_criterion():
  '''The registry dispatches on the name and passes the parameters on.'''
  criterion = build_criterion('sed_scaled_pos_neg_focal_loss', gamma=0.0,
                              alpha_0=1.0, alpha_1=1.0, secondary_factor=1.0)
  assert isinstance(criterion, SedScaledPosNegFocalLoss)
  assert set(CRITERION_REGISTRY) == {
      'focal_loss_standard',
      'sed_removed_focal_loss',
      'sed_scaled_pos_neg_focal_loss',
      'sed_scaled_pos_neg_focal_loss_augd',
      'bce_controlled_standard_loss',
  }


def test_build_criterion_rejects_an_unknown_name():
  '''An unknown name is a clear error that names the valid ones.'''
  with pytest.raises(ValueError) as error:
    build_criterion('not_a_criterion')
  message = str(error.value)
  assert 'not_a_criterion' in message
  for name in CRITERION_REGISTRY:
    assert name in message


def test_the_registry_maps_every_name_to_its_criterion():
  '''The five reachable criteria are the five registered ones.'''
  assert CRITERION_REGISTRY['focal_loss_standard'] is FocalLossStandard
  assert CRITERION_REGISTRY['sed_removed_focal_loss'] is SedRemovedFocalLoss
  assert (CRITERION_REGISTRY['sed_scaled_pos_neg_focal_loss']
          is SedScaledPosNegFocalLoss)
  assert (CRITERION_REGISTRY['sed_scaled_pos_neg_focal_loss_augd']
          is SedScaledPosNegFocalLossAugd)
  assert (CRITERION_REGISTRY['bce_controlled_standard_loss']
          is BCEControlledStandardLoss)
