#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the Keras criterion registry.

The names are the ones the configs use, and they are the same constants
the configuration layer validates against.
'''

import pytest

from birdcall.config_params.criterion_params import CRITERION_NAMES
from birdcall.losses.tf_registry import CRITERION_REGISTRY
from birdcall.losses.tf_registry import build_criterion


def test_the_criterion_registry_holds_the_five_criteria() -> None:
  '''The five reachable criteria are registered under their config names.'''
  assert sorted(CRITERION_REGISTRY) == [
      'bce_controlled_standard_loss',
      'focal_loss_standard',
      'sed_removed_focal_loss',
      'sed_scaled_pos_neg_focal_loss',
      'sed_scaled_pos_neg_focal_loss_augd',
  ]
  # The registry and the configuration layer agree on the names.
  assert sorted(CRITERION_REGISTRY) == sorted(CRITERION_NAMES)


def test_the_criteria_are_built_with_their_parameters() -> None:
  '''The hyper-parameters reach the criterion.'''
  criterion = build_criterion('sed_scaled_pos_neg_focal_loss', gamma=0.0,
                               alpha_0=1.0, alpha_1=1.0,
                               secondary_factor=1.0)
  assert criterion.gamma == 0.0
  assert criterion.alpha_0 == 1.0
  assert criterion.alpha_1 == 1.0
  assert criterion.secondary_factor == 1.0
  assert criterion.loss_keys == ['bce_loss', 'F_loss', 'FScaled_loss',
                                 'F_loss_0', 'F_loss_1']


def test_an_unknown_criterion_raises() -> None:
  '''An unknown criterion name is a clear error.'''
  with pytest.raises(ValueError, match='Unknown criterion'):
    build_criterion('sed_scaled_focal_loss')


def test_the_bce_controlled_criterion_has_no_loss_keys() -> None:
  '''The BCE criterion publishes no component metrics, on purpose.'''
  criterion = build_criterion('bce_controlled_standard_loss', beta=0.5)
  assert criterion.loss_keys == []
  assert criterion.beta == 0.5
