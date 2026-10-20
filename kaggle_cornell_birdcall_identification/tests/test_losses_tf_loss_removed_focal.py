#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the focal loss with the secondary species removed.

The secondary species must receive **exactly zero** gradient here,
which is the hard version of the down-weighting the scaled criterion
applies continuously.
'''

import numpy as np
import pytest

from birdcall.losses.tf_loss_removed_focal import SedRemovedFocalLoss

CLASSES = 264
BATCH = 2
SNIPPETS = 2


def batch(seed: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''Builds a deterministic batch of labels, secondary labels and scores.

  Args:
    seed: The seed of the generator.

  Returns:
    A tuple ``(labels, secondary labels, predictions)``.
  '''
  rng = np.random.default_rng(seed)
  labels = (rng.random((BATCH, SNIPPETS, CLASSES)) > 0.9).astype('float32')
  secondary = np.zeros_like(labels)
  secondary[:, :, :8] = labels[:, :, :8]
  predictions = rng.random((BATCH, SNIPPETS, CLASSES)).astype('float32')
  return labels, secondary, predictions


def test_the_loss_keys_are_published() -> None:
  '''The three component names are the documented ones.'''
  assert SedRemovedFocalLoss().loss_keys == ['bce_loss', 'F_loss',
                                             'FRemoved_loss']


def test_the_removed_loss_is_below_the_plain_focal_loss() -> None:
  '''The secondary positions contribute nothing.'''
  labels, secondary, predictions = batch()
  target = {'all_labels': labels, 'secondary_labels': secondary}
  criterion = SedRemovedFocalLoss(gamma=0.0)
  removed = float(criterion(y_true=target, y_pred=predictions))
  focal = float(criterion.components['F_loss'])
  assert removed < focal


def test_the_secondary_positions_are_exactly_zeroed() -> None:
  '''The removed positions carry no gradient at all.'''
  labels, secondary, predictions = batch()
  criterion = SedRemovedFocalLoss(gamma=0.0)
  criterion(y_true={'all_labels': labels, 'secondary_labels': secondary},
           y_pred=predictions)
  removed = float(criterion.components['FRemoved_loss'])
  # At gamma=0 the focal term is the BCE, so the removed loss is the
  # mean of the BCE over the positions that are not secondary.
  bce = -(labels * np.log(predictions) +
          (1.0 - labels) * np.log(1.0 - predictions))
  expected = float((bce * (1.0 - secondary)).mean())
  assert removed == pytest.approx(expected, abs=1e-5)


def test_without_secondary_labels_the_two_losses_agree() -> None:
  '''With no secondary label, the criterion is the plain focal loss.'''
  labels, _, predictions = batch()
  criterion = SedRemovedFocalLoss(gamma=0.0)
  value = float(criterion(y_true=labels, y_pred=predictions))
  focal = float(criterion.components['F_loss'])
  assert value == pytest.approx(focal, abs=1e-6)


def test_a_swapped_call_is_visibly_different() -> None:
  '''The argument order is inverted, and a swap does not raise.'''
  labels, secondary, predictions = batch()
  criterion = SedRemovedFocalLoss(gamma=0.0)
  straight = float(criterion(
      y_true={'all_labels': labels, 'secondary_labels': secondary},
      y_pred=predictions))
  swapped = float(criterion(y_true=predictions, y_pred=labels))
  assert straight != pytest.approx(swapped, abs=1e-4)
  assert np.isfinite(swapped)


def test_the_components_are_published() -> None:
  '''The three components are the instrumentation of the criterion.'''
  labels, secondary, predictions = batch()
  criterion = SedRemovedFocalLoss(gamma=0.0)
  criterion(y_true={'all_labels': labels, 'secondary_labels': secondary},
           y_pred=predictions)
  assert sorted(criterion.components) == sorted(criterion.loss_keys)


def test_the_config_round_trips() -> None:
  '''The configuration of the criterion can be read back.'''
  criterion = SedRemovedFocalLoss(gamma=3.0, alpha=0.5)
  config = criterion.get_config()
  assert config['gamma'] == 3.0
  assert config['alpha'] == 0.5
