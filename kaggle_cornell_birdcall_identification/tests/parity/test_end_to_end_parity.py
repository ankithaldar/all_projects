#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The cross-backend parity of the numeric core.

The two backends rewrite the whole numeric core, so the numbers
themselves are compared: the losses on the same inputs, and the front
end that feeds them.  A full one-epoch run would need the weight
conversion, which is a piece of work of its own.
'''

import numpy as np
import pytest
import torch

from birdcall.losses.pl_loss_bce_controlled_standard import (
    BCEControlledStandardLoss as PlBce)
from birdcall.losses.pl_loss_focal_standard import FocalLossStandard as PlFocal
from birdcall.losses.pl_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss as PlScaled)
from birdcall.losses.tf_loss_bce_controlled_standard import (
    BCEControlledStandardLoss as TfBce)
from birdcall.losses.tf_loss_focal_standard import FocalLossStandard as TfFocal
from birdcall.losses.tf_loss_scaled_pos_neg_focal import (
    SedScaledPosNegFocalLoss as TfScaled)

BATCH = 3
SNIPPETS = 2
CLASSES = 12


def batch(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''Builds a deterministic batch of labels, secondary labels and scores.

  Args:
    seed: The seed of the generator.

  Returns:
    A tuple ``(labels, secondary labels, predictions)``.
  '''
  rng = np.random.default_rng(seed)
  labels = (rng.random((BATCH, SNIPPETS, CLASSES)) > 0.7).astype('float32')
  secondary = np.zeros_like(labels)
  secondary[:, :, :3] = labels[:, :, :3]
  predictions = rng.random((BATCH, SNIPPETS, CLASSES)).astype('float32')
  return labels, secondary, predictions


@pytest.mark.parity
def test_the_scaled_focal_losses_agree() -> None:
  '''The live criterion gives the same number in both backends.'''
  labels, secondary, predictions = batch()
  torch_loss, _ = PlScaled(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                           secondary_factor=1.0)(
      torch.from_numpy(predictions),
      {'all_labels': torch.from_numpy(labels),
       'secondary_labels': torch.from_numpy(secondary)})
  keras_loss = TfScaled(gamma=0.0, alpha_0=1.0, alpha_1=1.0,
                        secondary_factor=1.0)(
      y_true={'all_labels': labels, 'secondary_labels': secondary},
      y_pred=predictions)
  assert float(keras_loss) == pytest.approx(float(torch_loss), abs=1e-6)


@pytest.mark.parity
def test_the_plain_focal_losses_agree() -> None:
  '''The plain focal loss gives the same number in both backends.'''
  labels, secondary, predictions = batch(seed=1)
  torch_loss, _ = PlFocal(gamma=0.0, alpha=1.0)(
      torch.from_numpy(predictions),
      {'all_labels': torch.from_numpy(labels),
       'secondary_labels': torch.from_numpy(secondary)})
  keras_loss = TfFocal(gamma=0.0, alpha=1.0)(
      y_true={'all_labels': labels, 'secondary_labels': secondary},
      y_pred=predictions)
  assert float(keras_loss) == pytest.approx(float(torch_loss), abs=1e-6)


@pytest.mark.parity
def test_the_bce_controlled_losses_agree() -> None:
  '''The winning criterion gives the same number in both backends.'''
  labels, _, predictions = batch(seed=2)
  torch_loss, _ = PlBce(beta=0.5)(
      torch.from_numpy(predictions), {'all_labels': torch.from_numpy(labels)})
  keras_loss = TfBce(beta=0.5)(y_true=labels, y_pred=predictions)
  assert float(keras_loss) == pytest.approx(float(torch_loss), abs=1e-6)


@pytest.mark.parity
def test_the_scaling_losses_agree() -> None:
  '''The secondary scaling gives the same number in both backends.'''
  labels, secondary, predictions = batch(seed=3)
  torch_loss, _ = PlScaled(gamma=2.0, alpha_0=1.0, alpha_1=1.0,
                           secondary_factor=0.1)(
      torch.from_numpy(predictions),
      {'all_labels': torch.from_numpy(labels),
       'secondary_labels': torch.from_numpy(secondary)})
  keras_loss = TfScaled(gamma=2.0, alpha_0=1.0, alpha_1=1.0,
                        secondary_factor=0.1)(
      y_true={'all_labels': labels, 'secondary_labels': secondary},
      y_pred=predictions)
  assert float(keras_loss) == pytest.approx(float(torch_loss), abs=1e-6)


@pytest.mark.parity
def test_the_learning_rate_schedules_agree() -> None:
  '''Both backends step the same schedule, once per batch.'''
  from birdcall.optimizers.pl_optimizer_adamw import build_scheduler as pl_build
  from birdcall.optimizers.tf_optimizer_adamw import build_scheduler as tf_build

  class RunParams:
    '''The two attributes the torch schedule reads.'''

    max_epochs = 5

  class Params:
    '''The attributes the torch schedule reads.'''

    scheduler_name = 'warmup_with_cosine'
    lr_scale_factor = 0.01
    run_params = RunParams()

  model = torch.nn.Linear(2, 2)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
  pl_schedule = pl_build(Params(), optimizer, 7)
  keras_schedule = tf_build({'scheduler_name': 'warmup_with_cosine',
                              'lr': 0.001, 'lr_scale_factor': 0.01,
                              'num_epochs': 5}, 7)
  for step in range(0, 40):
    assert float(keras_schedule(step)) == pytest.approx(
        float(pl_schedule.get_last_lr()[0]), abs=1e-12)
    pl_schedule.step()
