#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the AdamW optimiser and of the learning rate schedule.

The schedule is checked against the golden fixture of
``birdcall.engines.schedules``: for ``lr=0.001``, ``lr_scale_factor=0.01``,
``epoch_length=7`` and ``num_epochs=5`` the steps 0, 1, 2, 33 and 34 give
``1e-5``, ``0.00015143``, ``0.00029286``, ``0.00016292`` and
``0.00013223``.
'''

from pathlib import Path

import pytest
import torch

from birdcall.config_params.checkpoint_params import CheckpointParams
from birdcall.config_params.dataset_params import DatasetParams
from birdcall.config_params.parameters import Parameters
from birdcall.config_params.run_params import RunParams
from birdcall.engines.schedules import schedule_multiplier
from birdcall.optimizers.pl_optimizer_adamw import build_optimizer
from birdcall.optimizers.pl_optimizer_adamw import build_scheduler

GOLDEN_LR = 0.001
GOLDEN_SCALE = 0.01
GOLDEN_EPOCH_LENGTH = 7
GOLDEN_EPOCHS = 5
GOLDEN = {
    0: 0.00001,
    1: 0.00015143,
    2: 0.00029286,
    33: 0.00016292,
    34: 0.00013223,
}


def _parameters(**overrides) -> Parameters:
  '''Builds a validated configuration for the tests.

  Args:
    **overrides: The attributes that differ from the default.

  Returns:
    The configuration.
  '''
  dataset = DatasetParams(root_dir=Path('/tmp'),
                          csv_dir=Path('/tmp/fold.csv'),
                          period=30,
                          isTraining=True,
                          num_test_samples=1)
  payload = {
      'train_ds_params': dataset,
      'valid_ds_params': dataset,
      'test_ds_params': dataset,
      'lr': GOLDEN_LR,
      'lr_scale_factor': GOLDEN_SCALE,
      'scheduler_name': 'warmup_with_cosine',
      'run_params': RunParams(max_epochs=GOLDEN_EPOCHS,
                              epoch_length=GOLDEN_EPOCH_LENGTH),
      'checkpoint_params': CheckpointParams(save_dir=Path('/tmp/ckpt')),
  }
  payload.update(overrides)
  return Parameters(**payload)


def _model() -> torch.nn.Module:
  '''Builds a tiny model.

  Returns:
    The model.
  '''
  return torch.nn.Linear(3, 4)


def test_optimizer_is_the_originals_adamw():
  '''The optimiser keeps every argument of the original's.'''
  parameters = _parameters(weight_decay=0.001)
  optimizer = build_optimizer(parameters, _model())
  assert isinstance(optimizer, torch.optim.AdamW)
  group = optimizer.param_groups[0]
  assert group['lr'] == pytest.approx(0.001)
  assert group['weight_decay'] == pytest.approx(0.001)
  assert group['betas'] == (0.9, 0.999)
  assert group['eps'] == pytest.approx(1e-08)
  assert group['amsgrad'] is True


def test_optimizer_rejects_an_unknown_name():
  '''An unknown optimiser name is a clear error.'''
  parameters = _parameters()
  parameters.optimizer_name = 'sgd'
  with pytest.raises(ValueError, match='Unknown optimizer'):
    build_optimizer(parameters, _model())


def test_the_schedule_matches_the_golden_fixture():
  '''The learning rate curve is the original's, step for step.'''
  parameters = _parameters()
  optimizer = build_optimizer(parameters, _model())
  scheduler = build_scheduler(parameters, optimizer, GOLDEN_EPOCH_LENGTH)
  learning_rates = [optimizer.param_groups[0]['lr']]
  for _ in range(34):
    scheduler.step()
    learning_rates.append(optimizer.param_groups[0]['lr'])
  for step, expected in GOLDEN.items():
    assert learning_rates[step] == pytest.approx(expected, abs=5e-9), (
        f'the learning rate of step {step} is {learning_rates[step]}')


def test_the_schedule_agrees_with_the_shared_multiplier():
  '''Every step of the curve is the shared multiplier of the step.'''
  parameters = _parameters()
  optimizer = build_optimizer(parameters, _model())
  scheduler = build_scheduler(parameters, optimizer, GOLDEN_EPOCH_LENGTH)
  for step in range(GOLDEN_EPOCHS * GOLDEN_EPOCH_LENGTH):
    assert optimizer.param_groups[0]['lr'] == pytest.approx(
        GOLDEN_LR * schedule_multiplier('warmup_with_cosine', step,
                                        GOLDEN_EPOCH_LENGTH, GOLDEN_EPOCHS,
                                        GOLDEN_SCALE))
    scheduler.step()


def test_the_warmup_reaches_the_base_rate_at_the_phase_boundary():
  '''The two phases agree at the boundary, which is the base rate.

  The linear phase covers the first ``epoch_length`` steps, so its last
  sample is the step before the peak of the triangle, and the cosine
  phase starts exactly at the peak.
  '''
  parameters = _parameters()
  optimizer = build_optimizer(parameters, _model())
  scheduler = build_scheduler(parameters, optimizer, GOLDEN_EPOCH_LENGTH)
  learning_rates = [optimizer.param_groups[0]['lr']]
  for _ in range(GOLDEN_EPOCH_LENGTH):
    scheduler.step()
    learning_rates.append(optimizer.param_groups[0]['lr'])
  assert learning_rates[GOLDEN_EPOCH_LENGTH - 1] < GOLDEN_LR
  assert learning_rates[GOLDEN_EPOCH_LENGTH] == pytest.approx(GOLDEN_LR)


def test_the_none_schedule_keeps_the_learning_rate():
  '''The schedule named none does not move the learning rate.'''
  parameters = _parameters(scheduler_name='none')
  optimizer = build_optimizer(parameters, _model())
  scheduler = build_scheduler(parameters, optimizer, GOLDEN_EPOCH_LENGTH)
  for _ in range(10):
    assert optimizer.param_groups[0]['lr'] == pytest.approx(GOLDEN_LR)
    scheduler.step()


def test_the_one_cycle_schedule_starts_at_the_base_rate():
  '''The cosine phase alone starts at the base learning rate.'''
  parameters = _parameters(scheduler_name='one_cycle_cosine')
  optimizer = build_optimizer(parameters, _model())
  scheduler = build_scheduler(parameters, optimizer, GOLDEN_EPOCH_LENGTH)
  assert optimizer.param_groups[0]['lr'] == pytest.approx(GOLDEN_LR)
  scheduler.step()
  assert optimizer.param_groups[0]['lr'] < GOLDEN_LR


def test_the_scheduler_is_a_lambda_lr():
  '''The schedule is the LambdaLR the trainer drives per step.'''
  parameters = _parameters()
  optimizer = build_optimizer(parameters, _model())
  scheduler = build_scheduler(parameters, optimizer, GOLDEN_EPOCH_LENGTH)
  assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
  assert scheduler.optimizer is optimizer
