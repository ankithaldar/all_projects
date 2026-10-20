#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the Keras optimiser and of the batch stepped schedule.

The schedule is the one place where a near miss changes the run
quietly, so its first values are checked against the golden fixture of
the shared module.
'''


import numpy as np
import pytest
import tensorflow as tf

from birdcall.engines.schedules import schedule_multiplier
from birdcall.optimizers.tf_optimizer_adamw import BatchSteppedSchedule
from birdcall.optimizers.tf_optimizer_adamw import build_optimizer
from birdcall.optimizers.tf_optimizer_adamw import build_scheduler


GOLDEN = {
    0: 1e-5,
    1: 0.00015143,
    2: 0.00029286,
    33: 0.00016292,
    34: 0.00013223,
}

def schedule() -> BatchSteppedSchedule:
  '''Builds the schedule of the golden fixture.

  Returns:
    The schedule.
  '''
  return build_scheduler({
      'scheduler_name': 'warmup_with_cosine',
      'lr': 0.001,
      'lr_scale_factor': 0.01,
      'num_epochs': 5,
  }, 7)

def test_the_schedule_matches_the_golden_fixture() -> None:
  '''The first values are the reference values.'''
  built = schedule()
  for step, expected in GOLDEN.items():
    assert float(built(step)) == pytest.approx(expected, abs=1e-8)

def test_the_schedule_matches_the_shared_implementation() -> None:
  '''The schedule is the shared one, step for step.'''
  built = schedule()
  for step in range(0, 40):
    expected = 0.001 * schedule_multiplier('warmup_with_cosine', step, 7, 5,
                                           0.01)
    assert float(built(step)) == pytest.approx(expected, abs=1e-12)

def test_the_phase_boundary_is_at_the_epoch_length() -> None:
  '''The linear phase covers exactly the first epoch_length steps.'''
  built = schedule()
  last_linear = float(built(6))
  first_cosine = float(built(7))
  assert last_linear != pytest.approx(first_cosine, abs=1e-9)
  assert float(built(6)) == pytest.approx(
      0.001 * schedule_multiplier('warmup_with_cosine', 6, 7, 5, 0.01),
      abs=1e-12)

def test_the_schedule_steps_per_batch() -> None:
  '''The schedule advances once per batch, not once per epoch.'''
  built = schedule()
  values = [float(built(step)) for step in range(5)]
  # The linear phase of the warmup rises, one batch at a time.
  assert values == sorted(values)
  assert len(set(values)) == 5

def test_the_schedule_works_inside_a_graph() -> None:
  '''The schedule is callable with a tensor step.'''
  built = schedule()

  @tf.function
  def value(step):
    return built(step)

  assert float(value(tf.constant(1, tf.int32))) == pytest.approx(
      GOLDEN[1], abs=1e-8)

def test_the_none_schedule_is_constant() -> None:
  '''The "none" schedule does not move the learning rate.'''
  built = build_scheduler({'scheduler_name': 'none', 'lr': 0.001}, 7)
  assert float(built(0)) == pytest.approx(0.001, abs=1e-12)
  assert float(built(100)) == pytest.approx(0.001, abs=1e-12)

def test_an_unknown_schedule_is_refused() -> None:
  '''An unknown schedule name raises.'''
  with pytest.raises(ValueError, match='Unknown schedule'):
    build_scheduler({'scheduler_name': 'cosine'}, 7)

def test_a_non_positive_epoch_length_is_refused() -> None:
  '''An epoch length of zero is refused.'''
  with pytest.raises(ValueError, match='epoch_length'):
    BatchSteppedSchedule('none', 0.001, 0, 5, 0.01)

def test_the_schedule_config_round_trips() -> None:
  '''The configuration of the schedule can be read back.'''
  config = schedule().get_config()
  assert config['schedule_name'] == 'warmup_with_cosine'
  assert config['base_lr'] == 0.001
  assert config['epoch_length'] == 7
  assert config['num_epochs'] == 5
  assert config['scale_factor'] == 0.01

def test_the_optimiser_is_adamw_with_amsgrad() -> None:
  '''The optimiser is the original's AdamW.'''
  optimizer = build_optimizer({'lr': 0.001, 'weight_decay': 0.01}, None)
  assert isinstance(optimizer, tf.keras.optimizers.AdamW)
  assert optimizer.amsgrad is True
  assert float(optimizer.weight_decay) == pytest.approx(0.01)
  assert float(optimizer.learning_rate) == pytest.approx(0.001, abs=1e-9)

def test_the_epsilon_is_the_original_one() -> None:
  '''Keras spells eps as epsilon, and the value is 1e-8.'''
  optimizer = build_optimizer({'lr': 0.001, 'epsilon': 1e-8}, None)
  assert float(optimizer.epsilon) == pytest.approx(1e-8)

def test_the_betas_are_the_original_ones() -> None:
  '''The betas are (0.9, 0.999).'''
  optimizer = build_optimizer({'lr': 0.001}, None)
  assert float(optimizer.beta_1) == pytest.approx(0.9)
  assert float(optimizer.beta_2) == pytest.approx(0.999)

def test_the_gradient_accumulation_is_optional() -> None:
  '''Keras only accepts an accumulation of two or more.'''
  optimizer = build_optimizer({'lr': 0.001,
                               'gradient_accumulation_steps': 1}, None)
  assert optimizer.gradient_accumulation_steps is None
  accumulated = build_optimizer({'lr': 0.001,
                                  'gradient_accumulation_steps': 4}, None)
  assert accumulated.gradient_accumulation_steps == 4

def test_the_learning_rate_can_be_a_schedule() -> None:
  '''The schedule is attached to the optimiser.'''
  optimizer = build_optimizer({'lr': 0.001}, None)
  optimizer.learning_rate = schedule()
  assert float(optimizer.learning_rate) == pytest.approx(GOLDEN[0], abs=1e-8)

def test_the_clipping_is_optional() -> None:
  '''The gradient clipping is disabled in every shipped config.'''
  optimizer = build_optimizer({'lr': 0.001, 'clipnorm': None}, None)
  assert optimizer.clipnorm is None
  clipped = build_optimizer({'lr': 0.001, 'clipnorm': 1.0}, None)
  assert float(clipped.clipnorm) == pytest.approx(1.0)

def test_the_optimizer_takes_a_model_it_does_not_need() -> None:
  '''The model argument is accepted for symmetry with the torch port.'''
  optimizer = build_optimizer({'lr': 0.001}, object())
  assert isinstance(optimizer, tf.keras.optimizers.AdamW)
  assert np.isfinite(float(optimizer.learning_rate))
