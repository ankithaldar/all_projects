#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the learning rate schedules.

The formulas are taken from ``pytorch-ignite 0.4.0.post1``.  They are
reproduced here because the schedule is the one place where a near-miss
changes the run quietly.

The golden fixture is published in
``docs/ports/pytorch-lightning-guide.md``: for ``lr=0.001``,
``lr_scale_factor=0.01``, ``epoch_length=7`` and ``num_epochs=5`` the
iterations 0, 1, 2, 33 and 34 give ``1e-5``, ``0.00015143``,
``0.00029286``, ``0.00016292`` and ``0.00013223``.
'''

import pytest

from birdcall.engines.schedules import SCHEDULE_NAMES
from birdcall.engines.schedules import cosine_multiplier
from birdcall.engines.schedules import linear_multiplier
from birdcall.engines.schedules import one_cycle_cosine_multiplier
from birdcall.engines.schedules import schedule_multiplier
from birdcall.engines.schedules import warmup_with_cosine_multiplier

LR = 0.001
SCALE_FACTOR = 0.01
EPOCH_LENGTH = 7
NUM_EPOCHS = 5


def test_the_three_schedule_names_are_published():
  '''The names are the ones the configuration layer validates.'''
  assert SCHEDULE_NAMES == ('none', 'warmup_with_cosine', 'one_cycle_cosine')


def test_the_golden_fixture_of_the_warmup_schedule():
  '''The five published values of the warmup schedule.'''
  expected = [0.00001, 0.00015143, 0.00029286, 0.00016292, 0.00013223]
  for index, value in zip([0, 1, 2, 33, 34], expected):
    multiplier = warmup_with_cosine_multiplier(index, EPOCH_LENGTH,
                                               NUM_EPOCHS, SCALE_FACTOR)
    assert multiplier * LR == pytest.approx(value, rel=1e-4, abs=1e-9)


def test_the_linear_phase_runs_over_a_two_epoch_triangle():
  '''Only the rising half of the triangle is traversed.'''
  # The first step of the linear phase is the scale factor.
  assert warmup_with_cosine_multiplier(0, EPOCH_LENGTH, NUM_EPOCHS,
                                        SCALE_FACTOR) == pytest.approx(
                                            SCALE_FACTOR)
  # The last step of the linear phase is still below one.
  assert warmup_with_cosine_multiplier(EPOCH_LENGTH - 1, EPOCH_LENGTH,
                                        NUM_EPOCHS,
                                        SCALE_FACTOR) < 1.0


def test_the_cosine_phase_restarts_its_own_counter():
  '''The boundary between the two phases is not smoothed.'''
  linear = warmup_with_cosine_multiplier(EPOCH_LENGTH - 1, EPOCH_LENGTH,
                                         NUM_EPOCHS, SCALE_FACTOR)
  cosine = warmup_with_cosine_multiplier(EPOCH_LENGTH, EPOCH_LENGTH,
                                         NUM_EPOCHS, SCALE_FACTOR)
  assert cosine == pytest.approx(1.0)
  assert linear != pytest.approx(cosine)


def test_the_cosine_phase_decreases_over_the_run():
  '''The cosine phase never rises again.'''
  values = [warmup_with_cosine_multiplier(index, EPOCH_LENGTH, NUM_EPOCHS,
                                          SCALE_FACTOR)
            for index in range(EPOCH_LENGTH, NUM_EPOCHS * EPOCH_LENGTH)]
  assert all(first >= second
             for first, second in zip(values, values[1:]))
  assert values[-1] < values[0]
  # The cosine phase's own counter restarts at zero, so its last step is
  # still above the scale factor.
  assert values[-1] > SCALE_FACTOR


def test_the_linear_multiplier_is_a_triangle():
  '''The linear schedule goes down and then up again.'''
  assert linear_multiplier(0, 10, 1.0, 0.0) == pytest.approx(1.0)
  assert linear_multiplier(5, 10, 1.0, 0.0) == pytest.approx(0.0)
  assert linear_multiplier(10, 10, 1.0, 0.0) == pytest.approx(1.0)


def test_the_cosine_multiplier_goes_from_start_to_end():
  '''The cosine schedule starts at the start and ends at the end.'''
  assert cosine_multiplier(0, 10, 1.0, 0.0) == pytest.approx(1.0)
  assert cosine_multiplier(10, 10, 1.0, 0.0) == pytest.approx(0.0)
  assert cosine_multiplier(5, 10, 1.0, 0.0) == pytest.approx(0.5)


def test_the_one_cycle_schedule_is_the_cosine_alone():
  '''The cosine phase covers the whole run.'''
  for index in (0, 3, 17, 34):
    assert one_cycle_cosine_multiplier(index, NUM_EPOCHS, EPOCH_LENGTH,
                                        SCALE_FACTOR) == pytest.approx(
                                            cosine_multiplier(
                                                index,
                                                NUM_EPOCHS * EPOCH_LENGTH,
                                                1.0, SCALE_FACTOR))


def test_the_none_schedule_is_a_flat_one():
  '''The flat schedule never scales the learning rate.'''
  for index in (0, 7, 34):
    assert schedule_multiplier('none', index, EPOCH_LENGTH, NUM_EPOCHS,
                                SCALE_FACTOR) == pytest.approx(1.0)


def test_the_dispatch_returns_the_named_schedule():
  '''The dispatch is the one the engine calls.'''
  for index in (0, 5, 20):
    assert schedule_multiplier(
        'warmup_with_cosine', index, EPOCH_LENGTH, NUM_EPOCHS,
        SCALE_FACTOR) == pytest.approx(
            warmup_with_cosine_multiplier(index, EPOCH_LENGTH, NUM_EPOCHS,
                                          SCALE_FACTOR))
    assert schedule_multiplier(
        'one_cycle_cosine', index, EPOCH_LENGTH, NUM_EPOCHS,
        SCALE_FACTOR) == pytest.approx(
            one_cycle_cosine_multiplier(index, NUM_EPOCHS, EPOCH_LENGTH,
                                        SCALE_FACTOR))


def test_an_unknown_schedule_is_refused():
  '''An unknown name names the valid ones.'''
  with pytest.raises(ValueError, match='Unknown schedule'):
    schedule_multiplier('cyclic', 0, EPOCH_LENGTH, NUM_EPOCHS, SCALE_FACTOR)
