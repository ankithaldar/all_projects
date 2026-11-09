#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the world engine and scenario builder.'''

import pytest

from world_of_supply.facility import (
    RetailerCell,
    ToyFactoryCell,
    WarehouseCell,
)
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.world import Control


@pytest.fixture
def world():
  '''Build the shared seeded world fixture.

  Returns:
    World: Fresh world built with scenario defaults and seed 11.
  '''
  return WorldBuilder.build(ScenarioConfig(), seed=11)


def test_scenario_registers_nine_facilities(world):
  '''Verify default scenario registers nine facilities by tier.

  Args:
    world: Seeded world built by the world fixture.

  Returns:
    None: Test completes without assertion failure.
  '''
  assert len(world.facilities) == 9
  assert len(world.get_facilities(ToyFactoryCell)) == 3
  assert len(world.get_facilities(WarehouseCell)) == 2
  assert len(world.get_facilities(RetailerCell)) == 2


def test_all_supplier_pairs_are_connected(world):
  '''Verify every consumer source is reachable by rail path.

  Args:
    world: Seeded world built by the world fixture.

  Returns:
    None: Test completes without assertion failure.
  '''
  for facility in world.facilities.values():
    if facility.consumer is None:
      continue
    for source in facility.consumer.sources:
      start = (facility.x, facility.y)
      goal = (source.x, source.y)
      path = world.find_path(start, goal)
      assert path is not None, f'{facility.id} -> {source.id} unreachable'


def test_act_returns_sheet_per_facility_and_advances_time(world):
  '''Verify one tick returns a sheet per facility and advances time.

  Args:
    world: Seeded world built by the world fixture.

  Returns:
    None: Test completes without assertion failure.
  '''
  outcome = world.act(Control({}))
  assert set(outcome.facility_step_balance_sheets) == set(world.facilities)
  assert world.time_step == 1


def test_global_balance_equals_sum_of_facilities(world):
  '''Verify global balance change equals sum of facility sheets.

  Args:
    world: Seeded world built by the world fixture.

  Returns:
    None: Test completes without assertion failure.
  '''
  global_before = world.economy.global_balance().total()
  sheets = world.act(Control({})).facility_step_balance_sheets.values()
  per_facility = sum(sheet.total() for sheet in sheets)
  global_after = world.economy.global_balance().total()
  assert global_after - global_before == per_facility


def test_seed_produces_identical_layouts():
  '''Verify identical seeds produce identical facility layouts.

  Returns:
    None: Test completes without assertion failure.
  '''
  first = WorldBuilder.build(ScenarioConfig(), seed=99)
  second = WorldBuilder.build(ScenarioConfig(), seed=99)
  cells_first = [type(c).__name__ for column in first.grid for c in column]
  cells_second = [type(c).__name__ for column in second.grid for c in column]
  assert cells_first == cells_second
