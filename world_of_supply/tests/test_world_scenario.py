#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the world engine and scenario builder.'''

import pytest

from world_of_supply.facility import RetailerCell, ToyFactoryCell, WarehouseCell
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.world import Control


@pytest.fixture
def world():
  return WorldBuilder.build(ScenarioConfig(), seed=11)


def test_scenario_registers_nine_facilities(world):
  assert len(world.facilities) == 9
  assert len(world.get_facilities(ToyFactoryCell)) == 3
  assert len(world.get_facilities(WarehouseCell)) == 2
  assert len(world.get_facilities(RetailerCell)) == 2


def test_all_supplier_pairs_are_connected(world):
  for facility in world.facilities.values():
    if facility.consumer is None:
      continue
    for source in facility.consumer.sources:
      path = world.find_path((facility.x, facility.y), (source.x, source.y))
      assert path is not None, f'{facility.id} -> {source.id} unreachable'


def test_act_returns_sheet_per_facility_and_advances_time(world):
  outcome = world.act(Control({}))
  assert set(outcome.facility_step_balance_sheets) == set(world.facilities)
  assert world.time_step == 1


def test_global_balance_equals_sum_of_facilities(world):
  global_before = world.economy.global_balance().total()
  per_facility = sum(sheet.total() for sheet in world.act(Control({})).facility_step_balance_sheets.values())
  global_after = world.economy.global_balance().total()
  assert global_after - global_before == per_facility


def test_seed_produces_identical_layouts():
  first = WorldBuilder.build(ScenarioConfig(), seed=99)
  second = WorldBuilder.build(ScenarioConfig(), seed=99)
  cells_first = [type(c).__name__ for column in first.grid for c in column]
  cells_second = [type(c).__name__ for column in second.grid for c in column]
  assert cells_first == cells_second
