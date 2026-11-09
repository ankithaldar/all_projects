#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for storage capacity rules and manufacturing BOM use.'''

from collections import Counter

from world_of_supply.facility import FacilityConfig, LumberFactoryCell
from world_of_supply.manufacturing import BillOfMaterials
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.storage import StorageEconomy, StorageUnit


def make_storage(capacity=20):
  '''Build a storage unit with unit holding cost one.

  Args:
    capacity: Maximum units the storage may hold.

  Returns:
    StorageUnit: Empty storage unit ready for deposits.
  '''
  return StorageUnit(max_capacity=capacity, economy=StorageEconomy(1))


def build_factory_with_inputs(inputs, capacity=20):
  '''Build a lumber factory with a custom recipe and empty storage.

  Args:
    inputs: Mapping of input product id to units per lot.
    capacity: Storage capacity override for the factory.

  Returns:
    LumberFactoryCell: Factory whose storage holds no stock.
  '''
  world = WorldBuilder.build(ScenarioConfig())
  factory = world.get_facilities(LumberFactoryCell)[0]
  factory.bom = BillOfMaterials(Counter(inputs), output_product_id='toy_car')
  factory.storage.max_capacity = capacity
  factory.storage.stock_levels.clear()
  return factory


def test_add_units_respects_all_or_nothing():
  '''Verify oversized all-or-nothing deposit is fully rejected.

  Returns:
    None: Test completes without assertion failure.
  '''
  storage = make_storage()
  deposited = storage.try_add_units({'steel': 25}, all_or_nothing=True)
  assert deposited == {}
  assert storage.used_capacity() == 0


def test_partial_deposit_when_not_all_or_nothing():
  '''Verify partial deposit fills storage up to capacity.

  Returns:
    None: Test completes without assertion failure.
  '''
  storage = make_storage(capacity=5)
  deposited = storage.try_add_units({'steel': 8}, all_or_nothing=False)
  assert deposited == {'steel': 5}
  assert storage.used_capacity() == 5


def test_take_units_is_atomic():
  '''Verify withdrawal fails atomically when stock is short.

  Returns:
    None: Test completes without assertion failure.
  '''
  storage = make_storage()
  storage.try_add_units({'steel': 3})
  assert not storage.try_take_units({'steel': 4})
  assert storage.stock_levels['steel'] == 3


def test_take_available_returns_at_most_stock():
  '''Verify take_available returns at most the stocked amount.

  Returns:
    None: Test completes without assertion failure.
  '''
  storage = make_storage()
  storage.try_add_units({'toy_car': 4})
  taken = storage.take_available('toy_car', 9)
  assert taken == 4
  assert storage.used_capacity() == 0


def test_storage_books_holding_cost():
  '''Verify storage act books per-unit holding cost as loss.

  Returns:
    None: Test completes without assertion failure.
  '''
  storage = make_storage()
  storage.try_add_units({'steel': 6})
  sheet = storage.act(None)
  assert sheet.loss == -6


def test_manufacturing_consumes_bom_and_produces_output():
  '''Verify manufacturing consumes BOM inputs and produces output.

  Returns:
    None: Test completes without assertion failure.
  '''

  class Control:
    '''Minimal manufacturing control for one production lot.'''

    production_rate = 2
    unit_price = None
    consumer_product_id = None
    consumer_source_id = None
    consumer_quantity = None

  factory = build_factory_with_inputs({'lumber': 1})
  factory.storage.try_add_units({'lumber': 3})
  factory.manufacturing.act(Control())
  assert factory.storage.stock_levels['toy_car'] == 2
  assert factory.storage.stock_levels['lumber'] == 1


def test_manufacturing_requires_full_bom():
  '''Verify manufacturing produces nothing without the full BOM.

  Returns:
    None: Test completes without assertion failure.
  '''

  class Control:
    '''Minimal manufacturing control for one production lot.'''

    production_rate = 1
    unit_price = None
    consumer_product_id = None
    consumer_source_id = None
    consumer_quantity = None

  factory = build_factory_with_inputs({'lumber': 1, 'steel': 1})
  factory.storage.try_add_units({'lumber': 2})
  sheet = factory.manufacturing.act(Control())
  assert sheet.total() == 0
  assert 'toy_car' not in factory.storage.stock_levels


def test_facility_config_defaults_documented():
  '''Verify FacilityConfig documents its starting balance default.

  Returns:
    None: Test completes without assertion failure.
  '''
  config = FacilityConfig(bill_of_materials=BillOfMaterials(Counter()))
  assert config.initial_balance == 1000
  assert isinstance(config.initial_balance, int)
