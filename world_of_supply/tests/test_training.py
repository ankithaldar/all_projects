#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the curriculum mapping and model-description helpers.'''

from world_of_supply.rl.training import (
    CURRICULUM_ATTR,
    apply_curriculum,
    make_policy_mapping_fn,
)


def test_toy_only_mapping_starts_frozen_elsewhere():
  mapping, state = make_policy_mapping_fn(train_toy_factories_only=True)
  assert mapping('ToyFactoryCell_3p') == 'ppo_producer'
  assert mapping('ToyFactoryCell_3c') == 'ppo_consumer'
  assert mapping('WarehouseCell_6p') == 'frozen_producer'
  assert mapping('RetailerCell_9c') == 'frozen_consumer'
  assert state['trainable_prefixes'] == {'ToyFactoryCell'}


def test_default_mapping_trains_everything():
  mapping, state = make_policy_mapping_fn(train_toy_factories_only=False)
  assert mapping('SteelFactoryCell_1p') == 'ppo_producer'
  assert state['all_trainable'] is True


def test_curriculum_promotes_prefixes_at_thresholds():
  mapping, state = make_policy_mapping_fn(train_toy_factories_only=True)

  assert mapping('WarehouseCell_6p') == 'frozen_producer'
  promoted = apply_curriculum(state, iteration=2, n_iterations=8, schedule=((0.25, 'WarehouseCell'),))
  assert promoted == {'ToyFactoryCell', 'WarehouseCell'}
  assert mapping('WarehouseCell_6p') == 'ppo_producer'
  assert mapping('WarehouseCell_6c') == 'ppo_consumer'
  assert mapping('RetailerCell_9p') == 'frozen_producer'


def test_apply_curriculum_is_idempotent():
  _, state = make_policy_mapping_fn(train_toy_factories_only=True)
  schedule = ((0.25, 'WarehouseCell'), (0.35, 'WarehouseCell'))
  apply_curriculum(state, iteration=3, n_iterations=10, schedule=schedule)
  apply_curriculum(state, iteration=4, n_iterations=10, schedule=schedule)
  assert state['trainable_prefixes'] == {'ToyFactoryCell', 'WarehouseCell'}
