#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Seam tests for RL agents, env helpers, observations, training.'''

import numpy as np
import pytest

gymnasium = pytest.importorskip('gymnasium')

from world_of_supply.economy import BalanceSheet
from world_of_supply.rl.agents import FACILITY_PRODUCER_PRICE_INDEX
from world_of_supply.rl.agents import RETAILER_FACILITY_TYPES, is_retailer
from world_of_supply.rl.agents import producer_price_index
from world_of_supply.rl.env import EnvConfig, WorldOfSupplyEnv
from world_of_supply.rl.env import _build_agent_ids, _build_spaces
from world_of_supply.rl.env import _collect_facility_types
from world_of_supply.rl.env import _collect_product_ids
from world_of_supply.rl.env import _max_sources_per_facility
from world_of_supply.rl.observations import ObservationEncoder, normalize
from world_of_supply.rl.training import _apply_to_envs, _build_module_specs
from world_of_supply.rl.training import _build_policy_specs, _configure
from world_of_supply.rl.training import _probe_spaces
from world_of_supply.scenario import ScenarioConfig, WorldBuilder


class _SpecRecorder:
  '''Record keyword arguments passed to a fake specification class.'''

  def __init__(self, **kwargs):
    '''Store constructor keyword arguments for later inspection.

    Args:
      **kwargs: Values recorded verbatim as specification fields.

    Returns:
      None: Recorder is mutated in place.
    '''
    self.kwargs = kwargs


@pytest.fixture(scope='module')
def reference_world():
  '''Build the shared reference world for env helper seams.

  Returns:
    World: Fresh world built with scenario defaults.
  '''
  return WorldBuilder.build(ScenarioConfig(), seed=0)


def test_agents_retailer_and_price_index_seams():
  '''Verify retailer registry and producer price index lookups.

  Returns:
    None: Test completes without assertion failure.
  '''
  assert is_retailer('RetailerCell_8') is True
  assert is_retailer('SteelFactoryCell_1') is False
  assert producer_price_index('ToyFactoryCell') == 2
  assert producer_price_index('Nope') == 0
  assert RETAILER_FACILITY_TYPES == frozenset({'RetailerCell'})
  assert FACILITY_PRODUCER_PRICE_INDEX['RetailerCell'] == 6
  assert FACILITY_PRODUCER_PRICE_INDEX['WarehouseCell'] == 4


def test_env_collect_helpers(reference_world):
  '''Verify env helpers collect products, types, and sources.

  Args:
    reference_world: Reference world supplying facilities.

  Returns:
    None: Test completes without assertion failure.
  '''
  assert _collect_product_ids(reference_world) == ['lumber', 'steel', 'toy_car']
  types = _collect_facility_types(reference_world)
  expected_types = {'SteelFactoryCell': 0, 'LumberFactoryCell': 1}
  expected_types['ToyFactoryCell'] = 2
  expected_types['WarehouseCell'] = 3
  expected_types['RetailerCell'] = 4
  assert types == expected_types
  assert _max_sources_per_facility(reference_world) == 3
  agent_ids = _build_agent_ids(reference_world)
  assert len(agent_ids) == 18
  assert agent_ids[:2] == ['SteelFactoryCell_1p', 'SteelFactoryCell_1c']


def test_env_build_spaces_assigns_role_spaces(reference_world):
  '''Verify build spaces assigns producer and consumer spaces.

  Args:
    reference_world: Reference world supplying agent ids.

  Returns:
    None: Test completes without assertion failure.
  '''
  agent_ids = _build_agent_ids(reference_world)
  spaces = _build_spaces(
      agent_ids, total_dim=10, n_products=3, max_sources=2,
  )
  producer_space, consumer_space, observation_space, action_space = spaces
  assert list(producer_space.nvec) == [8, 6]
  assert list(consumer_space.nvec) == [3, 2, 6]
  assert set(observation_space.spaces) == set(agent_ids)
  first_producer = agent_ids[0]
  first_consumer = agent_ids[1]
  assert action_space.spaces[first_producer] is producer_space
  assert action_space.spaces[first_consumer] is consumer_space
  assert observation_space.spaces[first_producer].shape == (10,)


def test_env_accumulate_folds_noop_sheets():
  '''Verify accumulate folds no-op sheets into target sheets.

  Returns:
    None: Test completes without assertion failure.
  '''

  class Addition:
    '''Minimal outcome stand-in carrying no-op tick sheets.'''

    facility_step_balance_sheets = {
        'facility_a': BalanceSheet(5, -2),
        'facility_b': BalanceSheet(1, 0),
    }

  target = {'facility_a': BalanceSheet(10, -1)}
  WorldOfSupplyEnv._accumulate(target, Addition())
  assert target['facility_a'].total() == 12
  assert target['facility_b'].total() == 1


def test_env_set_iteration_publishes_curriculum_state():
  '''Verify set_iteration publishes iteration and total counts.

  Returns:
    None: Test completes without assertion failure.
  '''
  env = WorldOfSupplyEnv(EnvConfig(episode_duration=200, downsampling_rate=10))
  env.set_iteration(3, 10)
  assert env.current_iteration == 3
  assert env.n_iterations == 10


def test_observation_encode_helpers(reference_world):
  '''Verify observation helpers encode each feature block.

  Args:
    reference_world: Reference world supplying facilities.

  Returns:
    None: Test completes without assertion failure.
  '''
  env = WorldOfSupplyEnv(
      EnvConfig(episode_duration=200, downsampling_rate=10),
  )
  encoder = env.encoder
  facility = list(reference_world.facilities.values())[0]
  identity = encoder._encode_identity(facility)
  assert identity['is_positive_balance'] == 1
  assert sum(identity['facility_type']) == 1
  bom = encoder._encode_bom(facility)
  assert set(bom) == {'bom_inputs', 'bom_outputs'}
  storage = encoder._encode_storage(facility)
  assert storage['storage_capacity'] == facility.storage.max_capacity
  assert storage['storage_utilization'] == 0.0
  distribution = encoder._encode_distribution(facility)
  assert distribution['distributor_in_transit_orders'] == 0
  consumer = encoder._encode_consumer(facility)
  assert 'consumer_source_export_mask' in consumer
  global_state = encoder._encode_global(reference_world)
  assert global_state['global_time'] == 0.0
  assert len(global_state['global_storage_utilization']) == 9


def test_observation_normalize_guards_constant_vectors():
  '''Verify normalize maps constants to zeros and ranges to unit.

  Returns:
    None: Test completes without assertion failure.
  '''
  constant = normalize(np.array([1.0, 1.0, 1.0]))
  assert constant.tolist() == [0.0, 0.0, 0.0]
  scaled = normalize(np.array([0.0, 5.0, 10.0]))
  assert scaled.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_training_probe_spaces_matches_env():
  '''Verify probe spaces returns live producer and consumer spaces.

  Returns:
    None: Test completes without assertion failure.
  '''
  probed = _probe_spaces(WorldOfSupplyEnv, {'env': EnvConfig()})
  producer_space, consumer_space, observation_space = probed
  assert list(producer_space.nvec) == [8, 6]
  assert list(consumer_space.nvec) == [3, 3, 6]
  assert observation_space.shape == (56,)


def test_training_build_policy_and_module_specs():
  '''Verify policy and module specs map roles to model configs.

  Returns:
    None: Test completes without assertion failure.
  '''
  env = WorldOfSupplyEnv(EnvConfig())
  producer_space = env.action_space[env.agent_ids[0]]
  consumer_space = env.action_space[env.agent_ids[1]]
  observation = env.observation_space[env.agent_ids[0]]
  policy_specs = _build_policy_specs(
      _SpecRecorder, producer_space, consumer_space, observation,
  )
  expected_policies = {'ppo_producer', 'ppo_consumer'}
  expected_policies |= {'frozen_producer', 'frozen_consumer'}
  assert set(policy_specs) == expected_policies
  assert policy_specs['ppo_producer'].kwargs['action_space'] is producer_space
  assert policy_specs['ppo_consumer'].kwargs['action_space'] is consumer_space
  module_specs = _build_module_specs(
      _SpecRecorder, object, producer_space, consumer_space,
      observation, False,
  )
  producer_config = module_specs['ppo_producer'].kwargs['model_config']
  consumer_config = module_specs['ppo_consumer'].kwargs['model_config']
  assert producer_config['hidden_size'] == 128
  assert consumer_config['hidden_size'] == 256
  assert producer_config['use_lstm'] is False


def test_training_configure_and_apply_to_envs():
  '''Verify configure sets attributes and apply visits each env.

  Returns:
    None: Test completes without assertion failure.
  '''

  class FakeConfig:
    '''Minimal config stand-in exposing a settable learning rate.'''

    lr = 0.0

  configured = _configure(FakeConfig(), lr=0.1)
  assert configured.lr == 0.1
  visited = []

  class FakeVectorEnv:
    '''Minimal vector-env stand-in exposing child environments.'''

    def __init__(self, envs):
      '''Store child environments for traversal.

      Args:
        envs: Child environments, possibly containing None.

      Returns:
        None: Stand-in is mutated in place.
      '''
      self.envs = envs

  class FakeEnv:
    '''Minimal env stand-in exposing an unwrapped self reference.'''

    unwrapped = None

  fake = FakeEnv()
  fake.unwrapped = fake
  vector = FakeVectorEnv([fake, None])
  _apply_to_envs(vector, visited.append)
  assert visited == [fake]
  _apply_to_envs(None, visited.append)
  assert visited == [fake]
  assert isinstance(ObservationEncoder, type)
