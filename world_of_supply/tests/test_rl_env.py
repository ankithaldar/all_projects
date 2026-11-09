#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the multi-agent RL environment (requires gymnasium).'''

import pytest

gymnasium = pytest.importorskip('gymnasium')

from world_of_supply.rl.env import EnvConfig, WorldOfSupplyEnv
from world_of_supply.rl.env import coerce_env_config
from world_of_supply.scenario import ScenarioConfig


@pytest.fixture(scope='module')
def env():
  '''Build the shared module-scoped RL environment.

  Returns:
    WorldOfSupplyEnv: Environment with duration 200 and rate 10.
  '''
  config = EnvConfig(
      scenario=ScenarioConfig(),
      episode_duration=200,
      downsampling_rate=10,
  )
  return WorldOfSupplyEnv(config)


def test_observation_space_covers_all_agents(env):
  '''Verify observation and action spaces cover every agent.

  Args:
    env: Shared RL environment fixture.

  Returns:
    None: Test completes without assertion failure.
  '''
  assert len(env.agent_ids) == 18
  for agent_id in env.agent_ids:
    assert agent_id in env.observation_space.spaces
    assert agent_id in env.action_space.spaces


def test_reset_returns_normalized_observations_for_every_agent(env):
  '''Verify reset returns normalized observations for every agent.

  Args:
    env: Shared RL environment fixture.

  Returns:
    None: Test completes without assertion failure.
  '''
  observations, info = env.reset(seed=0)
  assert set(observations) == set(env.agent_ids)
  assert isinstance(info, dict)
  for vector in observations.values():
    assert vector.shape == (env.encoder.total_dim,)
    assert float(vector.min()) >= 0.0 and float(vector.max()) <= 1.0


def test_step_downsamples_time_and_returns_full_payloads(env):
  '''Verify step downsamples time and returns full payloads.

  Args:
    env: Shared RL environment fixture.

  Returns:
    None: Test completes without assertion failure.
  '''
  env.reset(seed=1)
  spaces = env.action_space.spaces.items()
  actions = {agent_id: space.sample() for agent_id, space in spaces}
  time_before = env.world.time_step
  results = env.step(actions)
  observations, rewards, terminated, truncated, infos = results
  rate = env.config.downsampling_rate
  assert env.world.time_step == time_before + rate
  assert set(rewards) == set(env.agent_ids)
  assert not any(terminated.values())
  assert set(infos) == set(env.agent_ids)


def test_episode_truncates_at_horizon():
  '''Verify episode truncates once the horizon is reached.

  Returns:
    None: Test completes without assertion failure.
  '''
  small_env = WorldOfSupplyEnv(
      EnvConfig(episode_duration=20, downsampling_rate=10),
  )
  small_env.reset(seed=2)
  for _ in range(2):
    spaces = small_env.action_space.spaces.items()
    actions = {agent_id: space.sample() for agent_id, space in spaces}
    _, _, _, truncated, _ = small_env.step(actions)
  assert truncated['__all__'] is True


def test_env_config_coercion_shapes():
  '''Verify env config coercion accepts every supported shape.

  Returns:
    None: Test completes without assertion failure.
  '''
  assert coerce_env_config(None).episode_duration == 1000
  assert coerce_env_config(EnvConfig(episode_duration=5)).episode_duration == 5
  wrapped = {'env': EnvConfig(episode_duration=7)}
  assert coerce_env_config(wrapped).episode_duration == 7
  assert coerce_env_config({'episode_duration': 9}).episode_duration == 9
  config = {'env': EnvConfig(episode_duration=100, downsampling_rate=10)}
  assert WorldOfSupplyEnv(config).config.episode_duration == 100


def test_reward_shaper_shapes_curriculum():
  '''Verify reward shaper shapes retailer and non-retailer sheets.

  Returns:
    None: Test completes without assertion failure.
  '''
  from world_of_supply.economy import BalanceSheet
  from world_of_supply.rl.rewards import RetailerProfitRewardShaper

  shaper = RetailerProfitRewardShaper()
  sheets = {
      'RetailerCell_8': BalanceSheet(profit=100),
      'SteelFactoryCell_1': BalanceSheet(loss=-40),
  }
  early = shaper.shape(sheets, iteration=0, n_iterations=10)
  late = shaper.shape(sheets, iteration=10, n_iterations=10)
  expected = {'RetailerCell_8p', 'RetailerCell_8c'}
  expected |= {'SteelFactoryCell_1p', 'SteelFactoryCell_1c'}
  assert set(early) == expected
  assert late['RetailerCell_8p'] != early['RetailerCell_8p']
