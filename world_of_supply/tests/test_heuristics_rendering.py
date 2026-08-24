#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for scripted RL-level heuristics and rendering components.'''

import pytest

gymnasium = pytest.importorskip('gymnasium')

from world_of_supply.rl.env import EnvConfig, WorldOfSupplyEnv


@pytest.fixture(scope='module')
def env():
  return WorldOfSupplyEnv(EnvConfig(episode_duration=200))


def test_scripted_actions_respect_action_space_bounds(env):
  env.reset(seed=0)
  actions = env.scripted_actions(seed=0)
  assert set(actions) == set(env.agent_ids)
  for agent_id, action in actions.items():
    nvec = env.action_space.spaces[agent_id].nvec
    for dimension, categories in enumerate(nvec):
      assert 0 <= action[dimension] < categories


def test_scripted_episode_runs_to_horizon(env):
  env.reset(seed=3)
  done = False
  steps = 0
  while not done:
    _, rewards, terminated, truncated, _ = env.step(env.scripted_actions(seed=5))
    done = truncated.get('__all__', False) or terminated.get('__all__', False)
    steps += 1
    assert steps <= 25
  expected_steps = env.config.episode_duration // env.config.downsampling_rate
  assert steps == expected_steps


def test_status_formatter_emits_all_sections(env):
  from world_of_supply.rendering.status import WorldStatusFormatter

  env.reset(seed=0)
  formatter = WorldStatusFormatter()
  status = formatter.status(env.world)
  text = str(status)
  for facility_id in list(env.world.facilities)[:3]:
    assert facility_id in text


def test_railroad_sprite_picks_crossing_glyph():
  from world_of_supply.rendering.sprites import railroad_glyph

  grid_neighbors = {
      (x, y): (x in (0, 2)) or (y in (0, 2))
      for x in range(-1, 4)
      for y in range(-1, 4)
  }
  glyph = railroad_glyph(1, 1, lambda x, y: grid_neighbors.get((x, y), False))
  assert glyph == '╬'


def test_ascii_renderer_produces_image(env):
  pillow = pytest.importorskip('PIL')
  from world_of_supply.rendering.renderer import AsciiWorldRenderer

  env.reset(seed=0)
  image = AsciiWorldRenderer().render(env.world)
  assert isinstance(image, pillow.Image.Image)
  assert image.size[0] > 0 and image.size[1] > 0
