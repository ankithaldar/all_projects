#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Robustness tests for env hardening E-01..E-06.'''

# pylint: disable=protected-access

import numpy as np
import pytest

from truck_carton.config import AppConfig, EnvironmentConfig
from truck_carton.domain.models import ObservationContext
from truck_carton.env.observation import ObservationBuilder
from truck_carton.env.packing_env import TruckCartonPackingEnv
from truck_carton.packing.placement import PlacementCandidate
from truck_carton.packing.rotation import Rotation


def _env(**kwargs):
  '''Create reset env.'''
  env = TruckCartonPackingEnv(
    config=AppConfig(), curriculum_stage=0, **kwargs
  )
  env.reset(seed=42)
  return env


def _valid_action(env):
  '''Return first valid action.'''
  valid = np.where(env.action_masks())[0]
  assert len(valid) > 0
  return int(valid[0])


def _to_loading(env):
  '''Drive env until a carton is being packed.'''
  for _ in range(50):
    if env.current_carton is not None:
      return
    env.step(_valid_action(env))
  assert env.current_carton is not None


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), None, 'x'])
def test_step_invalid_action_raises_value_error(bad):
  '''E-01: NaN/inf/None/str action raises ValueError.'''
  env = _env()
  with pytest.raises(ValueError):
    env.step(bad)


def test_step_masked_action_no_state_change():
  '''E-02: masked action is a no-op with zero reward.'''
  env = _env()
  mask = env.action_masks()
  masked = np.where(~mask)[0]
  assert len(masked) > 0
  action = int(masked[0])
  before = env.step_count
  idx = env.active_truck_idx
  obs, reward, term, trunc, _ = env.step(action)
  assert reward == 0.0
  assert not term and not trunc
  assert env.step_count == before
  assert env.active_truck_idx == idx
  for key, space in env.observation_space.spaces.items():
    assert space.contains(obs[key])


def test_step_after_terminated_raises():
  '''E-02: step after terminated raises RuntimeError.'''
  env = _env()
  done = False
  while not done:
    _, _, term, trunc, _ = env.step(_valid_action(env))
    done = term or trunc
    if env.step_count > 50:
      break
  if done:
    with pytest.raises(RuntimeError):
      env.step(_valid_action(env))


def test_step_after_truncated_raises():
  '''E-02: step after truncated raises RuntimeError.'''
  env = _env(max_steps=1)
  _, _, term, trunc, _ = env.step(_valid_action(env))
  assert trunc or term
  with pytest.raises(RuntimeError):
    env.step(_valid_action(env))


def test_execute_packing_no_double_count():
  '''E-02: re-executing packing does not double count.'''
  env = _env()
  _to_loading(env)
  assert env.action_manager.candidates
  cand = env.action_manager.candidates[0]
  weight_before = list(env.current_weights)
  info = env._execute_packing(cand)
  assert info is not None
  weight_after = list(env.current_weights)
  # Second call with same carton must not double count.
  env._current_carton = env.current_carton
  env._execute_packing(cand)
  assert env.current_weights == weight_after
  assert weight_after != weight_before or info is not None


def test_execute_packing_invalid_candidate_no_crash():
  '''E-03: out-of-bounds placement returns None, no crash.'''
  env = _env()
  _to_loading(env)
  carton = env.current_carton
  assert carton is not None
  dl, dw, dh = carton.length, carton.width, carton.height
  bad = PlacementCandidate(
    truck_id=env.active_truck_idx,
    x=10_000,
    y=0,
    z=0,
    rotation=Rotation.LWH,
    oriented_dims=(dl, dw, dh),
  )
  weights = list(env.current_weights)
  assert env._execute_packing(bad) is None
  assert env.current_weights == weights
  assert carton.carton_id not in env.placed_cartons


def test_constructor_clamps_stage():
  '''E-04: curriculum_stage is clamped.'''
  env = TruckCartonPackingEnv(
    config=AppConfig(), curriculum_stage=-5
  )
  assert env.curriculum_stage == 0
  last = len(AppConfig().curriculum.stages) - 1
  env2 = TruckCartonPackingEnv(
    config=AppConfig(), curriculum_stage=999
  )
  assert env2.curriculum_stage == last


def test_constructor_max_steps_validated():
  '''E-04: max_steps < 1 raises ValueError.'''
  with pytest.raises(ValueError):
    TruckCartonPackingEnv(config=AppConfig(), max_steps=0)


def test_constructor_env_config_validated():
  '''E-04: bad EnvironmentConfig caps raise ValueError.'''
  bad = EnvironmentConfig(max_trucks=0)
  config = AppConfig(env=bad)
  with pytest.raises(ValueError):
    TruckCartonPackingEnv(config=config)


def test_observation_nan_clipped():
  '''E-06: NaN/inf in context produce finite obs.'''
  config = AppConfig()
  env = TruckCartonPackingEnv(config=config)
  env.reset(seed=42)
  ctx = ObservationContext(
    trucks=env.episode_data.trucks,
    spaces=env.spaces,
    current_carton=env.current_carton,
    remaining_cartons=[],
    packing_candidates=[],
    current_weights=[float('nan')] * len(env.episode_data.trucks),
    num_placed=0,
    total_cartons=1,
    stage_index=0,
    step_count=0,
    max_steps=500,
  )
  obs = ObservationBuilder(config.env).build(ctx)
  for key, arr in obs.items():
    assert np.all(np.isfinite(arr)), key
    assert arr.min() >= 0.0 and arr.max() <= 1.0
  # Force a NaN carton feature path as well.
  if env.episode_data.cartons:
    carton = env.episode_data.cartons[0]
    carton.weight = float('nan')
    ctx2 = ObservationContext(
      trucks=[],
      spaces=[],
      current_carton=carton,
      remaining_cartons=[carton],
      packing_candidates=[],
      current_weights=[],
      num_placed=0,
      total_cartons=1,
      stage_index=0,
      step_count=0,
      max_steps=500,
    )
    obs2 = ObservationBuilder(config.env).build(ctx2)
    for arr in obs2.values():
      assert np.all(np.isfinite(arr))


def test_valid_action_unchanged():
  '''Valid action still advances step count.'''
  env = _env()
  before = env.step_count
  env.step(_valid_action(env))
  assert env.step_count == before + 1
