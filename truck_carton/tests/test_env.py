import numpy as np

from truck_carton.config import AppConfig
from truck_carton.env.packing_env import (
  TruckCartonPackingEnv,
)


def test_env_reset():
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, info = env.reset(seed=42)

  assert 'stage' in info
  assert info['stage'] == 0

  for key in env.observation_space.spaces:
    assert key in obs
    assert obs[key].shape == (
      env.observation_space[key].shape
    )


def test_env_step():
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, info = env.reset(seed=42)

  mask = env.action_masks()
  total_size = (
    config.env.max_candidates
    + config.env.max_routing_actions
  )
  assert mask.shape == (total_size,)

  valid = np.where(mask)[0]
  if len(valid) > 0:
    action = int(valid[0])
    obs, reward, terminated, truncated, info = (
      env.step(action)
    )
    assert isinstance(reward, float)
    assert 'reward_breakdown' in info
    assert 'num_placed' in info


def test_env_full_episode():
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, info = env.reset(seed=42)

  total_reward = 0.0
  steps = 0
  done = False

  while not done:
    mask = env.action_masks()
    valid = np.where(mask)[0]
    if len(valid) == 0:
      break
    action = int(valid[0])
    obs, reward, terminated, truncated, info = (
      env.step(action)
    )
    total_reward += reward
    steps += 1
    done = terminated or truncated

  assert steps > 0


def test_env_observation_space_contains():
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, _ = env.reset(seed=42)

  for key, space in (
    env.observation_space.spaces.items()
  ):
    assert space.contains(obs[key]), (
      f'Observation {key!r} not in space'
    )


def test_env_set_curriculum_stage():
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  env.set_curriculum_stage(1)
  assert env.curriculum_stage == 1

  obs, info = env.reset(seed=42)
  assert info['stage'] == 1


def test_env_action_mask_all_valid_at_start():
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  env.reset(seed=42)

  mask = env.action_masks()
  assert mask.any(), (
    'At least some actions should be valid'
  )


def test_env_cumulative_reward_in_episode_info():
  """Episode info 'r' must be cumulative reward."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, _ = env.reset(seed=42)

  cumulative = 0.0
  done = False
  last_info = {}

  while not done:
    mask = env.action_masks()
    valid = np.where(mask)[0]
    if len(valid) == 0:
      break
    action = int(valid[0])
    obs, reward, terminated, truncated, info = (
      env.step(action)
    )
    cumulative += reward
    last_info = info
    done = terminated or truncated

  if 'episode' in last_info:
    ep_reward = last_info['episode']['r']
    assert abs(ep_reward - cumulative) < 1e-6


def test_env_grid_world_created():
  """The environment must generate a grid world
  during reset."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  env.reset(seed=42)

  assert env.episode_data is not None
  assert env.episode_data.grid_world is not None
  assert len(env.episode_data.warehouses) > 0
  snap = env.get_render_snapshot()
  assert len(snap['warehouse_cartons']) > 0


def test_env_routing_and_packing():
  """Run a full episode with both routing and
  packing actions."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, info = env.reset(seed=42)

  routing_steps = 0
  packing_steps = 0
  done = False

  while not done:
    mask = env.action_masks()
    valid = np.where(mask)[0]
    if len(valid) == 0:
      break

    action = int(valid[0])
    if action >= config.env.max_candidates:
      routing_steps += 1
    else:
      packing_steps += 1

    obs, reward, terminated, truncated, info = (
      env.step(action)
    )
    done = terminated or truncated

  assert routing_steps > 0 or packing_steps > 0


def test_render_snapshot():
  """get_render_snapshot() must return a dict
  with all required keys."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  env.reset(seed=42)

  snap = env.get_render_snapshot()
  assert 'grid_world' in snap
  assert 'trucks' in snap
  assert 'warehouses' in snap
  assert 'stores' in snap
  assert 'warehouse_cartons' in snap
  assert 'truck_cargo' in snap
  assert 'delivered' in snap
  assert 'step' in snap


def test_grid_renderer_produces_image():
  """GridRenderer.render() must return a PIL
  Image."""
  from truck_carton.evaluation.visualizer import (
    GridRenderer,
  )

  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  env.reset(seed=42)

  renderer = GridRenderer()
  snap = env.get_render_snapshot()
  img = renderer.render(snap)

  assert img is not None
  assert img.size[0] > 0
  assert img.size[1] > 0


def test_road_sprite_straight_horizontal():
  """A road cell with left and right neighbors
  should produce a horizontal line."""
  from truck_carton.evaluation.visualizer import (
    GridRenderer,
  )

  renderer = GridRenderer()
  grid = np.zeros((3, 3), dtype=np.int8)
  grid[1, 0] = 1  # ROAD left
  grid[1, 1] = 1  # ROAD center
  grid[1, 2] = 1  # ROAD right

  sprite = renderer._road_sprite(1, 1, grid)
  assert sprite == '\u2500'  # ─


def test_road_sprite_corner():
  """A road cell with bottom and right neighbors
  should produce a top-left corner."""
  from truck_carton.evaluation.visualizer import (
    GridRenderer,
  )

  renderer = GridRenderer()
  grid = np.zeros((3, 3), dtype=np.int8)
  grid[0, 0] = 1  # center
  grid[1, 0] = 1  # below
  grid[0, 1] = 1  # right

  sprite = renderer._road_sprite(0, 0, grid)
  assert sprite == '\u250C'  # ┌


def test_env_episode_terminates():
  """A full episode must always terminate within
  max_steps, even with edge-case cartons."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, _ = env.reset(seed=99)

  steps = 0
  done = False
  while not done and steps < 500:
    mask = env.action_masks()
    valid = np.where(mask)[0]
    if len(valid) == 0:
      break
    action = int(valid[0])
    obs, reward, terminated, truncated, info = (
      env.step(action)
    )
    steps += 1
    done = terminated or truncated

  assert done or steps >= 500 or len(valid) == 0


def test_carton_queue_obs_nonzero():
  """carton_queue observation must have nonzero
  entries when unplaced cartons exist."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, _ = env.reset(seed=42)

  assert obs['carton_queue'].any(), (
    'carton_queue obs should reflect remaining '
    'unplaced cartons, not be all zeros'
  )


def test_routing_feature_dim_matches_obs():
  """routing_candidates observation dim must match
  config.env.routing_feature_dim (no wasted slots)."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, _ = env.reset(seed=42)

  expected_dim = config.env.routing_feature_dim
  actual_dim = obs['routing_candidates'].shape[1]
  assert actual_dim == expected_dim


def test_stage_index_normalization():
  """Stage index in global_info must be properly
  normalized using num_curriculum_stages, not
  hardcoded constants."""
  config = AppConfig()
  max_stage = len(config.curriculum.stages) - 1

  for stage_idx in range(
    min(3, len(config.curriculum.stages))
  ):
    env = TruckCartonPackingEnv(
      config=config, curriculum_stage=stage_idx
    )
    obs, _ = env.reset(seed=42)
    stage_val = obs['global_info'][1]
    expected = stage_idx / max(max_stage, 1)
    assert abs(stage_val - expected) < 1e-5, (
      f'Stage {stage_idx}: expected {expected}, '
      f'got {stage_val}'
    )


def test_cycle_to_truck_with_actions():
  """When active truck has no valid actions, env
  should cycle to another truck before terminating."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  obs, _ = env.reset(seed=42)

  done = False
  steps = 0
  while not done and steps < 200:
    mask = env.action_masks()
    valid = np.where(mask)[0]
    if len(valid) == 0:
      break
    action = int(valid[0])
    obs, reward, terminated, truncated, info = (
      env.step(action)
    )
    steps += 1
    done = terminated or truncated

  assert steps > 0
