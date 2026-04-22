import numpy as np

from truck_carton.config import AppConfig
from truck_carton.env.packing_env import (
  TruckCartonPackingEnv,
)
from truck_carton.evaluation.heuristic import (
  HeuristicAgent,
  RandomAgent,
)
from truck_carton.evaluation.metrics import (
  MetricsCollector,
)


def _run_episode(env, agent, seed=42):
  """Run one full episode, return total reward and
  step count."""
  obs, info = env.reset(seed=seed)
  total_reward = 0.0
  steps = 0
  done = False
  while not done:
    masks = env.action_masks()
    if not masks.any():
      break
    action = agent.predict(env, masks)
    obs, reward, terminated, truncated, info = (
      env.step(action)
    )
    total_reward += reward
    steps += 1
    done = terminated or truncated
  return total_reward, steps


def test_heuristic_agent_runs_full_episode():
  """HeuristicAgent completes a full episode
  without errors."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  agent = HeuristicAgent(config)
  reward, steps = _run_episode(env, agent)
  assert steps > 0
  assert isinstance(reward, float)


def test_random_agent_runs_full_episode():
  """RandomAgent completes a full episode."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  agent = RandomAgent(seed=42)
  reward, steps = _run_episode(env, agent)
  assert steps > 0


def test_heuristic_returns_valid_action():
  """Every action returned by HeuristicAgent must
  be in the valid mask."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  agent = HeuristicAgent(config)
  obs, _ = env.reset(seed=42)

  for _ in range(20):
    masks = env.action_masks()
    if not masks.any():
      break
    action = agent.predict(env, masks)
    assert masks[action], (
      f'Action {action} not in valid mask'
    )
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
      break


def test_heuristic_beats_random():
  """Heuristic should achieve higher reward than
  random across multiple episodes."""
  config = AppConfig()
  n_episodes = 10

  heuristic_rewards = []
  random_rewards = []

  for seed in range(n_episodes):
    env_h = TruckCartonPackingEnv(
      config=config, curriculum_stage=0
    )
    agent_h = HeuristicAgent(config)
    r_h, _ = _run_episode(env_h, agent_h, seed=seed)
    heuristic_rewards.append(r_h)

    env_r = TruckCartonPackingEnv(
      config=config, curriculum_stage=0
    )
    agent_r = RandomAgent(seed=seed)
    r_r, _ = _run_episode(env_r, agent_r, seed=seed)
    random_rewards.append(r_r)

  mean_h = np.mean(heuristic_rewards)
  mean_r = np.mean(random_rewards)
  assert mean_h >= mean_r, (
    f'Heuristic ({mean_h:.2f}) should beat '
    f'random ({mean_r:.2f})'
  )


def test_heuristic_places_cartons():
  """Heuristic should place at least some cartons."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  agent = HeuristicAgent(config)
  _run_episode(env, agent, seed=42)
  assert len(env.placed_cartons) > 0


def test_heuristic_delivers_cartons():
  """Heuristic should deliver at least some cartons
  to stores."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  agent = HeuristicAgent(config)
  _run_episode(env, agent, seed=42)
  assert env.num_delivered > 0


def test_heuristic_respects_weight_limits():
  """Heuristic should not exceed truck weight limits
  (validated by metrics)."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  agent = HeuristicAgent(config)
  reward, _ = _run_episode(env, agent, seed=42)

  collector = MetricsCollector()
  metrics = collector.compute(
    episode_data=env.episode_data,
    spaces=env.spaces,
    placed_cartons=env.placed_cartons,
    current_weights=env.current_weights,
    total_reward=reward,
    curriculum_stage=0,
  )
  assert metrics.weight_violation_rate == 0.0


def test_heuristic_no_support_violations():
  """Heuristic uses gravity-aware placement, so
  support violations should be zero."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  agent = HeuristicAgent(config)
  reward, _ = _run_episode(env, agent, seed=42)

  collector = MetricsCollector()
  metrics = collector.compute(
    episode_data=env.episode_data,
    spaces=env.spaces,
    placed_cartons=env.placed_cartons,
    current_weights=env.current_weights,
    total_reward=reward,
    curriculum_stage=0,
  )
  assert metrics.support_violation_rate == 0.0


def test_heuristic_across_stages():
  """Heuristic runs on multiple curriculum stages
  without errors."""
  config = AppConfig()
  for stage in [0, 1, 2]:
    env = TruckCartonPackingEnv(
      config=config, curriculum_stage=stage
    )
    agent = HeuristicAgent(config)
    reward, steps = _run_episode(
      env, agent, seed=42
    )
    assert steps > 0, (
      f'Stage {stage}: no steps executed'
    )


def test_heuristic_metrics_collection():
  """Full metrics pipeline works with heuristic."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  agent = HeuristicAgent(config)
  reward, _ = _run_episode(env, agent, seed=42)

  collector = MetricsCollector()
  metrics = collector.compute(
    episode_data=env.episode_data,
    spaces=env.spaces,
    placed_cartons=env.placed_cartons,
    current_weights=env.current_weights,
    total_reward=reward,
    curriculum_stage=0,
  )
  assert 0.0 <= metrics.completion_rate <= 1.0
  assert 0.0 <= metrics.fleet_volumetric_utilization <= 1.0
  assert 0.0 <= metrics.grouping_compliance_rate <= 1.0
  assert 0.0 <= metrics.fragility_violation_rate <= 1.0
  assert 0.0 <= metrics.support_violation_rate <= 1.0
  assert 0.0 <= metrics.weight_violation_rate <= 1.0
  assert 0.0 <= metrics.priority_accessibility_score <= 1.0


def test_random_agent_deterministic():
  """RandomAgent with same seed gives same actions."""
  config = AppConfig()
  env1 = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  env2 = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  agent1 = RandomAgent(seed=99)
  agent2 = RandomAgent(seed=99)

  r1, _ = _run_episode(env1, agent1, seed=42)
  r2, _ = _run_episode(env2, agent2, seed=42)
  assert abs(r1 - r2) < 1e-6


def test_heuristic_predict_with_no_valid_actions():
  """Heuristic returns 0 when no actions are valid."""
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=0
  )
  env.reset(seed=42)
  agent = HeuristicAgent(config)

  masks = np.zeros(600, dtype=np.bool_)
  action = agent.predict(env, masks)
  assert action == 0
