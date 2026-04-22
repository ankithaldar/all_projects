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
    assert mask.shape == (
        config.env.max_candidates,
    )

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
    assert info['num_placed'] > 0


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
            f"Observation '{key}' not in space"
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
    """Episode info 'r' must be cumulative reward,
    not the final single-step reward."""
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

    assert 'episode' in last_info
    ep_reward = last_info['episode']['r']
    assert abs(ep_reward - cumulative) < 1e-6


def test_env_no_carton_skip_on_failed_placement():
    """If the action decodes to None (invalid), the
    current carton should not advance."""
    config = AppConfig()
    env = TruckCartonPackingEnv(
        config=config, curriculum_stage=0
    )
    env.reset(seed=42)

    initial_carton = env._current_carton
    queue_len = len(env._queue)

    # Step with an action index beyond candidates
    # (should decode to None).
    bad_action = config.env.max_candidates - 1
    env.step(bad_action)

    # Carton should NOT have advanced if placement
    # failed.
    if initial_carton is not None:
        assert env._current_carton == initial_carton
        assert len(env._queue) == queue_len
