#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for agent callbacks, masking and masked agents.'''

# pylint: disable=protected-access,import-outside-toplevel
# pylint: disable=wrong-import-position,unused-import,unused-argument
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import os
import sys
import tempfile
import types
from unittest import mock

import numpy as np
import pytest


def _install_sb3_stubs():
  '''Install minimal sb3 stubs if the real packages are missing.'''
  try:
    import stable_baselines3  # noqa: F401
    import sb3_contrib  # noqa: F401
    return
  except ImportError:
    pass

  class _FakeBaseCallback:
    '''Minimal stand-in for BaseCallback.'''

    def __init__(self, verbose=0):
      '''Store verbose and initialise callback state.'''
      self.verbose = verbose
      self.locals = {}
      self.logger = mock.MagicMock()
      self.n_calls = 0
      self.model = mock.MagicMock()

  class _FakeMaskablePPO:
    '''Minimal stand-in recording constructor arguments.'''

    instances = []

    def __init__(self, *args, **kwargs):
      '''Record arguments and create a fake model.'''
      self.args = args
      self.kwargs = kwargs
      self.saved = []
      _FakeMaskablePPO.instances.append(self)

    def learn(self, total_timesteps=None, callback=None):
      '''Record a learn call.'''
      self.learned = (total_timesteps, callback)

    def predict(self, obs, deterministic=True, action_masks=None):
      '''Return a canned action.'''
      return np.array([1]), {}

    def save(self, path):
      '''Record a save call.'''
      self.saved.append(path)

    @classmethod
    def load(cls, path, env=None):
      '''Return a fake loaded model.'''
      instance = cls.__new__(cls)
      instance.args = ()
      instance.kwargs = {'loaded_path': path}
      instance.saved = []
      return instance

  class _FakeActionMasker:
    '''Minimal stand-in for ActionMasker.'''

    def __init__(self, env, mask_fn):
      '''Store the wrapped env and mask function.'''
      self.env = env
      self.mask_fn = mask_fn

    def action_masks(self):
      '''Delegate to the stored mask function.'''
      return self.mask_fn(self.env)

  sb3 = types.ModuleType('stable_baselines3')
  sb3_common = types.ModuleType('stable_baselines3.common')
  sb3_callbacks = types.ModuleType('stable_baselines3.common.callbacks')
  sb3_callbacks.BaseCallback = _FakeBaseCallback
  sb3.common = sb3_common
  sb3_common.callbacks = sb3_callbacks

  contrib = types.ModuleType('sb3_contrib')
  contrib.MaskablePPO = _FakeMaskablePPO
  contrib_wrappers = types.ModuleType(
    'sb3_contrib.common.wrappers'
  )
  contrib_wrappers.ActionMasker = _FakeActionMasker
  contrib_common = types.ModuleType('sb3_contrib.common')
  contrib_common.wrappers = contrib_wrappers
  contrib.common = contrib_common

  sys.modules.setdefault('stable_baselines3', sb3)
  sys.modules.setdefault('stable_baselines3.common', sb3_common)
  sys.modules.setdefault(
    'stable_baselines3.common.callbacks', sb3_callbacks
  )
  sys.modules.setdefault('sb3_contrib', contrib)
  sys.modules.setdefault('sb3_contrib.common', contrib_common)
  sys.modules.setdefault(
    'sb3_contrib.common.wrappers', contrib_wrappers
  )


_install_sb3_stubs()

from src.agent import action_mask_fn  # noqa: E402
from src.agent.action_mask_fn import compute_action_mask  # noqa: E402
from src.agent.callbacks import (  # noqa: E402
  CheckpointCallback,
  TensorBoardCallback,
)
from src.agent.masked_agent import (  # noqa: E402
  MaskedAgent,
  TierAgent,
  _mask_fn,
  _tier_mask_fn,
)


def _fake_logger():
  '''Return a logger mock collecting record calls.'''
  logger = mock.MagicMock()
  logger.records = []

  def _record(key, value):
    '''Append a record to the list.'''
    logger.records.append((key, value))

  logger.record.side_effect = _record
  return logger


class TestComputeActionMask:
  '''Tests for the action-mask helpers.'''

  def test_compute_action_mask_delegates_to_env(self, env):
    '''compute_action_mask returns the env mask unchanged.'''
    env.reset()
    expected = env.action_masks()
    result = compute_action_mask(env)
    assert np.array_equal(result, expected)

  def test_mask_fn_delegates_to_env(self, env):
    '''The private _mask_fn also delegates to the env.'''
    env.reset()
    assert np.array_equal(_mask_fn(env), env.action_masks())

  def test_tier_mask_fn_delegates(self, env_config):
    '''_tier_mask_fn delegates to a tier env.'''
    from src.cat_game_env.multi_agent import (  # noqa: PLC0415
      MultiAgentOrchestrator,
    )

    orch = MultiAgentOrchestrator(env_config)
    orch.reset()
    tier_env = orch.tier_envs[orch.tier_numbers[0]]
    assert np.array_equal(
      _tier_mask_fn(tier_env), tier_env.action_masks()
    )

  def test_action_mask_fn_module_alias(self):
    '''The module-level function is the same callable.'''
    assert action_mask_fn.compute_action_mask is compute_action_mask


class TestTensorBoardCallback:
  '''Tests for TensorBoardCallback.'''

  def _callback(self):
    '''Create a callback with a fake logger attached.'''
    callback = TensorBoardCallback()
    callback.logger = _fake_logger()
    return callback

  def test_empty_infos_returns_true(self):
    '''With no infos nothing is logged and training continues.'''
    callback = self._callback()
    callback.locals = {'infos': []}
    assert callback._on_step() is True
    assert callback.logger.records == []

  def test_missing_infos_returns_true(self):
    '''With no infos key training continues.'''
    callback = self._callback()
    callback.locals = {}
    assert callback._on_step() is True

  def test_logs_reward_and_action_aggregates(self):
    '''Reward components and action counts are averaged.'''
    callback = self._callback()
    callback.locals = {
      'infos': [
        {
          'reward_components': {'a': 2.0, 'b': 1},
          'action_info': {
            'applied': {'x': 1},
            'rejected': {},
          },
          'tick': 7,
        },
        {
          'reward_components': {'a': 4.0},
          'action_info': {
            'applied': {},
            'rejected': {'y': 2, 'z': 1},
          },
          'tick': 3,
        },
      ]
    }
    assert callback._on_step() is True
    records = dict(callback.logger.records)
    assert records['reward/a'] == 3.0
    assert records['reward/b'] == 1.0
    assert records['actions/applied_count'] == 0.5
    assert records['actions/rejected_count'] == 1.0
    assert records['env/tick'] == 7

  def test_non_numeric_reward_components_ignored(self):
    '''Non-numeric reward components are skipped.'''
    callback = self._callback()
    callback.locals = {
      'infos': [
        {
          'reward_components': {'note': 'hi', 'a': 5},
          'action_info': {},
          'tick': 1,
        }
      ]
    }
    assert callback._on_step() is True
    keys = [key for key, _ in callback.logger.records]
    assert 'reward/note' not in keys
    assert 'reward/a' in keys

  def test_info_without_optional_keys(self):
    '''Infos missing action_info still log zero counts.'''
    callback = self._callback()
    callback.locals = {'infos': [{'tick': 2}]}
    assert callback._on_step() is True
    records = dict(callback.logger.records)
    assert records['actions/applied_count'] == 0.0
    assert records['env/tick'] == 2


class TestCheckpointCallback:
  '''Tests for CheckpointCallback.'''

  def test_creates_save_directory(self):
    '''Construction creates the checkpoint directory.'''
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'ckpts')
      callback = CheckpointCallback(
        save_freq=10, save_path=path, verbose=0
      )
      assert os.path.isdir(path)
      assert callback._save_freq == 10

  def test_saves_on_frequency(self):
    '''A save happens exactly on the configured frequency.'''
    with tempfile.TemporaryDirectory() as tmp:
      callback = CheckpointCallback(
        save_freq=5, save_path=tmp, verbose=0
      )
      callback.n_calls = 5
      callback.model = mock.MagicMock()
      assert callback._on_step() is True
      callback.model.save.assert_called_once_with(
        os.path.join(tmp, 'checkpoint_5')
      )

  def test_no_save_off_frequency(self):
    '''No save happens off the configured frequency.'''
    with tempfile.TemporaryDirectory() as tmp:
      callback = CheckpointCallback(
        save_freq=5, save_path=tmp, verbose=1
      )
      callback.n_calls = 4
      callback.model = mock.MagicMock()
      assert callback._on_step() is True
      callback.model.save.assert_not_called()

  def test_verbose_prints_on_save(self):
    '''Verbose mode prints the checkpoint path.'''
    with tempfile.TemporaryDirectory() as tmp:
      callback = CheckpointCallback(
        save_freq=1, save_path=tmp, verbose=1
      )
      callback.n_calls = 1
      callback.model = mock.MagicMock()
      with mock.patch('builtins.print') as mock_print:
        assert callback._on_step() is True
      mock_print.assert_called_once()


def _agent_config():
  '''Return a minimal valid agent config.'''
  return {
    'policy': 'MultiInputPolicy',
    'hyperparameters': {'learning_rate': 0.001},
    'training': {
      'log_dir': 'output/logs',
      'checkpoint_freq': 123,
      'model_dir': 'output/models',
    },
  }


class TestMaskedAgent:
  '''Tests for MaskedAgent with stubbed sb3.'''

  def test_init_wraps_env_and_builds_model(self, env):
    '''Construction wraps the env and stores config.'''
    agent = MaskedAgent(env, _agent_config())
    assert agent.model is not None
    assert agent._config['policy'] == 'MultiInputPolicy'

  def test_predict_returns_action(self, env):
    '''predict returns the model's action.'''
    agent = MaskedAgent(env, _agent_config())
    action = agent.predict({'obs': np.zeros(3)})
    assert np.array_equal(action, np.array([1]))

  def test_train_uses_default_callbacks(self, env):
    '''train without callbacks builds the default list.'''
    agent = MaskedAgent(env, _agent_config())
    agent._model = mock.MagicMock()
    agent.train(total_timesteps=10)
    _, kwargs = agent.model.learn.call_args
    callbacks = kwargs['callback']
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], TensorBoardCallback)
    assert isinstance(callbacks[1], CheckpointCallback)
    assert callbacks[1]._save_freq == 123

  def test_train_with_explicit_callbacks(self, env):
    '''train forwards explicit callbacks unchanged.'''
    agent = MaskedAgent(env, _agent_config())
    agent._model = mock.MagicMock()
    sentinel = [object()]
    agent.train(total_timesteps=5, callbacks=sentinel)
    _, kwargs = agent.model.learn.call_args
    assert kwargs['callback'] is sentinel

  def test_save_creates_parent_directory(self, env):
    '''save creates parent directories and saves the model.'''
    agent = MaskedAgent(env, _agent_config())
    agent._model = mock.MagicMock()
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'sub', 'model')
      agent.save(path)
      assert os.path.isdir(os.path.join(tmp, 'sub'))
      agent.model.save.assert_called_once_with(path)

  def test_save_without_directory_component(self, env):
    '''save works for a bare filename.'''
    agent = MaskedAgent(env, _agent_config())
    agent._model = mock.MagicMock()
    with tempfile.TemporaryDirectory() as tmp:
      cwd = os.getcwd()
      os.chdir(tmp)
      try:
        agent.save('model')
      finally:
        os.chdir(cwd)
      agent.model.save.assert_called_once_with('model')

  def test_load_returns_agent(self, env):
    '''load classmethod builds an agent from a path.'''
    with mock.patch(
      'src.agent.masked_agent.MaskablePPO'
    ) as mock_ppo:
      mock_ppo.load.return_value = mock.MagicMock()
      agent = MaskedAgent.load(
        'some/path', env, _agent_config()
      )
      assert agent.model is mock_ppo.load.return_value
      mock_ppo.load.assert_called_once()

  def test_mask_fn_used_by_wrapper(self, env):
    '''The wrapper mask function matches the env mask.'''
    env.reset()
    assert np.array_equal(_mask_fn(env), env.action_masks())


class TestTierAgent:
  '''Tests for TierAgent with stubbed sb3.'''

  def _tier_env(self, env_config):
    '''Build an orchestrator and return its first tier env.'''
    from src.cat_game_env.multi_agent import (  # noqa: PLC0415
      MultiAgentOrchestrator,
    )

    orch = MultiAgentOrchestrator(env_config)
    orch.reset()
    return orch.tier_envs[orch.tier_numbers[0]]

  def test_init_stores_tier(self, env_config):
    '''Construction stores the tier number.'''
    tier_env = self._tier_env(env_config)
    agent = TierAgent(tier_env, _agent_config())
    assert agent.tier == tier_env.tier
    assert agent.model is not None

  def test_predict_returns_action(self, env_config):
    '''predict returns the model's action.'''
    agent = TierAgent(self._tier_env(env_config), _agent_config())
    assert np.array_equal(
      agent.predict({'obs': np.zeros(2)}), np.array([1])
    )

  def test_save_and_load(self, env_config):
    '''save delegates to the model and load rebuilds.'''
    tier_env = self._tier_env(env_config)
    agent = TierAgent(tier_env, _agent_config())
    agent._model = mock.MagicMock()
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'tiermodel')
      agent.save(path)
      agent.model.save.assert_called_once_with(path)
    with mock.patch(
      'src.agent.masked_agent.MaskablePPO'
    ) as mock_ppo:
      mock_ppo.load.return_value = mock.MagicMock()
      loaded = TierAgent.load('p', tier_env, _agent_config())
      assert loaded.tier == tier_env.tier
      assert loaded.model is mock_ppo.load.return_value


if __name__ == '__main__':
  pytest.main([__file__])
