#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for agent callbacks, masking and masked agents.'''

# pylint: disable=protected-access,import-outside-toplevel
# pylint: disable=wrong-import-position,unused-import,unused-argument
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
import types
from contextlib import contextmanager
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
)


class _RecordingMaskablePPO:
  '''Recording stand-in for MaskablePPO.

  Records every construction so tests can assert on the
  arguments ``BaseMaskedAgent`` passes, whichever
  ``MaskablePPO`` (stub or real) was imported.
  '''

  instances = []

  def __init__(self, *args, **kwargs):
    '''Record constructor arguments and predict state.'''
    self.args = args
    self.kwargs = kwargs
    self.predict_calls = []
    _RecordingMaskablePPO.instances.append(self)

  def predict(self, obs, deterministic=True, action_masks=None):
    '''Record the call and return a known action.

    Args:
      obs: Observation passed through from the agent.
      deterministic: Determinism flag from the agent.
      action_masks: Action masks passed through from the
        agent.

    Returns:
      A known action array and an empty info dict.
    '''
    self.predict_calls.append(
      {
        'obs': obs,
        'deterministic': deterministic,
        'action_masks': action_masks,
      }
    )
    return np.array([1]), {}


class _RecordingActionMasker:
  '''Recording stand-in for ActionMasker.'''

  instances = []

  def __init__(self, env, mask_fn):
    '''Record the wrapped env and mask function.'''
    self.env = env
    self.mask_fn = mask_fn
    _RecordingActionMasker.instances.append(self)


@contextmanager
def _patched_construction():
  '''Patch model and wrapper with the recording fakes.

  Yields:
    Nothing; construction inside the block uses the fakes.
  '''
  with mock.patch(
    'src.agent.masked_agent.MaskablePPO', _RecordingMaskablePPO
  ), mock.patch(
    'src.agent.masked_agent.ActionMasker', _RecordingActionMasker
  ):
    yield


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
    '''The single-source mask delegates to the env.'''
    from src.agent import masked_agent  # noqa: PLC0415

    assert not hasattr(masked_agent, '_mask_fn')
    assert not hasattr(masked_agent, '_tier_mask_fn')
    env.reset()
    assert np.array_equal(
      compute_action_mask(env), env.action_masks()
    )

  def test_tier_mask_fn_delegates(self, env_config):
    '''compute_action_mask delegates to a tier env.'''
    from src.cat_game_env.multi_agent import (  # noqa: PLC0415
      MultiAgentOrchestrator,
    )

    orch = MultiAgentOrchestrator(env_config)
    orch.reset()
    tier_env = orch.tier_envs[orch.tier_numbers[0]]
    assert np.array_equal(
      compute_action_mask(tier_env), tier_env.action_masks()
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
    '''Verbose mode logs the checkpoint path via logging.'''
    with tempfile.TemporaryDirectory() as tmp:
      callback = CheckpointCallback(
        save_freq=1, save_path=tmp, verbose=1
      )
      callback.n_calls = 1
      callback.model = mock.MagicMock()
      with mock.patch('src.agent.callbacks._LOGGER') as mock_log:
        with mock.patch('builtins.print') as mock_print:
          assert callback._on_step() is True
      mock_print.assert_not_called()
      mock_log.info.assert_called_once()
      assert 'checkpoint_1' in mock_log.info.call_args[0][1]

  def test_verbose_logs_on_save(self, caplog):
    '''Verbose mode logs the checkpoint path via logging.'''
    with tempfile.TemporaryDirectory() as tmp:
      callback = CheckpointCallback(
        save_freq=1, save_path=tmp, verbose=1
      )
      callback.n_calls = 1
      callback.model = mock.MagicMock()
      with caplog.at_level(logging.INFO, logger='src.agent.callbacks'):
        assert callback._on_step() is True
      assert 'checkpoint_1' in caplog.text


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
    '''predict forwards obs and returns the model's action.'''
    with _patched_construction():
      agent = MaskedAgent(env, _agent_config())
    obs, _ = env.reset()
    action = agent.predict(obs)
    assert np.array_equal(action, np.array([1]))
    assert agent.model is _RecordingMaskablePPO.instances[-1]
    assert len(agent.model.predict_calls) == 1
    call = agent.model.predict_calls[0]
    assert call['obs'] is obs
    assert call['deterministic'] is True
    assert call['action_masks'] is None

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
    assert np.array_equal(
      compute_action_mask(env), env.action_masks()
    )


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
    '''predict forwards obs and returns the model's action.'''
    tier_env = self._tier_env(env_config)
    with _patched_construction():
      agent = TierAgent(tier_env, _agent_config())
    obs = tier_env.build_obs(0)
    action = agent.predict(obs)
    assert np.array_equal(action, np.array([1]))
    assert agent.model is _RecordingMaskablePPO.instances[-1]
    assert len(agent.model.predict_calls) == 1
    call = agent.model.predict_calls[0]
    assert call['obs'] is obs
    assert call['deterministic'] is True
    assert call['action_masks'] is None

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


class TestSingleSourceMask:
  '''Tests that action masking has a single source of truth.'''

  def test_mask_fn_is_compute_action_mask(self):
    '''masked_agent defines no duplicate mask helpers.'''
    from src.agent import masked_agent  # noqa: PLC0415

    assert not hasattr(masked_agent, '_mask_fn')
    assert not hasattr(masked_agent, '_tier_mask_fn')

  def test_tier_mask_fn_is_compute_action_mask(self):
    '''compute_action_mask serves both env kinds.'''
    from src.agent import masked_agent  # noqa: PLC0415

    assert masked_agent.compute_action_mask is compute_action_mask

  def test_wrapper_uses_compute_action_mask(self, env):
    '''The ActionMasker wrapper receives compute_action_mask.'''
    model_before = len(_RecordingMaskablePPO.instances)
    wrapper_before = len(_RecordingActionMasker.instances)
    with _patched_construction():
      MaskedAgent(env, _agent_config())
    instance = _RecordingMaskablePPO.instances[model_before]
    assert instance.args[0] == 'MultiInputPolicy'
    wrapped = instance.args[1]
    assert wrapped is _RecordingActionMasker.instances[wrapper_before]
    assert wrapped.env is env
    assert wrapped.mask_fn is compute_action_mask


class TestSharedBase:
  '''Tests that TierAgent reuses the MaskedAgent implementation.'''

  def test_tier_agent_subclasses_masked_agent(self):
    '''TierAgent and MaskedAgent share a common base class.'''
    from src.agent.masked_agent import (  # noqa: PLC0415
      BaseMaskedAgent,
    )

    assert issubclass(TierAgent, BaseMaskedAgent)
    assert issubclass(MaskedAgent, BaseMaskedAgent)

  def test_tier_agent_inherits_shared_methods(self):
    '''TierAgent inherits predict, save and load unchanged.'''
    assert TierAgent.predict is MaskedAgent.predict
    assert TierAgent.save is MaskedAgent.save
    assert TierAgent.load.__func__ is MaskedAgent.load.__func__

  def test_masked_agent_default_n_steps(self, env):
    '''MaskedAgent keeps its n_steps default of 2048.'''
    before = len(_RecordingMaskablePPO.instances)
    with _patched_construction():
      agent = MaskedAgent(env, _agent_config())
    instance = _RecordingMaskablePPO.instances[before]
    assert agent.model is instance
    assert instance.kwargs['n_steps'] == 2048
    assert instance.kwargs['verbose'] == 1

  def test_tier_agent_default_n_steps(self, env_config):
    '''TierAgent keeps its n_steps default of 512.'''
    orch = TestTierAgent()._tier_env(env_config)
    before = len(_RecordingMaskablePPO.instances)
    with _patched_construction():
      agent = TierAgent(orch, _agent_config())
    instance = _RecordingMaskablePPO.instances[before]
    assert agent.model is instance
    assert instance.kwargs['n_steps'] == 512
    assert instance.kwargs['verbose'] == 0


class TestDocstrings:
  '''Tests that stub docstrings were expanded to Google style.'''

  def test_masked_agent_docstrings(self):
    '''MaskedAgent members document Args, Returns and Raises.'''
    assert 'Args:' in MaskedAgent.__doc__
    assert 'Args:' in MaskedAgent.__init__.__doc__
    assert 'Args:' in MaskedAgent.train.__doc__
    assert 'Returns:' in MaskedAgent.predict.__doc__
    assert 'Raises:' in MaskedAgent.load.__doc__
    assert 'trusted' in MaskedAgent.load.__doc__.lower()

  def test_callback_docstrings(self):
    '''Callbacks document Args and Returns.'''
    assert 'Args:' in CheckpointCallback.__init__.__doc__
    assert 'Returns:' in TensorBoardCallback._on_step.__doc__
    assert 'Returns:' in CheckpointCallback._on_step.__doc__


class TestSec01Integrity:
  '''Tests for SEC-01 trusted-only model loading and sidecars.'''

  def _writing_model(self):
    '''Return a model mock whose save writes bytes to the path.'''

    model = mock.MagicMock()

    def _save(path):
      '''Write deterministic bytes to the save path.'''
      with open(path, 'wb') as handle:
        handle.write(b'fake-model-bytes')

    model.save.side_effect = _save
    return model

  def test_save_writes_sha256_sidecar(self, env):
    '''save records a SHA256 sidecar next to the model.'''
    agent = MaskedAgent(env, _agent_config())
    agent._model = self._writing_model()
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'model')
      agent.save(path)
      sidecar = path + '.sha256'
      assert os.path.isfile(sidecar)
      expected = hashlib.sha256(b'fake-model-bytes').hexdigest()
      with open(sidecar, encoding='utf-8') as handle:
        assert handle.read().strip() == expected

  def test_load_without_sidecar_warns_but_loads(
    self, env, caplog
  ):
    '''load warns when no sidecar exists but still loads.'''
    with mock.patch(
      'src.agent.masked_agent.MaskablePPO'
    ) as mock_ppo:
      mock_ppo.load.return_value = mock.MagicMock()
      with caplog.at_level(
        logging.WARNING, logger='src.agent.masked_agent'
      ):
        agent = MaskedAgent.load('no/sidecar/here', env, _agent_config())
      assert agent.model is mock_ppo.load.return_value
      assert 'sidecar' in caplog.text.lower()

  def test_load_verifies_valid_sidecar(self, env):
    '''load succeeds when the sidecar digest matches.'''
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'model')
      with open(path, 'wb') as handle:
        handle.write(b'fake-model-bytes')
      digest = hashlib.sha256(b'fake-model-bytes').hexdigest()
      with open(path + '.sha256', 'w', encoding='utf-8') as handle:
        handle.write(digest)
      with mock.patch(
        'src.agent.masked_agent.MaskablePPO'
      ) as mock_ppo:
        mock_ppo.load.return_value = mock.MagicMock()
        agent = MaskedAgent.load(path, env, _agent_config())
        assert agent.model is mock_ppo.load.return_value

  def test_load_rejects_tampered_sidecar(self, env):
    '''load raises when the sidecar digest does not match.'''
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'model')
      with open(path, 'wb') as handle:
        handle.write(b'tampered-bytes')
      with open(path + '.sha256', 'w', encoding='utf-8') as handle:
        handle.write('0' * 64)
      with mock.patch(
        'src.agent.masked_agent.MaskablePPO'
      ):
        with pytest.raises(ValueError, match='SHA256'):
          MaskedAgent.load(path, env, _agent_config())

  def test_tier_agent_save_and_load_sidecar(self, env_config):
    '''TierAgent inherits sidecar save and verified load.'''
    tier_env = TestTierAgent()._tier_env(env_config)
    agent = TierAgent(tier_env, _agent_config())
    agent._model = self._writing_model()
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'tiermodel')
      agent.save(path)
      assert os.path.isfile(path + '.sha256')
      with mock.patch(
        'src.agent.masked_agent.MaskablePPO'
      ) as mock_ppo:
        mock_ppo.load.return_value = mock.MagicMock()
        loaded = TierAgent.load(path, tier_env, _agent_config())
        assert loaded.model is mock_ppo.load.return_value


if __name__ == '__main__':
  pytest.main([__file__])
