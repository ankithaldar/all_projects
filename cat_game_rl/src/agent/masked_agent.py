#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Masked PPO agents for single and tier envs.'''

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from src.agent.action_mask_fn import compute_action_mask
from src.agent.callbacks import CheckpointCallback, TensorBoardCallback
from src.cat_game_env.crafting_env import CraftingEnv
from src.cat_game_env.multi_agent import TierEnv

logger = logging.getLogger(__name__)

_AgentT = TypeVar('_AgentT', bound='BaseMaskedAgent')


def _artifact_path(path: str) -> Optional[str]:
  '''Resolve the on-disk model file for a save path.

  Args:
    path: Save path passed to ``MaskablePPO.save``.

  Returns:
    The model file path, or ``None`` when no file exists yet.
  '''
  if os.path.isfile(path):
    return path
  zipped = path + '.zip'
  if os.path.isfile(zipped):
    return zipped
  return None


def _sha256_of(path: str) -> str:
  '''Compute the SHA256 hex digest of a file.

  Args:
    path: File to hash.

  Returns:
    The SHA256 hex digest of the file contents.
  '''
  digest = hashlib.sha256()
  with open(path, 'rb') as handle:
    for chunk in iter(lambda: handle.read(65536), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _write_sidecar(path: str) -> None:
  '''Record a SHA256 sidecar for a saved model.

  The sidecar lives at ``path + '.sha256'`` and contains the
  hex digest of the saved model file.

  Args:
    path: Save path passed to ``MaskablePPO.save``.
  '''
  artifact = _artifact_path(path)
  if artifact is None:
    logger.warning(
      'No model file at %s; skipping SHA256 sidecar.', path
    )
    return
  with open(path + '.sha256', 'w', encoding='utf-8') as handle:
    handle.write(_sha256_of(artifact) + '\n')


def _verify_sidecar(path: str) -> None:
  '''Verify a saved model against its SHA256 sidecar.

  Only load models from trusted sources: a matching sidecar
  proves the file is unchanged, while a missing sidecar means
  the model cannot be verified and loading proceeds with a
  warning for backwards compatibility.

  Args:
    path: Save path passed to ``MaskablePPO.load``.

  Raises:
    ValueError: The sidecar digest does not match the model
      file, or the sidecar exists but the model file is gone.
  '''
  sidecar = path + '.sha256'
  if not os.path.isfile(sidecar):
    logger.warning(
      'No SHA256 sidecar at %s; loading unverified model. '
      'Only load models from trusted sources.',
      sidecar,
    )
    return
  artifact = _artifact_path(path)
  if artifact is None:
    raise ValueError(
      f'Missing model file for SHA256 verification: {path}'
    )
  with open(sidecar, encoding='utf-8') as handle:
    expected = handle.read().strip()
  actual = _sha256_of(artifact)
  if actual != expected:
    raise ValueError(
      f'SHA256 mismatch for {path}: '
      f'expected {expected}, got {actual}'
    )


class BaseMaskedAgent:
  '''Shared masked-PPO agent implementation.

  Args:
    env: Environment to wrap with the shared action mask.
    config: Configuration mapping with optional ``policy``,
      ``hyperparameters`` and ``training`` sections.
    n_steps_default: Default ``n_steps`` when the config does
      not override it.
    verbose_default: SB3 verbosity passed to the model.
    use_tensorboard: Whether to pass a TensorBoard log dir.
    policy_override: Policy name forced for every config, or
      ``None`` to read the policy from the config.
  '''

  def __init__(
    self,
    env: Union[CraftingEnv, TierEnv],
    config: Dict[str, Any],
    *,
    n_steps_default: int = 2048,
    verbose_default: int = 1,
    use_tensorboard: bool = True,
    policy_override: Optional[str] = None,
  ):
    '''Initialize the agent and build the masked model.

    Args:
      env: Environment to wrap with the shared action mask.
      config: Configuration mapping with optional ``policy``,
        ``hyperparameters`` and ``training`` sections.
      n_steps_default: Default ``n_steps`` when the config does
        not override it.
      verbose_default: SB3 verbosity passed to the model.
      use_tensorboard: Whether to pass a TensorBoard log dir.
      policy_override: Policy name forced for every config, or
        ``None`` to read the policy from the config.
    '''
    self._config = config
    hp = config.get('hyperparameters', {})

    wrapped_env = ActionMasker(env, compute_action_mask)

    if policy_override is not None:
      policy = policy_override
    else:
      policy = config.get('policy', 'MultiInputPolicy')

    extra: Dict[str, Any] = {}
    if use_tensorboard:
      extra['tensorboard_log'] = config.get('training', {}).get(
        'log_dir', 'output/logs'
      )

    self._model = MaskablePPO(
      policy,
      wrapped_env,
      learning_rate=hp.get('learning_rate', 3e-4),
      n_steps=hp.get('n_steps', n_steps_default),
      batch_size=hp.get('batch_size', 64),
      n_epochs=hp.get('n_epochs', 10),
      gamma=hp.get('gamma', 0.99),
      gae_lambda=hp.get('gae_lambda', 0.95),
      clip_range=hp.get('clip_range', 0.2),
      ent_coef=hp.get('ent_coef', 0.01),
      vf_coef=hp.get('vf_coef', 0.5),
      max_grad_norm=hp.get('max_grad_norm', 0.5),
      verbose=verbose_default,
      **extra,
    )

  @property
  def model(self) -> MaskablePPO:
    '''Return the underlying MaskablePPO model.

    Returns:
      The SB3 ``MaskablePPO`` model backing this agent.
    '''
    return self._model

  def predict(
    self,
    obs: Dict[str, np.ndarray],
    action_masks: Optional[np.ndarray] = None,
    deterministic: bool = True,
  ) -> np.ndarray:
    '''Predict an action for an observation.

    Args:
      obs: Observation dictionary for the environment.
      action_masks: Optional action mask overriding the
        environment mask for this prediction.
      deterministic: Whether to sample greedily.

    Returns:
      The predicted action array.
    '''
    action, _ = self._model.predict(
      obs,
      deterministic=deterministic,
      action_masks=action_masks,
    )
    return action

  def save(self, path: str) -> None:
    '''Save the model and record a SHA256 sidecar.

    Args:
      path: Destination save path for the model.
    '''
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    self._model.save(path)
    _write_sidecar(path)

  @classmethod
  def load(
    cls: Type[_AgentT],
    path: str,
    env: Union[CraftingEnv, TierEnv],
    config: Dict[str, Any],
  ) -> _AgentT:
    '''Load a saved model as an agent.

    Only load models from trusted sources. When a SHA256
    sidecar exists next to the model it is verified and a
    mismatch raises; without a sidecar the model is loaded
    unverified with a warning, so older saves still work.

    Args:
      path: Save path of the model to load.
      env: Environment to wrap with the shared action mask.
      config: Configuration mapping stored on the agent.

    Returns:
      A new agent instance wrapping the loaded model.

    Raises:
      ValueError: The sidecar digest does not match, or the
        sidecar exists but the model file is missing.
    '''
    _verify_sidecar(path)
    agent = cls.__new__(cls)
    agent._config = config
    tier = getattr(env, 'tier', None)
    if tier is not None:
      agent._tier = tier
    wrapped_env = ActionMasker(env, compute_action_mask)
    agent._model = MaskablePPO.load(path, env=wrapped_env)
    return agent


class MaskedAgent(BaseMaskedAgent):
  '''Masked PPO agent for a crafting environment.

  Args:
    env: Crafting environment to train and act in.
    config: Configuration mapping with optional ``policy``,
      ``hyperparameters`` and ``training`` sections.
  '''

  def __init__(self, env: CraftingEnv, config: Dict[str, Any]):
    '''Initialize the agent and build the masked model.

    Args:
      env: Crafting environment to train and act in.
      config: Configuration mapping with optional ``policy``,
        ``hyperparameters`` and ``training`` sections.
    '''
    super().__init__(
      env,
      config,
      n_steps_default=2048,
      verbose_default=1,
      use_tensorboard=True,
    )

  def train(
    self, total_timesteps: int, callbacks: list | None = None
  ) -> None:
    '''Train the model for a number of timesteps.

    Args:
      total_timesteps: Number of environment steps to train
        the model for.
      callbacks: Optional SB3 callbacks; when omitted, the
        default TensorBoard and checkpoint callbacks are
        built from the training config.
    '''
    if callbacks is None:
      train_cfg = self._config.get('training', {})
      callbacks = [
        TensorBoardCallback(),
        CheckpointCallback(
          save_freq=train_cfg.get('checkpoint_freq', 50000),
          save_path=train_cfg.get('model_dir', 'output/models'),
        ),
      ]
    self._model.learn(
      total_timesteps=total_timesteps,
      callback=callbacks,
    )


class TierAgent(BaseMaskedAgent):
  '''Masked PPO agent for a single tier environment.

  Args:
    tier_env: Tier environment to train and act in.
    config: Configuration mapping with optional
      ``hyperparameters`` section.
  '''

  def __init__(self, tier_env: TierEnv, config: Dict[str, Any]):
    '''Initialize the agent and build the masked model.

    Args:
      tier_env: Tier environment to train and act in.
      config: Configuration mapping with optional
        ``hyperparameters`` section.
    '''
    self._tier = tier_env.tier
    super().__init__(
      tier_env,
      config,
      n_steps_default=512,
      verbose_default=0,
      use_tensorboard=False,
      policy_override='MultiInputPolicy',
    )

  @property
  def tier(self) -> int:
    '''Return the tier number of the wrapped environment.

    Returns:
      The tier number of the wrapped tier environment.
    '''
    return self._tier
