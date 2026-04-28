from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from src.cat_game_env.crafting_env import CraftingEnv
from src.cat_game_env.frame_skipper import FrameSkipWrapper
from src.cat_game_env.multi_agent import TierEnv
from src.agent.action_mask_fn import compute_action_mask
from src.agent.callbacks import TensorBoardCallback, CheckpointCallback


def _mask_fn(env: CraftingEnv) -> np.ndarray:
  return env.action_masks()


class MaskedAgent:
  def __init__(self, env: CraftingEnv, config: Dict[str, Any]):
    self._config = config
    hp = config.get("hyperparameters", {})

    wrapped_env = ActionMasker(env, _mask_fn)

    policy = config.get("policy", "MultiInputPolicy")
    self._model = MaskablePPO(
      policy,
      wrapped_env,
      learning_rate=hp.get("learning_rate", 3e-4),
      n_steps=hp.get("n_steps", 2048),
      batch_size=hp.get("batch_size", 64),
      n_epochs=hp.get("n_epochs", 10),
      gamma=hp.get("gamma", 0.99),
      gae_lambda=hp.get("gae_lambda", 0.95),
      clip_range=hp.get("clip_range", 0.2),
      ent_coef=hp.get("ent_coef", 0.01),
      vf_coef=hp.get("vf_coef", 0.5),
      max_grad_norm=hp.get("max_grad_norm", 0.5),
      verbose=1,
      tensorboard_log=config.get("training", {}).get(
        "log_dir", "output/logs"
      ),
    )

  @property
  def model(self) -> MaskablePPO:
    return self._model

  def train(self, total_timesteps: int, callbacks: list | None = None) -> None:
    if callbacks is None:
      train_cfg = self._config.get("training", {})
      callbacks = [
        TensorBoardCallback(),
        CheckpointCallback(
          save_freq=train_cfg.get("checkpoint_freq", 50000),
          save_path=train_cfg.get("model_dir", "output/models"),
        ),
      ]
    self._model.learn(
      total_timesteps=total_timesteps,
      callback=callbacks,
    )

  def predict(
    self,
    obs: Dict[str, np.ndarray],
    action_masks: Optional[np.ndarray] = None,
    deterministic: bool = True,
  ) -> np.ndarray:
    action, _ = self._model.predict(
      obs,
      deterministic=deterministic,
      action_masks=action_masks,
    )
    return action

  def save(self, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    self._model.save(path)

  @classmethod
  def load(
    cls, path: str, env: CraftingEnv, config: Dict[str, Any]
  ) -> MaskedAgent:
    agent = cls.__new__(cls)
    agent._config = config
    wrapped_env = ActionMasker(env, _mask_fn)
    agent._model = MaskablePPO.load(path, env=wrapped_env)
    return agent


def _tier_mask_fn(env: TierEnv) -> np.ndarray:
  return env.action_masks()


class TierAgent:
  def __init__(self, tier_env: TierEnv, config: Dict[str, Any]):
    self._config = config
    self._tier = tier_env.tier
    hp = config.get("hyperparameters", {})

    wrapped = ActionMasker(tier_env, _tier_mask_fn)
    self._model = MaskablePPO(
      "MultiInputPolicy",
      wrapped,
      learning_rate=hp.get("learning_rate", 3e-4),
      n_steps=hp.get("n_steps", 512),
      batch_size=hp.get("batch_size", 64),
      n_epochs=hp.get("n_epochs", 10),
      gamma=hp.get("gamma", 0.99),
      gae_lambda=hp.get("gae_lambda", 0.95),
      clip_range=hp.get("clip_range", 0.2),
      ent_coef=hp.get("ent_coef", 0.01),
      vf_coef=hp.get("vf_coef", 0.5),
      max_grad_norm=hp.get("max_grad_norm", 0.5),
      verbose=0,
    )

  @property
  def tier(self) -> int:
    return self._tier

  @property
  def model(self) -> MaskablePPO:
    return self._model

  def predict(
    self,
    obs: Dict[str, np.ndarray],
    action_masks: Optional[np.ndarray] = None,
    deterministic: bool = True,
  ) -> np.ndarray:
    action, _ = self._model.predict(
      obs, deterministic=deterministic, action_masks=action_masks,
    )
    return action

  def save(self, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    self._model.save(path)

  @classmethod
  def load(
    cls, path: str, tier_env: TierEnv, config: Dict[str, Any]
  ) -> TierAgent:
    agent = cls.__new__(cls)
    agent._config = config
    agent._tier = tier_env.tier
    wrapped = ActionMasker(tier_env, _tier_mask_fn)
    agent._model = MaskablePPO.load(path, env=wrapped)
    return agent
