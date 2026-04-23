from __future__ import annotations

import os
from typing import Any, Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardCallback(BaseCallback):
  def __init__(self, verbose: int = 0):
    super().__init__(verbose)

  def _on_step(self) -> bool:
    infos = self.locals.get("infos", [])
    for info in infos:
      reward_components = info.get("reward_components", {})
      for key, value in reward_components.items():
        if isinstance(value, (int, float)):
          self.logger.record(f"reward/{key}", value)

      action_info = info.get("action_info", {})
      applied = action_info.get("applied", {})
      rejected = action_info.get("rejected", {})
      self.logger.record("actions/applied_count", len(applied))
      self.logger.record("actions/rejected_count", len(rejected))

      tick = info.get("tick", 0)
      self.logger.record("env/tick", tick)
    return True


class CheckpointCallback(BaseCallback):
  def __init__(
    self,
    save_freq: int = 50000,
    save_path: str = "output/models",
    verbose: int = 0,
  ):
    super().__init__(verbose)
    self._save_freq = save_freq
    self._save_path = save_path
    os.makedirs(save_path, exist_ok=True)

  def _on_step(self) -> bool:
    if self.n_calls % self._save_freq == 0:
      path = os.path.join(
        self._save_path, f"checkpoint_{self.n_calls}"
      )
      self.model.save(path)
      if self.verbose:
        print(f"Saved checkpoint to {path}")
    return True
