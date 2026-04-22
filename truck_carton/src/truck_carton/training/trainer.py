from __future__ import annotations

from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
  EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from truck_carton.config import AppConfig
from truck_carton.curriculum.manager import (
  CurriculumManager,
)
from truck_carton.env.packing_env import (
  TruckCartonPackingEnv,
)
from truck_carton.training.callbacks import (
  CurriculumCallback,
  MetricsCallback,
)


class Trainer:
  """Orchestrates the complete training pipeline
  with MaskablePPO and curriculum learning."""

  def __init__(
    self,
    config: AppConfig,
    output_dir: str = './output',
  ) -> None:
    self._config = config
    self._output_dir = Path(output_dir)
    self._output_dir.mkdir(
      parents=True, exist_ok=True
    )

  def train(self) -> MaskablePPO:
    env = self._make_env()
    eval_env = self._make_env()

    curriculum = CurriculumManager(
      self._config.curriculum
    )

    tc = self._config.training
    model = MaskablePPO(
      'MultiInputPolicy',
      env,
      learning_rate=tc.learning_rate,
      n_steps=tc.n_steps,
      batch_size=tc.batch_size,
      n_epochs=tc.n_epochs,
      gamma=tc.gamma,
      gae_lambda=tc.gae_lambda,
      clip_range=tc.clip_range,
      ent_coef=tc.ent_coef,
      vf_coef=tc.vf_coef,
      policy_kwargs={
        'net_arch': list(tc.net_arch)
      },
      verbose=1,
      tensorboard_log=str(
        self._output_dir / 'tb_logs'
      ),
      seed=tc.seed,
    )

    callbacks = [
      CurriculumCallback(
        curriculum, verbose=1
      ),
      MetricsCallback(),
      EvalCallback(
        eval_env,
        n_eval_episodes=tc.eval_episodes,
        eval_freq=tc.eval_freq,
        best_model_save_path=str(
          self._output_dir / 'models'
        ),
        log_path=str(
          self._output_dir / 'logs'
        ),
        verbose=1,
      ),
    ]

    model.learn(
      total_timesteps=tc.total_timesteps,
      callback=callbacks,
    )

    final_path = (
      self._output_dir / 'models' / 'final_model'
    )
    model.save(str(final_path))
    return model

  def _make_env(self) -> Monitor:
    env = TruckCartonPackingEnv(
      config=self._config
    )
    return Monitor(env)
