#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Shared episode loop for evaluation scripts.

Single home for the reset->masks->predict->step loop
previously duplicated in evaluate.py, visualize.py and
dashboard.py. Callers keep their distinct tails via the
on_step callback (metrics, frame capture, Streamlit live
updates) and stop conditions via should_continue.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
  from truck_carton.env.packing_env import (
    TruckCartonPackingEnv,
  )

PredictFn = Callable[
  ['TruckCartonPackingEnv', Any, np.ndarray], int
]
StepFn = Callable[
  [int, Any, float, dict, bool], None
]
ContinueFn = Callable[[], bool]


@dataclass
class EpisodeResult:
  '''Outcome of one shared episode run.

  Attributes:
    total_reward: Sum of step rewards.
    steps: Number of env.step calls executed.
    obs: Final observation returned by the last step.
    info: Info dict from the last step ({} if none).
    terminated: Whether the env terminated the episode.
    truncated: Whether the env truncated the episode.
  '''

  total_reward: float = 0.0
  steps: int = 0
  obs: Any = None
  info: dict = field(default_factory=dict)
  terminated: bool = False
  truncated: bool = False

  @property
  def done(self) -> bool:
    '''Return whether the episode ended.'''
    return self.terminated or self.truncated


def model_predict(
  model: Any,
  obs: Any,
  masks: np.ndarray,
) -> int:
  '''Predict a masked action with a trained model.

  Args:
    model: Trained model with a masked predict API.
    obs: Current observation.
    masks: Boolean action mask.

  Returns:
    Predicted action index as an int.
  '''
  action, _ = model.predict(
    obs,
    action_masks=masks,
    deterministic=True,
  )
  return int(action)


def random_predict(
  rng: np.random.Generator,
  masks: np.ndarray,
) -> int:
  '''Pick a uniformly random valid action.

  Args:
    rng: Generator used for the choice.
    masks: Boolean action mask.

  Returns:
    Random valid action index, or 0 if none are valid.
  '''
  valid = np.where(masks)[0]
  if len(valid) == 0:
    return 0
  return int(rng.choice(valid))


def make_predict(
  model: Any = None,
  agent: Any = None,
  rng: np.random.Generator | None = None,
) -> PredictFn:
  '''Build a predict function for run_episode.

  Model takes precedence, then a heuristic/random agent
  with predict(env, masks), then uniform random choice.

  Args:
    model: Optional trained model.
    agent: Optional agent with predict(env, masks).
    rng: Generator for the random fallback.

  Returns:
    Callable matching the PredictFn signature.
  '''
  generator = rng or np.random.default_rng(42)

  def _predict(
    env: TruckCartonPackingEnv,
    obs: Any,
    masks: np.ndarray,
  ) -> int:
    if model is not None:
      return model_predict(model, obs, masks)
    if agent is not None:
      return int(agent.predict(env, masks))
    return random_predict(generator, masks)

  return _predict


def run_episode(
  env: TruckCartonPackingEnv,
  obs: Any,
  predict: PredictFn,
  *,
  on_step: StepFn | None = None,
  should_continue: ContinueFn | None = None,
  max_steps: int | None = None,
) -> EpisodeResult:
  '''Run one episode from an already-reset observation.

  The env must already be reset; obs is the reset output.
  Loop: masks -> break if none valid -> predict ->
  step -> accumulate reward -> on_step callback.
  Stops on terminated/truncated, empty masks, a false
  should_continue(), or max_steps.

  Args:
    env: Packing environment to step.
    obs: Observation from env.reset().
    predict: Action selector (env, obs, masks) -> int.
    on_step: Optional (step, obs, reward, info, done)
      callback invoked after every step.
    should_continue: Optional gate checked each step;
      the episode stops when it returns False.
    max_steps: Optional cap on steps executed.

  Returns:
    EpisodeResult with totals and final state.
  '''
  result = EpisodeResult(obs=obs)
  while not result.done:
    if max_steps is not None and result.steps >= max_steps:
      break
    if should_continue is not None and not should_continue():
      break
    masks = env.action_masks()
    if not masks.any():
      break
    action = int(predict(env, result.obs, masks))
    result.obs, reward, terminated, truncated, info = env.step(
      action
    )
    result.total_reward += float(reward)
    result.steps += 1
    result.terminated = bool(terminated)
    result.truncated = bool(truncated)
    result.info = info if isinstance(info, dict) else {}
    if on_step is not None:
      on_step(
        result.steps,
        result.obs,
        float(reward),
        result.info,
        result.done,
      )
  return result
