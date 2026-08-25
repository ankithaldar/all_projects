#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''RLlib RLModule wrappers around :class:`FacilityNet` (new API stack).'''

from __future__ import annotations

try:
  from ray.rllib.core.columns import Columns
  from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
  from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

  HAS_RAY_RLMODULES = True
except ImportError:
  HAS_RAY_RLMODULES = False

import numpy as np
import torch

from world_of_supply.rl.models import FacilityNet

_ACTION_DIST_INPUTS_KEY = 'action_dist_inputs'
_VF_PREDS_KEY = 'vf_preds'


def make_multi_categorical_cls(branch_sizes: list[int]):
  '''Create a TorchMultiCategorical bound to fixed branch sizes.

  RLLib connectors call ``from_logits`` without ``input_lens``; this closure
  binds the category counts of one specific action space.

  Args:
    branch_sizes: Category count per MultiDiscrete branch.

  Returns:
    type: Bound distribution class.
  '''
  from ray.rllib.core.distribution.torch.torch_distribution import (
      TorchMultiCategorical,
  )

  class BoundMultiCategorical(TorchMultiCategorical):

    @classmethod
    def from_logits(cls, logits, input_lens=None, temperatures=None, **kwargs):
      '''Build the distribution, defaulting to the bound branch sizes.

      Args:
        logits: Concatenated per-branch logits.
        input_lens: Branch sizes; falls back to the bound values.
        temperatures: Optional sampling temperatures.
        **kwargs: Forwarded to the base implementation.

      Returns:
        TorchMultiCategorical: Ready-to-sample distribution.
      '''
      resolved = input_lens if input_lens is not None else list(branch_sizes)
      return super().from_logits(logits, resolved, temperatures=temperatures, **kwargs)

  return BoundMultiCategorical


def _module_config(module) -> dict:
  '''Resolve the configuration object across Ray versions.

  Ray renamed ``config`` to ``cfg`` during the 2.x series; this helper keeps
  the module compatible with either field name.

  Args:
    module: RLModule instance.

  Returns:
    dict: Model configuration dictionary.
  '''
  config_obj = getattr(module, 'cfg', None)
  if config_obj is None:
    config_obj = getattr(module, 'config', None)
  model_config = getattr(config_obj, 'model_config', None) or {}
  return dict(model_config)


if HAS_RAY_RLMODULES:

  class FacilityRLModule(TorchRLModule, ValueFunctionAPI):
    '''FacilityNet-backed module emitting PPO-compatible outputs.

    Produces ``action_dist_inputs`` (concatenated MultiDiscrete logits) for
    all forward modes, ``vf_preds`` for training, and implements
    :class:`ValueFunctionAPI` so RLLib's GAE connector can query values.
    '''

    def setup(self) -> None:
      '''Build the underlying FacilityNet from the module configuration.'''
      super().setup()
      config = _module_config(self)
      obs_space = self._obs_space()
      action_space = self._action_space()
      nvec = list(getattr(action_space, 'nvec', []))
      self.net = FacilityNet(
          obs_dim=int(obs_space.shape[0]),
          action_nvec=nvec,
          hidden_size=int(config.get('hidden_size', 256)),
          hidden_layers=int(config.get('hidden_layers', 1)),
          lstm_cell_size=int(config.get('lstm_cell_size', 64)),
          use_lstm=bool(config.get('use_lstm', False)),
      )
      self.action_dist_cls = make_multi_categorical_cls(nvec)

    def _obs_space(self):
      '''Return the observation space bound to this module.

      Returns:
        object: Space object carrying a ``shape`` attribute.
      '''
      config_obj = getattr(self, 'cfg', None) or getattr(self, 'config')
      return config_obj.observation_space or config_obj.space

    def _action_space(self):
      '''Return the action space bound to this module.

      Returns:
        object: Space object; MultiDiscrete expected.
      '''
      config_obj = getattr(self, 'cfg', None) or getattr(self, 'config')
      return config_obj.action_space

    def _forward(self, batch: dict) -> dict:
      '''Shared computation producing logits and value predictions.

      When the module is recurrent, RLLib-provided ``state_in`` tensors are
      fed into the LSTM so state actually carries across timesteps.

      Args:
        batch: Input batch containing observations under ``'obs'``.

      Returns:
        dict: Action-distribution inputs plus value predictions.
      '''
      obs = batch['obs']
      tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
      state = None
      if self.net.use_lstm:
        raw_state = batch.get(Columns.STATE_IN)
        if raw_state is not None:
          state = [torch.as_tensor(np.asarray(part), dtype=torch.float32) for part in raw_state]
      logits, values, _ = self.net(tensor, state)
      return {_ACTION_DIST_INPUTS_KEY: logits, _VF_PREDS_KEY: values}

    def forward_inference(self, batch: dict, **kwargs) -> dict:
      '''Compute inference outputs (no value head required).

      Args:
        batch: Input batch.
        **kwargs: Ignored framework hooks.

      Returns:
        dict: Action-distribution inputs only.
      '''
      return {_ACTION_DIST_INPUTS_KEY: self._forward(batch)[_ACTION_DIST_INPUTS_KEY]}

    def forward_exploration(self, batch: dict, **kwargs) -> dict:
      '''Compute exploration outputs identical to inference here.

      Args:
        batch: Input batch.
        **kwargs: Ignored framework hooks.

      Returns:
        dict: Action-distribution inputs only.
      '''
      return {_ACTION_DIST_INPUTS_KEY: self._forward(batch)[_ACTION_DIST_INPUTS_KEY]}

    def forward_train(self, batch: dict, **kwargs) -> dict:
      '''Compute training outputs including value predictions.

      Args:
        batch: Training batch.
        **kwargs: Ignored framework hooks.

      Returns:
        dict: Action-distribution inputs plus value predictions.
      '''
      return self._forward(batch)

    def compute_values(self, input_batch: dict, embeddings=None, **kwargs):
      '''Predict per-timestep state values (ValueFunctionAPI).

      Args:
        input_batch: Batch containing observations under the standard key.
        embeddings: Optional encoder embeddings (unused).
        **kwargs: Ignored framework hooks.

      Returns:
        torch.Tensor: Value estimates shaped ``(batch,)``.
      '''
      observations = input_batch.get(Columns.OBS, None)
      if observations is None:
        observations = np.asarray(input_batch['obs'])
      tensor = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
      _, values, _ = self.net(tensor)
      return values

    def get_initial_state(self) -> dict:
      '''Provide the recurrent initial state.

      Returns:
        dict: ``{'h', 'c'}`` zero tensors when recurrent, else empty dict.
      '''
      if not self.net.use_lstm:
        return {}
      state_h, state_c = self.net.initial_state()
      return {'h': state_h, 'c': state_c}

  __all__ = ['FacilityRLModule', 'HAS_RAY_RLMODULES']
else:
  FacilityRLModule = None  # type: ignore[assignment,misc]
  __all__ = ['HAS_RAY_RLMODULES']
