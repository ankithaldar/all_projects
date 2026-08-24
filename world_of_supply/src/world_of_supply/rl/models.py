#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''PyTorch policy/value network for facility agents.

This is the direct PyTorch port of the legacy TensorFlow ``FacilityNet``
(Dense → optional LSTM → linear policy logits + scalar value head).
'''

from __future__ import annotations

import torch
from torch import nn


class FacilityNet(nn.Module):
  '''MLP/LSTM trunk with multi-discrete policy and value heads.

  The policy head emits one concatenated logit per action dimension of a
  ``MultiDiscrete`` space; the categorical split happens downstream.

  Attributes:
    use_lstm: Whether an LSTM sits between the trunk and the heads.
    cell_size: LSTM hidden size (0 when ``use_lstm`` is False).
  '''

  def __init__(
      self,
      obs_dim: int,
      action_nvec: list[int],
      hidden_size: int = 256,
      lstm_cell_size: int = 64,
      use_lstm: bool = False,
  ) -> None:
    '''Build the network.

    Args:
      obs_dim: Flattened observation dimension.
      action_nvec: Category counts per MultiDiscrete branch.
      hidden_size: Width of the fully connected trunk.
      lstm_cell_size: Hidden size of the optional LSTM.
      use_lstm: Enable the recurrent layer.
    '''
    super().__init__()
    self.use_lstm = use_lstm
    self.cell_size = lstm_cell_size if use_lstm else 0
    self.action_nvec = list(action_nvec)

    self.trunk = nn.Sequential(
        nn.Linear(obs_dim, hidden_size),
        nn.ReLU(),
    )
    trunk_out = hidden_size
    if use_lstm:
      self.lstm = nn.LSTM(hidden_size, lstm_cell_size, batch_first=True)
      trunk_out = lstm_cell_size
    self.policy_head = nn.Linear(trunk_out, int(sum(action_nvec)))
    self.value_head = nn.Linear(trunk_out, 1)

  def initial_state(self, batch_size: int = 1) -> list[torch.Tensor]:
    '''Create zeroed recurrent state tensors.

    Args:
      batch_size: Leading batch dimension.

    Returns:
      list[torch.Tensor]: ``[h, c]`` when LSTM is enabled, else empty list.
    '''
    if not self.use_lstm:
      return []
    options = (
        torch.zeros(1, batch_size, self.cell_size),
        torch.zeros(1, batch_size, self.cell_size),
    )
    return [options[0], options[1]]

  def forward(
      self,
      observations: torch.Tensor,
      state: list[torch.Tensor] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    '''Compute policy logits and value estimates.

    Args:
      observations: Float tensor shaped ``(batch, obs_dim)`` or
        ``(batch, seq_len, obs_dim)`` when recurrent.
      state: Optional ``[h, c]`` tensors for the LSTM path.

    Returns:
      tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]: Logits,
      scalar values, and the new recurrent state.
    '''
    new_state: list[torch.Tensor] = []
    if self.use_lstm:
      if observations.dim() == 2:
        observations = observations.unsqueeze(1)
      h, c = state if state is not None else self.initial_state(observations.shape[0])
      features, (h, c) = self.lstm(self.trunk(observations), (h, c))
      features = features[:, -1, :]
      new_state = [h.contiguous(), c.contiguous()]
    else:
      features = self.trunk(observations)
    logits = self.policy_head(features)
    values = self.value_head(features).squeeze(-1)
    return logits, values, new_state
