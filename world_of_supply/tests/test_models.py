#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the PyTorch FacilityNet policy/value network.'''

import pytest

torch = pytest.importorskip('torch')

from world_of_supply.rl.models import FacilityNet


def test_forward_shapes_match_multi_discrete_heads():
  net = FacilityNet(obs_dim=50, action_nvec=[8, 6], hidden_size=32)
  logits, values, state = net(torch.zeros(4, 50))

  assert logits.shape == (4, 14)
  assert values.shape == (4,)
  assert state == []


def test_lstm_variant_carries_state():
  net = FacilityNet(obs_dim=50, action_nvec=[3, 2, 6], hidden_size=16, lstm_cell_size=8, use_lstm=True)
  initial = net.initial_state(batch_size=4)
  assert len(initial) == 2

  logits, values, new_state = net(torch.zeros(4, 50), initial)
  assert logits.shape == (4, 11)
  assert values.shape == (4,)
  assert all(tensor.shape[0] == 1 for tensor in new_state)


def test_gradient_flows_through_both_heads():
  net = FacilityNet(obs_dim=10, action_nvec=[5, 5], hidden_size=8)
  logits, values, _ = net(torch.randn(2, 10))
  loss = logits.sum() + values.sum()
  loss.backward()

  grads = [parameter.grad for parameter in net.parameters()]
  assert all(grad is not None for grad in grads)


def test_multi_layer_trunk_matches_legacy_widths():
  net = FacilityNet(obs_dim=50, action_nvec=[8, 6], hidden_size=128, hidden_layers=2)
  linear_layers = [m for m in net.trunk if isinstance(m, torch.nn.Linear)]
  assert len(linear_layers) == 2
  assert linear_layers[0].out_features == 128
  assert linear_layers[1].in_features == 128
  logits, _, _ = net(torch.zeros(2, 50))
  assert logits.shape == (2, 14)
