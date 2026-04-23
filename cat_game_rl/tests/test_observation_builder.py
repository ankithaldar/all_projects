from __future__ import annotations

import numpy as np
import pytest

from src.core.items import NUM_ITEMS, NUM_CRAFTABLE
from src.core.inventory import Stash
from src.core.coin_generator import CoinGenerator
from src.core.slot_scheduler import SlotScheduler
from src.core.target_provider import TargetProvider
from src.env.observation_builder import ObservationBuilder


class TestObservationBuilder:
  @pytest.fixture
  def obs_builder(self) -> ObservationBuilder:
    return ObservationBuilder(max_ticks=2016)

  def test_build_space(self, obs_builder: ObservationBuilder):
    space = obs_builder.build_space()
    assert "stash" in space.spaces
    assert "coins" in space.spaces
    assert "slots" in space.spaces
    assert "time_fraction" in space.spaces
    assert "current_tick" in space.spaces
    assert "targets_remaining" in space.spaces
    assert "targets_total" in space.spaces

  def test_build_space_shapes(self, obs_builder: ObservationBuilder):
    space = obs_builder.build_space()
    assert space["stash"].shape == (NUM_ITEMS,)
    assert space["coins"].shape == (1,)
    assert space["slots"].shape == (NUM_CRAFTABLE, 3)
    assert space["time_fraction"].shape == (1,)
    assert space["current_tick"].shape == (1,)
    assert space["targets_remaining"].shape == (NUM_ITEMS,)
    assert space["targets_total"].shape == (NUM_ITEMS,)

  def test_build_obs(
    self,
    obs_builder: ObservationBuilder,
    fresh_stash: Stash,
    funded_coins: CoinGenerator,
    slot_scheduler: SlotScheduler,
    target_provider: TargetProvider,
  ):
    obs = obs_builder.build_obs(
      fresh_stash, funded_coins, slot_scheduler, target_provider, tick=100
    )
    assert obs["stash"].shape == (NUM_ITEMS,)
    assert obs["coins"].shape == (1,)
    assert obs["slots"].shape == (NUM_CRAFTABLE, 3)
    assert obs["time_fraction"].shape == (1,)
    assert obs["current_tick"].shape == (1,)
    assert obs["current_tick"][0] == 100
    assert obs["coins"][0] == 100_000

  def test_obs_in_space(
    self,
    obs_builder: ObservationBuilder,
    fresh_stash: Stash,
    funded_coins: CoinGenerator,
    slot_scheduler: SlotScheduler,
    target_provider: TargetProvider,
  ):
    space = obs_builder.build_space()
    obs = obs_builder.build_obs(
      fresh_stash, funded_coins, slot_scheduler, target_provider, tick=0
    )
    assert space.contains(obs)

  def test_time_fraction(
    self, obs_builder, fresh_stash,
    funded_coins, slot_scheduler, target_provider
  ):
    obs = obs_builder.build_obs(
      fresh_stash, funded_coins, slot_scheduler, target_provider, tick=1008
    )
    assert obs["time_fraction"][0] == pytest.approx(0.5, rel=1e-3)
