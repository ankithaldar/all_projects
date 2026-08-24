#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the scripted control policy at the World level.'''

from world_of_supply.policies import ScriptedSupplyChainPolicy
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.world import Control


def test_policy_controls_every_facility():
  world = WorldBuilder.build(ScenarioConfig(), seed=4)
  control = ScriptedSupplyChainPolicy(seed=4).compute_control(world)
  assert set(control.facility_controls) == set(world.facilities)


def test_scripted_chain_runs_and_earns_retail_revenue():
  world = WorldBuilder.build(ScenarioConfig(), seed=6)
  policy = ScriptedSupplyChainPolicy(seed=6)

  retailers = world.get_facilities(type(next(iter(
      (f for f in world.facilities.values() if f.__class__.__name__ == 'RetailerCell')
  ))))
  sold_any = False
  for _ in range(80):
    world.act(policy.compute_control(world))
    if any(r.seller.economy.total_units_sold > 0 for r in retailers):
      sold_any = True
      break

  assert sold_any, 'retailers should sell stock within 80 scripted ticks'


def test_control_type_is_importable_alias():
  control = Control({})
  assert control.facility_controls == {}
