#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the scripted control policy at the World level.'''

from world_of_supply.facility import RetailerCell
from world_of_supply.policies import ScriptedSupplyChainPolicy
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.world import Control


def test_policy_controls_every_facility():
  '''Verify scripted policy controls every registered facility.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(ScenarioConfig(), seed=4)
  control = ScriptedSupplyChainPolicy(seed=4).compute_control(world)
  assert set(control.facility_controls) == set(world.facilities)


def test_scripted_chain_runs_and_earns_retail_revenue():
  '''Verify scripted chain sells retail stock within 80 ticks.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(ScenarioConfig(), seed=6)
  policy = ScriptedSupplyChainPolicy(seed=6)
  retailers = world.get_facilities(RetailerCell)
  sold_any = False
  for _ in range(80):
    world.act(policy.compute_control(world))
    sold = [r.seller.economy.total_units_sold > 0 for r in retailers]
    if any(sold):
      sold_any = True
      break
  assert sold_any, 'retailers should sell stock within 80 scripted ticks'


def test_control_type_is_importable_alias():
  '''Verify Control stores the facility controls mapping verbatim.

  Returns:
    None: Test completes without assertion failure.
  '''
  control = Control({})
  assert control.facility_controls == {}
