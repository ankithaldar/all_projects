#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Root environment entity. Manages all ItemFacility instances, GameClock, GameEconomy.
Implements the step() function that the RL agent calls. Defines the state and reward calculation.
'''


# imports
from collections import Counter
#    script imports
from env_game_clock import GameClock
from env_item_facility import ItemFacility
from env_game_economy import GameEconomy
from env_bill_of_materials import BillOfMaterials
from env_utils import (
  BASE_ITEMS,
  TIME_LIMIT,
  ITEM_DEFAULTS,
  INIT_ECONOMY,
  TARGET_ITEM_COUNTS,
  yaml_reader,
  parse_materials,
  get_total_counts
)
# imports


# constants
# constants


# classes
class GameWorld:
  '''
  Root environment entity. Manages all ItemFacility instances, GameClock, GameEconomy.
  Implements the step() function that the RL agent calls. Defines the state and reward calculation.
  '''

  def __init__(self):
    self.clock = GameClock(time_limit=TIME_LIMIT)
    self.economy = GameEconomy()
    self.item_facilities = {}

  def check_presents(self):
    if self.clock.current_time % 5 == 0:
      self.economy.update_coins(gained_coins=210)

  def check_terminate_condition(self):
    return all(
      facility.total_crafted_count >= facility.total_target_count
      for _, facility in self.item_facilities.items()
    )

  def step(self, action_batch_size:dict):
    if not self.clock.is_time_limit_reached():
      self.check_presents()

      for item, facility in self.item_facilities.items():
        facility.step(action_batch_size[item])

      self.clock.tick()

      return self.check_terminate_condition()
    else:
      return self.clock.is_time_limit_reached()

# classes


# functions
def load_init_economy(world: GameWorld) -> GameWorld:
  init_economy_dict = yaml_reader(INIT_ECONOMY)
  # load stating coins
  world.economy.coins = init_economy_dict['coins']
  # load starting materials
  world.economy.items_in_stash = Counter(
    parse_materials(init_economy_dict['materials'])
  )

  return world


def load_item_facilities(world: GameWorld) -> GameWorld:
  # fix fanal world targets for crafting
  targets = parse_materials(
    yaml_reader(TARGET_ITEM_COUNTS)['materials']
  )

  items_full_targets = get_total_counts(targets)
  items_crafted = get_total_counts(world.economy.items_in_stash)

  # adding item facilities
  item_defaults = yaml_reader(ITEM_DEFAULTS)

  for item in item_defaults['materials']:
    world.item_facilities[item['item_name']] = ItemFacility(
      name=item['item_name'],
      bom=BillOfMaterials(
        inputs=Counter(item['req_unit_raw']),
        init_cost=item['init_cost'],
        req_time=item['time']
      ),
      sources = [
        world.item_facilities[i] for i in item['req_unit_raw']
        if i not in BASE_ITEMS
      ],
      target_count=targets.get(item['item_name'], 0),
      total_target_count=items_full_targets[item['item_name']],
      total_crafted_count=items_crafted[item['item_name']],
      game_economy=world.economy,
      clock=world.clock
    )
  return world



def worldbuilder_create():
  return load_item_facilities(
    world=load_init_economy(
      world=GameWorld()
    )
  )
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
