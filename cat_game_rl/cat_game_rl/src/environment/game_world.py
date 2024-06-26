#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cat Game World'''


# imports
from collections import Counter
#    script imports
from bill_of_materials import BillOfMaterials
from game_clock import GameClock
from game_economy import GameEconomy
from item_facilities import ItemFacility
from rewards import RewardBalanceSheet
from utils import (BASE_ITEMS, BATCH_SIZE, INIT_ECONOMY,
                   ITEM_DEFAULTS, TARGET_ITEM_COUNTS, RUN_TIME, get_total_counts,
                   parse_materials, yaml_reader)
# imports


# constants
# constants


# classes
class CatGameWorld:
  '''Cat Game World'''

  def __init__(self, run_time):
    self.economy: GameEconomy = GameEconomy()
    self.clock: GameClock = GameClock(run_time)
    self.item_facilities = {}

  def check_presents(self):
    if self.clock.time % 5 == 0:
      self.economy.update_coins(gained_coins=210)

  def check_terminate_condition(self):
    return all(
      facility.total_crafted_count >= facility.total_target_count
      for _, facility in self.item_facilities.items()
    )

  @property
  def world_reward_memory(self) -> dict:
    return {
      item_name: facility.reward_memory
      for item_name, facility in self.item_facilities.items()
    }

  @property
  def total_rewards(self) -> RewardBalanceSheet:
    return sum(
      rewards.total()
      for _, facility in self.item_facilities.items()
      for rewards in facility.reward_memory
    )

  def is_one_min_crafting(self):
    # if one minute crafting is happening in the event
    if self.clock.flag_one_min_crafting:

      # if crafting is happening, set end time to one minute later
      for _, facility in self.item_facilities.items():
        if facility.is_crafting:
          facility.set_one_min_manufacturing_time()


  def act(self):
    self.check_presents()

    if self.clock.is_within_runtime():
      # move clock ahead by 1
      self.is_one_min_crafting()
      self.clock.tick()

# classes


# functions
# load initial economy conditions
def load_init_economy(world: CatGameWorld) -> CatGameWorld:
  init_economy_dict = yaml_reader(INIT_ECONOMY)

  # load stating coins
  world.economy.coins = init_economy_dict['coins']
  # load starting materials
  world.economy.items_in_stash = Counter(
    parse_materials(init_economy_dict['materials'])
  )

  return world


# load item facilities for handle crafting
def load_item_facilities(world: CatGameWorld) -> CatGameWorld:
  # fix fanal world targets for crafting
  targets = parse_materials(
    yaml_reader(TARGET_ITEM_COUNTS)['materials']
  )

  items_full_targets = get_total_counts(targets)
  items_crafted = get_total_counts(world.economy.items_in_stash)

  # adding item facilities
  item_defaults = yaml_reader(ITEM_DEFAULTS)

  for each in item_defaults['materials']:
    world.item_facilities[each['item_name']] = ItemFacility(
      name=each['item_name'],
      bom=BillOfMaterials(
        inputs=Counter(each['req_unit_raw']),
        init_cost=each['init_cost'],
        req_time=each['time']
      ),
      target_count=targets.get(each['item_name'], 0),
      total_target_count=items_full_targets[each['item_name']],
      total_crafted_count=items_crafted[each['item_name']],
      sources = [
        world.item_facilities[i] for i in each['req_unit_raw']
        if i not in BASE_ITEMS
      ],
      game_economy=world.economy,
      clock=world.clock
    )

  return world



# world Builder
def worldbuilder_create():
  return load_item_facilities(
    world=load_init_economy(
      world=CatGameWorld(RUN_TIME)
    )
  )
# functions


# main
def main():
  world = worldbuilder_create()

  while not world.check_terminate_condition():
    for item, _ in world.item_facilities.items():
      world.item_facilities[item].act(BATCH_SIZE[item])

    if world.clock.is_within_runtime():
      world.act()
    else:
      break

  print(
    world.clock.time,
    world.economy.items_in_stash,
    world.total_rewards,
    # sum(final_reward)
  )


# if main script
if __name__ == '__main__':
  main()
