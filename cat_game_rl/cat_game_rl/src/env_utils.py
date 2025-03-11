#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Utility functions & constants'''


# imports
from collections import Counter
from pathlib import Path
from functools import reduce
import yaml

try:
  from yaml import CLoader
except ImportError:
  from yaml import Loader as CLoader
#    script imports
# imports


# constants
# run time of game episodes
RUN_DAYS = 25
TIME_LIMIT = RUN_DAYS * 24 * 60

# YAML initialization file paths
ENV_FOLDER = Path(__file__).parent.resolve()
INIT_ECONOMY = ENV_FOLDER / 'loaders'/'init_game_economy.yml'
ITEM_DEFAULTS = ENV_FOLDER / 'loaders'/'item_default_config.yml'
TARGET_ITEM_COUNTS = ENV_FOLDER / 'loaders'/'crafting_target.yml'

# Others
BASE_ITEMS = ['tree', 'cotton', 'rock', 'quartz']
ITEM_COUNTS = Counter()
# constants


# classes
# classes


# functions
def yaml_reader(file_path: Path) -> dict:
  '''Read YAML Files'''
  if file_path.exists():
    with file_path.open('r') as f:
      return yaml.load(f, Loader=CLoader)
  else:
    print(f'Config file not found at {file_path.absolute()}.')
    raise FileNotFoundError



def parse_materials(materials: list) -> dict:
  '''parse dict with materials'''
  # return dict(ChainMap(*materials))
  return reduce(lambda a, b: {**a, **b}, materials)



def get_raw_material_count(item, counts, required_item_raw):
  '''recursive function to get total materials'''
  for prev_item, prev_qty in required_item_raw[item].items():
    ITEM_COUNTS[prev_item] += counts * prev_qty
    if prev_item not in BASE_ITEMS:
      get_raw_material_count(prev_item, counts * prev_qty, required_item_raw)



def get_total_counts(target_items):
  # target_items = {'artifact': 1}
  item_defaults = yaml_reader(ITEM_DEFAULTS)

  required_item_raw = {}
  for each in item_defaults['materials']:
    required_item_raw[each['item_name']] = each['req_unit_raw']
    ITEM_COUNTS[each['item_name']] = 0

  # get total items to be crafted
  for item, counts in target_items.items():
    if counts > 0:
      ITEM_COUNTS[item] += counts
      if item not in BASE_ITEMS:
        get_raw_material_count(item, counts, required_item_raw)

  return Counter(ITEM_COUNTS)
#functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
