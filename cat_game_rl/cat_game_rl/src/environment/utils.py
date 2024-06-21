#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Utility functions & constants'''



# imports
import logging
from collections import Counter
from datetime import datetime
from functools import reduce
from pathlib import Path

import yaml

try:
  from yaml import CLoader
except ImportError:
  from yaml import Loader as CLoader
#    script imports
# imports


# constants
DEBUG = False

if DEBUG:
  # define logger
  LOG_PATH = Path.cwd()/'logs'
  # define logger folder
  for path in [LOG_PATH]:
    if not path.exists():
      path.mkdir(parents=True)

  # define logger file
  logging.basicConfig(
    filename=LOG_PATH/f'log_cgrl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.DEBUG
  )
#

BASE_ITEMS = ['tree', 'cotton', 'rock', 'quartz']
# GAME_TYPE = '1MC' # Normal
# GAME_TYPE = 'Normal' # Normal
ITEM_COUNTS = Counter()

ENV_FOLDER = Path(__file__).parent.resolve()

# initialization files
INIT_ECONOMY = ENV_FOLDER / 'loaders'/'init_game_economy.yml'
ITEM_DEFAULTS = ENV_FOLDER / 'loaders'/'item_default_config.yml'
TARGET_ITEM_COUNTS = ENV_FOLDER / 'loaders'/'crafting_target.yml'

RUN_DAYS = 25
RUN_TIME = RUN_DAYS * 24 * 60

# DEFAULT BATCH SIZE
BATCH_SIZE = {
  'string': 10,
  'wood': 10,
  'metal': 12,
  'ribbon': 5,
  'needles': 3,
  'sparkles': 2,
  'bronze': 3,
  'silver': 2,
  'gold': 1,
  'amethyst': 3,
  'pendant': 1,
  'necklace': 1,
  'orb': 2,
  'water': 2,
  'fire': 1,
  'waterstone': 1,
  'firestone': 1,
  'elementstone': 1,
  'artifact': 1
}
# constants


# classes
# classes


# functions
# logging function
def write_to_log(msg:str='', log_level:str='info'):
  log_type = {
    'critical': 50,
    'error': 40,
    'warning': 30,
    'info': 20,
    'debug': 10,
    'notset': 0
  }
  logging.log(
    level=log_type.get(log_level.lower(), 20),
    msg=f'{datetime.now().strftime("%b %d, %Y %H:%M:%S")} | {msg}'
  )


# Read YAML Files
def yaml_reader(file_path: Path) -> dict:
  if file_path.exists():
    with file_path.open('r') as f:
      return yaml.load(f, Loader=CLoader)
  else:
    print(f'Config file not found at {file_path.absolute()}.')
    raise FileNotFoundError


# parse dict with materials
def parse_materials(materials: list) -> dict:
  # return dict(ChainMap(*materials))
  return reduce(lambda a, b: {**a, **b}, materials)


# recursive function to get total materials
def get_raw_material_count(item, counts, required_item_raw):
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

# functions


# main

def main():
  items = {
    'string': 1,
    'wood': 1,
    'metal': 1,
    'ribbon': 1,
    'needles': 1,
    'sparkles': 1,
    'bronze': 1,
    'silver': 1,
    'gold': 1,
    'amethyst': 1,
    'pendant': 1,
    'necklace': 1,
    'orb': 1,
    'water': 1,
    'fire': 1,
    'elementstone': 1,
    'firestone': 1,
    'waterstone': 1,
    'artifact': 1
  }
  for item, qty in items.items():
    global ITEM_COUNTS
    ITEM_COUNTS = Counter()
    print(item, get_total_counts({item: qty}))
    print('\n\n\n\n\n')




# if main script
if __name__ == '__main__':
  main()
