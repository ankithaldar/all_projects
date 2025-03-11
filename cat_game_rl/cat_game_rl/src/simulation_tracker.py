#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
from uuid import uuid4
#    script imports
from db_feeder import DBFeeder
# imports


# constants
# constants


# classes
class SimulationTracker:
  '''Tracke all simulation metrics'''

  def __init__(self, game_world):
    self.game_world = game_world
    self.db_feeder = DBFeeder()
    self.uuid = uuid4()

# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
