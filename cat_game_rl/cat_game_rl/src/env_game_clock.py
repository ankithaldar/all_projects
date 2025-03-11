#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Manages the in-game time.'''


# imports
#    script imports
# imports


# constants
# constants


# classes
class GameClock:
  '''Manages the in-game time.'''

  def __init__(self, time_limit):
    self.__current_time = 0
    self.time_limit = time_limit
    self.flag_one_min_crafting = False

  @property
  def current_time(self):
    '''Represents the current game time in minutes. Initialized to 0 at the start of each episode.'''
    return self.__current_time

  def tick(self):
    ''' Increments the game time by one time step (e.g., 1 minute).'''
    self.__current_time += 1

  def get_time(self):
    '''Returns the current game time.'''
    return self.current_time

  def is_time_limit_reached(self):
    '''Checks if the time limit has been reached.'''
    return True if self.current_time >= self.time_limit else False

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
