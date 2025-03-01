#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Clock module to track game time and activity'''


# imports
#    script imports
# imports


# constants
# constants


# classes
class GameClock:
  '''Clock module to track game time and activity'''

  def __init__(self, run_time):
    self.run_time = run_time
    self.__time = 1

  @property
  def time(self) -> int:
    return self.__time


  def tick(self) -> None:
    self.__time += 1

  def is_within_runtime(self) -> bool:
    if self.time <= self.run_time:
      return True

    return False
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
