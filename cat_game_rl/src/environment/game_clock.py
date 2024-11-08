#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''GameClock'''


# imports
from abc import ABC
#    script imports
# imports


# constants
# constants


# classes

# ##############################################################################
class EventType(ABC):
  '''base class for event types'''

  def __init__(self) -> None:
    pass

  def is_one_min_crafting(self) -> bool:
    pass


class MiniEvent(EventType):
  '''Mini Event'''

  event_count = 0

  def __init__(self, game_clock, event_start, runs_for_days=1) -> None:
    self.game_clock = game_clock
    self.event_start = event_start
    self.event_end = self.event_start + (24 * 60) * runs_for_days - 1
    self.runs_for_days = runs_for_days
    MiniEvent.event_count += 1
    self.event_name = f'{__class__.__name__}_{MiniEvent.event_count}'

  def is_one_min_crafting(self) -> bool:
    return False


class EventBasket(EventType):
  '''Event Basket'''

  event_count = 0

  def __init__(self, game_clock, event_start, runs_for_days=7) -> None:
    self.game_clock = game_clock
    self.event_start = event_start
    self.event_end = self.event_start + (24 * 60) * runs_for_days - 1
    self.runs_for_days = runs_for_days
    EventBasket.event_count += 1
    self.event_name = f'{__class__.__name__}_{EventBasket.event_count}'

  def is_one_min_crafting(self) -> bool:
    if self.game_clock.time >= self.event_start and self.game_clock.time <= self.event_end:
      # merge all slots into a single list
      one_min_craft_slots = [
        # starts at 0000hrs UTC
        *[self.event_start + (24 * 60) * i for i in range(self.runs_for_days)],
        # starts at 0600hrs UTC
        *[self.event_start + (6 * 60) + (24 * 60) * i for i in range(self.runs_for_days)],
        # starts at 1800hrs UTC
        *[self.event_start + (18 * 60) + (24 * 60) * i for i in range(self.runs_for_days)]
      ]
      one_min_craft_slots.sort()

      for slots in one_min_craft_slots:
        if self.game_clock.time >= slots and self.game_clock.time <= slots + 30 - 1:
          return True

    return False

# ##############################################################################


class GameClock:
  '''Clock module to track game time and activity'''

  def __init__(self, run_time):
    self.run_time = run_time
    self.__time = 1

    # 25 day event order

    self.event_order = {
      f'{event.event_name}': event for event in [
        MiniEvent(game_clock=self, event_start=1),
        EventBasket(game_clock=self, event_start=1441),
        MiniEvent(game_clock=self, event_start=11521),
        EventBasket(game_clock=self, event_start=12961),
        MiniEvent(game_clock=self, event_start=23041),
        EventBasket(game_clock=self, event_start=24481),
        MiniEvent(game_clock=self, event_start=34561),
      ]
    }

  @property
  def current_event_name(self) -> str:
    for _, event in self.event_order.items():
      if self.time >= event.event_start and self.time <= event.event_end:
        return event.event_name

    return 'No current event'

  @property
  def current_running_event(self) -> EventType:
    return self.event_order.get(self.current_event_name)

  @property
  def time(self) -> int:
    return self.__time

  @property
  def flag_one_min_crafting(self) -> bool:
    return self.current_running_event.is_one_min_crafting()


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
  gc = GameClock(run_time=25 * 24 * 60)
  while gc.time < gc.run_time:
    print(gc.time, gc.flag_one_min_crafting, gc.current_event_name)
    gc.tick()


# if main script
if __name__ == '__main__':
  main()
