#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
import numpy as np
#  script imports
# imports


# constants
WEEK_COUNT=52
SEED=42

np.random.seed(SEED)
# constants


# classes
# classes


# functions
def check_demand_value(x):
  return 0.0 if x < 0 else x


def fill_demand_history(dh, rn_1, rn_2):
  dh.append(
    np.round(
      check_demand_value(
        np.random.normal(rn_1, rn_2)
      )
    )
  )
  return dh


def create_demand_history():
  demand_hist = []

  for _ in range(WEEK_COUNT):
    for _ in range(4):
      demand_hist = fill_demand_history(demand_hist, 3, 1.5)

    demand_hist = fill_demand_history(demand_hist, 6, 1)

    for _ in range(2):
      demand_hist = fill_demand_history(demand_hist, 12, 2)

  return demand_hist
# functions


# main
def main():
  print(create_demand_history())


# if main script
if __name__ == '__main__':
  main()
