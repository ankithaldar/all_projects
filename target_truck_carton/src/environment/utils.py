#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
import random
#    script imports
from truck import Truck
from carton import Carton
# imports


# constants
UNITS = 'mm'
# constants


# classes
# classes


# functions
def generate_cartons(num_cartons: int = 1):
  min_max_length = (200, 400)
  min_max_width = (170, 400)
  min_max_height = (250, 400)

  cartons = []

  for _ in range(num_cartons):
    carton_id = f'CARTON_{len(cartons) + 1}'
    dimensions = (random.randint(*min_max_length), random.randint(*min_max_width), random.randint(*min_max_height))
    cartons.append(Carton(dimensions, carton_id, weight=random.randint(35, 50)))

  return cartons


def fetch_cartons(num_cartons: int = 1):
  # can be used to fetch from sql database
  return generate_cartons(num_cartons)



def generate_trucks(num_trucks: int = 1):
  min_max_length = (5898, 12031)
  min_max_width = (2352, 2352)
  min_max_height = (2394, 2394)
  min_max_payload = (24000, 30480)

  trucks = []

  for _ in range(num_trucks):
    truck_id = f'TRUCK_{len(trucks) + 1}'
    dimensions = (random.randint(*min_max_length), random.randint(*min_max_width), random.randint(*min_max_height))
    trucks.append(Truck(dimensions, truck_id, max_payload=random.randint(*min_max_payload)))

  return trucks


def fetch_trucks(num_trucks: int = 1):
  # can be used to fetch from sql database
  return generate_trucks(num_trucks)
# functions


# main
def main():
  trucks = fetch_trucks(1)
  cartons = fetch_cartons(10)

  while sum([truck.max_payload for truck in trucks]) >= sum([carton.weight for carton in cartons]):
    cartons = [*cartons, *fetch_cartons(10)]

  print(len(cartons))

  pass


# if main script
if __name__ == '__main__':
  main()
