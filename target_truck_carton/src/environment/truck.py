#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Represents a truck for transporting cartons.'''


# imports
from typing import Tuple
from dataclasses import dataclass
# import numpy as np
#    script imports
from carton import Carton
# from facility import Store, Warehouse
# imports


# constants
# constants


# classes
@dataclass
class Truck:
  '''
  Represents a truck for transporting cartons.

  Attributes:
    length (int): The length of the truck.
    width (int): The width of the truck.
    height (int): The height of the truck.
    truck_id (int | str): The unique identifier for the truck.
    max_payload (int): The maximum payload capacity of the truck in kilograms.
    truck_space (np.ndarray): A 3D numpy array representing the space in the truck.
    cartons (dict): A dictionary mapping cartons to their positions in the truck.
  '''

  def __init__(self, dimensions:Tuple, truck_id:int|str, max_payload:int=10000):
    '''
    Args:
      dimensions (Tuple[int, int, int]): The dimensions of the truck (length, width, height).
      truck_id (int | str): The unique identifier for the truck.
      max_payload (int): The maximum payload capacity of the truck in kilograms. Defaults to 10000.
    '''
    self.length = dimensions[0]
    self.width = dimensions[1]
    self.height = dimensions[2]
    self.truck_id = truck_id # can take in registration number
    self.max_payload = max_payload  #kg

    # self.truck_space = np.zeros((self.length, self.width, self.height), dtype=bool)
    self.cartons = {}

  @property
  def volume(self) -> int:
    '''
    Calculates the volume of the truck.

    Returns:
      int: The volume of the truck.
    '''
    return self.length * self.width * self.height

  # def add_source_destination(self, destination: List[Store], source: Warehouse):
  #   self.source = source
  #   self.destination = destination

  def add_carton(self, carton: Carton):
    '''
    Adds a carton to the truck if there is enough space and weight capacity.

    Args:
      carton (Carton): The carton to be added.
    '''
    if self.can_add_carton(carton):
      next_available_position = self.find_available_volumetric_position()
      self.cartons[carton] = next_available_position
      carton.load_in_truck(self)
    pass

  def remove_carton(self, carton: Carton):
    # remove truck object from carton
    pass

  def remaining_weight_capacity(self) -> int:
    '''
    Calculates the remaining weight capacity of the truck.

    Returns:
      int: The remaining weight capacity.
    '''
    return self.max_payload - sum([carton.weight for _, carton in self.cartons.items()])

  def remaining_volume_capacity(self) -> int:
    '''
    Calculates the remaining volume capacity of the truck.

    Returns:
      int: The remaining volume capacity.
    '''
    return (self.volume) - sum([carton.volume for _, carton in self.cartons.items()])

  def can_add_carton(self, carton) -> bool:
    '''
    Calculates the remaining volume capacity of the truck.

    Returns:
      int: The remaining volume capacity.
    '''
    if self.remaining_weight_capacity() >= carton.weight and self.remaining_volume_capacity() >= carton.volume:
      return True
    return False

  def find_available_volumetric_position(self) -> Tuple[int, int, int]:
    '''
    Finds the next available volumetric position in the truck.

    Returns:
      Tuple[int, int, int]: The coordinates of the next available position.
    '''
    return (0, 0, 0)

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
