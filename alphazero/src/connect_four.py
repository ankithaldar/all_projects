#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Alpha Zero from Scratch - Connect Four'''


# imports
import numpy as np
#    script imports
# imports


# constants
# constants


# classes
class ConnectFour:
  '''Connect Four'''

  def __init__(self):
    self.row_count = 6
    self.column_count = 7
    self.action_size = self.column_count
    self.in_a_row = 4
    self.game_name = self.__class__.__name__

  def get_initial_state(self):
    return np.zeros((self.row_count, self.column_count))


  def get_next_state(self, state, action, player):
    row = np.max(np.where(state[:, action] == 0))
    state[row, action] = player
    return state


  def get_valid_moves(self, state):
    return (state[0] == 0).astype(np.uint8)


  def check_win(self, state, action):
    if action is None:
      return False

    row = np.min(np.where(state[:, action] != 0))
    column = action
    player = state[row][column]

    def count(offset_row, offset_column):
      for i in range(1, self.in_a_row):
        r = row + offset_row * i
        c = action + offset_column * i
        if (
          r < 0
          or r >= self.row_count
          or c < 0
          or c >= self.column_count
          or state[r][c] != player
        ):
          return i - 1
      return self.in_a_row - 1

    return (
      count(1, 0) >= self.in_a_row - 1 # vertical
      or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
      or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
      or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
    )


  def get_value_and_terminated(self, state, action):
    if self.check_win(state, action):
      return 1, True
    if np.sum(self.get_valid_moves(state)) == 0:
      return 0, True
    return 0, False


  def get_opponent(self, player):
    return -player


  def get_opponent_value(self, value):
    return -value


  def change_perspective(self, state, player):
    return state * player


  def get_encoded_state(self, state):
    encoded_state = np.stack(
      (state == -1, state == 0, state == 1)
    ).astype(np.float32)

    return encoded_state

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