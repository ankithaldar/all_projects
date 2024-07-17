#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Alpha Zero from Scratch - Tic Tac Toe'''


# imports
import numpy as np
#    script imports
# imports


# constants
# constants


# classes
class TicTacToe:
  '''TicTacToe

  This class represents a Tic-Tac-Toe game environment. It provides methods for
  initializing the game state, making moves, checking for win conditions, and
  determining game termination.

  Attributes:
    row_count (int): Number of rows in the game board (default: 3).
    column_count (int): Number of columns in the game board (default: 3).
    action_size (int): Total number of possible actions (row_count * column_count).
    game_name (str): game_name (str): Name of the game.

  Methods:
    __init__(): Initializes a new TicTacToe game instance with default parameters.
    get_initial_state(): Returns an initial game state represented as a NumPy
      array of shape (row_count, column_count) filled with zeros.
    get_next_state(state, action, player): Updates the game state by placing the
      given player's mark at the specified action location.
    get_valid_moves(state): Returns a NumPy array indicating valid moves
      (empty cells) in the given state.
    check_win(state, action): Checks if the player who made the action has won
      the game.
    get_row_column_for_action(action): Converts an action index to row and
      column coordinates.
    get_value_and_terminated(state, action): Determines the game value
      (1 for win, 0 for draw, -1 for loss) and whether the game is terminated.
    get_opponent(player): Returns the opponent's player value.
    get_opponent_value(value): Returns the opponent's value.
    change_perspective(state, player): Changes the perspective of the game
      state for the given player.
    get_encoded_state(state): Encodes the game state into a NumPy array suitable
      for machine learning models.
  '''

  def __init__(self):
    self.row_count = 3
    self.column_count = 3
    self.action_size = self.row_count * self.column_count
    self.game_name = self.__class__.__name__


  def get_initial_state(self):
    return np.zeros((self.row_count, self.column_count))


  def get_next_state(self, state, action, player):
    row, column = self.get_row_column_for_action(action=action)
    state[row, column] = player
    return state


  def get_valid_moves(self, state):
    return (state.reshape(-1) == 0).astype(np.uint8)


  def check_win(self, state, action):
    if action is None:
      return False

    row, column = self.get_row_column_for_action(action=action)
    player = state[row, column]

    return (
      np.sum(state[row, :]) == player * self.column_count
      or np.sum(state[:, column]) == player * self.row_count
      or np.sum(np.diag(state)) == player * self.row_count
      or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
    )


  def get_row_column_for_action(self, action):
    return (
      action // self.column_count, # row
      action % self.column_count # column
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
