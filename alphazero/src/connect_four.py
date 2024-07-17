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
  '''
  Represents a Connect Four game.

  Attributes:
    row_count (int): Number of rows in the board.
    column_count (int): Number of columns in the board.
    action_size (int): Number of possible actions (columns).
    in_a_row (int): Number of consecutive pieces needed to win.
    game_name (str): Name of the game.
  '''

  def __init__(self):
    self.row_count = 6
    self.column_count = 7
    self.action_size = self.column_count
    self.in_a_row = 4
    self.game_name = self.__class__.__name__

  def get_initial_state(self):
    '''
    Returns the initial state of the game board.

    Returns:
      np.ndarray: A 2D numpy array representing the empty board.
    '''
    return np.zeros((self.row_count, self.column_count))


  def get_next_state(self, state, action, player):
    '''
    Returns the new state of the game after a player makes a move.

    Args:
      state (np.ndarray): The current state of the game board.
      action (int): The column where the player wants to place a piece.
      player (int): The player making the move (1 or -1).

    Returns:
      np.ndarray: The updated game board after the move.
    '''
    row = np.max(np.where(state[:, action] == 0))
    state[row, action] = player
    return state


  def get_valid_moves(self, state):
    '''
    Returns a binary vector indicating valid moves.

    Args:
      state (np.ndarray): The current state of the game board.

    Returns:
      np.ndarray: A binary vector where 1 indicates a valid move and 0 indicates an invalid move.
    '''
    return (state[0] == 0).astype(np.uint8)


  def check_win(self, state, action):
    '''
    Checks if a player has won after making a move.

    Args:
      state (np.ndarray): The current state of the game board.
      action (int): The column where the player made the move.

    Returns:
      bool: True if a player has won, False otherwise.
    '''
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
    '''
    Determines the game outcome and whether the game is over.

    Args:
      state (np.ndarray): The current state of the game board.
      action (int): The column where the player made the move.

    Returns:
      tuple: A tuple containing the game value (1 for win, -1 for loss, 0 for draw) and a boolean indicating if the game is over.
    '''
    if self.check_win(state, action):
      return 1, True
    if np.sum(self.get_valid_moves(state)) == 0:
      return 0, True
    return 0, False


  def get_opponent(self, player):
    '''
    Returns the opponent of the given player.

    Args:
      player (int): The player to get the opponent for.

    Returns:
      int: The opponent's player ID.
    '''
    return -player


  def get_opponent_value(self, value):
    '''
    Returns the opponent's value.

    Args:
      value (int): The value to get the opponent's value for.

    Returns:
      int: The opponent's value.
    '''
    return -value


  def change_perspective(self, state, player):
    '''
    Changes the perspective of the game state.

    Args:
      state (np.ndarray): The current state of the game board.
      player (int): The player whose perspective to adopt.

    Returns:
      np.ndarray: The state from the specified player's perspective.
  '''
    return state * player


  def get_encoded_state(self, state):
    '''
    Encodes the game state for input to a neural network.

    Args:
      state (np.ndarray): The current state of the game board.

    Returns:
      np.ndarray: The encoded state.
  '''
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
