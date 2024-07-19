#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
import numpy as np
# #    script imports
# from  import TicTacToe
# imports


# constants
# constants


# functions
def test_tictactoe(game, mcts):
  '''
  Interactive Tic-Tac-Toe game using MCTS for AI player.

  This function allows human player interaction with the Tic-Tac-Toe game environment
  and utilizes the provided MCTS instance for the AI player's decision-making.

  Args:
    game: A TicTacToe game environment instance.
    mcts: An MCTS instance for AI player decision-making.
  '''

  tictactoe = game
  player = 1
  state = tictactoe.get_initial_state()


  while True:
    print(state)

    if player == 1:
      valid_moves = tictactoe.get_valid_moves(state)
      print('valid_moves', [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
      action = int(input(f'{player}:'))

      if valid_moves[action] == 0:
        print('action not valid')
        continue

    else:
      neutral_state = tictactoe.change_perspective(state, player)
      mcts_probs = mcts.search(neutral_state)
      action = np.argmax(mcts_probs)

    state = tictactoe.get_next_state(state, action, player)

    value, is_terminal = tictactoe.get_value_and_terminated(state, action)

    if is_terminal:
      print(state)
      if value == 1:
        print(player, 'won')
      else:
        print('draw')
      break

    player = tictactoe.get_opponent(player)



# test_get_initial_state: Checks if the initial state is a 3x3 matrix of zeros.
# test_get_next_state: Verifies that placing a piece updates the correct position in the state.
# test_get_valid_moves: Asserts that valid moves are correctly identified.
# test_check_win: Tests different win conditions (row, column, diagonals).
# test_get_value_and_terminated: Checks for win, draw, and ongoing game scenarios.
# test_get_opponent: Ensures correct opponent determination.
# test_get_opponent_value: Verifies opponent value calculation.
# test_change_perspective: Checks if the state is correctly changed from one player's perspective to another.
# test_get_encoded_state: Asserts the shape of the encoded state.

# functions
