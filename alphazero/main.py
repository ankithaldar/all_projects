#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
#    script imports
from src.tictactoe import TicTacToe
from src.mcts import MCTS
from tests.test_tictactoe import test_tictactoe
# imports


# constants
# constants


# classes
# classes


# functions
def run_tests():
  tictactoe = TicTacToe()
  mcts = MCTS(
    game=tictactoe,
    args={
      'C': 1.41,
      'num_searches': 1000
    }
  )

  test_tictactoe(
    game=tictactoe,
    mcts=mcts
  )
# functions


# main
def main():
  run_tests()


# if main script
if __name__ == '__main__':
  main()
