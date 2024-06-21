#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
#    script imports
from src.tictactoe import TicTacToe
from src.mcts import MCTS
from src.model import ResNet
from tests.test_tictactoe import test_tictactoe
# imports


# constants
# constants


# classes
# classes


# functions
def run_tests():
  tictactoe = TicTacToe()
  model = ResNet(game=tictactoe, num_res_blocks=4, num_hidden=64)
  mcts = MCTS(
    game=tictactoe,
    args={
      'C': 2,
      'num_searches': 1000
    },
    model=model
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
