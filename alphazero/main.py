#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
#    script imports
from src.tictactoe import TicTacToe
from src.mcts import MCTS
from src.model import ResNet
from src.alphazero import AlphaZero
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


def test_self_play():
  tictactoe = TicTacToe()

  model = ResNet(tictactoe, 4, 64)

  args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 10,
    'num_epochs': 4,
    'batch_size': 64
  }

  alpha_zero = AlphaZero(model, tictactoe, args)
  alpha_zero.learn()
# functions


# main
def main():
  # run_tests()
  test_self_play()


# if main script
if __name__ == '__main__':
  main()
