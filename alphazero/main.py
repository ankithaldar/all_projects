#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
#    script imports
from src.tictactoe import TicTacToe
from src.connect_four import ConnectFour
# from src.mcts import MCTS
from src.model import ResNet
from src.alphazero import AlphaZero
# imports


# constants
# constants


# classes
# classes


# functions
def test_tictactoe_self_play():
  tictactoe = TicTacToe()

  model = ResNet(tictactoe, 4, 64)

  args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
  }

  alpha_zero = AlphaZero(model, tictactoe, args)
  alpha_zero.learn()



def test_connectfour_self_play():
  connect_four = ConnectFour()

  model = ResNet(connect_four, 9, 128)

  args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
  }

  alpha_zero = AlphaZero(model, connect_four, args)
  alpha_zero.learn()
# functions


# main
def main():
  # run_tests()
  test_tictactoe_self_play()
  test_connectfour_self_play()


# if main script
if __name__ == '__main__':
  main()