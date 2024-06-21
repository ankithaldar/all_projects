#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
#    script imports
from src.tictactoe import TicTacToe
from tests.test_tictactoe import test_tictactoe
# imports


# constants
# constants


# classes
# classes


# functions
def run_tests():
  test_tictactoe(game=TicTacToe())
# functions


# main
def main():
  run_tests()


# if main script
if __name__ == '__main__':
  main()
