#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Alpha Zero from Scratch - Self Play'''


# imports
import numpy as np
import tensorflow as tf
#    script imports
from src.mcts import MCTS
# imports


# constants
# constants

# https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras


# classes
class AlphaZero:
  '''Self Play'''

  def __init__(self, model, optimizer, game, args):
    self.model = model
    self.optimizer = optimizer
    self.game = game
    self.args = args
    self.mcts = MCTS(game, args, model)


  def self_play(self):
    memory = []
    player = 1
    state = self.game.get_initial_state()

    while True:
      neutral_state = self.game.change_perspective(state, player)
      action_probs = self.mcts.search(neutral_state)

      memory.append((neutral_state, action_probs, player))

      action = np.random.choice(self.game.action_size, p=action_probs)

      state = self.game.get_next_state(state, action, player)

      value, is_terminal = self.game.get_value_and_terminated(state, action)

      if is_terminal:
        return_memory = []
        for hist_neutral_state, hist_action_probs, hist_player in memory:
          hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
          return_memory.append((
            self.game.get_encoded_state(hist_neutral_state),
            hist_action_probs,
            hist_outcome
          ))
        return return_memory

      player = self.game.get_opponent(player)


  def train(self, memory):
    pass


  def learn(self):
    for iteration in range(self.args['num_iterations']):
      memory = []

      for _ in range(self.args['num_selfPlay_iterations']):
        memory += self.self_play()

      for _ in range(self.args['num_epochs']):
        self.train(memory)

      tf.keras.models.save_model(self.model, f'model_{self.game.game_name}_{iteration}.h5')
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
