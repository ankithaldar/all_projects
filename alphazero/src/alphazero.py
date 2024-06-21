#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Alpha Zero from Scratch - Self Play'''


# imports
import numpy as np
import tensorflow as tf
import random
import requests
import os
#    script imports
from src.mcts import MCTS
# imports


# constants
# DISCORD_URL = 'https://discord.com/api/v9/channels/1259362351107407975/messages'
# SESSION = tls_client.Session(
#   client_identifier='chrome_120',
#   random_tls_extension_order=True
# )
# constants

# https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras


# classes
class AlphaZero:
  '''Self Play'''

  def __init__(self, model, game, args):
    self.model = model
    self.game = game
    self.args = args
    self.mcts = MCTS(game, args, model)
    self.define_optimizer()


  def define_optimizer(self):
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


  def send_to_discord(self, message):
    payload = {
      'content': message,
    }
    requests.post(os.environ['DISCORD_WEBHOOK'], json=payload)


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
    random.shuffle(memory)
    for batch_idx in range(0, len(memory), self.args['batch_size']):
      sample = memory[batch_idx:min(len(memory) - 1, batch_idx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
      state, policy_targets, value_targets = zip(*sample)

      state, policy_targets, value_targets = tf.convert_to_tensor(state), tf.convert_to_tensor(policy_targets), tf.convert_to_tensor(value_targets)

      with tf.GradientTape() as tape:
        out_policy, out_value = self.model(state)

        # Compute the loss value for this minibatch.
        policy_loss = tf.keras.losses.categorical_crossentropy(out_policy, policy_targets)
        value_loss = tf.keras.losses.MSE(out_value, value_targets)
        # loss = policy_loss + value_loss

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient([policy_loss, value_loss], self.model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


  def learn(self):
    for iteration in range(self.args['num_iterations']):
      memory = []

      for _ in range(self.args['num_selfPlay_iterations']):
        memory += self.self_play()

      for num_epoch in range(self.args['num_epochs']):
        self.train(memory)
        try:
          if os.environ['DISCORD_WEBHOOK']:
            self.send_to_discord(f'Iteration: {iteration}, Epoch: {num_epoch}')
        except KeyError:
          pass

      tf.keras.models.save_model(self.model, f'model_{self.game.game_name}_{iteration}.keras')
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
