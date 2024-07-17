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
# constants

# https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras


# classes
class AlphaZero:
  '''AlphaZero Agent for Self-Play and Reinforcement Learning

  This class implements the AlphaZero algorithm for playing games using
  self-play and reinforcement learning. AlphaZero combines Monte Carlo Tree
  Search (MCTS) with a neural network model to achieve high performance in
  various games.

  Attributes:
    model: A TensorFlow model used for policy and value estimation.
    optimizer: An optimizer used for training the model.
    game: The game environment (TicTacToe, Go, etc.).
    args: A dictionary containing hyperparameters for AlphaZero.
    mcts: An instance of the MCTS class for searching the game tree.
  '''

  def __init__(self, model, game, args):
    self.model = model
    self.optimizer = self.define_optimizer()
    self.game = game
    self.args = args
    self.mcts = MCTS(game, args, model)


  def define_optimizer(self):
    '''Defines the optimizer used for training the model.

    Returns:
      A TensorFlow optimizer (default: Adam with learning rate 1e-3 and weight decay 1e-4).
    '''
    return tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-4)


  def send_to_discord(self, message):
    '''Sends a message to a Discord webhook (for logging purposes).

    Args:
      message: The message content to send.

    Raises:
      KeyError: If the environment variable 'DISCORD_WEBHOOK' is not set.
    '''
    payload = {
      'content': message,
    }
    try:
      if os.environ['DISCORD_WEBHOOK']:
        requests.post(os.environ['DISCORD_WEBHOOK'], json=payload)
    except KeyError:
      pass


  def self_play(self):
    '''Performs a single self-play game simulation.

    Returns:
      A list of tuples containing encoded game states, action probabilities, and player outcomes for training the model.
    '''
    memory = []
    player = 1
    state = self.game.get_initial_state()

    while True:
      neutral_state = self.game.change_perspective(state, player)
      action_probs = self.mcts.search(neutral_state)

      memory.append((neutral_state, action_probs, player))

      temperature_action_probs = action_probs ** (1 / self.args['temperature'])
      temperature_action_probs = temperature_action_probs / np.sum(temperature_action_probs)
      action = np.random.choice(self.game.action_size, p=temperature_action_probs) # change to p=temperature_action_probs

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
    '''Trains the model using a minibatch of self-play data.

    Args:
      memory: A list of tuples containing encoded game states, action probabilities, and player outcomes.
    '''
    random.shuffle(memory)
    for batch_idx in range(0, len(memory), self.args['batch_size']):
      sample = memory[batch_idx:min(len(memory) - 1, batch_idx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
      state, policy_targets, value_targets = zip(*sample)

      state, policy_targets, value_targets = tf.convert_to_tensor(state), tf.convert_to_tensor(policy_targets), tf.convert_to_tensor(value_targets)

      with tf.GradientTape() as tape:
        out_policy, out_value = self.model(state)

        # Compute the loss value for this minibatch.
        policy_loss = tf.keras.losses.categorical_crossentropy(policy_targets, out_policy)
        value_loss = tf.keras.losses.MSE(value_targets, out_value)
        # loss = policy_loss + value_loss

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient([policy_loss, value_loss], self.model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


  def learn(self):
    '''Main training loop for the AlphaZero agent.

    This function iteratively performs self-play, trains the model, and saves the model checkpoints.
    '''
    for iteration in range(self.args['num_iterations']):
      memory = []

      # self.model.eval()
      for _ in range(self.args['num_selfPlay_iterations']):
        memory += self.self_play()


      # self.model.train()
      for num_epoch in range(self.args['num_epochs']):
        self.train(memory)
        self.send_to_discord(f'Iteration: {iteration}, Epoch: {num_epoch}')

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
