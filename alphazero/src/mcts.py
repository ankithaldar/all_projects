#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Alpha Zero from Scratch - monte carlo tree search'''


# imports
import numpy as np
import math
import tensorflow as tf
#    script imports
# imports


# constants
# constants


# classes
class Node:
  '''Node for Monte Carlo Tree Search (MCTS)

  This class represents a node in the MCTS tree. Each node holds information
  about the game state, the action that led to this state, the visit count
  (number of times this node has been visited), the value sum (sum of values
  obtained from simulations starting at this node), and references to its parent
  node and child nodes.

  Attributes:
    game: The game environment.
    args: A dictionary containing MCTS hyperparameters.
    state: The game state represented by this node.
    parent: The parent node of this node.
    action_taken: The action that led to this state (None for the root node).
    prior: The prior probability of this node (used during exploration).
    visit_count: The number of times this node has been visited.
    value_sum: The sum of values obtained from simulations starting at this node.
    children: A list of child nodes.
  '''

  def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
    self.game = game
    self.args = args
    self.state = state
    self.parent = parent
    self.action_taken = action_taken
    self.prior = prior

    self.children = []

    self.visit_count = visit_count
    self.value_sum = 0


  def is_fully_expanded(self):
    '''Checks if all valid actions can be taken from this state.

    Returns:
      True if all valid actions can be taken from this state, False otherwise.
    '''
    return len(self.children) > 0


  def select(self):
    '''Selects the best child node based on the Upper Confidence Bound (UCB) formula.

    Returns:
      The child node with the highest UCB value.
    '''
    best_child = None
    best_ucb = -np.inf

    for child in self.children:
      ucb = self.get_ucb(child)
      if ucb > best_ucb:
        best_child = child
        best_ucb = ucb

    return best_child


  def get_ucb(self, child):
    '''Calculates the UCB value for a child node.

    Args:
      child: The child node for which to calculate UCB.

    Returns:
      The UCB value of the child node.
    '''

    if child.visit_count == 0:
      q_value = 0
    else:
      q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
    return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior


  def expand(self, policy):
    '''Expands the node by creating child nodes for each valid action.

    Args:
      policy: A probability distribution over possible actions.
    '''
    for action, prob in enumerate(policy):
      if prob > 0:
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player=-1)

        child = Node(self.game, self.args, child_state, self, action, prob)
        self.children.append(child)


  def backpropagate(self, value):
    '''Propagates the value obtained from a simulation back up the tree.

    Args:
      value: The value obtained from a simulation.
    '''
    self.value_sum += value
    self.visit_count += 1

    value = self.game.get_opponent_value(value)
    if self.parent is not None:
      self.parent.backpropagate(value)


class MCTS:
  '''Monte Carlo Tree Search (MCTS)

  This class implements the MCTS algorithm for playing games. MCTS uses a tree
  structure to explore the game state space and select the most promising action.

  Attributes:
    game: The game environment.
    args: A dictionary containing MCTS hyperparameters.
    model: A TensorFlow model used to evaluate game states.
  '''

  def __init__(self, game, args, model):
    self.game = game
    self.args = args
    self.model = model


  def search(self, state):
    root = Node(self.game, self.args, state, visit_count=1)

    policy, _ = self.model(
      tf.expand_dims(input=self.game.get_encoded_state(state), axis=0)
    )
    policy = tf.squeeze(tf.nn.softmax(policy), axis=None).numpy()
    policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']]* self.game.action_size)

    valid_moves = self.game.get_valid_moves(state)
    policy *= valid_moves
    policy /= np.sum(policy)
    root.expand(policy)

    for _ in range(self.args['num_searches']):
      node = root

      while node.is_fully_expanded():
        node = node.select()

      value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
      value = self.game.get_opponent_value(value)

      if not is_terminal:
        policy, value = self.model(
          tf.expand_dims(input=self.game.get_encoded_state(node.state), axis=0)
        )

        policy = tf.squeeze(tf.nn.softmax(policy), axis=None).numpy()
        valid_moves = self.game.get_valid_moves(node.state)
        policy *= valid_moves
        policy /= np.sum(policy)
        value = tf.squeeze(value).numpy()
        node.expand(policy)

      node.backpropagate(value)

    action_probs = np.zeros(self.game.action_size)
    for child in root.children:
      action_probs[child.action_taken] = child.visit_count
    action_probs /= np.sum(action_probs)
    return action_probs


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
