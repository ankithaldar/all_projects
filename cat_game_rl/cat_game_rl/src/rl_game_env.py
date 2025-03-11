#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This class wraps the game environment and provides an interface for the RL agent.
It keeps the RL-specific logic separate from the game logic.
'''


# imports
import torch
import torch.optim as optim
import random
from dataclasses import dataclass
import numpy as np
#    script imports
from env_game_world import worldbuilder_create
# from simulation_tracker import SimulationTracker
from rl_state_calculator import StateCalculator
from rl_action_calculator import ActionCalculator
from rl_reward_calculator import RewardCalculator
from model import QNetwork
# imports


# constants
EPS = 1
EPS_DECAY = 0.001
EPS_MIN = 0.01
GAMMA = 0.99
MAX_ITEM_BATCH_SIZE = 20

# RL SIMULATION VARIABLES
NUM_EPISODES = 1000
LEARN_BATCH_SIZE = 32             # minibatch size
LEARN_RATE = 1e-4                 # learning rate

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# constants


# classes
@dataclass
class Experience:
  '''Experience tuple for agent'''
  def __init__(self, state, action, reward, next_state, done):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done


class RLGameEnv:
  '''
  This class wraps the game environment and provides an interface for the RL agent.
  It keeps the RL-specific logic separate from the game logic.
  '''

  def __init__(self):
    self.reference_world = worldbuilder_create()
    # self.simulation_tracker = SimulationTracker(self.reference_world)
    self.state_calculator = StateCalculator(self.reference_world)
    self.action_calculator = ActionCalculator(self.reference_world, MAX_ITEM_BATCH_SIZE)
    self.reward_calculator = RewardCalculator(self.reference_world)

    self.state_tensor = self.state_calculator.state_to_torch_tensor()

    self.agent = QNetwork(
      state_size=len(self.state_calculator.world_to_state().keys()),
      action_size=self.action_calculator.action_size
    )
    self.agent_optimizer = optim.Adam(self.agent.parameters(), lr=LEARN_RATE)


    self.epsilon = EPS
    self.epsilon_decay = EPS_DECAY
    self.epsilon_min = EPS_MIN


  def reset(self):
    '''Resets the game environment and returns the initial state.'''
    self.reference_world = worldbuilder_create()
    # self.simulation_tracker = SimulationTracker(self.reference_world)
    self.state_calculator.world = self.reference_world

    self.state_tensor = self.state_calculator.state_to_torch_tensor()

    self.epsilon = EPS
    self.epsilon_decay = EPS_DECAY
    self.epsilon_min = EPS_MIN


  def step(self):
    action = self.agent_select_acion()

    self.state = self.state_calculator.world_to_state()

    terminate = self.reference_world.step(
      action_batch_size=self.action_calculator.action_to_dict(
        action=action
      )
    )
    return (
      # state
      self.state_calculator.state_to_torch_tensor(),
      # action
      action,
      # reward
      self.reward_calculator.calculate_total_rewards(
        prev_state=self.state,
        current_state=self.state_calculator.world_to_state()
      ),
      # terminate condition
      terminate
    )

  def agent_select_acion(self):
    '''Selects an action based on the current state.'''
    if random.random() > EPS:
      with torch.no_grad():
        return self.agent(self.state)
    else:
      # Explore: return a random action (normalized batch sizes)
      return torch.rand(self.action_calculator.action_size)

  def agent_optimize_model(self, memory):
    if len(memory) < LEARN_BATCH_SIZE:
      return

    experiences = random.sample(memory, LEARN_BATCH_SIZE)

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

    # error
    q_values = self.agent(states).gather(1, actions).squeeze() #action_batch.unsqueeze(1)

    # Calculate the expected Q-values for the next states
    next_q_values = self.agent(next_states)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1-dones)

    # Calculate the loss (Mean Squared Error)
    # error
    loss = torch.nn.MSELoss()(q_values, expected_q_values.squeeze())

    # Optimize the Q-network
    self.agent_optimizer.zero_grad()
    loss.backward()
    self.agent_optimizer.step()

    # Update epsilon for exploration
    if self.epsilon > self.epsilon_min:
      self.epsilon -= self.epsilon_decay


# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  rl_env = RLGameEnv()
  memory=[]

  for episode in range(NUM_EPISODES):
    rl_env.reset()
    state = rl_env.state_tensor
    done = False
    total_reward = 0

    while not done:
      next_state, action, reward, done = rl_env.step()
      memory.append(
        Experience(
          state=state,
          action=action,
          reward=reward,
          next_state=next_state,
          done=done
        )
      )

      rl_env.agent_optimize_model(memory)

      total_reward += reward
      state = next_state

      if rl_env.reference_world.clock.current_time%10000 == 0:
        print(total_reward)

    print(f'Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {rl_env.epsilon}')

# if main script
if __name__ == '__main__':
  main()
