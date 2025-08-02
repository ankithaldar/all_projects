#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Defines DQN for RL Agent learning'''


# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
#    script imports
# imports


# constants
# constants


# classes
class QNetwork(nn.Module):
  '''Defines DQN for RL Agent learning'''

  def __init__(self, state_size:int, action_size:int, seed:int=42):
    super(QNetwork, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, action_size)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

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
