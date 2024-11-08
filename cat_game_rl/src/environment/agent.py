#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''Agent'''


# imports
from abc import ABC
#   script imports
# imports


class Agent(ABC):
  '''Abstract base class for an agent in Reinforcement Learning (RL).

    This class defines the core interface for an RL agent, requiring all
    subclasses to implement the act method.

    Args:
      control: An object representing the control input to the agent. The
      specific format of the control object will depend on the
      environment the agent interacts with.
  '''

  def act(self, control):
    pass
