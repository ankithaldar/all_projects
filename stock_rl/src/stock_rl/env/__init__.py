#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Reinforcement learning environments.

Only what the project actually uses is here: a Gymnasium-compatible
``PortfolioEnv`` over aligned daily bar panels, and the two Protocols
describing what an RL algorithm may expect from an environment.

Implemented as a package rather than a module because the eventual
hedge and baseline environments belong beside this one, and because the
package name is the single word the design docs use for it.
'''

from stock_rl.env.gym import Env
from stock_rl.env.nse import PortfolioEnv, RewardWeights

__all__ = ['Env', 'PortfolioEnv', 'RewardWeights']
