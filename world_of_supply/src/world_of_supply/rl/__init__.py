#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Reinforcement-learning subpackage: env, encoders, models, training.

Heavy dependencies (torch, ray) are imported lazily so the environment,
encoders, and heuristics remain usable without them.
'''

from world_of_supply.rl.actions import ActionDecoder, PRICE_LEVELS, QUANTITY_LEVELS, RATE_LEVELS
from world_of_supply.rl.agents import (
    consumer_agent_id,
    facility_id_of,
    is_consumer,
    is_producer,
    producer_agent_id,
)
from world_of_supply.rl.env import EnvConfig, WorldOfSupplyEnv
from world_of_supply.rl.observations import ObservationEncoder, normalize
from world_of_supply.rl.rewards import RetailerProfitRewardShaper, RewardShaper

__all__ = [
    'ActionDecoder',
    'PRICE_LEVELS',
    'QUANTITY_LEVELS',
    'RATE_LEVELS',
    'consumer_agent_id',
    'facility_id_of',
    'is_consumer',
    'is_producer',
    'producer_agent_id',
    'EnvConfig',
    'WorldOfSupplyEnv',
    'FacilityNet',
    'ObservationEncoder',
    'normalize',
    'RetailerProfitRewardShaper',
    'RewardShaper',
]


def __getattr__(name: str):
  '''Lazily resolve torch-dependent symbols.

  Args:
    name: Attribute name.

  Returns:
    object: The resolved attribute.

  Raises:
    AttributeError: If the attribute is unknown.
    ImportError: If PyTorch is required but not installed.
  '''
  if name == 'FacilityNet':
    from world_of_supply.rl.models import FacilityNet

    return FacilityNet
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
