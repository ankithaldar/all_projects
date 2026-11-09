#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Multi-agent Gymnasium/RLLib environment wrapping the supply-chain sim.'''

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
  from ray.rllib.env.multi_agent_env import MultiAgentEnv as _MultiAgentEnvBase
except ImportError:
  from gymnasium import Env as _MultiAgentEnvBase

from gymnasium.spaces import Box, Dict as DictSpace, MultiDiscrete

from world_of_supply.economy import BalanceSheet
from world_of_supply.rl.actions import PRICE_LEVELS, ActionDecoder
from world_of_supply.rl.agents import (
    consumer_agent_id,
    facility_type_name,
    is_producer,
    producer_agent_id,
)
from world_of_supply.rl.observations import ObservationEncoder
from world_of_supply.rl.rewards import RewardShaper, RetailerProfitRewardShaper
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.world import Control


@dataclass
class EnvConfig:
  '''Environment-level settings.

  Attributes:
    scenario: World construction parameters.
    episode_duration: Simulation ticks per episode.
    downsampling_rate: Sim ticks per decision (>= 1).
    global_reward_weight_producer: Trust of producers in the global signal.
    global_reward_weight_consumer: Trust of consumers in the global signal.
    seed: Default reset seed when none is passed to :meth:`reset`.
  '''

  scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
  episode_duration: int = 1000
  downsampling_rate: int = 20
  global_reward_weight_producer: float = 0.9
  global_reward_weight_consumer: float = 0.9
  seed: int | None = None


def coerce_env_config(config) -> 'EnvConfig':
  '''Normalize RLLib-supplied environment configuration.

  RLLib forwards ``env_config`` to the env constructor verbatim, so the
  value may be an :class:`EnvConfig`, a ``{'env': EnvConfig}`` wrapper (the
  shape used by :func:`world_of_supply.rl.training.build_ppo_algorithm`), a
  dict of EnvConfig fields, or ``None``.

  Args:
    config: Any supported configuration shape.

  Returns:
    EnvConfig: Normalized configuration.

  Raises:
    TypeError: If the shape cannot be interpreted.
  '''
  if config is None:
    return EnvConfig()
  if isinstance(config, EnvConfig):
    return config
  if isinstance(config, dict):
    if set(config) == {'env'}:
      inner = config['env']
      if isinstance(inner, EnvConfig):
        return inner
      return EnvConfig(**inner)
    return EnvConfig(**config)
  config_type = type(config).__name__
  raise TypeError(f'Cannot interpret env config of type {config_type}')


def _collect_product_ids(reference_world) -> list[str]:
  '''Collect the sorted global product id ordering.

  Args:
    reference_world: Reference world supplying facility BOMs.

  Returns:
    list[str]: Sorted product ids.
  '''
  products = set()
  for facility in reference_world.facilities.values():
    products.add(facility.bom.output_product_id)
    products.update(facility.bom.inputs.keys())
  return sorted(products)


def _collect_facility_types(reference_world) -> dict[str, int]:
  '''Assign an index to each facility class of the reference world.

  Args:
    reference_world: Reference world supplying facilities.

  Returns:
    dict[str, int]: Facility class name to one-hot column.
  '''
  facility_types: dict[str, int] = {}
  for facility in reference_world.facilities.values():
    class_name = facility_type_name(facility)
    if class_name not in facility_types:
      facility_types[class_name] = len(facility_types)
  return facility_types


def _max_sources_per_facility(reference_world) -> int:
  '''Find the widest consumer source list of the reference world.

  Args:
    reference_world: Reference world supplying facilities.

  Returns:
    int: Maximum number of sources, zero when no consumer exists.
  '''
  extents = []
  for facility in reference_world.facilities.values():
    if facility.consumer is not None:
      extents.append(len(facility.consumer.sources))
  return max(extents, default=0)


def _build_agent_ids(reference_world) -> list[str]:
  '''Build producer and consumer agent ids for every facility.

  Args:
    reference_world: Reference world supplying facility ids.

  Returns:
    list[str]: Agent ids ordered by facility.
  '''
  agent_ids = []
  for facility_id in reference_world.facilities:
    agent_ids.append(producer_agent_id(facility_id))
    agent_ids.append(consumer_agent_id(facility_id))
  return agent_ids


def _build_spaces(agent_ids, total_dim, n_products, max_sources):
  '''Build observation and action spaces keyed by agent id.

  Args:
    agent_ids: Agent ids receiving a space entry.
    total_dim: Flattened observation dimension per agent.
    n_products: Global product count.
    max_sources: Maximum suppliers any facility has.

  Returns:
    tuple: ``(producer_space, consumer_space, observation_space,
    action_space)``.
  '''
  obs_box = Box(low=0.0, high=1.0, shape=(total_dim,), dtype=np.float32)
  producer_space = MultiDiscrete([len(PRICE_LEVELS), 6])
  consumer_width = max(max_sources, 1)
  consumer_space = MultiDiscrete([n_products, consumer_width, 6])
  observation_space = DictSpace(
      {agent_id: obs_box for agent_id in agent_ids}
  )
  action_space = DictSpace({
      agent_id: (
          producer_space if is_producer(agent_id) else consumer_space
      )
      for agent_id in agent_ids
  })
  return producer_space, consumer_space, observation_space, action_space


class WorldOfSupplyEnv(_MultiAgentEnvBase):
  '''RLlib-compatible multi-agent environment over the simulated economy.

  Two agents (producer/consumer) control each facility. One decision step
  advances the simulation by ``downsampling_rate`` ticks: the first tick
  applies decoded controls, the remaining ones run untouched so slow
  logistics can progress between decisions.

  Attributes:
    config: Environment configuration.
    observation_space: Dict space keyed by agent id.
    action_space: Dict space keyed by agent id.
  '''

  def __init__(self, config: EnvConfig | None = None) -> None:
    '''Derive spaces from a reference world and prepare calculators.

    Args:
      config: Environment configuration; defaults are used when omitted.

    Returns:
      None: The env is mutated in place.
    '''
    super().__init__()
    self.config = coerce_env_config(config)
    self.reference_world = WorldBuilder.build(self.config.scenario, seed=0)
    self.product_ids = _collect_product_ids(self.reference_world)
    self.facility_types = _collect_facility_types(self.reference_world)
    self.max_sources_per_facility = _max_sources_per_facility(
        self.reference_world
    )
    self.agent_ids = _build_agent_ids(self.reference_world)
    self.possible_agents = list(self.agent_ids)
    self.agents = list(self.agent_ids)

    self.encoder = ObservationEncoder(
        product_ids=self.product_ids,
        facility_types=self.facility_types,
        max_sources_per_facility=self.max_sources_per_facility,
        episode_duration=self.config.episode_duration,
        reference_facility_count=len(self.reference_world.facilities),
    )
    self.decoder = ActionDecoder(self.product_ids)
    producer_weight = self.config.global_reward_weight_producer
    consumer_weight = self.config.global_reward_weight_consumer
    self.reward_shaper: RewardShaper = RetailerProfitRewardShaper(
        producer_global_weight=producer_weight,
        consumer_global_weight=consumer_weight,
    )

    total_dim = self.encoder.total_dim
    n_products = len(self.product_ids)
    spaces = _build_spaces(
        self.agent_ids,
        total_dim,
        n_products,
        self.max_sources_per_facility,
    )
    producer_space, consumer_space, obs, act = spaces
    self._producer_action_space = producer_space
    self._consumer_action_space = consumer_space
    self.observation_space = obs
    self.action_space = act

    self.world = None
    self.current_iteration = 0
    self.n_iterations = 0

  def reset(self, *, seed: int | None = None, options: dict | None = None):
    '''Build a fresh world and return initial observations.

    Args:
      seed: Random seed; falls back to ``config.seed`` when omitted.
      options: Unused Gymnasium hook.

    Returns:
      tuple[dict, dict]: Observations and empty info per Gymnasium API.
    '''
    super().reset(seed=seed)
    effective_seed = seed if seed is not None else self.config.seed
    self.world = WorldBuilder.build(self.config.scenario, seed=effective_seed)
    observations, raws = self.encoder.encode_world(self.world)
    self._latest_raw = raws
    return observations, {}

  def step(self, action_dict: dict[str, object]):
    '''Apply one decision and its downsampling tail.

    Args:
      action_dict: Actions keyed by agent id.

    Returns:
      tuple: ``(obs, rewards, terminated, truncated, infos)`` following the
      Gymnasium multi-agent API.
    '''
    control = self.decoder.decode(action_dict, self.world)
    outcome = self.world.act(control)
    for _ in range(self.config.downsampling_rate - 1):
      no_op_outcome = self.world.act(Control(facility_controls={}))
      self._accumulate(outcome.facility_step_balance_sheets, no_op_outcome)

    rewards = self.reward_shaper.shape(
        outcome.facility_step_balance_sheets,
        self.current_iteration,
        self.n_iterations,
    )
    observations, raws = self.encoder.encode_world(self.world)
    self._latest_raw = raws
    done = self.world.time_step >= self.config.episode_duration
    terminated = {agent_id: False for agent_id in observations}
    terminated['__all__'] = False
    truncated = {agent_id: done for agent_id in observations}
    truncated['__all__'] = done
    return observations, rewards, terminated, truncated, raws

  @staticmethod
  def _accumulate(target: dict[str, BalanceSheet], addition) -> None:
    '''Fold no-op tick sheets into the primary outcome.

    Args:
      target: Sheets of the decision tick, modified in place.
      addition: Outcome of a no-op tick contributing extra sheets.

    Returns:
      None: The target sheets are mutated in place.
    '''
    for facility_id, sheet in addition.facility_step_balance_sheets.items():
      target[facility_id] = target.get(facility_id, BalanceSheet()) + sheet

  def set_iteration(self, iteration: int, n_iterations: int) -> None:
    '''Publish curriculum progress used by reward shaping.

    Args:
      iteration: Current training iteration.
      n_iterations: Total planned iterations.

    Returns:
      None: The env is mutated in place.
    '''
    self.current_iteration = iteration
    self.n_iterations = n_iterations

  def scripted_actions(self, seed: int | None = None) -> dict:
    '''Convenience accessor producing the heuristic baseline actions.

    Args:
      seed: Optional seed for supplier choice randomness.

    Returns:
      dict: Agent actions mirroring the hand-coded policy.
    '''
    from world_of_supply.rl.heuristics import ScriptedAgentController

    controller = ScriptedAgentController(
        self.encoder,
        len(self.product_ids),
        self.max_sources_per_facility,
        seed=seed,
    )
    return controller.actions(self.world)
