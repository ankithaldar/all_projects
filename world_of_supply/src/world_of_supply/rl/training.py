#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Training entry points: PPO via RLLib 2.x and scripted baseline evaluation.'''

from __future__ import annotations

from world_of_supply.rl.agents import is_producer

TRAINABLE_POLICIES = ('ppo_producer', 'ppo_consumer')
FROZEN_POLICIES = ('frozen_producer', 'frozen_consumer')
TOY_FACTORY_PREFIX = 'ToyFactoryCell'


def make_policy_mapping_fn(train_toy_factories_only: bool = False):
  '''Create a policy-mapping function over agent ids.

  Args:
    train_toy_factories_only: When True, only toy-factory agents map to the
      trainable PPO policies; all others use frozen clones, mirroring the
      legacy curriculum setup.

  Returns:
    Callable[[str], str]: Mapping from agent id to policy name.
  '''

  def mapping_fn(agent_id: str, episode=None) -> str:
    '''Resolve the policy for one agent.

    Args:
      agent_id: Agent id ending in producer/consumer suffix.
      episode: Optional running episode (new API stacks pass it).

    Returns:
      str: Policy name.
    '''
    trainable = True
    if train_toy_factories_only and not agent_id.startswith(TOY_FACTORY_PREFIX):
      trainable = False
    role = 'producer' if is_producer(agent_id) else 'consumer'
    prefix = 'ppo' if trainable else 'frozen'
    return f'{prefix}_{role}'

  return mapping_fn


def _configure(config, **settings):
  '''Apply scalar settings to an algorithm config across Ray versions.

  Newer Ray releases expose ``lr``/``train_batch_size``/``gamma`` as plain
  attributes while older ones accepted them on ``training()``; this helper
  prefers direct attributes and falls back to keyword arguments.

  Args:
    config: RLLib AlgorithmConfig instance.
    **settings: Setting name to value pairs.

  Returns:
    AlgorithmConfig: The same instance for chaining.
  '''
  for key, value in settings.items():
    if hasattr(type(config), key) or key in getattr(config, '__dict__', {}):
      setattr(config, key, value)
    else:
      config.training(**{key: value})
  return config


def build_ppo_algorithm(
    env_config: dict,
    lr: float = 2e-4,
    gamma: float = 0.99,
    vf_loss_coeff: float = 20.0,
    vf_clip_param: float = 200.0,
    train_batch_size: int = 2000,
    rollout_fragment_length: int = 50,
    num_workers: int = 2,
    use_lstm: bool = False,
    train_toy_factories_only: bool = False,
):
  '''Assemble a PPO algorithm over :class:`WorldOfSupplyEnv`.

  Uses the modern RLlib AlgorithmConfig builder (new API stack, PyTorch).
  Frozen policy clones share the same network class as trainable ones but
  are excluded from updates via ``policies_to_train``.

  Args:
    env_config: Serialized :class:`EnvConfig` fields for workers.
    lr: Learning rate.
    gamma: Discount factor.
    vf_loss_coeff: Value-function loss coefficient.
    vf_clip_param: Value clipping range.
    train_batch_size: Batch size per training step.
    rollout_fragment_length: Rollout fragment length.
    num_workers: Parallel rollout workers.
    use_lstm: Enable the recurrent variant of FacilityNet.
    train_toy_factories_only: Freeze all non-toy-factory agents.

  Returns:
    Algorithm: Built (not yet trained) PPO algorithm.
  '''
  from ray.rllib.algorithms.ppo import PPOConfig
  from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
  from ray.rllib.core.rl_module.rl_module import RLModuleSpec
  from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
  from ray.rllib.policy.policy import PolicySpec

  from world_of_supply.rl.env import WorldOfSupplyEnv
  from world_of_supply.rl.rl_modules import FacilityRLModule

  probe_env = WorldOfSupplyEnv(env_config['env'])
  module_class = FacilityRLModule if FacilityRLModule is not None else TorchRLModule
  producer_space = probe_env.action_space[probe_env.agent_ids[0]]
  consumer_space = probe_env.action_space[probe_env.agent_ids[1]]
  observation_space = probe_env.observation_space[probe_env.agent_ids[0]]
  policy_specs = {
      name: PolicySpec(
          policy_class=None,
          observation_space=observation_space,
          action_space=producer_space if name.endswith('producer') else consumer_space,
      )
      for name in (*TRAINABLE_POLICIES, *FROZEN_POLICIES)
  }

  config = PPOConfig()
  config = _configure(config, lr=lr, gamma=gamma, train_batch_size=train_batch_size)
  runners = config.env_runners if hasattr(config, 'env_runners') else config.rollout
  worker_key = 'num_env_runners' if hasattr(config, 'env_runners') else 'num_rollout_workers'
  config = runners(
      **{worker_key: num_workers, 'rollout_fragment_length': rollout_fragment_length}
  )
  config = (
      config.training(
          vf_loss_coeff=vf_loss_coeff,
          vf_clip_param=vf_clip_param,
      )
      .multi_agent(
          policies=policy_specs,
          policy_mapping_fn=make_policy_mapping_fn(train_toy_factories_only),
          policies_to_train=list(TRAINABLE_POLICIES),
      )
      .rl_module(
          rl_module_spec=MultiRLModuleSpec(
              rl_module_specs={
                  name: RLModuleSpec(
                      module_class=module_class,
                      observation_space=observation_space,
                      action_space=(
                          producer_space if name.endswith('producer') else consumer_space
                      ),
                      model_config={
                          'hidden_size': 256,
                          'lstm_cell_size': 64,
                          'use_lstm': use_lstm,
                      },
                  )
                  for name in (*TRAINABLE_POLICIES, *FROZEN_POLICIES)
              }
          )
      )
  )
  return config.build(env=WorldOfSupplyEnv)


def train(algorithm, n_iterations: int, log=lambda iteration, result: None) -> object:
  '''Run PPO iterations with curriculum progress reporting.

  Args:
    algorithm: Built RLLib algorithm.
    n_iterations: Number of training iterations.
    log: Callback ``(iteration, result_dict)`` invoked after each iteration.

  Returns:
    object: The trained algorithm.
  '''
  for iteration in range(n_iterations):
    algorithm.workers.foreach_worker(
        lambda worker: worker.foreach_env(lambda env: env.set_iteration(iteration, n_iterations))
    )
    result = algorithm.train()
    log(iteration, result)
  return algorithm


def evaluate_scripted(env, episodes: int = 3, seed: int | None = None) -> list[float]:
  '''Score the hand-coded baseline directly against the environment.

  Args:
    env: A :class:`WorldOfSupplyEnv` instance.
    episodes: Number of episodes to run.
    seed: Optional seed for reproducible supplier choices.

  Returns:
    list[float]: Total undiscounted reward per episode.
  '''
  episode_returns: list[float] = []
  for episode_index in range(episodes):
    observations, _ = env.reset(seed=None if seed is None else seed + episode_index)
    actions_seed = None if seed is None else seed + episode_index
    done = False
    total_reward = 0.0
    while not done:
      action_dict = env.scripted_actions(seed=actions_seed)
      observations, rewards, terminated, truncated, _ = env.step(action_dict)
      total_reward += sum(rewards.values())
      done = bool(truncated.get('__all__', False)) or bool(terminated.get('__all__', False))
    episode_returns.append(total_reward)
  return episode_returns
