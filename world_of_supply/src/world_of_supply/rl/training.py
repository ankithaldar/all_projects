#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Training entry points: PPO via RLLib 2.x and scripted baseline evaluation.'''

from __future__ import annotations

from world_of_supply.rl.agents import is_producer

TRAINABLE_POLICIES = ('ppo_producer', 'ppo_consumer')
FROZEN_POLICIES = ('frozen_producer', 'frozen_consumer')
TOY_FACTORY_PREFIX = 'ToyFactoryCell'
CURRICULUM_ATTR = 'world_of_supply_curriculum_state'

# Legacy parity: producer MLP [128, 128], consumer MLP [256, 256].
POLICY_MODEL_CONFIGS = {
    'producer': {'hidden_size': 128, 'hidden_layers': 2},
    'consumer': {'hidden_size': 256, 'hidden_layers': 2},
}


def make_policy_mapping_fn(
    train_toy_factories_only: bool = False, state: dict | None = None
):
  '''Create a policy-mapping function over agent ids.

  The mapping consults a mutable state dict so a training curriculum can
  promote additional facility classes to the trainable policies mid-run
  (restoring the legacy ``update_policy_map`` capability).

  Args:
    train_toy_factories_only: When True, only toy-factory agents start on
      the trainable PPO policies; all others use frozen clones.
    state: Optional mutable dict ``{'trainable_prefixes': set[str],
      'all_trainable': bool}``; created when omitted.

  Returns:
    tuple: ``(mapping_fn, state)`` — the callable maps agent ids to policy
    names, the state enables curriculum updates.
  '''
  if state is None:
    trainable_prefixes = {TOY_FACTORY_PREFIX}
    if not train_toy_factories_only:
      trainable_prefixes = set()
    state = {
        'trainable_prefixes': trainable_prefixes,
        'all_trainable': not train_toy_factories_only,
    }

  def mapping_fn(agent_id: str, episode=None) -> str:
    '''Resolve the policy for one agent.

    Args:
      agent_id: Agent id ending in producer/consumer suffix.
      episode: Optional running episode (new API stacks pass it).

    Returns:
      str: Policy name.
    '''
    prefixes = tuple(state['trainable_prefixes'])
    trainable = state['all_trainable']
    trainable = trainable or agent_id.startswith(prefixes)
    role = 'producer' if is_producer(agent_id) else 'consumer'
    prefix = 'ppo' if trainable else 'frozen'
    return f'{prefix}_{role}'

  return mapping_fn, state


def apply_curriculum(
    state: dict | None, iteration: int, n_iterations: int, schedule
) -> set[str]:
  '''Promote facility-class prefixes to trainable policies at thresholds.

  Mirrors the legacy curriculum: e.g. ``((0.25, 'WarehouseCell'),)``
  promotes warehouses once 25% of the iterations have completed.

  Args:
    state: Mapping state created by :func:`make_policy_mapping_fn`.
    iteration: Current iteration index.
    n_iterations: Total planned iterations.
    schedule: Sequence of ``(fraction, prefix)`` thresholds.

  Returns:
    set[str]: The currently trainable prefixes (empty when all agents are
    trainable regardless of state).
  '''
  if state is None or state.get('all_trainable'):
    return set()
  for fraction, prefix in schedule or ():
    if iteration >= int(fraction * n_iterations):
      state['trainable_prefixes'].add(prefix)
  return set(state['trainable_prefixes'])


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


def _probe_spaces(env_class, env_config: dict):
  '''Probe role spaces from a throwaway environment instance.

  Args:
    env_class: Environment class to instantiate for probing.
    env_config: Serialized :class:`EnvConfig` fields for workers.

  Returns:
    tuple: ``(producer_space, consumer_space, observation_space)``.
  '''
  probe_env = env_class(env_config['env'])
  producer_space = probe_env.action_space[probe_env.agent_ids[0]]
  consumer_space = probe_env.action_space[probe_env.agent_ids[1]]
  observation_space = probe_env.observation_space[probe_env.agent_ids[0]]
  return producer_space, consumer_space, observation_space


def _build_policy_specs(
    policy_spec, producer_space, consumer_space, observation_space
):
  '''Build PolicySpec entries for trainable and frozen policies.

  Args:
    policy_spec: PolicySpec class used to build the specs.
    producer_space: Producer action space.
    consumer_space: Consumer action space.
    observation_space: Shared observation space.

  Returns:
    dict: Policy name to :class:`PolicySpec`.
  '''
  specs = {}
  for name in (*TRAINABLE_POLICIES, *FROZEN_POLICIES):
    is_producer_policy = name.endswith('producer')
    if is_producer_policy:
      action_space = producer_space
    else:
      action_space = consumer_space
    specs[name] = policy_spec(
        policy_class=None,
        observation_space=observation_space,
        action_space=action_space,
    )
  return specs


def _build_module_specs(
    module_spec, module_class, producer_space, consumer_space,
    observation_space, use_lstm: bool
):
  '''Build RLModuleSpec entries for every policy.

  Args:
    module_spec: RLModuleSpec class used to build the specs.
    module_class: RLModule implementation class.
    producer_space: Producer action space.
    consumer_space: Consumer action space.
    observation_space: Shared observation space.
    use_lstm: Enable the recurrent variant of FacilityNet.

  Returns:
    dict: Policy name to :class:`RLModuleSpec`.
  '''
  specs = {}
  for name in (*TRAINABLE_POLICIES, *FROZEN_POLICIES):
    is_producer_policy = name.endswith('producer')
    role = 'producer' if is_producer_policy else 'consumer'
    model_config = dict(POLICY_MODEL_CONFIGS[role])
    model_config['lstm_cell_size'] = 64
    model_config['use_lstm'] = use_lstm
    if is_producer_policy:
      action_space = producer_space
    else:
      action_space = consumer_space
    specs[name] = module_spec(
        module_class=module_class,
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config,
    )
  return specs


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
    env_class=None,
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
    env_class: Optional env subclass (e.g. with a custom reward shaper);
      defaults to :class:`WorldOfSupplyEnv`.

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

  probed = _probe_spaces(WorldOfSupplyEnv, env_config)
  producer_space, consumer_space, observation_space = probed
  if FacilityRLModule is not None:
    module_class = FacilityRLModule
  else:
    module_class = TorchRLModule
  policy_specs = _build_policy_specs(
      PolicySpec, producer_space, consumer_space, observation_space
  )
  module_specs = _build_module_specs(
      RLModuleSpec, module_class, producer_space, consumer_space,
      observation_space, use_lstm,
  )
  mapping_fn, curriculum_state = make_policy_mapping_fn(
      train_toy_factories_only
  )

  config = PPOConfig()
  config = _configure(
      config, lr=lr, gamma=gamma, train_batch_size=train_batch_size
  )
  environment_class = env_class or WorldOfSupplyEnv
  config = config.environment(env=environment_class, env_config=env_config)
  if hasattr(config, 'env_runners'):
    runners = config.env_runners
  else:
    runners = config.rollout
  if hasattr(config, 'env_runners'):
    worker_key = 'num_env_runners'
  else:
    worker_key = 'num_rollout_workers'
  worker_settings = {
      worker_key: num_workers,
      'rollout_fragment_length': rollout_fragment_length,
  }
  config = runners(**worker_settings)
  config = config.training(
      vf_loss_coeff=vf_loss_coeff,
      vf_clip_param=vf_clip_param,
  )
  config = config.multi_agent(
      policies=policy_specs,
      policy_mapping_fn=mapping_fn,
      policies_to_train=list(TRAINABLE_POLICIES),
  )
  module_spec = MultiRLModuleSpec(rl_module_specs=module_specs)
  config = config.rl_module(rl_module_spec=module_spec)
  algorithm = config.build(env=environment_class)
  setattr(algorithm, CURRICULUM_ATTR, curriculum_state)
  return algorithm


def describe_model(use_lstm: bool = False) -> str:
  '''Render the architecture of the producer and consumer policy networks.

  Replacement for the legacy ``print_model_summaries``: builds one
  FacilityNet per role exactly as the trainer would and returns their
  ``repr`` strings.

  Args:
    use_lstm: Match the recurrent trainer variant.

  Returns:
    str: Multi-line architecture description.
  '''
  from world_of_supply.rl.env import EnvConfig, WorldOfSupplyEnv
  from world_of_supply.rl.models import FacilityNet

  env = WorldOfSupplyEnv(EnvConfig())
  lines = []
  for role in ('producer', 'consumer'):
    agent_id = env.agent_ids[0] if role == 'producer' else env.agent_ids[1]
    obs_dim = int(env.observation_space[agent_id].shape[0])
    action_nvec = list(env.action_space[agent_id].nvec)
    net = FacilityNet(
        obs_dim=obs_dim,
        action_nvec=action_nvec,
        hidden_size=POLICY_MODEL_CONFIGS[role]['hidden_size'],
        hidden_layers=POLICY_MODEL_CONFIGS[role]['hidden_layers'],
        lstm_cell_size=64,
        use_lstm=use_lstm,
    )
    lines.append(f'=== {role} ({type(net).__name__}) ===')
    lines.append(str(net))
  return '\n'.join(lines)


def _apply_to_envs(env, fn):
  '''Apply a callable to every underlying environment instance.

  New RLLib runners wrap environments in a Gymnasium ``VectorEnv`` whose
  children live under ``.envs`` and may carry additional Gymnasium wrappers;
  older paths expose the bare environment. Local placeholder runners without
  an environment are skipped safely.

  Args:
    env: An environment, vector environment, or ``None``.
    fn: Callable receiving each concrete (unwrapped) environment.

  Returns:
    None: Envs are mutated in place by ``fn``.
  '''
  if env is None:
    return
  sub_envs = getattr(env, 'envs', None)
  if sub_envs is None:
    candidates = [env]
  else:
    candidates = list(sub_envs)
  for candidate in candidates:
    if candidate is not None:
      fn(getattr(candidate, 'unwrapped', candidate))


def train(
    algorithm, n_iterations: int, log=lambda iteration, result: None,
    curriculum=()
) -> object:
  '''Run PPO iterations with curriculum progress reporting.

  Works across Ray 2.x releases by resolving the env-runner group through
  whichever accessor the installed version exposes. When a curriculum
  schedule is given, facility-class prefixes are promoted to the trainable
  policies as iteration thresholds pass (legacy ``update_policy_map``).

  Args:
    algorithm: Built RLLib algorithm.
    n_iterations: Number of training iterations.
    log: Callback ``(iteration, result_dict)`` invoked after each iteration.
    curriculum: Sequence of ``(fraction, prefix)`` promotion thresholds.

  Returns:
    object: The trained algorithm.
  '''

  def resolve_worker_set():
    '''Fetch the collection of rollout workers version-tolerantly.

    Returns:
      object: EnvRunnerGroup (new Ray) or WorkerSet (older Ray).
    '''
    try:
      return algorithm.env_runner_group
    except AttributeError:
      workers = algorithm.workers
      return workers() if callable(workers) else workers

  curriculum_state = getattr(algorithm, CURRICULUM_ATTR, None)
  promoted_seen: set[str] = set()
  for iteration in range(n_iterations):
    promoted = apply_curriculum(
        curriculum_state, iteration, n_iterations, curriculum
    )
    if promoted and promoted != promoted_seen:
      message = sorted(promoted)
      print(f'curriculum: trainable facility prefixes now {message}')
      promoted_seen = set(promoted)
    worker_set = resolve_worker_set()
    if hasattr(worker_set, 'foreach_env_runner'):
      worker_set.foreach_env_runner(
          lambda runner: _apply_to_envs(
              runner.env, lambda env: env.set_iteration(iteration, n_iterations)
          )
      )
    else:
      worker_set.foreach_worker(
          lambda worker: worker.foreach_env(
              lambda env: env.set_iteration(iteration, n_iterations)
          )
      )
    result = algorithm.train()
    log(iteration, result)
  return algorithm


def evaluate_scripted(
    env, episodes: int = 3, seed: int | None = None
) -> list[float]:
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
    if seed is None:
      reset_seed = None
    else:
      reset_seed = seed + episode_index
    env.reset(seed=reset_seed)
    actions_seed = None if seed is None else seed + episode_index
    done = False
    total_reward = 0.0
    while not done:
      action_dict = env.scripted_actions(seed=actions_seed)
      _, rewards, terminated, truncated, _ = env.step(action_dict)
      total_reward += sum(rewards.values())
      all_terminated = bool(terminated.get('__all__', False))
      all_truncated = bool(truncated.get('__all__', False))
      done = all_terminated or all_truncated
    episode_returns.append(total_reward)
  return episode_returns
