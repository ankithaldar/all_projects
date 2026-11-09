#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Command-line interface for simulating, rendering, evaluating.

Covers the scripted simulation loop, RL baseline scoring, and PPO
training entry points.
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from world_of_supply.policies import ScriptedSupplyChainPolicy
from world_of_supply.rendering.renderer import AsciiWorldRenderer
from world_of_supply.rendering.status import WorldStatusFormatter
from world_of_supply.scenario import WorldBuilder


def _should_render_frame(tick: int, ticks: int) -> bool:
  '''Decide whether one tick should be rendered to disk.

  Args:
    tick: Zero-based tick index.
    ticks: Total number of ticks in the simulation.

  Returns:
    bool: True on every ``ticks // 10``-th tick, at least once.
  '''
  return (tick + 1) % max(ticks // 10, 1) == 0


def _frame_filename(tick: int) -> str:
  '''Build the output file name for one rendered tick.

  Args:
    tick: Zero-based tick index.

  Returns:
    str: Frame file name such as ``frame_00003.png``.
  '''
  return f'frame_{tick:05d}.png'


def _tick_line(tick: int, ticks: int, world) -> str:
  '''Format the console line printed after one simulation tick.

  Args:
    tick: Zero-based tick index.
    ticks: Total number of ticks in the simulation.
    world: World snapshot after the tick.

  Returns:
    str: Tick progress line with the global balance.
  '''
  balance = world.economy.global_balance()
  return f'--- tick {tick + 1}/{ticks}: global balance {balance}'


def _render_frame(renderer, output: Path, tick: int, world) -> Path:
  '''Render one tick to disk and report the frame path.

  Args:
    renderer: World renderer producing PNG images.
    output: Directory receiving the rendered frame.
    tick: Zero-based tick index.
    world: World snapshot to render.

  Returns:
    Path: Path of the saved frame image.
  '''
  frame_path = output / _frame_filename(tick)
  renderer.render(world).save(frame_path)
  return frame_path


def _run_ticks(world, policy, ticks: int, renderer, output) -> None:
  '''Run the scripted policy for the requested number of ticks.

  Args:
    world: Mutable world advanced in place.
    policy: Policy computing controls for the world.
    ticks: Number of ticks to simulate.
    renderer: Optional renderer saving periodic frames.
    output: Optional directory receiving rendered frames.

  Returns:
    None: The world is mutated in place.
  '''
  for tick in range(ticks):
    world.act(policy.compute_control(world))
    print(_tick_line(tick, ticks, world))
    should_render = renderer is not None and _should_render_frame(tick, ticks)
    if should_render:
      frame_path = _render_frame(renderer, output, tick, world)
      print(f'    saved {frame_path}')


def _print_status(world) -> None:
  '''Print the final world status as YAML.

  Args:
    world: World snapshot after the simulation.

  Returns:
    None: Output is printed; no value is returned.
  '''
  formatter = WorldStatusFormatter()
  status_text = yaml.dump(formatter.status(world)).replace('\'', '')
  print(status_text)


def run_simulation(ticks: int, seed: int | None, render_dir: str | None) -> int:
  '''Run the scripted supply chain for a number of ticks.

  Args:
    ticks: Number of ticks to simulate.
    seed: Optional scenario seed.
    render_dir: When set, save one PNG per rendered tick into
      this folder.

  Returns:
    int: Exit code.
  '''
  world = WorldBuilder.build(seed=seed)
  policy = ScriptedSupplyChainPolicy(seed=seed)
  renderer = AsciiWorldRenderer() if render_dir else None
  output = Path(render_dir) if render_dir else None
  if output:
    output.mkdir(parents=True, exist_ok=True)

  _run_ticks(world, policy, ticks, renderer, output)
  _print_status(world)
  return 0


def _baseline_env(seed: int | None):
  '''Build the RL environment used for scripted baseline scoring.

  Args:
    seed: Base random seed.

  Returns:
    WorldOfSupplyEnv: Configured RL environment.
  '''
  from world_of_supply.rl.env import EnvConfig, WorldOfSupplyEnv

  return WorldOfSupplyEnv(
      EnvConfig(episode_duration=200, downsampling_rate=20, seed=seed)
  )


def run_baseline(episodes: int, seed: int | None) -> int:
  '''Evaluate the scripted agents through the RL environment.

  Args:
    episodes: Number of episodes to run.
    seed: Base random seed.

  Returns:
    int: Exit code.
  '''
  from world_of_supply.rl.training import evaluate_scripted

  env = _baseline_env(seed)
  returns = evaluate_scripted(env, episodes=episodes, seed=seed)
  for index, episode_return in enumerate(returns):
    print(f'episode {index}: total reward {episode_return:.2f}')
  return 0


def _training_progress_line(iteration: int, result: dict) -> str:
  '''Format one compact training progress line.

  Resolves metric locations across RLLib result-schema versions.

  Args:
    iteration: Iteration index.
    result: RLLib result dictionary.

  Returns:
    str: Progress line with reward, timesteps, and extras.
  '''
  runners = result.get('env_runners') or {}
  reward = result.get('episode_reward_mean')
  if reward is None:
    reward = runners.get('episode_return_mean')
  timesteps = result.get('timesteps_total')
  if timesteps is None:
    timesteps = runners.get('num_env_steps_sampled_lifetime')
  extras = []
  for source, key in (
      (runners, 'episode_return_max'),
      (runners, 'episode_return_min'),
      (runners, 'episode_len_mean'),
  ):
    if source.get(key) is not None:
      extras.append(f'{key}={source[key]}')
  suffix = ' '.join(extras)
  line = f'iteration {iteration}: reward_mean={reward} '
  line = f'{line}timesteps={timesteps} {suffix}'
  return line.rstrip()


def _log_training_progress(iteration: int, result: dict) -> None:
  '''Print compact training progress.

  Args:
    iteration: Iteration index.
    result: RLLib result dictionary.

  Returns:
    None: Output is printed; no value is returned.
  '''
  print(_training_progress_line(iteration, result))


def _training_curriculum(curriculum_warehouses: bool) -> tuple:
  '''Build the warehouse curriculum schedule for training.

  Args:
    curriculum_warehouses: Promote warehouses to trainable
      policies at 25% and 35% of the iterations when set.

  Returns:
    tuple: Curriculum entries, possibly empty.
  '''
  if curriculum_warehouses:
    return ((0.25, 'WarehouseCell'), (0.35, 'WarehouseCell'))
  return ()


def run_training(
    iterations: int,
    train_toy_factories_only: bool,
    curriculum_warehouses: bool = False,
) -> int:
  '''Launch PPO training via RLLib.

  Args:
    iterations: Number of training iterations.
    train_toy_factories_only: Freeze non-toy-factory agents.
    curriculum_warehouses: Promote warehouses to trainable
      policies at 25% and 35% of the iterations, mirroring the
      legacy curriculum.

  Returns:
    int: Exit code.
  '''
  from world_of_supply.rl.env import EnvConfig
  from world_of_supply.rl.training import (
      build_ppo_algorithm,
      describe_model,
      train,
  )

  print(describe_model())
  curriculum = _training_curriculum(curriculum_warehouses)
  algorithm = build_ppo_algorithm(
      {'env': EnvConfig()},
      train_toy_factories_only=train_toy_factories_only,
  )
  train(algorithm, iterations, _log_training_progress, curriculum=curriculum)
  return 0


def _add_simulate_parser(subparsers):
  '''Add the ``simulate`` subcommand arguments.

  Args:
    subparsers: Argument subparsers collection.

  Returns:
    argparse.ArgumentParser: The ``simulate`` subparser.
  '''
  simulate = subparsers.add_parser(
      'simulate', help='Run the scripted supply chain'
  )
  simulate.add_argument('--ticks', type=int, default=60)
  simulate.add_argument('--seed', type=int, default=None)
  simulate.add_argument('--render-dir', type=str, default=None)
  return simulate


def _add_baseline_parser(subparsers):
  '''Add the ``baseline`` subcommand arguments.

  Args:
    subparsers: Argument subparsers collection.

  Returns:
    argparse.ArgumentParser: The ``baseline`` subparser.
  '''
  baseline = subparsers.add_parser(
      'baseline', help='Score the scripted agents in the RL env'
  )
  baseline.add_argument('--episodes', type=int, default=3)
  baseline.add_argument('--seed', type=int, default=42)
  return baseline


def _add_training_parser(subparsers):
  '''Add the ``train`` subcommand arguments.

  Args:
    subparsers: Argument subparsers collection.

  Returns:
    argparse.ArgumentParser: The ``train`` subparser.
  '''
  training = subparsers.add_parser(
      'train', help='Train PPO policies (requires ray)'
  )
  training.add_argument('--iterations', type=int, default=5)
  training.add_argument('--toy-only', action='store_true')
  training.add_argument('--curriculum-warehouses', action='store_true')
  return training


def build_parser() -> argparse.ArgumentParser:
  '''Create the argument parser.

  Returns:
    argparse.ArgumentParser: Configured CLI parser.
  '''
  parser = argparse.ArgumentParser(
      prog='world-of-supply',
      description='Supply-chain MARL sandbox',
  )
  subparsers = parser.add_subparsers(dest='command', required=True)
  _add_simulate_parser(subparsers)
  _add_baseline_parser(subparsers)
  _add_training_parser(subparsers)
  return parser


def main(argv: list[str] | None = None) -> int:
  '''CLI entry point.

  Args:
    argv: Argument vector; defaults to ``sys.argv[1:]``.

  Returns:
    int: Process exit code.
  '''
  args = build_parser().parse_args(argv)
  if args.command == 'simulate':
    return run_simulation(args.ticks, args.seed, args.render_dir)
  if args.command == 'baseline':
    return run_baseline(args.episodes, args.seed)
  return run_training(
      args.iterations, args.toy_only, args.curriculum_warehouses
  )


if __name__ == '__main__':
  sys.exit(main())
