#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Command-line interface for simulating, rendering, evaluating, and training.'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from world_of_supply.policies import ScriptedSupplyChainPolicy
from world_of_supply.rendering.renderer import AsciiWorldRenderer
from world_of_supply.rendering.status import WorldStatusFormatter
from world_of_supply.scenario import WorldBuilder


def run_simulation(ticks: int, seed: int | None, render_dir: str | None) -> int:
  '''Run the scripted supply chain for a number of ticks.

  Args:
    ticks: Number of ticks to simulate.
    seed: Optional scenario seed.
    render_dir: When set, save one PNG per rendered tick into this folder.

  Returns:
    int: Exit code.
  '''
  world = WorldBuilder.build(seed=seed)
  policy = ScriptedSupplyChainPolicy(seed=seed)
  renderer = AsciiWorldRenderer() if render_dir else None
  output = Path(render_dir) if render_dir else None
  if output:
    output.mkdir(parents=True, exist_ok=True)

  for tick in range(ticks):
    world.act(policy.compute_control(world))
    print(f'--- tick {tick + 1}/{ticks}: global balance {world.economy.global_balance()}')
    if renderer is not None and (tick + 1) % max(ticks // 10, 1) == 0:
      frame_path = output / f'frame_{tick:05d}.png'
      renderer.render(world).save(frame_path)
      print(f'    saved {frame_path}')

  formatter = WorldStatusFormatter()
  status_text = yaml.dump(formatter.status(world)).replace('\'', '')
  print(status_text)
  return 0


def run_baseline(episodes: int, seed: int | None) -> int:
  '''Evaluate the scripted agents through the RL environment.

  Args:
    episodes: Number of episodes to run.
    seed: Base random seed.

  Returns:
    int: Exit code.
  '''
  from world_of_supply.rl.env import EnvConfig, WorldOfSupplyEnv
  from world_of_supply.rl.training import evaluate_scripted

  env = WorldOfSupplyEnv(EnvConfig(episode_duration=200, downsampling_rate=20, seed=seed))
  returns = evaluate_scripted(env, episodes=episodes, seed=seed)
  for index, episode_return in enumerate(returns):
    print(f'episode {index}: total reward {episode_return:.2f}')
  return 0


def run_training(iterations: int, train_toy_factories_only: bool) -> int:
  '''Launch PPO training via RLLib.

  Args:
    iterations: Number of training iterations.
    train_toy_factories_only: Freeze non-toy-factory agents.

  Returns:
    int: Exit code.
  '''
  from world_of_supply.rl.env import EnvConfig
  from world_of_supply.rl.training import build_ppo_algorithm, train

  def log(iteration: int, result: dict) -> None:
    '''Print compact training progress.

    Resolves metric locations across RLLib result-schema versions.

    Args:
      iteration: Iteration index.
      result: RLLib result dictionary.
    '''
    runners = result.get('env_runners') or {}
    reward = result.get('episode_reward_mean')
    if reward is None:
      reward = runners.get('episode_return_mean')
    timesteps = result.get('timesteps_total')
    if timesteps is None:
      timesteps = runners.get('num_env_steps_sampled_lifetime')
    print(f'iteration {iteration}: reward_mean={reward} timesteps={timesteps}')

  algorithm = build_ppo_algorithm(
      {'env': EnvConfig()},
      train_toy_factories_only=train_toy_factories_only,
  )
  train(algorithm, iterations, log)
  return 0


def build_parser() -> argparse.ArgumentParser:
  '''Create the argument parser.

  Returns:
    argparse.ArgumentParser: Configured CLI parser.
  '''
  parser = argparse.ArgumentParser(prog='world-of-supply', description='Supply-chain MARL sandbox')
  subparsers = parser.add_subparsers(dest='command', required=True)

  simulate = subparsers.add_parser('simulate', help='Run the scripted supply chain')
  simulate.add_argument('--ticks', type=int, default=60)
  simulate.add_argument('--seed', type=int, default=None)
  simulate.add_argument('--render-dir', type=str, default=None)

  baseline = subparsers.add_parser('baseline', help='Score the scripted agents in the RL env')
  baseline.add_argument('--episodes', type=int, default=3)
  baseline.add_argument('--seed', type=int, default=42)

  training = subparsers.add_parser('train', help='Train PPO policies (requires ray)')
  training.add_argument('--iterations', type=int, default=5)
  training.add_argument('--toy-only', action='store_true')

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
  return run_training(args.iterations, args.toy_only)


if __name__ == '__main__':
  sys.exit(main())
