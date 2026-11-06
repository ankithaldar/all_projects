#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Entry point for evaluating a trained agent or
baseline heuristic.
'''

import argparse

import numpy as np

from truck_carton.config import AppConfig
from truck_carton.env.packing_env import (
  TruckCartonPackingEnv,
)
from truck_carton.evaluation.episode_runner import (
  make_predict,
  run_episode,
)
from truck_carton.evaluation.heuristic import (
  HeuristicAgent,
  RandomAgent,
)
from truck_carton.evaluation.metrics import (
  MetricsCollector,
)
from truck_carton.logging_config import load_model_safely


def main() -> None:
  '''Main.'''
  parser = argparse.ArgumentParser(
    description='Evaluate a packing agent or baseline'
  )
  parser.add_argument(
    '--model',
    type=str,
    default=None,
    help='Path to saved model (omit for baseline)',
  )
  parser.add_argument(
    '--agent',
    type=str,
    default='heuristic',
    choices=['heuristic', 'random', 'model'],
    help='Agent type when --model not given',
  )
  parser.add_argument(
    '--episodes',
    type=int,
    default=50,
    help='Number of evaluation episodes',
  )
  parser.add_argument(
    '--stage',
    type=int,
    default=0,
    help='Curriculum stage (0-4)',
  )
  parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Random seed',
  )
  parser.add_argument(
    '--allow-trusted-load',
    action='store_true',
    help='Opt in to loading a checkpoint: SB3 load '
    'unpickles it, so only load models you fully trust. '
    'Path must stay in an allowlisted model dir.',
  )
  parser.add_argument(
    '--model-dir',
    action='append',
    default=None,
    metavar='DIR',
    help='Allowlisted model dir (repeatable; default: '
    './output and ./models under cwd, or '
    'TRUCK_CARTON_MODEL_DIRS)',
  )
  args = parser.parse_args()

  # Load first so a rejected path fails before any work.
  model = None
  if args.model:
    model = load_model_safely(
      args.model,
      allow_dirs=args.model_dir,
      allow_trusted=args.allow_trusted_load,
    )

  config = AppConfig()
  env = TruckCartonPackingEnv(config=config, curriculum_stage=args.stage)
  collector = MetricsCollector()

  # Select agent
  agent = None
  if model is not None:
    agent_name = f'MaskablePPO ({args.model})'
  elif args.agent == 'random':
    agent = RandomAgent(seed=args.seed)
    agent_name = 'Random Baseline'
  else:
    agent = HeuristicAgent(config)
    agent_name = 'Heuristic Baseline'

  print(f'Agent: {agent_name}')
  print(f'Stage: {args.stage} | Episodes: {args.episodes}')
  print()

  predict = make_predict(model=model, agent=agent)
  all_metrics = []

  for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed + ep)
    result = run_episode(env, obs, predict)
    total_reward = result.total_reward

    metrics = collector.compute(
      episode_data=env.episode_data,
      spaces=env.spaces,
      placed_cartons=env.placed_cartons,
      current_weights=env.current_weights,
      total_reward=total_reward,
      curriculum_stage=args.stage,
      truck_travel=env.truck_travel,
      num_delivered=env.num_delivered,
    )
    all_metrics.append(metrics)

    print(
      f'Episode {ep + 1}/{args.episodes}: '
      f'placed='
      f'{metrics.num_placed}/'
      f'{metrics.num_total}, '
      f'delivered='
      f'{env.num_delivered}/'
      f'{metrics.num_total}, '
      f'vol_util='
      f'{metrics.fleet_volumetric_utilization:.2%}, '
      f'reward={metrics.total_reward:.2f}'
    )

  print(f'\n=== {agent_name} Results ===')
  print(
    'Completion rate:    '
    f'{np.mean([m.completion_rate for m in all_metrics]):.2%}'
  )
  print(
    'Vol utilization:    '
    f'{np.mean([m.fleet_volumetric_utilization for m in all_metrics]):.2%}'
  )
  print(
    'Avg displacement:   '
    f'{np.mean([m.avg_displacement_per_stop for m in all_metrics]):.2f}'
  )
  print(
    'Grouping score:     '
    f'{np.mean([m.grouping_compliance_rate for m in all_metrics]):.2%}'
  )
  print(
    'Fragility viol:     '
    f'{np.mean([m.fragility_violation_rate for m in all_metrics]):.2%}'
  )
  print(
    'Support viol:       '
    f'{np.mean([m.support_violation_rate for m in all_metrics]):.2%}'
  )
  print(
    'Weight viol:        '
    f'{np.mean([m.weight_violation_rate for m in all_metrics]):.2%}'
  )
  print(
    'Priority score:     '
    f'{np.mean([m.priority_accessibility_score for m in all_metrics]):.2%}'
  )
  print(
    'Delivery rate:      '
    f'{np.mean([m.delivery_completion_rate for m in all_metrics]):.2%}'
  )
  print(
    'Avg travel/truck:   '
    f'{np.mean([m.avg_travel_per_truck for m in all_metrics]):.1f}'
  )
  print(
    f'Mean reward:        {np.mean([m.total_reward for m in all_metrics]):.2f}'
  )


if __name__ == '__main__':
  main()
