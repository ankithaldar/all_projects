#!/usr/bin/env python3
"""Entry point for evaluating a trained agent."""

import argparse

import numpy as np
from sb3_contrib import MaskablePPO

from truck_carton.config import AppConfig
from truck_carton.env.packing_env import (
  TruckCartonPackingEnv,
)
from truck_carton.evaluation.metrics import (
  MetricsCollector,
)


def main() -> None:
  parser = argparse.ArgumentParser(
    description='Evaluate a trained packing agent'
  )
  parser.add_argument(
    '--model', type=str, required=True,
    help='Path to saved model',
  )
  parser.add_argument(
    '--episodes', type=int, default=50,
    help='Number of evaluation episodes',
  )
  parser.add_argument(
    '--stage', type=int, default=0,
    help='Curriculum stage (0-2)',
  )
  args = parser.parse_args()

  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=args.stage
  )
  model = MaskablePPO.load(args.model)
  collector = MetricsCollector()

  all_metrics = []

  for ep in range(args.episodes):
    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
      masks = env.action_masks()
      action, _ = model.predict(
        obs,
        action_masks=masks,
        deterministic=True,
      )
      obs, reward, terminated, truncated, info = (
        env.step(int(action))
      )
      total_reward += reward
      done = terminated or truncated

    metrics = collector.compute(
      episode_data=env.episode_data,
      spaces=env.spaces,
      placed_cartons=env.placed_cartons,
      current_weights=env.current_weights,
      total_reward=total_reward,
      curriculum_stage=args.stage,
    )
    all_metrics.append(metrics)

    print(
      f'Episode {ep + 1}/{args.episodes}: '
      f'placed='
      f'{metrics.num_placed}/'
      f'{metrics.num_total}, '
      f'vol_util='
      f'{metrics.fleet_volumetric_utilization:.2%}, '
      f'reward={metrics.total_reward:.2f}'
    )

  print('\n=== Aggregated Results ===')
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
    'Mean reward:        '
    f'{np.mean([m.total_reward for m in all_metrics]):.2f}'
  )


if __name__ == '__main__':
  main()
