#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Run the genetic-algorithm baseline for the crafting game.

Runs the GA scheduler and exports the best schedule found.
'''

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from src.core.items import CraftingTree, ItemId, ITEM_NAME_TO_ID
from src.ga.ga_scheduler import GaScheduler
from src.ga.chromosome import Chromosome


def main() -> None:
  parser = argparse.ArgumentParser(
    description='Run GA baseline for crafting game'
  )
  parser.add_argument(
    '--config', default='config/ga.yaml', help='GA config path'
  )
  parser.add_argument(
    '--generations', type=int, default=None, help='Override n_generations'
  )
  parser.add_argument(
    '--output-dir', default='output/ga_results', help='Output directory'
  )
  args = parser.parse_args()

  with open(args.config, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)['ga']

  with open('config/training.yaml', 'r', encoding='utf-8') as f:
    train_config = yaml.safe_load(f)
  env_cfg = train_config.get('environment', {})
  initial_coins = env_cfg.get('initial_coins', 0)
  initial_stash = env_cfg.get('initial_stash', {})

  tree = CraftingTree.from_yaml('config/crafting_tree.yaml')

  with open('config/targets.yaml', 'r', encoding='utf-8') as f:
    targets_data = yaml.safe_load(f)['targets']
  targets = {ITEM_NAME_TO_ID[k.lower()]: v for k, v in targets_data.items()}

  scheduler = GaScheduler(
    config, tree, targets, initial_coins, initial_stash
  )
  hof = scheduler.run(
    n_generations=args.generations,
    output_dir=args.output_dir,
  )

  best_genes = scheduler.get_best_schedule(hof)
  if best_genes is not None:
    schedule = Chromosome.decode(best_genes)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(
      args.output_dir, f'ga_batch_schedule_{timestamp}.txt'
    )

    hdr_tick = 'tick_index'
    hdr_elapsed = 'elapsed_minutes'
    hdr_item = 'item_name'
    hdr_batch = 'batch_size_decided'
    with open(output_path, 'w', encoding='utf-8') as f:
      f.write(
        f'{hdr_tick:<12}{hdr_elapsed:<18}'
        f'{hdr_item:<20}{hdr_batch}\n'
      )
      f.write('-' * 68 + '\n')
      for t, tick_actions in enumerate(schedule):
        for item_id_int, batch_size in tick_actions.items():
          item_name = ItemId(item_id_int).name.lower()
          f.write(
            f'{t:<12}{t * 5:<18}{item_name:<20}{batch_size}\n'
          )

    print(f'\nBest schedule exported to {output_path}')
    best = hof[0]
    print(
      f'Best fitness: cost={best.fitness.values[0]:.0f}, '
      f'time={best.fitness.values[1]:.0f} ticks, '
      f'waste={best.fitness.values[2]:.0f}, '
      f'cost_per_item={best.fitness.values[3]:.1f}'
    )
    print(f'Pareto front size: {len(hof)}')


if __name__ == '__main__':
  main()
