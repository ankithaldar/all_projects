#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Entry point for visualizing packing results.'''

import argparse

import matplotlib.pyplot as plt

from truck_carton.config import AppConfig
from truck_carton.env.packing_env import (
  TruckCartonPackingEnv,
)
from truck_carton.evaluation.episode_runner import (
  make_predict,
  run_episode,
)
from truck_carton.evaluation.visualizer import (
  GridRenderer,
  PackingVisualizer,
)
from truck_carton.logging_config import load_model_safely


def main() -> None:
  '''Main.'''
  parser = argparse.ArgumentParser(description='Visualize packing results')
  parser.add_argument(
    '--model',
    type=str,
    default=None,
    help='Path to saved model',
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
    '--gif',
    type=str,
    default=None,
    help='Save animation as GIF to this path',
  )
  parser.add_argument(
    '--mode',
    type=str,
    default='grid',
    choices=['grid', 'packing', 'both'],
    help='Visualization mode',
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

  renderer = GridRenderer()
  obs, _ = env.reset(seed=args.seed)

  # Capture initial frame
  renderer.capture_frame(env.get_render_snapshot())

  def _capture(
    unused_step,
    unused_obs,
    unused_reward,
    unused_info,
    unused_done,
  ):
    renderer.capture_frame(env.get_render_snapshot())

  run_episode(
    env,
    obs,
    make_predict(model=model),
    on_step=_capture,
  )

  delivered = env.num_delivered
  total = len(env.episode_data.cartons)
  print(f'Delivered {delivered}/{total} cartons in {env.step_count} steps')

  if args.gif:
    renderer.save_gif(args.gif)
    print(f'Saved animation to {args.gif}')

  if args.mode in ('grid', 'both'):
    final = renderer.render(env.get_render_snapshot())
    final.show()

  if args.mode in ('packing', 'both'):
    viz = PackingVisualizer()
    viz.render_all_trucks(
      env.episode_data,
      env.spaces,
      env.placed_cartons,
    )
    plt.show()


if __name__ == '__main__':
  main()
