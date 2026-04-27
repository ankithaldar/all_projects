#!/usr/bin/env python3
"""Entry point for visualizing packing results."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO

from truck_carton.config import AppConfig
from truck_carton.env.packing_env import (
  TruckCartonPackingEnv,
)
from truck_carton.evaluation.visualizer import (
  GridRenderer,
  PackingVisualizer,
)


def main() -> None:
  parser = argparse.ArgumentParser(
    description='Visualize packing results'
  )
  parser.add_argument(
    '--model', type=str, default=None,
    help='Path to saved model',
  )
  parser.add_argument(
    '--stage', type=int, default=0,
    help='Curriculum stage (0-4)',
  )
  parser.add_argument(
    '--seed', type=int, default=42,
    help='Random seed',
  )
  parser.add_argument(
    '--gif', type=str, default=None,
    help='Save animation as GIF to this path',
  )
  parser.add_argument(
    '--mode', type=str, default='grid',
    choices=['grid', 'packing', 'both'],
    help='Visualization mode',
  )
  args = parser.parse_args()

  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=args.stage
  )

  model = None
  if args.model:
    model = MaskablePPO.load(args.model)

  renderer = GridRenderer()
  obs, info = env.reset(seed=args.seed)

  # Capture initial frame
  renderer.capture_frame(
    env.get_render_snapshot()
  )

  done = False
  while not done:
    masks = env.action_masks()
    if not masks.any():
      break

    if model is not None:
      action, _ = model.predict(
        obs,
        action_masks=masks,
        deterministic=True,
      )
    else:
      valid_actions = np.where(masks)[0]
      action = int(
        np.random.choice(valid_actions)
      )

    obs, reward, terminated, truncated, info = (
      env.step(int(action))
    )
    done = terminated or truncated

    renderer.capture_frame(
      env.get_render_snapshot()
    )

  delivered = env.num_delivered
  total = len(env.episode_data.cartons)
  print(
    f'Delivered {delivered}/{total} cartons'
    f' in {env.step_count} steps'
  )

  if args.gif:
    renderer.save_gif(args.gif)
    print(f'Saved animation to {args.gif}')

  if args.mode in ('grid', 'both'):
    final = renderer.render(
      env.get_render_snapshot()
    )
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
