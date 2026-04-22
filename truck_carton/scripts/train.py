#!/usr/bin/env python3
"""Entry point for training the packing agent."""

import argparse
from dataclasses import replace

from truck_carton.config import AppConfig
from truck_carton.training.trainer import Trainer


def main() -> None:
  parser = argparse.ArgumentParser(
    description='Train the truck-carton agent'
  )
  parser.add_argument(
    '--timesteps', type=int, default=None,
    help='Override total timesteps',
  )
  parser.add_argument(
    '--output', type=str, default='./output',
    help='Output directory',
  )
  parser.add_argument(
    '--seed', type=int, default=None,
    help='Random seed',
  )
  args = parser.parse_args()

  config = AppConfig()

  if (
    args.timesteps is not None
    or args.seed is not None
  ):
    tc = config.training
    if args.timesteps is not None:
      tc = replace(
        tc, total_timesteps=args.timesteps
      )
    if args.seed is not None:
      tc = replace(tc, seed=args.seed)
    config = replace(config, training=tc)

  trainer = Trainer(
    config, output_dir=args.output
  )
  trainer.train()
  print(
    'Training complete. Model saved to'
    f' {args.output}/models/'
  )


if __name__ == '__main__':
  main()
