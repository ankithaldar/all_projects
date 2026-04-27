#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cat_game_env.crafting_env import CraftingEnv
from src.cat_game_env.frame_skipper import FrameSkipWrapper
from src.agent.masked_agent import MaskedAgent


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Train RL agent for crafting game"
  )
  parser.add_argument(
    "--config", default="config/training.yaml", help="Training config path"
  )
  parser.add_argument(
    "--timesteps", type=int, default=None, help="Override total timesteps"
  )
  parser.add_argument(
    "--frame-skip", type=int, default=None, help="Override frame skip"
  )
  args = parser.parse_args()

  with open(args.config, "r") as f:
    config = yaml.safe_load(f)

  env_config = {
    "crafting_tree_path": "config/crafting_tree.yaml",
    "targets_path": "config/targets.yaml",
    "max_batch_size": config["environment"]["max_batch_size"],
    "max_ticks": config["environment"]["max_ticks"],
    "initial_coins": config["environment"]["initial_coins"],
    "initial_stash": config["environment"].get("initial_stash", {}),
    "reward_weights": config.get("reward_weights"),
  }

  env = CraftingEnv(env_config)

  frame_skip = args.frame_skip or config["environment"].get("frame_skip", 1)
  if frame_skip > 1:
    env = FrameSkipWrapper(env, skip=frame_skip)

  agent = MaskedAgent(env, config)

  total_timesteps = args.timesteps or config["training"]["total_timesteps"]
  print(
    f"Training for {total_timesteps} timesteps "
    f"with frame_skip={frame_skip}"
  )
  agent.train(total_timesteps)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  model_dir = config["training"].get("model_dir", "output/models")
  os.makedirs(model_dir, exist_ok=True)
  save_path = os.path.join(model_dir, f"final_model_{timestamp}")
  agent.save(save_path)
  print(f"Model saved to {save_path}")


if __name__ == "__main__":
  main()
