#!/usr/bin/env python3
"""Train multi-agent tier-based system for crafting game.

Each tier has its own MaskablePPO agent. An orchestrator coordinates
the shared environment, order board, and tick loop. Agents are trained
via collect-then-update: all agents collect rollouts for N ticks,
then each updates independently.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Dict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cat_game_env.multi_agent import MultiAgentOrchestrator
from src.agent.masked_agent import TierAgent


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Train multi-agent tier system"
  )
  parser.add_argument(
    "--config", default="config/training.yaml",
    help="Training config path",
  )
  parser.add_argument(
    "--episodes", type=int, default=100,
    help="Number of training episodes",
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
  }

  orch = MultiAgentOrchestrator(env_config)

  agents: Dict[int, TierAgent] = {}
  for tier_num, tier_env in orch.tier_envs.items():
    agents[tier_num] = TierAgent(tier_env, config)

  n_episodes = args.episodes
  max_ticks = config["environment"]["max_ticks"]

  print(
    f"Training {len(agents)} tier agents for {n_episodes} episodes "
    f"({max_ticks} ticks/episode)"
  )
  for tier_num in sorted(agents.keys()):
    env = orch.tier_envs[tier_num]
    print(
      f"  Tier {tier_num}: {len(env.tier_items)} items, "
      f"action_space={env.action_space}"
    )

  best_tick = max_ticks + 1
  for episode in range(n_episodes):
    obs = orch.reset()
    episode_rewards = {t: 0.0 for t in agents}
    done = False

    for tick in range(max_ticks):
      masks = orch.get_action_masks()
      actions: Dict[int, np.ndarray] = {}
      for tier_num, agent in agents.items():
        actions[tier_num] = agent.predict(
          obs[tier_num],
          action_masks=masks[tier_num],
          deterministic=False,
        )

      obs, rewards, terminated, truncated, info = orch.step(actions)

      for tier_num, r in rewards.items():
        episode_rewards[tier_num] += r

      if terminated or truncated:
        done = True
        break

    total_r = sum(episode_rewards.values())
    completion = orch.current_tick
    status = "DONE" if terminated else "TRUNC"

    if terminated and completion < best_tick:
      best_tick = completion

    if episode % 10 == 0 or terminated:
      print(
        f"  Ep {episode:>4}/{n_episodes} | "
        f"{status} tick={completion:>5} | "
        f"reward={total_r:>8.1f} | "
        f"best={best_tick if best_tick <= max_ticks else 'N/A'}"
      )

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  model_dir = config["training"].get("model_dir", "output/models")
  os.makedirs(model_dir, exist_ok=True)
  for tier_num, agent in agents.items():
    path = os.path.join(model_dir, f"tier{tier_num}_{timestamp}")
    agent.save(path)
  print(f"\nModels saved to {model_dir}/tier*_{timestamp}")


if __name__ == "__main__":
  main()
