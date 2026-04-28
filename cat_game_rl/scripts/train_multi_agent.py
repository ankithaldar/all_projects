#!/usr/bin/env python3
"""Train multi-agent tier-based system for crafting game.

Each tier agent trains independently via SB3's MaskablePPO.learn().
The TierEnv.step() delegates to the shared orchestrator, with other
tiers using no-op actions during the focal agent's training. Agents
are trained round-robin: each tier gets a training batch, then the
next tier trains, cycling for the configured number of rounds.
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


def evaluate_episode(
  orch: MultiAgentOrchestrator,
  agents: Dict[int, TierAgent],
  max_ticks: int,
) -> tuple[int, float]:
  obs = orch.reset()
  total_reward = 0.0
  for _ in range(max_ticks):
    masks = orch.get_action_masks()
    actions = {}
    for tier_num, agent in agents.items():
      actions[tier_num] = agent.predict(
        obs[tier_num], action_masks=masks[tier_num],
      )
    obs, rewards, terminated, truncated, _ = orch.step(actions)
    total_reward += sum(rewards.values())
    if terminated or truncated:
      break
  return orch.current_tick, total_reward


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Train multi-agent tier system"
  )
  parser.add_argument(
    "--config", default="config/training.yaml",
    help="Training config path",
  )
  parser.add_argument(
    "--rounds", type=int, default=10,
    help="Training rounds (each tier trains once per round)",
  )
  parser.add_argument(
    "--steps-per-round", type=int, default=4096,
    help="Timesteps per tier per round",
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
  max_ticks = config["environment"]["max_ticks"]

  orch = MultiAgentOrchestrator(env_config)

  agents: Dict[int, TierAgent] = {}
  for tier_num, tier_env in orch.tier_envs.items():
    agents[tier_num] = TierAgent(tier_env, config)

  print(
    f"Training {len(agents)} tier agents, "
    f"{args.rounds} rounds x {args.steps_per_round} steps/tier"
  )
  for tier_num in sorted(agents.keys()):
    env = orch.tier_envs[tier_num]
    print(
      f"  Tier {tier_num}: {len(env.tier_items)} items, "
      f"action_space={env.action_space}"
    )

  tick, reward = evaluate_episode(orch, agents, max_ticks)
  print(f"\nPre-training eval: tick={tick}, reward={reward:.1f}\n")

  for rnd in range(args.rounds):
    for tier_num in sorted(agents.keys()):
      agent = agents[tier_num]
      agent.model.learn(
        total_timesteps=args.steps_per_round,
        reset_num_timesteps=False,
      )

    tick, reward = evaluate_episode(orch, agents, max_ticks)
    status = "DONE" if tick < max_ticks else "TRUNC"
    print(
      f"  Round {rnd + 1:>3}/{args.rounds} | "
      f"{status} tick={tick:>5} | reward={reward:>8.1f}"
    )

  print("\nFinal evaluation:")
  tick, reward = evaluate_episode(orch, agents, max_ticks)
  print(f"  tick={tick}, reward={reward:.1f}")

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  model_dir = config["training"].get("model_dir", "output/models")
  os.makedirs(model_dir, exist_ok=True)
  for tier_num, agent in agents.items():
    path = os.path.join(model_dir, f"tier{tier_num}_{timestamp}")
    agent.save(path)
  print(f"\nModels saved to {model_dir}/tier*_{timestamp}")


if __name__ == "__main__":
  main()
