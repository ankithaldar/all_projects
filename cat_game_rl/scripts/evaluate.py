#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.items import ItemId, CRAFTABLE_ITEM_IDS
from src.env.crafting_env import CraftingEnv
from src.agent.masked_agent import MaskedAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument(
        "--model", required=True, help="Path to saved model (.zip)"
    )
    parser.add_argument(
        "--config", default="config/training.yaml", help="Training config path"
    )
    parser.add_argument(
        "--output", default=None, help="Output batch_schedule.txt path"
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
        "reward_weights": config.get("reward_weights"),
    }

    env = CraftingEnv(env_config)
    agent = MaskedAgent.load(args.model, env, config)

    obs, info = env.reset()
    schedule_rows = []
    total_reward = 0.0

    for tick_idx in range(config["environment"]["max_ticks"]):
        mask = env.action_masks()
        action = agent.predict(obs, action_masks=mask)

        for i, item_id_int in enumerate(CRAFTABLE_ITEM_IDS):
            batch_size = int(action[i])
            if batch_size > 0:
                schedule_rows.append({
                    "tick_index": tick_idx,
                    "elapsed_minutes": tick_idx * 5,
                    "item_name": ItemId(item_id_int).name.lower(),
                    "batch_size_decided": batch_size,
                })

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"output/batch_schedule_{timestamp}.txt"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"{'tick_index':<12}{'elapsed_minutes':<18}{'item_name':<20}{'batch_size_decided'}\n")
        f.write("-" * 68 + "\n")
        for row in schedule_rows:
            f.write(
                f"{row['tick_index']:<12}"
                f"{row['elapsed_minutes']:<18}"
                f"{row['item_name']:<20}"
                f"{row['batch_size_decided']}\n"
            )

    print(f"Total reward: {total_reward:.2f}")
    print(f"Schedule exported to {output_path}")
    print(f"Targets complete: {env.targets.is_complete()}")
    print(f"Final tick: {env.current_tick}")


if __name__ == "__main__":
    main()
