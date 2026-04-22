#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.items import CraftingTree, ItemId, ITEM_NAME_TO_ID, CRAFTABLE_ITEM_IDS
from src.ga.ga_scheduler import GaScheduler
from src.ga.chromosome import Chromosome


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GA baseline for crafting game")
    parser.add_argument(
        "--config", default="config/ga.yaml", help="GA config path"
    )
    parser.add_argument(
        "--generations", type=int, default=None, help="Override n_generations"
    )
    parser.add_argument(
        "--output-dir", default="output/ga_results", help="Output directory"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)["ga"]

    tree = CraftingTree.from_yaml("config/crafting_tree.yaml")

    with open("config/targets.yaml", "r") as f:
        targets_data = yaml.safe_load(f)["targets"]
    targets = {ITEM_NAME_TO_ID[k.lower()]: v for k, v in targets_data.items()}

    scheduler = GaScheduler(config, tree, targets)
    hof = scheduler.run(
        n_generations=args.generations,
        output_dir=args.output_dir,
    )

    best_genes = scheduler.get_best_schedule(hof)
    if best_genes is not None:
        schedule = Chromosome.decode(best_genes)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.output_dir, f"ga_batch_schedule_{timestamp}.txt"
        )

        with open(output_path, "w") as f:
            f.write(
                f"{'tick_index':<12}{'elapsed_minutes':<18}"
                f"{'item_name':<20}{'batch_size_decided'}\n"
            )
            f.write("-" * 68 + "\n")
            for t, tick_actions in enumerate(schedule):
                for item_id_int, batch_size in tick_actions.items():
                    item_name = ItemId(item_id_int).name.lower()
                    f.write(
                        f"{t:<12}{t * 5:<18}{item_name:<20}{batch_size}\n"
                    )

        print(f"\nBest schedule exported to {output_path}")
        best = hof[0]
        print(
            f"Best fitness: cost={best.fitness.values[0]:.0f}, "
            f"time={best.fitness.values[1]:.0f} ticks, "
            f"waste={best.fitness.values[2]:.0f}"
        )
        print(f"Pareto front size: {len(hof)}")


if __name__ == "__main__":
    main()
