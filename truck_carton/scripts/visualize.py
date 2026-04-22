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
    PackingVisualizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualize packing results'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to saved model (random if omitted)',
    )
    parser.add_argument(
        '--stage', type=int, default=0,
        help='Curriculum stage (0-2)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed',
    )
    args = parser.parse_args()

    config = AppConfig()
    env = TruckCartonPackingEnv(
        config=config, curriculum_stage=args.stage
    )

    model = None
    if args.model:
        model = MaskablePPO.load(args.model)

    obs, info = env.reset(seed=args.seed)
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

    print(
        f'Placed {len(env._placed)}'
        f'/{len(env._episode.cartons)} cartons'
    )

    viz = PackingVisualizer()
    viz.render_all_trucks(
        env._episode, env._spaces, env._placed
    )
    plt.show()


if __name__ == '__main__':
    main()
