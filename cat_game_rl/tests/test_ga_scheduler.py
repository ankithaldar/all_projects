from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.core.items import CraftingTree, ItemId
from src.ga.ga_scheduler import GaScheduler


@pytest.fixture
def ga_config():
    return {
        "population_size": 10,
        "n_generations": 2,
        "crossover_prob": 0.7,
        "mutation_prob": 0.3,
        "mut_batch_indpb": 0.02,
        "mut_time_shift_prob": 0.15,
        "sparse_init_density": 0.05,
        "max_init_batch": 10,
        "time_block_size": 48,
        "seed": 42,
    }


@pytest.fixture
def targets():
    return {ItemId.STRING: 5}


class TestGaScheduler:
    def test_run_completes(
        self,
        ga_config: dict,
        crafting_tree: CraftingTree,
        targets: dict,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = GaScheduler(ga_config, crafting_tree, targets)
            hof = scheduler.run(n_generations=2, output_dir=tmpdir)
            assert len(hof) > 0

    def test_pareto_front_non_dominated(
        self,
        ga_config: dict,
        crafting_tree: CraftingTree,
        targets: dict,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = GaScheduler(ga_config, crafting_tree, targets)
            hof = scheduler.run(n_generations=2, output_dir=tmpdir)

            for i, ind_a in enumerate(hof):
                for j, ind_b in enumerate(hof):
                    if i == j:
                        continue
                    dominated = all(
                        a <= b for a, b in zip(
                            ind_a.fitness.values, ind_b.fitness.values
                        )
                    ) and any(
                        a < b for a, b in zip(
                            ind_a.fitness.values, ind_b.fitness.values
                        )
                    )
                    assert not dominated, (
                        f"Individual {i} dominates {j} in Pareto front"
                    )

    def test_jsonl_log_created(
        self,
        ga_config: dict,
        crafting_tree: CraftingTree,
        targets: dict,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = GaScheduler(ga_config, crafting_tree, targets)
            scheduler.run(n_generations=2, output_dir=tmpdir)

            log_path = os.path.join(tmpdir, "ga_population_log.jsonl")
            assert os.path.exists(log_path)

            with open(log_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 2  # 2 generations

            entry = json.loads(lines[0])
            assert "gen" in entry
            assert "pareto_front_size" in entry
            assert "best_cost" in entry

    def test_get_best_schedule(
        self,
        ga_config: dict,
        crafting_tree: CraftingTree,
        targets: dict,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = GaScheduler(ga_config, crafting_tree, targets)
            hof = scheduler.run(n_generations=2, output_dir=tmpdir)
            best = scheduler.get_best_schedule(hof)
            assert best is not None
            assert best.shape == (2016, 19)

    def test_empty_hof(
        self,
        ga_config: dict,
        crafting_tree: CraftingTree,
        targets: dict,
    ):
        scheduler = GaScheduler(ga_config, crafting_tree, targets)
        assert scheduler.get_best_schedule([]) is None
