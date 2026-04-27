from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
from deap import algorithms, base, creator, tools

from src.core.items import CraftingTree, ItemId, NUM_CRAFTABLE
from src.ga.chromosome import Chromosome, MAX_TICKS
from src.ga.operators import GaOperators


if not hasattr(creator, "FitnessMulti"):
  creator.create(
    "FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0)
  )
if not hasattr(creator, "Individual"):
  creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)


class GaScheduler:
  def __init__(
    self,
    config: Dict[str, Any],
    crafting_tree: CraftingTree,
    targets: Dict[ItemId, int],
  ):
    self._config = config
    self._tree = crafting_tree
    self._targets = targets
    self._operators = GaOperators(crafting_tree, config)
    self._log_path = None

    self._toolbox = base.Toolbox()
    self._setup_toolbox()

  def _setup_toolbox(self) -> None:
    density = self._config.get("sparse_init_density", 0.05)
    max_init = self._config.get("max_init_batch", 10)

    def _init_individual():
      genes = Chromosome.create_random(density, max_init)
      ind = creator.Individual(genes)
      return ind

    self._toolbox.register("individual", _init_individual)
    self._toolbox.register(
      "population",
      tools.initRepeat,
      list,
      self._toolbox.individual,
    )

    self._toolbox.register("evaluate", self._evaluate)
    self._toolbox.register("mate", self._operators.cx_time_block)
    self._toolbox.register("mutate", self._operators.mutate)
    self._toolbox.register("select", tools.selNSGA2)

  def _evaluate(
    self, individual: np.ndarray
  ) -> tuple[float, float, float, float]:
    return Chromosome.evaluate(individual, self._tree, self._targets)

  def run(
    self,
    n_generations: int | None = None,
    output_dir: str = "output/ga_results",
  ) -> List:
    n_gen = n_generations or self._config.get("n_generations", 500)
    pop_size = self._config.get("population_size", 200)
    cx_prob = self._config.get("crossover_prob", 0.7)
    mut_prob = self._config.get("mutation_prob", 0.3)
    seed = self._config.get("seed", 42)

    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    self._log_path = os.path.join(output_dir, "ga_population_log.jsonl")

    pop = self._toolbox.population(n=pop_size)

    fitnesses = list(map(self._toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
      ind.fitness.values = fit

    hof = tools.ParetoFront()
    hof.update(pop)

    for gen in range(n_gen):
      offspring = tools.selTournamentDCD(pop, len(pop))
      offspring = [creator.Individual(ind.copy()) for ind in offspring]

      for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_prob:
          self._toolbox.mate(child1, child2)
          del child1.fitness.values
          del child2.fitness.values

      for mutant in offspring:
        if random.random() < mut_prob:
          self._toolbox.mutate(mutant)
          del mutant.fitness.values

      invalids = [ind for ind in offspring if not ind.fitness.valid]
      fitnesses = list(map(self._toolbox.evaluate, invalids))
      for ind, fit in zip(invalids, fitnesses):
        ind.fitness.values = fit

      pop = self._toolbox.select(pop + offspring, pop_size)
      hof.update(pop)

      self._log_generation(gen, pop, hof)

      if gen % 50 == 0:
        best = hof[0] if hof else None
        best_fit = best.fitness.values if best else None
        print(
          f"Gen {gen}/{n_gen} | "
          f"Pop: {len(pop)} | "
          f"Front: {len(hof)} | "
          f"Best: {best_fit}"
        )

    return list(hof)

  def _log_generation(
    self, gen: int, pop: List, hof: tools.ParetoFront
  ) -> None:
    if not self._log_path:
      return

    front_entries = []
    for ind in hof:
      front_entries.append({
        "cost": ind.fitness.values[0],
        "time": ind.fitness.values[1],
        "waste": ind.fitness.values[2],
        "cost_per_item": ind.fitness.values[3],
        "rank": 1,
      })

    fits = [ind.fitness.values for ind in pop]
    costs = [f[0] for f in fits]
    times = [f[1] for f in fits]
    wastes = [f[2] for f in fits]
    cpis = [f[3] for f in fits]

    entry = {
      "gen": gen,
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "pop_size": len(pop),
      "pareto_front_size": len(hof),
      "best_cost": min(costs) if costs else 0,
      "best_time": min(times) if times else 0,
      "best_waste": min(wastes) if wastes else 0,
      "best_cost_per_item": min(cpis) if cpis else 0,
      "avg_cost": sum(costs) / len(costs) if costs else 0,
      "avg_time": sum(times) / len(times) if times else 0,
      "avg_waste": sum(wastes) / len(wastes) if wastes else 0,
      "avg_cost_per_item": sum(cpis) / len(cpis) if cpis else 0,
      "front": front_entries[:10],
    }

    with open(self._log_path, "a") as f:
      f.write(json.dumps(entry) + "\n")

  def get_best_schedule(self, hof: List) -> np.ndarray | None:
    if not hof:
      return None
    return np.array(hof[0])
