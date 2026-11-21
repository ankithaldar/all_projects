# Learning behavior and the curriculum check

This page records the training behavior actually implemented in this folder. It is a factual supplement to the main README (§6 and §9). If it disagrees with the code, the code wins.

## Spawn rules (`js/evolution.js`)

| Target generation | child 0 | children 1–19 | RNG draws |
| --- | --- | --- | --- |
| **51 only** (one-time full restart) | fresh `new NeuralNetwork(rng)` — the elite clone is deliberately dropped | 19 more fresh nets | 20 × 800 = **16 000** |
| every other generation, **including 101, 151, 201** | `best.clone()` exact, zero draws | `best.clone().mutate(0.25, 0.30, rng)` | mutation draws only |

- Constants: `POPULATION = 20`, `MUTATION_RATE = 0.25`, `MUTATION_SCALE = 0.30`, `MAX_GENERATIONS = 5000` (`js/evolution.js:120-127`).
- The 800 draws per fresh net are 400 Xavier-like weights × 2 draws per Box–Muller Gaussian; the 34 biases are zero-filled (`js/nn.js:172-176,251-263`). `NeuralNetwork.randomInit(rng)` and `new NeuralNetwork(rng)` take the same path (`js/nn.js:321-327`).
- The target generation is threaded explicitly at both call sites: `spawnNextGeneration` passes `this.generation + 1` (`js/evolution.js:317-327`); `resumeFromBest` passes `completedGeneration + 1` (`js/evolution.js:372-390`).
- Resume byte-identity comes from the validated uint32 `rngState` cursor: the constructor restores it with `setState` (`js/evolution.js:183-198`), then the same `_spawnChildren` routine runs on both the uninterrupted and resumed paths.

## Evaluation constants (`js/driver-sim.js:103-108`)

`SIM_DT = 1/120` s, `EVAL_CAP_S = 45` s, `STUCK_TIMEOUT_S = 12` s, `PROGRESS_ADVANCE_PX = 20` px, `LAP_BONUS_TRACK_LENGTHS = 2`, `LAP_BONUS_LAPS = 100`. Fitness is net signed odometer + `laps × 2 × track.length`; kill order is crash > finished > stuck > time-cap. Difficulty is display/audit only.

## Reproducing the 200-generation check

From `random_codes/0003_car_racer`:

```sh
node tests/f1-curriculum.mjs
```

The 9-leg suite (seed 4242) runs generations 1–50 on the default circuit and 51–250 on `Rotation.resolveGeneratedTrack(4242, 1, 2)` — hash `9a44a5c9dadd0597`, halfWidth `49.75`. It asserts gen-250 best fitness ≥ **38 970**, the mean best over gens 201–250 ≥ **50 000** and greater than the gens 51–100 mean, and at least 1 of the 20 trained drivers completing ≥1 clean lap on R1; a falsifier shows stationary and wall-hugging policies cannot. It also byte-checks that the restart boundary set is exactly `{51}` (101/151/201 stay exact-clone + mutants) and that resume is byte-identical at completed generations 50/100/150/200. Exit 0 prints `legs passed: 9 / 9` and `F1-CURRICULUM OK — all 9 legs passed`.

Determinism is per engine: the same seed on the same JS engine gives byte-identical populations; cross-engine last-ulp divergence is real.
