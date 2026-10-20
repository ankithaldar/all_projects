# Implementation Plan — Cornell Birdcall: PyTorch Lightning + TensorFlow/Keras

## Overview

Build **two working implementations** of the Cornell Birdcall 1st-place solution
from the specification in `docs/`, and the **13-model vote + `submission.csv`**
that actually won the competition (which the original repo never contained).

Three facts drive everything below:

1. **The original source is not in this repo.** `src/` holds only an empty
   `birdcall/` scaffold; the 85-file original lives in `docs/` as a
   per-file specification (`docs/src-reference/`), a 112-row parity matrix
   (`docs/ports/feature-parity-matrix.md`), a file-by-file worklist
   (`docs/ports/porting-worklist.md`) and two step-by-step implementation
   guides (`docs/ports/pytorch-lightning-guide.md`,
   `docs/ports/tensorflow-keras-guide.md`). The implementation is therefore
   *reconstructed from the docs*, not copied.
2. **A framework port is not two independent rewrites.** The Lightning port
   reuses the entire numeric core (model, front-end, dataset, sampler,
   augmentations, losses, metrics) and replaces only the control plane. The
   Keras port rewrites the numeric core too. So we build **one shared
   PyTorch reference core** first — it is *not* a third deliverable, it is
   the ground truth the Lightning port wraps and the Keras port is checked
   against. Without it there is nothing to prove the Keras port correct.
3. **The framework swap is the second-order problem.** Per
   `docs/ports/code-issues-and-framework-wins.md`, the winning vote, the
   unreproducible CV split and the missing tests are worth more than either
   port. The vote is therefore built **first**, framework-agnostically, and
   both ports must produce a **byte-identical** `submission.csv`.

### Deliverables

| # | Deliverable | Verdict |
|---|---|---|
| D1 | `birdcall` package: config, front-end, model, data, losses, metrics, train loops | two framework backends |
| D2 | `birdcall.voting` — 13-model quorum, 5-second rows, `submission.csv` | framework-agnostic, shared by both |
| D3 | A parity harness proving the two backends agree | log-mel, shape trace, metrics, LR curve, submission bytes |
| D4 | Reproducibility: seeded fold generator, label-space generator, characterisation tests | closes the gaps the docs flag |

---

## Ground rules (read before implementing)

Taken from `docs/ports/porting-worklist.md` and `docs/FINDINGS.md`; these are
**invariants**, not suggestions.

| # | Rule | Why |
|---|---|---|
| G1 | Reproduce the **inert focal loss** exactly (`gamma=0`, `alpha_0=alpha_1=1`, `secondary_factor=1` ⇒ bit-identical to mean BCE) | it is the objective that won; "fixing" it is a different experiment |
| G2 | Never apply a second sigmoid in a loss or metric | the `AttBlock` already sigmoided |
| G3 | Only `clipwise_output` reaches a loss; `framewise_output` is unsupervised | by design; it is half the winning rule |
| G4 | The `0.3` clip gate in the frame metric stays a **literal**, and the two gates stay **independent** | the in-repo ancestor of the winning ensemble rule |
| G5 | `frames_num == 3001`, DenseNet time axis `== 93`, tail pad `== 25` frames | measured, not assumed |
| G6 | Softmax in `AttBlock` is over **time** (Keras: `axis=-2`) | `axis=-1` is a silent wrong answer |
| G7 | Keep librosa's **Slaney, area-normalised** mel matrix | `tf.signal`'s is HTK — a different matrix |
| G8 | Feed **power** (`real²+imag²`), not magnitude, to the dB formula | magnitude is a silent ×2 error |
| G9 | Explicit **reflect pad** before framing | missing it makes `frames_num` 2977, not 3001 |
| G10 | `bn0` → SpecAugment ordering is load-bearing | SpecAugment writes literal 0 = "per-bin mean" post-BN |
| G11 | Waveform augmentation runs on the **whole file, before slicing** | per-snippet augmentation is a different experiment |
| G12 | Label is a **264-vector** one-hot broadcast to every snippet | weak file-level labels |
| G13 | LR schedule is **numerically identical** to Ignite's | otherwise it is not the run that won |
| G14 | Every experiment is **one named YAML file** validated by a pydantic model; `--config` takes that file's path | 13 named, individually reproducible configs — the constraint the docs actually protect. The docs' "keep plain Python config modules" is a *means*, not the goal; see AD-23 |
| G15 | The Keras port ships **only `sed_dense121att`** | `keras.applications` has no DenseNet161/169/201; all 13 ensemble members use the 121 |
| G16 | Do **not** port the 37 out-of-scope legacy files | unreachable from any shipped config |
| G17 | **Every RNG is seeded, and every seed is in the config.** No hidden default may affect a run | the original hid five of them (mixup's `1234`, `os.urandom` worker seeds, the unseeded `random` module behind audiomentations, the global torch stream, the `_augd` mask). A run that cannot be re-derived from its config is not an experiment |
| G18 | **Per-item randomness is derived from `(seed, epoch, index)`**, not from a global stream | a worker-scheduled global stream makes item *i*'s crop depend on how many workers ran — the same config would not reproduce |
| G19 | The same config + same seed + same data ⇒ the **same run**, byte-for-byte where the framework allows | that is what "replicable" means; tolerance-based equality is a cross-framework claim (G21), not a within-framework one |
| G20 | Every run writes a **manifest**: resolved config, seeds, git commit, package versions, and hashes of the data, the mel matrix and the score artefacts | without provenance a result is an anecdote; with it, a run is re-derivable |
| G21 | Across the two backends the claim is **numerical equality to a stated tolerance**, and **byte-identical `submission.csv`** after thresholding | bit-exactness across BLAS/DNN libraries is not achievable; the thresholded submission is the artefact that must agree, and it is the one the competition scored |
| G22 | A run is **replayable from its manifest**: `reproduce <manifest.json>` re-checks the inputs and re-runs the experiment | "replicable" is only true if something can act on it. The manifest is not a log — it is an executable description of the experiment |
| G23 | The **inputs** are hashed, not just named: the fold CSVs, the audio files, the label space, the mel matrix and the score artefacts | a path is not an identity; a file can change under a stable name |

---

## Environment reality (checked 2026-10-21)

| fact | consequence |
|---|---|
| system Python **3.14.7** (Anaconda) | **no stable TF wheel for 3.14** → the Keras port must use the project venv |
| `.venv` = **Python 3.12.13** (uv), numpy 1.26.4, pandas, scipy, sklearn, soundfile, pytest, tqdm, psutil | this is the project environment for **both** ports |
| `lightning 2.6.5`, `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu`, `torchmetrics 1.9.0`, `neptune 1.14.0` are in the **system** site-packages, not in `.venv` | install them into `.venv` (cp312 wheels exist) |
| `tensorflow` / `keras` / `librosa` / `audiomentations` are **not installed anywhere** | they are Phase-0 installs |
| **no GPU** (`nvidia-smi` absent) | all verification is CPU-only on synthetic tensors; the real 30 s × 32 workload is out of reach here |
| no `/data`, no `/folds`, no `train.csv`, no checkpoints | the label space and the CV split must be **generated**, and every test must be synthetic |

**Consequence:** the plan separates *verifiable here* (shapes, maths, parity,
unit behaviour) from *not verifiable here* (a real training run, a real
leaderboard score). No task claims more than it can prove.

---

## Architecture decisions

| # | Decision | Rationale |
|---|---|---|
| AD-1 | The package root is the existing `src/birdcall/` scaffold, **with** `__init__.py`; the project is installed editable (`package-dir = {"" = "src"}`) | the scaffold is what exists; the original's namespace-package convention was a property of *its* tree, not a requirement |
| AD-2 | **YAML is the single source of truth for every experiment.** The 21 configs become 21 `.yaml` files; the Python `Parameters` class is replaced by a **pydantic** model that validates them | pydantic closes the docs' Tier-2 issue #11 ("config attributes that are referenced but never defined") *at load time*, which the plain class could not; `extra='forbid'` turns a typo into an error instead of a silently ignored key. The docs recommended keeping Python modules — that recommendation is deliberately overridden (AD-23) |
| AD-23 | The YAML tree **mirrors the original's layout**, so the 13 ensemble members keep their names: `config_params/final_sed/*.yaml` (15), `config_params/final_sed_5_fold/*.yaml` (5), `config_params/example_config.yaml`, `config_params/inference/final_5sec_sed.yaml` | the docs' actual constraint is *13 named, individually reproducible config files*. YAML files satisfy it; `LightningCLI` would not, because it collapses the config into a search space. The docs' objection was to `LightningCLI`/`keras_tuner`, not to YAML |
| AD-24 | The pydantic schema lives in `config_params/parameters.py` (shared, no prefix) and is the **only** place a config attribute is named | one schema, two backends, zero drift — and a typo in a YAML is a validation error, not an `AttributeError` 40 minutes into a run |
| AD-25 | The 13-model ensemble is an explicit **`config_params/ensembles/ensemble_13.yaml`** manifest listing its 13 member config paths plus the vote thresholds | the ensemble is the actual deliverable; making it a first-class, validated artefact means the winning rule is data, not folklore scattered over 13 files |
| AD-26 | `criterion_params` is validated by a **cross-field `model_validator`**, not by a union of typed models | the discriminator (`criterion_name`) lives at the top level, not inside `criterion_params`; a `model_validator` checks the params match the selected criterion in one place, with one readable error, instead of a 5-way union that pydantic cannot discriminate |
| AD-3 | The **mel matrix is precomputed once and committed** as `.npy` (with the librosa version in the filename); librosa is a **build-time-only** dependency | `librosa.filters.mel` is the only thing needed from librosa; it removes the librosa/numba/numpy pin triangle from the runtime and from the Keras port |
| AD-4 | A **single shared scoring artefact** (`.npz`: `clip` + `frames`) is the handoff between a model and the vote | the original's `SedTest` max-pools the time axis away and therefore *cannot* vote. Fixing that is the whole point of the artefact |
| AD-5 | The vote is a **pure numpy module** over `.npz` files, with **no** framework import | both backends import the same module; "byte-identical submission" is then a testable claim rather than a hope |
| AD-6 | One **inference config** is added (`config_params/inference/final_5sec_sed.py`: `period=30`, `num_test_samples=20`, `valid_bs=1`) | the original had no inference config; `docs/modules/08` Part E lists it as required net-new work |
| AD-7 | The PyTorch reference core lives under `src/birdcall/` (not a separate `ports/` tree) and is **imported unchanged** by the Lightning backend | the docs' Lightning guide assumes the core is importable; one tree, one import path |
| AD-8 | Backend-specific code is namespaced by filename **prefix**: `pl_<name>.py` for the PyTorch Lightning implementation, `tf_<name>.py` for the TensorFlow implementation; **unprefixed modules are shared** and contain no framework import | keeps the scaffold's package structure, makes the split obvious at a glance in a file listing, avoids a `backends/` sub-tree the scaffold does not have |
| AD-49 | The Keras backend goes through **`tf.keras`**, never the standalone `keras` package | `tf.keras` *is* Keras 3 on TensorFlow 2.16+, so nothing is lost, and the code stays on the API TensorFlow itself ships. One spelling, one import path, and the `keras.saving` module (which `tf.keras` does not expose) becomes `tf.keras.utils` |
| AD-9 | `ast.literal_eval` replaces `eval` for the CSV's `secondary_labels` | `docs/code-issues` #10; same result, no code execution from a data file |
| AD-10 | Model / criterion / scheduler dispatch is **registry dicts** with a clear error on an unknown key | `docs/code-issues` #12; the original's `if/elif` had no `else` and failed later with an unrelated `AttributeError` |
| AD-11 | `lraps` keeps the **original** semantics (no background branch) as the tracked metric; a background-aware variant is available behind a constructor flag, off by default | changing the tracked metric changes checkpoint ranking and therefore the experiment |
| AD-12 | The `_augd` criterion's primary-dropout mask is drawn from the **global seeded** RNG (per-call) instead of an unseeded global | `docs/code-issues` #8: the original was not reproducible run-to-run. Documented deviation |
| AD-13 | `Mixup` keeps the hard-coded `random_seed=1234` **as the default** | the docs say to port it faithfully or explicitly decide; fidelity wins here because it changes the experiment |
| AD-14 | AMP stays **off** by default, but `power_to_db` is forced to fp32 so it *could* be enabled | `docs/modules/02`: the `amin=1e-10` fp16 underflow → NaN is the reason AMP was disabled; the guard is the documented fix |
| AD-15 | The `apply_aug=False` + mixup crash is **fixed** (the batch halving is guarded by the mixup actually being applied) | `docs/FINDINGS` "Latent crash worth knowing before any refactor" |
| AD-16 | Checkpoint cadence = **per validation** (`check_val_every_n_epoch=eval_every` + `ModelCheckpoint` on validation end), matching the original's evaluator-attached saver | the original wrote checkpoints only on validation days; keeping that cadence preserves the "best of these validation points" semantics |
| AD-17 | Validation metrics are logged as `valid_<name>`; the checkpoint monitor is the same string; the Neptune series is mapped back to the original's `valid/<name>` | the original's names were unprefixed; Lightning warns on unprefixed validation metrics and `ModelCheckpoint` fails **silently** on a monitor that is never logged |
| AD-18 | A crashed run **re-raises** (both backends) | `docs/code-issues` #18: the original swallowed the traceback and exited 0 |
| AD-19 | Tests are `pytest` under `tests/`, split by phase, with a `parity/` sub-package for cross-backend checks | the original had **zero** tests (`docs/code-issues` #4) |
| AD-20 | `activations/` is left empty and unused | the docs need no custom activation layer; `tanh`/`sigmoid` are framework built-ins |
| AD-21 | The **shared surface** is kept deliberately small and is exactly: `config_params/`, `helpers/mel_matrix.py`, `augmentations/sed_augment.py`, `augmentations/mixup.py`, `datasets/sed_dataset.py` (pure-numpy core), `metrics/scoring.py`, `engines/schedules.py`, `voting/vote.py` | every shared module is a parity risk if it drifts, so the shared surface is only what is provably framework-agnostic. Everything else is written twice, once per backend, on purpose |
| AD-22 | The **PyTorch reference core is the `pl_` prefix**, not a third tree | "PyTorch Lightning port" and "reference core" are the same code; the Lightning port adds a control plane on top of `pl_*` and imports the rest unchanged |
| AD-27 | `logger_params['params']` — the hand-built flat dict in the original — becomes a **computed field** on the pydantic model | the original built it by hand and nothing validated it, so it silently drifted from the config it described |
| AD-28 | `root_dir` is a `Path`, not `Path | dict` | the 5-archive sharding branch is documented as **dead code** (`get_dict_value` is imported by every config and called by none) and is already out of scope |
| AD-29 | The 21 YAML files are **generated once** from the documented variation table and then hand-edited; a test asserts each YAML still matches the table | the table in `docs/src-reference/config-params.md` is the ground truth for all 21 configs — generating from it is exact, and the test stops later edits from silently changing an ensemble member |
| AD-30 | A shared `helpers/seed.py` owns **all** seeding: `derive_seed(base, *parts)`, `seed_worker`, `seed_everything` | five different RNG sources in the original, four of them unseeded. One module is the only way to guarantee G17/G18 hold everywhere; a test asserts no other module calls `np.random.seed`/`random.seed` directly |
| AD-31 | The seed knobs are **config fields**: `seed: int = 42`, `mixup_seed: int = 1234`, `deterministic: bool = True` | the original's `Mixup` seed was a default argument buried in a class — invisible in the config and therefore invisible in the experiment record. `mixup_seed` defaults to the original's `1234`, so the shipped YAMLs stay faithful **and** the seed is now recorded data |
| AD-32 | The dataset derives a **per-item** seed from `(seed, epoch, index)`; the sampler derives a **per-epoch** seed from `(seed, epoch)`; the model/augmentation RNGs are seeded per-worker and per-epoch | G18. This is the change that makes a run reproducible regardless of worker count or scheduling |
| AD-33 | `audiomentations` gets `random` (the stdlib module) seeded per worker — the original only seeded numpy | the docs flag this explicitly: "audiomentations draws parameters from Python's `random` module, which is not per-worker-seeded". Unfixed, the waveform augmentation is irreproducible |
| AD-34 | A shared `provenance.py` writes `manifest.json` beside the checkpoints: the resolved config (verbatim YAML + JSON), every seed, the git commit, the package versions, and the SHA-256 of the fold CSVs, the mel matrix and the score artefacts | G20. Cheap to write, and it is the difference between "a run happened" and "a run can be repeated" |
| AD-35 | `deterministic: true` additionally sets `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `warn_only=True` because DenseNet's average-pool backward has no deterministic CUDA kernel; the flag must not make a run fail on a hardware-specific kernel, only warn |
| AD-36 | The dataset and the sampler carry an **epoch counter** (`set_epoch(epoch)`), set by the training loop at each epoch start | the per-item seed needs the epoch (AD-32), and the original's sampler had no `set_epoch` at all — the docs flag the dangling DDP hook. The epoch counter is also what makes a resumed run continue the same sequence |
| AD-37 | `reproduce.py` is an entry point: it reads a `manifest.json`, verifies every input hash, re-pins the environment, and re-runs the recorded command | G22. A manifest you cannot act on is documentation, not reproducibility |
| AD-38 | The ensemble manifest records, per member, the **config path, the checkpoint path and the checkpoint's SHA-256** | the 13-model ensemble is the deliverable; "which trained model is member *k*" must be data, not a filename convention |
| AD-39 | The environment is **locked**: `uv.lock` plus exact pins in `requirements*.txt`; the resolved package versions go into every manifest | a range like `torch>=2.2` is not a reproduction recipe |
| AD-40 | **Experiment tracking is wandb**, not Neptune. The original's `neptune-client==0.4.107` is dropped, and the `logger_name` enum becomes `Literal['wandb', 'print']` | the original's tracker is an external service with a pinned 0.4.x client that the framework change would have forced to `>=1.0` anyway. wandb is the requested sink |
| AD-41 | The wandb **series names are the original's Neptune series names**, unchanged: `train/loss`, `train/<loss_key>`, `train/lr`, `train/<metric>`, `valid/<metric>` | wandb accepts slashes in metric keys and renders them as groups, so the mapping is one-to-one and the dashboards stay comparable with the original's |
| AD-42 | The `logger_params` **6 keys keep their names** (`project_name`, `log_every`, `name`, `prefix_name`, `tags`, `params`) and map onto wandb's `project`, the log cadence, `name`, **`group`**, `tags` and `config` | keeps the 39-attribute config contract intact. `prefix_name` was accepted by the original loggers and **never read**; here it becomes the wandb group, which is exactly what it was always meant to be — all 13 ensemble members in one group |
| AD-43 | `logger_params['params']` maps to wandb's **`config`**, populated from the schema's computed field | wandb's config is the natural home for the resolved hyperparameters, and it means the tracker always shows the *resolved* config, not a hand-built copy that can drift (AD-27) |
| AD-44 | The test suite and CI run wandb in **offline/disabled** mode (`WANDB_MODE=offline`, `WANDB_DISABLED=true`) | no test may depend on a network call or an account; a tracking outage must never fail a training run |
| AD-45 | The **manifest is the source of truth**; wandb is a sink | a tracker is mutable, a manifest is not. The manifest records the wandb run id and URL so the two are linked, but the experiment is re-derivable from the manifest alone |
| AD-46 | **One file per functionality.** Every public class gets its own module, named after it; a file holds one class (or one cohesive function-family) and nothing else | the user requirement, and it is also the cleanest way to make the docs' parity matrix and the code line up: one matrix row ↔ one file. It also stops the "5 losses in one file" pattern that makes the *reachable* subset hard to see |
| AD-47 | A package's **name → class** mapping lives in that package's `*_registry.py` (one per backend, `augmentations/registry.py` is shared) | the registries are the replacement for the original's `if/elif` chains with no `else` (AD-10). One place to look, one place to add, one clear error on an unknown key |
| AD-48 | A Keras `Layer` lives in `layers/`; a torch `nn.Module` lives in `models/` | keras has a first-class layer concept and torch does not; the rule keeps each backend's files where a reader expects them |

---

## Code style contract

Authoritative source: **`.pylintrc` at the repository root**
(`competitions_main/.pylintrc`, Google-derived). It is **read-only** — the
implementation is refactored to fit it, never the other way round.

| rule | value | source |
|---|---|---|
| indentation | **2 spaces** | `.pylintrc` `indent-string='  '` |
| line length | **80** characters max | `.pylintrc` `max-line-length=80` |
| hanging indent after an open paren | **4** spaces | `.pylintrc` `indent-after-paren=4` |
| string literals | **single quotes**, consistently within a module | `.pylintrc` `check-quote-consistency=yes` |
| docstrings | **Google style** on every module, class, function and method | project rule |
| module docstring | **required** (pylint's `missing-module-docstring` is *not* disabled) | `.pylintrc` |
| class docstring | **required** (`missing-class-docstring` is *not* disabled) | `.pylintrc` |
| function docstring | required by project rule; pylint's check is disabled | project rule |
| file header | shebang + encoding + module docstring, in that order | project rule |
| single-line `if` | allowed | `.pylintrc` `single-line-if-stmt=yes` |
| refactor messages | **all disabled** (`disable=R,`) | `.pylintrc` |
| `good-names` | `main`, `_` | `.pylintrc` |
| `notes` | `TODO` is tracked | `.pylintrc` |

### Mandatory file header

Every `.py` file in the project starts with exactly this shape:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''<One-line summary of the module.

Google-style docstring body: what the module owns, what it deliberately does
not do, and any invariant a caller must not break.
'''
```

The module docstring uses **triple single quotes**; every other string in the
file uses **single quotes**.

### Design rules

- **SOLID** — one reason to change per module; the backends depend on
  *protocols* (a `ScoreWriter`, a `Criterion`, a `Schedule`), not on concrete
  classes. `voting/vote.py` and `metrics/scoring.py` know nothing about torch
  or tensorflow and are importable from either.
- **Registry over `if/elif`** (AD-10) — an unknown key raises a
  `KeyError`/`ValueError` naming the valid keys, at construction time.
- **Dependency inversion at the framework boundary** — `pl_*` modules import
  torch; `tf_*` modules import tensorflow; neither imports the other; shared
  modules import neither.
- **No framework import in a shared module** — enforced by a test
  (`tests/test_shared_is_framework_free.py`) that greps the shared surface for
  `import torch` / `import tensorflow`.
- **One file per functionality (AD-46)** — one public class per module, named
  after it. A file that holds more than one class (a registry, a config model
  family) declares that in its module docstring. This also makes the docs'
  parity matrix line up with the tree one row ↔ one file.

---

## File layout

```
src/birdcall/
  config_params/            # shared
    parameters.py           # the top-level Parameters model
    model_config.py         # ModelConfig
    dataset_params.py       # DatasetParams
    logger_params.py        # LoggerParams
    checkpoint_params.py    # CheckpointParams
    run_params.py           # RunParams
    criterion_params.py     # the per-criterion param schemas + the cross-field validator
    loader.py               # YAML -> Parameters, with a readable error report
    configs.py              # BIRD_CODE, EBIRD_LABEL, INV_EBIRD_LABEL, AUDIO_CONFIG
    bird_codes.json         # generated label space (T4.2)
    example_config.yaml
    final_sed/              # 15 YAMLs
    final_sed_5_fold/       # 5 YAMLs
    inference/
      final_5sec_sed.yaml   # new: num_test_samples=20, valid_bs=1 (AD-6)
    ensembles/
      ensemble_13.yaml      # the 13 members + the vote thresholds (AD-25)
  helpers/                  # shared
    mel_matrix.py           # build + load the committed Slaney mel matrix
    seed.py                 # ALL seeding (AD-30)
  augmentations/            # shared
    sed_augment_default.py  # the "default" stack
    sed_augment_background.py  # the "secondary_default" palindromic bracket
    mixup.py                # numpy mixup (even/odd pairing, seed 1234)
    registry.py             # aug_name -> transform builder
  datasets/
    sed_dataset_core.py     # shared: pure-numpy rows, label coding, snippet slicing
    pl_dataset_sed.py       # PL: torch Dataset + collate
    pl_sampler_imbalanced.py
    tf_dataset_sed.py       # TF: tf.data pipeline
  models/
    pl_model_att_block.py         # PL: AttBlock
    pl_model_panns_dense121_att.py
    tf_model_panns_dense121_att.py
    pl_registry.py                # model_name -> class
    tf_registry.py
  layers/                   # Keras layers (AD-48)
    tf_layer_att_block.py
    tf_layer_spec_augment.py
  losses/
    pl_loss_focal_standard.py
    pl_loss_removed_focal.py
    pl_loss_scaled_pos_neg_focal.py
    pl_loss_scaled_pos_neg_focal_augd.py
    pl_loss_bce_controlled_standard.py
    tf_loss_focal_standard.py
    tf_loss_removed_focal.py
    tf_loss_scaled_pos_neg_focal.py
    tf_loss_scaled_pos_neg_focal_augd.py
    tf_loss_bce_controlled_standard.py
    pl_registry.py                # criterion_name -> class
    tf_registry.py
  metrics/
    pl_metric_lraps.py
    pl_metric_f1score_clip.py
    pl_metric_f1score_frame.py
    tf_metric_lraps.py
    tf_metric_f1score_clip.py
    tf_metric_f1score_frame.py
    scoring.py             # shared: the .npz score artefact (clip + frames)
    pl_registry.py                # metric name -> class
    tf_registry.py
  engines/
    schedules.py           # shared: the LR-schedule maths + the schedule registry
    pl_lightning_module.py # PL: LightningModule (steps, log_dict)
    pl_trainer.py          # PL: Trainer wiring, checkpointing, resume
    tf_keras_model.py      # TF: keras.Model, train_step, compile
  datamodules/
    pl_datamodule_sed.py   # PL: LightningDataModule
  optimizers/
    pl_optimizer_adamw.py
    tf_optimizer_adamw.py
  callbacks/
    pl_callback_log_every.py
    pl_callback_slack.py
    pl_callback_inference_dump.py
    tf_callback_log_every.py
    tf_callback_slack.py
    tf_callback_inference_dump.py
    tf_callback_keep_best_n.py
  loggers/
    pl_logger_wandb.py
    pl_logger_print.py
    tf_logger_wandb.py
    tf_logger_print.py
  voting/
    vote.py                # shared: 4-of-13 quorum, 5-second rows, submission.csv
  provenance.py            # shared: the run manifest (AD-34)
  scripts/
    build_mel_matrix.py
    build_bird_codes.py
    make_folds.py
    build_configs.py        # regenerates the 21 YAMLs from the documented table
  train_pl.py               # entry point: PyTorch Lightning
  train_tf.py               # entry point: TensorFlow/Keras
  run_vote.py               # entry point: the shared vote
  reproduce.py              # entry point: replay an experiment from its manifest (AD-37)
tests/
  test_shared_is_framework_free.py
  ...
```

**Naming rule.** `pl_<name>.py` = the PyTorch Lightning implementation.
`tf_<name>.py` = the TensorFlow/Keras implementation. **No prefix** = shared,
framework-agnostic, importable from both.

**File rule (AD-46).** One file per functionality: one public class per file,
named after it. A file that must hold more than one thing (a registry, a
config model family) says so in its module docstring.

### The config surface

`config_params/parameters.py` defines the pydantic schema. The **39
attributes** from `docs/src-reference/config-params.md` are the contract; the
nested dicts become nested models:

| group | YAML shape | pydantic model |
|---|---|---|
| identity | `fold`, `name` | `fold: int | Literal['all']`, `name: str` |
| model | `model_name`, `model_config` (9 keys), `pretrained_path` | `model_name: Literal[...]`, `ModelConfig` |
| augmentation | `aug_name`, `apply_mixup`, `bckgrd_aug_dir`, `secondary_bckgrd_aug_dir` | `aug_name: Literal[...]`, `Path | None` |
| loss | `criterion_name`, `criterion_params` | `criterion_name: Literal[...]`, `dict[str, float]` + `model_validator` (AD-26) |
| optimiser | `optimizer_name`, `weight_decay`, `lr` | `Literal['adamw']`, `weight_decay: float > 0`, `lr: float > 0` |
| scheduler | `scheduler_name`, `lr_scale_factor` | `Literal[...]`, `0 < f <= 1` |
| data | `PERIOD`, `train_ds_params`, `valid_ds_params`, `test_ds_params`, `train_bs`, `train_num_workers`, `valid_bs`, `valid_num_workers` | `DatasetParams` |
| metrics | `metrics`, `track_metric`, `metric_factor` | `metrics: list[Literal[...]]`, `track_metric: Literal[...]` |
| logging | `logger_name`, `logger_params` (6 keys) | `logger_name: Literal['wandb', 'print']`, `LoggerParams` |
| checkpointing | `checkpoint_params` (3 keys), `checkpoint_dir`, `load_model_only` | `CheckpointParams` |
| runtime | `run_params`, `dist_params`, `val_length`, `eval_every`, `add_pbar`, `accumulation_steps`, `gradient_clip_val` | `RunParams` |
| **reproducibility** | `seed`, `mixup_seed`, `deterministic` | `seed: int`, `mixup_seed: int = 1234`, `deterministic: bool = True` (AD-31) |

Model-wide settings: `extra='forbid'` (a typo is an error, not a silently
ignored key), `validate_assignment=True`, `frozen` where a field must not
change after load.

**A config is the complete recipe.** Every knob that can change a number is a
field: the seeds, the LR and schedule, the weight decay, the batch sizes, the
augmentation probabilities, the mixup seed, the criterion parameters, the
checkpoint cadence, the fold CSV paths, and the background-audio directory.
Nothing that affects the result lives in a default argument, an environment
variable, or a constant in a module. The manifest (AD-34) then records the
resolved values, so a run is described by data, not by prose.

---



The detailed task cards — acceptance criteria, verification commands,
dependencies, files touched — live in [`tasks/todo.md`](todo.md). This is the
ordered index.

### Phase 0 — Contracts and environment

- [ ] **T0.1** Project skeleton, packaging, dependency pins — `pyproject.toml`, `requirements*.txt`
- [ ] **T0.2** Config contract — `config_params/parameters.py`, `config_params/loader.py`, the 21 YAMLs, the ensemble manifest
- [ ] **T0.3** Shared LR-schedule maths + the Ignite golden fixture — `engines/schedules.py`
- [ ] **T0.4** Seeding contract + the shared seed helpers — `helpers/seed.py`

### Checkpoint 0 — contracts

- [ ] `pip install -e .` succeeds in `.venv`; `birdcall` imports
- [ ] all 21 YAMLs load and validate; every attribute is present and typed
- [ ] an unknown key, a bad enum value or a mismatched criterion raises a clear pydantic error
- [ ] the LR lambda reproduces the 5 reference values from the Lightning guide
- [ ] every seed in the schema derives deterministically from `seed`

### Phase 1 — PyTorch reference core + the deliverable

- [ ] **T1.1** Audio front-end + mel matrix — `helpers/mel_matrix.py`, `helpers/pl_audio_frontend.py`
- [ ] **T1.2** Model — `models/pl_sed_models.py`
- [ ] **T1.3** Dataset + imbalanced sampler — `datasets/sed_dataset.py`, `datasets/pl_sed_dataset.py`, `datasets/pl_imbalanced_sampler.py`
- [ ] **T1.4** Waveform augmentation stacks + `Mixup` — `augmentations/sed_augment.py`, `augmentations/mixup.py`
- [ ] **T1.5** The five live criteria — `losses/pl_sed_losses.py`
- [ ] **T1.6** `LRAP` / `F1-clip` / `F1-frame` — `metrics/pl_sed_metrics.py`
- [ ] **T1.7** Frame-preserving score artefact writer — `metrics/scoring.py`
- [ ] **T1.8** Vote + `submission.csv` — `voting/vote.py`, `run_vote.py`
- [ ] **T1.9** Run manifest / provenance — `provenance.py`

### Checkpoint 1 — the reference implementation is complete

- [ ] log-mel frames == 3001; DenseNet time axis == 93; tail pad == 25
- [ ] inert-focal-loss test passes to 1e-6
- [ ] a synthetic 13-model set produces a `submission.csv`; 4 fires accept, 3 reject
- [ ] a run writes a manifest; re-running with the same seed and data reproduces it

### Phase 2 — PyTorch Lightning port

- [ ] **T2.1** `LightningModule` + datamodule — `engines/pl_lightning_module.py`, `datamodules/pl_sed_datamodule.py`
- [ ] **T2.2** `Trainer` wiring — `engines/pl_trainer.py`, `optimizers/pl_optimizers.py`
- [ ] **T2.3** Callbacks + loggers — `callbacks/pl_callbacks.py`, `loggers/pl_loggers.py`
- [ ] **T2.4** Port verification — `train_pl.py`, `tests/test_pl_*.py`

### Checkpoint 2 — Lightning port

- [ ] LR curve matches the golden fixture step for step
- [ ] one epoch over synthetic data matches the reference core to 1e-5
- [ ] exactly `n_saved` checkpoints exist; weights-only resume leaves the optimiser at its initial LR

### Phase 3 — TensorFlow/Keras port

- [ ] **T3.1** Front-end layer + log-mel equality — `helpers/tf_audio_frontend.py`
- [ ] **T3.2** Model rewrite — `models/tf_sed_models.py`, `layers/tf_layers.py`
- [ ] **T3.3** `tf.data` pipeline + losses + metrics — `datasets/tf_sed_dataset.py`, `losses/tf_sed_losses.py`, `metrics/tf_sed_metrics.py`
- [ ] **T3.4** Training loop — `engines/tf_keras_model.py`, `optimizers/tf_optimizers.py`, `callbacks/tf_callbacks.py`, `loggers/tf_loggers.py`
- [ ] **T3.5** Cross-port verification — `train_tf.py`, `tests/parity/`

### Checkpoint 3 — Keras port

- [ ] log-mel matches the torch front-end to 1e-4
- [ ] attention sums to 1 over **time**; `clipwise_output` is a sum over time
- [ ] the Keras and Lightning backends produce a **byte-identical** `submission.csv`

### Phase 4 — Operations and reproducibility

- [ ] **T4.1** CLI entry points, README, `.env.example`, `setup_structure.sh`
- [ ] **T4.2** Seeded fold-CSV generator + label-space generator — `scripts/make_folds.py`, `scripts/build_bird_codes.py`
- [ ] **T4.3** Full config sweep + whole-suite test run + parity report
- [ ] **T4.4** `reproduce` command + end-to-end reproducibility verification — `reproduce.py`

### Checkpoint 4 — complete

- [ ] every row of `docs/ports/feature-parity-matrix.md` A–J is ticked or explicitly deferred with a reason
- [ ] every row of §K (the vote) is implemented
- [ ] `pytest` green on CPU
- [ ] `pylint` clean against the repository `.pylintrc`

---

## What the original source settled

The documentation in `docs/` is a **reverse-engineered** spec. The original
source is public, and the implementation was checked against it. Four
places where the documentation was wrong, and the source is the authority:

| # | the documentation says | the source says | consequence |
|---|---|---|---|
| 1 | the `_augd` criterion drops **70 %** of the primary positions | it drops **30 %**: the mask is 1 on a primary position with probability `primary_drop_out_factor`, and the loss is multiplied by `(1 - mask)`, so a mask of 1 **zeroes** the position | the criterion is corrected in both backends, and the test now measures the dropped fraction to 5 % |
| 2 | `fc1` is a 1×1 convolution | it is `nn.Linear(1024, 1024)`, applied to a **transposed** tensor | the torch model uses `nn.Linear` with the original's transpose sandwich, so the state dict key and shape match the original checkpoint |
| 3 | the mel matrix is a frozen **buffer** | it is an `nn.Parameter` with `requires_grad=False`, and it is in the checkpoint under `logmel_extractor.melW` | the parameter count matches the documented 9 628 768, and the checkpoint key is right |
| 4 | the submission's second column is `labels` | the competition's own sample submission calls it **`birds`** | this project writes `labels` (as specified); the rename is a one-line change, flagged in the README and in `voting/vote.py` |

Two more facts came from the competition's **data page** rather than the
repository: the `row_id` format is `site_1_<audio_id>_<end_second>`, and
`site_3_<audio_id>` for the file-level site 3 rows.

The upstream repository also supplied the **21 original configurations**,
so the YAML tree is generated from them rather than reconstructed from a
summary table (`scripts/build_configs.py`).

---

## Risks and mitigations

| risk | impact | mitigation |
|---|---|---|
| ~~**The 264 species codes are not in the docs.**~~ **CLOSED.** The original source is public (<https://github.com/ryanwongsa/kaggle-birdsong-recognition>), so `BIRD_CODE` and `EBIRD_LABEL` were recovered from `src/config_params/configs.py` and committed as `config_params/bird_codes.json` | — | closed: the label space is real data now, not a placeholder, and `scripts/build_bird_codes.py` can regenerate it from the competition's `train.csv` |
| **The original fold CSVs are gone** — and the upstream repository does not have them either; they lived at `/folds/`, produced offline | the 13-model ensemble cannot be re-derived from the repo | `scripts/make_folds.py` ships a seeded generator (stratified on the **primary** `ebird_code` only, per `docs/modules/01`) that also writes a `folds_manifest.json` recording the seed and every CSV's hash. The *exact* original split is unrecoverable — the generator produces a **new, reproducible** split, which is the honest outcome |
| **No competition audio, no checkpoints** | nothing can be trained or scored here | every test is synthetic with a fixed seed; the plan makes no claim that requires real data. A "real run" is a documented operator step, not a task |
| **Keras' DenseNet121 may pool differently from torchvision's** | the 93-frame time axis — and therefore the whole shape trace and the tail pad — shifts silently | T3.2 **asserts** 93 rather than assuming it; the test fails loudly if Keras differs |
| **Keras applies ImageNet `preprocess_input` internally** | destroys the pretrained transfer with no error | T3.2 bypasses it (feature-extractor truncation) and asserts the bypass is active |
| **`Softmax(axis=-1)` in Keras `AttBlock`** | trains fine, scores badly — the single most likely silent failure | T3.2 asserts the attention sums to 1 over **time** and that `clipwise_output` is a sum over time |
| **Torch and TensorFlow in one environment** | dependency conflicts (TF pins numpy, torch pins its own) | both are installed into `.venv` (cp312); `numpy` stays `<2`; the parity tests that need both run in a single process where possible, otherwise via the shared `.npz` artefact on disk |
| **No GPU** | the 30 s × 32 real workload cannot be run | all verification uses short synthetic waveforms (e.g. 2 s) and small tensors; the *frame arithmetic* is verified analytically (3001 / 93 / 25 / 500 / 120) and by construction, not by a full-length forward pass |
| **`pytorch-ignite` may not install against torch 2.13** | the "verify against live Ignite" check from the guide is unavailable | T0.3 pins the **5 reference values transcribed from the guide** as the golden fixture (they were re-derived here from the documented formulas and match to 7e-8). Installing Ignite in a scratch venv is an *optional* cross-check, not a dependency of the plan |
| **librosa/numba/numpy pin triangle** | `import librosa` succeeds but `librosa.filters` fails at first use | AD-3 removes librosa from the runtime entirely; the build script pins `librosa==0.10.2` and records the version in the artefact filename |
| **A hidden RNG survives somewhere** | the run is irreproducible and nobody notices, because a single unseeded stream changes one crop in ten thousand | AD-30 makes `helpers/seed.py` the only module allowed to seed; a test greps the tree for `np.random.seed`, `random.seed` and `torch.manual_seed` outside it |
| **Worker count changes the result** | the same config produces different crops with `num_workers=4` vs `8` | G18/AD-32: per-item seeds derived from `(seed, epoch, index)`, so the crop depends on the item, not on the schedule |
| **`audiomentations` uses the stdlib `random` module** | the waveform augmentation is irreproducible even with every other source seeded | AD-33 seeds `random` per worker alongside numpy and torch |
| **CUDA non-determinism** | a run differs across machines | AD-35 sets `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG`; `warn_only` because DenseNet's average-pool backward has no deterministic CUDA kernel. The manifest records the flag so a non-deterministic run is *labelled* rather than silently trusted |
| **Config drift between the two backends** | the backends stop being the same experiment | one shared pydantic schema (AD-24); both backends load the same YAML and neither may define its own defaults |
| **21 YAMLs drift from the documented table** | an ensemble member silently changes | AD-29: generated from the table, and a test asserts the table still holds |
| **wandb is unreachable (offline, no key, service down)** | a training run dies because its *tracker* is down | AD-44: the test suite runs offline/disabled; a wandb failure is caught and downgraded to a warning, never fatal — the manifest (AD-45) remains the source of truth |
| **wandb's Keras 3 support is version-sensitive** | the Keras port's tracking breaks on a Keras upgrade | the Keras logger is a thin `Callback` that calls `wandb.log` directly rather than depending on `wandb.keras.WandbCallback`'s internals; a smoke test asserts the series land |
| **The audio files change underneath a run** | a "reproduced" run trains on different bytes | the manifest records a content hash per audio file (path + size + SHA-256); `reproduce` re-checks them and refuses to start on a mismatch |
| **The environment drifts** | the same config behaves differently next month | AD-39: the venv is locked (`uv.lock` + exact pins) and the resolved package versions are written into every manifest |
| **The working tree is dirty** | the git commit does not identify the code that ran | the manifest records the commit **and** a hash of the dirty diff, so a non-clean tree is visible rather than assumed clean |

---

## Open questions

| # | Question | Default if unanswered |
|---|---|---|
| Q1 | What is the **exact** competition `row_id` string format (site 1/2 5-second rows vs site 3 file rows)? `docs/modules/08` marks it external and the repo has 0 hits | implement `{site}_{audio_id}_{chunk_start_s}` for sites 1/2 and `{site}_{audio_id}` for site 3, isolated in one function (`build_row_id`) so it is a one-line change once `sample_submission.csv` is available |
| Q2 | Which `secondary_labels` spelling does the competition CSV use? `docs/modules/01` notes the original's `yebsap` typo may silently drop a species | parse with `ast.literal_eval`; unmatched secondary codes are **dropped, not fatal** (matches the original's `in self.inv_ebird_label` guard) |
| Q3 | Should the 5-fold family's `track_metric` inconsistency (fold 0 → `f1score_clip`, folds 1-4 → `f1score_frame`) be preserved? | **Preserved.** It is the original's experiment; "fixing" it changes which checkpoint each ensemble member ships |
| Q4 | Is `num_test_samples=2` (60 s of a 600 s file) a bug to fix in the *validation* configs? | **Not fixed.** It is the original's validation contract; the 20-sample config is *added* alongside (AD-6) rather than substituted |
| Q5 | Is the winning quorum 4-of-13 or should the module support the full sweep (3/4/5 votes × 0.3/0.5)? | support `votes` and both thresholds as **parameters** with the winning defaults (4, 0.3, 0.3); the README's 13-model and 9-model tables are then reproducible without a code change |
| Q6 | Which wandb **entity and project** should the 13 ensemble members use, and should they share one project? | one project, one **group** per member family (`prefix_name`), the entity from `WANDB_ENTITY` in the environment. The project name is a config field (`logger_params['project_name']`), so the choice is data, not code |
| Q7 | Should the print logger remain an option, given wandb already writes to the console? | **Yes** — keep `logger_name: Literal['wandb', 'print']`. The original had two loggers and the print logger is the one that works with no account at all, which is what the offline test suite uses |

---

## Definition of done

A port is finished when **all** of these are true. (Condensed from
`docs/ports/porting-worklist.md` § "Definition of done".)

**Status, 2026-10-21.** Everything below is done except the weight
conversion, the one-epoch parity against live Ignite, the DDP sharding
of the imbalanced sampler, and a real training run. The four are listed
with their reasons in `tasks/todo.md`; each is blocked by something this
environment does not have — a checkpoint, a multi-node setup, a GPU, or
the competition audio — rather than by unfinished work.

**Coverage**
- [ ] every row of `docs/ports/feature-parity-matrix.md` A–J is ticked, or explicitly deferred with a reason recorded in this file
- [ ] every row of §K (the vote) is implemented
- [ ] all 21 configs construct and the port reads all 39 attributes
- [ ] the eleven out-of-scope items stay out of scope

**Numerics**
- [ ] the log-mel matches the reference front-end to 1e-4 on a fixed waveform
- [ ] `clipwise_output` matches the reference model to 1e-4 on a fixed input
- [ ] the criterion at `gamma=0` equals `nn.BCELoss()` to 1e-6
- [ ] the three metrics reproduce the reference values to 1e-6
- [ ] the LR curve matches the golden fixture, step for step, to 1e-12
- [ ] a one-epoch run matches the reference per-step loss to 1e-5

**Operations**
- [ ] checkpoints keep the best `n_saved` by the tracked metric, and the monitor name resolves
- [ ] a weights-only resume leaves the optimiser at its initial state
- [ ] a full resume restores the epoch counter
- [ ] the wandb run carries the same metric series as the original's Neptune run (`train/loss`, `train/<loss_key>`, `train/lr`, `train/<metric>`, `valid/<metric>`)
- [ ] the inference artefact keeps the frame axis
**Reproducibility**
- [ ] every RNG is seeded from a config field; no module seeds outside `helpers/seed.py` (asserted by a grep test)
- [ ] re-running the same config with the same seed on the same data gives **identical** per-step losses and identical checkpoint hashes
- [ ] the result does not depend on `num_workers` or on batch scheduling
- [ ] every run writes a manifest with the resolved config, the seeds, the git commit (and dirty-diff hash), the package versions, and the SHA-256 of the fold CSVs, the audio, the label space, the mel matrix and the score artefacts
- [ ] `reproduce <manifest.json>` re-checks every input hash and re-runs the recorded command
- [ ] the fold generator is seeded and its seed is committed; the label space is committed with its source hash
- [ ] the ensemble manifest names, per member, the config, the checkpoint and the checkpoint's hash

**The two that are not about the code**
- [ ] the 13-model vote produces a submission — and **both** backends produce the same one
- [ ] the fold generator and its seed are committed, so the ensemble can be re-derived
