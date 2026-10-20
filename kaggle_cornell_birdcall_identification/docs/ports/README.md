# Porting guides — PyTorch Lightning and TensorFlow/Keras

These three documents are the **specification** for porting the Cornell Birdcall
solution off PyTorch Ignite onto **PyTorch Lightning** and **TensorFlow/Keras 3**.

| Document | What it is | Read it when |
|---|---|---|
| [code-issues-and-framework-wins.md](code-issues-and-framework-wins.md) | The 23 major issues in the original code, ranked by impact, with a verdict on whether a framework change actually helps each one | **First, if the question is "is porting worth it?"** |
| [feature-parity-matrix.md](feature-parity-matrix.md) | Exhaustive inventory of everything the original does, with the Lightning and Keras target for each item | **First, if the decision is made.** This is the checklist that guarantees nothing is left out. |
| **[porting-worklist.md](porting-worklist.md)** | **The file-by-file execution order** — what to port, in what order, what traps each file holds, and how you know you got it right | **When you sit down to do the port.** This is the one you work through. |
| [pytorch-lightning-guide.md](pytorch-lightning-guide.md) | Step-by-step implementation guide for the Lightning port | When writing the Lightning port. |
| [tensorflow-keras-guide.md](tensorflow-keras-guide.md) | Step-by-step implementation guide for the Keras port | When writing the Keras port. |

> **Before you start:** read
> [code-issues-and-framework-wins.md](code-issues-and-framework-wins.md). The
> single largest issue in this repository is not a framework problem — the
> winning 13-model vote does not exist in the code, the cross-validation
> cannot be reproduced, and there are no tests. A framework port fixes none of
> those three.

The two existing analyses in [`../framework-comparisons/`](../framework-comparisons/) are
the *reasoning* behind these decisions — why Lightning's callback list is a
regression, why `Softmax(axis=-1)` is a silent wrong answer in Keras, why the
mel scale matters. **Read them if you want the why; use these three documents if
you want the what.** Nothing here contradicts them; where they disagree on a
detail, the framework-comparisons analysis is the authority and this directory
points at it.

---

## Scope

**In scope — everything the live SED pipeline does:**

- the five PANNs model variants and the whole `forward()` shape trace
- the in-graph audio front-end (STFT → log-mel → SpecAugment)
- `SedDataset` (snippet slicing, label coding, background injection) and
  `ImbalancedDatasetSampler`
- both waveform augmentation stacks and the mixup
- all five registered criteria, their `loss_keys` contract, and the
  `(scalar, dict)` return shape
- the three validation metrics, the `RunningAverage` loss-key metrics, and the
  inference dump
- the optimiser, both LR schedules, gradient accumulation, gradient clipping,
  AMP, the validation cadence, checkpointing/resuming, the LR finder, DDP,
  the progress bar, and both loggers (Neptune + print/Slack)
- the whole `Parameters` config surface (39 attributes) and all 21 config files
- the entry point and its `importlib` config resolution

**Out of scope — and why (each row is a decision, not an oversight):**

| Excluded | Reason |
|---|---|
| `src/engine/main_engine*.py`, `src/train*.py` (v1–v4) | The pre-SED mel-*image* classification line. Superseded by `sed_engine.py` + `sed_train.py`; unreachable from any shipped config. Also runtime-broken: `helpers/audio_utils.py` has no `mono_to_color`, which `dataloaders/audio_dataset.py:112` imports. |
| `src/dataloaders/audio_dataset*.py` (v1–v4) | Same legacy line. `v1`–`v3` are hard-broken on the missing `mono_to_color`; `v4` is the intermediate `AudioDatasetV4`. |
| `src/augmentations/base_augment.py`, `default_augment.py`, `spec_augment.py` | Legacy image-pipeline augmentation. `spec_augment.py` raises `ImportError` at import time (same missing `mono_to_color`). |
| `src/models/aux_*.py`, `classifier_*.py`, `densenet121.py`, `densenet161.py` | Legacy model zoo, dispatched by nothing. |
| The 11 loss classes the SED engine cannot reach | Only `main_engine*` selects them. `sed_scaled_focal_loss.py` is imported by *nothing at all*. |
| `src/metrics/custom_f1score.py`, `custom_lraps.py`, `custom_test.py` | Legacy metric family (sigmoid + mean-pool); the SED path uses `Sed*` metrics. |
| `src/augmentations/noise_maker.py` | Reference implementation of the pink-noise generator. The actual WAVs were produced by an external Kaggle kernel (`README.md:127`). The *function* is not called from the repo. |
| The 13-model vote, 5-second row slicing, `submission.csv` emission | **Never existed in this repo.** The vote lives in an external Kaggle kernel (`README.md:130`). See [below](#the-thirteen-model-vote). |

Nothing in the "out of scope" table is reachable from the winning
configuration, so dropping it cannot change a single number the solution
produces. **37 of the 85 files under `src/` are out of scope**, plus two
dead symbols inside live files (`AUDIO_CONFIG`, `get_dict_value`) — see
[`../src-reference/README.md`](../src-reference/README.md) for the per-file
status of every one. If you want the port to also cover the legacy research
paths, that is a *second* porting project with a second parity matrix — say
so and it gets written.

---

## Does the code match the docs 1:1?

**No — in neither direction.** But the correspondence is *complete*, and the
shape of the mismatch is deliberate. Measured from the knowledge graph and the
"Covers" tables:

| question | answer | evidence |
|---|---|---|
| **1 chapter → 1 code file?** | **No.** Only **2 of 14** chapters map to exactly one file (`04` → `sed_models.py`, `08` → `sed_test.py`). The spread is **2 → 39** files per chapter. | the per-chapter reference counts |
| **1 code file → 1 chapter?** | **No.** **0 of 10** `src/` packages are touched by exactly one chapter. `engine` is read by **13 of 14** chapters, `models` by 11, `dataloaders` by 11, `metrics` by 9. | the chapter → package table |
| **1 chapter → 1 package?** | **Essentially yes** — for **7 of 14** chapters, the "Covers" column names exactly one `src/` package (`01`→`dataloaders`, `02`→`helpers`, `03`→`augmentations`, `04`→`models`, `05`→`engine`, `07`→`loss`, `09`→`config_params`). One chapter owns two (`10`→`metrics`+`logger`); six are cross-cutting by design (`00`, `06`, `08`, `11`, `12`, `13`). | `docs/PLAN.md`'s "Covers" column |
| **Is every code file documented?** | **Yes.** All **85** files now have a per-file entry with a status, a public API, its exact behaviour and its traps — in [`../src-reference/`](../src-reference/README.md). | the reference |
| **Is every file *referenced* by a topic chapter?** | **72 of 84.** The 12 unreferenced are individual `config_params/final_sed*` config files, covered by their directory's chapter (09) but not cited individually. | the reference graph |
| **Does every chapter have code?** | **Yes**, all 14. | the reference graph |

**So the map is a topic → package map with a deliberately overlapping
reference graph — not a 1:1 file map, and not meant to be one.** A 1:1
reference graph would mean the docs never explained how the parts connect:
the engine *should* be read by the losses chapter (it selects the criterion),
the model *should* be read by the data-pipeline chapter (the dataset feeds it).

**The overlap is not a defect, but the file-level 1:1 claim would be.** One
file can genuinely belong to several chapters, and the *reverse* — one chapter
covering 39 files — is why the per-file reference exists.

---

## The thirteen-model vote

This is the part most likely to be forgotten, because it is where the
competition was actually won and where the repo contains **no code**.

`README.md:37-62` is the only record:

- 13 models = 4-fold no-mixup + 5-fold no-mixup + 4-fold mixup, all
  `sed_dense121att`
- a species is accepted for a 5-second row when **≥4 of the 13 models** fire
- with **clipwise ≥ 0.3 AND framewise ≥ 0.3**
- that variant placed 2nd on public (0.616) and **1st on private (0.681)**
- the 9-model (no-mixup) ensemble used **3 votes, clip 0.3 / frame 0.5**
- 2 submissions per day was the binding constraint on threshold search

The in-repo ancestor of that decision rule survives as a hard-coded literal in
`src/metrics/sed_f1score_frame.py:35,40`:

```python
(np.sum((pf >= self.threshold) & (pc >= 0.3))) > 0.0
```

The 0.3 is the same 0.3. **Both ports must keep that literal and the
per-head independence** — the frame gate and the clip gate are separate
thresholds, not one.

Three things the repo does *not* contain, and the port therefore cannot
inherit: 5-second row slicing, site 1/2/3 handling, the `nocall` fallback,
and `submission.csv` emission. If the ported pipeline is meant to produce a
submittable file, those four must be written — and they are *new work*, not
porting. Section "Vote and submission" in the parity matrix marks them.

---

## The two ports, side by side

| | PyTorch Lightning | TensorFlow/Keras 3 |
|---|---|---|
| Reuse from the original tree | **Almost everything.** The model, losses, dataset, sampler, augmentations and front-end are all plain PyTorch and port unchanged. | **Almost nothing.** Every tensor op, the front-end, the losses, the metrics, the dataset and the sampler are rewritten. |
| What is actually replaced | The Ignite **control plane** only: the event graph in `base_engine.setup()`. | The control plane **and** the entire numeric core. |
| Expected size of the new code | ~200–300 lines | ~900–1200 lines |
| Numerical risk | **Low.** The model and loss maths are untouched; only the LR schedule needs proving equivalent. | **High.** The mel scale, the dB formula, the reflect padding, the softmax axis and the frame-padding all change silently if you get them wrong. |
| Verdict in the existing analysis | "Would have made the author's life easier at 3am" | "Would have been a mistake for this solution" — the control plane maps badly and the tuned recipe would have to be re-tuned. |

---

## The rule this directory follows

> **A port is not done when it runs. It is done when the parity matrix has no
> unchecked row.**

The matrix has a *verify* column for a reason: for each item there is a
concrete check — a number, an equality, an API call — that proves the ported
behaviour matches the original. A row is only ticked when its check passes.

Three of the checks are known to be non-trivial and are called out in the
guides:

1. **The LR schedule must be numerically identical to Ignite's.** The exact
   formulas are transcribed in
   [pytorch-lightning-guide.md](pytorch-lightning-guide.md#lr-schedules) and
   are verifiable against a live Ignite installation.
2. **The log-mel matrix must stay Slaney-normalised.** `tf.signal`'s matrix
   is HTK and unnormalised; `librosa.filters.mel` is Slaney and
   area-normalised. They are different matrices. Keep librosa's.
3. **The `AttBlock` softmax axis must be over time, not classes.** In Keras
   the axes come from the right; `Softmax(axis=-1)` is a silent wrong answer
   that trains fine and scores badly.

---

## Repository conventions the ports must respect

- `src/` is the import root; there is **no `__init__.py` anywhere** and the
  package works through implicit namespace packages. Keep it that way — a port
  that adds `__init__.py` files changes how every existing import resolves.
- Configs are **plain Python modules** resolved with
  `importlib.import_module` and instantiated with `module.Parameters()`. Do
  not switch to YAML/`LightningCLI`/`absl.flags` for the shipped configs:
  the 13-model ensemble depends on 13 *named, individually reproducible*
  config files.
- `src/config_params/` contains **CRLF line endings** in 20 of its 22 files
  (`docs/README.md:61`). Line-anchored `grep` silently misses them. If you
  regenerate or rewrite configs, preserve or normalise deliberately.
- The `Parameters` attribute set is the public config contract. See the
  matrix's config section — every one of the 39 attributes must keep its
  meaning in both ports.
- Paths in the shipped configs are absolute (`/data/`, `/folds/`,
  `/saved_models/`, `/pinknoise`). They are not shipped with the repo and
  are not reproducible from it (`docs/modules/09`).
