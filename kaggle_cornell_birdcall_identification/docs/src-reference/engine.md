# `src/engine/` — 6 files

The control plane. `BaseEngine` is the largest and most-connected object in
the repository; `SedEngine` is the live subclass; the other four are the
superseded research generations.

| file | LOC | status |
|---|---|---|
| [`base/base_engine.py`](#basebase_enginepy) | 375 | **LIVE** |
| [`sed_engine.py`](#sed_enginepy) | 226 | **LIVE** |
| [`main_engine.py`](#main_enginepy) | 118 | LEGACY + **BROKEN** |
| [`main_engine_v2.py`](#main_engine_v2py) | 168 | LEGACY + **BROKEN** |
| [`main_engine_v3.py`](#main_engine_v3py) | 168 | LEGACY + **BROKEN** (L153) |
| [`main_engine_v4.py`](#main_engine_v4py) | 221 | LEGACY + **BROKEN** (L43, L48, L205) |

---

## `base/base_engine.py`

**Status: LIVE.** The Ignite event graph, the 375-line control plane, and the
single most-connected object in the knowledge graph (82 edges, betweenness
0.200).

**Purpose.** Builds and wires the training machinery: two Ignite `Engine`s
(trainer + evaluator), the dataloaders, the model, the criterion, the
optimiser, the scheduler, the metrics, the checkpoint saver, the LR finder and
the loggers.

**Public API** — 31 methods

```python
class BaseEngine(object):
    def __init__(self, local_rank, hparams)
    # setup
    def _init_distribution(self)                 # rank/device/auto_* plumbing
    def log_basic_info(self, logger)
    def load_trainer_from_checkpoint(self)
    def setup_checkpoint_saver(self, to_save)
    def attach_metrics(self, engine, metrics)
    def setup(self)                              # the event graph
    # the three steps
    def train_step(self, engine, batch)
    def eval_step(self, engine, batch)
    def test_step(self, engine, batch)
    def loss_backpass(self, loss)                # the AMP fork of backward
    def prepare_batch(self, batch, model='valid')
    def output_transform(self, x, y, y_pred, loss=None)
    # runs
    def train(self, run_params)
    def validate(self, dl)
    def evaluate(self, dl)
    def get_batch(self)
    # factories (the 11 _init_* hooks)
    def _init_scheduler(self)      def _init_model(self)
    def _init_optimizer(self)      def _init_criterion_function(self)
    def _init_logger(self)         def _init_metrics(self)
    def _init_train_dataloader(self)   def _init_valid_dataloader(self)
    def _init_test_dataloader(self)    def _init_augmentation(self)
    def _init_score_function(self)
    # utilities
    def clip_gradients(self)
    def lr_finder(self, min_lr=0.00003, max_lr=10.0, num_iter=None)
    def loss_fn(self, y_pred, y)                # NotImplementedError
```

**The event graph** (`setup`, L134-214) — nine ordering-sensitive
registrations, in this exact order:

```
setup()
├── _init_distribution()                    # rank, device, auto_dataloader/model/optim
├── trainer = Engine(self.train_step)
├── load_trainer_from_checkpoint()          # BEFORE the scheduler is attached
├── trainer.add_event_handler(ITERATION_STARTED, scheduler)
├── attach_metrics(trainer, train_metrics)
├── trainer.add_event_handler(EPOCH_STARTED, set_epoch)
├── common.setup_common_training_handlers(...)
├── evaluator = Engine(self.eval_step)
├── trainer.add_event_handler(EPOCH_COMPLETED, complete_clear)
├── trainer.add_event_handler(EPOCH_COMPLETED(every=eval_every), validate(valid_loader))
├── attach_metrics(evaluator, validation_metrics)
├── setup_checkpoint_saver(to_save)         # attaches to evaluator.COMPLETED
└── if rank == 0: logger._init_logger(...)  # logger._add_train_events/_add_eval_events
```

**Behaviour**

- `USE_AMP` is **always `False`** (L6-13). The `try` block prints
  "Found NVIDIA Apex, using mixed precision" and then
  "NOT USING APEX BECAUSE CANNOT CREATE MEL SPEC…" — the first line lies on
  every path.
- `XLA_USE_BF16 = os.getenv("XLA_USE_BF16")` (L28) with **no default**,
  swallowed by a bare `except` (L29-30), so it is the integer `0` where a
  string was intended; it is compared against `1` in `clip_gradients` (L346).
- `complete_clear` (L175-180) nulls `engine.state.batch` / `engine.state.output`
  and calls `gc.collect()` at each epoch end.
- `train_step` (L237-249) calls `optimizer.step()` only when
  `engine.state.iteration % accumulation_steps == 0` (L245) and zeroes grads
  after. `self.scheduler` is stepped at `ITERATION_STARTED` — **every**
  iteration, independently of the optimiser step.
- `clip_gradients` (L327-350) is a verbatim copy of Lightning's own
  `training_tricks` code, with `EPSILON = 1e-6` and `EPSILON_FP16 = 1e-5`
  (L31-32).
- `lr_finder` (L353-374) removes the validation handler and the scheduler
  before sweeping with Ignite's `FastaiLRFinder`.
- `train` (L216-223) wraps `trainer.run()` in a `try/except` that **prints the
  traceback and continues**, then closes the Neptune experiment normally. A
  crashed 50-epoch run exits 0 and looks finished in the tracker.

**Gotchas**

- ⚠️ `_init_*` dispatch has **no `else` branch** and no key validation — an
  unknown `model_name` fails later as
  `AttributeError: 'SedEngine' object has no attribute 'model'`.
- ⚠️ The scheduler steps every iteration while the optimiser steps every
  `accumulation_steps` — with `accumulation_steps > 1` the LR runs ahead.
- ⚠️ `setup_checkpoint_saver` attaches the saver to the **evaluator's**
  `COMPLETED` event (L128), so checkpoints are written only on validation
  days, i.e. every `eval_every` epochs.
- ⚠️ `evaluate` (L230-235) is **dead code** — nothing calls it, and the
  `mode='test'` branch it would use is broken.
- ⚠️ `load_trainer_from_checkpoint` (L103-118) restores only
  `{"model": …}` when `load_model_only=True` — which is **every** shipped
  config — so the optimiser, the scheduler and the RNG state are never
  restored.

---

## `sed_engine.py`

**Status: LIVE.** The only engine the shipped path instantiates.

**Public API** — 12 overrides of the base class

```python
class SedEngine(BaseEngine):
    def __init__(self, local_rank, hparams)
    def prepare_batch(self, batch, mode='valid')
    def loss_fn(self, y_pred, y)
    def output_transform(self, x, y, y_pred, loss=None, dict_loss={}, mode='valid')
    def _init_optimizer(self)        def _init_criterion_function(self)
    def _init_scheduler(self)         def _init_logger(self)
    def _init_metrics(self)           def _init_model(self)
    def _init_augmentation(self)      def _init_train_dataloader(self)
    def _init_valid_dataloader(self)  def _init_test_dataloader(self)
```

**Behaviour**

- `prepare_batch` (L16-69) returns `((x, mixup_lambda), y)` where `y` is the
  label dict. In `train` mode with mixup on, the lambda is drawn with
  `get_lambda(batch_size=bs*c, device=x.device)` (L26) and the **labels** are
  mixed and halved to `(B·C)//2, 1, 264` (L28-30). The `test` branch (L57-69)
  references an **undefined local `x`** → `NameError`.
- `loss_fn` (L71-73) calls `self.criterion(y_pred["clipwise_output"], y)` —
  **only the clipwise head**. `framewise_output` never reaches a loss.
- `_init_criterion_function` (L87-102) registers exactly **five** criteria:
  `focal_loss_standard`, `sed_removed_focal_loss`,
  `sed_scaled_pos_neg_focal_loss`, `sed_scaled_pos_neg_focal_loss_augd`,
  `bce_controlled_standard_loss`. The other eleven loss classes are
  unreachable from here.
- `_init_scheduler` (L104-128) supports `none`, `warmup_with_cosine` and
  `one_cycle_cosine`. `warmup_with_cosine` builds
  `ConcatScheduler([LinearCyclical(lr·f → lr, cycle=2E), CosineAnnealing(lr → lr·f, cycle=N·E)], durations=[E])`.
- `_init_metrics` (L140-174) registers a `RunningAverage` for `train_avg_loss`
  and `valid_avg_loss` plus **one per `criterion.loss_keys` entry**
  (L155-156), and then `SedLRAPS` / `SedF1ScoreClip` / `SedF1ScoreFrame` under
  the **unprefixed** names `lraps`, `f1score_clip`, `f1score_frame` (L160-174).
  `SedTest` is registered only `if "test" in hparams.metrics` (L172) — no
  shipped config sets that.
- `_init_model` (L176-198) supports five names and honours `pretrained_path`
  **only** for `sed_ccnn14att` (L179-186), deleting the `att_block` weights
  before loading.
- `output_transform` (L75-81) has a **mutable default** `dict_loss={}`.

**Gotchas**

- ⚠️ `hparams.metrics` names are unprefixed, so the checkpoint's
  `track_metric="f1score_clip"` matches `engine.state.metrics["f1score_clip"]`
  exactly. A port that renames the metric must remap the monitor too.
- ⚠️ `hparams.save_dir` and `hparams.combine_files` (L174) are **not defined
  in any config** — the `SedTest` branch is unreachable, not merely unused.
- ⚠️ `criterion.loss_keys` is read unguarded (L155) — `BCELoss` and
  `SmoothBCELoss` define no `loss_keys`, so a config selecting one would
  raise `AttributeError`.

---

## `main_engine.py` — v1

**Status: LEGACY + BROKEN.** Imported by `train.py:2`.

**Purpose.** The v1 training engine: the v1 dataset + the DCASE classifier +
BCE losses.

**Public API.** `MainEngine(BaseEngine)` with the same 12 overrides as
`SedEngine`; the model branch accepts only `"dcase"`, the augmentation branch
only `"baseline"`, the criterion branches `"bce"` and `"smooth_bce"`, and the
metric branch only `"f1score"`.

**Broken at**

- **L32** `AttributeError: 'MainEngine' object has no attribute 'ls_fn'` —
  `loss_fn` calls `self.ls_fn`, which is never assigned anywhere in the
  repository. `self.criterion` is built but never called.
- With a shipped config it fails **earlier**: `_init_model` has no
  `sed_dense121att` branch and `_init_augmentation` has no `"default"` branch,
  so `self.model` / `self.tfms` are never set and `_init_train_dataloader`
  (L108) raises `AttributeError` first.

**Undefined hparams** — `num_classes` (L99), `smooth` (L55).

---

## `main_engine_v2.py` — v2

**Status: LEGACY + BROKEN.** Imported by `train_v2.py:2`.

**Purpose.** Multi-target batches (coded labels + a clean flag), DenseNet and
DCASE models, noisy/curated losses, and F1@50/60/70 + LRAPS + test metrics.

**Broken at**

- **L35** `AttributeError: 'MainEngineV2' object has no attribute 'ls_fn'` —
  the same dead attribute as v1.
- **L153** `ImportError` — the `"spec_aug"` branch imports
  `augmentations/spec_augment.py`, which fails at import time.

**Undefined hparams** — `beta` (L61, L64), `q` (L61, L64), `smooth` (L58),
`num_classes` (L140, L143, L146), `save_dir` (L135), `combine_files` (L135).

**Gotchas**

- `output_transform` has a **mutable default** `dict_loss={}` (L38).
- `self.criterion` is created but `loss_fn` never calls it.
- The criterion branches `"lq"` / `"lsoft"` build `NoisyCuratedLoss`, whose
  `"lq"` branch is inert.

---

## `main_engine_v3.py` — v3

**Status: LEGACY.** Imported by `train_v3.py:2`. **The `ls_fn` bug does not
exist here** — `loss_fn` calls `self.criterion(y_pred_i, y)` (L39) instead.

**Purpose.** Multi-head (auxiliary) model with per-head loss weighting, and
per-head F1 metrics.

**Behaviour**

- `loss_fn` (L35-41) iterates the model's four heads with **hard-coded**
  `loss_factors = [1.0, 0.4, 0.2, 0.1]` — not read from hparams.
- The model branch accepts only `"aux_dense121"` (L144-146), which returns
  exactly four heads.
- Metrics are registered as `f"f1score50_{i}"` for `i in range(1)` — head 0
  only (L124-137).

**Broken at** — L153 `ImportError` via the `"spec_aug"` branch, which imports
`augmentations/spec_augment.py` (fails at import time on its missing
`mono_to_color`).

**Undefined hparams** — `beta` (L68, L71), `q` (L68, L71), `smooth` (L65),
`num_classes` (L146), `save_dir` (L141), `combine_files` (L141).

**Gotchas**

- ⚠️ The F1 wiring uses `x["y_pred"][i]` (L126-136) while `x["y"]` stays
  batched — the prediction loses its batch axis and the shapes mismatch
  `f1_score`. The v2 wiring is shape-consistent; v3's is not.
- `self.criterion` is called with a **tuple** target `(coded_labels, clean)`,
  which matches `NoisyCuratedLoss.forward` but would break `BCELoss`.

---

## `main_engine_v4.py` — v4

**Status: LEGACY + BROKEN.** Imported by `train_v4.py:2`. The "final" pre-SED
engine.

**Purpose.** Multi-head model with a per-head `beta` schedule from hparams,
audiomentations-based augmentation, criterion-key-driven metrics, and
`primary_codes` in the batch.

**Broken at**

- **L43** `AttributeError: 'Parameters' object has no attribute 'loss_factors'`
  on the first training step.
- **L48** `TypeError` — `self.criterion(y_pred_i, y, betas[i], betas[i]+0.05)`
  passes 4 positional args, but `BCEControlledStandardLoss.forward(self,
  y_pred, target)` takes 2.
- **L205** `_init_augmentation` only handles `aug_name == "default"`, so the
  five `final_sed_5_fold` configs (`aug_name="secondary_default"`) leave
  `self.tfms` unset → `AttributeError` at L211.

**Undefined hparams** — `loss_factors` (L43), `betas` (L44),
`primary_loss_factor` (L80), `gamma` (L86, L89), `alpha` (L86, L89),
`num_classes` (L202), `save_dir` (L197), `combine_files` (L197).

**Gotchas**

- ⚠️ v4 reads `gamma`/`alpha` as **top-level** hparams instead of via
  `**hparams.criterion_params` — which is exactly how the live `sed_engine.py`
  does it. That single difference is the root cause of the whole
  missing-attribute class of bugs.
- `_init_metrics` (L170-176) registers a train+valid `RunningAverage` for
  every `criterion.loss_keys` entry × 4 heads.
- `prepare_batch` has no fall-through `return` — an unrecognised `mode`
  returns `None`.
- `output_transform` has a **mutable default** `dict_loss={}` (L54).

---

## The undefined-hparams table

The universe of attributes any config defines:

```
accumulation_steps, add_pbar, apply_mixup, aug_name, bckgrd_aug_dir,
checkpoint_dir, checkpoint_params, criterion_name, criterion_params,
dist_params, eval_every, fold, gradient_clip_val, load_model_only,
logger_name, logger_params, lr, lr_scale_factor, metric_factor, metrics,
model_config, model_name, name, optimizer_name, PERIOD, pretrained_path,
run_params, scheduler_name, secondary_bckgrd_aug_dir, test_ds_params,
track_metric, train_bs, train_ds_params, train_num_workers, valid_bs,
valid_ds_params, valid_num_workers, val_length, weight_decay
```

Every attribute the **base class** reads is defined in every config. The gaps
are entirely in the legacy subclasses:

| engine | undefined hparams (with line numbers) |
|---|---|
| `main_engine.py` | `num_classes` (99), `smooth` (55) |
| `main_engine_v2.py` | `beta` (61, 64), `q` (61, 64), `smooth` (58), `num_classes` (140, 143, 146), `save_dir` (135), `combine_files` (135) |
| `main_engine_v3.py` | `beta` (68, 71), `q` (68, 71), `smooth` (65), `num_classes` (146), `save_dir` (141), `combine_files` (141) |
| `main_engine_v4.py` | `loss_factors` (43), `betas` (44), `primary_loss_factor` (80), `gamma` (86, 89), `alpha` (86, 89), `num_classes` (202), `save_dir` (197), `combine_files` (197) |
| `sed_engine.py` | `save_dir` (174), `combine_files` (174) — only on the unreachable `SedTest` branch |

`beta`, `q`, `gamma` and `alpha` **do** exist as keys inside
`criterion_params` in all 21 configs — the legacy engines just read them from
the wrong level.

---

## See also

- [modules/05 — Training engine](../modules/05-training-engine.md)
- [modules/06 — Checkpointing and resuming](../modules/06-checkpointing-and-resuming.md)
- [modules/11 — Legacy research paths](../modules/11-legacy-research-paths.md)
- [ports/pytorch-lightning-guide.md](../ports/pytorch-lightning-guide.md) §Phase 6 — the Trainer mapping
