# PyTorch Lightning implementation guide

Step-by-step specification for porting the Cornell Birdcall solution from
PyTorch Ignite to **PyTorch Lightning 2.x**.

Read [feature-parity-matrix.md](feature-parity-matrix.md) alongside this — every
step here carries the matrix IDs it discharges. The *why* behind the structural
choices is in
[`../framework-comparisons/pytorch-lightning-port.md`](../framework-comparisons/pytorch-lightning-port.md).

**The port is mostly reuse.** The model, the losses, the dataset, the sampler,
the augmentations and the audio front-end are plain PyTorch and are imported
unchanged. What is replaced is the **control plane**: the nine ordering-sensitive
event registrations in `base_engine.setup()`.

---

## Phase 0 — Environment (resolve this first)

The repo pins six packages and **not** torch/torchvision/torchaudio — the
README's GCP setup assumes a preinstalled torch from the `pytorch-latest-gpu`
image, so the torch version is whatever that image carried. A Lightning pin
implies a torch version range, so this must be resolved before anything else.

| Package | Original pin | Port requirement |
|---|---|---|
| `pytorch-ignite` | `==0.4.0.post1` | **Dropped.** The port does not import ignite. |
| `torch`, `torchvision`, `torchaudio` | *unpinned* | **Pin them explicitly.** Pick a Lightning-compatible pair and write it down; this is the single biggest reproducibility improvement the port can make. |
| `lightning` | — | `lightning>=2.2,<3` (the `lightning` package, not the legacy `pytorch-lightning` alias). |
| `torchmetrics` | — | Comes with Lightning. Only needed if you implement the metrics as `torchmetrics.Metric` rather than hand-rolled accumulators. |
| `neptune-client` | `==0.4.107` | ⚠️ **Must be upgraded.** The original wraps ignite's Neptune logger, which speaks the 0.4.x API. Lightning's `NeptuneLogger` speaks `neptune>=1.0`. This is an unavoidable consequence of the framework change. |
| `audiomentations` | `==0.11.0` | Unchanged. |
| `SoundFile` | `==0.10.3.post1` | Unchanged. |
| `tqdm`, `psutil` | `==4.35.0`, `==5.6.7` | Unchanged. |
| `librosa` | *unpinned* | Unchanged — but note `librosa.filters` imports `numba`, and `numba>=0.60` removed `numba.decorators`, which librosa 0.10's `util/decorators.py` imports unguarded. **Pin `numba<0.60` and `numpy<1.27` together, or pick a librosa/numba pair that agrees.** |
| `scikit-learn` | *unpinned* | Unchanged — `SedLRAPS` and the F1 metrics use it. |

The `librosa`/`numba`/`numpy` triangle is a real trap and it is easy to miss
because `import librosa` can succeed while `librosa.filters` fails at first
use — and `librosa.filters` is exactly what the front-end calls.

### ⚠️ The original does not run on a modern librosa

`src/helpers/sed_audio_utils.py:58` calls

```python
fft_window = librosa.util.pad_center(fft_window, n_fft)
```

librosa **0.10.0** made `size` keyword-only, so on any modern librosa
`PANNsDense121Att.__init__` **raises** and the model cannot be constructed at
all. Verified on librosa 0.10.2:

```
TypeError: pad_center() takes 1 positional argument but 2 were given
```

This is not a porting problem — it means **the original does not run today**
either. Three ways out, in order of preference:

1. **Change the call to `pad_center(fft_window, size=n_fft)`** — one line, and
   it is what librosa wants. A port should do this; the ported front-end will
   not have the problem at all.
2. Pin `librosa<0.10` — which also pins `numba` (librosa 0.9 needs numba
   ≤ 0.58) and `numpy<1.24`. Heavy, and it blocks validating a Keras
   front-end against the same environment.
3. Precompute the window outside the model — changes the design; only worth
   it for a port that drops the vendored front-end entirely.

### ⚠️ `torchvision.models.densenet121(pretrained=True)` is deprecated

`src/models/sed_models.py:367, 492, 615, 738`. torchvision 0.13 deprecated
the `pretrained` boolean in favour of `weights=`; it still works in 0.29 with
a warning and **will be removed**. All four `densenet{121,161,169,201}`
symbols still exist in torchvision 0.29, so the Lightning port can keep all
five model variants. The Keras port cannot: `keras.applications` has **only
DenseNet121** — 161/169/201 have no equivalent. Since all 13 ensemble
members use the 121, restricting the Keras port to it is a defensible
decision, but it must be an explicit one.

---

## Phase 1 — Entry point and config resolution

The `--config` contract must not change: it is a **dotted Python module path**,
resolved with `importlib`, and the config class is instantiated with
`module.Parameters()`. All 21 config files keep working unchanged.

```
src/                       # the import root; no __init__.py anywhere
  ports/
    pytorch_lightning/
      sed_train.py          # entry point
      engine.py             # dataloaders + Trainer wiring
      lightning_module.py   # model + criterion + steps
      sed_metrics.py        # ignite-free metric accumulators
      sed_scheduler.py      # LR schedule lambdas
      callbacks.py          # CompleteClearGC, LogEvery, the inference dump
      voting.py             # the 13-model vote (new — see Phase 11)
```

**Keep the namespace-package layout.** `src/` has no `__init__.py` and works
through implicit namespace packages; adding `__init__.py` files changes how every
existing import resolves. The port directory inherits the same convention.

**The entry point** mirrors `src/sed_train.py`:

- argparse `--config` → `importlib.import_module` → `module.Parameters()`
- `torch.backends.cudnn.benchmark = True` (keep it)
- `pl.seed_everything(42, workers=True)` replaces `manual_seed(42 + rank)`
  (see Phase 9 for the per-rank offset)
- `idist.Parallel(**dist_params)` is replaced by `Trainer(strategy=...)`

**Do not use `LightningCLI`.** It would change the config model from "one
Python file per experiment" to "one model + one YAML", and the 13-model
ensemble depends on 13 *named, individually reproducible* config files. The
repo's own analysis reaches the same conclusion.

**Keep the config's 39 attributes** — they are the public contract:

| Group | Attributes |
|---|---|
| Identity | `fold`, `name` |
| Model | `model_name`, `model_config` (9 keys), `pretrained_path` |
| Augmentation | `aug_name`, `apply_mixup`, `bckgrd_aug_dir`, `secondary_bckgrd_aug_dir` |
| Loss | `criterion_name`, `criterion_params` |
| Optimiser | `optimizer_name`, `weight_decay`, `lr` |
| Scheduler | `scheduler_name`, `lr_scale_factor` |
| Data | `PERIOD`, `train_ds_params`, `valid_ds_params`, `test_ds_params`, `train_bs`, `train_num_workers`, `valid_bs`, `valid_num_workers` |
| Metrics | `metrics`, `track_metric`, `metric_factor` |
| Logging | `logger_name`, `logger_params` (6 keys) |
| Checkpointing | `checkpoint_params` (3 keys), `checkpoint_dir`, `load_model_only` |
| Runtime | `run_params` (`max_epochs`, `epoch_length`), `dist_params`, `val_length`, `eval_every`, `add_pbar`, `accumulation_steps`, `gradient_clip_val` |

---

## Phase 2 — The LightningModule

The original's seven `_init_*` hooks become two registry dicts plus plain
`__init__` code. The repo's own analysis recommends the registry over the
if/elif, because the if/elif has **no `else` and no key validation** — an
unknown `model_name` in the original leaves the attribute unset and fails later
as a confusing `AttributeError`.

**Model registry** (matrix A5–A9):

| `hparams.model_name` | class |
|---|---|
| `sed_ccnn14att` | `PANNsCNN14Att` |
| `sed_dense121att` | `PANNsDense121Att` |
| `sed_dense161att` | `PANNsDense161Att` |
| `sed_dense169att` | `PANNsDense169Att` |
| `sed_dense201att` | `PANNsDense201Att` |

**Criterion registry** (matrix E1) — exactly the five the SED engine registers:

| `hparams.criterion_name` | class | `loss_keys` |
|---|---|---|
| `focal_loss_standard` | `FocalLossStandard` | `bce_loss`, `F_loss` |
| `sed_removed_focal_loss` | `SedRemovedFocalLoss` | `bce_loss`, `F_loss`, `FRemoved_loss` |
| `sed_scaled_pos_neg_focal_loss` | `SedScaledPosNegFocalLoss` | `bce_loss`, `F_loss`, `FScaled_loss`, `F_loss_0`, `F_loss_1` |
| `sed_scaled_pos_neg_focal_loss_augd` | `SedScaledPosNegFocalLossAugd` | same 5 |
| `bce_controlled_standard_loss` | `BCEControlledStandardLoss` | *(empty)* |

Two details the original gets wrong and the port should not reproduce:

- `getattr(self.criterion, "loss_keys", [])` — `BCELoss` and `SmoothBCELoss`
  define no `loss_keys`, and the original's `_init_metrics` would raise
  `AttributeError` on them. They are not in the SED registry, but a defensive
  `getattr` costs nothing.
- `BCEControlledStandardLoss` has an **empty** `loss_keys`, so it contributes
  zero extra metrics. That is the winning criterion's actual behaviour — do not
  "fix" it.

**`prepare_batch`** (matrix G16) ports almost verbatim. Lightning has already
moved the batch to the device, so the original's explicit `convert_tensor`
calls disappear; everything else — the mixup lambda, the label mixing, the
`(B·C)//2` halving — stays. The original's `mode='test'` branch references an
**undefined local `x`** and raises `NameError`; the port's branch reads
`batch["waveforms"]` like the other two.

**The mixup lambda** is drawn with `self.mixup_augmenter.get_lambda(batch_size=bs*c, device=x.device)`.
Note `Mixup.__init__` hard-codes `random_seed=1234`, so every run and every rank
draws the **same** sequence of `Beta(1,1)` lambdas. That is pre-existing
behaviour; the port inherits it (and the analysis notes it as a wart, not a
bug to fix mid-port).

---

## Phase 3 — The three steps

**`training_step`** (matrix E3, E4, G18):

1. `(x, mixup_lambda), y = self.prepare_batch(batch, mode="train")`
2. `y_pred = self.model((x, mixup_lambda))` — the model takes the **tuple**;
   the mixup lambda is consumed inside `preprocess` on the spectrogram
3. `loss, dict_loss = self.criterion(y_pred["clipwise_output"], y)` — the
   **clipwise head only**
4. log `train_loss` and `train_<key>` for each `dict_loss` entry
5. return `loss`

**`validation_step`**: the same forward under `torch.no_grad()` (Lightning
gives you that automatically in validation), the loss logged as `val_avg_loss`
and `val_<key>`, and — critically — **the validation metrics are updated here,
not returned**, because they are epoch-wise accumulators.

**`test_step`**: the original's `test_step` returns `{"y_pred", "x", "y"}`
without computing a loss. Keep that shape; the dump callback consumes it.

**Metric naming.** The original's validation metrics are `valid_avg_loss`,
`valid_<loss_key>`, and — **without a prefix** — `lraps`, `f1score_clip`,
`f1score_frame`. `hparams.track_metric` is `"f1score_clip"`, matching the
unprefixed name.

Lightning warns when a validation metric is not prefixed `val_`, and
`ModelCheckpoint(monitor=...)` looks the monitor up in
`trainer.callback_metrics` by exact name. Pick one and apply it consistently:

- **Recommended**: log as `val_lraps`, `val_f1score_clip`, `val_f1score_frame`,
  and map the checkpoint monitor to `"val_" + hparams.track_metric` (unless the
  configured name already starts with `val_`).
- **Alternative**: log under the bare names to match the original exactly, and
  accept Lightning's warning.

Do not mix the two — a monitor name that is not logged is the Lightning
equivalent of the original's `score_function` `KeyError`, and it fails
**silently**: the checkpoint callback simply never fires.

---

## Phase 4 — Metrics

The three validation metrics subclass `ignite.metrics.Metric` and use
`@reinit__is_reduced` / `@sync_all_reduce`. Lightning has no equivalent
protocol, so there are two routes:

1. **`torchmetrics.Metric`** — a `reset`/`update`/`compute` interface that
   matches ignite's almost exactly. This is the smallest change.
2. **Hand-rolled accumulators** — a `reset`/`update`/`compute` class of your
   own, updated in `validation_step` and reduced in `on_validation_epoch_end`.

Either way, **the arithmetic is copied unchanged**:

- `SedLRAPS` — `sklearn.label_ranking_average_precision_score` per item, on
  `clipwise_output.max(axis=1)[0]` vs `target[:,0,:]`, averaged.
- `SedF1ScoreClip` — micro-F1 on the clip head, with the **background branch**:
  a row with no positive label scores 1.0 only when nothing fires.
- `SedF1ScoreFrame` — micro-F1 on the frame head **ANDed** with a hard-coded
  0.3 clip gate: `(pf >= threshold) & (pc >= 0.3)`.

Three things that must not drift:

- **No sigmoid is applied** in any of the three — the `AttBlock` already
  applied it. The original says so in a comment in each file.
- **Reduce, then divide.** The ignite metrics accumulate a sum and a count and
  divide at `compute()` time. Averaging per-batch F1 values instead is a
  different number.
- The 0.3 in `SedF1ScoreFrame` is the in-repo ancestor of the winning ensemble
  threshold. It is a literal in the source, not a config value. Keep it a
  literal, and keep it on the **clip** side of the AND.

**The `RunningAverage` per loss key** (matrix F5) is replaced by Lightning's
`self.log` / `self.log_dict` with `on_epoch=True`. The mechanism changes; the
intent — five copies of the same number, one per key — does not.

---

## Phase 5 — LR schedules (must be numerically identical)

This is the one place where a "close enough" port silently changes the
training run. The formulas are transcribed from **pytorch-ignite 0.4.0.post1**,
`ignite/contrib/handlers/param_scheduler.py`.

**`LinearCyclicalScheduler.get_param`**

```
p = event_index / cycle_size
value = end_value + (start_value - end_value) * abs(p - 0.5) * 2
```

A triangle: at `p=0` the value is `start_value`, at `p=0.5` it is `end_value`,
at `p=1` it is back to `start_value`.

**`CosineAnnealingScheduler.get_param`**

```
p = event_index / cycle_size
value = start_value + (end_value - start_value) / 2 * (1 - cos(pi * p))
```

**`ConcatScheduler` boundary semantics.** With
`ConcatScheduler([linear, cosine], durations=[E])`:

- the linear phase runs for **exactly `E`** events, with its own `event_index`
  counting `0 … E-1`;
- the cosine phase then runs, with its own `event_index` **restarting at 0**.

**`ParamScheduler.__call__` ordering.** It reads `get_param()` with the
*current* `event_index` and increments it afterwards, so the first scheduled
event uses index 0.

**The repo's `warmup_with_cosine`** therefore is:

| phase | scheduler | start | end | cycle size | duration |
|---|---|---|---|---|---|
| 1 | linear | `lr · lr_scale_factor` | `lr` | `2 · epoch_length` | `epoch_length` iterations |
| 2 | cosine | `lr` | `lr · lr_scale_factor` | `num_epochs · epoch_length` | the rest |

with `epoch_length = run_params["epoch_length"] or len(train_loader)` and
`num_epochs = run_params["max_epochs"]`.

**Why the triangle's falling half is never used:** the linear phase lasts one
epoch but its cycle is two epochs long, so only the *rising* half is traversed
— it starts at `lr · 0.01` and reaches exactly `lr` at the end of the first
epoch, where the cosine phase takes over at `lr`. The curve is continuous.

**`one_cycle_cosine`** is the cosine phase alone, over the whole run.

**Transcribing to `LambdaLR`.** `LambdaLR` calls the lambda with
`last_epoch`, which starts at 0 and increments once per `scheduler.step()`.
Driven by Lightning with `interval="step"`, batch `t` sees the lambda's value
at `t` — the same index Ignite used for iteration `t`. So the lambda is:

- for `t < epoch_length`: the linear value at `t` with `cycle_size = 2·E`;
- for `t ≥ epoch_length`: the cosine value at `t − epoch_length` with
  `cycle_size = N·E`;

each divided by the base LR, because `LambdaLR` multiplies.

Return it from `configure_optimizers` as
`{"optimizer": …, "lr_scheduler": {"scheduler": LambdaLR(…), "interval": "step", "frequency": 1}}`.

**Verify it against live Ignite.** Install `pytorch-ignite` in a scratch
environment, drive the original `ConcatScheduler` for `N·E` iterations, and
compare against the lambda's output. The two must agree to within floating
point noise. A worked reference point, with `lr=0.001`, `lr_scale_factor=0.01`,
`epoch_length=7`, `num_epochs=5`:

| iteration | LR (Ignite) |
|---|---|
| 0 | 0.00001000 |
| 1 | 0.00015143 |
| 2 | 0.00029286 |
| … | … |
| 33 | 0.00016292 |
| 34 | 0.00013223 |

Those numbers are what a correct transcription produces. If yours differ, the
phase boundary or the cycle size is wrong.

⚠️ **One behavioural difference with `accumulation_steps > 1`.** Ignite steps
its scheduler on **every iteration** (`Events.ITERATION_STARTED`) but its
optimiser only every `N` iterations, so the LR runs ahead of the optimiser.
Lightning's `interval="step"` scheduler advances with the optimiser step, so
they move together. **Every shipped config has `accumulation_steps=1`**, where
the two are identical — but if you set it above 1, the LR curve changes. Decide
which semantics you want and say so.

---

## Phase 6 — The Trainer

Every `hparams` field that reaches the trainer:

| `hparams` | Lightning `Trainer` kwarg | Notes |
|---|---|---|
| `run_params["max_epochs"]` | `max_epochs` | ⚠️ On resume, `fit(ckpt_path=…)` **restores `max_epochs` from the checkpoint** — changing it in the config has no effect. Same trap as Ignite. |
| `run_params["epoch_length"]` | *(none)* | Ignite truncates an epoch to a fixed iteration count; Lightning has no equivalent. Every shipped config sets it to `None`, so it is moot. If you need it, truncate the dataloader. |
| `accumulation_steps` | `accumulate_grad_batches` | See the Phase 5 caveat. |
| `gradient_clip_val` | `gradient_clip_val` | The original's 23-line `clip_gradients` is a verbatim copy of Lightning's own code — delete it. |
| `eval_every` | `check_val_every_n_epoch` | ⚠️ In Ignite, `eval_every` is **two knobs in one field**: validation frequency *and* checkpoint frequency (the saver is attached to the evaluator's `COMPLETED`). Lightning **decouples** them. You must decide: checkpoint every validation, or every epoch? The original checkpoints every validation. |
| `add_pbar` | `enable_progress_bar` | — |
| `dist_params` | `strategy`, `devices`, `accelerator` | See Phase 9. |
| `logger_name` | `logger` | See Phase 8. |
| — | `default_root_dir` | The original had no equivalent; it wrote checkpoints to `checkpoint_params["save_dir"]`. |
| — | `num_sanity_val_steps` | The original ran no sanity check. Set it to 0 to keep the behaviour identical. |

---

## Phase 7 — Checkpointing and resuming

**Saving** (matrix G9). The original's
`Checkpoint(to_save, DiskSaver(save_dir, require_empty=False), n_saved=5, filename_prefix=prefix, score_function=score_function, score_name="score", global_step_transform=global_step_from_engine(trainer))`
maps onto:

```
ModelCheckpoint(
    dirpath=checkpoint_params["save_dir"],
    filename=checkpoint_params["prefix_name"],
    monitor=<the tracked metric>,
    mode="max" if metric_factor >= 0 else "min",
    save_top_k=checkpoint_params["n_saved"],
    save_last=True,
)
```

`save_last=True` is a genuine improvement the original lacked: it caps crash
loss at one epoch instead of nine.

⚠️ `save_top_k=0` means **save nothing**, not "save all"; `save_top_k=-1` means
save all. A classic off-by-one.

⚠️ The monitor name must be a metric you actually log, or the callback
**silently never fires**.

**Loading** (matrix G10). Two modes:

- `load_model_only=True` — the original restores only `{"model": …}` from the
  checkpoint file (the `Checkpoint` handler stores the weights under the
  `"model"` key). The port loads that key into the model and leaves the
  optimiser at its initial state. **This is the mode every shipped config uses**
  (`load_model_only = True` in all 21).
- `load_model_only=False` — a full resume: `trainer.fit(..., ckpt_path=checkpoint_dir)`
  restores model + optimiser + scheduler + epoch.

---

## Phase 8 — Logging

**Neptune** (matrix H1). The original's `MyNeptuneLogger` wraps **ignite's**
`NeptuneLogger`, which speaks the `neptune-client` **0.4.x** API. Lightning's
`NeptuneLogger` speaks **`neptune>=1.0`**. The client must be upgraded; this
is not optional and not a porting choice.

What must still be logged:

| when | names |
|---|---|
| every `log_every` iterations | `train/loss`, `train/<loss_key>`, `train/lr` |
| end of each training epoch | `train/<metric>` |
| end of each validation | `valid/<metric>` |

Plus `logger_params["params"]` (the flattened config) and `logger_params["tags"]`.

Lightning logs the learning rate itself under `lr-<optimizer_name>`, so the
`train/lr{i}` series is covered — but the *name* differs. If a downstream
Neptune dashboard depends on the exact names, map them in the logger.

**Print** (matrix H2). The original's `PrintLogger` writes to stdout every
`log_every` iterations. Lightning's progress bar already writes to stdout, so
the file logger can be disabled and a hand-rolled callback can reproduce the
per-iteration print.

⚠️ **The `every=N` filter** (matrix H4). The original registers
`Events.ITERATION_COMPLETED(every=log_every)`. `Trainer(log_every_n_steps=N)`
controls *logging* frequency globally — it is **not** a per-handler event
filter, and the repo's own analysis calls this a genuine regression in
expressiveness. A counter callback is the faithful replacement.

**Slack** (matrix H3). `os.environ["SLACK_URL"]` → one message per training
epoch, and **one per metric key** at validation end. Reimplement as a callback.

---

## Phase 9 — Distributed

`idist.Parallel(**dist_params)` → `Trainer(strategy="ddp", accelerator=…, devices=…)`.

⚠️ **`ImbalancedDatasetSampler` is not a `DistributedSampler`.** Under plain
Ignite + DDP every rank sees the same samples. Lightning's DDP sampler wrapper
shards a *custom* sampler rather than replacing it, so the inverse-frequency
weighting survives — **verify this**, because it is the difference between a
correct and an incorrect distributed run.

`manual_seed(42 + rank)` → `seed_everything(42 + trainer.global_rank, workers=True)`.
The per-rank offset is manual in Lightning; it is not automatic.

⚠️ **`Mixup` ignores the seed entirely** — `Mixup.__init__` hard-codes
`random_seed=1234`, so every rank draws the same `Beta(1,1)` sequence. Pre-existing
behaviour; unchanged by the port, and worth knowing before you interpret a
multi-rank run.

---

## Phase 10 — LR finder

`BaseEngine.lr_finder` uses Ignite's `FastaiLRFinder` with
`min_lr=3e-5`, `max_lr=10.0`, `num_iter=len(loader)-1`, and — before sweeping —
it **removes the validation handler and the scheduler**.

Lightning's `Tuner(trainer).lr_find(model, dataloader, min_lr=…, max_lr=…)`
has no callback-removal equivalent, so the sweep must run on a **throwaway
trainer** (a second `Trainer` with the same settings but no validation and no
scheduler) rather than the live one. That is the cleanest way to preserve the
original's intent.

---

## Phase 11 — The inference path and the vote

### The dump (matrix F6)

The original's `SedTest` is an ignite `Metric` that writes one JSON per audio
file. It has **four independent blockers** in the original tree:

1. **Syntax error** — a missing comma after `"filename": filename`
   (`src/metrics/sed_test.py:59`). This is the only file in `src/` that does
   not parse.
2. **`hparams.save_dir` and `hparams.combine_files` are not defined in any
   config** — only `checkpoint_params["save_dir"]` exists. The branch is
   unreachable.
3. **`SedEngine.prepare_batch(mode='test')` references an undefined local `x`**
   → `NameError` on the first call.
4. **`save_dir` must be a `pathlib.Path`** — a `str` has no `.mkdir`.

The port fixes all four. The **output format stays byte-identical**:

```json
{
  "filename": "...",
  "predicted_frames": [264 floats],
  "predicted_clip":  [264 floats],
  "targets":         [264 floats],
  "lraps_score_clip":  0.0,
  "lraps_score_frames": 0.0
}
```

written to `save_dir/{filename}.json`, with the optional merge into
`save_dir/final/merged_results.json` reproducing the original's text-level
concatenation (`'[' + ','.join(file_texts) + ']'`).

The dump is **max-pooled over the entire time axis**
(`pred_frames.max(axis=1)[0].max(axis=1)[0]`), which is exactly why it cannot
produce 5-second rows. Keep that for the dump — it is the original's
behaviour — and read the *frame* scores for the vote from a path that keeps
them.

### The vote (matrix K1–K6, all new)

No vote code exists in the repo. The rule is in `README.md:45` only:

> 4 of 13 models fire, with `clipwise_output ≥ 0.3` **and**
> `framewise_output ≥ 0.3`, within the audio snippet.

The 13 models are 4-fold no-mixup + 5-fold no-mixup + 4-fold mixup, all
`sed_dense121att`. The 9-model variant used 3 votes, clip 0.3 / frame 0.5.

The vote is **framework-agnostic** — it operates on saved score files, not on
tensors — so it should live in a module both ports can import, and the same
implementation must serve both. The four things the repo does *not* contain and
the vote therefore needs are listed as K3–K6: 5-second row slicing, site 1/2/3
row construction, the `nocall` fallback, and `submission.csv` emission.

The in-repo ancestor of the 0.3/0.3 rule is the literal in
`src/metrics/sed_f1score_frame.py:35,40`. Keep the two gates independent.

---

## Phase 12 — What does *not* improve

The repo's own analysis is explicit, and it is worth repeating because it stops
the port from being judged on line counts:

| Component | Why the framework does not help |
|---|---|
| The model | A `nn.Module` with log-mel inside `forward()`, a hand-rolled BatchNorm shuffle, two reduction dances, and a reshape that changes the meaning of the batch axis. |
| `AttBlock`'s `tanh`+softmax | A model-internal detail. |
| The dual heads | The model returns a dict. The framework does not care. |
| The `(scalar, dict)` loss contract | Lightning can return any loss; the dict is logged via `self.log_dict`. |
| The three metrics | Same interface, same code. |
| `SedDataset` | A `torch.utils.data.Dataset`. Untouched. |
| `ImbalancedDatasetSampler` | A `torch.utils.data.Sampler`. Untouched. |
| The vendored front-end | `nn.Module`s inside the model. Untouched. |
| Mixup | Split between `prepare_batch` (labels) and `preprocess` (log-mel). Untouched. |

Roughly 180 of the 375 lines of `base_engine.py` disappear; the remaining ~195
are SED-specific logic that Lightning cannot express. **A port that deletes
more than that has deleted something it should not have.**

---

## Traps, ranked by expected damage

1. **A `monitor` name that is never logged** → the checkpoint callback silently
   never fires, and you train for 50 epochs with no checkpoints.
2. **A non-identical LR schedule** → the run is quietly not the run that won.
   Verify against live Ignite (Phase 5).
3. **A metric name mismatch** — logging `f1score_clip` but monitoring
   `val_f1score_clip` (or the reverse) → the same silent no-checkpoint failure.
4. **Dropping the `dict_loss` keys** → the `loss_keys` instrumentation
   disappears, and the Neptune run loses the five per-key series.
5. **Applying a sigmoid in the metrics** → the `AttBlock` already applied it;
   a second sigmoid silently changes every F1.
6. **Averaging per-batch F1 instead of reducing then dividing** → a different
   number from the original's.
7. **Dropping the `SedF1ScoreFrame` 0.3 clip gate** → the frame metric
   degenerates to a frame-only metric and the checkpoint ranking changes.
8. **Checkpointing every epoch instead of every validation** → `eval_every=2`
   means twice as many checkpoints, and the "best" is now selected over a
   different set of validation points.
9. **Forgetting `load_model_only`** → a weights-only resume that silently also
   restores the optimiser, changing the effective learning rate.
10. **Using `LightningCLI`** → the 13-config reproducibility contract breaks.
11. **Assuming `seed_everything` fixes the mixup** → it does not; `Mixup` has
    its own hard-coded seed.
12. **Leaving `num_sanity_val_steps` at the default (2)** → the original ran
    no sanity check, and it costs a validation pass.

---

## Verification plan

Run these in order. Each one has caught a real class of porting bug.

| # | Check | Passes when |
|---|---|---|
| 1 | **LR schedule equivalence** | The ported lambda matches live Ignite for `N·E` steps, max delta < 1e-12 (Phase 5) |
| 2 | **Criterion identity** | `SedScaledPosNegFocalLoss(gamma=0, alpha_0=1, alpha_1=1, secondary_factor=1)` equals `nn.BCELoss()` to 1e-6 — i.e. the port did not "fix" the inert focal loss |
| 3 | **Metric identity** | The ported `SedF1ScoreClip` / `SedF1ScoreFrame` / `SedLRAPS` reproduce the ignite originals' values on a fixed synthetic batch |
| 4 | **Head wiring** | The loss receives only `clipwise_output`; `framewise_output` receives no gradient |
| 5 | **One-epoch parity** | With a fixed seed and a tiny synthetic dataset, the ported run's per-step loss matches the original Ignite run's to 1e-5 |
| 6 | **Checkpoint behaviour** | Exactly `n_saved` checkpoints exist, the best one is the highest `track_metric`, and `save_last` exists |
| 7 | **Resume behaviour** | A weights-only resume leaves the optimiser at its initial LR; a full resume restores the epoch counter |
| 8 | **Inference dump** | The JSON keys and value shapes match the original's `SedTest` exactly |
| 9 | **Vote** | A synthetic 13-model set with exactly 4 fires at ≥0.3/≥0.3 is accepted; 3 fires is rejected |
| 10 | **Config sweep** | All 21 config files construct and the port reads every attribute (the `hparams.smooth`, `hparams.num_classes`, `hparams.save_dir` etc. that the original referenced but never defined) |

Check 5 is the one that matters most: it is the only check that exercises the
whole control plane end to end. To run it you need the original Ignite path
working as well — which means installing `pytorch-ignite==0.4.0.post1`
alongside a torch it accepts, in a separate environment, and pointing both at
the same synthetic dataset.
