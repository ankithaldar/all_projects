# Porting worklist

The **file-by-file checklist** for porting the Cornell Birdcall solution from
PyTorch Ignite to PyTorch Lightning or TensorFlow/Keras.

This is the companion to [feature-parity-matrix.md](feature-parity-matrix.md),
which is organised by *capability*. This one is organised by *file* — it is
what you work through, in order, when you sit down to do the port. Every entry
says: what must survive the port, what will bite you, and how you know you
got it right.

| | |
|---|---|
| [Prerequisites](#0-prerequisites--resolve-these-first) | the environment, and two blockers that stop the original running today |
| [1. The front-end](#1-the-front-end--helperssed_audio_utilspy) | STFT → log-mel → SpecAugment, in-graph |
| [2. The model](#2-the-model--modelssed_modelspy) | the five PANNs variants |
| [3. The data](#3-the-data--dataloaders) | `SedDataset` and the sampler |
| [4. The augmentations](#4-the-augmentations--augmentations) | two waveform stacks and the mixup |
| [5. The losses](#5-the-losses--loss) | five live criteria |
| [6. The metrics](#6-the-metrics--metrics) | LRAP, two F1 metrics, the dump |
| [7. The control plane](#7-the-control-plane--engine) | the part the framework replaces |
| [8. The loggers](#8-the-loggers--logger) | Neptune, print, Slack |
| [9. The configs](#9-the-configs--config_params) | 21 configs, 39 attributes |
| [10. The entry point](#10-the-entry-point) | `sed_train.py` and its CLI |
| [11. Inference and the vote](#11-inference-and-the-vote) | the dump, and the six things that do not exist yet |
| [12. Weight conversion](#12-weight-conversion) | the 744-tensor checkpoint, and the 19 you must remap by hand |
| [Definition of done](#definition-of-done) | the checklist to close the port |

**Where the full behavioural spec lives.** Every behaviour below is stated in
full, with line numbers, in [`../src-reference/`](../src-reference/README.md) —
one page per package. Those pages are the source of truth; this worklist is
the execution order and the trap list.

| porting worklist section | full spec |
|---|---|
| 1. front-end | [src-reference/helpers.md](../src-reference/helpers.md) |
| 2. model | [src-reference/models.md](../src-reference/models.md) |
| 3. data | [src-reference/dataloaders.md](../src-reference/dataloaders.md) |
| 4. augmentations | [src-reference/augmentations.md](../src-reference/augmentations.md) |
| 5. losses | [src-reference/loss.md](../src-reference/loss.md) |
| 6. metrics | [src-reference/metrics.md](../src-reference/metrics.md) |
| 7. control plane | [src-reference/engine.md](../src-reference/engine.md) |
| 8. loggers | [src-reference/logger.md](../src-reference/logger.md) |
| 9. configs | [src-reference/config-params.md](../src-reference/config-params.md) |
| 10. entry point | [src-reference/entrypoints.md](../src-reference/entrypoints.md) |

---

## 0. Prerequisites — resolve these first

### The two API blockers

**Blocker 1 — `librosa.util.pad_center` (fatal, and it stops the original running today).**

`src/helpers/sed_audio_utils.py:58` calls:

```python
fft_window = librosa.util.pad_center(fft_window, n_fft)
```

librosa **0.10.0** made `size` keyword-only:

```
librosa 0.10.2:  pad_center(data, *, size, axis=-1, **kwargs)   -> TypeError: takes 1 positional argument but 2 were given
```

So on any modern librosa, `PANNsDense121Att.__init__` **raises**, and the
entire model cannot be constructed. This is not a porting problem — it means
**the original does not run today** either. Three ways out:

| option | consequence |
|---|---|
| pin `librosa<0.10` | also pins `numba` (librosa 0.9 needs numba ≤ 0.58) and `numpy<1.24`. Heavy, and it blocks the Keras port's `tf.signal` rewrite from being validated against the same env. |
| change the call to `pad_center(fft_window, size=n_fft)` | **one line**, and it is what librosa wants. A port should do this, and the ported front-end will not have the problem at all. |
| precompute the window outside the model | changes the design; only worth it for a Keras port that drops the vendored front-end entirely. |

**Blocker 2 — `torchvision.models.densenet121(pretrained=True)` (deprecated, not yet fatal).**

`src/models/sed_models.py:367, 492, 615, 738`. torchvision 0.13 deprecated
the `pretrained` boolean in favour of `weights=`; it still works in 0.29 with
a warning, and **will be removed**. All four `densenet{121,161,169,201}`
symbols still exist in torchvision 0.29. In Keras, `keras.applications` has
**only DenseNet121** — 161/169/201 have no equivalent, so a port that must
keep all five variants either ports the torchvision weights by hand or
declares the 121-only scope explicitly (the 121 is what all 13 ensemble
members use).

### The dependency matrix

| package | original | Lightning port | Keras port |
|---|---|---|---|
| `torch` / `torchvision` | *unpinned* | **pin them explicitly** — a Lightning pin implies a torch range, and the original's torch is "whatever the GCP image had" | not needed, except for the weight converter |
| `pytorch-ignite` | `==0.4.0.post1` | dropped | dropped |
| `lightning` | — | `>=2.2,<3` | — |
| `tensorflow` / `keras` | — | — | `tensorflow>=2.16` + `keras>=3.5`; **Python ≤ 3.13** (no stable TF wheel for 3.14) |
| `librosa` | *unpinned* | `>=0.10` **after** fixing the `pad_center` call | needed only to precompute the mel matrix; `tf.signal`'s is HTK and **not** a drop-in |
| `numba` | *unpinned* | `numba>=0.60` is fine once the `pad_center` call is fixed | ditto |
| `audiomentations` | `==0.11.0` | unchanged | unchanged — **numpy in, numpy out, drops into `tf.data.map` unchanged** |
| `scikit-learn` | *unpinned* | unchanged | **not usable in a graph** — see §6 |
| `neptune-client` | `==0.4.107` | **must become `>=1.0`** for `NeptuneLogger` | **must become `>=1.0`** for `neptune-keras` |
| `pandas`, `soundfile`, `tqdm`, `psutil` | *unpinned* | unchanged | unchanged |

### The seeding trap

`Mixup.__init__(self, mixup_alpha, random_seed=1234)` hard-codes the seed as
a default argument, and the engine constructs `Mixup(mixup_alpha=1.)` without
overriding it. **Every run, every epoch and every DDP rank draws the same
sequence of `Beta(1,1)` lambdas.** Port it faithfully and the port is
faithful; fix it and you have changed the experiment. Decide explicitly.

---

## 1. The front-end — `helpers/sed_audio_utils.py`

**Port this first.** Everything downstream depends on its output, and it is
the only part of the pipeline with non-trivial numerics.

| what | spec |
|---|---|
| `STFT` | a `Conv1d` with **frozen** DFT×window weights, `bias=False`; `hop_length` defaults to `win_length // 4`; `center=True` **reflect-pads** by `n_fft//2` first |
| `Spectrogram` | `real**2 + imag**2` — **power, not magnitude** |
| `LogmelFilterBank` | `melW = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax).T`, shape `(513, 64)`, frozen; `power_to_db` = `10*log10(clamp(x, 1e-10)) - 10*log10(max(1e-10, 1.0))` |
| `DropStripes` | zeroes `stripes_num` random bands along dim 2 (time) / dim 3 (freq), **in place, per batch item**; **no-op when not training** |
| `SpecAugmentation` | time-dropper (dim 2) then freq-dropper (dim 3); configured `time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2` |

**Traps**

- ⚠️ `amin = 1e-10` underflows to 0 in fp16 → `log10(0) = -inf` → NaN. This
  is why `USE_AMP` is hard-disabled. Any port that enables AMP must force
  `power_to_db` to fp32.
- ⚠️ `top_db` is `None` in every shipped config — the clamp is dead code in
  the final path.
- ⚠️ `LogmelFilterBank.forward` takes `(B, C, T)` and returns `(B, T, M)`;
  `Spectrogram` returns `(B, 1, T, F)`. Opposite axis orders.
- ⚠️ In Keras: `tf.signal.frame` has **no reflect padding** — pad explicitly
  or `frames_num` becomes 2977 instead of 3001.
- ⚠️ In Keras: keep **librosa's** mel matrix. `tf.signal`'s is HTK with
  unit-peak triangles; librosa's is Slaney with area normalisation.
- ⚠️ In Keras: `SpecAugmentation` must take an explicit `training` flag, and
  must draw an **independent** band per batch item.

**Acceptance check** — the ported front-end and the original's must produce
the same `logmel` on the same waveform, to 1e-4. That single check covers the
mel scale, the dB formula, the reflect padding and the power-vs-magnitude
question at once. It is the highest-value test in the whole port.

---

## 2. The model — `models/sed_models.py`

**Reuse it unchanged in the Lightning port** — it is plain `nn.Module` code
and Lightning does not touch it. In the Keras port it is a full rewrite.

| what | spec |
|---|---|
| 5 variants | `PANNsCNN14Att`, `PANNsDense121Att`, `PANNsDense161Att`, `PANNsDense169Att`, `PANNsDense201Att` |
| output | a **dict**: `{"clipwise_output": (B,C,O), "framewise_output": (B,C,T,O)}` |
| `AttBlock` | `softmax(tanh(att(x)), dim=-1) * sigmoid(cla(x))`, summed over time; returns `(x, norm_att, cla)` |
| `interpolate` | ×32 **interleaved** nearest-neighbour upsample on the time axis |
| `pad_framewise_output` | repeat the **last frame** until `frames_num` (25 times, because `93 × 32 = 2976 < 3001`) |
| `bn0` | `BatchNorm2d(mel_bins)`, applied **before** SpecAugment |
| mixup | applied to the **spectrogram inside the model**, after SpecAugment; the batch axis changes meaning mid-forward (`b = (b*c)//2`) |
| init | xavier-uniform + zero bias; BN bias 0 / weight 1; the **backbone is deliberately not re-initialised** |

**Traps**

- ⚠️ `apply_aug=False` **breaks mixup** — the `b = (b*c)//2` branch still
  fires against an unmixed spectrogram.
- ⚠️ `PANNsCNN14Att` does **not** accept `top_db`; the four Dense variants
  do. A config with `sed_ccnn14att` **and** `top_db` raises `TypeError`.
- ⚠️ `bn_att` and `temperature` are **dead parameters**.
- ⚠️ In Keras the `AttBlock` softmax must be over **time**: `Softmax(axis=-2)`.
  `axis=-1` is a silent wrong answer — it trains and scores badly.
- ⚠️ In Keras the DenseNet **applies ImageNet `preprocess_input`**, which
  torchvision's `.features` does not. Applied to dB-valued log-mel already
  normalised by `bn0`, it destroys the pretrained transfer with no error.
- ⚠️ In Keras, `tf.pad(SYMMETRIC)` is **not** "repeat the last frame"; use
  `tf.concat([x, tf.repeat(x[:, -1:, :], n, 1)], 1)`.

**Acceptance check** — the same input must give the same `clipwise_output` to
1e-4. In the Lightning port this is free; in the Keras port it is the single
hardest test in the project, and it should be written before the rest.

---

## 3. The data — `dataloaders/`

| what | spec |
|---|---|
| `SedDataset` | 10 kwargs; returns `{"filename", "waveforms", "all_labels", "primary_labels", "secondary_labels"}` |
| label coding | a **264-vector** one-hot, broadcast to every snippet — *not* a per-frame matrix |
| training slice | a short clip is placed at a **random** offset; a long clip is **randomly cropped** |
| validation slice | deterministic contiguous slices, tail zero-pad, long files truncated |
| transform order | the waveform transform is applied to the **whole file** *before* slicing |
| background rows | label-less WAVs appended as extra rows → hard negatives on all 264 classes |
| `ImbalancedDatasetSampler` | inverse-frequency weights, `torch.multinomial(..., replacement=True)` |

**Traps**

- ⚠️ `eval()` on the CSV's `secondary_labels` column — use `ast.literal_eval`.
- ⚠️ `item["duration"]` is a **sample count**, not seconds.
- ⚠️ `root_dir` may be a `Path` **or** a dict (5-archive sharding) — the dict
  branch is dead (`get_dict_value` is imported by every config and called by
  none).
- ⚠️ `ImbalancedDatasetSampler` is **not** a `DistributedSampler` — under DDP
  every rank can see the same samples. Lightning's DDP wrapper *shards* a
  custom sampler rather than replacing it, so the weighting survives — verify.
  Keras: `sample_from_datasets` is the declarative equivalent but **does not
  shard**.
- ⚠️ In Keras, `steps_per_epoch` cannot always be inferred after
  `sample_from_datasets` — compute `cardinality()` before any lossy op.

**Acceptance check** — a fixed synthetic batch must produce the same
`waveforms` and label tensors, and the same snippet count `(B, C, S)`.

---

## 4. The augmentations — `augmentations/`

| what | spec |
|---|---|
| `sed_default_augment` | `AddGaussianNoise(0.001, 0.015, p=0.3)` → `AddGaussianSNR(p=0.3)` → optional `AddBackgroundNoise(dir, p=0.5)` |
| `sed_background_augment` | a **palindromic bracket**: noise(p=0.2) → SNR(p=0.2) → Gain(−15..15 dB, p=0.3) → optional background(p=0.2) → optional short-noises(p=0.6) → **the noise/SNR/gain trio again** |
| `Mixup` | `lam ~ Beta(a,a)` per **pair**, even↔odd; `do_mixup` mixes them |

**Traps**

- ⚠️ The bracket applies the same transform **twice** — `Gain` can reach
  **+30 dB with no clipping guard**, and SNR is perturbed twice.
- ⚠️ `AddGaussianSNR`'s `min_SNR`/`max_SNR` are **linear amplitude ratios,
  not dB**; the default range is `1:1000 … 1:1`.
- ⚠️ The `transform(x, sr)` signature differs from the legacy
  `transform(x)` — the two families are not interchangeable.
- ⚠️ `Mixup`'s hard-coded seed (see §0).
- ✅ `audiomentations` is numpy-in/numpy-out — **it ports to Keras unchanged**.
  This is the one augmentation win.

**Acceptance check** — the composed transform must return an array of the
same length, and the mixup must produce `λ·x_even + (1−λ)·x_odd` on the
label tensors with the batch halved to `(B·C)//2`.

---

## 5. The losses — `loss/`

Only **five** criteria are reachable from the SED path. Port those five and
nothing else.

| `hparams.criterion_name` | class | `loss_keys` |
|---|---|---|
| `sed_scaled_pos_neg_focal_loss` | `SedScaledPosNegFocalLoss` | `bce_loss, F_loss, FScaled_loss, F_loss_0, F_loss_1` |
| `sed_scaled_pos_neg_focal_loss_augd` | `SedScaledPosNegFocalLossAugd` | same 5 |
| `bce_controlled_standard_loss` | `BCEControlledStandardLoss` | *(empty)* |
| `focal_loss_standard` | `FocalLossStandard` | `bce_loss, F_loss` |
| `sed_removed_focal_loss` | `SedRemovedFocalLoss` | `bce_loss, F_loss, FRemoved_loss` |

**The contract:** every criterion returns **`(scalar_tensor, dict_of_scalars)`**
and publishes `loss_keys`. The engine turns each key into a metric, so the
dict is instrumentation, not decoration.

**Traps**

- ⚠️ **The focal loss is inert at the shipped parameters.** `gamma=0` ⇒
  `(1−pt)^0 = 1`; `alpha_0 = alpha_1 = 1`; `secondary_factor = 1.0` ⇒
  `FScaled_loss == plain BCE`, bit for bit. **Reproduce the collapse.** A
  "fixed" focal loss is a different experiment with a different result.
- ⚠️ The prediction is **already sigmoided** (the `AttBlock` applied it).
  Applying a sigmoid again silently changes every number.
- ⚠️ Only `clipwise_output` ever reaches a loss. `framewise_output` is
  unsupervised — by design, and it is half the winning decision rule.
- ⚠️ `_augd` **randomly drops 70 % of the primary-species positions**
  (`primary_drop_out_factor=0.3`) using the global, unseeded RNG.
- ⚠️ `BCEControlledStandardLoss.loss_keys` is **empty** — it contributes zero
  extra metrics. That is correct; do not "fix" it.
- ⚠️ In Keras the **argument order inverts** (`call(y_true, y_pred)`), and
  `reduction` defaults to `sum_over_batch_size` (divides by `B`, not
  `B·S·O` — a factor of ~792,264).

**Acceptance check** — `SedScaledPosNegFocalLoss(gamma=0, alpha_0=1,
alpha_1=1, secondary_factor=1)` must equal `nn.BCELoss()` to 1e-6, and the
five `loss_keys` must appear as logged metrics.

---

## 6. The metrics — `metrics/`

| what | spec |
|---|---|
| `SedLRAPS` | sklearn LRAP per item on `clipwise_output.max(axis=1)[0]` vs `target[:,0,:]` |
| `SedF1ScoreClip` | micro-F1 on the clip head, with the **background branch** |
| `SedF1ScoreFrame` | micro-F1 on the frame head **ANDed** with a hard-coded **0.3** clip gate |
| `SedTest` | one JSON per audio file + an optional merge |

**Traps**

- ⚠️ **No sigmoid is applied in any of the three** — the `AttBlock` did it.
- ⚠️ **Reduce, then divide.** These are means of *per-item* scores, not
  global scores.
- ⚠️ The **0.3** in `SedF1ScoreFrame` is a literal, not a config value, and
  it is the in-repo ancestor of the winning ensemble rule. Keep it, and keep
  the two gates independent.
- ⚠️ `SedLRAPS` has **no background branch** — a label-less row scores a
  perfect 1.0. The F1 metrics do have the branch and score it 0.0. The three
  metrics disagree about the same rows.
- ⚠️ In Keras there is **no label-ranking metric of any kind**. LRAP must be
  reimplemented vectorised (~20 lines) or paid for with `tf.py_function` per
  item. `keras.metrics.AUC` is **not** LRAP (query-independent vs
  query-dependent).
- ⚠️ In Keras `keras.metrics.F1Score` defaults `threshold=None` = multiclass
  argmax — **categorically wrong** for 264 labels. Pass `threshold=0.5`.
- ⚠️ In Keras, metrics use `add_variable`, not `add_weight`; and they need
  `keras.saving.register_keras_serializable` + `get_config`/`from_config` or
  **all 13 checkpoints fail to reload**.

**Acceptance check** — the ported metrics must reproduce the originals' values
on a fixed synthetic batch, to 1e-6.

---

## 7. The control plane — `engine/`

This is the part the framework replaces, and the only part.

| original | Lightning | Keras |
|---|---|---|
| `Engine(self.train_step)` | `LightningModule.training_step` | `Model.train_step` override |
| `Engine(self.eval_step)` | `validation_step` | `Model.evaluate` / a `Callback` |
| `Events.EPOCH_COMPLETED(every=eval_every)` | `check_val_every_n_epoch` | ⚠️ `save_freq` as an int means **batches** |
| `Checkpoint(…, n_saved, score_function)` | `ModelCheckpoint(monitor, save_top_k)` | ⚠️ Keras 3 has **no `save_top_k`** |
| `Events.ITERATION_COMPLETED(every=N)` | ⚠️ no equivalent — hand-roll a counter | ⚠️ no equivalent |
| `handler.remove()` | ⚠️ no equivalent | ⚠️ no equivalent |
| `engine.state` | ⚠️ no single source of truth | ⚠️ no single source of truth |
| `idist.Parallel` | `strategy="ddp"` | `tf.distribute` |
| `clip_gradients` (23 lines) | `gradient_clip_val` | `clipnorm` |
| `lr_finder` (23 lines) | `Tuner.lr_find` | manual |
| `complete_clear` + `gc.collect()` | automatic | manual |

**Traps**

- ⚠️ **The LR schedule must be numerically identical to Ignite's** — see the
  Lightning guide, Phase 5, for the exact formulas and the reference values.
  This is the one place where the Lightning port can silently diverge.
- ⚠️ The **monitor name** must be a metric you actually log, or
  `ModelCheckpoint` never fires — silently.
- ⚠️ `eval_every` is **two knobs in one field** in Ignite (validation
  frequency *and* checkpoint frequency). Lightning decouples them; you must
  choose.
- ⚠️ `fit(ckpt_path=…)` restores `max_epochs` **from the checkpoint**, so
  raising it in the config on resume has no effect.
- ⚠️ `accumulation_steps > 1` changes the LR curve in Lightning (the
  scheduler advances with the optimiser step; Ignite's advances every
  iteration). Every shipped config is 1, where they coincide.
- ⚠️ In Keras the **mixup batch halving** must happen in a `train_step`
  override, because the batch axis changes meaning mid-forward.

**Acceptance check** — a one-epoch run against the original Ignite path on
the same synthetic data, per-step loss matching to 1e-5. That is the only
check that exercises the whole control plane end to end.

---

## 8. The loggers — `logger/`

| what | spec |
|---|---|
| per-iteration | `train/loss`, `train/<loss_key>`, `train/lr` |
| per-epoch | `train/<metric>` |
| per-validation | `valid/<metric>` |
| Slack | one message per training epoch; **one per metric key** at validation end |
| Neptune payload | `project_name`, `log_every`, `name`, `prefix_name`, `tags`, `params` |

**Traps**

- ⚠️ The Neptune **client version changes** — ignite's logger speaks
  `neptune-client` 0.4.x, Lightning's and `neptune-keras` both need
  `neptune>=1.0`.
- ⚠️ The `every=N` event filter has no equivalent in either framework.
- ⚠️ The Slack URL is read at **import time** in a bare `except`.
- ⚠️ `BaseEngine.train` swallows exceptions — a crashed run exits 0 and looks
  finished. Lightning and Keras both re-raise, which is a win but a changed
  operational contract.

**Acceptance check** — the metric names in the experiment tracker must match
the original's series exactly, or the dashboards break.

---

## 9. The configs — `config_params/`

**Reuse the 21 config files unchanged.** They are plain Python classes and
both ports read them the same way.

| what | spec |
|---|---|
| resolution | `importlib.import_module(<dotted path>)` → `module.Parameters()` |
| `Parameters` | **39 attributes** — the full table is in [src-reference/config-params.md](../src-reference/config-params.md) |
| `BIRD_CODE` | 264 species codes → class indices 0..263; the index order is the label order everywhere |

**Traps**

- ⚠️ The `--config` value is a **dotted module path**, not a file path.
- ⚠️ **21 attributes vary** across the 21 configs and 18 do not — the
  variation table is in the reference. Two of them bite:
  - `track_metric` is **inconsistent across the ensemble** (the 5-fold family
    tracks `f1score_frame` except fold 0, which tracks `f1score_clip`);
  - `weight_decay` spans **100×** (`0.0001` / `0.001` / `0.01`).
- ⚠️ **20 of the 22 files in `src/config_params/` are CRLF** — 79 of the 84
  `.py` files in the whole tree are. Line-anchored `grep` misses them.
- ⚠️ `hparams.save_dir` and `hparams.combine_files` are read by the live
  `sed_engine.py` but **defined by no config**.

**Acceptance check** — all 21 configs construct and the port reads every
attribute. A dataclass with required fields catches the gaps the original
cannot.

---

## 10. The entry point

Keep the argparse + `importlib` shape and the `--config` contract. Add
whatever the new framework needs (a `Trainer(...)` or a `model.compile(...)`),
but do **not** switch to `LightningCLI` or `absl.flags`: the 13-model
ensemble depends on 13 *named, individually reproducible* config files.

**Traps**

- ⚠️ `cd src` is mandatory — the tree has **no `__init__.py`** and works
  through implicit namespace packages.
- ⚠️ `torch.backends.cudnn.benchmark = True` is set at import time.
- ⚠️ The original's `idist.Parallel(**dist_params)` is single-process with
  every shipped config (`dist_params = {}`).

---

## 11. Inference and the vote

### The dump (port `SedTest`)

The original is broken in four ways; the port should fix all four and keep
the **output format byte-identical**:

```json
{"filename": "...", "predicted_frames": [264], "predicted_clip": [264],
 "targets": [264], "lraps_score_clip": 0.0, "lraps_score_frames": 0.0}
```

written to `save_dir/{filename}.json`, with an optional merge into
`save_dir/final/merged_results.json`.

The four fixes: the missing comma at `sed_test.py:59`; `save_dir` must be a
`pathlib.Path`; `hparams.save_dir` / `hparams.combine_files` are undefined in
every config; and `sed_engine.py:60` references an undefined local `x` in
the `mode='test'` branch.

**⚠️ The dump max-pools the entire time axis** — a 10-minute file yields one
record, not 120 five-second rows. That is fine for the dump (it is the
original's behaviour) and **fatal for the vote**, which needs the frame axis
preserved.

### The vote (does not exist in the original — new work)

| what | source of truth |
|---|---|
| 4 of 13 models must fire | `README.md:45` |
| clipwise ≥ 0.3 **and** framewise ≥ 0.3 | `README.md:45`; the in-repo ancestor is `sed_f1score_frame.py:35,40` |
| 5-second rows | external kernel only — **new** |
| site 1/2/3 rows | external kernel only — **new** |
| `nocall` fallback | external kernel only — **new** |
| `submission.csv` (`row_id`, `labels`) | external kernel only — **new** |

The 13 models are 4-fold no-mixup + 5-fold no-mixup + 4-fold mixup, all
`sed_dense121att`. The 9-model variant used 3 votes, clip 0.3 / frame 0.5.

**Make the vote framework-agnostic** — it operates on saved score files, so
both ports should import the *same* module, and the acceptance check is that
both produce a **byte-identical `submission.csv`**.

---

## 12. Weight conversion

The original's checkpoints are ignite `Checkpoint` files containing
`{"model": <state_dict>, "trainer": ..., "optimizer": ..., "scheduler": ...}`.
The `"model"` entry is a plain `torch` `state_dict`.

For the winning model (`PANNsDense121Att`, `classes_num=264`):

| | |
|---|---|
| tensors in the state_dict | **744** |
| total elements | **9,713,195** |
| trainable parameters | **9,628,768** (the README's "~9.63M") |
| frozen front-end elements | 1,083,456 |
| DenseNet backbone elements | 7,037,625 (725 tensors, 31 top-level modules) |

**The 19 tensors a port must remap by hand** — everything else is the
torchvision DenseNet, which `keras.applications.DenseNet121` supplies (with
the ImageNet-`preprocess_input` caveat from §2):

| state_dict key | shape |
|---|---|
| `spectrogram_extractor.stft.conv_real.weight` | (513, 1, 1024) |
| `spectrogram_extractor.stft.conv_imag.weight` | (513, 1, 1024) |
| `logmel_extractor.melW` | (513, 64) |
| `bn0.weight` / `bn0.bias` | (64,) |
| `bn0.running_mean` / `bn0.running_var` | (64,) |
| `bn0.num_batches_tracked` | () |
| `fc1.weight` / `fc1.bias` | (1024, 1024) / (1024,) |
| `att_block.att.weight` / `.bias` | (264, 1024, 1) / (264,) |
| `att_block.cla.weight` / `.bias` | (264, 1024, 1) / (264,) |
| `att_block.bn_att.weight` / `.bias` | (264,) |
| `att_block.bn_att.running_mean` / `.running_var` | (264,) |
| `att_block.bn_att.num_batches_tracked` | () |

**Note the two frozen buffers.** `conv_real`/`conv_imag` are the STFT's
DFT×window filters and `melW` is the mel matrix — in Keras these are
**constants**, not weights, and a converted checkpoint should **not** carry
them. The BatchNorm `running_mean`/`running_var`/`num_batches_tracked` do
carry over.

**The other four variants** differ only in the backbone and in `fc1`:
`fc1` is 1024→1024 (CNN14, Dense121), 2208→2048 (Dense161), 1664→1024
(Dense169), 1920→1024 (Dense201). Only Dense121 has a Keras equivalent.

---

## Definition of done

A port is finished when **all** of these are true.

**Coverage**
- [ ] Every row of [feature-parity-matrix.md](feature-parity-matrix.md)
      A–J is ticked.
- [ ] Every row of §K (the vote) is implemented or explicitly deferred with a
      reason.
- [ ] All 21 configs construct and the port reads all 39 attributes.
- [ ] The eleven out-of-scope items are still out of scope (or their
      de-scoping is recorded).

**Numerics**
- [ ] The log-mel matches the original to 1e-4 on a fixed waveform.
- [ ] `clipwise_output` matches the original to 1e-4 on a fixed input.
- [ ] The criterion at `gamma=0` equals `nn.BCELoss()` to 1e-6.
- [ ] The three metrics reproduce the originals' values to 1e-6.
- [ ] The LR curve matches Ignite's, step for step, to 1e-12.
- [ ] A one-epoch run matches the original's per-step loss to 1e-5.

**Operations**
- [ ] Checkpoints keep the best `n_saved` by the tracked metric, and the
      monitor name resolves.
- [ ] Weights-only resume leaves the optimiser at its initial state.
- [ ] The full-resume path restores the epoch counter.
- [ ] The Neptune run carries the same metric series.
- [ ] The inference dump's JSON keys are byte-identical to the original's.

**The two that are not about the code**
- [ ] The 13-model vote produces a submission — and **both** ports produce
      the same one.
- [ ] The fold CSVs (or their generator, with the seed) are preserved, so
      the ensemble can be re-derived.

---

## What *not* to port

**37 of the 85 files** are unreachable from any shipped config, and a further
two symbols (`AUDIO_CONFIG`, `get_dict_value`) are dead inside a live file.
Porting them faithfully ports their bugs.

| do not port | why |
|---|---|
| `engine/main_engine*.py` | the pre-SED line; also broken (`ls_fn`, undefined hparams) |
| `train.py`, `train_v2/3/4.py` | the same, in entry-point form |
| `dataloaders/audio_dataset*.py` | broken on `get_snippets` / `mono_to_color` |
| `augmentations/base_augment.py`, `default_augment.py`, `spec_augment.py` | legacy image pipeline; `spec_augment.py` fails at import |
| `models/aux_*.py`, `classifier_*.py`, `densenet121.py`, `densenet161.py` | legacy model zoo |
| the other 11 loss classes | only `main_engine*` selects them |
| `metrics/custom_*.py` | the legacy metric family |
| `augmentations/noise_maker.py` | imported by nothing; the pink-noise WAVs came from an external kernel |
| `configs.AUDIO_CONFIG` | defined, never instantiated |
| `configs.get_dict_value` | imported by every config, called by none |

The full argument for each exclusion is in
[ports/README.md](README.md#scope).
