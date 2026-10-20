# TensorFlow / Keras 3 implementation guide

Step-by-step specification for porting the Cornell Birdcall solution from
PyTorch Ignite to **TensorFlow 2.x + Keras 3**.

Read [feature-parity-matrix.md](feature-parity-matrix.md) alongside this, and
read [code-issues-and-framework-wins.md](code-issues-and-framework-wins.md)
first if you have not already decided the port is worth doing.

The *why* behind every structural choice is in
[`../framework-comparisons/tensorflow-keras-port.md`](../framework-comparisons/tensorflow-keras-port.md).
This document is the *what*.

**This is the harder port, and it is not close.** The Lightning port reuses the
entire numeric core; this one rewrites it. The model, the front-end, the
losses, the metrics, the dataset and the sampler are all rewritten. The
control plane is *also* rewritten, and it is the part that maps worst.

---

## Phase 0 — Environment

| Package | Original | Port requirement |
|---|---|---|
| `tensorflow` | — | `>=2.16` (Keras 3 is the default `tf.keras`). Verified working: TF 2.20 / Keras 3.15 on Python 3.12–3.13. |
| `keras` | — | `>=3.5`. **Do not install `tf-keras`** unless you specifically need the Keras 2 API — it shadows `tf.keras` and changes `ModelCheckpoint`'s signature. |
| `torch`, `torchvision` | *unpinned* | **Dropped.** Nothing in the port needs them — with one exception, see the mel matrix below. |
| `librosa` | *unpinned* | ⚠️ **Still needed** for the mel filterbank. `tf.signal.linear_to_mel_weight_matrix` is HTK with unit-peak triangles; `librosa.filters.mel` is Slaney with area normalisation. They are **different matrices** for the same `(sr, n_fft, n_mels, fmin, fmax)`. Keep librosa's — and precompute it to a `.npy` at build time so the port does not need librosa at runtime. |
| `numba` | — | Pulled in by librosa. `numba>=0.60` removed `numba.decorators`, which librosa 0.10's `util/decorators.py` imports unguarded — **pin `numba<0.60` with `numpy<1.27`, or avoid importing `librosa.filters` at runtime.** |
| `audiomentations` | `==0.11.0` | **Unchanged and free.** It is numpy-in / numpy-out and drops into `tf.data.map` unchanged. This is a genuine parity win. |
| `SoundFile` | `==0.10.3.post1` | Unchanged. |
| `pandas` | *unpinned* | Unchanged (fold CSVs). |
| `scikit-learn` | *unpinned* | ⚠️ **Not usable inside a Keras graph.** `label_ranking_average_precision_score` and `f1_score` do not trace. The F1 metrics can be reimplemented in TF (easy); **LRAP cannot** and needs its own vectorised implementation. |
| `neptune-client` | `==0.4.107` | ⚠️ **Must be upgraded** to `>=1.0` for `neptune-keras`. The original's client speaks the 0.4.x API. |
| `psutil`, `tqdm` | `==5.6.7`, `==4.35.0` | Unchanged. |

**Python version.** TF 2.20 supports Python 3.9–3.13. Python 3.14 has **no
stable TF wheel** — only a release candidate paired with `keras-nightly`. Use
3.12 or 3.13.

**The `add_weight(trainable=False)` mechanism.** The original's frozen STFT and
mel buffers are `nn.Parameter` with `requires_grad = False`. In Keras the
direct counterpart is `add_weight(..., trainable=False)`, which also keeps the
buffer out of `model.trainable_variables` and therefore out of the optimiser —
which matters, because the original passes `model.parameters()` to AdamW.

**Precompute the mel matrix.** librosa is a heavy dependency for one matrix.
Compute `librosa.filters.mel(sr=32000, n_fft=1024, n_mels=64, fmin=50,
fmax=14000).T` once, save it, and load it with `np.load`. Record the librosa
version in the file name.

### ⚠️ The original's `pad_center` call is incompatible with modern librosa

`src/helpers/sed_audio_utils.py:58` calls

```python
fft_window = librosa.util.pad_center(fft_window, n_fft)
```

librosa **0.10.0** made `size` keyword-only, so on any modern librosa the
model cannot be constructed at all:

```
TypeError: pad_center() takes 1 positional argument but 2 were given
```

For a Keras port this is **moot** — the vendored front-end is replaced by
`tf.signal` wholesale (§2a), so the call disappears. But it matters for the
*validation* step: to compare the ported front-end against the original you
must be able to run the original, so the comparison environment needs
`librosa<0.10` (which pins `numba` and `numpy` down with it) **or** the
one-line fix `pad_center(fft_window, size=n_fft)`.

### ⚠️ `torchvision.models.densenet121(pretrained=True)` is deprecated

`src/models/sed_models.py:367, 492, 615, 738`. Deprecated in torchvision 0.13
in favour of `weights=`; still works in 0.29 with a warning, and **will be
removed**. All four `densenet{121,161,169,201}` symbols still exist in
torchvision 0.29, so the *conversion* of all five variants is possible —
but `keras.applications` has **only DenseNet121**. 161/169/201 have no Keras
equivalent, and since all 13 ensemble members use the 121, restricting the
Keras port to it is a defensible decision — but it must be an explicit one,
not an accident.

---

## Phase 1 — Entry point and config

Same as the Lightning port: keep `importlib` + `module.Parameters()` + the
`--config` dotted path. All 21 config files keep working unchanged.

`class Parameters` → a `@dataclass` is the natural translation, but it is
**optional**. The only thing it buys is validation of the attribute set, and
that is worth having — the original references `hparams.smooth`,
`hparams.num_classes`, `hparams.save_dir` and `hparams.combine_files`, none
of which any config defines.

**Do not use `keras_tuner`.** It optimises a search space; the solution needs
13 frozen, individually reproducible configs. The existing analysis is
explicit: it "would be actively wrong for a 13-named-checkpoint ensemble".

---

## Phase 2 — The model

This is the largest single piece of work.

### The shape trace you are porting towards

For `sed_dense121att`, `B=28`, `C=1` (train):

| stage | shape |
|---|---|
| `input_x` | (28, 1, 960000) |
| `input_x.reshape(b*c, s)` | (28, 960000) — the snippet axis folds into the batch |
| Spectrogram | (28, 1, 3001, 513) |
| LogmelFilterBank | (28, 1, 3001, 64) |
| transpose 1↔3 | (28, 64, 3001, 1) |
| `bn0` BatchNorm2d(64) | (28, 64, 3001, 1) |
| transpose 1↔3 | (28, 1, 3001, 64) |
| expand to 3 channels | (28, 3, 3001, 64) |
| `densenet121.features` | (28, 1024, 93, 2) — **measured, not assumed** |
| mean over the frequency axis | (28, 1024, 93) |
| max_pool1d + avg_pool1d, k=3 s=1 p=1 | (28, 1024, 93) |
| `fc1` 1024→1024 | (28, 1024, 93) |
| `att_block` att/cla Conv1d 1024→264 | (28, 264, 93) |
| `norm_att` softmax over time | (28, 264, 93) |
| `clipwise_output` = sum over time | (28, 264) |
| `segmentwise_output` (cla transposed) | (28, 93, 264) |
| `interpolate` ×32 | (28, 2976, 264) |
| `pad_framewise_output` to 3001 | (28, 3001, 264) — **25 frames of tail repeat** |
| final reshape | `framewise_output` (28, 1, 3001, 264), `clipwise_output` (28, 1, 264) |

The DenseNet number is worth stating precisely: `densenet121(weights=None).features`
on a `(1, 3, 3001, 64)` tensor gives `conv0 → (1,64,1501,32)`,
`transition1 → (1,128,375,8)`, `transition2 → (1,256,187,4)`,
`transition3 → (1,512,93,2)`, final **`(1,1024,93,2)`**. The transitions use
*valid* pooling, which is why 93 and not 94. Hence `interpolate_ratio = 32` and
`93 × 32 = 2976 < 3001`.

⚠️ **Measure the Keras equivalent.** `keras.applications.DenseNet121` may
pool differently. Do not assume 93 — assert it.

### 2a. The front-end

The vendored `sed_audio_utils.py` collapses from ~250 lines to ~6:

```
x       = tf.pad(inputs, [[0,0],[512,512]], mode="REFLECT")
frames  = tf.signal.frame(x, frame_length=1024, frame_step=320)     # (B*C, 3001, 1024)
spec    = tf.abs(tf.signal.rfft(frames, fft_length=[1024])) ** 2    # (B*C, 3001, 513)
mel     = tf.matmul(spec, melW)                                    # (B*C, 3001, 64)
logmel  = 10.0 * tf.math.log10(tf.maximum(mel, 1e-10))
```

Four things must be exactly right:

1. **The reflect pad is explicit.** `tf.signal.frame` has no reflect padding.
   Without it, `frames_num` becomes **2977** instead of 3001, and
   `pad_framewise_output` then mis-pads. Silent, and it shifts the model.
2. **Feed power, not magnitude.** The original computes `real² + imag²`. Feed
   magnitude and you get a factor-of-2 error that no test catches.
3. **Keep librosa's mel matrix.** `tf.signal`'s is HTK, unnormalised. The
   mismatch is partly muted by the `bn0` sitting directly downstream — but only
   because `bn0` is directly downstream, and only until you change the order.
4. **The dB formula is one line.** `10*log10(clamp(x, amin=1e-10))` with
   `ref=1.0` — there is **no `tf.signal.magnitude_to_db`** (all 34 `tf.signal`
   symbols were enumerated to confirm). Do not build on `tf.raw_ops.MagnitudeToDb`;
   its existence could not be confirmed.

`top_db` is `None` in every shipped config, so the clamp is dead code in the
final path. If you need it: `tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)`.

### 2b. The BatchNorm transpose dance disappears

The original transposes twice solely to land PyTorch's channel axis on axis 1.
Keras is channels-last:

```
self.bn0 = keras.layers.BatchNormalization(axis=-1)
```

`(B*C, 3001, 64)` → normalised over the 3001 axis, per mel bin. **Semantically
identical, two transposes deleted.** The same applies to `AttBlock.bn_att`
(`BatchNorm1d(264)` over the time axis → `BatchNormalization(axis=-1)`).

⚠️ **Keep the ordering:** `bn0` → SpecAugment. The transpose removal must not
move the normalisation relative to the augmentation.

### 2c. `AttBlock` — the axes-from-the-right trap

The original, on `(n_samples, n_in, n_time)`:

```
norm_att = softmax(tanh(att(x)), dim=-1)      # softmax over n_time
cla      = sigmoid(cla(x))
out      = sum(norm_att * cla, dim=2)         # sum over n_time
```

In Keras the tensor arrives channels-last, so the time axis is **-2** and the
class axis is **-1**. `Softmax(axis=-1)` is a **silent wrong answer**: the
softmax moves from time to classes. It trains. It scores badly.

**This is the single highest-probability silent failure in the entire port.**
Write the axis with a comment, and assert the attention sums to 1 over the
time axis in a test.

### 2d. The frame tail padding

`pad_framewise_output` repeats the **last frame** 25 times.

```
tf.concat([x, tf.repeat(x[:, -1:, :], n, 1)], axis=1)
```

⚠️ `tf.pad(..., mode="SYMMETRIC")` reflects the whole tensor — wrong.
`tf.pad(..., mode="CONSTANT")` injects 25 frames of "no bird" — also wrong, and
it silently poisons the frame head that the vote consumes.

### 2e. The remaining shape operations

| Original | Keras | Note |
|---|---|---|
| `x[:, :, None, :].repeat(1,1,ratio,1).reshape(...)` | `tf.repeat(x, ratio, axis=1)` | `tf.repeat` is **interleaved** — each frame repeated `ratio` times consecutively. Exactly right. |
| `x.expand(x.shape[0], 3, x.shape[2], x.shape[3])` | `tf.broadcast_to` | ⚠️ **not** `tf.tile` — `tile` copies. |
| `torch.mean(x, dim=3)` | `tf.reduce_mean(x, axis=-1)` | the frequency axis is last in Keras |
| `F.max_pool1d` / `F.avg_pool1d` | `MaxPool1D` / `AveragePool1D` | `padding="same"`, k=3, s=1 |
| `F.dropout(p, training=self.training)` | `keras.layers.Dropout(p)` | the `training` flag must be threaded |
| `models.densenet121(pretrained=True).features` | `keras.applications.DenseNet121(include_top=False, weights="imagenet")` | ⚠️ see 2f |

### 2f. The DenseNet transfer trap

⚠️ **`keras.applications.DenseNet121` applies ImageNet `preprocess_input`
internally** — `x /= 255.0` then mean/std normalisation. torchvision's
`.features` applies **nothing**.

The input here is dB-valued log-mel already normalised by `bn0`. Applying
ImageNet normalisation to it **destroys the pretrained transfer, with no
error**. Bypass it: use `include_top=False` and take the layers up to the last
convolution block, or re-tune from scratch (which invalidates the ensemble).

⚠️ **`keras.applications` has no DenseNet161, 169 or 201.** Four of the five
model variants have no direct Keras equivalent. Options: port the torchvision
DenseNet161/169/201 weights into a Keras implementation, or port only
`sed_dense121att` — which is the model **all 13 ensemble members use**
(`README.md:47`: "Ensemble diversity across folds/splits/mixup, not across
architectures"). Porting only the 121 is defensible and should be an explicit
decision, not an accident.

### 2g. SpecAugment

`DropStripes` and `SpecAugmentation` become one Keras layer with:

- ⚠️ an explicit `training` argument — a layer that omits `training` from
  `call()` applies SpecAugment **at inference**, which corrupts
  `framewise_output` — exactly the tensor the 13-model vote consumes.
- ⚠️ **per-batch-item** band randomness. The original draws an independent
  `bgn` per batch element. A shared `axis_mask` gives the whole batch the same
  band, which weakens the augmentation with no symptom. Use per-item masks.
- The original mutates in place; use masks instead.

### 2h. The mixup batch halving

The batch axis **changes meaning** inside `forward()`: at mixup time
`b = (b*c)//2; c = 1`, and `framewise_output.reshape(b, c, T, O)` undoes it.

In PyTorch a `reshape` of a contiguous tensor is free metadata and changing
what an axis means two lines apart is normal. **In Keras this forces a
`train_step` override** — mandatory, not optional — because the halving must
happen where the label tensor is also halved.

⚠️ `Model.train_step` is a documented, stable Keras 3 API, but the `master`
branch is mid-refactor. **The durable extension point is `compute_loss`**;
`train_step` remains the right place for the mixup halving, but **pin your
Keras version and expect churn**.

---

## Phase 3 — The dataset

`tf.data` is the natural fit and the pipeline is simple, but three behaviours
must be preserved exactly.

**The waveform transform runs on the whole file, before slicing** (matrix D3).
`audiomentations` is numpy-in/numpy-out, so it drops into `tf.data.map`
unchanged — but the *order* is load-bearing: augmenting per-snippet is a
different experiment.

**Label coding** (matrix C5): one-hot over 264 classes, **broadcast to every
snippet**. A weak file-level label becomes every frame's target. The result
is a 264-vector, not a 3001×264 matrix.

**Snippet slicing** (matrix C6/C7): random placement padding for training,
deterministic slicing for validation.

⚠️ **`eval()` on the CSV's `secondary_labels` column** — do **not** carry it
into the TF port. Use `ast.literal_eval` (same result, no arbitrary code
execution). The original's NaN cells raise `TypeError` today; the port should
not.

**The imbalanced sampler.** `tf.data.Dataset.sample_from_datasets` is the
declarative equivalent of `ImbalancedDatasetSampler` and is **strictly better**
— 6 lines against 25. But ⚠️ it **does not shard across ranks**: under
`tf.distribute` every replica sees the same samples. The Lightning port gets
sharding for free from Lightning's DDP sampler wrapper; here you must shard
the index space yourself.

⚠️ `steps_per_epoch` cannot always be inferred after `sample_from_datasets`
(or after `.filter()`, which makes the cardinality `UNKNOWN`). Compute
`cardinality()` **before** any lossy operation and assert it is finite.

---

## Phase 4 — The loss

**The `(scalar, dict)` contract has no Keras equivalent.** A `keras.losses.Loss`
returns a scalar. The `loss_keys` dict is a first-class part of this
solution's instrumentation — it produces 5 extra metrics per run in Neptune.

The solution: a `keras.metrics.Metric` per loss key, updated inside the loss's
`call` (or in a custom `train_step`), each with `add_variable` state. Keras 3
metrics use **`add_variable`**, not `add_weight` — `add_weight` is the layer
API. Mixing them up is a loud failure, so it is cheap, but it is a common
copy-paste error.

**The argument order inverts.** PyTorch: `criterion(y_pred, y_target)`. Keras:
`call(y_true, y_pred)`. Identical shapes, identical dtypes, **no error**. Use
keyword arguments everywhere, and override
`compute_loss(self, x, y, y_pred, sample_weight, training=True)` — the
durable extension point.

**`reduction` defaults to `sum_over_batch_size`**, which divides by `B`, not by
`B·S·O` — a factor of ~792,264 with the shipped shapes. Set
`reduction="sum_over_batch_size"` **and** do the `.mean()` inside `call`, or
use `reduction=None` and reduce yourself.

**The sigmoid is inside the model.** The `AttBlock` already applies it, so
every loss must use `from_logits=False`. Applying it twice silently changes
every number.

**Port the inertness.** `SedScaledPosNegFocalLoss` with `gamma=0` is exactly
BCE. The Keras port must reproduce `(1-pt)^gamma` **and** the collapse —
otherwise the port is not the same experiment. `BinaryCrossentropy` with
`from_logits=False` happens to match the collapsed form; but port the general
form so the `gamma` knob still exists.

**`tf.stop_gradient` replaces `torch.no_grad()`** for the mask construction
inside the loss.

---

## Phase 5 — The metrics

**There is no Keras label-ranking metric of any kind.** `SedLRAPS` must be
reimplemented — vectorised, roughly 20 lines — or paid for with
`tf.py_function` per item (which will not trace and will be slow). This is the
single hardest metric to port.

**`keras.metrics.AUC(multi_label=True, average="macro")` is NOT LRAP** —
query-independent versus query-dependent. It would corrupt `track_metric` in
the `final_sed_dense121_mix_*` configs.

**`keras.metrics.F1Score`'s `threshold` defaults to `None`, which means
multiclass argmax** — categorically wrong for a 264-label multi-label
problem. Pass `threshold=0.5` explicitly.

**The background branch and the dual-threshold AND must be hand-written.**
Neither exists in the Keras metrics:

- empty target ⇒ 1.0 only if nothing fires;
- the frame metric is `(pf >= 0.5) & (pc >= 0.3)` — the 0.3 is the in-repo
  ancestor of the winning ensemble threshold and must stay a literal.

**Registration.** `keras.saving.register_keras_serializable(package, name)`
is the API — there is **no `keras.saving.register_keras_metric`**. The class
needs `get_config()` / `from_config()`. **Without it, all 13 checkpoints fail
to reload.**

**`add_variable`, not `add_weight`** — see Phase 4.

**No sigmoid in the metrics** — the `AttBlock` already applied it.

---

## Phase 6 — The training loop

`model.compile(optimizer=…, loss=…, metrics=…)` then `model.fit(...)`.

The `fit` knobs that matter:

| Original | Keras | Note |
|---|---|---|
| `accumulation_steps` | `optimizer.gradient_accumulation_steps` | ✅ |
| `gradient_clip_val` | `optimizer.clipnorm` | ✅ — but the original's is **norm-2** clipping with `EPSILON=1e-6`; `clipnorm` is also norm-2. Disabled in every shipped config anyway. |
| `max_epochs` | `fit(epochs=…)` | ⚠️ Keras does **not** restore the epoch counter on resume — `fit(initial_epoch=N)` is manual. |
| `eval_every=2` | ⚠️ `ModelCheckpoint(save_freq=…)` as an **int means batches**, not epochs — a silent unit change | validation cadence itself is `validation_data=…` + `validation_freq=…`, but `validation_freq` is in **epochs** while `save_freq` as an int is in **batches**. Read both docstrings. |
| `run_params["epoch_length"]` | `fit(steps_per_epoch=…)` | ⚠️ may not be inferrable — see Phase 3. |
| — | `compile(jit_compile="auto")` | ⚠️ **Keras 3's `fit()` has no `jit_compile`** — it is on `compile()`. `fit(jit_compile=True)` is Keras 2 and is silently ignored. |
| `add_pbar` | `keras.callbacks.ProgbarLogger` | ✅ |
| `USE_AMP` | `keras.mixed_precision` | ⚠️ **The same wall.** `10*log10(1e-10)` underflows in fp16 → `log10(0) = −inf` → NaN. Force `power_to_db` to fp32, and **validate before trusting it**. |

**The `every=N` filter** (matrix H4) has no Keras equivalent. The original's
`Events.ITERATION_COMPLETED(every=log_every)` is a per-handler filter;
Keras callbacks have no event filter at all. A hand-rolled counter in a custom
callback is the faithful replacement.

**The `engine.state` single source of truth is gone.** Nine ordering-sensitive
registrations collapse into a callback list whose order silently matters.

**`fit()` raises on a crash** — the same improvement as Lightning. The
original's `train()` swallowed the traceback and closed the Neptune run
normally, so a crashed run looked finished.

---

## Phase 7 — The LR schedule

`keras.optimizers.schedules` is the natural home and the two-phase concat
becomes a `piecewise` or a single lambda.

**But the formulas must be identical to Ignite's** — the same requirement as
the Lightning port, and the same reference values. Transcribe:

```
LinearCyclical:  p = event_index / cycle_size
                 v = end_value + (start_value - end_value) * abs(p - 0.5) * 2
CosineAnnealing: p = event_index / cycle_size
                 v = start_value + (end_value - start_value) / 2 * (1 - cos(pi * p))
```

with the ConcatScheduler boundary: the linear phase for exactly `epoch_length`
iterations, then the cosine phase with its own counter restarting at 0.

⚠️ **A Keras schedule steps per *batch* by default**, which matches the
original (the Ignite scheduler is attached to `ITERATION_STARTED`). But if you
use `keras.optimizers.schedules.CosineDecay` — which steps **per epoch** — the
curve is wrong. Use a batch-stepped schedule, and verify the first few values
against the reference table in the Lightning guide (they are framework
independent).

---

## Phase 8 — Checkpointing and resuming

**Keras 3's `ModelCheckpoint` has no `save_top_k`.** The verified signature is
`(filepath, monitor, verbose, save_best_only, save_weights_only, mode,
save_freq, initial_value_threshold)`. The original's `n_saved=5` keep-best-N
behaviour needs a **custom callback plus file management** — write it, do not
assume it exists.

**Keras does not restore the epoch counter.** `fit(initial_epoch=N)` is manual.

**`load_weights` does not restore optimizer slots.** For a true resume use
`tf.train.Checkpoint` + `tf.train.CheckpointManager`, which is a *different
programming model* (manual variable and step management) rather than a
declarative object set.

The mitigation: **the original's resume path is weights-only**
(`load_model_only = True` in all 21 configs), so the gap is largely
theoretical. Implement weights-only first, and only then decide whether a full
resume is worth the `tf.train.Checkpoint` complexity.

---

## Phase 9 — Logging

⚠️ **The Neptune client must be upgraded** from 0.4.x to `>=1.0` for
`neptune-keras`. This is the same unavoidable consequence as in the Lightning
port, and it applies to the *original* too if you ever wanted to run it
side by side.

The logged series must be preserved (matrix H1): per-iteration `train/loss`,
`train/<loss_key>`, `train/lr`; per-epoch `train/<metric>`; at validation end
`valid/<metric>`; plus `params` and `tags`.

The Slack webhook (matrix H3) is a custom callback: one message per training
epoch, and **one per metric key** at validation end.

---

## Phase 10 — Distributed

`tf.distribute.MirroredStrategy` (single node) or `MultiWorkerMirroredStrategy`
(multi node).

⚠️ **`strategy.run_id` could not be verified** to exist on
`MirroredStrategy` or `MultiWorkerMirroredStrategy`. Use the verified idiom
instead: `tf.distribute.get_replica_context().replica_id_in_sync_group`.

⚠️ **VRAM is preallocated by default**, and
`tf.config.experimental.set_memory_growth(logical_devices, True)` is
**process-global** and must run **before any GPU operation**. Fatal for a
13-model ensemble on a 16 GB T4.

⚠️ **The imbalanced sampler does not shard** (Phase 3).

---

## Phase 11 — Inference and the vote

The same four blockers on `sed_test.py` (see the Lightning guide, Phase 11),
the same JSON output format, and the same six new items (K1–K6) that do not
exist in the original.

**The vote is framework-agnostic** — it operates on saved score files. Put it
in a shared module so the Lightning port and the Keras port use the *same*
implementation and produce the *same* submission. That is the only way to
compare the two ports at the level that matters.

---

## Phase 12 — The trade-offs, stated honestly

| Verdict | Items |
|---|---|
| **Easier** | The front-end (~250 lines → ~6), the DenseNet transfer, the BatchNorm shuffle, the metrics plumbing, the LR schedule, the imbalanced sampler, and a pile of small plumbing (`clipnorm`, `gradient_accumulation_steps`, `Dropout`, the sigmoid clamp) |
| **Harder** | The control plane (no priority, no removal, no `engine.state`, no `every=N`), the in-graph front-end as a Keras layer, the `AttBlock` axis, the LRAP metric (no equivalent exists), the F1 thresholds, the `(scalar, dict)` loss contract, checkpoint/resume, the 13-model ensemble's memory, and SpecAugment's per-item randomness |
| **Net-neutral** | The waveform augmentation (`audiomentations` drops in unchanged), the config/entry point, and AMP (TF is no better and arguably worse) |

**The one-line answer, from the existing analysis:**

> The portable core is the signal-processing / loss / metric *logic*, and it
> ports almost cleanly. What does not port is the *control plane*.

**A note on the architecture TF actually wants:** computing log-mel in
`tf.data` and caching it is strictly better than what the repo does — the repo
re-reads and re-analyses the same 30 s windows 50 times per fold across 13
models. But adopting it changes the `bn0` → `SpecAugment` ordering and
invalidates the tuned recipe. **For a solution that already won, that is not a
trade you make.** The value of that observation is not "use TF instead"; it is
that **the on-the-fly front-end is the single biggest inefficiency in the
original, and that was true in PyTorch too.**

---

## Traps, ranked by expected damage

1. **`Softmax(axis=-1)` in `AttBlock`** — a silent wrong answer. Correct is
   `axis=-2`. Trains; scores badly.
2. **The loss argument order inverts** — `call(y_true, y_pred)`. No error.
3. **Keras `reduction` divides by `B`, not `B·S·O`** — a factor of ~792,264.
4. **Keras' DenseNet applies ImageNet `preprocess_input`** — destroys the
   pretrained transfer, silently.
5. **HTK mel instead of Slaney** — different matrices; muted only by `bn0`.
6. **`tf.pad(SYMMETRIC)` is not "repeat the last frame"** — use
   `tf.concat` + `tf.repeat`.
7. **A layer that omits `training` applies SpecAugment at eval** — corrupts the
   tensor the vote consumes.
8. **A shared SpecAugment mask gives the whole batch one band** — weakens the
   augmentation invisibly.
9. **No reflect padding before `tf.signal.frame`** — `frames_num` becomes 2977.
10. **Feeding magnitude instead of power to the dB formula** — a factor of 2.
11. **Keras 3 `ModelCheckpoint` has no `save_top_k`** — `n_saved=5` needs a
    custom callback.
12. **Keras does not restore the epoch counter** on resume.
13. **`save_freq` as an int means batches, not epochs** — a silent unit change.
14. **`fit()` has no `jit_compile`** — it is on `compile()`.
15. **VRAM is preallocated** and `set_memory_growth` must run first.
16. **`sample_from_datasets` breaks `steps_per_epoch` inference** — compute the
    cardinality first.
17. **`keras.metrics.F1Score` defaults to argmax** — pass `threshold=0.5`.
18. **`AUC` is not LRAP** — query-independent vs query-dependent.
19. **`add_variable`, not `add_weight`, in a Metric.**
20. **No `keras.saving.register_keras_metric`** — it is
    `register_keras_serializable`, and the class needs `get_config`.
21. **TFLite int8 quantisation invalidates the 0.3 vote threshold** — the one
    genuine TF advantage, unusable without re-tuning the thing it optimises.
22. **`keras_tuner` is the wrong tool** for a 13-named-checkpoint ensemble.

---

## Verification plan

| # | Check | Passes when |
|---|---|---|
| 1 | **Log-mel equality** | The TF front-end and the original torch front-end produce the same `logmel` on the same waveform (tol 1e-4) — this single check covers the mel scale, the dB formula, the reflect pad and the power-vs-magnitude question at once |
| 2 | **Shape trace** | `frames_num == 3001`; the DenseNet output's time axis is **93**; the tail pad is exactly 25 frames |
| 3 | **Attention axis** | The attention distribution sums to 1 over the **time** axis, and the clipwise output is a sum over time |
| 4 | **Loss identity** | The ported `SedScaledPosNegFocalLoss` at `gamma=0` equals `BinaryCrossentropy` — i.e. the inertness was reproduced |
| 5 | **Loss argument order** | A deliberately swapped call produces a visibly different loss (i.e. the test would catch the inversion) |
| 6 | **Metric identity** | The ported LRAP / F1 clip / F1 frame reproduce the original ignite metrics' values on a fixed synthetic batch |
| 7 | **Background branch** | A background row with no fire scores 1.0 on the F1 metrics and (after the fix) does **not** score 1.0 on LRAP |
| 8 | **LR schedule** | The first values match the reference table, and the phase boundary is at exactly `epoch_length` |
| 9 | **One-epoch parity** | Against the original Ignite run on the same synthetic data, the per-step loss matches to 1e-5 |
| 10 | **Vote** | A synthetic 13-model set with exactly 4 fires at ≥0.3/≥0.3 is accepted; 3 fires is rejected — **and the Keras and Lightning ports produce the identical submission** |
| 11 | **Config sweep** | All 21 config files construct and the port reads every attribute |

Check 10 is the one that decides whether the two ports are interchangeable:
if the shared voting module produces byte-identical `submission.csv` from both
ports' score dumps, the ports agree at the level the competition was won.
