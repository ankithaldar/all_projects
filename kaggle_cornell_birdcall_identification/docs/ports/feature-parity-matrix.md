# Feature parity matrix

The complete inventory of the Cornell Birdcall solution, with the Lightning and
Keras target for every item and a concrete check for each.

**How to use it.** Work top to bottom. A row is *done* when its check passes.
A row that cannot be ticked is a *decision* — record it in the port's own
notes with the reason, never silently drop it. The "out of scope" rows in
[`README.md`](README.md#scope) are already decided; everything else is in.

### Where the full behavioural spec lives

Every row below is a *capability*. The **full spec** — the exact signatures,
the line numbers, the traps, the values that must be preserved — is in
[`../src-reference/`](../src-reference/README.md), one page per package. This
matrix says *what* must survive the port; the reference says *exactly what it
is*.

| matrix section | capability | full spec | worklist section |
|---|---|---|---|
| **A** | Model | [src-reference/models.md](../src-reference/models.md) | [§2](porting-worklist.md#2-the-model--modelssed_modelspy) |
| **B** | Audio front-end | [src-reference/helpers.md](../src-reference/helpers.md) | [§1](porting-worklist.md#1-the-front-end--helperssed_audio_utilspy) |
| **C** | Data | [src-reference/dataloaders.md](../src-reference/dataloaders.md) | [§3](porting-worklist.md#3-the-data--dataloaders) |
| **D** | Augmentation | [src-reference/augmentations.md](../src-reference/augmentations.md) | [§4](porting-worklist.md#4-the-augmentations--augmentations) |
| **E** | Loss | [src-reference/loss.md](../src-reference/loss.md) | [§5](porting-worklist.md#5-the-losses--loss) |
| **F** | Metrics | [src-reference/metrics.md](../src-reference/metrics.md) | [§6](porting-worklist.md#6-the-metrics--metrics) |
| **G** | Training control | [src-reference/engine.md](../src-reference/engine.md) | [§7](porting-worklist.md#7-the-control-plane--engine) |
| **H** | Logging | [src-reference/logger.md](../src-reference/logger.md) | [§8](porting-worklist.md#8-the-loggers--logger) |
| **I** | Config | [src-reference/config-params.md](../src-reference/config-params.md) | [§9](porting-worklist.md#9-the-configs--config_params) |
| **J** | Entry points | [src-reference/entrypoints.md](../src-reference/entrypoints.md) | [§10](porting-worklist.md#10-the-entry-point) |
| **K** | Vote & submission | *does not exist in the original* | [§11](porting-worklist.md#11-inference-and-the-vote) |

The **[porting worklist](porting-worklist.md)** is the execution order — the
matrix is what to build, the worklist is the order to build it in, with the
traps and the acceptance check for each step.

**Column key**

- **Original** — the source of truth in the current tree (`file:line`).
- **Must preserve** — the behaviour that defines the solution. If this is
  wrong, the solution is wrong; the framework is irrelevant.
- **Lightning / Keras** — where the behaviour lives after the port. `reuse`
  means the original file is imported unchanged.
- **Verify** — a check that proves it.

Legend: ✅ = trivially reusable · ⚠️ = must be rewritten, watch the trap ·
🆕 = does not exist in the original, new work · ⛔ = out of scope, decided.

---

## A. Model — `src/models/sed_models.py`

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| A1 | `init_layer:33` | Xavier-uniform weights, zero bias | reuse | `keras.initializers.GlorotUniform` + zero bias | weight std ≈ √(2/(fan_in+fan_out)) |
| A2 | `init_bn:41` | BN bias 0, weight 1 | reuse | reuse `BatchNormalization` defaults | BN weight all 1, bias all 0 |
| A3 | `ConvBlock:81` | 2×(conv3×3, no bias → BN → ReLU) then pool; `pool_type` ∈ avg/max/avg+max | reuse | `Conv2D(use_bias=False)` + `BatchNormalization` + `ReLU` + pool | output shape matches |
| A4 | `AttBlock:131` | `att`/`cla` are `Conv1d(k=1, bias=True)`; `bn_att` is `BatchNorm1d`; **softmax over time** of `tanh(att(x))`; `cla` → sigmoid; sum over time; returns `(x, norm_att, cla)` | reuse | ⚠️ `Softmax(axis=-2)` — Keras axes come from the right, `axis=-1` is a silent wrong answer | attention sums to 1 over the **time** axis, not the class axis |
| A5 | `PANNsCNN14Att:180` | 5 conv blocks (64→1024), 2 avg-pool, dropout 0.2, `fc1` 1024→1024, `AttBlock(1024, classes)` | reuse | rewrite as `keras.Model` | output dict has both heads with right shapes |
| A6 | `PANNsDense121Att:319` | **The winning model.** DenseNet121 `.features` (pretrained), `fc1` 1024→1024, `AttBlock(1024, classes)`, `interpolate_ratio=32` | reuse | ⚠️ `keras.applications.DenseNet121(include_top=False)` — and **bypass its ImageNet `preprocess_input`**, which torchvision's `.features` does not apply | same logits on the same input as the torch model (tol 1e-4) |
| A7 | `PANNsDense161Att:443` | DenseNet161 `.features`, `fc1` 2208→2048 | reuse | ⚠️ **`keras.applications` has no DenseNet161** — port the torchvision weights manually or use the 121 variant | shapes match |
| A8 | `PANNsDense169Att:566` | DenseNet169 `.features`, `fc1` 1664→1024 | reuse | ⚠️ same — no Keras equivalent | shapes match |
| A9 | `PANNsDense201Att:689` | DenseNet201 `.features`, `fc1` 1920→1024 | reuse | ⚠️ same — no Keras equivalent | shapes match |
| A10 | `interpolate:46` | ×ratio nearest-neighbour upsample on the time axis: each frame repeated `ratio` times consecutively | reuse | `tf.repeat(x, ratio, axis=1)` — interleaved, which is the same thing | `x[:, ::ratio] == original` |
| A11 | `pad_framewise_output:62` | Pad to `frames_num` by **repeating the last frame** (not zeros, not symmetric) | reuse | ⚠️ `tf.concat([x, tf.repeat(x[:, -1:, :], n, 1)], 1)` — `SYMMETRIC` pad is wrong, `CONSTANT` pad injects 25 "no bird" frames | last 25 frames all equal frame 3000 |
| A12 | `forward:311-316` | Output is a **dict**: `clipwise_output` (B,C,O) and `framewise_output` (B,C,T,O) | reuse | return the same dict | both keys present, correct shapes |
| A13 | `interpolate_ratio = 32` | 32× upsample; `93 × 32 = 2976 < 3001` → 25 frames of tail padding | reuse | reuse | padding count is exactly 25 |
| A14 | `bn0` + transpose dance `386-388` | `BatchNorm2d(mel_bins)` applied **before** SpecAugment; the transpose exists only to land PyTorch's channel axis on axis 1 | reuse | ⚠️ `BatchNormalization(axis=-1)` — **both transposes disappear**; keep the *ordering* (bn0 → SpecAugment), it is load-bearing | SpecAugment sees normalised input |
| A15 | `cnn_feature_extractor:241` / `head:420,422` | Dropout 0.2 in the CNN, dropout 0.5 twice in the head, all gated on `self.training` | reuse | ⚠️ `Dropout(p)` needs the `training` flag threaded through `call()` | dropout off at inference |
| A16 | `x1 + x2` pool fusion `292-294` | max_pool1d + avg_pool1d, k=3, s=1, p=1, summed | reuse | `MaxPool1D` + `AveragePool1D`, summed | equal to the torch output |
| A17 | `fc1` per variant | 1024→1024 / 2208→2048 / 1664→1024 / 1920→1024 | reuse | `Dense(units, activation='relu')` | param count matches |
| A18 | `densenet121(pretrained=True)` | ImageNet-pretrained backbone | reuse | ⚠️ `weights='imagenet'` + **no** `preprocess_input` (see A6) | feature extractor output equal to torchvision's |
| A19 | `apply_aug` flag | `True` in every shipped config. **`False` silently breaks mixup** — the model's `b = (b*c)//2` branch still fires | reuse | reuse | training with `apply_aug=False` + mixup must not crash |
| A20 | `preprocess:390-395` | Mixup is applied to the **spectrogram**, inside the model, after SpecAugment | reuse | ⚠️ requires a `train_step` override — the batch axis changes meaning mid-forward | mixed spectrogram = `λ·x_even + (1-λ)·x_odd` |
| A21 | `x.expand(..., 3, ...)` | 1 mel channel → 3 RGB channels (a view, zero copy) | reuse | `tf.tile`/`broadcast_to` (not `tile` — it copies) | shape (B·C, 3, T, M) |
| A22 | `torch.mean(x, dim=3)` | Mean over the **frequency** axis | reuse | `tf.reduce_mean(x, axis=-1)` — axis order differs | shape (B·C, 1024, 93) |
| A23 | measured shape trace | DenseNet121 `.features` on (1,3,3001,64) → **(1, 1024, 93, 2)**; transitions use *valid* pooling, hence 93 not 94 | reuse | ⚠️ Keras' DenseNet pooling may differ — **measure it**, do not assume 93 | the 93 is confirmed, not assumed |

---

## B. Audio front-end — `src/helpers/sed_audio_utils.py` (vendored PANNs)

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| B1 | `DFTBase:31-56` | Unitary DFT matrix `W[n,n] = exp(∓2πi·xy/n)` | reuse | not needed | — |
| B2 | `STFT:40-107` | Conv1d with **frozen** DFT×window weights; `center=True` + `pad_mode='reflect'`; hop defaults to `win//4`; returns (real, imag) (B, 513, T) | reuse | ⚠️ `tf.signal.frame` has **no reflect padding** — explicit `tf.pad(..., mode="REFLECT")` first, or `frames_num` becomes 2977 | frames_num == 3001 |
| B3 | `Spectrogram:110-140` | `power=2.0` → `real² + imag²` (power, **not** magnitude) | reuse | `tf.abs(rfft)**2` | feeding magnitude instead of power is a silent ×2 error |
| B4 | `LogmelFilterBank:144-196` | `melW = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax).T`, shape (513, 64), frozen; `power_to_db` = `10·log10(clamp(x, amin=1e-10)) − 10·log10(max(amin, ref=1.0))`; `top_db=None` in every shipped config | reuse | ⚠️ keep **librosa's** mel matrix — `tf.signal.linear_to_mel_weight_matrix` is HTK with unit-peak triangles, librosa's is Slaney with area normalisation. Different matrices. And there is **no** `tf.signal.magnitude_to_db`; write the one line. | `logmel` matches the torch front-end on the same waveform (tol 1e-4) |
| B5 | `DropStripes:199-243` | Zeroes `stripes_num` random bands along dim 2 (time) / dim 3 (freq); **no-op when `self.training` is False**; mutates in place, per batch item | reuse | ⚠️ `call(x, training=...)` must gate it, or SpecAugment corrupts `framewise_output` at inference — which is exactly what the vote consumes | inference output identical with and without SpecAugment configured |
| B6 | `SpecAugmentation:246-271` | time-dropper (dim 2) then freq-dropper (dim 3); configured `time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2` | reuse | ⚠️ use **masks**, not in-place mutation; and draw an **independent** band per batch item — a shared `axis_mask` gives the whole batch the same band and weakens the augmentation invisibly | band positions differ across batch items |
| B7 | frames count | `S = 32000·30 = 960000`, `n_fft=1024`, `hop=320` → **3001** frames, 513 linear bins | reuse | same, if B2 is right | frames_num == 3001 |
| B8 | `freeze_parameters=True` | `requires_grad = False` on the STFT and mel buffers | reuse | `add_weight(..., trainable=False)` — the direct counterpart; it also keeps the buffer out of `trainable_variables` and therefore out of the optimiser | optimiser param count excludes the frozen buffers |

---

## C. Data — `src/dataloaders/`

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| C1 | `SedDataset.__init__:11-43` | 10 kwargs: `root_dir, csv_dir, period, bird_code, inv_ebird_label, isTraining, num_test_samples, transform, background_audio_dir` | reuse | rewrite | constructor signature identical |
| C2 | `pd.read_csv` + `df.secondary_labels.apply(eval)` | The CSV's `secondary_labels` column holds **Python literals** and is executed with `eval`. NaN cells raise `TypeError` | reuse | ⚠️ do **not** carry `eval` into a TF port; parse the literal with `ast.literal_eval` — same result, no arbitrary execution | same label set as the original for every row |
| C3 | background injection `:34-43` | Label-less WAVs from `background_audio_dir` are appended as extra rows with **no** `ebird_code` → hard negatives on all 264 classes | reuse | rewrite | background rows have an all-zero label vector |
| C4 | `get_audio:55-85` | `.mp3` → `.wav` filename rewrite; `root_dir` may be a `Path` **or** a dict (5-archive sharding) | reuse | rewrite | correct file resolved for both root shapes |
| C5 | `convert_labels_to_coded:87-92` | One-hot over 264 classes, **broadcast to every snippet** (weak file-level label → every frame) | reuse | rewrite | a 264-vector, not 3001×264 |
| C6 | `__getitem__:103-115` (train) | Random placement of a short clip inside the 30 s window; random crop of a long clip | reuse | rewrite | two calls on the same file can differ (train) |
| C7 | `__getitem__:117-125` (valid) | Deterministic slicing; tail zero-pad; long files truncated | reuse | rewrite | two calls on the same file are identical (valid) |
| C8 | `num_test_samples` | 1 in train, 2 in valid/test | reuse | rewrite | batch tensor is (B, C, S) with the right C |
| C9 | `ImbalancedDatasetSampler:5-24` | Inverse-frequency weights, `torch.multinomial(..., replacement=True)`, one epoch = `len(dataset)` samples | reuse | ⚠️ `tf.data.Dataset.sample_from_datasets` is the declarative equivalent — but ⚠️ **it does not shard across ranks** | class balance per epoch is skewed toward rare classes |
| C10 | `init_workers_fn:51-53` | Per-worker numpy seeding from `os.urandom` | reuse | n/a (tf.data has its own threading) | two workers disagree on their random crops |
| C11 | `additional_loader_params:22-24` | `{"worker_init_fn": self.init_workers_fn}` | reuse | n/a | — |
| C12 | `get_label:45-46` | Returns the **primary** ebird code — this is what the sampler balances on | reuse | rewrite | sampler uses the primary label, not the full label set |

---

## D. Augmentation — `src/augmentations/`

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| D1 | `sed_default_augment:21-33` | `AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3)` → `AddGaussianSNR(p=0.3)` → optional `AddBackgroundNoise(dir, p=0.5)` | reuse | ✅ `audiomentations` is numpy-in/numpy-out — **drops into `tf.data.map` unchanged** | transform output length == input length |
| D2 | `sed_background_augment` | `AddGaussianNoise(p=0.2)`, `AddGaussianSNR(p=0.2)`, `Gain(−15..15 dB, p=0.3)`, optional `AddBackgroundNoise(p=0.2)`, optional `AddShortNoises(min_time_between_sounds=0.0, max=15.0, burst_probability=0.5, p=0.6)`, **then the noise/SNR/gain trio repeated** | reuse | ✅ unchanged | the palindromic bracket is preserved |
| D3 | `sed_dataset.py:99` | The waveform transform is applied to the **entire file, before slicing** into 30 s snippets | reuse | ⚠️ preserve the order — transforming per-snippet is a different experiment | a 10-min file is augmented as one signal |
| D4 | `Mixup:27-47` | `lam ~ Beta(alpha, alpha)`, even/odd pairing, `random_state = np.random.RandomState(1234)` — **hard-coded**, so every run and every rank draws the same sequence | reuse | ✅ numpy-based, unchanged | same lambdas on run 2 as on run 1 |
| D5 | `do_mixup:49-59` | `x[0::2]·λ[0::2] + x[1::2]·λ[1::2]` — mixes **even with odd** | reuse | ✅ same | mixed row is the convex combination of its pair |
| D6 | `sed_engine.prepare_batch:28-30` | The **labels** are mixed too, and the batch is halved to `(B·C)//2` | reuse | ⚠️ the halving must happen in `train_step` in Keras, where the batch axis changes meaning mid-forward | label tensor shape is `(B·C)//2, 1, 264` |
| D7 | `helpers/sed_audio_utils.py` | SpecAugment (see B6) runs **inside the model** after `bn0` | reuse | see B6 | — |
| D8 | `noise_maker.powerlaw_psd_gaussian` | 1/f^α Gaussian noise, unit variance (Timmer & Koenig 1995) | ⛔ | ⛔ | not called from the repo; the WAVs come from an external kernel |
| D9 | `base_augment`, `default_augment`, `spec_augment` | Legacy image pipeline | ⛔ | ⛔ | `spec_augment.py` raises `ImportError` (`mono_to_color` does not exist) |

---

## E. Loss — `src/loss/`

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| E1 | `SedEngine._init_criterion_function:87-102` | Exactly **five** registered criteria: `focal_loss_standard`, `sed_removed_focal_loss`, `sed_scaled_pos_neg_focal_loss`, `sed_scaled_pos_neg_focal_loss_augd`, `bce_controlled_standard_loss` | reuse (a registry dict replaces the if/elif) | rewrite (a registry dict) | an unknown name raises a **clear** error, not a later `AttributeError` |
| E2 | `loss_keys` on each | `focal_loss_standard` → `["bce_loss","F_loss"]`; `sed_removed_focal_loss` → `["bce_loss","F_loss","FRemoved_loss"]`; `sed_scaled_pos_neg_focal_loss` → `["bce_loss","F_loss","FScaled_loss","F_loss_0","F_loss_1"]`; `sed_scaled_pos_neg_focal_loss_augd` → same 5; `bce_controlled_standard_loss` → `[]` | reuse | ⚠️ Keras losses return a **scalar**; the dict needs a `Metric`-per-key shim | every key appears in the logged metrics |
| E3 | return contract | Every loss returns **`(scalar_tensor, dict_of_scalars)`** | reuse | ⚠️ a Keras `Loss` returns a scalar — the dict must be surfaced separately (a per-key metric, or a `compute_loss` override) | 5 extra metrics logged, matching `loss_keys` |
| E4 | `sed_engine.loss_fn:72` | The loss consumes **only** `y_pred["clipwise_output"]`. `framewise_output` is never touched by any loss — it inherits all signal through the `AttBlock` attention weights | reuse | ⚠️ `call(y_true, y_pred)` — the **argument order inverts**. Use keyword arguments and a `compute_loss(self, x, y, y_pred, sample_weight, training=True)` override | gradient reaches only the clip head path |
| E5 | inertness | `gamma=0` ⇒ `(1-pt)^0 = 1` ⇒ the focal losses are **exactly BCE**; `alpha_0=alpha_1=1`; `secondary_factor=1.0` ⇒ the secondary mask is 1 | reuse | rewrite identically | `SedScaledPosNegFocalLoss(gamma=0, alpha_0=1, alpha_1=1, secondary_factor=1)` output == `nn.BCELoss()` output (tol 1e-6) |
| E6 | `sed_scaled_pos_neg_focal_loss_augd` | After the scaled loss, the **primary** species positions are randomly zeroed with probability `primary_drop_out_factor=0.3` ⇒ **70 % of primary positions are dropped** | reuse | rewrite identically | ~70 % of primary positions contribute zero |
| E7 | `BCEControlledStandardLoss` | The **winning** criterion: `beta=0.5` L-soft BCE on an already-sigmoided prediction, `.mean(1).mean()`; `y_pred` is **not** re-sigmoided | reuse | ⚠️ `BinaryCrossentropy(from_logits=False)` — the sigmoid is inside the model | equals `F.binary_cross_entropy(y_pred, (β·y_true + (1-β)·y_pred)·y_true).mean(1).mean()` |
| E8 | the other 11 loss classes | Legacy V1–V4 loss zoo | ⛔ | ⛔ | only `main_engine*` selects them; `sed_scaled_focal_loss.py` is imported by nothing |
| E9 | `torch.no_grad()` inside the loss | The masks are computed under `no_grad` | reuse | ⚠️ `tf.stop_gradient` | no gradient flows through the mask construction |

---

## F. Metrics — `src/metrics/`

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| F1 | `SedLRAPS:5-31` | Label-ranking AP on `clipwise_output.max(axis=1)[0]` vs `target[:,0,:]`, averaged | reuse (`torchmetrics.Metric` or a hand-rolled accumulator) | ⚠️ **no Keras label-ranking metric exists** — reimplement vectorised LRAP (~20 lines) or pay `tf.py_function` per item | LRAP of a perfect ranking == 1.0 |
| F2 | `SedF1ScoreClip:8-45` | Micro-F1 on the clip head. **Background branch**: no positive label ⇒ 1.0 only if nothing fires above threshold | reuse | ⚠️ `keras.metrics.F1Score` defaults `threshold=None` = multiclass argmax — **categorically wrong** for 264-label multi-label. Pass `threshold=0.5` explicitly, and hand-write the empty-target branch | a background row with no fire scores 1.0 |
| F3 | `SedF1ScoreFrame:7-45` | Micro-F1 on the frame head **ANDed with a hard-coded 0.3 clip gate**: `(pf ≥ threshold) & (pc ≥ 0.3)` | reuse | ⚠️ same two traps as F2, plus the AND | a frame that passes but whose clip is < 0.3 contributes nothing |
| F4 | `SedLRAPS`/`F1Score*` sigmoid | **No sigmoid is applied** — the `AttBlock` already applied it | reuse | ⚠️ same | the metric sees probabilities in [0,1] |
| F5 | `RunningAverage` per `loss_key` | One running mean per criterion loss key, on both train and validation | ✅ `self.log`/`self.log_dict` | ⚠️ needs the per-key metric shim from E2/E3 | logged values equal the mean over the epoch |
| F6 | `SedTest:21-82` | Writes one JSON per audio file: `filename, predicted_frames, predicted_clip, targets, lraps_score_clip, lraps_score_frames`; optionally merges into `final/merged_results.json`; returns the mean frame LRAP | ⚠️ rewrite as a callback — see the four blockers in the guide | ⚠️ same | the JSON keys match byte-for-byte |
| F7 | `SedTest` max-pooling | `pred_frames.max(axis=1)[0].max(axis=1)[0]` — the **entire time axis is destroyed**. This is why it cannot emit 5-second rows | ⚠️ preserve for the dump, and **fix separately** for the vote (see K3) | ⚠️ same | — |
| F8 | `custom_f1score`, `custom_lraps`, `custom_test` | Legacy family (sigmoid + mean-pool) | ⛔ | ⛔ | the SED path uses the `Sed*` metrics |

---

## G. Training control — `src/engine/`

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| G1 | `sed_engine._init_optimizer:84` | `AdamW(lr, betas=(0.9,0.999), eps=1e-08, weight_decay=hparams.weight_decay, amsgrad=True)` | reuse | ⚠️ `keras.optimizers.AdamW(..., amsgrad=True)` exists; `epsilon` not `eps` | param group hyper-parameters match |
| G2 | `warmup_with_cosine:105-118` | `ConcatScheduler([LinearCyclical(lr·f → lr, cycle=2E), CosineAnnealing(lr → lr·f, cycle=N·E)], durations=[E])` | ⚠️ `LambdaLR` — the exact formulas are in the guide and are **verified against live Ignite** | ⚠️ `keras.optimizers.schedules` — two phases, the first for exactly `E` steps | the LR curve equals Ignite's, step for step (max delta < 1e-12) |
| G3 | `one_cycle_cosine:120-128` | A single `CosineAnnealing(lr → lr·f, cycle=N·E)` | `LambdaLR` | schedule | same |
| G4 | `scheduler_name="none"` | No scheduler at all | ✅ | ✅ | LR constant |
| G5 | `accumulation_steps` | `optimizer.step()` only every N iterations; `zero_grad()` after | ✅ `accumulate_grad_batches=N` | ⚠️ `gradient_accumulation_steps=N` on the optimizer | gradients accumulate over N batches |
| G6 | `clip_gradients:327-350` | Norm-2 clipping, `EPSILON=1e-6` (`EPSILON_FP16=1e-5` under AMP); **disabled in every shipped config** (`gradient_clip_val=0`) | ✅ `gradient_clip_val=N` (the original's code is a copy of Lightning's own) | ✅ `clipnorm=N` | — |
| G7 | `USE_AMP` `:6-13` | **Always False.** Apex is detected but explicitly not used, because the in-graph log-mel produced NaNs under fp16 (`amin=1e-10` underflows to 0 → `log10(0) = −inf`) | ✅ do **not** enable `precision=16` without wrapping `power_to_db` in `autocast(enabled=False)` | ⚠️ same wall; Keras' automatic loss scaling is harder to opt out of per-layer | no NaN in the first 10 steps at fp16 |
| G8 | `eval_every` | Validation every 2 epochs; the **checkpoint saver fires on the evaluator's `COMPLETED`**, so checkpoints are written once per validation, not once per epoch | ✅ `check_val_every_n_epoch=2` + `ModelCheckpoint` — ⚠️ Lightning **decouples** the two cadences; you must choose | ⚠️ `save_freq` as an int means **batches**, not epochs — a silent unit change | a checkpoint appears every 2 epochs |
| G9 | `setup_checkpoint_saver:120-128` | `Checkpoint(to_save, DiskSaver(save_dir, require_empty=False), n_saved=5, filename_prefix=prefix, score_function=score_function, score_name="score", global_step_transform=global_step_from_engine)` | ✅ `ModelCheckpoint(dirpath, filename, monitor, mode, save_top_k=n_saved, save_last=True)` | ⚠️ **Keras 3 `ModelCheckpoint` has no `save_top_k`** — `n_saved=5` needs a custom callback plus file management | 5 checkpoints kept, ranked by the tracked metric |
| G10 | `load_trainer_from_checkpoint:103-118` | Two modes: `load_model_only=True` restores only `{"model": ...}`; `False` restores model + trainer + optimizer + scheduler | ✅ `trainer.fit(ckpt_path=...)` for a full resume; a manual `load_state_dict` for weights-only | ⚠️ Keras does not restore the epoch counter; `tf.train.Checkpoint` + `CheckpointManager` is a different programming model | weights-only load leaves the optimiser at its initial state |
| G11 | `complete_clear:175-180` | Nulls the batch/output references and `gc.collect()`s at each epoch end — the OOM lever is in-flight batches | ✅ Lightning does it, but an explicit `gc.collect()` is cheap insurance | ⚠️ Keras holds the batch in the step function — clear it explicitly | peak memory flat across epochs |
| G12 | `lr_finder:353-374` | Ignite's `FastaiLRFinder`, `min_lr=3e-5`, `max_lr=10`, `num_iter=len(loader)-1`; it **detaches** the validation handler and the scheduler for the sweep | ✅ `Tuner(trainer).lr_find(...)` — ⚠️ but Lightning has **no callback removal**, so the sweep must run on a throwaway trainer | ⚠️ no equivalent — a manual loop | the suggested LR matches |
| G13 | `_init_distribution:52-92` | `idist.Parallel`, `auto_dataloader` (which **shards** the sampler), `auto_model`, `auto_optim`; `manual_seed(42 + rank)` | ✅ `strategy="ddp"`, `seed_everything(42 + global_rank)` — ⚠️ `ImbalancedDatasetSampler` is not a `DistributedSampler`; Lightning's DDP wrapper shards it, but verify | ⚠️ `tf.distribute` — and the imbalanced sampler does **not** shard | each rank sees a disjoint slice |
| G14 | `ProgressBar(persist=False)` | A tqdm bar on the trainer and the evaluator | ✅ `enable_progress_bar` | ✅ `keras.callbacks.ProgbarLogger` | — |
| G15 | `train:216-223` | The `try/except` around `trainer.run()` **swallows the traceback**, prints it, and continues to `_end_of_training()`. A crashed 50-epoch run exits 0 and looks complete in Neptune | ✅ Lightning **re-raises** — a win, but it changes the operational contract: wrap `fit()` explicitly if you want the old behaviour | ⚠️ same | a crash is loud |
| G16 | `prepare_batch:16-69` | Builds `(x, mixup_lambda)` and the label dict; handles `train`/`valid`/`test` | ✅ port verbatim (Lightning already moved the batch to the device) | ⚠️ rewrite; the `test` branch in the original references an **undefined local `x`** → `NameError` | the `test` branch runs |
| G17 | `output_transform:75-81` | Returns `{"loss", "x", "y_pred", "y", "dict_loss"}` for train/valid, `{"y_pred","x","y"}` for test | ✅ the `training_step` return value | ✅ | — |
| G18 | `loss_fn:71-73` | `self.criterion(y_pred["clipwise_output"], y)` | ✅ | ⚠️ see E4 | — |
| G19 | `_init_*` dispatch | 5 model names, 5 criterion names, 3 scheduler names, 1 optimiser name, 2 logger names, 2 augmentation names | ✅ **two registry dicts** — the repo's own analysis recommends this over the if/elif | ✅ same | an unknown key raises a clear error |
| G20 | `_init_score_function:318-325` | `metric_factor × engine.state.metrics[track_metric]`; `track_metric="f1score_clip"`, `metric_factor=1` | ✅ `ModelCheckpoint(monitor=..., mode="max" if factor≥0 else "min")` — ⚠️ the monitor name must be one you actually log, or the callback **silently never fires** | ⚠️ same | a checkpoint is written when the metric improves |
| G21 | `_init_score_function` on `hparams.track_metric` | `hparams.track_metric` is `"f1score_clip"`, **not** `"valid_f1score_clip"` | ⚠️ decide the naming: log as `val_f1score_clip` and map the monitor, or log under the bare name — but Lightning warns on validation metrics not prefixed `val_` | ⚠️ same | — |
| G22 | `evaluate:230-235` | Creates a third Engine for testing — **dead code**, never called, and `mode='test'` was broken | ✅ make it work: `trainer.test(...)` with the dump callback attached | ✅ | the dump runs end to end |

---

## H. Logging — `src/logger/`

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| H1 | `MyNeptuneLogger` | Per-iteration: `train/loss`, `train/<loss_key>`, `train/lr{i}`; per-epoch: `train/<metric>`; at eval end: `valid/<metric>`. Also `params` and `tags` from `logger_params` | ⚠️ `lightning.pytorch.loggers.NeptuneLogger` — ⚠️ the original wraps ignite's logger, which speaks **neptune-client 0.4.x**; Lightning's speaks **neptune ≥ 1.0**. The client version changes with the framework — an unavoidable dependency change, not a porting choice | ⚠️ `keras` + the `neptune-keras` callback, or a custom `Callback` | every metric in H1 appears in the Neptune run |
| H2 | `PrintLogger` | The same series to stdout, at `log_every` intervals | ✅ console | ✅ | — |
| H3 | Slack webhook | One Slack message per epoch (train) and **one per metric key** (valid), via `os.environ["SLACK_URL"]` | ⚠️ reimplement as a Lightning callback | ⚠️ same | a message arrives per epoch |
| H4 | `base/utils.add_iteration_handlers` | `Events.ITERATION_COMPLETED(every=N)` — a **per-handler** filter. `Trainer(log_every_n_steps=N)` is not the same thing | ⚠️ a hand-rolled counter callback — the analysis calls this a genuine regression in expressiveness | ⚠️ same | the log fires every N steps, not every step |
| H5 | `logger_params` | `project_name, log_every, name, prefix_name, tags, params` — `params` is the flattened model/loss/optimiser config | ✅ pass through | ✅ | the run carries the same hyper-parameters |

---

## I. Config — `src/config_params/`

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| I1 | `Parameters` (39 attributes) | Every attribute and its meaning — see the table in the Lightning guide | ✅ reuse the class unchanged | ✅ reuse the class unchanged | the port reads the same 39 names |
| I2 | `importlib.import_module(config)` + `module.Parameters()` | The `--config` argument is a **dotted Python module path** | ✅ unchanged | ✅ unchanged | `--config config_params.final_sed.final_sed_dense121_mix_fold0` resolves |
| I3 | `configs.py:31+` | `BIRD_CODE` (264 entries, code→index), `EBIRD_LABEL`, `INV_EBIRD_LABEL` (index→`"Genus species_Common Name"`) | ✅ reuse | ✅ reuse | 264 species, index order identical |
| I4 | `AUDIO_CONFIG` | Defined but never instantiated (legacy) | ⛔ | ⛔ | — |
| I5 | `get_dict_value` / `fill_range` | The 5-archive sharding map. Imported by every config, **called by none** | ⛔ (keep the import working) | ⛔ | — |
| I6 | 21 config files | `final_sed/*` (15), `final_sed_5_fold/*` (5), `example_config.py` — 13 of them are the shipped ensemble members | ✅ all 21 keep working unchanged | ✅ all 21 keep working unchanged | every config constructs and the port reads its attributes |
| I7 | CRLF line endings | 20 of the 22 files in `src/config_params/` are CRLF. Line-anchored `grep` silently misses them | ⚠️ do not silently rewrite them | ⚠️ same | — |

---

## J. Entry points

| ID | Original | Must preserve | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| J1 | `sed_train.py:1-25` | argparse `--config` → `importlib` → `Parameters()` → `idist.Parallel` → `run()`; `torch.backends.cudnn.benchmark = True` | ✅ keep the argparse + importlib shape | ✅ keep it | `python <port>/sed_train.py --config …` starts a run |
| J2 | `train.py`, `train_v2/3/4.py` | Legacy entry points | ⛔ | ⛔ | — |
| J3 | `startup.sh` | The environment setup script (Apex clone, pip install) | ✅ update the dependency list only | ✅ update the dependency list only | the script still provisions the env |

---

## K. Vote and submission — 🆕 **does not exist in the original**

| ID | Required behaviour | Source of truth | Lightning | Keras | Verify |
|---|---|---|---|---|---|
| K1 | 4 of 13 models must fire for a species to be accepted | `README.md:45` | a shared voting module, framework-agnostic | the same module | a synthetic 13-model set with exactly 4 fires → accepted; 3 → rejected |
| K2 | `clipwise ≥ 0.3` **and** `framewise ≥ 0.3` — two independent gates | `README.md:45`; the in-repo ancestor is `sed_f1score_frame.py:35,40` | the same | the same | a model that passes the clip gate but not the frame gate does not vote |
| K3 | 5-second rows | external kernel only | 🆕 the frame axis must be **kept**, not max-pooled (F7) | 🆕 same | a 10-minute file yields 120 rows |
| K4 | Site 1/2/3 row construction | external kernel only | 🆕 | 🆕 | — |
| K5 | `nocall` fallback | external kernel only | 🆕 | 🆕 | a row with no species → `nocall` |
| K6 | `submission.csv` (`row_id`, `labels`) | external kernel only | 🆕 | 🆕 | the CSV has the right row count and format |

**K1–K6 are new work.** The original repo has no vote code, no 5-second
slicing, no `row_id`, no `to_csv` — a repo-wide search for `submission`,
`row_id`, `site` and `to_csv` returns nothing. Porting the solution faithfully
does **not** produce these; they are what turns a trained model into a
leaderboard entry.

---

## Coverage summary

| Section | Items | In scope | Out of scope | New |
|---|---|---|---|---|
| A. Model | 23 | 23 | 0 | 0 |
| B. Front-end | 8 | 8 | 0 | 0 |
| C. Data | 12 | 12 | 0 | 0 |
| D. Augmentation | 9 | 7 | 2 | 0 |
| E. Loss | 9 | 8 | 1 | 0 |
| F. Metrics | 8 | 7 | 1 | 0 |
| G. Training control | 22 | 22 | 0 | 0 |
| H. Logging | 5 | 5 | 0 | 0 |
| I. Config | 7 | 5 | 2 | 0 |
| J. Entry points | 3 | 2 | 1 | 0 |
| K. Vote & submission | 6 | 0 | 0 | 6 |
| **Total** | **112** | **99** | **7** | **6** |

The 7 out-of-scope rows are D8, D9, E8, F8, I4, I5 and J2. Each is listed
with its reason in [`README.md`](README.md#scope). Every one of them is
unreachable from the winning configuration.

**Before calling a port done, every row in A–J must be ticked and every row in
K must be either implemented or explicitly deferred with a reason.**
