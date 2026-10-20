# Verified Findings — Cornell Birdcall 1st Place Solution

Accumulated as agents report. Every line here is **verified against the code**,
not inferred from the README. This is the fact base for the master synthesis.

## Repo shape

- Live chain: `sed_train.py → SedEngine → BaseEngine → {SedDataset, sed_models,
  helpers/sed_audio_utils, loss/*sed_*, metrics/sed_*, augmentations/sed_*}`
- `src/config_params/` = `configs.py`, `example_config.py`, `final_sed/` (**15**
  files), `final_sed_5_fold/` (**5** files) = 21 configs.
- The `final/`, `old/`, `final_5sec_sed_5_fold/` dirs named in the repo's own
  `directory_structure.txt` **do not exist and were never tracked in git**.
  Stale artifact — do not trust it.
- No `__init__.py` anywhere under `src/` — implicit namespace packages; must run
  from inside `src/`.
- `requirements.txt` pins ignite, neptune-client, audiomentations, tqdm, psutil,
  SoundFile. **torch / torchvision / torchaudio / librosa are NOT pinned.**

### 1. The "focal loss" is inert — the real objective is plain mean BCE

All 21 configs ship these four values (the two `_augd` configs add a fifth key,
`"primary_drop_out_factor": 0.3`, consumed by `sed_scaled_pos_neg_focal_loss_augd`):

```python
criterion_params = {"gamma": 0.0, "alpha_0": 1.0, "alpha_1": 1.0, "secondary_factor": 1.0}
```

**Verified numerically, not by inspection:** with these values
`SedScaledPosNegFocalLoss` is **bit-identical** to `nn.BCELoss()` — both
`0.8186740875244141` on the same input.

| Knob | Shipped value | Effect |
|---|---|---|
| `gamma` | `0.0` | `(1-pt)^0 ≡ 1` → **focal term off** |
| `alpha_0` / `alpha_1` | `1.0` / `1.0` | `(α₀(1-m) + α₁m)` ≡ `1` → **pos/neg split off** |
| `secondary_factor` | `1.0` | `y_secondary_mask ≡ 1.0` → **secondary masking off** |

⇒ The optimised objective is **exactly `mean BCE`** over clamped sigmoid
`clipwise_output` against clip-constant multi-hot targets. The machinery is
present, instrumented and calibrated — but numerically inert at these values.

Only 5 of 16 loss files are even reachable from `SedEngine`; **13 are dead**.
`sed_scaled_focal_loss.py` is the conceptual bridge in the progression
(`focal_loss_standard` → `removed` → `scaled` → `pos_neg`) and is imported by
nothing.

### 1b. `loss_keys` registers 12 metrics, 3 of which are duplicates

`_init_metrics` does `for lk in self.criterion.loss_keys` →
`RunningAverage(x["dict_loss"][lk])` on both train and valid. For the final
criterion (`["bce_loss","F_loss","FScaled_loss","F_loss_0","F_loss_1"]`) all
five values are **identical** under the shipped params, so the Neptune curves
`train_bce_loss`, `train_F_loss` and `train_FScaled_loss` are three copies of the
same number. Useful signal *would* appear the moment `gamma`/`alpha_0`/
`alpha_1`/`secondary_factor` are moved off identity — the instrumentation is
already there, waiting on hyperparameters.

The `_augd` variant's Bernoulli(0.3) primary-dropout measures only a **0.11 %**
loss reduction, and because `eval_step` calls the same `loss_fn`, it makes the
**validation loss non-deterministic**.

### 1c. Why the identity values are defensible, not sloppy

The metric is row-wise micro-F1, which is already species-balanced (every
(row, species) pair contributes one TP; TN ignored). With a species-uniform
sampler the batch is already ~32 distinct species, so within a batch the positive
fraction is ~1/264 and *stable*. Pos/neg reweighting would correct a bias that
the sampler has already removed. Every imbalance knob set to neutral is
consistent with that — though no experiment log exists to confirm the intent.

## The three headline insights

### 1. The "focal loss" is inert — the real objective is plain mean BCE

All 21 configs ship these four values (the two `_augd` configs add a fifth key,
`"primary_drop_out_factor": 0.3`, consumed by `sed_scaled_pos_neg_focal_loss_augd`):

```python
criterion_params = {"gamma": 0.0, "alpha_0": 1.0, "alpha_1": 1.0, "secondary_factor": 1.0}
```

Substituted into `SedScaledPosNegFocalLoss`:

| Knob | Shipped value | Effect |
|---|---|---|
| `gamma` | `0.0` | `(1-pt)^0 ≡ 1` → **focal term off** |
| `alpha_0` / `alpha_1` | `1.0` / `1.0` | `F_loss_0 + F_loss_1` reconstructs **unweighted** BCE |
| `secondary_factor` | `1.0` | `y_secondary_mask ≡ 1.0` → **secondary scaling off** |

⇒ The optimised objective is **exactly `mean BCE`** over clamped sigmoid
`clipwise_output` against clip-constant multi-hot targets. All the focal/scaling
machinery is present, logged and calibrated — but numerically inert at these
values. The class is the vehicle; the weights are the knob; every shipped config
leaves every knob neutral.

The **only** active term anywhere is the `_augd` variants' 30 % Bernoulli
primary-dropout (`primary_drop_out_factor=0.3`), and that one lives inside the
loss, not the dataset.

### 2. AMP is dead code, and we know exactly why

`base_engine.py:6-13`:

```python
try:
    from apex import amp
    print("Found NVIDIA Apex, using mixed precision")
    print("NOT USING APEX BECAUSE CANNOT CREATE MEL SPEC, REQUIRES TUNING, PROBLEMS HAVE OCCURRED")
    USE_AMP = False
except:
    print("NVIDIA Apex not found, not using mixed precision")
    USE_AMP = False
```

`USE_AMP = False` in **both** branches — the import detection is vestigial and the
"Found NVIDIA Apex" print is a lie.

Root cause, verified numerically: `LogmelFilterBank` uses `amin=1e-10`. Under
fp16 this underflows to `0.0`, and `log10(0) = -inf`. fp16's smallest normal is
6.1e-5. The fp32 path is finite at exactly −100 dB. Fixing it requires wrapping
`power_to_db` in `autocast(enabled=False)`, **not** merely raising `amin`.

### 3. Training supervises one head; inference uses both

`sed_engine.py:71-73`:

```python
def loss_fn(self, y_pred, y):
    loss, dict_loss = self.criterion(y_pred["clipwise_output"], y)
```

**`framewise_output` is never touched by any loss.** The frame head inherits all
its signal through the `AttBlock`'s `norm_att` attention weights — i.e. the
attention *is* the weak-label propagation mechanism. That is precisely why the
frame and clip thresholds must be tuned **separately** at inference: the frame
score is not calibrated to a probability of "bird present in this 5 s".

And the winning decision rule survives in-repo, in
`metrics/sed_f1score_frame.py`:

```python
score = f1_score([t], [(pf >= self.threshold) & (pc >= 0.3)], average='micro')
```

Frame `>= self.threshold` (0.5, wired at `sed_engine.py:170`) AND clip `>= 0.3`
— **a hard-coded literal, not configurable**. Asymmetric by accident.

#### Four independent blockers on the test path

| # | Blocker | Consequence |
|---|---|---|
| 1 | `sed_test.py:59` missing comma | `SyntaxError`; file cannot be imported |
| 2 | `"test"` absent from `self.metrics` in all 21 configs | metric never registered |
| 3 | `save_dir`/`combine_files` defined by **no** config | `AttributeError`; `save_dir` would also be a `str`, so `.mkdir` fails |
| 4 | `sed_engine.py:60` uses undefined `x` in the `mode=='test'` branch | `NameError` — `x` is bound only in the `'valid'` branch at `:48` |

Fixing #1 does **not** produce a submission pipeline.

#### The structural blocker, worse than the syntax error

`SedTest.update` double max-pools `(B,S,3001,264)` → `(B,264)`: **1 row where
120 are needed.** Structurally incapable of a 5-second vote regardless of bugs.

Frame arithmetic, **measured** by running `densenet121.features` on a real
tensor: 10 min @ 32 kHz = 19,200,000 samples → **60,001** mel frames → **1,875**
post-DenseNet → ×32 = 60,000 → padded to 60,001. **500 frames = 5 s; 120 rows per
file.** Needs `num_test_samples = 20`; **every shipped config has `2`** — 60 s of a
600 s file, remainder silently truncated.

#### Two further corrections, verified against the pinned versions

- `@sync_all_reduce("_score", ...)` on `SedTest.compute` **does not raise** in
  ignite `0.4.0.post1` — it uses `getattr(self, attr, None)` so the missing
  `_score` is silently skipped. The real bug is subtler: `_num_items` *is* reduced
  while `_score_frame` is **not** — global denominator over a per-rank numerator,
  silently wrong under DDP, no error raised.
- **CORRECTED 2026-10-07:** 20 of the 22 `.py` files in `src/config_params/` are
  CRLF — `configs.py` plus **19 of the 20** `final_sed*` configs. LF: `example_config.py`
  **and `final_sed/final_sed_dense121_mix_2.py`** (verified with `file`). A
  line-based `grep` silently misses 19 of the 21 `Parameters` configs. This already bit the
  exploration agents and will bite anyone auditing this repo.

#### Where the missing inference logic lived

`directory_structure.txt` lists `./src/config_params/final_5sec_sed_5_fold`
(plus `old/`, `final/`) — **absent from the tree and never committed**. Exactly
where 5-second configs would live. Corroborates that inference used a different
`period`/`num_test_samples` than any shipped training config.

## Model & front-end

- `sed_audio_utils.py` is **vendored PANNs** (`qiuqiangkong/audioset_tagging_cnn`,
  MIT, header at `sed_models.py:1-23`), not torchaudio. `freeze_parameters` and
  `DropStripes` are PANNs-only signatures. No `sed_torchaudio.py` exists.
- Concrete dims (verified by running the real `nn.Conv1d`), `sr=32000`,
  `n_fft=1024`, `hop=320`, `center=True`, `period=30`:
  `L=960,000` samples → `(N,1,3001,513)` power → 64 mels. So `T=3001`,
  `F=513` @ 31.25 Hz, ~0.01 s per frame.
- **The load-bearing trick nobody comments:** `bn0 = BatchNorm2d(64)` runs
  *before* `SpecAugmentation`, which writes **literal 0**. In raw dB, 0 means
  full-scale power; post-BN, 0 means "the per-bin mean". That is why the
  `transpose(1,3) / bn0 / transpose(1,3)` sandwich exists.
- Dense mel bank `(513, 64)`, not sparse. `librosa.filters.mel` is called with
  **no `htk=`** and librosa is not in `requirements.txt` → HTK-vs-Slaney mel
  scale is **unpinned**. *Inferred* HTK from the mid-2020 competition date
  (librosa flipped the default in 0.8.0, Jan 2021); not proven.
- `SpecAugmentation` is PANNs', **not** torchaudio's: `DropStripes(dim=2)`=time,
  `dim=3`=mel on the 4-D `(N,1,T,F)` tensor — **not** `dim=1`/`dim=2`. No
  `replace_with_zero` flag exists.
- `AttBlock`: `softmax(tanh(att(x)), dim=-1)` replaces upstream
  `clamp(att(x), 0, 1)`. DenseNet121 `.features` (ImageNet) replaces the PANNs
  CNN14 stack; the 1-channel logmel is widened to 3 via `x.expand(...)` — a
  **view**, zero copy, zero params. **CORRECTED 2026-10-07:** DenseNet121
  `.features` has **five** stride-2 stages — `conv0` (stride 2), `pool0` (MaxPool
  stride 2), `transition1-3` (AvgPool stride 2 each) — so time is reduced ×32
  (floor at each stage: 3001 → 93, measured), hence `interpolate_ratio = 32`
  (93×32 = 2976, then padded by 25 to 3001). The earlier "3 maxpools → T/8" was wrong.
- `top_db` defaults to `80.0` in the class but **every** config passes `None`, so
  the clamping branch is dead. `PANNsCNN14Att` has no `top_db` parameter at all.

## Data

- Layout: `/data/{first_letter}/{ebird_code}/{filename}.wav` — `root_dir` may be a
  `Path` or a `dict` keyed by first letter. `.mp3` → `.wav` rewritten at load;
  training ran on a pre-resampled 32 kHz wav mirror of the Kaggle data.
- `PERIOD = 30 s`. `num_test_samples` = **1** train, **2** valid/test. So train
  items are `(1, 30 s)`, valid/test `(2, 30 s)`. The competition's 5-second rows
  exist **only** in the external kernel.
- Short files are zero-padded and the real audio is placed at a **random** offset
  (`np.random.randint(effective_length - duration)`, one-arg → `low=0`) — a
  deliberate augmentation.
- Labels are **file-level and broadcast across every snippet and frame**
  (`coded_labels[:, label_index] = 1`). This is the weak-label SED contract.

## Metrics

- Ignite `Metric` default `usage=EpochWise()` → `reset()` on EPOCH_STARTED,
  `update()` per iteration, `compute()` on EPOCH_COMPLETED. All accumulation is
  **global over the epoch**, never per-batch.
- `sed_f1score_clip` / `sed_f1score_frame` produce **one scalar per file**
  (`f1_score` on a 1×264 matrix, then `sum/count`). Despite the name,
  `f1score_frame` is **file-level** — it max-pools the frame axis away and uses
  frame evidence only as a gate. No 5-second rows, no site-3 handling.
- Background items (`t.sum()==0`) score 1.0 if nothing crosses threshold, else 0.0
  — so false positives on background are maximally punished. Background audio is
  injected into the validation loader from `/background/data_ssw`.
- `SedLRAPS` uses `clipwise_output` only, threshold-free. sklearn's LRAP
  **returns 1.0 when a sample has zero positives** (verified: `1.0` for an
  all-zero target, vs `0.015` for a 3-positive target; sklearn 1.9.1 source
  confirms `if relevant.size == 0 ...: aux = 1.`). So background items score a
  **free perfect 1.0** and **inflate** `lraps` rather than dragging it down. This
  matters because `track_metric="lraps"` in two configs drives checkpoint
  selection — so those runs selected checkpoints on a metric that rewards
  predicting nothing on background.
- `sed_test.py` **does not compile** — `SyntaxError` at line 60, missing comma
  after `"filename": filename` on line 59. It is unreachable dead code: `"test"`
  is in no config's `metrics`, and `save_dir`/`combine_files` are defined by no
  config. It writes one JSON per audio file (264 raw clip scores + 264 raw frame
  scores), and imports an **unused** `INV_BIRD_CODE` — the exact hook needed to
  build a submission's `labels` column. Submission assembly moved to the kernel.
- **There is no submission-generation code in this repo.** `row_id` = 0 hits.

## Config / CV

- Cross-validation is **not reproducible from this repo.** Each fold config points
  at a pre-built CSV (`/folds/fold_{n}_train.csv`, `/folds5/...`). The split was
  made offline; the CSVs are absent.
- Stratification label is the **primary `ebird_code` only** (`get_label` returns a
  single scalar) — secondary labels are deliberately not used for splitting.
- The `*_2` configs are **half-lr fine-tunes warm-started from their own best
  checkpoint** (`lr 1e-3 → 5e-4`), and the `checkpoint_dir` paths embed the
  stage-1 scores: fold0 0.6328, fold1 0.6134, fold2 0.6305, fold3 0.6370.
- `weight_decay` is honoured **only** by `SedEngine`. All four legacy engines drop
  it, so their config values were inert.
- `gradient_clip_val = 0` everywhere ⇒ `clip_gradients` returns immediately;
  **gradient clipping is disabled in every shipped config** (even though the
  function is a verbatim copy from PyTorch Lightning).
- `accumulation_steps = 1` everywhere ⇒ no gradient accumulation.
- `max_epochs = 50`; `eval_every` = 2 (folds) / 10 (whole-data); `n_saved` = 5
  (4-fold) / 10 (5-fold).
- 5-fold family is internally inconsistent: fold0 tracks `f1score_clip`,
  folds 1–4 track `f1score_frame` — so four models select checkpoints by a
  different metric than the fifth within one ensemble.

## Logging

- Neptune run is created fresh in `__init__`; **no `with_id`/`with_hash`**, so
  resuming from `checkpoint_dir` does **not** resume the Neptune run.
- Slack fires on **ungated** `EPOCH_COMPLETED`, not on `eval_every`. With
  `eval_every=2`, odd epochs re-log **stale** validation metrics.
- `WeightsScalarHandler` / `GradsScalarHandler` attachment is **commented out** —
  no weight or gradient-norm curves are logged.
- `logger_params["prefix_name"]` is accepted by both loggers and **never read**.

## Augmentation — the two-stage design

Waveform-domain (audiomentations, in the DataLoader worker) and spectrogram-domain
(SpecAugment + mixup, inside `forward()`) augmentations are **independent and
both live**. They are not alternatives.

| `aug_name` | Used by | Train pipeline (in order) |
|---|---|---|
| `"default"` | all `final_sed/*` | `AddGaussianNoise(0.001–0.015, p=.3)` → `AddGaussianSNR(p=.3)` → `AddBackgroundNoise("/pinknoise", p=.5)` |
| `"secondary_default"` | all `final_sed_5_fold/*` | `AddGaussianNoise(p=.2)` → `AddGaussianSNR(p=.2)` → `Gain(±15 dB, p=.3)` → `AddBackgroundNoise(p=.2)` → `AddShortNoises(p=.6)` → **then the first three again** |

Valid/test transform is a **pure identity function** — zero augmentation at eval.

> [!warning] audiomentations 0.11.0 `AddGaussianSNR` trap
> In the pinned 0.11.0, `AddGaussianSNR(min_SNR=0.001, max_SNR=1.0)` treats those as a
> **linear ratio of noise-std to signal-std**, *not* dB. So `AddGaussianSNR(p=0.3)`
> can inject noise at up to **100 % of the signal's own std**. Any doc describing this
> as "SNR noise between X dB and Y dB" is wrong. Deterministic and reproducible
> *because* the version is pinned.
> Note `AddBackgroundNoise(min_snr_in_db=3, max_snr_in_db=30)` **is** dB. Mixed units
> in the same pipeline.

- The transform runs on the **whole file before windowing** (`sed_dataset.py:98-99`),
  not on the 30 s snippet. `sf.read` loads the entire multi-minute file, augmentation
  hits the full-length signal, and only then does the dataset crop.
- Consequence for `AddBackgroundNoise`: the 1-minute pink noise is always **tiled**
  and cropped to the file length, since `max_noise_offset = 0` when the noise is
  shorter than the signal.
- `init_workers_fn` reseeds only `np.random` per worker; audiomentations draws
  parameters from Python's `random` module, which is **not** per-worker-seeded.

### Mixup — convex lerp, even/odd pairing, batch halved

`Mixup.get_lambda` uses a **fixed-seed private RNG** (`np.random.RandomState(1234)`)
with no per-epoch reseeding ⇒ the λ sequence is byte-identical every epoch and
identical across DDP ranks. `mixup_alpha=1.` ⇒ `Beta(1,1)` = `U(0,1)`.

`do_mixup` is a **lambda-weighted convex combination of a fixed even/odd pairing** —
not a swap, not a random permutation:

- `out = x[0::2]*λ_even + x[1::2]*λ_odd`, output batch is **exactly half** the input.
- The `transpose(0,-1)` sandwich makes it rank-agnostic, which is why the same
  function serves both `(N, 264)` labels and `(N, 1, 3001, 64)` log-mels.
- Labels: `(32,1,264)` → `reshape(32,264)` → `(16,264)` → `reshape(16,1,264)`. The
  `1` is **synthetic** — reinserted purely to preserve the 3-D contract the loss
  expects.
- The **waveform is not mixed in `prepare_batch`**. `mixup_lambda` rides along in
  the input tuple and mixup happens on the **log-mel after `bn0`**, inside
  `preprocess`. So it is *not* input-space mixup — log-mel + BN are nonlinear.
- Halved-batch bookkeeping is **duplicated** in the model (`b=(b*c)//2; c=1`). Both
  copies must stay in sync or shapes desynchronise.
- Gated on three conditions: `self.training and self.apply_aug and mixup_lambda is not None`.
- Because the sampler is species-uniform, even/odd neighbours are almost always
  *different* species — which is what makes mixup produce meaningful soft targets here.

### Pink noise

`noise_maker.py` vendors `felixpatzelt/colorednoise` (MIT). Algorithm is
**Timmer & Koenig (1995)**: shape each complex Fourier component's amplitude as
`f^(-β/2)` and draw real/imag parts directly as scaled Gaussians, rather than
filtering white noise (which is numerically unstable and only approximately
power-law). `exponent=1` ⇒ pink/flicker. Enforces Hermitian symmetry by zeroing
Nyquist and DC imaginary parts; normalises to unit variance.

Used **nowhere in the pipeline** — it is the offline source of `/pinknoise`,
generated in a separate Kaggle kernel.

### Sampler

`ImbalancedDatasetSampler` (vendored from ufoym) computes `weights = 1/count[label]`,
so each class contributes exactly `1.0` total weight mass ⇒ **every class receives
exactly `1/264` of sampling probability**, regardless of its true frequency.
`multinomial` with `replacement=True`, `num_samples == len(dataset)`.

Strictly **sample-aware**, no batch balancing, no stratification. With 264 classes
and `train_bs=32`, a batch is almost always 32 samples from 32 *different* species —
which is precisely what makes the pos/neg loss separation and `secondary_factor`
meaningful.

Two landmines:
- `get_label` is **accepted and never used**; the sampler calls the dataset's bound
  method instead.
- `sed_dataset.get_label` does `dataset.data[idx]["ebird_code"]` with no `.get()`,
  but background rows are appended as `{"filename": bk}` with **no such key** ⇒
  constructing the sampler on a dataset with `background_audio_dir` raises
  `KeyError`. Survives only because background audio is injected into the
  **valid/test** params, which get no sampler.
- **No `set_epoch`** method, yet `base_engine.py:151-154` still has the DDP hook.
  Harmless at `world_size==1` (the actual submission runs); ignite's
  `auto_dataloader` wrapper usually exposes `set_epoch` so it resolves there.
- The sampler, dropout, and SpecAugment's `torch.randint` all draw from the **same
  global torch RNG stream**.

## Front-end gotchas found

- `sed_audio_utils.py` is **vendored from torchaudio ≤ 0.12** (the `STFT`-as-two-
  `nn.Conv1d`-with-DFT-matrix implementation torchaudio replaced in 0.13). Its ISC
  header says "librosa" — misleading, because torchaudio itself vendored librosa.
- `Spectrogram.__init__` accepts `freeze_parameters` but **hard-codes `True`** when
  constructing the inner `STFT`, so a caller's `False` is silently ignored.
- `librosa.filters.mel` is called with **no `htk=`**, so librosa defaults apply:
  `htk=False` ⇒ **Slaney** mel scale, `norm='slaney'` ⇒ **area-normalised** (not
  unit peak). librosa is **not in `requirements.txt`**, so this is unpinned.
  Note the vendored librosa calls still import librosa — `sed_audio_utils` is *not*
  librosa-free.
- `raise ParameterError(...)` on the negative-`top_db` branch — symbol never
  imported ⇒ `NameError`.
- `self.W` is a 16.8 MB `complex128` numpy array held on a **plain attribute** —
  never a registered buffer, never freed, and duplicated per model instance.
- `SpecAugmentation` (`DropStripes`) masks **in place** on a view of the log-mel.
  `distance` can be `0` (`torch.randint(low=0, high=drop_width)`), so effective
  stripe count is random in `[0, stripes_num]`.
- **The `bn0`-before-SpecAugment ordering is load-bearing and uncommented.** `bn0`
  normalises per mel bin; SpecAugment then writes literal `0`, which post-BN means
  "the per-bin mean", not silence. That is also why the
  `transpose(1,3) / bn0 / transpose(1,3)` sandwich exists.
- Concrete masking at final config: time up to `2 × 64` frames = 1.28 s; frequency up
  to `2 × 8` of 64 bins = **25 % of the mel band** — far more aggressive than PANNs'
  AudioSet setting.
- `audio_utils.py:10` — `pathname.suffix == ".wav"`, **missing the dot**, so always
  False and every read falls through to `librosa.load`.
- `normalise()` derives `min_val`/`max_val` from the **first** array only, then reuses
  them for all subsequent ones.

## Model — measured, not inferred

- **Only `PANNsDense121Att` is live.** All 21 configs set
  `model_name = "sed_dense121att"`. `PANNsDense161/169/201Att` are reachable from
  `sed_engine.py:190-198` but **no config ever names them** — dead-but-runnable
  research code. 9.63 M params, **72.2 % ImageNet-pretrained** (measured).
- **`densenet121.features` downsamples time by ×32, not ×8.** Five stride-2 ops:
  strided `conv0`, `pool0`, and the `AvgPool2d` inside three `_Transition` blocks.
  Frequency collapses **64→2**. So `interpolate_ratio = 32` is inherited from
  upstream PANNs and is *exactly* right for DenseNet by coincidence.
  **The `ConvBlock` stack in `PANNsCNN14Att` is only ×32 over 5 blocks too.**
- **`AttBlock` has no hidden width.** `att`/`cla` are `Conv1d(in_features, 264, 1)`.
  So README's "reduced the AttBlock size to 1024" means **`fc1`'s output width**.
  fc1: D121 1024→1024, D161 2208→**2048**, D169 1664→1024, D201 1920→1024.
  Backbone out dims (measured): 1024 / 2208 / 1664 / 1920.
- **`AttBlock.bn_att` and `.temperature` are dead** — created and initialised,
  never referenced in `forward`. Upstream PANNs has the same defect.

### The `tanh` swap — README is wrong about the baseline

> [!warning] README says `torch.clamp` was replaced. Upstream PANNs actually uses
> `softmax(clamp(x, -10, 10))`, **not** `clamp(x, 0, 1)`.

Both forms keep the softmax. The difference is the *pre-softmax squashing*: with
`clamp(−10, 10)`, saturated entries get **exactly zero gradient** (measured: 49.5 %
of entries sit in the flat tail), whereas `tanh` bounds smoothly to (−1, 1) and
keeps a live gradient everywhere. *Inferred motive:* with ≤100 labels/class,
killing half the attention gradient is a meaningful optimisation handicap.

### Latent crash worth knowing before any refactor

`apply_aug=False` **together with** `apply_mixup=True` → the model's
`do_mixup` call is skipped but `b = (b*c)//2; c = 1` still executes.
Reproduced: `RuntimeError: shape '[1, 1, 501, 264]' is invalid`. Affects all
5 SED model classes. Latent only because every shipped config sets
`apply_aug: True`. Setting `apply_aug=False` to disable SpecAugment **silently
breaks mixup**.

Also: `PANNsCNN14Att` cannot be built from any shipped config — it has no
`top_db` parameter but every config passes one, and the dispatch key is
misspelled `"sed_ccnn14att"`.

## Batch/memory arithmetic (computed, not estimated)

At `sr=32000`, `period=30`, `num_test_samples=2`:

| Quantity | Value |
|---|---|
| samples per snippet | `32000 × 30 = 960,000` |
| mel frames per snippet | `1 + 960000//320 = 3001` |
| hop | `320/32000 = 10 ms` exactly |
| valid batch waveform tensor | `(32, 2, 960000)` = **245.8 MB float32** |
| 10-min file | `19,200,000` samples → `60,001` frames |
| one 5-second row | `500` frames |
| rows per 10-min file | `120` |

**This is the OOM lever.** A validation batch carries ~246 MB of raw waveform
alone, before the log-mel (`(64, 2, 3001, 64)` float32 ≈ 98 MB) or any
activations. With `train_num_workers=8` and `num_workers` prefetching, peak host
RAM is dominated by **in-flight batches, not the dataset** — which is why
`train_bs` varies 24/28/32 across configs while `valid_bs` is pinned at 32.
Under mixup the batch halves to 16, buying headroom; that is plausibly *why*
`train_bs` went 24 → 32 with `apply_mixup=True`.

The model then folds the snippet axis into batch (`sed_models.py:404`
`input_x.reshape(b*c, s)`), so a valid batch of 32 is **64 forward passes** at
3001 frames each — the real driver of validation wall-clock.

## The official competition metric

> [!info] Provenance
> The Kaggle page itself **cannot be scraped** — `webfetch` against
> `kaggle.com/competitions/birdsong-recognition` returns empty (JS-rendered,
> bot-blocked), and `websearch` returned no results from this environment. The
> README quotes the evaluation description verbatim and is treated as the
> in-repo source of truth. The formula below was confirmed against the
> Wikipedia F-score reference. The **submission file schema** is external
> knowledge, flagged as such.

The competition scores **row-wise micro-averaged F1**. Micro-F1 is the harmonic
mean of *global* micro-precision and *global* micro-recall:

```
F1 = 2·TP / (2·TP + FP + FN)
```

computed over the **pooled** TP/FP/FN of every (row, species) cell in the whole
submission — not averaged per row. Three consequences that matter:

- **TP/FP/FN pool across rows and species.** A species is credited once per
  correct row-species pair. Because FN and FP pool in the same denominator,
  *precision and recall are traded off inside a single number* — you cannot
  "win" recall without paying for it in precision. This is precisely why the
  author tuned a **vote count** and **two thresholds** rather than training a
  single well-calibrated model.
- **Every species weighs equally regardless of its true frequency.** A `comros`
  row and a 30-sample species row each contribute one TP. So the metric is
  *already* species-balanced — which is what makes the species-uniform
  `ImbalancedDatasetSampler` the right training distribution rather than a
  merely plausible one.
- **Only positive predictions enter.** TN are ignored entirely.

This is the correct explanation for the README's core claim that thresholding
both heads was "key to reducing false positives": under micro-F1, one spurious
row-species pair adds to the same `2·TP + FP + FN` denominator as one missed
pair, so a vote gate (4-of-13) is a direct precision mechanism.

### Why the in-loop metric is a poor proxy

`sed_f1score_clip` / `sed_f1score_frame` score **one file at a time**, then
`sum/count` — a macro average over files with a hard 0.5 threshold and **no
vote gate**. The official metric is micro over the pooled submission **after**
13-model voting at tuned thresholds. So:

- in-loop `f1score_clip ≈ 0.61–0.64` and final LB `≈ 0.68` are **not
  comparable numbers** — they are different estimators of different things;
- `track_metric = "f1score_clip"` is a *checkpoint-selection proxy*, not a
  leaderboard estimate. It ranks checkpoints; it does not predict the score.

## Known-broken / dead

- `sed_test.py` — SyntaxError, unreachable.
- `audio_dataset.py` V1 — `NameError`: imports `*` from `helpers/audio_utils.py`
  then calls `get_snippets` and `mono_to_color`, neither of which is defined
  there. `spec_augment.py:8` imports `mono_to_color` too. **V1 cannot run.**
- `MainEngine` V1 / V2 `loss_fn` — calls undefined `self.ls_fn`.
- `aux_densnet161.py` — calls `F.*` without importing `torch.nn.functional`.
- `sed_scaled_focal_loss.py` — class is named `SedRemovedFocalLoss`, colliding
  with `sed_removed_focal_loss.py`; imported by nothing.
- `classifier_m_series.py` — dispatched by no engine.
- `sed_audio_utils.py` `raise ParameterError(...)` — symbol never imported →
  `NameError` on that branch.
- `noisy_loss.NoisyLoss` — declares 4 `loss_keys`, returns 2 → `KeyError`.
- `smooth_bce_loss` — zeroes only **row 0**: `loss[0, intersection] = 0.0`.
- `audio_utils.py:10` — `pathname.suffix == ".wav"`, missing the dot; always False.
- `bce_controlled_loss` — adds a `(bs,)` tensor to a `(bs*s,)` one; only valid
  when `s == 1`. Also returns python `0` where the engine calls `.detach()`.
- ~~`base_engine.py:28` — `os.getenv(XLA_USE_BF16)` with **no default** raises
  `TypeError` if unset~~ — **CORRECTED.** `os.getenv` returns `None` for an
  unset var; it cannot raise. The real defect is downstream: `XLA_USE_BF16==1`
  compares a `str` to an `int`, so the fp16-eps branch is **dead code**.
  `clip_gradients` always uses `EPSILON = 1e-6`.
- ~~`ConcatScheduler` built with `durations=[epoch_length]` — length 1 for two
  schedulers~~ — **CORRECTED. This was my error.** Verified against the pinned
  ignite `0.4.0.post1` wheel (`ignite/contrib/handlers/param_scheduler.py:446`):
  `if len(schedulers) != len(durations) + 1: raise ValueError`. Two schedulers
  with `durations=[epoch_length]` (length 1) is the **required** shape, not a bug.
  The idiom is copied verbatim from the class's own docstring example.
- `lr_finder` does not restore the optimizer LR after the sweep.
- `train()` **swallows the traceback** then calls `_end_of_training()` — a crashed
  run exits **0** and appears "finished" in Neptune. Silent-failure hazard.
- Checkpoint saver is **not** rank-gated (`:207`); only the logger is. Under DDP
  every rank writes to the same `DiskSaver` dir.
- `setup_common_training_handlers` installs far less than its name suggests here:
  `stop_on_nan=False`, `lr_scheduler=None`, `to_save=None`, so no
  `TerminateOnNan`, `LRScheduler`, `Checkpoint`, or `TimeProfiler`. Only
  `empty_cuda_cache` and progress bars.

## Ignite 0.4.0.post1 semantics verified from the wheel

Extracted `pytorch_ignite-0.4.0.post1` to check claims against the *pinned*
version rather than whatever is installed locally:

- `State.event_to_attr` maps `EPOCH_COMPLETED → "epoch"`, so
  `Events.EPOCH_COMPLETED(every=2)` filters on **epoch number**, not iteration.
  With `max_epochs=50` that is **25 validation runs**, at epochs 2,4,…,50.
- `ConcatScheduler` raises unless `len(schedulers) == len(durations) + 1`
  (`param_scheduler.py:446`).
- `@sync_all_reduce` uses `getattr(self, attr, None)`, so a missing attribute is
  **silently skipped** rather than raising.
- `Metric.attach` default `usage=EpochWise()` ⇒ `reset()` on `EPOCH_STARTED`,
  `update()` per `ITERATION_COMPLETED`, `compute()` on `EPOCH_COMPLETED`.
  Accumulation is global over the epoch.
- ignite 0.4.0 ships the same memory-cleanup TODO commented out that
  `base_engine.complete_clear` reimplements (`engine.py:775`
  `# self.state.batch = None`). The author's handler is a deliberate
  backport, not a reinvention.
- `sed_audio_utils.py` holds `self.W`, a 16.8 MB numpy complex128 array, on a
  plain attribute — never a buffer, never freed.

### Correction log — 2026-10-07 recheck pass
- modules/01: stereo failure mode, Sakpsucker misspelling, dict-branch path, get_dict_value count, secondary_labels format.
- modules/02: provenance (torchlibrosa), HTK deleted, librosa<0.10 pin, top_db=80 exception.
- modules/07: 3001 frame shape, 11+2 loss split, threshold sweep evidence.
- modules/08: birds/nocall format (external), period=30 arithmetic note.
- modules/06: checkpoint {epoch}, max_epochs override, state contents.
- modules/05: XLA_USE_BF16 NameError.
