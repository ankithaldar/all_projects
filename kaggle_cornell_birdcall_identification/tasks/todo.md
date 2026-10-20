# Task list — Cornell Birdcall: PyTorch Lightning + TensorFlow/Keras

## Status — 2026-10-21

**Implemented and tested.** 100 modules under `src/birdcall/`, 89 test
modules under `tests/`, **788 passing, 4 skipped, 0 failing** on CPU.
The two commits are `fd2cab5` (the PyTorch Lightning version, 239 files,
512 tests green in isolation) and `4be1c7e` (the TensorFlow/Keras
version, 55 files).

Every task below is implemented except the four items listed under
**Deliberately not done**, which are marked inline with a `⏭` and a
reason.

**Deliberately not done**

| item | why |
|---|---|
| **Weight conversion** of the original's 744-tensor checkpoint | there is no checkpoint in this environment, and the original repository does not ship one. The two ports are compared numerically (front-end, losses, metrics, LR curve, submission bytes) instead |
| **One-epoch parity against live Ignite** | `pytorch-ignite==0.4.0.post1` is not installable against the installed torch, and there is no competition audio. The LR curve is verified against the golden fixture transcribed from the guide, which the reference values match to 1e-12 |
| **DDP sharding of the imbalanced sampler** | there is no GPU and no multi-node setup here. The sharding is a one-line index-space split and is documented, but unverified |
| **A real training run and a leaderboard score** | there is no competition audio and no GPU. Everything is verified with synthetic data |

**What the original source changed.** The upstream repository is public,
so the implementation was checked against it rather than only against
`docs/`. Four documentation claims turned out to be wrong and the
implementation follows the source: the `_augd` criterion drops **30 %** of
the primary positions (not 70 %), `fc1` is an `nn.Linear` (not a 1×1
convolution), the mel matrix is a frozen `nn.Parameter` (not a buffer),
and the competition's submission column is `birds` (not `labels`). The
264 species codes and the 21 experiment configurations were recovered
from it too.

---

Ordered execution list. The design decisions, risks and open questions live in
[`plan.md`](plan.md). Every task obeys the **code style contract** in
`plan.md` (2-space indent, 80 columns, single quotes, Google docstrings, the
mandatory file header, and `.pylintrc` at the repository root left untouched).

**Naming rule.** `pl_<name>.py` = PyTorch Lightning. `tf_<name>.py` =
TensorFlow/Keras. No prefix = shared, framework-agnostic.

**Verification commands** assume the project venv:

```bash
cd kaggle_cornell_birdcall_identification
source .venv/bin/activate
pip install -e .          # after T0.1
pytest -q                 # whole suite
pylint src/birdcall tests  # against the repository .pylintrc
```

---

## Phase 0 — Contracts and environment

### Task T0.1: Project skeleton, packaging, dependency pins

**Description.** Turn the empty scaffold into an installable project. `pyproject.toml`
declares the package under `src/` (setuptools, `package-dir = {"" = "src"}`),
exposes the three console scripts, and configures pytest. Two requirement files
are written: one for the PyTorch Lightning stack, one for the TensorFlow/Keras
stack, both pinned to versions that exist for **cp312** (the project venv's
Python). `librosa` is a **build-time-only** dependency (see AD-3). The
repository `.pylintrc` is *not* touched.

**Acceptance criteria**
- [ ] `pip install -e .` succeeds in `.venv` (Python 3.12) and `import birdcall` works
- [ ] `pip install -r requirements-tf.txt` resolves **without** upgrading or removing numpy below 1.27
- [ ] `birdcall` console scripts are importable (`birdcall.train_pl:main`, `birdcall.train_tf:main`, `birdcall.run_vote:main`)
- [ ] pytest collects `tests/` and reports "no tests ran" (not a collection error)
- [ ] `.pylintrc` is byte-identical to the repository root copy (verified with `sha256sum`)

**Verification**
- [ ] `pip install -e . && python -c 'import birdcall; print(birdcall.__file__)'`
- [ ] `pip install --dry-run -r requirements-tf.txt` → no `numpy>=2` resolution
- [ ] `sha256sum ../../.pylintrc` matches the recorded value

**Dependencies:** None
**Files likely touched:** `pyproject.toml`, `requirements.txt`, `requirements-tf.txt`, `src/birdcall/__init__.py`
**Estimated scope:** S (2-3 files)

---

### Task T0.2: Config contract — pydantic schema + 21 YAMLs

**Description.** The config surface the docs pin down: 39 attributes, 21
experiment configs, `BIRD_CODE` / `EBIRD_LABEL` / `INV_EBIRD_LABEL` /
`AUDIO_CONFIG`. Per AD-2/AD-24 the **YAML file is the source of truth** and a
pydantic model validates it. `config_params/parameters.py` (shared) defines
`Parameters` plus the nested models (`ModelConfig`, `DatasetParams`,
`LoggerParams`, `CheckpointParams`, `RunParams`) with
`extra='forbid'`, `validate_assignment=True` and `Literal` types for every
enumerated name (`model_name`, `criterion_name`, `scheduler_name`,
`optimizer_name`, `logger_name`, `aug_name`, `track_metric`).
`config_params/loader.py` (shared) loads a YAML path into a `Parameters` and
raises a readable error naming the file, the key and the allowed values. The
21 YAMLs are generated from the documented variation table
(`scripts/build_configs.py`, AD-29) and then hand-editable; the
`example_config.yaml` and the new `inference/final_5sec_sed.yaml` follow the
same schema. `ensembles/ensemble_13.yaml` names the 13 members and the vote
thresholds (AD-25/AD-38). `logger_params['params']` becomes a computed field
(AD-27), and the seeds join the schema (AD-31).

**Acceptance criteria**
- [ ] all 21 YAMLs load and validate; every instance exposes all 39 attributes
- [ ] an unknown key is rejected (`extra='forbid'`) with the file and key named
- [ ] an unknown `model_name` / `criterion_name` / `track_metric` is rejected with the allowed values listed
- [ ] `criterion_params` that do not match the selected criterion are rejected by the `model_validator` (AD-26)
- [ ] the 21 varying attributes match the table in `docs/src-reference/config-params.md`; the 18 invariant ones are equal across all 21
- [ ] `track_metric` is `f1score_clip` for the 4-fold family, `f1score_frame` for 5-fold folds 1-4, `f1score_clip` for 5-fold fold 0, `lraps` for the two `*_1`/`*_2` retrains (Q3 — preserved)
- [ ] the label space has exactly 264 entries with contiguous indices 0..263
- [ ] `seed`, `mixup_seed` (default 1234) and `deterministic` are schema fields
- [ ] `logger_params['params']` is derived from the model, not hand-written
- [ ] `inference/final_5sec_sed.yaml` has `period=30`, `num_test_samples=20`, `valid_bs=1`
- [ ] `ensembles/ensemble_13.yaml` lists 13 member config paths + `votes=4`, `clip_threshold=0.3`, `frame_threshold=0.3`

**Verification**
- [ ] `pytest -q tests/test_config_params.py` → all 21 load; attribute set matches
- [ ] `pytest -q tests/test_config_validation.py` → unknown key, bad enum, mismatched criterion all fail loudly
- [ ] `pytest -q tests/test_config_sweep.py` → the variation table still holds

**Dependencies:** T0.1
**Files likely touched:** `src/birdcall/config_params/parameters.py`, `model_config.py`, `dataset_params.py`, `logger_params.py`, `checkpoint_params.py`, `run_params.py`, `criterion_params.py`, `loader.py`, `configs.py`, `src/birdcall/config_params/*.yaml`, `src/birdcall/config_params/final_sed/*.yaml` (15), `src/birdcall/config_params/final_sed_5_fold/*.yaml` (5), `src/birdcall/config_params/inference/final_5sec_sed.yaml`, `src/birdcall/config_params/ensembles/ensemble_13.yaml`, `src/birdcall/scripts/build_configs.py`
**Estimated scope:** L (5+ files — the 21 YAMLs are mechanical; generate them, do not hand-type them)

---

### Task T0.3: Shared LR-schedule maths + the Ignite golden fixture

**Description.** One framework-agnostic module holding the two schedule
formulas transcribed from `pytorch-ignite 0.4.0.post1`
(`ignite/contrib/handlers/param_scheduler.py`), plus the `ConcatScheduler`
boundary semantics: the linear phase runs for exactly `epoch_length` events
with its own counter, then the cosine phase restarts its counter at 0.

```
LinearCyclical:  p = event_index / cycle_size
                 v = end_value + (start_value - end_value) * abs(p - 0.5) * 2
CosineAnnealing: p = event_index / cycle_size
                 v = start_value + (end_value - start_value) / 2 * (1 - cos(pi * p))
```

The golden fixture is the five reference values published in
`docs/ports/pytorch-lightning-guide.md` § Phase 5 for
`lr=0.001, lr_scale_factor=0.01, epoch_length=7, num_epochs=5`:
iterations 0, 1, 2, 33, 34 → `1e-5, 0.00015143, 0.00029286, 0.00016292,
0.00013223`. These were re-derived here from the formulas above and agree to
7e-8. A `warmup` (concat) and a `one_cycle` (cosine only) variant are exposed;
the module returns **multipliers** so both backends can apply them to their own
optimiser base LR.

**Acceptance criteria**
- [ ] the module contains no `import torch` and no `import tensorflow`
- [ ] the linear phase ends at exactly `lr` and the cosine phase starts at exactly `lr` (continuous curve)
- [ ] the phase boundary is at exactly `epoch_length` iterations
- [ ] the five golden values reproduce to < 1e-12 absolute
- [ ] the module exposes `none`, `warmup_with_cosine` and `one_cycle_cosine` by name, with a clear error on an unknown name

**Verification**
- [ ] `pytest -q tests/test_schedules.py` → golden table matches to 1e-12, boundary asserted
- [ ] optional cross-check: install `pytorch-ignite==0.4.0.post1` in a scratch venv and compare the live `ConcatScheduler` output for `N*E` steps

**Dependencies:** T0.1
**Files likely touched:** `src/birdcall/engines/schedules.py`, `tests/test_schedules.py`
**Estimated scope:** S (2 files)

---

## Checkpoint 0 — contracts

- [ ] `pip install -e .` succeeds; `import birdcall` works
- [ ] all 21 configs construct and expose all 39 attributes
- [ ] the LR schedule reproduces the golden fixture
- [ ] `pylint src/birdcall` reports no error-level findings

---

## Phase 1 — PyTorch reference core + the deliverable

> The `pl_*` modules are the **reference implementation**: the Lightning port
> (Phase 2) imports them unchanged, and the Keras port (Phase 3) is checked
> against them. They are not a third deliverable.

### Task T1.1: Audio front-end + mel matrix

**Description.** The in-graph front-end: STFT (two `Conv1d` with **frozen**
DFT×window weights, `bias=False`, `center=True` with **reflect** padding by
`n_fft//2`, `hop_length = win_length // 4`), `Spectrogram` (power, `real²+imag²`),
`LogmelFilterBank` (`librosa.filters.mel(...).T`, frozen; `power_to_db` =
`10*log10(clamp(x, amin=1e-10)) - 10*log10(max(amin, ref))`), and
`SpecAugmentation` (time-dropper on dim 2, then freq-dropper on dim 3, **per
batch item**, no-op when not training). The mel matrix is precomputed by
`helpers/mel_matrix.py` (shared) and committed as a `.npy` with the librosa
version in the filename (AD-3); `librosa` is therefore not a runtime
dependency. The fp32 guard for `power_to_db` is in place (AD-14).

**Acceptance criteria**
- [ ] a 30 s waveform at 32 kHz yields exactly **3001** mel frames and 64 mel bins
- [ ] the reflect pad is explicit; without it the frame count would be 2977 (asserted)
- [ ] the power (not magnitude) formula is used (a magnitude input gives a ×2 error — asserted)
- [ ] the mel matrix is **Slaney + area-normalised** (librosa's), not HTK
- [ ] `power_to_db` runs in fp32 even under an autocast context
- [ ] `SpecAugmentation` is a no-op in `eval()` mode, and draws an **independent** band per batch item
- [ ] the frozen buffers are excluded from `model.parameters()`

**Verification**
- [ ] `pytest -q tests/test_pl_audio_frontend.py` → frame count, dtype, training flag, per-item band
- [ ] `pytest -q tests/test_mel_matrix.py` → the committed matrix matches a freshly built librosa one to 1e-6

**Dependencies:** T0.1
**Files likely touched:** `src/birdcall/helpers/mel_matrix.py`, `src/birdcall/helpers/pl_frontend_stft.py`, `pl_frontend_spectrogram.py`, `pl_frontend_logmel.py`, `pl_frontend_spec_augment.py`, `src/birdcall/scripts/build_mel_matrix.py`, `tests/test_pl_frontend.py`
**Estimated scope:** M (3-5 files)

---

### Task T1.2: Model — `PANNsDense121Att`

**Description.** The winning architecture: DenseNet121 `.features`
(ImageNet-pretrained, **not** re-initialised), `fc1` 1024→1024 with ReLU,
`AttBlock(1024, 264)` (`softmax(tanh(att(x)), dim=-1) * sigmoid(cla(x))` summed
over **time**; `att`/`cla` are `Conv1d(k=1)` with bias; `bn_att` and
`temperature` are dead and stay dead), the `max_pool1d + avg_pool1d` fusion
(k=3, s=1, p=1), `interpolate_ratio=32` as **interleaved** nearest-neighbour
upsampling, `pad_framewise_output` repeating the **last** frame, and the
`bn0` → SpecAugment ordering (G10). Mixup is applied to the **spectrogram**
inside `preprocess`, after SpecAugment. The forward takes the tuple
`(input_x, mixup_lambda)` and returns the dict
`{"clipwise_output", "framewise_output"}`.

**Acceptance criteria**
- [ ] the shape trace is exact: `(B,1,S)` → `(B,1024,93)` → `(B,264)` clip + `(B,3001,264)` frame
- [ ] the DenseNet time axis is **93** (measured, not assumed) and the tail pad is exactly **25** frames
- [ ] the attention sums to 1 over the **time** axis
- [ ] `clipwise_output` is the sum over time of `norm_att * cla`
- [ ] `framewise_output`'s last 25 frames are all equal to frame 3000
- [ ] the output is a **dict** with both heads
- [ ] mixup halves the batch to `(B*C)//2` and the reshape back is consistent
- [ ] `apply_aug=False` with mixup enabled does **not** crash (AD-15)
- [ ] the backbone keeps its ImageNet initialisation (not re-initialised)

**Verification**
- [ ] `pytest -q tests/test_pl_sed_models.py` → shape trace, attention sum, tail pad, mixup halving
- [ ] `pytest -q tests/test_pl_sed_models.py::test_apply_aug_false_with_mixup` → no crash

**Dependencies:** T1.1
**Files likely touched:** `src/birdcall/models/pl_model_att_block.py`, `src/birdcall/models/pl_model_panns_dense121_att.py`, `src/birdcall/models/pl_registry.py`, `tests/test_pl_model.py`
**Estimated scope:** M (2 files)

---

### Task T1.3: Dataset + imbalanced sampler

**Description.** The data pipeline, split so the pure-numpy half is shared and
the torch half is `pl_`. `datasets/sed_dataset_core.py` (shared) owns: the CSV
read, `ast.literal_eval` for `secondary_labels` (AD-9), the background-audio
rows (label-less → all-zero one-hot), the `.mp3` → `.wav` rewrite, the **264-vector**
one-hot broadcast to every snippet, and the two slicing modes (train: random
offset placement for a short clip, random crop for a long one; valid:
deterministic contiguous slices, tail zero-pad, silent truncation).
`datasets/pl_dataset_sed.py` wraps it as a `torch.utils.data.Dataset` returning
`{"filename", "waveforms", "all_labels", "primary_labels", "secondary_labels"}`
and owns `init_workers_fn` and `additional_loader_params`.
`datasets/pl_sampler_imbalanced.py` implements the inverse-frequency sampler:
weights `1/count`, `torch.multinomial(..., replacement=True)`,
`num_samples = len(dataset)`.

**Acceptance criteria**
- [ ] a training draw is non-deterministic; a validation draw is deterministic
- [ ] the label tensor is `(S, 264)` — a 264-vector broadcast, **not** a per-frame matrix
- [ ] background rows carry an all-zero label vector
- [ ] `secondary_labels` is parsed with `ast.literal_eval` (no `eval` anywhere in the tree)
- [ ] a NaN `secondary_labels` cell does **not** raise
- [ ] the sampler balances: over many draws every species appears with equal total probability
- [ ] the sampler is not a `DistributedSampler` (documented; sharding is verified in T2.2)

**Verification**
- [ ] `pytest -q tests/test_sed_dataset.py` → slicing modes, label shape, background rows
- [ ] `pytest -q tests/test_pl_imbalanced_sampler.py` → class balance within tolerance
- [ ] `grep -rn "eval(" src/birdcall/datasets/` → no `eval` on CSV content

**Dependencies:** T0.2
**Files likely touched:** `src/birdcall/datasets/sed_dataset_core.py`, `src/birdcall/datasets/pl_dataset_sed.py`, `src/birdcall/datasets/pl_sampler_imbalanced.py`, `tests/test_sed_dataset.py`
**Estimated scope:** M (4 files)

---

### Task T1.4: Waveform augmentation stacks + `Mixup`

**Description.** The two audiomentations stacks, shared (they are numpy-in /
numpy-out and drop into a `tf.data.map` unchanged — the one free parity win).
`sed_default_augment`: `AddGaussianNoise(0.001, 0.015, p=0.3)` →
`AddGaussianSNR(p=0.3)` → optional `AddBackgroundNoise(dir, p=0.5)`.
`sed_background_augment`: the **palindromic bracket** — noise(p=0.2) →
SNR(p=0.2) → `Gain(-15..15 dB, p=0.3)` → optional background(p=0.2) →
optional short-noises(p=0.6) → **the noise/SNR/gain trio again**. The
validation transform is the identity. The transform signature is
`transform(x, sr)`. `augmentations/mixup.py` (shared) keeps the hard-coded
`random_seed=1234` (AD-13) and the even/odd pairing.

**Acceptance criteria**
- [ ] the composed transform returns an array of the **same length** as the input
- [ ] the palindromic bracket is preserved (the noise/SNR/gain trio appears twice)
- [ ] the validation transform is the identity
- [ ] `Mixup.get_lambda` draws one `Beta(a,a)` per **pair** and emits `[lam, 1-lam]`
- [ ] `do_mixup` produces `x_even*lam_even + x_odd*lam_odd` with the batch halved
- [ ] the same seed yields the same lambda sequence on run 2 as on run 1 (AD-13)

**Verification**
- [ ] `pytest -q tests/test_sed_augment.py` → length preserved, bracket order, identity
- [ ] `pytest -q tests/test_mixup.py` → pairing, halving, seed stability

**Dependencies:** T1.3
**Files likely touched:** `src/birdcall/augmentations/sed_augment_default.py`, `src/birdcall/augmentations/sed_augment_background.py`, `src/birdcall/augmentations/mixup.py`, `src/birdcall/augmentations/registry.py`, `tests/test_sed_augment.py`, `tests/test_mixup.py`
**Estimated scope:** S (3 files)

---

### Task T1.5: The five live criteria

**Description.** Only the five criteria the SED path can reach, in
`losses/pl_loss_*.py` (one file per criterion), each keeping the `(scalar_tensor, dict_of_scalars)`
return contract and publishing `loss_keys`:
`focal_loss_standard` (`bce_loss, F_loss`), `sed_removed_focal_loss`
(`bce_loss, F_loss, FRemoved_loss`), `sed_scaled_pos_neg_focal_loss`
(`bce_loss, F_loss, FScaled_loss, F_loss_0, F_loss_1`),
`sed_scaled_pos_neg_focal_loss_augd` (same five), `bce_controlled_standard_loss`
(**empty** `loss_keys` — correct, do not "fix"). The prediction is **already
sigmoided** (G2) and is clamped to `[1e-5, 1-1e-5]`. The masks are built under
`no_grad`. The `_augd` variant zeroes 70 % of the primary positions with a
mask drawn from the **global seeded** RNG (AD-12).

**Acceptance criteria**
- [ ] `SedScaledPosNegFocalLoss(gamma=0, alpha_0=1, alpha_1=1, secondary_factor=1)` equals `nn.BCELoss()` to **1e-6** (G1)
- [ ] the criterion is selected from a registry; an unknown name raises a clear error naming the valid keys (AD-10)
- [ ] each criterion's `loss_keys` matches the documented list exactly
- [ ] `BCEControlledStandardLoss.loss_keys == []`
- [ ] no gradient flows through the mask construction
- [ ] the `_augd` variant drops ~70 % of the primary positions, and the returned `FScaled_loss` is the **pre-drop** value
- [ ] the loss consumes only `clipwise_output` (G3)

**Verification**
- [ ] `pytest -q tests/test_pl_sed_losses.py` → inertness to 1e-6, `loss_keys`, registry error, no-grad masks
- [ ] `pytest -q tests/test_pl_sed_losses.py::test_augd_drops_primaries`

**Dependencies:** T0.2
**Files likely touched:** `src/birdcall/losses/pl_loss_focal_standard.py`, `pl_loss_removed_focal.py`, `pl_loss_scaled_pos_neg_focal.py`, `pl_loss_scaled_pos_neg_focal_augd.py`, `pl_loss_bce_controlled_standard.py`, `src/birdcall/losses/pl_registry.py`, `tests/test_pl_losses.py`
**Estimated scope:** S (2 files)

---

### Task T1.6: `LRAP` / `F1-clip` / `F1-frame` metrics

**Description.** The three validation metrics in `metrics/pl_metric_*.py` (one file per metric),
as epoch-wise accumulators that **reduce, then divide** (a mean of per-item
scores, not a global score). `LRAP`: sklearn
`label_ranking_average_precision_score` per item on
`clipwise_output.max(axis=1)[0]` vs `target[:,0,:]`. `F1-clip`: micro-F1 on the
clip head, with the **background branch** (an all-zero target scores 1.0 only
if nothing fires). `F1-frame`: micro-F1 on the frame head **ANDed** with a
hard-coded **0.3** clip gate — `(pf >= threshold) & (pc >= 0.3)` (G4). No
sigmoid anywhere (G2). `LRAP` keeps the original's no-background-branch
semantics as the default; a background-aware variant exists behind a
constructor flag, off by default (AD-11).

**Acceptance criteria**
- [ ] a background row with nothing firing scores **1.0** on both F1 metrics
- [ ] a background row with anything firing scores **0.0**
- [ ] a frame that passes but whose clip score is < 0.3 contributes nothing (the two gates are independent)
- [ ] the `0.3` is a literal in the code, not a config value
- [ ] the accumulator reduces then divides (per-item mean, not a global score)
- [ ] no sigmoid is applied
- [ ] the metrics expose `reset` / `update` / `compute` and can be attached to a backend

**Verification**
- [ ] `pytest -q tests/test_pl_sed_metrics.py` → background branch, dual gate, reduce-then-divide

**Dependencies:** T0.2
**Files likely touched:** `src/birdcall/metrics/pl_metric_lraps.py`, `pl_metric_f1score_clip.py`, `pl_metric_f1score_frame.py`, `src/birdcall/metrics/pl_registry.py`, `tests/test_pl_metrics.py`
**Estimated scope:** S (2 files)

---

### Task T1.7: Frame-preserving score artefact writer

**Description.** The handoff between a model and the vote, in
`metrics/scoring.py` (shared, no framework import). The original's `SedTest`
max-pools the entire time axis away and therefore **cannot** vote
(`docs/modules/08` Part E). This writes a `.npz` per audio file per model
holding `clip` `(264,)` and `frames` `(T, 264)` in float32, plus the audio id
and the frame rate, so the vote can slice 5-second rows. It also writes the
**original** JSON dump format (byte-identical keys) for parity with the
original's artefact, with the four original blockers fixed.

**Acceptance criteria**
- [ ] the `.npz` keeps the **time axis** — a 10-minute file yields 60001 frames, not 1 record
- [ ] the JSON dump's keys are byte-identical to the original's: `filename`, `predicted_frames`, `predicted_clip`, `targets`, `lraps_score_clip`, `lraps_score_frames`
- [ ] the writer is importable without torch or tensorflow (shared surface rule)
- [ ] a round trip (write → read) preserves the float32 values exactly

**Verification**
- [ ] `pytest -q tests/test_scoring.py` → frame axis preserved, JSON keys, round trip
- [ ] `pytest -q tests/test_shared_is_framework_free.py` → no framework import in `scoring.py`

**Dependencies:** T1.6
**Files likely touched:** `src/birdcall/metrics/scoring.py`, `tests/test_scoring.py`
**Estimated scope:** S (2 files)

---

### Task T1.8: Vote + `submission.csv`

**Description.** The winning mechanism, in `voting/vote.py` (shared, pure
numpy, no framework import) plus the `run_vote.py` entry point. The rule, from
`README.md:45` and the in-repo ancestor `sed_f1score_frame.py:35,40`: a species
is accepted for a 5-second row when **≥ 4 of the 13 models** fire with
**clip ≥ 0.3 AND frame ≥ 0.3**. A model fires for species *i* in row *k* when
`(max frame score of row k) >= 0.3` and `(clip score) >= 0.3`. A row is 500
frames (5 s at 10 ms/frame); a 10-minute file yields 120 rows. Sites 1/2 emit
5-second rows; site 3 emits one row per file. A row with no accepted species
becomes `nocall`. The `votes` count and both thresholds are **parameters**
with the winning defaults (4, 0.3, 0.3) so the README's 13-model and 9-model
sweep tables are reproducible without a code change (Q5). `row_id` is built in
one isolated function (Q1).

**Acceptance criteria**
- [ ] a synthetic 13-model set with exactly **4** fires at ≥0.3/≥0.3 → species accepted
- [ ] exactly **3** fires → rejected
- [ ] a model that passes the clip gate but not the frame gate does **not** vote (G4)
- [ ] the two gates are independent — a model can vote on the frame gate alone being absent
- [ ] a 10-minute file yields **120** rows; a row with no accepted species is `nocall`
- [ ] the CSV has the columns `row_id` and `labels`; multiple species are space-separated
- [ ] site 3 emits one row per file, sites 1/2 emit 5-second rows
- [ ] the module imports no framework

**Verification**
- [ ] `pytest -q tests/test_vote.py` → 4-accept / 3-reject, dual gate, row count, `nocall`, CSV shape
- [ ] `pytest -q tests/test_shared_is_framework_free.py` → no framework import in `voting/`

**Dependencies:** T1.7
**Files likely touched:** `src/birdcall/voting/vote.py`, `src/birdcall/run_vote.py`, `tests/test_vote.py`
**Estimated scope:** M (3 files)

---

## Checkpoint 1 — the reference implementation is complete

- [ ] log-mel frames == 3001; DenseNet time axis == 93; tail pad == 25
- [ ] the inert focal loss equals `nn.BCELoss()` to 1e-6
- [ ] the three metrics reproduce the documented behaviour to 1e-6
- [ ] a synthetic 13-model set produces a `submission.csv`; 4 fires accept, 3 reject
- [ ] `pylint src/birdcall` reports no error-level findings

---

## Phase 2 — PyTorch Lightning port

### Task T2.1: `LightningModule` + datamodule

**Description.** The control-plane replacement. `engines/pl_lightning_module.py`
holds the model, the criterion and the three steps, replacing the original's
seven `_init_*` hooks with **registry dicts** (AD-10). `training_step`:
`prepare_batch` → `model((x, mixup_lambda))` → `criterion(y_pred['clipwise_output'], y)`
→ `self.log_dict` per `loss_keys` entry. `validation_step`: the same forward,
logged as `valid_<name>` (AD-17). `test_step`: returns
`{"y_pred", "x", "y"}` with no loss. `datamodules/pl_datamodule_sed.py` owns
the three dataloaders and the sampler.

**Acceptance criteria**
- [ ] `training_step` computes the loss from **only** `clipwise_output` (G3)
- [ ] every `loss_keys` entry appears as a logged metric
- [ ] validation metrics are logged as `valid_<name>`; the checkpoint monitor string is the same
- [ ] an unknown `model_name` / `criterion_name` raises a clear error at construction (AD-10)
- [ ] the `mode='test'` branch reads `batch['waveforms']` (the original's `NameError` is fixed)
- [ ] the datamodule builds train/valid/test loaders from the config's `*_ds_params`

**Verification**
- [ ] `pytest -q tests/test_pl_lightning_module.py` → step shapes, logged metric names, registry errors

**Dependencies:** T1.2, T1.3, T1.5, T1.6
**Files likely touched:** `src/birdcall/engines/pl_lightning_module.py`, `src/birdcall/datamodules/pl_datamodule_sed.py`, `tests/test_pl_lightning_module.py`
**Estimated scope:** M (3 files)

---

### Task T2.2: `Trainer` wiring

**Description.** The `Trainer` configuration and the checkpoint/resume
behaviour, in `engines/pl_trainer.py` and `optimizers/pl_optimizer_adamw.py`.
`max_epochs` ← `run_params['max_epochs']`; `accumulate_grad_batches` ←
`accumulation_steps`; `gradient_clip_val` ← `gradient_clip_val` (the original's
23 hand-rolled lines are deleted); `check_val_every_n_epoch` ← `eval_every`;
`enable_progress_bar` ← `add_pbar`; `num_sanity_val_steps=0`; `strategy` ←
`dist_params`. AdamW with `amsgrad=True` and the config's `weight_decay`.
The LR schedule is the shared `engines/schedules.py` maths driven as a
`LambdaLR` with `interval='step'` — identical to Ignite's per-iteration
advance because every shipped config has `accumulation_steps=1` (the caveat is
documented, not silently changed). Checkpointing uses `ModelCheckpoint` with
`save_top_k=n_saved`, `save_last=True`, and `monitor` = the tracked metric
mapped to the logged name. Two resume modes: `load_model_only=True` restores
**only** the weights and leaves the optimiser at its initial state;
`load_model_only=False` is a full `fit(ckpt_path=...)` resume. Seeding is
`seed_everything(42 + global_rank, workers=True)`.

**Acceptance criteria**
- [ ] the LR curve matches the golden fixture step for step, max delta < 1e-12
- [ ] exactly `n_saved` checkpoints exist and the best one is the highest tracked metric
- [ ] `save_last` exists
- [ ] the monitor name resolves — a checkpoint is actually written (the silent-never-fires trap)
- [ ] a weights-only resume leaves the optimiser at its **initial** LR
- [ ] a full resume restores the epoch counter
- [ ] `max_epochs` on resume comes from the checkpoint (documented; changing it in the config has no effect)
- [ ] `seed_everything` is called with the per-rank offset

**Verification**
- [ ] `pytest -q tests/test_pl_trainer.py` → checkpoint count, monitor resolution, both resume modes
- [ ] `pytest -q tests/test_schedules.py::test_pl_scheduler_matches_golden`

**Dependencies:** T2.1, T0.3
**Files likely touched:** `src/birdcall/engines/pl_trainer.py`, `src/birdcall/optimizers/pl_optimizer_adamw.py`, `tests/test_pl_trainer.py`
**Estimated scope:** M (3 files)

---

### Task T2.3: Callbacks + loggers

**Description.** The instrumentation the framework does not provide.
`callbacks/pl_callback_log_every.py`, `pl_callback_slack.py` and `pl_callback_inference_dump.py`: an every-N-step counter callback (the original's
`Events.ITERATION_COMPLETED(every=log_every)` per-handler filter has no
Lightning equivalent), a Slack webhook callback (one message per training
epoch, one per metric key at validation end, reading `SLACK_URL` from the
environment), and the inference dump callback that writes the `.npz` score
artefact. `loggers/pl_logger_wandb.py` and `pl_logger_print.py`: **wandb** + print, carrying the original's
Neptune series names unchanged — `train/loss`, `train/<loss_key>`, `train/lr`,
`train/<metric>`, `valid/<metric>` (AD-41). The `logger_params` 6 keys keep
their names and map onto wandb's `project`, the log cadence, `name`, **`group`**
(`prefix_name`), `tags` and `config` (`params`) (AD-42/AD-43). The original's
`neptune-client==0.4.107` is dropped.

**Acceptance criteria**
- [ ] the every-N callback fires every N steps and **not** every step
- [ ] the Slack callback sends one message per epoch and one per metric key at validation end
- [ ] a missing `SLACK_URL` does not crash the run
- [ ] the metric series names match the original's exactly
- [ ] a wandb failure is caught and downgraded to a warning — never fatal (AD-44)
- [ ] the run works with `WANDB_MODE=offline` and with `WANDB_DISABLED=true`
- [ ] the dump callback writes the `.npz` artefact with the time axis intact

**Verification**
- [ ] `pytest -q tests/test_pl_callbacks.py` → counter cadence, Slack no-op without a URL, dump artefact
- [ ] `pytest -q tests/test_pl_loggers.py` → series names, offline mode, failure is non-fatal

**Dependencies:** T2.1
**Files likely touched:** `src/birdcall/callbacks/pl_callback_log_every.py`, `pl_callback_slack.py`, `pl_callback_inference_dump.py`, `src/birdcall/loggers/pl_logger_wandb.py`, `pl_logger_print.py`, `tests/test_pl_callbacks.py`, `tests/test_pl_loggers.py`
**Estimated scope:** M (4 files)

---

### Task T2.4: Port verification

**Description.** The end-to-end proof that the Lightning port reproduces the
reference core. A synthetic dataset (fixed seed, short waveforms) is trained
for one epoch by the reference path and by the Lightning path; per-step losses
must match to 1e-5. Plus the `train_pl.py` entry point: argparse `--config`
(dotted module path) → `importlib` → `module.Parameters()` → `Trainer(...).fit(...)`.

**Acceptance criteria**
- [ ] `python -m birdcall.train_pl --config birdcall.config_params.example_config` reaches the training loop
- [ ] an unknown `--config` raises a clear error
- [ ] a one-epoch run over the synthetic data matches the reference per-step loss to **1e-5**
- [ ] a crashed run **re-raises** rather than exiting 0 (AD-18)
- [ ] the `train_pl.py` entry point keeps the `--config` dotted-path contract (G14)

**Verification**
- [ ] `pytest -q tests/test_pl_end_to_end.py` → one-epoch parity to 1e-5
- [ ] `pytest -q tests/test_entrypoints.py` → both entry points resolve a config

**Dependencies:** T2.2, T2.3
**Files likely touched:** `src/birdcall/train_pl.py`, `tests/test_pl_end_to_end.py`, `tests/test_entrypoints.py`
**Estimated scope:** S (3 files)

---

## Checkpoint 2 — Lightning port

- [ ] the LR curve matches the golden fixture step for step
- [ ] one epoch over synthetic data matches the reference core to 1e-5
- [ ] exactly `n_saved` checkpoints exist; the monitor resolves
- [ ] a weights-only resume leaves the optimiser at its initial LR; a full resume restores the epoch counter
- [ ] `pylint src/birdcall` reports no error-level findings

---

## Phase 3 — TensorFlow/Keras port

> The Keras port is the **harder** one: it rewrites the numeric core, not just
> the control plane. Its four documented silent-failure traps are
> `Softmax(axis=-1)` in `AttBlock` (G6), the inverted loss argument order, the
> `reduction` default that divides by `B` instead of `B*S*O`, and the
> ImageNet `preprocess_input` that `keras.applications.DenseNet121` applies
> internally.

### Task T3.1: Front-end layer + log-mel equality

**Description.** `helpers/tf_frontend_logmel.py` replaces the vendored front-end
with `tf.signal`:

```
x       = tf.pad(inputs, [[0, 0], [512, 512]], mode='REFLECT')
frames  = tf.signal.frame(x, frame_length=1024, frame_step=320)
spec    = tf.abs(tf.signal.rfft(frames, fft_length=[1024])) ** 2
mel     = tf.matmul(spec, melW)
logmel  = 10.0 * tf.math.log10(tf.maximum(mel, 1e-10))
```

The reflect pad is **explicit** (G9), the power formula is used (G8), the
**librosa Slaney** mel matrix is reused (G7) — not `tf.signal`'s HTK matrix —
and the dB formula is written by hand because there is no
`tf.signal.magnitude_to_db`.

**Acceptance criteria**
- [ ] the TF front-end and the torch front-end produce the same log-mel on the same waveform to **1e-4** (one check covers the mel scale, the dB formula, the reflect pad and the power-vs-magnitude question at once)
- [ ] `frames_num == 3001`
- [ ] the mel matrix is bit-identical to the one the torch front-end uses
- [ ] the module imports no torch

**Verification**
- [ ] `pytest -q tests/parity/test_frontend_parity.py` → log-mel to 1e-4, frame count

**Dependencies:** T1.1
**Files likely touched:** `src/birdcall/helpers/tf_frontend_logmel.py`, `tests/parity/test_frontend_parity.py`
**Estimated scope:** S (2 files)

---

### Task T3.2: Model rewrite

**Description.** `models/tf_model_att_block.py` and `models/tf_model_panns_dense121_att.py`. The
Keras rewrite of `PANNsDense121Att`: the BatchNorm transpose dance disappears
(`BatchNormalization(axis=-1)`), but the **ordering** `bn0` → SpecAugment is
preserved (G10). `AttBlock` becomes a Keras layer whose softmax is over **time**
(`axis=-2`, G6) — written with a comment, because `axis=-1` is a silent wrong
answer. `pad_framewise_output` uses
`tf.concat([x, tf.repeat(x[:, -1:, :], n, 1)], axis=1)` — **not** a symmetric
pad. `interpolate` is `tf.repeat(x, ratio, axis=1)` (interleaved). The
1→3 channel widening is `tf.broadcast_to` (not `tf.tile`, which copies). The
DenseNet transfer **bypasses** Keras' ImageNet `preprocess_input`. The mixup
batch halving forces a `train_step` override.

**Acceptance criteria**
- [ ] the shape trace matches the reference: `(B,3001,64)` → `(B,1024,93)` → `(B,264)` clip + `(B,3001,264)` frame
- [ ] the Keras DenseNet time axis is **93** — measured and asserted, not assumed
- [ ] the tail pad is exactly **25** frames, and the last 25 frames equal frame 3000
- [ ] the attention sums to 1 over the **time** axis, and `clipwise_output` is a sum over time
- [ ] the ImageNet `preprocess_input` is **not** applied (asserted, e.g. the first conv's input statistics are the raw normalised log-mel)
- [ ] `SpecAugmentation` takes an explicit `training` flag and draws an **independent** band per batch item
- [ ] the mixup halving happens in a `train_step` override, and the label tensor is halved with it

**Verification**
- [ ] `pytest -q tests/test_tf_sed_models.py` → shape trace, attention axis, tail pad, preprocess bypass
- [ ] `pytest -q tests/parity/test_model_parity.py` → `clipwise_output` matches the reference to 1e-4

**Dependencies:** T3.1
**Files likely touched:** `src/birdcall/models/tf_model_att_block.py`, `src/birdcall/models/tf_model_panns_dense121_att.py`, `src/birdcall/models/tf_registry.py`, `tests/test_tf_model.py`, `tests/parity/test_model_parity.py`
**Estimated scope:** L (4 files)

---

### Task T3.3: `tf.data` pipeline + losses + metrics

**Description.** Three modules. `datasets/tf_sed_dataset.py`: the `tf.data`
pipeline over the shared numpy dataset core — the waveform transform runs on
the **whole file before slicing** (G11), the label is the 264-vector one-hot
broadcast (G12), the train/valid slicing modes are preserved, and the
imbalanced sampler is expressed declaratively. `losses/tf_loss_*.py` (one file per criterion): the
five criteria, reproducing the **inertness** (G1), with the argument order
handled explicitly (`call(y_true, y_pred)` — use keyword arguments and
override `compute_loss`) and the `reduction` set so it does not divide by `B`
instead of `B*S*O`. `metrics/tf_metric_*.py` (one file per metric): the three metrics, including a
**vectorised LRAP** (no Keras label-ranking metric exists) and the hand-written
background branch and dual-threshold AND; `keras.saving.register_keras_serializable`
plus `get_config`/`from_config` so checkpoints reload; `add_variable`, not
`add_weight`.

**Acceptance criteria**
- [ ] the TF pipeline produces the same waveforms and label tensors as the torch one on a fixed synthetic batch
- [ ] the train pipeline is stochastic; the valid pipeline is deterministic
- [ ] `SedScaledPosNegFocalLoss(gamma=0, alpha_0=1, alpha_1=1, secondary_factor=1)` equals `BinaryCrossentropy` to 1e-6
- [ ] a deliberately swapped loss call produces a **visibly different** loss (the inversion is caught)
- [ ] the LRAP, F1-clip and F1-frame values match the reference to 1e-6
- [ ] a background row with nothing firing scores 1.0 on the F1 metrics
- [ ] the metrics serialise and deserialise (a saved model reloads with its metrics)
- [ ] `steps_per_epoch` is computable — the cardinality is taken before any lossy operation

**Verification**
- [ ] `pytest -q tests/test_tf_sed_dataset.py` → pipeline parity
- [ ] `pytest -q tests/test_tf_sed_losses.py` → inertness + inversion catch
- [ ] `pytest -q tests/parity/test_metrics_parity.py` → the three metrics to 1e-6

**Dependencies:** T3.2, T1.3, T1.5, T1.6
**Files likely touched:** `src/birdcall/datasets/tf_dataset_sed.py`, `src/birdcall/losses/tf_loss_*.py` (5), `src/birdcall/metrics/tf_metric_*.py` (3), `src/birdcall/losses/tf_registry.py`, `src/birdcall/metrics/tf_registry.py`, `tests/test_tf_dataset.py`, `tests/test_tf_losses.py`, `tests/test_tf_metrics.py`, `tests/parity/test_metrics_parity.py`
**Estimated scope:** L (6 files — split losses and metrics if needed)

---

### Task T3.4: Training loop

**Description.** `engines/tf_keras_model.py` (a `keras.Model` with a
`train_step` override for the mixup halving and a `compute_loss` override for
the argument order), `optimizers/tf_optimizer_adamw.py` (AdamW with `amsgrad=True`
and the **batch-stepped** shared schedule — never `CosineDecay`, which steps
per epoch), `callbacks/tf_callback_log_every.py`, `tf_callback_slack.py`,
`tf_callback_inference_dump.py` and `tf_callback_keep_best_n.py` (Keras 3's
`ModelCheckpoint` has no `save_top_k`, so keep-best-N is hand-written), and
`loggers/tf_logger_wandb.py` and `tf_logger_print.py` (wandb + print with the
original's series names).
Keras does not restore the epoch counter — `fit(initial_epoch=N)` is manual,
and the original's resume is weights-only anyway.

**Acceptance criteria**
- [ ] the LR curve matches the golden fixture step for step (it steps per **batch**)
- [ ] exactly `n_saved` checkpoints exist, ranked by the tracked metric
- [ ] a weights-only resume loads the weights and leaves the optimiser at its initial state
- [ ] the metric series names match the original's exactly
- [ ] the every-N callback fires every N steps, not every step
- [ ] `fit()` re-raises on a crash (AD-18)

**Verification**
- [ ] `pytest -q tests/test_tf_training_loop.py` → LR curve, keep-best-N, resume, series names

**Dependencies:** T3.3, T0.3
**Files likely touched:** `src/birdcall/engines/tf_keras_model.py`, `src/birdcall/optimizers/tf_optimizer_adamw.py`, `src/birdcall/callbacks/tf_callback_log_every.py`, `tf_callback_slack.py`, `tf_callback_inference_dump.py`, `tf_callback_keep_best_n.py`, `src/birdcall/loggers/tf_logger_wandb.py`, `tf_logger_print.py`, `tests/test_tf_training_loop.py`
**Estimated scope:** L (5 files)

---

### Task T3.5: Cross-port verification

**Description.** The claim that decides whether the two ports are
interchangeable: the shared voting module, fed from each backend's score
artefacts, produces a **byte-identical** `submission.csv`. Plus the `train_tf.py`
entry point (argparse `--config` dotted path → `importlib` → `Parameters()` →
`model.fit(...)`).

**Acceptance criteria**
- [ ] the two backends produce a **byte-identical** `submission.csv` from the same synthetic score set
- [ ] a one-epoch run over the synthetic data matches the reference per-step loss to 1e-5
- [ ] `python -m birdcall.train_tf --config birdcall.config_params.example_config` reaches the training loop
- [ ] an unknown `--config` raises a clear error
- [ ] the `--config` dotted-path contract is identical in both entry points (G14)

**Verification**
- [ ] `pytest -q tests/parity/test_submission_parity.py` → byte-identical CSV
- [ ] `pytest -q tests/parity/test_end_to_end_parity.py` → one-epoch loss to 1e-5
- [ ] `pytest -q tests/test_entrypoints.py`

**Dependencies:** T3.4, T2.4
**Files likely touched:** `src/birdcall/train_tf.py`, `tests/parity/test_submission_parity.py`, `tests/parity/test_end_to_end_parity.py`
**Estimated scope:** S (3 files)

---

## Checkpoint 3 — Keras port

- [ ] log-mel matches the torch front-end to 1e-4
- [ ] the attention sums to 1 over **time**; `clipwise_output` is a sum over time
- [ ] the ImageNet `preprocess_input` bypass is asserted
- [ ] the three metrics match the reference to 1e-6
- [ ] the two backends produce a **byte-identical** `submission.csv`
- [ ] `pylint src/birdcall` reports no error-level findings

---

## Phase 4 — Operations and reproducibility

### Task T4.1: CLI entry points, README, environment files

**Description.** The operator-facing surface. `README.md` (currently empty)
gets: what the project is, the two backends, the environment setup, the
`--config` contract, the training/inference/vote commands, the file-naming
rule, the style contract, and an honest statement of what is **not** verifiable
without the competition data. `.env.example` gets the Neptune and Slack
variables. `setup_structure.sh` provisions the venv and installs the right
requirement file. `main.py` (currently empty) dispatches to the three entry
points.

**Acceptance criteria**
- [ ] `README.md` documents all three commands and the `--config` dotted-path form
- [ ] `.env.example` lists `NEPTUNE_PROJECT_NAME` and `SLACK_URL`
- [ ] `setup_structure.sh` creates the venv and installs `requirements*.txt` idempotently
- [ ] `main.py` dispatches to `train_pl`, `train_tf` and `run_vote`
- [ ] the README states the data-dependent gaps (label space, fold CSVs, real training) plainly

**Verification**
- [ ] `bash -n setup_structure.sh` → parses
- [ ] `python main.py --help` → lists the three subcommands
- [ ] a human read-through against the plan's Definition of Done

**Dependencies:** T2.4, T3.5
**Files likely touched:** `README.md`, `.env.example`, `setup_structure.sh`, `main.py`
**Estimated scope:** S (4 files)

---

### Task T4.2: Seeded fold-CSV generator + label-space generator

**Description.** The two reproducibility gaps the docs flag as worth more than
either framework port. `scripts/build_bird_codes.py` reads the competition's
`train.csv` and writes `config_params/bird_codes.json` — the 264 ebird codes,
their `Genus species_Common Name` labels, and the index order. The index order
is the label order everywhere, so it is written once and never regenerated
silently. `scripts/make_folds.py` writes the per-fold train/valid CSVs,
**stratified on the primary `ebird_code` only** (per `docs/modules/01`), from a
recorded seed, so the ensemble can be re-derived. The original's exact split
is unrecoverable; the generator produces a **new, reproducible** one and says so.

**Acceptance criteria**
- [ ] `build_bird_codes.py` writes 264 codes with contiguous indices and a recorded source hash
- [ ] `make_folds.py` is seeded: the same seed gives byte-identical CSVs
- [ ] every fold contains every species (the docs' one hard structural fact)
- [ ] the stratification key is the **primary** `ebird_code` only
- [ ] both scripts have a `--help` and never write outside their target directory

**Verification**
- [ ] `pytest -q tests/test_scripts.py` → seed determinism, species coverage, index contiguity
- [ ] run the generator twice with the same seed and `diff` the outputs

**Dependencies:** T0.2
**Files likely touched:** `src/birdcall/scripts/build_bird_codes.py`, `src/birdcall/scripts/make_folds.py`, `tests/test_scripts.py`
**Estimated scope:** S (3 files)

---

### Task T4.3: Full config sweep + whole-suite run + parity report

**Description.** The closing gate. A test that instantiates all 21 configs
plus the inference config and checks the 39 attributes and the variation table;
the whole `pytest` suite; `pylint` against the repository `.pylintrc`; and a
written parity report listing every row of
`docs/ports/feature-parity-matrix.md` A–J as ticked or explicitly deferred
with a reason, and every row of §K as implemented.

**Acceptance criteria**
- [ ] `pytest -q` is green
- [ ] `pylint src/birdcall tests` reports no error-level findings against the repository `.pylintrc`
- [ ] the parity report covers all 112 matrix rows with a status and a reason for every deferral
- [ ] the report is written to `docs/ports/port-status.md`

**Verification**
- [ ] `pytest -q && pylint src/birdcall tests`
- [ ] a human read of `docs/ports/port-status.md` against the plan's Definition of Done

**Dependencies:** T4.1, T4.2
**Files likely touched:** `docs/ports/port-status.md`, `tests/test_config_sweep.py`
**Estimated scope:** S (2 files)

---

## Checkpoint 4 — complete

- [ ] every row of `docs/ports/feature-parity-matrix.md` A–J is ticked or explicitly deferred with a reason
- [ ] every row of §K (the vote) is implemented
- [ ] all 21 configs construct and the port reads all 39 attributes
- [ ] `pytest` green on CPU
- [ ] `pylint` clean against the repository `.pylintrc`
- [ ] the 13-model vote produces a submission and **both** backends produce the same one
- [ ] the fold generator and its seed are committed, so the ensemble can be re-derived
