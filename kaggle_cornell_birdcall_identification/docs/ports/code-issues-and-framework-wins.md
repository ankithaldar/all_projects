# Major code issues, and what Lightning / Keras would (and would not) fix

An evidence-backed review of the Cornell Birdcall solution, ranked by impact,
with a verdict for each issue on whether a framework change actually helps.

Every claim below is verified against the working tree. File:line references
point at `src/` unless stated otherwise.

> **Status.** The issues below are all still open. One adjacent defect has
> been repaired: the **32 broken relative links** in the pre-existing
> `docs/` tree (a `modules/` prefix doubling up inside `docs/modules/`, plus
> four links from `docs/README.md` pointing at files that live in
> `docs/modules/`). All 32 now resolve; the fix changed only link targets,
> no prose.

**The headline finding:** the framework is the *second*-order problem. The
first-order problems are that the winning mechanism is not in the repo, that
the cross-validation cannot be reproduced, and that the audio front-end
recomputes the same log-mel spectra thousands of times. Lightning and Keras
fix none of those three. A framework swap is worth roughly 180 lines of
boilerplate; the three items above are worth the entire competition.

---

## Tier 1 — The solution as committed cannot reproduce its own result

### 1. The winning mechanism does not exist in the repo

The competition was won by a **13-model vote**, not by a better model:
4 of 13 models firing, with `clipwise_output ≥ 0.3` **and**
`framewise_output ≥ 0.3` per 5-second row (`README.md:45`). The jump from
7th to 1st on the Private Leaderboard came from that ensemble and those
thresholds.

There is no vote code, no 5-second row slicing, no site 1/2/3 handling, no
`nocall` fallback and no `submission.csv` emission anywhere in the tree. A
repo-wide search for `submission`, `row_id`, `site`, `to_csv` and `writerow`
returns nothing. The only surviving trace is a hard-coded `0.3` in
`src/metrics/sed_f1score_frame.py:35,40`.

**Why it matters:** the repository contains the *training* of 13 models and
none of the *use* of them. `src/metrics/sed_test.py` max-pools the entire time
axis (`pred_frames.max(axis=1)[0].max(axis=1)[0]`, line 50), so a 10-minute
file yields **one** record, not 120.

**Fix (framework-independent):** write the vote. It operates on saved score
files, so it belongs in a framework-agnostic module that both ports can import.
See [feature-parity-matrix.md](feature-parity-matrix.md) §K1–K6.

**Lightning:** no opinion. **Keras:** no opinion. Neither is the bottleneck.

---

### 2. Cross-validation is not reproducible from this repo

The 13 models are 4-fold no-mixup + 5-fold no-mixup + 4-fold mixup, each fold
a separate config file pointing at `/folds/fold_{n}_train.csv` and
`/folds/fold_{n}_test.csv`. **The fold CSVs are not in the repo**, and they
were produced offline. The data layout is also absolute: `/data/`,
`/pinknoise`, `/background/data_ssw`, `/saved_models/`.

The docs reach the same conclusion independently
(`docs/modules/09-config-and-cross-validation.md`).

**Why it matters:** you cannot re-derive the ensemble, verify the threshold, or
retrain a member and expect the vote to still work. The 13-model ensemble is
the deliverable, and its composition procedure is lost.

**Fix (framework-independent):** ship the fold CSVs, or ship the script that
generates them, with the RNG seed and the grouping key recorded. This is worth
more than either framework port.

**Lightning / Keras:** neither helps. A `dataclasses` config class (Keras) or
a registry (Lightning) is cosmetic next to missing data.

---

### 3. The inference path is broken in four independent ways

`src/metrics/sed_test.py` is the **only file in the entire tree that does not
compile** (verified with `compileall`): a missing comma after
`"filename": filename` at line 59. On top of that:

| # | Blocker | Where |
|---|---|---|
| 1 | Syntax error — missing comma | `sed_test.py:59` |
| 2 | `hparams.save_dir` and `hparams.combine_files` are **not defined in any config** (only `checkpoint_params["save_dir"]` exists) | `sed_engine.py:174` |
| 3 | `prepare_batch(mode='test')` references an **undefined local `x`** → `NameError` | `sed_engine.py:60` |
| 4 | `save_dir` must be a `pathlib.Path`; a `str` has no `.mkdir` | `sed_test.py:28` |

Plus `BaseEngine.evaluate()` — the only caller of the test path — is
**never invoked** by anything.

**Why it matters:** the ported pipeline would inherit four defects that the
original never exercised, because the branch is unreachable in every shipped
config (`metrics` is `["lraps","f1score_clip","f1score_frame"]` in all 21).

**Fix:** fix the four, and make the test path reachable.
**Lightning:** `Trainer.test()` + a callback. **Keras:** `model.evaluate()` +
a callback. Both are natural; the fixes are the work.

---

### 4. There are no tests

Zero test files in the repository. No CI. Nothing asserts that the loss is
what it claims to be, that the metrics are what they claim to be, or that the
model's shape trace is stable.

**Why it matters:** every defect in Tiers 2 and 3 below is a defect that a
twenty-line test would have caught.

**Fix:** a handful of characterisation tests before any port. The two that
matter most:

- `SedScaledPosNegFocalLoss(gamma=0, alpha_0=1, alpha_1=1, secondary_factor=1)`
  must equal `nn.BCELoss()` — this is the "focal loss is inert" claim, and it
  should be a test, not a comment.
- the ported LR schedule must equal Ignite's, step for step.

**Lightning / Keras:** neither supplies tests for you.

---

## Tier 2 — Silent numerical and metric defects

### 5. The focal loss is inert

`SedScaledPosNegFocalLoss` is the criterion in **19 of the 21** shipped
configs, with `gamma=0.0`, `alpha_0=alpha_1=1.0`, `secondary_factor=1.0`.
With `gamma=0`, `(1-pt)^0 = 1`; with both alphas 1 the positive/negative
split is a no-op; with `secondary_factor=1.0` the secondary mask is 1. The
"focal" loss is therefore **bit-identical to plain mean BCE**
(`src/loss/sed_scaled_pos_neg_focal_loss.py:37-46`).

The same collapse hits `FocalLossStandard` (`gamma=0` default) and
`FocalControlledStandardLoss`.

**Why it matters:** not because it is broken — the result is still real and the
model still trained — but because the code *claims* to be focal and is not.
Anyone reading the config sees `criterion_name="sed_scaled_pos_neg_focal_loss"`
and infers a focal objective that was never applied. It also means the
hyper-parameter surface (`gamma`, `alpha_0`, `alpha_1`, `secondary_factor`)
looks like it was tuned when it was inert.

**Fix:** rename it, or set `gamma` to a real value and **re-tune** (which
invalidates the ensemble — do not do that to a solution that won). The honest
move is to rename and document.

**Lightning / Keras:** neither changes the maths. A Keras port must reproduce
the collapse *exactly* if the port is to reproduce the result — which means
porting `(1-pt)^gamma` and then noting that `gamma=0` makes it 1.

---

### 6. Background audio gets a free perfect LRAP

`SedLRAPS` has no background guard: for a row with no positive label,
`label_ranking_average_precision_score` scores the ranking of an all-zero
target as **1.0** (`src/metrics/sed_lraps.py:22-27`).

The validation set deliberately injects label-less no-bird audio
(`background_audio_dir`, `sed_dataset.py:34-43`) to punish false positives.
Those rows are exactly the ones the LRAP metric scores as perfect.

`SedF1ScoreClip` and `SedF1ScoreFrame` *do* have the background branch and
score those rows 0.0 when something fires. So the three metrics disagree about
the same rows.

**Why it matters:** `lraps` is inflated by the number of background rows
present. It is not the tracked metric (`track_metric="f1score_clip"`), so the
checkpoint ranking is unaffected — but the reported LRAP is not a clean
quality number, and anyone comparing runs across folds is comparing
differently-composed sets.

**Fix:** add the same background branch to `SedLRAPS` that the F1 metrics
have, and report both numbers.

**Lightning / Keras:** neither fixes it — and in Keras it is *harder*, because
there is no label-ranking metric at all and the whole metric has to be
reimplemented (see the ledger).

---

### 7. The DDP metric reduction is mismatched

`src/metrics/sed_test.py:74` decorates `compute()` with
`@sync_all_reduce("_score", "_num_items")`, but the class has no `_score`
attribute — it has `_score_frame` and `_score_clip`. Ignite's
`sync_all_reduce` resolves missing attributes with a `getattr(..., None)`
default, so the denominator is globally reduced while the numerator is not.

**Why it matters:** under DDP the metric is silently wrong — a ratio of a
local sum to a global count. Moot today (`dist_params={}` in every config),
but it is a trap for the first multi-GPU run, which is also the run where you
will not be looking.

**Fix:** rename the attributes to match the decorator, or reduce both
numerators.

**Lightning / Keras:** neither has an equivalent hazard — but Keras has no
built-in metric that matches, so the whole thing must be rewritten anyway.

---

### 8. The `_augd` criterion silently drops 70 % of the primary species

`SedScaledPosNegFocalLossAugd` computes the scaled loss, then **randomly
zeroes the primary-species positions with probability
`primary_drop_out_factor=0.3`** — i.e. 70 % of primary positions contribute
nothing (`src/loss/sed_scaled_pos_neg_focal_loss_augd.py`). It is used by 2 of
21 configs (`nomix_fold2_augd`, `nomix_fold3_augd`).

**Why it matters:** it looks like a bug and is not — but it is a strong
regulariser that nobody would guess from the name, and it uses a global RNG
that is not seeded per-epoch, so it is not reproducible run to run.

**Fix:** document it loudly, and seed it.

---

### 9. The frame head is unsupervised

Only `clipwise_output` ever reaches a loss
(`src/engine/sed_engine.py:72`). `framewise_output` inherits all of its signal
through the `AttBlock`'s attention weights and is never directly optimised —
yet it is half of the winning decision rule (the 0.3 frame gate).

**Why it matters:** this is by design, and the repo's own docs explain it
well (`docs/modules/10`). But it means the frame gate's threshold was never
tuned against a supervised objective — it was tuned by leaderboard feedback
under a **2-submissions-per-day** budget. That is the real reason the recipe
looks arbitrary.

**Fix:** none needed. **But** if you ever port this, keep the two heads
independent — do not add a frame loss, and do not merge the thresholds.

---

## Tier 3 — Robustness and dead code

### 10. `eval()` on CSV content

`df.secondary_labels = df.secondary_labels.apply(eval)`
(`src/dataloaders/sed_dataset.py:28`) executes arbitrary Python from a CSV
cell. A NaN cell raises `TypeError`. A malicious or malformed CSV is a code
execution path.

**Fix:** `ast.literal_eval` — same result, no execution. One line.

---

### 11. Config attributes that are referenced but never defined

Verified by searching all 22 files in `src/config_params/`: **none** of these
exist, yet the code reads them:

| Attribute | Read at | Consequence |
|---|---|---|
| `hparams.smooth` | `main_engine.py` (smooth_bce) | `AttributeError` |
| `hparams.num_classes` | `main_engine.py` (dcase) | `AttributeError` |
| `hparams.save_dir`, `hparams.combine_files` | `sed_engine.py:174` (SedTest) | `AttributeError` — the whole test path is unreachable |
| `hparams.gamma`, `alpha`, `primary_loss_factor`, `beta`, `q`, `loss_factors`, `betas` | `main_engine_v4.py` | `AttributeError` |

**Why it matters:** every one of these is a *silent* configuration contract
violation. The config class is a plain object with no validation, so nothing
fails at construction — the failure surfaces minutes into a run, or never.

**Fix:** a dataclass with required fields, or a `__post_init__` that asserts
the attribute set. **This is exactly the kind of thing a dataclass buys** —
and it is a *framework-independent* fix that both ports should adopt.

---

### 12. The `_init_*` dispatch has no `else` and no key validation

`SedEngine._init_model` / `_init_criterion_function` / `_init_scheduler` /
`_init_optimizer` are long if/elif chains on config strings, with **no `else`
branch** (only `_init_logger` has one). An unknown `model_name` leaves
`self.model` unset and fails later as
`AttributeError: 'SedEngine' object has no attribute 'model'` — from a line
that has nothing to do with the typo.

**Fix (framework-independent):** registry dicts. **Lightning:** the port
should use them (the repo's own analysis recommends it). **Keras:** the same.
The framework is irrelevant — this is a plain-Python improvement.

---

### 13. Modules that are dead, or dead on arrival

| Module | Status |
|---|---|
| `src/loss/sed_scaled_focal_loss.py` | Imported by **nothing** (verified). Its class name `SedRemovedFocalLoss` also **shadows** the real one in `sed_removed_focal_loss.py` — so a careless import gets the wrong class. |
| `src/augmentations/spec_augment.py` | **ImportError at import time** — `from helpers.audio_utils import mono_to_color`, and `mono_to_color` does not exist in that file. |
| `src/dataloaders/audio_dataset.py` | Same missing `mono_to_color`, at line 112. |
| `src/loss/noisy_loss.py` | `loss_keys` declares **4** keys; `forward` returns **2** → `KeyError` in `main_engine_v4._init_metrics`. |
| `src/loss/custom_losses_for_targets.py` | The `noisy_type == "lq"` branch is `pass` → `self.noisy_loss` is never set → `AttributeError`. |
| `src/engine/main_engine*.py`, `train_v2/3/4.py`, `audio_dataset_v2/3/4.py`, `classifier_*.py`, `aux_*.py` | The legacy research line, unreachable from any shipped config. |

**Why it matters:** `src/` presents 85 files; roughly 30 of them are dead,
broken, or both. A reader cannot tell which is which without reading the
imports of every file.

**Fix (framework-independent):** delete the dead line, or move it to
`src/legacy/` with a README saying it does not run. Either is fine; leaving
it interleaved with the live path is not.

**Lightning / Keras:** neither helps — and a port that faithfully copies dead
code faithfully copies the bugs.

---

### 14. AMP is dead code with a contradictory log message

`src/engine/base/base_engine.py:6-13`:

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

The first line claims AMP is being used; the second says it is not. Both
branches set `USE_AMP = False`, so the log lies on every path.

The root cause is real and worth keeping: the in-graph log-mel computes
`10·log10(clamp(x, 1e-10))`, and under fp16 `1e-10` underflows to `0`, so
`log10(0) = −inf` → NaN. The `amin` fix is described but, per
`docs/modules/02-audio-frontend.md`, **untested**.

Also: `os.getenv("XLA_USE_BF16")` with no default, swallowed by a bare
`except` (`base_engine.py:27-30`), so `XLA_USE_BF16` is `0` — an `int` where a
`str` was intended, compared against `1` in `clip_gradients`.

**Fix:** delete the branch, or fix it and **re-validate** the whole recipe.
AMP halves the memory of a 30 s × 32-batch SED model, so it is worth a
deliberate attempt — but only with the `power_to_db` op forced to fp32, and
only after the NaN is actually gone.

**Lightning:** `precision="16-mixed"` is one kwarg, but it hits **the same
wall** — the fix is identical and framework-independent (force `power_to_db`
to fp32). Lightning does not make AMP work here. **Keras:** the same wall, and
`keras.mixed_precision` is arguably harder to opt out of per-layer.

---

### 15. CRLF line endings in 20 of 22 config files

Verified: 20 of the 22 `.py` files under `src/config_params/` are CRLF
(`configs.py` + 19 of 20 `final_sed*`), while `example_config.py` and
`final_sed/final_sed_dense121_mix_2.py` are LF.

**Why it matters:** line-anchored `grep` silently misses CRLF files. A
developer auditing the configs with `grep -n "criterion_name"` sees 2 of 22
files and concludes the rest differ, when they are identical modulo line
endings. This has already caused at least one documented confusion.

**Fix:** `.gitattributes` normalising `*.py text eol=lf`, or a one-off
normalisation commit. **Framework-independent.**

---

## Tier 4 — Performance

### 16. The log-mel front-end is recomputed on every single step

This is the single largest inefficiency in the repo, and the repo's own
framework analysis says so explicitly:

> "The most valuable thing in this document is not 'use TF instead' — it is the
> observation that the on-the-fly front-end is the single biggest inefficiency
> in the original, and that was true in PyTorch too."
> — `docs/framework-comparisons/tensorflow-keras-port.md`

The model takes a raw waveform and computes STFT → log-mel → SpecAugment
**inside `forward()`**, every step. With `period=30` s, `train_bs=32`, 50
epochs and 13 models, the same 30-second windows are re-analysed thousands of
times, and the STFT/mel buffers are frozen but the *arithmetic* is not cached.

**Fix (framework-independent):** compute log-mel in the data pipeline and
cache it. **But** this changes the `bn0` → `SpecAugment` ordering relative to
the normaliser and invalidates the tuned recipe — so it is a new experiment,
not a refactor. It also invalidates the waveform-domain augmentation, which
must then move to the mel domain.

**Lightning:** no effect. **Keras:** the TF port doc's own recommendation is
to compute log-mel in `tf.data` and cache it — *which is the right fix, and
it is not a Keras feature, it is a change of architecture*. Adopting it means
re-tuning a solution that already won.

---

### 17. The dataset re-reads and holds whole files in memory

`SedDataset.get_audio` reads the entire file with `sf.read`
(`sed_dataset.py:76`) and `__getitem__` slices it afterwards. A 10-minute
32 kHz WAV is ~38 MB as float32; the validation set loads them one at a time
per worker, but the memory profile is driven by whole-file reads, not by the
30 s window that is actually used.

**Also:** `num_test_samples=2` in the valid/test configs, while a 10-minute
file needs 20 snippets of 30 s to be fully covered — or 120 rows of 5 s. The
validation metric is computed on a truncated view of the data.

**Fix (framework-independent):** read only the window you need
(`sf.read(..., start=, stop=)`), and decide deliberately whether validation
should cover the whole file.

**Keras:** `tf.data` with `AudioDataset`/`tfio` would make this easy — one of
the few places where the framework genuinely helps the *plumbing* (though the
decision of what to cache is still yours).

---

## Tier 5 — Operational behaviour

### 18. A crashed run looks like a successful run

`BaseEngine.train` (`base_engine.py:216-223`):

```python
try:
    self.trainer.run(self.train_loader, **run_params)
except Exception as e:
    import traceback
    print(traceback.format_exc())
if self.rank == 0 and self.logger:
    self.logger._end_of_training()
```

The traceback is printed and swallowed; the Neptune experiment is then closed
normally; the process exits 0. A run that died in epoch 3 of 50 looks
identical to a finished run in the experiment tracker.

**Fix (framework-independent):** re-raise, or at least set a non-zero exit
code. **Lightning:** `Trainer.fit()` re-raises by default — a genuine
improvement, though it changes the operational contract. **Keras:** `fit()`
raises too.

---

### 19. Checkpoints are only written on validation days, and up to 9 epochs can be lost

The saver is attached to the **evaluator's** `COMPLETED` event
(`base_engine.py:128`), and validation runs every `eval_every=2` epochs. So
checkpoints are written twice per 50 epochs, and `n_saved=5` means 5
checkpoints per 10 epochs.

**Fix (framework-independent):** save every epoch. **Lightning:**
`ModelCheckpoint(save_last=True)` reduces crash loss from "up to 9 epochs" to
"1 epoch" — a concrete win. **Keras:** Keras 3's `ModelCheckpoint` has no
`save_top_k`, so keep-best-N needs a custom callback plus file management.

---

### 20. "Resume" is not a resume

`load_model_only = True` in **all 21** configs, so a resume restores only the
model weights. The RNG state is not checkpointed, so the augmentation and
mixup streams do not continue where they left off, and the epoch counter is
not restored either.

**Fix (framework-independent):** decide what "resume" means and implement that
one thing. **Lightning:** `fit(ckpt_path=…)` restores the full state, and is
strictly better — but note it also **restores `max_epochs` from the
checkpoint**, so raising `max_epochs` in the config on resume has no effect.

---

### 21. Mixup draws the same lambdas every run, every epoch, every rank

`Mixup.__init__(self, mixup_alpha, random_seed=1234)`
(`src/augmentations/mixup.py:28`) — the seed is a default argument, so
`Mixup(mixup_alpha=1.)` always uses 1234. The `Beta(1,1)` sequence is
therefore identical across runs, across epochs and across DDP ranks.

**Why it matters:** the mixup augmentation is far less random than it looks,
and in a multi-rank run every rank mixes the *same pairs* with the *same
weights*.

**Fix:** seed it per-epoch from the global RNG.

---

### 22. The imbalanced sampler is not sharded across ranks

`ImbalancedDatasetSampler` is a `torch.utils.data.Sampler`, not a
`DistributedSampler`. Under Ignite's `idist.auto_dataloader` every rank can
see the same samples.

**Fix (framework-independent):** shard the index space by rank.
**Lightning:** Lightning's DDP wrapper *shards* a custom sampler rather than
replacing it, so the inverse-frequency weighting survives — **verify this**,
it is the difference between a correct and an incorrect distributed run.

---

### 23. The real constraint was submissions, not compute

Two submissions per day, and a 13-model ensemble whose vote count and
thresholds were chosen by leaderboard feedback. The training loop ran 50
epochs on a T4 for each of 13 models, and the *only* thing that decided the
competition was a threshold.

**Why it matters:** it reframes the whole framework question. Faster training
would not have won this. A framework port is a code-quality project, not a
competition project.

---

## The ledger — what each framework actually buys

### PyTorch Lightning

**Genuinely fixes**

| Issue | How |
|---|---|
| 19 — checkpoints only on validation days | `ModelCheckpoint(save_last=True)` — crash loss from 9 epochs to 1 |
| 18 — a crashed run looks successful | `fit()` re-raises |
| 12 — no `else` in the dispatch | registry dicts, with a clear error |
| Gradient clipping (23 hand-rolled lines) | `gradient_clip_val` — the original's code is a copy of Lightning's own |
| LR finder (23 lines) | `Tuner.lr_find()` |
| 12 × `RunningAverage` objects | `self.log` / `self.log_dict` |
| 16 — `complete_clear` + `gc.collect()` | Lightning handles the references; keep an explicit `gc.collect()` as cheap insurance |
| DDP plumbing, progress bar, the `every=N` filter | mostly — see below |

**Does not fix** — items 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16,
17, 21, 22, 23. **None of the domain logic, none of the reproducibility
gaps, none of the metric defects, and not the front-end recomputation.**

**Makes worse**

- **The `every=N` event filter is gone.** `Trainer(log_every_n_steps=N)` is a
  global logging knob, not a per-handler filter. Needs a hand-rolled counter.
- **`handler.remove()` is gone.** The LR finder's detaching of the validation
  handler and the scheduler has no equivalent — run the sweep on a throwaway
  trainer.
- **No `engine.state`.** Six places read `engine.state` in the original; in
  Lightning the epoch, the step and the metrics live in three different
  places and can disagree.
- **Callback order is load-bearing.** Nine ordering-sensitive registrations
  collapse into a list whose order silently matters.

**Net:** a good trade for ~180 of the 375 lines of `base_engine.py`, and no
change to any number the solution produces.

### TensorFlow / Keras 3

**Genuinely fixes** (all plumbing, no domain logic)

| Issue | How |
|---|---|
| The vendored front-end (~250 lines of buffer maths) | `tf.signal` — ~6 lines. The single largest code reduction available |
| The DenseNet transfer | `keras.applications.DenseNet121(include_top=False)` |
| The BatchNorm transpose dance (twice) | `BatchNormalization(axis=-1)` — two transposes deleted |
| 12 × `RunningAverage` + `attach_metrics` | `compile(metrics=[...])` |
| The two-scheduler LR concat | `keras.optimizers.schedules` |
| The imbalanced sampler (25 lines) | `tf.data.Dataset.sample_from_datasets` — **strictly better** |
| Gradient clipping / accumulation / dropout / the sigmoid clamp | `clipnorm=`, `gradient_accumulation_steps=`, `Dropout(p)`, `BinaryCrossentropy` |

**Does not fix** — the same list as Lightning, plus two more that are Keras-
specific: **no label-ranking metric exists at all** (so `SedLRAPS` must be
reimplemented vectorised, ~20 lines, or paid for with `tf.py_function` per
item), and the **`(scalar, dict)` loss contract** has no Keras equivalent
(Keras losses return a scalar; the `loss_keys` dict needs a per-key metric
shim).

**Makes worse — and these are the ones that bite**

1. **`Softmax(axis=-1)` in `AttBlock` is a silent wrong answer.** The softmax
   must be over *time*; Keras' axes come from the right, so it is `axis=-2`.
   It trains. It scores badly. **Highest-probability silent failure in the
   whole port.**
2. **The in-graph log-mel front-end is an anti-pattern in Keras.** Nontrainable
   `add_weight` buffers, `tf.pad(REFLECT)`, HTK-vs-Slaney mel scale, no
   `tf.signal.magnitude_to_db`, and — the blocker — Keras' DenseNet applies
   ImageNet `preprocess_input` that torchvision's `.features` does not, which
   destroys the pretrained transfer **with no error**.
3. **The control plane maps worst here.** No priority, no removal, no
   `engine.state`, no uniform `every=N`. `base_engine.setup()` makes nine
   ordering-sensitive registrations; in Keras their order becomes a Python
   list index, and for a 13-config ensemble that is 13 chances to get it
   wrong silently.
4. **Checkpoint/resume is worse.** Keras 3 `ModelCheckpoint` has no
   `save_top_k`; the epoch counter is not restored; `load_weights` does not
   restore optimizer slots.
5. **The loss argument order inverts** (`call(y_true, y_pred)`), identically
   shaped, identically typed, no error.
6. **Keras' `reduction` default is `sum_over_batch_size`** — divides by `B`,
   not by `B·S·O`. A factor of ~792,264.
7. **`tf.pad(SYMMETRIC)` is not "repeat the last frame"** — `pad_framewise_output`
   needs `tf.concat([x, tf.repeat(x[:, -1:, :], n, 1)], 1)`.
8. **VRAM is preallocated** and `set_memory_growth` is process-global and must
   run first — fatal for a 13-model ensemble on a 16 GB T4.
9. **TFLite int8 quantisation invalidates the 0.3 vote threshold** — the one
   genuine TF advantage, and it is unusable without re-tuning the thing it
   would optimise.

**Net:** the existing analysis's verdict — "the portable core is the
signal-processing / loss / metric *logic*, and it ports almost cleanly. What
does not port is the *control plane*."

---

## Recommended order of work

**Before touching either framework**

1. Write the vote and the submission path (§1). Without them there is no
   deliverable to port.
2. Ship the fold CSVs or their generator (§2).
3. Add characterisation tests for the loss, the three metrics and the LR
   schedule (§4). These are the ground truth both ports are checked against.
4. Fix the four inference blockers (§3), the `eval()` (§10), the undefined
   `hparams` attributes (§11), the `NoisyLoss` key mismatch and the `lq`
   branch (§13).
5. Rename the inert focal loss, or set a real `gamma` and re-tune (§5).
6. Add the background branch to `SedLRAPS` and report both numbers (§6).

**Framework-independent wins, available in Ignite today**

7. Read only the window you need in `SedDataset` (§17).
8. Delete or quarantine the dead line (§13) and normalise the line endings (§15).
9. Seed the mixup (§21), shard the sampler under DDP (§22), re-raise on
   crash (§18), checkpoint every epoch (§19), and decide what "resume" means (§20).

**Then, if the port is still wanted**

10. **PyTorch Lightning first.** It reuses the entire numeric core — the model,
    the loss, the dataset, the sampler, the augmentations and the front-end are
    imported unchanged — so the parity risk is concentrated in one place: the
    LR schedule. That is a two-hour numerical check.
    Follow [pytorch-lightning-guide.md](pytorch-lightning-guide.md).
11. **Keras second, and only if the goal is a different deployment story**
    (TFLite, a Keras-native serving stack) rather than a better codebase. The
    code reduction is real; the numerical risk is in the front-end, the
    `AttBlock` axis and the frame padding, and the control plane is a
    regression. Follow [tensorflow-keras-guide.md](tensorflow-keras-guide.md).

**The honest summary:** the framework swap is worth about 180 lines of
boilerplate and a moderate reduction in numerical risk-to-benefit ratio. Steps
1–9 are worth the competition.

---

## Appendix — the debt map, from the knowledge graph

The issue ranking above is an assertion. The graph in `graphify-out/` makes it
measurable: `BaseEngine` is the most connected node in the repository
(**82 edges**, betweenness 0.200), and its neighbourhood touches **27
communities**.

### What the 82 edges actually are

| direction | relation | count |
|---|---|---|
| out | `method` | 31 |
| out | `inherits`, `calls` | 2 |
| in | `references` | 38 |
| in | `imports`, `inherits` | 5 + 5 |
| in | `contains` | 1 |

⚠️ **Read this before quoting the number.** 41 of the 82 edges are pure
Python structure — 31 `method` edges are the class *containing* its own
functions, and 10 are `imports`/`inherits`. **Exactly one edge is a `calls`
edge.** So "god node" here mostly means **"big class"**, not "deeply
entangled". The meaningful figure is the **38 inbound `references` from
documentation**, all but three of them `EXTRACTED`.

### Why it bridges 27 communities

Three compounding reasons, and the third is the interesting one:

1. **It is a God Object.** 31 methods, **11 of them `_init_*` dispatch
   hooks**. The class owns device and rank setup, sampler construction, the
   entire Ignite event graph, checkpoint save and load, metric attachment,
   gradient clipping, the LR finder, the three steps, the score function, and
   eleven component factories.
2. **Its own file is a community.** 20 of its 82 edges stay inside
   `base_engine.py` ("Base Engine and AMP Dead Code").
3. **The prose hangs from it.** 38 inbound `references` edges come from
   documentation spread across 27 communities — and the citation ranking is
   the tell:

| method | doc communities citing it |
|---|---|
| `.setup()` | **15** |
| `.setup_checkpoint_saver()` | 9 |
| `._init_distribution()` | 8 |
| `.train()` | 8 |
| `.load_trainer_from_checkpoint()` | 8 |
| `.train_step()` | 6 |
| `.clip_gradients()` | 5 |
| `.attach_metrics()` | 4 |
| `.lr_finder()` | 4 |
| `.eval_step()` | 4 |
| `._init_score_function()` | 3 |

`.setup()` alone is cited from **15 different communities** — it is where the
analysis of the whole repository is anchored.

### Which issues the centrality explains

Mapping the most-cited methods onto the issue list gives **9 of the 23**,
and every one of them is in the control plane:

| issue | lives in | cited by |
|---|---|---|
| #12 — no `else` in the `_init_*` dispatch | `_init_*` hooks | 11 communities |
| #19 — checkpoints only on validation days | `.setup_checkpoint_saver()` | 9 |
| #22 — the sampler is not sharded under DDP | `._init_distribution()` | 8 |
| #20 — "resume" is not a resume | `.load_trainer_from_checkpoint()` | 8 |
| #11 — `hparams` attributes read but never defined | `_init_*` hooks | 11 |
| #18 — a crashed run looks successful | `.train()` | 8 |
| #6 — background audio gets a free LRAP 1.0 | `.attach_metrics()` | 4 |
| #7 — the DDP metric reduction is mismatched | `.attach_metrics()` | 4 |
| #5 — the focal loss is inert | `.train_step()` → criterion dispatch | 6 |

**The graph localises the debt to specific methods.** That is the useful part:
these are not abstract smells, they are nine concrete lines in 375 lines of
`base_engine.py`.

### The part that matters for the port

Split the 27 communities by subject:

- **13 are port/framework-analysis communities** (32 edges) — "Lightning
  Training Step Port", "Keras Port Control Plane", "Lightning Checkpoint
  Callbacks", "Keras Fit and JIT", "Keras Backup and Restore", "Framework
  Comparison Matrix", …
- 14 are domain/code communities (50 edges).

**The port-side communities cluster around `BaseEngine` because the port *is*
a control-plane change, and `BaseEngine` is where the control plane lives.**
The graph independently rediscovers the conclusion the existing analysis
reaches by hand: roughly 180 of `base_engine.py`'s 375 lines disappear under
Lightning, and the remaining ~195 are SED-specific logic no framework can
express. The graph says *where* the 180 live: `setup()` plus the nine
event-graph methods around it.

There is a second-order effect worth naming. **The God Object's size is a
symptom of the event graph.** The `_init_*` dispatch has no `else` and no key
validation (#12) *because* there was nowhere to validate — every component
was constructed in one linear `setup()` sequence, so a typo could only fail
later. A framework port splits that sequence into a `LightningModule` and a
`Trainer(callbacks=[...])`, which is exactly where the missing validation
becomes visible and must be written down. **The port does not fix #12 — it
makes #12 impossible to hide.**

### And the part that does *not* follow

The four Tier-1 issues sit **outside** `BaseEngine`'s neighbourhood:

| Tier-1 concern | graph distance from `BaseEngine` |
|---|---|
| `SedTest` (the broken inference path) | 2 |
| "13-Model Ensemble (4 votes, 0.3 clip / 0.3 frame thresholds)" | 2 |
| "The Focal Loss Is Inert" | 2 |
| "Cross-Validation Is Not Reproducible from This Repo" | **3** |

`BaseEngine`'s **only** direct neighbour among the headline concerns is
`ImbalancedDatasetSampler` — which is issue #22, a control-plane issue.

So the graph says the same thing as the review, from the other direction: a
framework port reaches the control plane, where 9 issues live, and stops
short of the vote, the fold CSVs and the tests, where the remaining
competition-costing issues live. **The centrality measures where the port's
reach ends.**
