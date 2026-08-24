# RSNA Knee Abnormality Detection — Winning Solution Blueprint

> Metric: **macro-averaged ROC-AUC over 12 findings** (ranking metric, threshold-free).
> Test set: ~1300 studies. **Reports are NOT available at test time** — text is training-only signal.
> Series metadata (`Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`) IS available at test time.

---

## 0. The single insight that wins this competition

Only a small subset of studies has per-condition labels, but **every** training study has a
radiology report written by the radiologist who *knew the answers*. The reports are the labels —
in ~10 languages. Therefore:

```
Stage A  Text teacher:   reports ──► soft 12-dim probabilities for ALL train studies
Stage B  Image student:  volumes + series metadata ──► 12 probs, distilled from Stage A + gold labels
Stage C  Self-training:  student re-labels train set (noisy-student / mean-teacher loop)
```

At inference you ship only the **image student** (+ series metadata embeddings).
The text model is a *teacher*, never a test-time input.

Everything else is execution quality: leak-free CV, robust multilingual label extraction,
noise-tolerant losses, and a diverse ensemble blended on honest OOF.

---

## 1. Problem framing

| Aspect | Decision | Rationale |
|---|---|---|
| Unit of prediction | Study | Submission is per `StudyInstanceUID` |
| Inputs at inference | All series DICOMs + series metadata table | Reports withheld |
| Output | 12 independent sigmoid logits | Macro AUC = per-class ranking; no softmax coupling |
| Laterality | Keep native orientation; **never flip L/R** as augmentation without flipping Medial/Lateral targets too | Medial vs lateral compartments are target-defining |
| Metric nuance | Rare classes (Fracture, Baker's) dominate variance | Per-class recall@high-sensitivity matters more than accuracy |

## 2. Data preprocessing

1. **DICOM decode**: `pydicom` + `pylibjpeg`/`gdcm` plugins for JPEG-lossless/J2K transfer syntaxes.
   Decode once → cache `.npz` volumes keyed by `SeriesInstanceUID` (uint16, original HU-free MR intensities,
   plus stored `InstanceNumber` order). Kaggle kernels are I/O-bound on DICOM; caching turns days into hours.
2. **Slice ordering**: sort by `ImagePositionPatient · slice_normal` when tags exist in the 86-tag allowlist;
   fallback to `InstanceNumber`; final fallback SOP UID sort. Verify ordering visually for ≥20 series.
3. **Windowing/normalization**: per-series percentile normalization (p1–p99) → uint8; MR has no absolute scale,
   percentiles are scanner-invariant. Optionally keep raw int16 for learned windowing experiments.
4. **Resampling**: resize in-plane to 384–512 px; sample fixed number of slices per series
   (e.g., uniformly 32–48 slices, or all slices with stride) so batch shapes are static.
5. **Series routing**: group series by `(Anatomical_Plane, Fluid_Sensitive)`; log the distribution.
   Sagittal fluid-sensitive series carry most signal for ACL/meniscus/effusion — make sure your sampler sees them.

## 3. Weak supervision from multilingual reports (the core)

Three-tier cascade, each tier gated by measured precision on the gold-labeled subset:

### Tier 1 — Rule-based extraction (precision ≈ 0.95+, recall ≈ 40–60%)
- Sentence-split reports (language-aware: handle no-space scripts CJK via jieba/MeCab-lite or char-ngrams).
- Language ID with `fasttext lid.176` (or character-profile heuristic) to route lexicons.
- Per-finding lexicons in EN/ES/DE/FR/PT/IT/NL/PL/RU/TR/JA/ZH (see `src/knee/labeling/lexicons.py`):
  affirmative patterns ("full-thickness tear of the ACL", "Ruptura del LCA", "Vorderes Kreuzbandriss"…),
  finding-specific synonyms ("effusion", "joint fluid", "derrame", "Gelenkerguss", "joint effusion"…).
- **Negation & uncertainty scope**: cue list ("no", "without", "intact", "unremarkable", "sin", "ohne",
  "kein", "sans", "нет", "无"…) applied within a ±window around each match; negated mention ⇒ negative evidence,
  hedged mention ("?","possible","cannot exclude") ⇒ ignored.
- Aggregate per study: any affirmative mention ⇒ positive seed; mentions exist but none affirmative ⇒ negative seed;
  else unknown. Calibrate per-class thresholds on gold subset; emit confidence = precision measured on gold.

### Tier 2 — Neural text teacher (recall ≈ 85–95%, this is the workhorse)
- Backbone: `xlm-roberta-base` (or `microsoft/mdeberta-v3-base`) — multilingual by design, no translation needed.
- Train on **gold-labeled studies only**, input = report text, output = 12 BCE heads.
- Long reports: stride-sliding-window inference with max-pooling over windows.
- Predict soft probs for ALL unlabeled studies → these are distillation targets.
- Validate teacher itself against gold subset with held-out fold — its OOF macro-AUC upper-bounds what
  the image student can inherit from it.
- Optional boost: pseudo-label self-training of the teacher (round 2 trained on rules+round1 labels).

### Tier 3 — Zero-shot NLI fallback (only if gold subset too small)
- `mDeBERTa-v3-base-xnli-multilingual-nli-2mil` with entailment hypotheses
  ("There is a tear of the anterior cruciate ligament.") per language template. Lower precision;
  use only as a prior blended at low weight.

### Label fusion policy (`weak_labels.parquet`)
Priority: **gold hard label > Tier-1 rule seed (conf ≥ τ_class) > Tier-2 teacher prob × weight**.
Store per-class: `prob`, `source ∈ {gold, rule, teacher}`, and `weight` used in the loss.

## 4. Model architecture

Recommended primary: **2.5D per-series encoder + attention-MIL study aggregator** (proven winner on MRI Kaggle comps).

```
per series s:  stack K slices → (K,384,384) → treat K as channels (or 3 groups of K/3)
               timm backbone (tf_efficientnetv2_s / convnext_small.fb_in22k / maxvit_tiny_224)
               → GAP → 1280-d series feature
metadata:      [fluid_sensitive, fat_suppression, plane_onehot(3)] → MLP → 32-d embedding (added)
study level:   transformer/attention pooling over series features (masked MIL)
head:          linear → 12 sigmoid logits
```

Why 2.5D first: pretrained ImageNet weights transfer far better than from-scratch 3D; slices within a series are
highly redundant, so adjacent-slice stacking captures context cheaply; trains in hours on a P100/T4x2.

Secondary diversity models for the ensemble:
1. **True 3D CNN** (MONAI `SeresNet50`/`resnet18_3d` or torchvision video `r2plus1d_18`) on resampled 1.5–2 mm³ volumes.
2. **Video transformer** (`vivit`-style or `x3d`) if compute allows.
3. **Per-plane specialists**: separate sagittal/coronal encoders fused late (plane metadata makes routing trivial).
4. **MaxViT/CoAtNet 2.5D** — attention hybrid, strong but heavier.

Fusion principle (SRP): encoders, aggregators, and heads are separate swappable classes wired by config
(`target:` class path + `params:`), never hard-coded.

## 5. Training recipe

- Loss: **Asymmetric loss** (ASL, γ+ = 0.2, γ− = 2) or Focal; per-class pos-weight from label stats;
  optional label smoothing 0.05 on gold-only samples. Distillation term: `BCE(student, teacher_probs)` with
  temperature T=2, weight ramped 0→λ over epochs (λ≈0.5); gold samples use hard labels with weight 1.0,
  weak samples use `weight` column.
- **Engine: PyTorch Lightning 2.x.** Models are `LightningModule`s (`KneeStudyLitModule`,
  `TextTeacherLitModule`); the Trainer + callbacks (ModelCheckpoint, EarlyStopping, custom EMACallback)
  + logger are declared as `ComponentSpec`s in YAML and assembled by `engines/trainer_factory.py::build_trainer`.
  Benefits on Kaggle kernels: bf16-mixed precision in one flag, free multi-GPU DDP on T4×2,
  checkpointing with metric-monitored top-k, deterministic seeding hooks, and `trainer.predict`
  for the inference kernel.
- Optimizer: AdamW via config spec with differential LR (backbone ×0.25 vs head), warmup-cosine schedule,
  weight decay 1e-2, EMA of weights via callback (decay 0.999), evaluated/saved at validation time.
- Sampling: `WeightedRandomSampler` equalizing expected positive exposure per class per epoch
  (`datasets/sampler.py`); rare-class oversample factor capped at 8×.
- Augmentations (albumentations, 2.5D-consistent across the slice stack): RandomResizedCrop(0.7–1.0),
  affine rotate ≤12°, translate ≤5%, brightness/contrast/gamma jitter (simulates windowing), Gaussian noise/blur,
  grid distortion light, CoarseDropout on slices. **No horizontal flips.** Vertical/slice-axis flips OK.
- Epochs: 8–15 with EarlyStopping(monitor=`val_macro_auc`, mode=max, patience=3).

### 5.1 Optimization-enhancement suite (all config-declared, default-off)

Every item below is a swappable `ComponentSpec`/flag in YAML — enabling one is a config edit,
never an engine change (Open/Closed). Implemented in `optimizers/gradients.py`,
`optimizers/lookahead.py`, `losses/curriculum_weights.py`, `callbacks/{stability,adaptivity,swapping}.py`,
`augmentations/diff_augment.py`, integrated in `engines/study_lit_module.py`.

| # | Enhancement | Config surface | Default |
|---|---|---|---|
| 1 | Gradient clipping strategies | `train.grad_clip`: `NormClipping` / `PercentileClipping` (rolling q-th percentile of grad-norm history) / `AdaptiveClipping` (ceiling = multiplier × EMA of norms) | Trainer `gradient_clip_val=10` |
| 2 | Gradient noise injection | `train.grad_noise: eta>0` (decays as η/(1+t)^0.55, scaled per-tensor) | 0.0 |
| 3 | Curriculum controller | `train.curriculum`: `ConfidenceRampCurriculum` — hold on gold/high-confidence seeds for N epochs, then linearly admit soft-label samples by lowering the sample-weight floor | null |
| 4 | Dynamic loss weighting | `loss.dynamic_weights: true` → DWA balances supervised vs distillation terms from their epoch-mean descent rates; weights logged as `loss_weight_*` | false |
| 5 | Layer freeze/unfreeze scheduler | callback `callbacks.stability.ProgressiveUnfreezingCallback` with declarative rules `[{pattern: encoder., until_epoch: 2}]` | not wired |
| 6 | Parameter-specific optimizers | `optimizer.backbone_optimizer:` spec → manual optimization mode (separate encoder optimizer; PL accumulation handled via `trainer.should_accumulate`) | single AdamW |
| 7 | Model component swapping | callback `callbacks.swapping.ComponentSwapCallback` with `[{epoch, attribute, spec}]`; new module inherits old weights when shapes allow (`strict=False`) | not wired |
| 8 | Differentiable augmentation | `augment.diff_ops: [color, translation, cutout]` — pure-torch GPU ops applied inside `training_step` (translation is grid-sample = differentiable; never flips L/R) | [] |
| 9 | Adaptive batch size | callback `callbacks.adaptivity.AdaptiveBatchSizeCallback` doubles batch size while peak CUDA memory < target fraction; requires `KneeDataModule.set_batch_size` | not wired |
| 10 | AMP fallback | `train.amp_fallback: true` → `AmpInstabilityWatcher` trips after N non-finite losses; `fit_with_amp_fallback` restarts fit in fp32 resuming from newest checkpoint | false |
| 11 | Online HP tuning | callback `callbacks.adaptivity.OnlineHPTuner`: LR ×decay after val plateau (max reductions), optional weight-decay growth | not wired |
| 12 | Lookahead wrapper | `optimizer.lookahead:` spec wraps every built optimizer; param-group dicts shared so schedulers still work | null |
| 13 | Smart accumulation scheduler | callback `callbacks.adaptivity.SmartAccumulationCallback` halves/doubles `accumulate_grad_batches` inside [min,max] from memory-pressure bands | not wired |

### 5.2 Checkpoint resume & W&B artifact lifecycle

- `engines/trainer_factory.find_latest_checkpoint(dir, fold)` prefers Lightning's `last.ckpt`
  (exact trainer state incl. optimizer/scheduler/EMA shadow) else newest-mtime `fold{N}-*.ckpt`.
- `train.resume: true` (+ `--resume` on `scripts/train_image_student.py`) passes it as
  `trainer.fit(ckpt_path=...)`.
- W&B integration: deterministic per-fold run id `deterministic_run_id(run_name, fold)` +
  `resume='allow'` ⇒ re-running continues the same W&B run; `log_model='all'` mirrors every saved
  checkpoint into W&B `model` artifacts, so a crashed kernel resumes from cloud state when no local
  checkpoint exists. Logger silently disables without `WANDB_API_KEY` unless `WANDB_MODE=offline`.

## 6. Cross-validation strategy

- Rows are studies ⇒ plain multi-label **iterative stratification** (`iterative-stratification`, 5 folds,
  seeds fixed) preserving all 12 label marginals. If DICOM allowlist exposes PatientID, switch to
  grouped stratification to prevent cross-study patient leakage.
- **One frozen fold assignment reused across every experiment** (`train_folds.csv` committed) — mandatory
  for valid ensembling and comparison.
- Two validation lenses:
  1. **Gold-OOF** (only gold-labeled studies): unbiased proxy of LB. *This is the metric that decides experiments.*
  2. Weak-OOF (all studies vs fused labels): noisy, inflated — use only for stability monitoring.
- Never tune on public LB; expect prevalence shift between train/LB/final (stated by organizers) —
  AUC is rank-based so avoid probability calibration tricks that depend on prevalence; prefer pure rankers.

## 7. Class imbalance handling (summary)

| Lever | Setting |
|---|---|
| Loss | ASL/Focal + per-class pos_weight |
| Sampler | Balanced multilabel weighted sampling |
| Thresholds | Irrelevant for AUC — skip threshold tuning |
| Data | Oversample Fracture/Baker's-positive studies ≤8×; mixup on images (β=0.2, labels mixed) optional |

## 8. Ensembling strategy

1. Within-model: fold checkpoints averaged (logit-mean), EMA weights, 2–3 seeds.
2. Across-model: greedy hill-climb on gold-OOF — start with best model's OOF, add candidate ensemble members
   (weighted prob-average) keeping additions that improve gold-OOF macro AUC.
3. Rank-average (scipy rankdata → mean → min-max) as alternative blend; pick whichever wins gold-OOF.
4. TTA at inference: vertical flip, slice-axis reversal, two crop scales (mean over views). No LR flip.
5. Budget: inference of 1300 studies × ~4 series × 32 slices must fit kernel limits — precomputed npz cache +
   fp16 + batch 32 keeps full ensemble <2 h.

## 9. Kaggle execution plan

| # | Kernel (script) | Input | Output artifact |
|---|---|---|---|
| 1 | `scripts/prepare_volumes.py` | DICOMs | `volumes_cache/` dataset (~compressed npz) |
| 2 | `scripts/make_folds.py` | train.csv | `train_folds.csv` |
| 3 | `scripts/train_text_teacher.py` + `build_weak_labels.py` | reports + folds | `weak_labels.parquet` dataset |
| 4 | `scripts/train_image_student.py` | cache + folds + weak labels | fold ckpts + OOF |
| 5 | `scripts/self_train.py` | ckpts + cache | relabeled parquet → re-run 4 (2 rounds) |
| 6 | `scripts/infer.py` | ckpts + test cache | `submission.csv` |
| 7 | `scripts/blend_submissions.py` | submissions | final `submission.csv` |

Push every artifact as a versioned Kaggle Dataset so downstream kernels mount it read-only.
Secrets (W&B key, etc.) come from `.env` locally; on Kaggle use Add-ons→Secrets (fallback logic in
`src/knee/utils/env.py`).

## 10. Risks & pitfalls checklist

- [ ] Don't leak report content into features at test time (architecturally impossible here — enforce by tests).
- [ ] LR flip silently destroys Medial-vs-Lateral OA discrimination.
- [ ] J2K decoding speedups: decode parallel (multiprocessing, 4 workers), cache aggressively.
- [ ] Some series are localizers with <10 slices — downweight tiny series in aggregator (mask + length embedding).
- [ ] Rare classes may be absent in some folds' validation split → guard NaN in fold AUC, rely on 5-fold pooled OOF.
- [ ] Teacher trained on gold subset will memorize if reports duplicate phrasing — always hold out gold folds for teacher eval too.
- [ ] Final-round self-training can drift: cap at 2 rounds, keep round-0 model in ensemble as anchor.

## 12. Repository taxonomy (14 packages)

```
src/knee/
  activations/      activation factory (relu/gelu/silu/mish)
  augmentations/    2.5D stack transforms + GPU differentiable ops
  callbacks/        EMA, Discord, unfreezing, AMP watcher,
                    batch/accum tuners, online HP, component swaps
  config_params/    pydantic schemas + loader/instantiate factory
  datamodules/      fold-scoped LightningDataModule
  datasets/         DICOM IO, volume cache, study datasets,
                    balanced sampler, frozen folds
  engines/          LightningModules, trainer factory, text teacher,
                    rule/NLI labelers, weak-label builder, predictor,
                    greedy blender
  helpers/          secrets (.env/Kaggle), seeding, logging
  layers/           MIL/transformer aggregators, metadata encoder
  loggers/          Discord notifier transport
  losses/           ASL/focal/soft-BCE + curriculum/DWA weighting
  metrics/          macro ROC-AUC with NaN-class accounting
  models/           timm 2.5D + MONAI 3D encoders, composite model
  optimizers/       param groups, warmup-cosine, Lookahead,
                    clipping strategies, gradient noise
```

## 13. Code standards enforcement

- `.pylintrc` (Google Python style) is the lint authority: max line 80,
  2-space indent, module/class docstrings required (functions <12 lines
  exempt), Google naming regexes. Ruff mirrors formatting decisions
  (`line-length=80`, `indent-width=2`, `quote-style='single'`) so both
  tools agree; justified lazy imports / broad excepts carry inline
  pragmas with comments.
- Every Python file begins with the shebang + PEP 263 coding declaration.
- Tests: dependency-light smoke suite (`tests/`) runs without torch;
  torch-gated enhancement tests skip gracefully.

## 11. Expected score trajectory (macro AUC)

| Milestone | Estimate |
|---|---|
| Gold-only baseline (EffNetV2-S 2.5D) | 0.80–0.84 |
| + text-teacher distillation on all studies | 0.86–0.89 |
| + self-training round + 3D/plane-specialist ensemble | 0.88–0.91 |
| + tuned greedy blend | 0.89–0.92+ |

Gold-only ceiling is bounded by small labeled set — the distillation pipeline above is what separates medal range from baseline.
