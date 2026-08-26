# RSNA Knee Abnormalities Detection — MVP Blueprint

Multimodal (MRI + reports) detection of 12 knee abnormalities per study.
Metric: macro-averaged ROC-AUC across the 12 targets.

## 1. Objective and Hard Constraints

Produce a valid `submission.csv` from an offline Kaggle notebook using
PyTorch + Lightning + timm, targeting 2x T4 GPUs.

| Resource | Limit | Design consequence |
|---|---|---|
| Raw data | ~570 GB | Never copied. Mounted read-only under `/kaggle/input/`. Stream-decode per batch; only headers are indexed once |
| RAM | 30 GB | Streaming datasets; no whole-series in-RAM buffers |
| Disk (writable) | 20 GB | Only checkpoints / OOF / submission (<2 GB); pushed to a Kaggle Dataset after each session |
| Kernel runtime | 12 h/session | Training split across sessions via checkpoint-resume protocol (Section 6) |
| GPU quota | 30 h T4/week | MVP sized at 7-12 GPU-h (Section 8); every compute knob lives in YAML |
| Submission runtime | <= 9 h (competition rule) | Inference target < 45 min |

## 2. Measured Facts (from `notebooks/01_EDA.ipynb`)

| Finding | Design impact |
|---|---|
| 4,407 studies / 24,371 series (~5.5 series/study) | Cap 4 series x K=24 slices -> ~423k slice-forwards/epoch (~20-25 min on 2xT4). Full 5-fold x 4-epoch MVP = 7-12 GPU-h |
| 0% missing reports | Rule-based pseudo-labels cover every training study |
| Median voxel anisotropy 9.92x | Pure-3D convolutions ruled out; 2.5D confirmed |
| Oblique IOP vectors exist; some series unsorted or missing IPP | Slice order precomputed in `index.parquet` via projection onto slice normal; `InstanceNumber` fallback. Filename sort is unsafe |
| `PatientSex` present ~94% (M/F/O), `PatientAge` absent | Sex one-hot enters metadata vector; age dropped |
| 1000 sampled studies map to 1000 unique patients | GroupKFold by StudyUID; PatientID captured during header scan to re-group if multi-study patients appear |
| "200/200 corrupted" audit result was an EDA logic bug (rate over unchecked files) | Corrected audit moved to notebook 02; decoder registry keeps zeros-frame fallback regardless |
| Domain-shift analysis crashed before producing numbers | `index.parquet` stores `ScanningSequence` / `MagneticFieldStrength` so folds may stratify by protocol proxy; DANN deferred |
| Label-vs-text contradiction rates not yet quantified | Fixed contradiction-rate table becomes part of notebook 02 QC; label smoothing strength calibrated from it |
| 73 high-frequency DICOM tags discovered | Header-only index captures all tags per series (first slice) plus per-slice minimal set (IPP, IOP, InstanceNumber) |

## 3. MVP Scope

| In MVP | Deferred (backlog Section 11) |
|---|---|
| Header-only DICOM index (geometry, spacing, ordering) | Physical isotropic resampling |
| On-the-fly decode (gdcm/pylibjpeg fallback registry) | Decoded-pixel cache datasets |
| Percentile normalization + autocrop inside Dataset | MONAI bias-field / ghosting augmentations |
| Rule-based English pseudo-labels (negation-aware) | XLM-R multilingual labeler, soft-label distillation |
| 2.5D sampling: K slices/series -> attention pooling | Per-plane encoders, CLIP-style image-report pretraining |
| Shared backbone + plane/sequence embedding + FiLM(metadata) | TTA, snapshot ensembles |
| Weighted BCE (YAML-swappable Asymmetric Focal) | ONNX Runtime export, INT8/TensorRT |
| StratifiedGroupKFold 5 folds, OOF macro-AUC | Fold-count reduction experiments |
| Session-resume training via Kaggle Dataset versioning | Efficiency-prize micro-tuning |

## 4. Architecture Overview

```
StudyInstanceUID
 └── Series 1..N (plane, Fluid_Sensitive, Fat_Suppression, Sex from index.parquet)
      ├── ordered slice list (precomputed; zero DICOM parsing at train time)
      └── sample K=24 evenly spaced slices -> worker decodes JPEG/JP2
           -> rescale -> percentile normalize -> autocrop -> resize img_size
           └── shared Backbone (tf_efficientnetv2_s) -> slice embeddings
                -> AttentionPool -> series token (+ plane/seq embedding)
                 ▼
     StudyAggregator (learnable query cross-attends series tokens)
     ⊕ FiLM(metadata MLP: plane ⊕ FS ⊕ FL ⊕ sex ⊕ log counts ⊕ spacing stats)
                 ▼
          trunk MLP -> 12 sigmoid heads
```

Reports exist only at training time: gold labels where present,
rule-derived pseudo-labels elsewhere (Section 7).

## 5. Configuration System (zero hardcoded configuration)

### 5.1 Rules

- Every hyperparameter, path, seed, size, and class selection lives in `configs/*.yaml`.
- Any object is specified as:

```yaml
class_path: package.module.ClassName   # fully qualified import path
init_params:                           # kwargs for __init__, resolved recursively
  key: value
```

- Lists of such specs are allowed (augmentation pipelines, metrics).

### 5.2 Files

```
configs/
├── config.yaml          # experiment name, seed, paths (mounts, ckpt dataset slug,
│                        #   kaggle secret names), device, folds to run
├── data.yaml            # n_slices, img_size, series-selection policy,
│                        #   normalization percentiles, decode backend order
├── folds.yaml           # CV class_path + init_params (StratifiedGroupKFold)
├── augmentations.yaml   # albumentations pipelines: train / valid
├── model.yaml           # KneeNet tree: backbone, pools, aggregator, heads
├── loss.yaml            # criterion class_path + init_params
├── optimizer.yaml       # optimizer/scheduler class_paths, lr groups, warmup_epochs
├── train.yaml           # Trainer args (epochs, precision, accumulation),
│                        #   session_time_budget_h, checkpoint_every_n_epochs, resume policy
├── datamodule.yaml      # DataModule/Dataset class_paths, batch_size, workers
└── infer.yaml           # ckpt paths, fp16, batch_size, tta: [], submission_path
```

### 5.3 Loader API (`src/knee/config_params/loader.py`)

```python
def load_config(path: str, overrides: Sequence[str] | None = None) -> dict:
    """Load YAML and apply dot-path overrides (e.g. 'model.init_params.dropout=0.2')."""

def instantiate(spec: Any) -> Any:
    """Recursively resolve class_path/init_params specs into live objects."""
```

Implementation uses `yaml.safe_load` + `importlib`; no third-party config framework.

## 6. Storage and Resume Protocol

Training cannot finish one 12 h session; kernels are ephemeral; GPU quota resets weekly.

Artifacts (all tiny; total < 2 GB):

| Artifact | Location | Content |
|---|---|---|
| `index.parquet`, `labels_pseudo.csv`, `folds.csv` | Kaggle Dataset `<slug>-index` | ordered SOP lists, spacing, plane/FS/FL, sex, protocol tags; supervision; splits |
| `fold{k}/last.ckpt`, `fold{k}/done`, `run_state.json` | Kaggle Dataset `<slug>-ckpt` (versioned) | model+optimizer+scheduler+epoch+step+best metric; fold completion markers |

Flow (`src/knee/helpers/kaggle_io.py`):

1. Credentials from Kaggle `UserSecretsClient`; secret names read from `config.paths.*`.
2. `pull_latest(slug, dest)` downloads newest version if it exists.
3. Engine start: per fold — `done` marker -> skip; `last.ckpt` -> resume via `Trainer.fit(ckpt_path=...)`.
4. `TimeBudgetCallback` stops fit when wall clock exceeds `session_time_budget_h - time_margin_min`;
   checkpoints also written every `checkpoint_every_n_epochs`.
5. Session end: `push_version(slug, folder)` creates a new immutable dataset version (rollback history).
6. Push retries x3 with backoff; on failure the ckpt remains in `/kaggle/working` with logged manual recovery path.
7. Inference consumes whichever folds have `done` markers (partial ensembles degrade gracefully).

## 7. NLP Pseudo-Labeling (MVP rules)

- Per-target lexicon regexes on lowercased report text.
- Negation triggers (`no|without|intact|unremarkable|absent`) within a +/-6-token window flip to negative.
- Uncertain cues (`rule out|cannot exclude|question`) emit `-1` mask (excluded from that target's loss).
- Output `labels_pseudo.csv`: hard labels + `source in {gold, rules}`.
- QC in notebook 02: agreement kappa vs gold subset per class, prevalence sanity table,
  language census over all reports (sizes the future XLM-R backlog item),
  fixed contradiction-rate table (calibrates label smoothing).

## 8. Compute Budget (fits 30 GPU-h/week)

```
per epoch : 4407 studies x 4 series x 24 slices ~= 423k slices @384 px
            ~= 300-350 img/s (2xT4, bf16-mixed)  ~= 20-25 min (+decode overlap)
MVP total : 5 folds x 4 epochs                  ~= 7-12 GPU-h
inference : ~1300 studies                       ~= < 15 min
headroom  : >= 15 GPU-h for iteration/backlog
```

Levers if measurements disagree: `n_slices`, `img_size`, `max_series_per_study`,
backbone name — all in YAML. Notebook 01 records measured decode throughput.

## 9. Repository Layout

```
main.py                              # CLI: build-index | build-labels | build-folds | train | infer
configs/                             # Section 5.2
src/knee/
├── config_params/loader.py          # load_config, instantiate
├── helpers/
│   ├── dicom_io.py                  # DecoderRegistry: native->gdcm->pylibjpeg, zeros fallback
│   ├── geometry.py                  # normal-projection ordering; InstanceNumber fallback
│   ├── intensity.py                 # rescale, percentile normalize, autocrop
│   ├── nlp_labeling.py              # lexicons + negation window + uncertain mask
│   ├── folds.py                     # StratifiedGroupKFold builder -> folds.csv
│   ├── kaggle_io.py                 # dataset push/pull via kaggle CLI + UserSecrets
│   └── utils.py                     # seeding, timer, logger
├── datasets/
│   ├── series_dataset.py            # index-driven reader; decodes K slices per __getitem__
│   └── study_dataset.py             # N series -> tokens + metadata vector + label mask
├── datamodules/study_datamodule.py
├── augmentations/factory.py         # build A.Compose from YAML specs
├── layers/pooling.py                # AttentionPool2d, StudyAggregator, FiLM
├── models/
│   ├── backbones.py                 # timm wrapper with offline checkpoint support
│   ├── series_encoder.py            # backbone + temporal attention pooling
│   └── knee_net.py                  # full net: forward(series_batch, meta) -> logits[12]
├── losses/asymmetric_focal.py       # + WeightedBCE default; ignore_index=-1 support
├── metrics/auc.py                   # per-class + macro ROC-AUC
├── engines/
│   ├── train_module.py              # LightningModule; OOF dump per fold
│   └── inferencer.py                # eager fp16 predictor
├── callbacks/
│   ├── time_budget.py               # stop fit before session budget
│   ├── periodic_push.py             # save ckpt -> kaggle dataset version every N epochs
│   └── per_class_auc.py
└── loggers/csv_logger.py
notebooks/
├── 01_EDA.ipynb                     # done
├── 02_build_index_labels_folds.ipynb
├── 03_model_training.ipynb          # resume-aware loop; pushes ckpt dataset per session
└── 04_inference.ipynb               # loads pushed ckpts -> submission.csv
tests/
├── test_geometry.py                 # synthetic ordering incl. reversed normals
├── test_nlp_labeling.py             # negation/scope cases per target
├── test_loader.py                   # recursive instantiation, dot-path overrides
├── test_knee_net.py                 # shapes: variable #series/#slices
└── test_resume.py                   # push/pull roundtrip (mocked CLI), state fidelity
```

Style contract for all Python: shebang + coding header, Google docstrings,
2-space indentation, single quotes, SOLID boundaries
(Registry/Factory/Adapter/Strategy/Template Method/Facade as annotated above).

## 10. Verification Gates

- `pytest tests/` green.
- Every config file instantiates through `load_config` + `instantiate` without error.
- KneeNet forward passes shape tests for variable series/slice counts.
- Notebook 03 prints per-fold/per-class OOF AUC; sanity gate >= 0.80 macro on gold+pseudo CV.
- Notebook 04 asserts exact header, row count, and UID set equality before writing `submission.csv`.

## 11. Iteration Backlog (post-MVP)

1. XLM-R multilingual pseudo-labeler -> soft-target distillation.
2. ONNX Runtime fp16 export + parity tests (< 0.001 OOF delta).
3. TTA and snapshot ensembles.
4. MONAI MRI-specific augmentations (bias field, ghosting).
5. Per-plane encoder specialization.
6. Efficiency tuning: K ablation, fold pruning, INT8 quantization.
7. Contrastive image-report pretraining.
8. Protocol-stratified folds once domain-shift analysis is rerun cleanly.
