# 12 — Kaggle and Compute Constraints

> [!warning] Scope
> Only constraints that are **stated in the repo** (README / code) or **derivable
> from the code** are listed. Kaggle's kernel time/RAM limits for this competition
> are **not recorded anywhere in the repo**; they are marked *not documented* rather
> than guessed. Tags: **[README]**, **[code]**, **[derived]** (arithmetic shown).

## 1. Competition / evaluation constraints

| Constraint | Value | Source | How the solution responds |
|---|---|---|---|
| Test set | ~150 mp3 recordings, ~10 min each, "will not all fit in a notebook's memory at the same time" | **[README]** | must stream file-by-file in the inference kernel (not in repo) |
| Sites | 3 sites; sites 1–2 labelled per 5 s, site 3 at file level only | **[README]** | needs two output resolutions; nothing in-repo produces either (see [08](08-inference-and-submission.md)) |
| Metric | row-wise micro-averaged F1 | **[README]** | favours a high-precision vote (see [00 §5](00-overview-and-secret-sauce.md)) |
| Final selections | 2 submissions selected before private LB reveal | **[README]** | 13-model/4-vote and 9-model/3-vote picked |
| Daily submission cap | "max 2 submissions per day" — limited threshold exploration (frame 0.3 never tried on the 9-model ensemble) | **[README]** | |
| Labels | weak: file-level primary + secondary species, no timestamps | **[code]** `sed_dataset.py` | SED with attention pooling; labels broadcast to all snippets/frames |
| Data scarcity | "≤ 100 labelled samples per bird class" | **[README]** | smaller head, DenseNet121, heavy augmentation, class-balanced sampler |

## 2. Compute environment actually used (training)

| Item | Value | Source |
|---|---|---|
| Machine | GCP `n1-standard-8`, **1 × NVIDIA T4**, preemptible, 300 GB disk | **[README]** §0 |
| Image | `pytorch-latest-gpu` (Deep Learning VM) — torch version unrecorded | **[README]** |
| Precision | fp32 only; AMP disabled (`USE_AMP=False`), NaNs under fp16 because of `amin=1e-10` | **[code + README]** |
| Data on disk | `/data` (32 kHz wav mirror), `/pinknoise`, optional `/background/data_ssw`, `/folds`, `/folds5`, `/saved_models` | **[code]** configs |
| Wall-clock per model | **not documented** | — |
| Kaggle kernel limits (GPU hours, RAM) | **not documented** in repo | — |

## 3. Memory arithmetic at the shipped configs **[derived]**

Inputs: `sr=32000`, `PERIOD=30`, `hop=320`, `n_fft=1024`, 64 mels, 264 classes, fp32.

| Tensor | Shape | Size |
|---|---|---|
| One training snippet | 30 × 32000 = 960 000 samples | 3.84 MB |
| Train waveform batch, mix family | (32, 1, 960 000) | 122.9 MB |
| Train waveform batch, 4-fold nomix | (24, 1, 960 000) | 92.2 MB |
| Train waveform batch, 5-fold | (28, 1, 960 000) | 107.5 MB |
| Valid waveform batch | 32 files × 2 snippets → (64, 960 000) | 245.8 MB |
| Power spectrogram per snippet | (1, 3001, 513) | 6.16 MB |
| `framewise_output` per snippet | (3001, 264) | 3.17 MB |

Frames: `1 + 960000 / 320 = 3001` (`center=True`). DenseNet121 `.features`
reduces 3001 → 93 time steps (measured), interpolated ×32 → 2976, padded to 3001.

## 4. Inference-side arithmetic **[derived]**

A 10-minute test file = 19 200 000 samples → 60 001 frames → **120** five-second
rows (500 frames each). With `PERIOD=30`, covering it needs `num_test_samples=20`;
every shipped config uses `2` (= first 60 s). This is one more reason the
submission path must have lived outside the repo (see [08](08-inference-and-submission.md)).

## 5. Open items

- Record per-model training time and GPU memory when re-running (none committed).
- Pin torch/torchvision/torchaudio/librosa (unpinned in `requirements.txt`).
