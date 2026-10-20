# Cornell Birdcall Identification — 1st Place Solution: Technical Documentation

Written for the author's future self: every claim is traceable to `src/`, to the
repo's own `README.md`, or is explicitly marked *inferred*. The verified fact base and
its correction history is [`FINDINGS.md`](FINDINGS.md).

## The one-paragraph answer

A single architecture — PANNs-style SED with a DenseNet121 backbone and `tanh`
attention (`PANNsDense121Att`) — trained 13 times (4-fold no-mixup, 5-fold no-mixup,
4-fold mixup) with plain BCE on weak, file-level labels. At inference (external
Kaggle kernel, not in this repo) a species is accepted for a 5-second row when
**≥4 of 13 models** fire with **clip ≥ 0.3 AND frame ≥ 0.3**. That ensemble was 2nd
on public (0.616) and **1st on private (0.681)**, moving 7th → 1st. Details and
provenance: [00 — Overview and secret sauce](modules/00-overview-and-secret-sauce.md).

## Reading order

| # | Document | Covers |
|---|---|---|
| 00 | [Overview and secret sauce](modules/00-overview-and-secret-sauce.md) | end-to-end flow, the 13 models, why it won, what is missing |
| 01 | [Data pipeline](modules/01-data-pipeline.md) | `SedDataset`, `ImbalancedDatasetSampler`, audio + label decoding |
| 02 | [Audio front-end](modules/02-audio-frontend.md) | vendored PANNs STFT / log-mel / SpecAugmentation, AMP NaN root cause |
| 03 | [Augmentations](modules/03-augmentations.md) | audiomentations 0.11.0 waveform stage, SpecAugment, mixup |
| 04 | [Model architecture](modules/04-model-architecture.md) | `PANNsDense121Att`, `AttBlock`, shapes, parameter counts |
| 05 | [Training engine](modules/05-training-engine.md) | Ignite 0.4.0.post1 wiring, scheduler, accumulation, distributed |
| 06 | [Checkpointing and resuming](modules/06-checkpointing-and-resuming.md) | evaluator-attached saver, `load_model_only`, warm starts |
| 07 | [Losses](modules/07-losses.md) | 16 loss files, the inert focal loss, `loss_keys` |
| 08 | [Inference and submission](modules/08-inference-and-submission.md) | why `sed_test.py` cannot be the submission path; what the vote needs |
| 09 | [Config and cross-validation](modules/09-config-and-cross-validation.md) | all 21 configs, 13-model reconciliation, CV gaps |
| 10 | [Metrics and tracking](modules/10-metrics-and-tracking.md) | LRAP / F1 clip / F1 frame, Neptune, Slack |
| 11 | [Legacy research paths](modules/11-legacy-research-paths.md) | `train.py`…`train_v4.py` generations |
| 12 | [Kaggle and compute constraints](modules/12-kaggle-constraints.md) | stated limits, memory arithmetic |
| 13 | [Experiment log](modules/13-experiment-log.md) | the README LB tables and config-embedded scores — nothing more exists |

Framework ports (kept separate, not merged):
[PyTorch Lightning 2.x](framework-comparisons/pytorch-lightning-port.md) ·
[TensorFlow / Keras 3](framework-comparisons/tensorflow-keras-port.md) ·
[comparison matrix](frameworks/comparison-matrix.md).
TensorFlow/Keras does not appear anywhere in the repo; that port is counterfactual.

Porting specs (also kept separate):
[code issues and framework wins](ports/code-issues-and-framework-wins.md) ·
[feature parity matrix](ports/feature-parity-matrix.md) ·
[PyTorch Lightning guide](ports/pytorch-lightning-guide.md) ·
[TensorFlow/Keras guide](ports/tensorflow-keras-guide.md).
These are the actionable specs for actually doing the two ports: a row-by-row
checklist of every capability in the original, and a review of the 23 issues
that decide whether the port is worth doing.

Per-file reference (also kept separate): [`src-reference/`](src-reference/README.md)
covers **all 85 files** under `src/` — what each one does, its public API, and
whether it is live, legacy, dead or broken. It is the file-level index that
the topic chapters above are not.

## Running training

```bash
cd src                      # no __init__.py anywhere: implicit namespace packages
pip install -r ../requirements.txt   # torch/torchvision/torchaudio/librosa NOT pinned — preinstall
python sed_train.py --config config_params.final_sed.final_sed_dense121_nomix_fold0
```

`--config` is a **dotted module path**, imported with `importlib` (`sed_train.py`).
The fold configs also expect `/data` (32 kHz wav mirror), `/pinknoise`, `/folds/*.csv`
or `/folds5/*.csv`, and write to `/saved_models` — none of which ship with the repo.

## Most important corrections to first impressions

1. **The focal loss is inert**: at shipped params it is bit-identical to mean BCE ([07](modules/07-losses.md)).
2. **`sed_test.py` is not the submission pipeline**: it does not compile, and even fixed it cannot emit 5-second rows ([08](modules/08-inference-and-submission.md)).
3. **No mel caching**: SED computes log-mel on the GPU inside `forward()` ([02](modules/02-audio-frontend.md)).
4. **CV is not reproducible**: fold CSVs are absent ([09](modules/09-config-and-cross-validation.md)).
5. **AMP is off** in both branches of the apex `try/except` (fp16 underflow of `amin=1e-10`) ([02](modules/02-audio-frontend.md)).
6. **Line endings**: 20 of the 22 `.py` files under `src/config_params/` are CRLF (`configs.py` + 19 of 20 final configs); `example_config.py` and `final_sed/final_sed_dense121_mix_2.py` are LF. Line-anchored `grep` silently misses CRLF files.
