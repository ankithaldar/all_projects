# Framework Comparison Matrix — Ignite vs Lightning vs Keras

*Structural, line‑for‑line comparison of the three framework ports. The Keras port is counterfactual (TF/Keras absent from repo); the Lightning port is the author's migration reference.*

---

## Philosophy

| Aspect | PyTorch Ignite (repo) | PyTorch Lightning (port) | TensorFlow/Keras (port) |
|--------|----------------------|--------------------------|--------------------------|
| **Style** | Imperative, manual loop | Declarative, hooks + Trainer | Declarative, Keras API |
| **Config dispatch** | `importlib` + `class Parameters` → `if/elif` in `_init_*` | Dataclass + `LightningCLI` / `jsonargparse` | `HParams` + `tf.config` |
| **Training loop** | `engine.run(dataloader, epochs)` → manual `optimizer.step()` | `Trainer.fit(model)` → `training_step` + auto backward | `model.fit()` → auto everything |
| **Validation cadence** | `Events.EPOCH_COMPLETED(every=N)` → callback on evaluator | `Trainer(check_val_every_n_epoch=N)` | `model.fit(validation_freq=N)` |
| **Checkpointing** | `Checkpoint` + `DiskSaver` attached to evaluator | `ModelCheckpoint(monitor=..., mode=..., save_top_k=N)` | `ModelCheckpoint(monitor=..., save_best_only=True)` |
| **Distributed** | Manual `ddp` + `idist` + `sync_all_reduce` | `Trainer(strategy="ddp", devices=N)` + `seed_everything` | `tf.distribute.MirroredStrategy()` |
| **AMP** | Dead code (`USE_AMP=False` both branches) | Native `precision="16-mixed"` + `autocast(enabled=False)` guard | Native `mixed_precision.set_global_policy("mixed_float16")` |
| **LR Finder** | Manual handler removal/re‑add | `Tuner(trainer).lr_find()` | Not built‑in (use `lr_schedule` callbacks) |

---

## Code Volume Comparison

| Component | Ignite (lines in `base_engine.py` + `sed_engine.py`) | Lightning (estimated) | Keras (estimated) |
|-----------|------------------------------------------------------|----------------------|-------------------|
| **Engine / Module** | 375 + 210 = 585 | ~200 (`LightningModule` + `configure_optimizers`) | ~150 (`Model` subclass) |
| **Checkpoint / resume** | 45 (Checkpoint + DiskSaver + evaluator attach) | ~10 (`ModelCheckpoint` callback) | ~10 |
| **LR scheduling** | 60 (ConcatScheduler, warmup, one_cycle_cosine) | ~20 (`configure_optimizers` returns dict) | ~15 (`LearningRateSchedule` callback) |
| **Metrics wiring** | 40 (loss_keys, RunningAverage, attach) | ~15 (`self.log_dict` in `validation_step`) | ~15 (`model.compile(metrics=...)`) |
| **AMP** | 20 (dead code + apex try/except) | ~5 (`precision="16-mixed"`) | ~5 (`mixed_float16` policy) |
| **Distributed** | 35 (idist, manual seed, sync_all_reduce) | ~5 (`strategy="ddp"`) | ~15 (`MirroredStrategy` scope) |
| **TOTAL** | **~785** | **~265** | **~210** |

**Lines that do NOT disappear in Lightning/Keras**:
- Model architecture (logmel inside forward) — 100% custom
- `AttBlock` `softmax(tanh)` — 100% custom
- Dual heads (clip + frame) — 100% custom
- `SedScaledPosNegFocalLoss` — 100% custom
- `SedF1ScoreClip/Frame/LRAP` — 100% custom
- Audio augmentations (`audiomentations` + PANNs `SpecAugmentation`) — 100% custom
- Imbalanced sampler — 100% custom
- Inference / voting ensemble — 100% custom

**Conclusion**: The training framework is **~25 %** of the repo. The **75 % is domain logic** that no framework can express.

---

## Migration Table (Ignite → Lightning)

| Ignite concept | Lightning equivalent | Notes |
|----------------|----------------------|-------|
| `BaseEngine` | `LightningModule` | `__init__` holds model, criterion, optimizer, scheduler |
| `SedEngine` | subclass `LightningModule` | adds `training_step`, `validation_step`, `test_step` |
| `_init_model` | `self.model = ...` in `__init__` | Config string → registry dict |
| `_init_criterion_function` | `self.criterion = ...` in `__init__` | `training_step`: `loss, dict_loss = self.criterion(y_pred["clipwise_output"], y)` |
| `_init_optimizer` | `configure_optimizers()` returns `optimizer` | AdamW `amsgrad=True` |
| `_init_scheduler` | `configure_optimizers()` returns `{"optimizer": opt, "lr_scheduler": {...}}` | `interval="step", frequency=1` for step schedulers |
| `_init_train_dataloader` | `train_dataloader()` or `LightningDataModule` | Sampler logic unchanged |
| `_init_valid_dataloader` | `val_dataloader()` | |
| `_init_test_dataloader` | `test_dataloader()` | |
| `_init_metrics` | `self.log_dict(dict_loss, on_epoch=True)` in `training_step`/`validation_step` | `loss_keys` → running averages replaced by `self.log` |
| `_init_logger` | `Trainer(logger=[CSVLogger, NeptuneLogger])` | `self.log_hyperparams(cfg)` |
| `accumulation_steps` | `Trainer(accumulate_grad_batches=N)` | No manual `loss.backward()` in `training_step` |
| `clip_gradients` (manual) | `Trainer(gradient_clip_val=1.0)` | The manual `clip_gradients` is copied from Lightning source — use built‑in |
| `Events.EPOCH_COMPLETED(every=eval_every)` | `Trainer(check_val_every_n_epoch=eval_every)` | **Off‑by‑one**: Ignite fires after epoch N if `every=N`; Lightning validates after epoch N if `check_val_every_n_epoch=N` (same). |
| `Checkpoint(DiskSaver, score_function, n_saved)` | `ModelCheckpoint(monitor=track_metric, mode="max"/"min", save_top_k=n_saved)` | `metric_factor=-1` → `mode="min"`; `metric_factor=1` → `mode="max"` |
| `load_trainer_from_checkpoint` | `Trainer.fit(model, ckpt_path=...)` | Restores model + optimizer + scheduler + epoch + global_step automatically |
| `load_model_only` | `model.load_state_dict(torch.load(ckpt)["state_dict"])` | Manual; no built‑in "model only" mode |
| `complete_clear` gc handler | **Unnecessary** — Lightning clears `outputs` / `loss` between epochs automatically | |
| `LR finder` (manual handler removal) | `Tuner(trainer).lr_find()` | Native; no need to tear down validation/scheduler handlers |
| `idist` | `strategy="ddp"`, `devices=N` | `seed_everything(42+rank)` replaces manual `manual_seed` |
| `sync_all_reduce` | Built into metrics (`self.log(..., sync_dist=True)`) | |
| `setup_common_training_handlers` | `Trainer(callbacks=[...])` | `ModelCheckpoint`, `LearningRateMonitor`, `EarlyStopping` etc. |

---

## Migration Table (Ignite → Keras)

| Ignite concept | Keras equivalent | Notes |
|----------------|------------------|-------|
| `BaseEngine` | Subclass `tf.keras.Model` | `__init__` holds model, criterion, optimizer |
| `SedEngine` | Override `train_step`, `test_step` | No `validation_step` — validation is automatic with `validation_data` |
| `_init_model` | `self.model = ...` in `__init__` | DenseNet121 via `tf.keras.applications.DenseNet121` (include_top=False) |
| `_init_criterion_function` | `self.criterion = ...` in `__init__` | `train_step`: `loss, dict_loss = self.criterion(y_pred["clipwise_output"], y)` |
| `_init_optimizer` | `compile(optimizer=AdamW(amsgrad=True))` | |
| `_init_scheduler` | Custom `LearningRateSchedule` callback or `optimizer.lr.assign(...)` in `on_train_batch_begin` | Step‑based schedulers need per‑batch callback; epoch‑based → `LearningRateScheduler` |
| `_init_train_dataloader` | `tf.data.Dataset` pipeline | Sampler logic → `tf.data.experimental.shuffle` + `batch` |
| `_init_valid_dataloader` | `tf.data.Dataset` | |
| `_init_metrics` | `self.add_metric(..., aggregation="mean")` in `train_step`/`test_step` | `loss_keys` → separate metric per key |
| `_init_logger` | `model.compile(..., callbacks=[CSVLogger, NeptuneCallback])` | `NeptuneCallback` must be custom (no built‑in) |
| `accumulation_steps` | Not directly supported → manual gradient accumulation in `train_step` | |
| `clip_gradients` | `optimizer.clipnorm=1.0` or `clipvalue=1.0` | |
| `Events.EPOCH_COMPLETED(every=eval_every)` | `model.fit(validation_freq=eval_every)` | Same semantics |
| `Checkpoint` | `ModelCheckpoint(monitor=..., save_best_only=..., save_weights_only=False)` | |
| `load_trainer_from_checkpoint` | `model.load_weights(ckpt_path)` + `optimizer.build(...)` manual | No auto restore of optimizer state or epoch |
| `load_model_only` | `model.load_weights(ckpt_path)` | Direct equivalent |
| `complete_clear` | Unnecessary — TF handles memory | |
| `LR finder` | Not built‑in — use `tf.keras.callbacks.LearningRateScheduler` with manual search | |
| `idist` / `sync_all_reduce` | `MirroredStrategy().scope()` | `tf.distribute` handles all‑reduce automatically |
| `setup_common_training_handlers` | `callbacks=[...]` in `model.fit()` | |

---

## Would Lightning Have Won Faster?

| Argument | Weight |
|----------|--------|
| **Against**: The competition lever was a 13‑model vote, not iteration speed. | HIGH |
| **Against**: On a T4 with 30 s clips, step overhead is sub‑second — Ignite vs Lightning overhead is ~0.5 s/step < 1 % of runtime. | MEDIUM |
| **Against**: AMP fix (`autocast(enabled=False)`) is needed in *both* frameworks — same effort. | MEDIUM |
| **For**: `ModelCheckpoint(monitor=...)` saves real time vs manual evaluator‑attached Checkpoint. | MEDIUM |
| **For**: `log_dict` simplifies `loss_keys` metric wiring. | LOW |
| **For**: `Trainer(lr_find())` saves the manual handler setup/teardown. | LOW |
| **For**: Lightning's DDP handling (`strategy="ddp"`) is less error‑prone than manual `idist`. | MEDIUM |
| **For**: `accumulate_grad_batches` replaces manual accumulation loop. | LOW |

**Verdict**: **No.** The winning lever was the 13‑model ensemble vote (architecture + threshold), not the training framework. The domain logic (75 % of code) is orthogonal to the framework. Lightning would have saved ~200 lines of boilerplate and made checkpointing/logging easier, but would not have changed the public LB score by more than ±0.001.

---

## Summary Table

| Category | Ignite (repo) | Lightning (port) | Keras (port) |
|----------|---------------|------------------|--------------|
| **Lines of boilerplate** | ~785 | ~265 | ~210 |
| **Lines of domain logic** | ~1800+ | ~1800+ (identical) | ~1800+ (identical) |
| **AMP support** | Dead code | Native (with guard) | Native (with guard) |
| **Checkpointing** | Manual evaluator attach | 1‑liner `ModelCheckpoint` | 1‑liner `ModelCheckpoint` |
| **LR scheduling** | Custom `ConcatScheduler` | Native `configure_optimizers` | Custom callback |
| **Distributed** | Manual `idist` | Native `strategy="ddp"` | Native `MirroredStrategy` |
| **Metrics / logging** | `loss_keys` + RunningAverage | `self.log_dict` | `add_metric` |
| **LR finder** | Manual handler swap | Native `Tuner.lr_find()` | Manual |
| **Reproducibility** | Manual seed per rank | `seed_everything` | `tf.random.set_seed` |
| **Migration effort** | — | ~1 day | ~2 days |

---

*Generated 2026‑10‑06 from the three port chapters and source code.*

## Cross‑references

- `docs/framework-comparisons/pytorch-lightning-port.md` — full Lightning port
- `docs/framework-comparisons/tensorflow-keras-port.md` — full Keras port
- `docs/00-overview-and-secret-sauce.md` — the reverse‑engineered thesis