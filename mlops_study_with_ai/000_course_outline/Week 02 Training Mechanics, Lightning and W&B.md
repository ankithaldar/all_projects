# Week 2: Training Mechanics, Lightning and W&B

*Phase A (Foundations) of the 26-week plan · 15 hours · Deliverable: Project P1, the Training Template · written 13 Nov 2026. Links marked ✓ were confirmed live in search on that date.*

This file expands Week 2 of the [master plan](https://claude.ai/artifact/FD56r2oaE9BXX8VZPiQAeo). It contains no solution code: every coding task is written as an exercise with a pass condition, so you do the typing.

## 1. Week at a glance

**Goal.** Take the makemore MLP you code along with in the videos and turn it into a reproducible, tracked, tested training template built on PyTorch Lightning and Weights & Biases, while learning why initialization and normalization decide whether a network trains at all.

**By Sunday you can:**

- Predict the initial loss of a classifier before running it, and diagnose the model when the number is wrong.
- Show, with plots you generated, why saturated tanh units, bad init scale and missing normalization stall training.
- Structure training code as LightningModule, LightningDataModule and Trainer with callbacks, with no data logic inside the model.
- Make a run reproducible (same seed, same loss curve) and prove it with a test.
- Log configs, metrics, tables and versioned checkpoints to W&B, and reload a checkpoint from an Artifact in a clean process.
- Ship it with config files, tests and a green CI pipeline.

**Hour budget (15 h, matches the master plan rhythm)**

| Block | Hours | Content |
| --- | --- | --- |
| Cheat sheet and orientation | H1 (plus 10 min in H15) | CS1, Karpathy’s Recipe, self-test, five-line notes |
| Code-along | H2–H6 (5 h) | makemore Part 2 (MLP) and Part 3 (activations, gradients, BatchNorm) |
| Build | H7–H12 (6 h) | P1: Lightning port, DataModule, Trainer, W&B, config files, init and BatchNorm experiments |
| Hardening | H13–H14 (2 h) | ML tests, CI, pre-commit, Docker smoke test |
| Review | H15 (about 50 min) | checklist, README, demo, tag |

**Suggested session split:** five 3-hour sessions (H1–3, H4–6, H7–9, H10–12, H13–15), or any rhythm that keeps the hours in order. The order matters: later hours reuse what earlier hours produce.

**Definition of done for Week 2**

1. One repo (the Week 1 `ml-template` extended, or a new `p1-training-template`) with a single command that trains the model.
2. Same seed gives the same loss curve within a tolerance you state, enforced by a test.
3. A W&B project with at least 8 comparable runs, each carrying its config, git SHA and a checkpoint Artifact with `latest` and `best` aliases.
4. A results matrix of 4 init schemes × BatchNorm on/off, with five lines of written findings.
5. The seven test categories from H13 passing, CI green on a pull request, Docker smoke test passing.
6. A README with the results table and diagnostic plots, a 2-minute demo, one ADR, and your five-line CS1 notes.

## 2. Everything to cover this week

| # | Topic | Hours |
| --- | --- | --- |
| 1 | Initial-loss sanity check (about ln of the vocabulary size) | H1, H4, H13 |
| 2 | Overfit-one-batch debugging ritual | H1, H2, H13 |
| 3 | Context windows and embedding lookup | H2 |
| 4 | Minibatch training, learning-rate search, learning-rate decay | H3, H9 |
| 5 | Train, dev and test splits; over- and underfitting | H3, H8 |
| 6 | Saturated tanh, dead units, init scale (fan-in and gain) | H4, H12 |
| 7 | BatchNorm: train vs eval behavior, running statistics, momentum, redundant bias | H5, H13 |
| 8 | Diagnostics: activation and gradient statistics, update-to-data ratio | H6, H12 |
| 9 | LightningModule | H7 |
| 10 | LightningDataModule | H8 |
| 11 | Trainer, callbacks (ModelCheckpoint, EarlyStopping, LR monitor), `fast_dev_run` | H9 |
| 12 | Reproducibility: `seed_everything`, deterministic mode | H9, H13 |
| 13 | Learning-rate finder | H9 |
| 14 | W&B runs, config, metrics, tags | H10 |
| 15 | W&B Artifacts, aliases, reloading from an Artifact | H10 |
| 16 | W&B Tables | H10 |
| 17 | Sweeps or Hydra multirun | H11 |
| 18 | Config files, CLI overrides, fail-fast validation | H11 |
| 19 | ML unit tests: shapes, init loss, overfit, seed, checkpoint, eval mode | H13 |
| 20 | CI, pre-commit, Docker smoke test | H14 |
| 21 | README, ADR, demo | H15 |

**Deliberately skipped this week:** makemore Part 1 (unless the H1 self-test fails), Part 4 (backprop by hand) and Part 5 (WaveNet), all three papers behind this material, multi-GPU, TorchMetrics in depth, Lightning Fabric, W&B Registry and Weave, advanced Hydra plugins. Part 4 is a good stretch if you finish early.

## 3. Resource pack

**Videos**

- [Karpathy: Building makemore Part 2, MLP](https://youtu.be/TCH_1BHY58I) ✓ (about 1 h 15). Core. H2–H3.
- [Karpathy: Building makemore Part 3, Activations & Gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc) ✓ (about 2 h). Core. H4–H6.
- [Sebastian Raschka: Deep Learning Fundamentals, Unit 5, Organizing your PyTorch code with Lightning](https://lightning.ai/courses/deep-learning-fundamentals/overview-organizing-your-code-with-pytorch-lightning/) ✓. Watch 5.1 (Lightning basics), 5.2 (MLP with the Trainer), 5.4 (reproducibility), 5.5 (DataModules), 5.6 (logging), 5.8 (callbacks, optional). Bite-sized videos. H7–H9.
- [Deep Learning Fundamentals course page](https://lightning.ai/courses/deep-learning-fundamentals/) ✓: Unit 6 has 6.1 (checkpointing and early stopping), 6.2 (learning rates), 6.6 (BatchNorm) and 6.8 (debugging). Use 6.1 and 6.2 in H9.
- Optional: [makemore Part 1, bigrams](https://www.youtube.com/watch?v=PaCmpygFfXo) ✓ (only if the H1 self-test fails); [Part 4, backprop ninja](https://youtu.be/q8SA3rM6ckI) ✓ and [Part 5, WaveNet](https://youtu.be/t3YJ5hKiMQ0) ✓ (stretch only).

**GitHub and notebooks**

- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) ✓: lecture notebooks, [Part 2 notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb) ✓ and [Part 3 notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb) ✓. Use them to check your work, not to type from.
- [karpathy/makemore](https://github.com/karpathy/makemore) ✓: the names dataset (`names.txt` in the repo root).
- [Lightning-AI/dl-fundamentals](https://github.com/Lightning-AI/dl-fundamentals) ✓: course code plus Unit 5 exercises, for example the [Lightning regression exercise](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit05-lightning/exercises/1_lightning-regression) ✓ and the [custom callback exercise](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit05-lightning/exercises/2_custom-callback) ✓. Do both as warm-ups if you have spare minutes.
- [wandb/examples](https://github.com/wandb/examples): runnable W&B demos.
- Reference only, after your own first attempt: [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) ✓. Study its folder layout and config groups; do not copy it wholesale (it is large, and its maintainers document Hydra-with-multi-GPU issues).

**Documentation**

- Lightning: [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) ✓ (includes the reproducibility section and `deterministic`), [WandbLogger API](https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.loggers.wandb.html) ✓, [LightningCLI (2.6.6 docs)](https://lightning.ai/docs/pytorch/2.6.6/reference/cli/lightning_cli.md) ✓, [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), [DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html), [checkpointing](https://lightning.ai/docs/pytorch/stable/common/checkpointing.html), [callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html). Lightning is reorganizing its docs (the Trainer page now lists `/core-api/trainer` as canonical), so if a path redirects or 404s, search the page title.
- W&B: start at the [docs examples index](https://docs.wandb.ai/examples) ✓, which links Experiments overview, Configure experiments, Artifacts overview, Create and use a dataset artifact, Tables, Run a sweep and Registry aliases. W&B reorganized its docs and some older deep links now 404, so navigate from this index. The [Lightning integration page](https://docs.coreweave.com/products/wandb/integrations/lightning) ✓ is now hosted on CoreWeave’s docs.
- [Hydra](https://hydra.cc/docs/intro/) (pick Hydra or LightningCLI in H11, not both).
- [pytest](https://docs.pytest.org/), [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html) ✓, [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), [pre-commit](https://pre-commit.com/), [GitHub Actions](https://docs.github.com/en/actions), [setup-uv action](https://github.com/astral-sh/setup-uv), [Made With ML: testing ML systems](https://madewithml.com/courses/mlops/testing/).

**Reading**

- [Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) ✓ (about 25 minutes; H1).
- CS1 (autograd and the training loop) in Appendix A of the [master plan](https://claude.ai/artifact/FD56r2oaE9BXX8VZPiQAeo) (H1).
- Skip unless curious: [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), [Batch Normalization](https://arxiv.org/abs/1502.03167), [Delving Deep into Rectifiers (Kaiming init)](https://arxiv.org/abs/1502.01852).

**One practical note.** Pick a single Lightning namespace for the whole project (use the unified `lightning` package consistently). Mixing `pytorch_lightning` and `lightning.pytorch` imports in one project produces confusing type and callback errors.

## 4. Hour-by-hour plan

Each hour lists the focus, the materials, the exercises (with IDs you can reference in your notes) and a “done when” test. Exercises are written for you to implement; none comes with code.

### Block 0: Orientation

### Hour 1: Cheat sheet, the Recipe, and a self-test

- **Focus:** start with the right mental model of training and a known baseline.
- **Materials:** CS1; Karpathy’s Recipe (read the sections on inspecting the data, setting up the training and evaluation skeleton, and overfitting; skim the rest).
- **Minute plan:** 15 CS1; 20 Recipe; 15 self-test; 10 setup (create the Week 2 branch, confirm `wandb login` works, note whether you have a GPU).
- **X1, pre-flight self-test (no tutorial open):** in a scratch notebook, (a) read the names list and build the vocabulary of 26 letters plus one boundary token; (b) state on paper the expected cross-entropy of a model that predicts uniformly over that vocabulary, then verify it numerically; (c) build the context-to-next-character training pairs for block size 3 and confirm the tensor shapes.
- **Branch rule:** if any step takes more than 15 minutes, watch makemore Part 1 at 2× (about 1 h) and pay for it by dropping the optional sweep and report tasks in H10–H11. Never pay from Hardening.
- **Done when:** you can state the uniform-predictor loss for 27 classes (about 3.30) without computing it, and your notes hold three sentences on why Karpathy says training fails silently and why you overfit one batch first.

### Block A: Code-along (H2–H6)

Rule for every code-along hour: type in your own notebook, and before each cell runs, write down the shape or value you expect. Every miss is a gap in understanding; track the count.

### Hour 2: makemore Part 2, first half (dataset, embeddings, hidden layer, loss)

- **Focus:** from characters to a trainable MLP.
- **Materials:** Part 2 video up to the first working training loop; Part 2 notebook for checking; `names.txt`.
- **X2.1 Predict, then run:** keep a tally of shape and value predictions made before executing each cell. Target at least 90% correct by the end of the hour.
- **X2.2 Rebuild from memory:** close the notebook and rewrite the context-window dataset builder; it must match the video’s tensors for the first five words.
- **X2.3 Overfit one batch:** train on a single batch of 32 examples until the loss is near zero; record how many steps that took. If it cannot get there, stop and find the bug.
- **Done when:** X2.1 tally is at least 90%, and X2.2 and X2.3 pass.

### Hour 3: makemore Part 2, second half, plus the mutate step

- **Focus:** minibatches, learning-rate search, splits, over- and underfitting, scaling the model.
- **Materials:** the rest of Part 2; the exercises in the video description (as I read them: beat the video’s best dev loss by tuning hyperparameters, and compute the uniform-initialization loss then tune the init so training starts near it).
- **X3.1 Swap the data:** replace names with another list of short strings (place names, Pokémon names, any 10k short words). Only the data file may change; note anything else you had to edit and fix the hard-coding.
- **X3.2 Block-size study:** train with block sizes 2, 3, 4 and 5; plot dev loss against block size and write a two-sentence conclusion.
- **X3.3 Learning-rate search:** sweep log-spaced learning rates, plot loss against rate, choose a rate and justify it; then add a step decay and record the dev-loss difference.
- **Done when:** your own notebook reproduces a dev loss close to the video’s (use your number as the reference for the rest of the week), and X3.1–X3.3 are saved with plots.

### Hour 4: makemore Part 3, first third (initialization)

- **Focus:** why a freshly initialized network starts with a bad loss and dead-looking units.
- **Materials:** Part 3 up to the end of the init-scale discussion; Part 3 notebook.
- **X4.1 Confidently wrong:** reproduce the oversized initial loss, fix it by shrinking the output layer’s weights and zeroing its bias, and check the new initial loss is within 10% of ln 27 (about 3.30).
- **X4.2 Saturation audit:** plot histograms of hidden pre-activations and tanh outputs before and after rescaling the hidden-layer init; record the percentage of tanh outputs with magnitude above 0.97 in each case.
- **X4.3 Init scale by hand:** work out the fan-in-based scale for your layer sizes, build the initialization manually, then compare with PyTorch’s built-in Kaiming initializer; the resulting weight standard deviations must agree within 5%.
- **Done when:** initial loss is within 10% of ln 27, and you have before-and-after saturation percentages.

### Hour 5: makemore Part 3, second third (Batch Normalization)

- **Focus:** what BatchNorm does to training, and the train-versus-eval trap.
- **Materials:** the BatchNorm segment of Part 3.
- **X5.1 Own BatchNorm:** implement the BatchNorm forward pass yourself following the video, including learnable scale and shift and running statistics with momentum. In train mode its output on a fixed batch must match PyTorch’s BatchNorm layer within 1e-5.
- **X5.2 Init-sensitivity table:** train with and without BatchNorm at three init scales (too small, right, too large) and tabulate dev loss. Write one sentence on what the table says about BatchNorm and init.
- **X5.3 Train vs eval trap:** feed the same single example inside batches of different composition, in train mode and then in eval mode. Document that train-mode output changes with its batch mates and eval-mode output does not. This is the root of a classic production bug.
- **X5.4 Running stats check:** compare the running mean and variance with statistics computed in one pass over the whole training set; note the gap.
- **Done when:** X5.1 matches to 1e-5, and the X5.3 note exists.

### Hour 6: makemore Part 3, final third (layer classes and diagnostics)

- **Focus:** the instruments you will reuse all course: activation, gradient and update statistics.
- **Materials:** the layer-class refactor (skim at 1.5×, since you use `torch.nn` next hour) and the diagnostics segment.
- **X6.1 Diagnostics function:** write a reusable routine that, for any stack of layers, produces per-layer activation mean, standard deviation and saturation percentage; per-layer gradient standard deviation; and the update-to-data ratio per parameter over time (the lecture’s rule of thumb is a ratio near 1e-3). Run it on three configs: bad init, good init, and BatchNorm.
- **X6.2 Interpretation:** for each config write five lines on what the plots show and what a healthy network looks like.
- **X6.3 Save the evidence:** export the three plot sets; they go in the README in H15.
- **Done when:** you can look at an unlabeled set of these plots and name which config produced them.

### Block B: Build, Project P1 (H7–H12)

Suggested repo layout (folders only): a source package holding the model, the data module, the callbacks and the training entrypoint; a `configs` folder; a `tests` folder; a `notes` folder for your predictions, interpretations and the ADR. Keep everything from the notebooks out of the package until it has been re-typed and understood.

### Hour 7: LightningModule

- **Focus:** port the Part 3 MLP into a LightningModule built from `torch.nn` layers, with exact parity to your notebook.
- **Materials:** Raschka Units 5.1 and 5.2 (watch at 1.25× and follow along in your repo); LightningModule and Trainer docs.
- **X7.1 Build the module:** hyperparameters as constructor arguments that are saved into every checkpoint; a forward pass; training and validation steps; optimizer configuration. Rule: the module never reads files or builds datasets.
- **X7.2 Smoke test first:** run a single batch through training and validation with `fast_dev_run` before any real run.
- **X7.3 Parity check:** with the same seed and data, train a few thousand steps on CPU. Initial loss and final train and dev loss must match your notebook within a tolerance you choose (suggest 0.05). If they do not, bisect in this order: init, data order, BatchNorm mode, optimizer.
- **X7.4 Logging:** log train and validation loss under separate prefixes, plus learning rate and global step.
- **Done when:** parity holds and the tolerance is written in your notes.

### Hour 8: LightningDataModule

- **Focus:** move all data logic out of the model and make splits deterministic.
- **Materials:** Raschka Unit 5.5; DataModule docs; `names.txt`.
- **X8.1 Build the DataModule:** a preparation step that reads the file; a setup step that builds the vocabulary, the context windows and a deterministic 80/10/10 split driven by a seed; train, validation and test loaders. Expose vocabulary size and block size as properties that the model reads.
- **X8.2 Leakage check:** split by word first, then build windows. Write a check that no word appears in two splits, and two sentences on why splitting windows would leak.
- **X8.3 Determinism:** two constructions with the same seed give identical splits; different seeds differ.
- **X8.4 Loader throughput:** measure examples per second with 0 and with 2 workers, record the numbers, and pick a default.
- **Done when:** splits are disjoint at word level, shapes are correct, and the module trains through the DataModule with no model changes.

### Hour 9: Trainer, callbacks, reproducibility, learning-rate finder

- **Focus:** let the Trainer do the loop work, and prove runs are repeatable.
- **Materials:** Raschka Units 5.4, 6.1 and 6.2; Trainer docs (reproducibility, callbacks, `deterministic`); PyTorch reproducibility notes.
- **X9.1 Reproducibility:** seed everything, including dataloader workers, and enable deterministic mode. Run twice on CPU; curves must match within tolerance. If you have a GPU, repeat there and note what stops being exactly repeatable.
- **X9.2 Callbacks:** checkpoint on best validation loss keeping the top 3; early stopping with a patience you can justify; learning-rate monitor.
- **X9.3 Learning-rate search, twice:** your manual log-spaced sweep from H3 versus Lightning’s learning-rate finder. The chosen rates should land within 2× of each other; write two sentences on why they can differ.
- **X9.4 Kill and resume:** stop a run midway and resume from the last checkpoint. The loss should continue without a jump. List what must be saved for that to be true (optimizer state, BatchNorm running statistics, epoch counter).
- **Done when:** X9.1 curves match, three top checkpoints exist, resume is smooth.

### Hour 10: Weights & Biases

- **Focus:** make every run explain itself.
- **Materials:** W&B docs index (Experiments overview, Configure experiments, Artifacts overview, Create and use a dataset artifact, Tables); WandbLogger API; the Lightning integration page; Raschka Unit 5.6.
- **X10.1 Logger and metadata:** set the project, a run-name convention (for example project-model-seed), tags and the full hyperparameter set. Add the git commit SHA and a dataset checksum as extra config entries.
- **X10.2 Checkpoint Artifacts:** enable model logging tied to your checkpoint callback. Confirm the `latest` and `best` aliases appear. Download the best Artifact into an empty directory, reload it in a fresh process, and reproduce the logged validation loss within tolerance.
- **X10.3 Samples table:** each epoch, log a W&B Table with 20 generated names produced with a fixed sampling seed, so you can judge quality by eye as well as by loss.
- **X10.4 Compare runs:** run three configurations and build a comparison view. Note one thing you would have missed without it.
- **X10.5 Offline mode:** run with W&B disabled and confirm training still works. CI needs this in H14.
- **Done when:** a clean-process reload from the Artifact reproduces the number on the dashboard.

### Hour 11: Config files and a mini-sweep

- **Focus:** no hyperparameter lives in code.
- **Materials:** Hydra intro (or LightningCLI); W&B Run a sweep (optional); peek at the reference template’s layout only after you have finished X11.1.
- **First 10 minutes:** choose Hydra or LightningCLI and draft the ADR “where configuration lives and why”.
- **X11.1 Externalize everything:** four config groups: data (block size, batch size, split seed, workers), model (hidden size, layers, embedding dimension, init scheme, BatchNorm flag), trainer (steps, seed, precision), logging (project, tags).
- **X11.2 One entrypoint:** a single train command with command-line overrides. The resolved config is saved next to each checkpoint and logged to W&B.
- **X11.3 Fail fast:** reject invalid combinations before training starts, with clear errors: a negative learning rate, BatchNorm with batch size 1, an unknown init scheme.
- **X11.4 Mini-sweep *(opt.)*:** 6 to 10 runs over hidden size and learning rate, through config multirun or W&B Sweeps; capture a parallel-coordinates view.
- **Done when:** a new experiment needs zero code edits, and invalid configs fail in under a second.

### Hour 12: The mutate step, init and BatchNorm experiments inside your template

- **Focus:** re-run Karpathy’s lessons with your own tooling, so they become yours.
- **Materials:** your X6.1 diagnostics from H6; the H4 and H5 notes.
- **X12.1 New config options:** init scheme (naive, small output layer, fan-in with gain, all-zero) and BatchNorm on or off.
- **X12.2 Diagnostics callback:** port X6.1 into a Lightning callback that logs, every N steps, per-layer activation mean, standard deviation and saturation percentage, gradient standard deviation, and update-to-data ratio to W&B. Measure its overhead and keep it under 10% of step time.
- **X12.3 The matrix:** run 4 init schemes × BatchNorm on/off = 8 short runs (about 2,000 steps each). Write predictions first (initial loss, and whether it trains), then compare. Explain any mismatch with ln 27 in three sentences.
- **X12.4 All-zero init (lecture exercise):** inspect gradients and activations to find which parameters receive learning signal and why the network only partly trains. Five-line explanation.
- **X12.5 Stretch:** after training with BatchNorm, fold it into the preceding linear layer and verify eval-mode outputs match within 1e-5. Write it as a test.
- **Done when:** the 8-run table has predictions beside outcomes, the findings note exists, and the callback overhead is measured.

### Block C: Hardening (H13–H14)

### Hour 13: Tests for ML code

- **Focus:** catch silent failures automatically.
- **Materials:** pytest docs; Made With ML testing lesson; PyTorch reproducibility notes.
- **Write seven test groups:**
  - **T1 Data:** shapes, dtypes, vocabulary size; splits disjoint; split determinism.
  - **T2 Model:** forward output shape; initial loss within 10% of ln(vocabulary size) under the scaled init.
  - **T3 Overfit one batch:** loss falls below your threshold within a step limit, on CPU, in under 30 seconds.
  - **T4 Seed reproducibility:** two 200-step runs with the same seed match within tolerance; different seeds differ.
  - **T5 Checkpoint round trip:** save, load into a fresh module, identical eval outputs.
  - **T6 Eval mode:** in eval mode, one example’s output does not depend on its batch mates (X5.3 as a test).
  - **T7 Config:** every shipped config loads and validates; invalid configs fail with the expected error.
- **Rules:** CPU only, tiny data, no network calls, W&B disabled by environment, whole suite under 2 minutes locally.
- **X13.1 Mutation check:** introduce three deliberate bugs (leak a word across splits, leave the model in the wrong mode at validation, use the wrong init scale) and confirm at least one test fails for each. Revert them.
- **Done when:** all seven groups pass and every injected bug is caught.

### Hour 14: CI, pre-commit, Docker smoke test

- **Focus:** the pipeline enforces what you just wrote.
- **Materials:** GitHub Actions docs; setup-uv action; uv, ruff and pre-commit docs; your Week 1 Dockerfile.
- **X14.1 Workflow:** on pull requests and pushes to main: install from the lockfile with caching, lint and format check, type check, tests, with W&B disabled. Fail if the lockfile is out of sync.
- **X14.2 Protect main:** require CI to pass before merge; add the status badge to the README.
- **X14.3 Docker smoke test:** CI builds the Week 1 image and runs a one-batch fast dev run inside the container on CPU.
- **X14.4 pre-commit:** mirror the CI checks locally (lint, format, large-file guard, trailing whitespace).
- **X14.5 Red then green:** open a pull request with a deliberately failing test, confirm CI blocks the merge, then fix and merge.
- **Done when:** the badge is green on main and the red-then-green history is visible in the PR list.

### Block D: Review (H15)

### Hour 15: Review, README, demo, notes

- **Minute plan:** 15 run the checklist (Section 5); 15 README; 10 ADR; 10 demo; 10 five-line CS1 notes and release tag.
- **X15.1 Checklist:** run the universal checklist (Section 7 of the master plan) plus the Week 2 add-ons below. Fix or log every miss as an issue.
- **X15.2 README:** what it is and why; one command to run it; the 8-run results table; the three diagnostic plot sets from X6.3; a link to the W&B project; a known-limitations list.
- **X15.3 ADR:** one page: where configuration lives, the options you weighed, and the trade-offs.
- **X15.4 Demo:** a 2-minute screen recording: one command, a W&B run appearing, a checkpoint Artifact, and a clean-process reload.
- **X15.5 Notes and tag:** five lines of CS1 in your own words; tag the release `v0.1-p1`.
- **Done when:** a friend could clone the repo, run one command and reproduce your headline loss.

## 5. Week 2 review checklist

Run the universal ten-item checklist from the master plan, then confirm these Week 2 items:

- Same seed gives the same loss curve within your stated tolerance, and test T4 enforces it.
- The LightningModule holds no data logic; the DataModule owns splits and loaders.
- Splits are made by word, so no word crosses splits (T1).
- Initial loss is checked by a test, not by eye (T2).
- Every W&B run shows config, git SHA, dataset checksum and a checkpoint Artifact with `latest` and `best` aliases.
- A checkpoint reloaded from an Artifact in a clean process reproduces its logged validation loss.
- Configs fail fast on invalid values; no hyperparameter is hard-coded.
- CI runs lint, types, tests and the Docker smoke test with W&B disabled; main is protected.
- No API keys or tokens in the repo or in logs; your `.env` is ignored.

## 6. If you fall behind

1. Cut in this order: X12.5, X11.4, X10.4, X9.3, the stretch exercises. Never cut H13 tests or H14 CI.
2. If Part 3 runs long, skim the layer-class refactor and diagnostics segments at 1.5× and build the diagnostics only once, in H12.
3. If W&B or CI fights you for more than 30 minutes, timebox it, log an issue with what you tried, and move on; finish it in H15.
4. If you are ahead: makemore Part 4 (backprop by hand), the two Lightning exercises in the dl-fundamentals repo, or a side-by-side of Hydra and LightningCLI for the ADR.

## 7. Prepare for Week 3

- Week 3 is Karpathy’s Let’s build GPT and Let’s build the GPT Tokenizer. Read CS2 and CS3 in Appendix A of the master plan before you start (about 30 minutes).
- Keep this repo: Week 3 adds a Transformer model and a tokenizer to the same template, selected by config, rather than starting a new project.
- Check that you can get a GPU for Week 4 (a rented 24 GB card or Colab is enough) and that your billing alarms from Week 1 are still on.
