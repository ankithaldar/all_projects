# Week 1: Dev scaffold and PyTorch core

*Companion to the 26-week plan, Phase A. 15 hours: 5 code-along, 6 build, 2 hardening, 1 cheat sheet, 1 review. This file contains no code, only exercises with acceptance criteria. Links were checked on 11 Nov 2026; a ✓ means the page was confirmed live in search that day.*

## 1. Goal and deliverables

By the end of Week 1 you can:

1. Stand up a reproducible, linted, tested, containerized Python project with CI.
2. Explain and implement backpropagation from scratch.
3. Show that your implementation agrees with PyTorch autograd.
4. Spend on cloud only after cost guardrails are in place.

**Deliverables**

- **Repo A, `ml-template` (tagged v0.1):** the skeleton every later project starts from.
- **Repo B, `week01-autograd`, created from Repo A:** your autograd engine, MLP, PyTorch parity checks, tests and plots.
- **`docs/guardrails.md`:** accounts, MFA, budgets, teardown rules.
- **`notes/`:** your CS1 write-up, a micrograd-to-PyTorch mapping, and a log of where you got stuck.

Section 7 has the full exit criteria.

## 2. Two cloud decisions to make before you spend a minute

The main plan’s Week 1 row says to create all three accounts and set billing alarms. Having checked the current free-credit rules, follow this section instead, because two credit clocks would otherwise run out before you need them.

**GCP: do not start the free trial this week.** The trial gives $300 in credit that lasts 90 days from signup. The trial billing account closes at 90 days or when the credit is spent, whichever comes first, and unused credit survives an upgrade to a paid account only until the 90 days are up. Sign up in Week 1 and the credit is gone around Week 13, before the GCP weeks (16, 19, 22, 25). Plan to start it in Week 15. This week you only write the GCP guardrail plan. New projects typically start with little or no GPU quota, so also plan to request GPU quota in Weeks 14–15. Sources: [Google Cloud free program](https://docs.cloud.google.com/free/docs/free-cloud-features) ✓ and [Free Trial FAQs](https://cloud.google.com/signup-faqs) ✓.

**AWS: choose the plan deliberately.** Accounts created since 15 July 2025 pick a Free plan or a Paid plan. Both come with $100 in credits plus up to $100 more for completing activities; setting up a cost budget is one of them. The Free plan ends after 6 months or when the credits run out, whichever is first, and then closes the account, which would land around your capstone. Its credits also apply only to services eligible under that plan. Credits expire 12 months after account creation on either plan. Because Weeks 15–25 use EKS, SageMaker and GPU instances, the Paid plan with a strict budget is the safer choice. I could not confirm from the sources which of those services the Free plan covers, so check AWS’s eligible-services list before relying on it. A write-up on AWS Builder Center also warns that enabling AWS Organizations, Control Tower or IAM Identity Center converts a Free-plan account to Paid and expires the credits, so avoid those until you mean to upgrade. It adds that AWS Budgets nets credits against cost by default, so a zero-spend budget stays quiet while credits drain; exclude credits when you create the budget. Sources: [AWS hands-on guide: control your costs](https://docs.aws.amazon.com/hands-on/latest/control-your-costs-free-tier-budgets/control-your-costs-free-tier-budgets.html) ✓, [The $0 AWS account in 2026](https://builder.aws.com/content/3Kkt4kNiDSUnO3n6db2weSNubzE/the-dollar0-aws-account-in-2026-free-plan-credits-guardrails) ✓, [earning additional credits](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans-activities.html) ✓.

## 3. Everything to cover this week

| Area | Topics | Hours |
| --- | --- | --- |
| Engineering foundations | src project layout; dependency management and lockfiles (uv); linting and formatting (ruff); testing (pytest); pre-commit hooks; task runner; README conventions | 1, 3, 13 |
| Cloud and tracking guardrails | MFA; non-root admin identity; budgets and alerts; credit clocks; W&B account, login and first run | 2, 12 |
| Calculus you will use in code | derivative as sensitivity; numerical derivatives; chain rule; local gradients for add, multiply, tanh, exp, power; gradient accumulation when a node is reused | 4–6 |
| Autograd mechanics | computation graph; topological order; backward pass; gradient accumulation; gradient reset | 5–7, 9 |
| Neural-net building blocks | neuron, layer, MLP; parameters; loss; gradient descent; learning-rate effects; the training loop | 6–7, 10 |
| PyTorch core | tensors, dtype, device; nn.Module; autograd; optimizers; DataLoader basics; train and eval modes; no-grad; checkpointing | 8, 12 |
| Verification habits | numerical gradient check; parity test against a reference; overfit one batch; initial-loss sanity check; seeded determinism; checkpoint-resume equality | 11–13 |
| Hardening | pytest suite; type hints and type checking; coverage; Docker image with layer caching and non-root user; CI; branch protection | 13–14 |
| Habits | notes, README, one ADR, the weekly checklist | 15 |

## 4. Resource library for Week 1

**Videos and code to replicate**

| Resource | Type | Time | Used in |
| --- | --- | --- | --- |
| [Karpathy: The spelled-out intro to neural networks and backpropagation, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) ✓ | Video (2 h 25 m) | primary code-along | Hours 4–7 |
| [Neural Networks: Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) ✓ | Playlist; this week is lecture 1 only | context | Hour 4 |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) ✓ | GitHub repo | reference solution; read after you finish | Hours 4, 7, 10 |
| [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) ✓ | GitHub repo with lecture notebooks | unblock yourself when stuck | Hours 4–7 |

**PyTorch reading and tutorials** (all pages confirmed on the official index)

| Resource | Type | Time | Used in |
| --- | --- | --- | --- |
| [Learn the Basics: Tensors](https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html) ✓ | Tutorial | 15 min | Hour 8 |
| [Learn the Basics: Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) ✓ | Tutorial | 15 min | Hour 8 |
| [Learn the Basics: Automatic differentiation](https://pytorch.org/tutorials/beginner/basics/autograd_tutorial.html) ✓ | Tutorial | 15 min | Hours 8, 11 |
| [Learn the Basics: Optimizing model parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) ✓ | Tutorial | 15 min | Hours 8, 12 |
| [Learn the Basics: Datasets and DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) ✓ | Tutorial | 10 min | Hour 12 |
| [Learn the Basics: Save and load the model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) ✓ | Tutorial | 10 min | Hour 12 |
| [Learn the Basics: Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) ✓ | Tutorial, optional skim | 10 min | Hour 8 |
| [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) | Reading, optional | 15 min | Hour 8 |
| [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html) | Reading | 10 min | Hour 12 |
| [CS231n: backprop intuitions](https://cs231n.github.io/optimization-2/) | Reading, optional | 15 min | Hour 4 |
| [CS231n: gradient checks](https://cs231n.github.io/neural-networks-3/) | Reading (gradient-check section only) | 10 min | Hour 11 |
| [scikit-learn make\_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) | API doc | 5 min | Hour 10 |

**Tooling documentation**

| Resource | Type | Used in |
| --- | --- | --- |
| [uv: working on projects](https://docs.astral.sh/uv/guides/projects/) ✓ and [project concepts](https://docs.astral.sh/uv/concepts/projects/) ✓ | Docs | Hours 1, 3 |
| [uv with Docker](https://docs.astral.sh/uv/guides/integration/docker/) and [uv with GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/); [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) ✓ | Docs and action | Hour 14 |
| [ruff](https://docs.astral.sh/ruff/) | Docs | Hours 3, 13 |
| [pytest getting started](https://docs.pytest.org/en/stable/) | Docs | Hours 3, 13 |
| [pre-commit](https://pre-commit.com/); optional: [Made With ML pre-commit lesson](https://madewithml.com/courses/mlops/pre-commit/) | Docs and lesson | Hour 3 |
| [src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) | Reading | Hour 3 |
| [Docker get started](https://docs.docker.com/get-started/) and [build best practices](https://docs.docker.com/build/building/best-practices/); [TechWorld with Nana](https://www.youtube.com/@TechWorldwithNana) for a video walkthrough | Docs and video | Hours 1, 14 |
| [GitHub Actions docs](https://docs.github.com/en/actions) and [protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) | Docs | Hour 14 |

**Cloud and tracking**

| Resource | Used in |
| --- | --- |
| [AWS hands-on: control your costs](https://docs.aws.amazon.com/hands-on/latest/control-your-costs-free-tier-budgets/control-your-costs-free-tier-budgets.html) ✓; [The $0 AWS account in 2026](https://builder.aws.com/content/3Kkt4kNiDSUnO3n6db2weSNubzE/the-dollar0-aws-account-in-2026-free-plan-credits-guardrails) ✓; [AWS IAM best practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html) | Hour 2 |
| [Google Cloud free program](https://docs.cloud.google.com/free/docs/free-cloud-features) ✓; [Cloud Billing budgets](https://cloud.google.com/billing/docs/how-to/budgets) | Hour 2 (reading only) |
| [Weights & Biases docs](https://docs.wandb.ai/) ✓ (Models quickstart and the PyTorch integration page) | Hours 2, 12 |

## 5. Coding exercises

These are the exercises that complete Week 1. Do them in order. Each has acceptance criteria you can check without me.

| ID | Exercise | Acceptance criteria | Hour |
| --- | --- | --- | --- |
| E1 | **Fresh-clone smoke test.** On a clean directory, clone Repo A, build the environment from the lockfile alone, then run the linter and the tests. | Works with no manual steps in under 5 minutes; the README lists the three commands you used | 3 |
| E2 | **Scaffold with guardrails.** Src-layout package; lint, format, type-check and test settings in the project file; pre-commit running ruff lint and format plus hygiene and secret-detection hooks; a task runner with setup, lint, format and test targets. | A deliberately badly formatted commit is blocked by pre-commit; every target runs from a clean clone | 3 |
| E3 | **Numerical derivatives.** Write a finite-difference derivative for a single-input function. Compare forward and centered differences across step sizes from large to very small and tabulate the error. | Table plus a three-sentence note on why too-small steps get worse (floating-point rounding) | 4 |
| E4 | **Manual backprop.** Pick an expression with at least three operations and one reused variable. Write each local derivative, apply the chain rule by hand, then verify every gradient with E3. | All gradients agree with the numerical ones to within 1e-6 relative error | 4 |
| E5 | **Value engine (code-along, then extend).** A scalar node holding data, gradient, parents and operation; add, multiply and tanh from the video; then exp, power, negate, subtract and divide; backward in topological order with correct accumulation for reused nodes. | The video’s example reproduces its gradients and matches your E4 hand results | 5–6 |
| E6 | **Neuron, layer, MLP and training (code-along).** Build them on your engine; train on the video’s tiny four-sample dataset with a squared-error loss and manual gradient descent. Then deliberately forget to reset gradients between steps. | Loss falls near zero with correct code; you have written down the symptom of the missing reset | 6–7 |
| E7 | **Rebuild from a blank file.** Close the video. In 60 minutes recreate the engine, the MLP and a smoke test as library code under src, using only your own notes. | Reproduces the E6 results; your notes list every point where you got stuck | 9 |
| E8 | **Train on a real toy dataset.** Two-class, two-feature dataset with interleaving half-circles plus noise; an MLP with two hidden layers; a margin-based or logistic loss with a small L2 penalty. Plot the loss curve and the decision boundary. | At least 95% training accuracy; both plots saved under docs | 10 |
| E9 | **Gradient-check harness.** A reusable checker that compares your analytic gradients against centered finite differences for each primitive op and for every parameter of a small MLP. Choose and justify a relative-error threshold using the CS231n guidance. | Everything passes; then break one op’s backward rule on purpose and show the checker catches it | 11 |
| E10 | **Extend the engine.** Add ReLU, sigmoid (built from exp), log and one non-smooth op such as absolute value. Add an evaluation switch that skips graph building. | Every new op passes E9; evaluation mode demonstrably builds no graph | 11 |
| E11 | **PyTorch parity.** Build the identical MLP with torch.nn in double precision, load weights exported from your engine, and run forward and backward on the same batch. Then train both for 100 plain gradient-descent steps. | Outputs and all parameter gradients agree to within 1e-9; loss curves overlay to within 1e-6 | 12 |
| E12 | **Hand-written PyTorch training loop.** Explicit steps with no high-level trainer: seeded DataLoader, train and eval modes, no-grad evaluation, optional gradient clipping, per-step loss logging, checkpoint save and resume. Log one run to W&B. | A resumed run reproduces the loss trajectory of an uninterrupted run within tolerance; the W&B run link is in the README | 12 |
| E13 | **Test suite.** Unit tests for ops and graph; gradient-check tests; shape and dtype tests for the PyTorch model; a seeded determinism test; an initial-loss test (roughly ln of the class count at random init); an overfit-one-batch test. Add type hints and a coverage report. | Suite runs on CPU in under 5 minutes; ruff and the type checker are clean | 13 |
| E14 | **Container.** Dockerfile for the template: pinned Python base, dependency layer before source for caching, non-root user, a .dockerignore. Run the tests inside the container. | Builds from clean; tests pass in the container; image size recorded in the README; a code-only change reuses the cached dependency layer | 14 |
| E15 | **CI.** GitHub Actions workflow on pull request and push: lint, type-check, tests with dependency caching, and a Docker build without push. Require it on main via branch protection. | Green run under 7 minutes; a PR with a failing test cannot be merged; status badge in the README | 14 |
| E16 | **Guardrails evidence.** AWS root MFA, a non-root admin identity and a monthly budget with email alerts and credits excluded; a written GCP plan; a W&B account with a first logged run. | `docs/guardrails.md` with screenshots, the credit expiry dates and the planned GCP start (Week 15) | 2, 12 |

## 6. Hour-by-hour plan

### Schedule at a glance

| Hour | Focus | Block |
| --- | --- | --- |
| 1 | Orient, cheat sheet CS1, toolchain | Cheat sheet 30 min + Review slot 30 min (setup) |
| 2 | Cost guardrails and W&B account | Build |
| 3 | Scaffold the `ml-template` repo | Build |
| 4 | micrograd part 1: derivatives, Value object, manual backprop | Code-along |
| 5 | micrograd part 2: backward for every operation | Code-along |
| 6 | micrograd part 3: more ops, PyTorch comparison, start of MLP | Code-along |
| 7 | micrograd part 4: training and the zero-grad bug | Code-along |
| 8 | PyTorch core tutorials | Code-along |
| 9 | Rebuild the engine from a blank file | Build |
| 10 | Train on a real toy dataset | Build |
| 11 | Gradient checking and extending the engine | Build |
| 12 | PyTorch parity and a hand-written training loop | Build |
| 13 | Tests and types | Hardening |
| 14 | Docker and CI | Hardening |
| 15 | Review, write-up and self-test | Cheat sheet 30 min + Review 30 min |

This week the Review slot absorbs 30 minutes of tool setup because there is nothing to review yet. Overall the week still totals 5 h code-along, 6 h build, 2 h hardening, 1 h cheat sheet and 1 h review.

### Cheat sheet CS1: autograd and the training loop (read in Hour 1)

- **Mental model:** the forward pass records a graph; calling backward fills the gradient on every parameter; the optimizer step consumes those gradients.
- **Loop order:** reset gradients, forward, loss, backward, optional clipping, step, scheduler step.
- **Gotchas:** forgetting to reset gradients; validating without eval mode and no-grad; pulling a Python number out of a tensor every step (forces a GPU sync); in-place operations on tensors autograd still needs.
- **Numbers:** initial cross-entropy should be about ln(number of classes) (ln 50,257 is about 10.8 for a GPT-2 sized vocabulary); if it isn’t, initialization or labels are off. Prove any loop by overfitting one batch.
- **From micrograd:** a node used twice must accumulate gradient from both uses; backward walks the graph in reverse topological order.

### Hour 1 · Orient, CS1 and toolchain

- **Focus:** know the week’s shape and have a working toolchain.
- **Materials:** CS1 above; [uv: working on projects](https://docs.astral.sh/uv/guides/projects/) ✓; [Docker get started](https://docs.docker.com/get-started/).
- **Do:**
  1. (10 min) Read Sections 1–3 and 7 of this file. Put the 15 hours in your calendar.
  2. (20 min) Read CS1. Write down three questions about it that you can’t yet answer; you will answer them in Hour 15.
  3. (30 min) Install and verify: Python through uv, git (identity set, GitHub authentication working), Docker (a hello-world container runs), and an editor with Python and ruff support. Decide local vs Colab for the code-along; this week needs CPU only.
- **Checkpoint:** every tool reports a version, and a throwaway repo pushes to GitHub.

### Hour 2 · Cost guardrails and W&B (Build)

- **Focus:** nothing can silently spend money, and experiment tracking is ready.
- **Materials:** Section 2 above; [AWS hands-on: control your costs](https://docs.aws.amazon.com/hands-on/latest/control-your-costs-free-tier-budgets/control-your-costs-free-tier-budgets.html) ✓; [The $0 AWS account in 2026](https://builder.aws.com/content/3Kkt4kNiDSUnO3n6db2weSNubzE/the-dollar0-aws-account-in-2026-free-plan-credits-guardrails) ✓; [AWS IAM best practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html); [Google Cloud free program](https://docs.cloud.google.com/free/docs/free-cloud-features) ✓; [W&B docs](https://docs.wandb.ai/) ✓.
- **Do:**
  1. (25 min) AWS: create the account (plan choice per Section 2), turn on MFA for the root user, create a non-root admin identity for daily use, and create a monthly cost budget with email alerts at 50%, 80% and 100% with credits excluded. Note the credit expiry date.
  2. (10 min) GCP: do not sign up for the trial. In your guardrails file write the planned start (Week 15), the budget thresholds you will set on day one, and the GPU-quota request you will file in Weeks 14–15.
  3. (15 min) W&B: create a free personal account, find your API key, authenticate your local tooling once, and keep the key in a password manager rather than a repo.
  4. (10 min) Draft `docs/guardrails.md`: accounts, MFA status, budgets with screenshots, credit clocks, and the teardown rule (nothing running at the end of a session).
- **Exercise:** E16 (AWS and W&B parts now; the first logged W&B run comes in Hour 12).
- **Checkpoint:** the budget is visible with alerts configured, root MFA is on, and the guardrails file is committed.

### Hour 3 · Scaffold the template repo (Build)

- **Focus:** a reusable `ml-template` that every later project starts from.
- **Materials:** [uv: working on projects](https://docs.astral.sh/uv/guides/projects/) ✓ and [project concepts](https://docs.astral.sh/uv/concepts/projects/) ✓; [src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/); [ruff](https://docs.astral.sh/ruff/); [pytest](https://docs.pytest.org/en/stable/); [pre-commit](https://pre-commit.com/); optional [Made With ML pre-commit lesson](https://madewithml.com/courses/mlops/pre-commit/).
- **Do:**
  1. (10 min) Create Repo A on GitHub. Initialize a uv project with a lockfile, a src-layout package, one placeholder module and one trivial test.
  2. (15 min) Configure ruff (lint and format), a type checker and pytest in the project file.
  3. (15 min) Add pre-commit with ruff lint, ruff format, whitespace and end-of-file fixers, a large-file guard and a secret-detection hook.
  4. (10 min) Add a task runner with setup, lint, format and test targets.
  5. (10 min) Write the README (what it is, the three commands to get going) and tag v0.1.
- **Exercises:** E2, then E1 as the last step.
- **Checkpoint:** a badly formatted commit is blocked, and a fresh clone passes in under 5 minutes.

### Hour 4 · micrograd part 1: derivatives and the Value object (Code-along)

- **Focus:** a derivative is a sensitivity; build the first expression graph and backpropagate through it by hand.
- **Materials:** [the micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0) ✓, roughly the first third (segments: intro, micrograd overview, derivatives of simple functions, the core Value object, manual backprop on a simple expression, manual backprop through a neuron; names follow YouTube’s chapter list); [karpathy/micrograd](https://github.com/karpathy/micrograd) ✓; [nn-zero-to-hero notebooks](https://github.com/karpathy/nn-zero-to-hero) ✓ for when you’re stuck; optional [CS231n backprop intuitions](https://cs231n.github.io/optimization-2/).
- **Do:**
  1. (5 min) Skim the micrograd README so you can see the target.
  2. (50 min) Code along. Type every cell yourself and never paste from the notebook. Pause before each result and predict the number.
  3. (5 min) Write three bullets: what a local derivative is, what the gradient on a node means, what the chain rule does in the graph.
- **Exercises:** E3 during the derivatives segment; E4 as your own version of the manual-backprop segment.
- **Checkpoint:** the gradients you derived by hand for your own expression match the numerical ones.

### Hour 5 · micrograd part 2: backward for every operation (Code-along)

- **Focus:** automate backpropagation across the whole graph.
- **Materials:** the video’s next stretch (implementing backward for each operation, then for a whole expression graph in topological order, then the fix for a node used multiple times).
- **Do:**
  1. (45 min) Code along through those segments. Your pace sets the exact boundary; what matters is finishing E5’s first half by the end of Hour 6.
  2. (10 min) Deliberately reintroduce the reused-node bug (overwriting instead of accumulating) and write down which expression exposes it.
  3. (5 min) Notes: why topological order, and why accumulate.
- **Exercise:** E5, first half (add, multiply, tanh, topological backward).
- **Checkpoint:** the video’s example expression gives the same gradients you computed by hand in Hour 4.

### Hour 6 · micrograd part 3: more operations, PyTorch comparison, MLP (Code-along)

- **Focus:** richer operations, an external reference, and the start of a neural net.
- **Materials:** the video’s segments on breaking tanh into primitive operations, comparing with PyTorch, and building a neuron, a layer and an MLP; the [PyTorch autograd tutorial](https://pytorch.org/tutorials/beginner/basics/autograd_tutorial.html) ✓ as a preview.
- **Do:**
  1. (25 min) Finish the remaining operations: exp, power, negate, subtract, divide (built from your existing ones where the video does so).
  2. (10 min) Compare your gradients with PyTorch’s on a second expression of your own, not only the video’s.
  3. (25 min) Begin Neuron, Layer and MLP.
- **Exercises:** E5 second half, then start E6.
- **Checkpoint:** an MLP forward pass runs and you can collect all its parameters.

### Hour 7 · micrograd part 4: training and the zero-grad bug (Code-along)

- **Focus:** loss, parameters, gradient descent, and the classic bug.
- **Materials:** the rest of the video: tiny dataset and loss, collecting parameters, gradient descent by hand; the bug segment near the 2 h 11 m mark ✓; then the full-source walkthrough. If you’re short on time, watch the final walkthrough and the PyTorch tanh-backward segment at 1.5× and read them again in Hour 10.
- **Do:**
  1. (30 min) Code along through training.
  2. (10 min) Vary the learning rate across four orders of magnitude and note where training diverges or crawls.
  3. (10 min) Reproduce the forgotten gradient reset and record exactly what you observe.
  4. (10 min) Read micrograd’s own source and list how its design differs from yours.
- **Exercise:** E6.
- **Checkpoint:** loss on the four-sample dataset is near zero, and your notes describe the missing-reset symptom.

### Hour 8 · PyTorch core (Code-along)

- **Focus:** map what you just built onto PyTorch.
- **Materials:** Learn the Basics pages for [Tensors](https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html) ✓, [Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) ✓, [Automatic differentiation](https://pytorch.org/tutorials/beginner/basics/autograd_tutorial.html) ✓ and [Optimizing model parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) ✓; optional [Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) ✓ and [autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html).
- **Do (about 15 minutes per page, using the page’s Colab button or local):**
  1. Tensors: predict dtype and device behavior before running each cell.
  2. Build the Neural Network: count each layer’s parameters by hand, then verify.
  3. Automatic differentiation: reproduce your Hour 4 hand-computed gradient in PyTorch.
  4. Optimizing model parameters: identify the loop’s steps and map each to your micrograd training loop.
- **Deliverable:** `notes/pytorch_map.md`, a table from micrograd concept to PyTorch concept (value node to tensor with gradients, backward, MLP to nn.Module, manual update to optimizer step, gradient reset).
- **Checkpoint:** the mapping table exists and you can explain each row.

### Hour 9 · Rebuild from a blank file (Build)

- **Focus:** the Rebuild step of Replicate, Mutate, Rebuild.
- **Materials:** your own notes only; open the micrograd repo only after being stuck for 10 minutes, and log each peek.
- **Do:** work in Repo B (created from the template), in library code under src rather than a notebook. Timebox 60 minutes: engine with operations and backward, then Neuron, Layer, MLP, then a smoke test. Log every stuck point.
- **Exercise:** E7.
- **Checkpoint:** the Hour 7 results reproduce from library code.

### Hour 10 · Train on a real toy dataset (Build)

- **Focus:** apply your engine to something that is not four numbers.
- **Materials:** [scikit-learn make\_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html); afterwards compare with micrograd’s [demo notebook](https://github.com/karpathy/micrograd/blob/master/demo.ipynb).
- **Do:**
  1. (10 min) Generate and standardize the dataset; hold out a test split.
  2. (30 min) Train the two-hidden-layer MLP on your engine.
  3. (15 min) Produce the loss-curve and decision-boundary plots.
  4. (5 min) Notes: what changed when you altered hidden width and learning rate.
- **Exercise:** E8.
- **Checkpoint:** at least 95% training accuracy, with both plots saved under docs.

### Hour 11 · Gradient checking and extending the engine (Build)

- **Focus:** learn to distrust and verify your own gradients.
- **Materials:** [CS231n gradient checks](https://cs231n.github.io/neural-networks-3/) (the gradient-check section); [PyTorch autograd docs](https://pytorch.org/docs/stable/autograd.html) for how a reference checker is framed.
- **Do:**
  1. (35 min) Build the checker: centered differences, relative error with a floor in the denominator, a justified threshold. Break one backward rule on purpose and confirm the checker flags it.
  2. (25 min) Add the new operations and the evaluation switch.
- **Exercises:** E9, E10.
- **Checkpoint:** everything passes, and the deliberate break was caught.

### Hour 12 · PyTorch parity and a hand-written training loop (Build)

- **Focus:** prove your engine agrees with PyTorch, then write the loop you will reuse for months.
- **Materials:** [Optimizing model parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) ✓; [Datasets and DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) ✓; [Save and load the model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) ✓; [reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html); [W&B docs](https://docs.wandb.ai/) ✓.
- **Do:**
  1. (25 min) E11: export your weights, rebuild the model in double precision, compare outputs and gradients, then compare 100 training steps.
  2. (30 min) E12: write the loop with explicit steps, checkpoint save and resume, and per-step logging.
  3. (5 min) Log one run to W&B and put the link in the README.
- **Exercises:** E11, E12, and the W&B part of E16.
- **Checkpoint:** parity numbers recorded, the resumed run matches the uninterrupted one, and the W&B link is in the README.

### Hour 13 · Tests and types (Hardening)

- **Focus:** turn the week’s scripts into a trustworthy suite.
- **Materials:** [pytest](https://docs.pytest.org/en/stable/); [ruff](https://docs.astral.sh/ruff/).
- **Do:** write the E13 test suite in this order: ops and graph unit tests; gradient-check tests; model shape and dtype tests; seeded determinism; initial-loss check; overfit-one-batch. Add type hints to public functions, run the type checker, and produce a coverage report. Keep every test CPU-only and fast.
- **Exercise:** E13.
- **Checkpoint:** the suite runs under 5 minutes on CPU, with ruff and the type checker clean.

### Hour 14 · Container and CI (Hardening)

- **Focus:** the same checks run identically on your machine, in a container and in CI.
- **Materials:** [uv with Docker](https://docs.astral.sh/uv/guides/integration/docker/); [uv with GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/); [setup-uv](https://github.com/astral-sh/setup-uv) ✓; [Docker build best practices](https://docs.docker.com/build/building/best-practices/); [GitHub Actions docs](https://docs.github.com/en/actions); [protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
- **Do:**
  1. (30 min) E14 in Repo A, then pull the changes into Repo B.
  2. (30 min) E15 in both repos. If you overrun, drop E14’s cache-reuse check, not CI.
- **Exercises:** E14, E15.
- **Checkpoint:** the container passes the tests, CI is green, and a failing PR is blocked from merging.

### Hour 15 · Review, write-up and self-test

- **Focus:** consolidate and close the week.
- **Do:**
  1. (10 min) Write `notes/cs1.md`: your own five-line version of the cheat sheet.
  2. (20 min) Answer your three Hour 1 questions and the six self-test questions in Section 7, out loud or in notes, without looking anything up. Re-watch only for the ones you miss.
  3. (15 min) Run the Section 8 checklist and fix anything quick.
  4. (10 min) Finish Repo B’s README (what, why, how to run, results table with parity numbers and accuracy, plots) and write one ADR (for example, why double precision for parity checks).
  5. (5 min) Tag v0.1 on both repos, push, and open Week 2’s first resources so you can start cold.
- **Checkpoint:** every box in Section 7 is checked.

## 7. Exit criteria for Week 1

Week 1 is done only when all of the following are true.

1. **Repo A (`ml-template`, v0.1):** fresh-clone setup in under 5 minutes; pre-commit blocks a bad commit; CI is green; the Docker image builds and runs the tests.
2. **Repo B (`week01-autograd`):** engine and MLP as tested library code under src; gradient check passes; PyTorch parity within the E11 tolerances; at least 95% accuracy on the toy dataset with plots; checkpoint-resume equality; a W&B run linked.
3. **Guardrails:** `docs/guardrails.md` with AWS root MFA and a credits-excluded budget (screenshots), the credit expiry dates, the GCP plan with a Week 15 start, and a working W&B login.
4. **Notes:** `cs1.md`, `pytorch_map.md` and the rebuild stuck-points log.
5. **Self-test, answered without notes:**
   - Why must gradients accumulate rather than overwrite at a node used more than once? Give a concrete expression.
   - Why reset gradients between steps, and what do you see if you don’t?
   - Why is a centered finite difference better than a forward difference, and why does an extremely small step hurt?
   - Why compare against PyTorch in double precision?
   - What should the initial loss be for a K-class classifier at random initialization, and what does a different value tell you?
   - What does eval mode change, and what does no-grad change, separately?

## 8. Weekly review checklist

**Every week**

- Typed code; ruff and the type checker clean; no logic that lives only in a notebook.
- Shape, dtype and device tests; a seeded determinism test; an overfit-one-batch test; suite under 5 minutes on CPU.
- No hard-coded paths or hyperparameters; secrets only from the environment or a secret manager.
- Pinned lockfile; seeds set; one command reproduces the headline result.
- Every run logs params, metrics, environment and artifacts; failed runs are tagged, not deleted.
- Docker image builds in CI, runs as non-root, has a .dockerignore and a pinned base.
- CI runs lint, test and build on every PR; main is protected; a status badge is in the README.
- Structured logs for anything served (not yet relevant this week).
- Teardown verified, budget alarms on, least-privilege IAM, nothing public without auth.
- README says what, why and how to run, with a results table; one ADR; a screenshot or plot.

**Week 1 add-ons**

- The template repo clones and passes CI from scratch in under 10 minutes.
- Pre-commit blocks a bad commit (demonstrate it once and keep the evidence).
- A billing-alarm screenshot is saved in `docs/`, with credits excluded from the budget.

## 9. Pitfalls and optional extras

**Pitfalls**

- **Pasting from the repo notebook.** It feels fast and teaches nothing. Type it, predict the next output, and only peek after 10 minutes.
- **Watching passively in Hours 4–7.** If you haven’t paused to predict at least five times an hour, slow down.
- **Over-engineering the template.** Keep only what Week 2 needs; every extra hook is an hour you don’t have.
- **Letting Docker and CI eat Hour 15.** Use the timeboxes; unfinished polish moves to Week 2’s Hardening block.
- **Floating-point confusion in gradient checks.** Use double precision, centered differences, and a relative error with a floor in the denominator.

**Optional extras (only if you finish early)**

- Read [CS231n backprop intuitions](https://cs231n.github.io/optimization-2/) to see the add, multiply and max gate patterns.
- Read [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) and note how it differs from your engine.
- Make your engine operate on small arrays instead of scalars; the video deliberately leaves tensors out, so this is the bridge to how PyTorch actually works.
