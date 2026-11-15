# Week 01 — PyTorch Foundations

This folder contains Week 01 PyTorch Foundations study materials: comprehensive notes, executable notebooks (Sections 1–15 curriculum), and generated report artifacts (plots/loss curves).

## Contents

### Notes
- `notes/001_notes.md` — **Week 01 — PyTorch Foundations (Comprehensive Study Notes)**
  - Includes a table of contents and study-topic coverage for **Sections 1–15** (PyTorch/NumPy foundations, tensor basics, indexing/shape manipulation, arithmetic/reductions, broadcasting, memory/views/contiguity/devices, autograd fundamentals, backpropagation/gradient mechanics, gradient verification/debugging, linear regression, training/validation, optimization experiments, logistic regression, classification metrics/evaluation, and assessment/review).
  - Includes Mermaid diagrams/mind maps/flowcharts across topics.
  - Authoritative artifacts/sections:
    - `## Autograd One-Page Cheat Sheet (Section 9)` (Mermaid.js, single-source)
    - `## Repository Review Checklist (Section 15C)`
    - `## Week 1 Cheat Sheets (Section 15D)` with `### 1. Tensor Operations and Broadcasting`, `### 2. Autograd and Backpropagation` (aligned with Section 9 cheat sheet), and `### 3. Gradient Descent and Optimization`

### Notebooks
- `notebooks/01_tensor_fundamentals.ipynb` — **Sections 1–2**
  - CUDA availability check and device selection (**CUDA → MPS → CPU**), NumPy array vs PyTorch tensor property comparison (including N/A semantics for NumPy device/`requires_grad`).
  - Tensor creation (scalar/vector/matrix/3D), zeros/ones (`*_like`), uniform/normal/integer random tensors, ranges (`arange`, `linspace`), inspection (shape/rank/dtype/device), float/int dtype experiments, and shape/dimension interpretation examples.

- `notebooks/02_tensor_manipulation.ipynb` — **Sections 3–5**
  - 10×10 matrix (1–100) indexing/slicing/submatrix extraction, valid reshapes, vector↔row/column conversion, explicit singleton-dimension setup and squeeze/unsqueeze, transpose, concatenation, stacking, and valid/invalid reshape explanations with caught non-fatal invalid examples.
  - Tensor arithmetic/reductions: element-wise ops, matrix multiplication, dot product, matrix-vector multiplication demo, row/column sums, mean, min/max along dimensions, argmax per row, and NumPy↔PyTorch numerical parity verification with explicit `np.allclose`/`torch.allclose` tolerances and visible confirmations.
  - Broadcasting experiments with predicted shapes: scalar+matrix, vector to every row, column vector to every column, compatible element-wise multiplication, 3D broadcasting, **5 valid + 5 invalid** examples (invalid cases caught with predicted outcome and exception type/message) and explanations.

- `notebooks/03_autograd.ipynb` — **Sections 6–8**
  - Memory/device/views: views vs copies, contiguity/non-contiguous tensors and memory layout, reshape/view behavior, in-place operation consequences, CPU/GPU transfers (when available), device compatibility checks, environment-safe caught incompatible/unavailable-device demo, and basic GPU memory awareness (conditional).
  - Autograd fundamentals: derivatives/partial derivatives concepts, computational graphs, forward computation, `requires_grad`, leaf/non-leaf tensors, tracked operations, disabled tracking, and Mermaid graph drawings.
  - Backpropagation/gradient mechanics: chain rule, `.backward()`, `.grad`, gradient accumulation and clearing, repeated backward behavior, `torch.no_grad()`, `torch.detach()`, and forward/backward graph drawings.

- `notebooks/04_linear_regression.ipynb` — **Sections 10–12**
  - From-scratch synthetic linear regression (1 feature, known coefficient/intercept, noise): dataset generation/inspection, train/validation split, trainable raw parameters, prediction, MSE loss, manual gradient descent update process.
  - Complete training pipeline (forward, loss, backward, manual update in `torch.no_grad()`, gradient clearing, epochs, train/val tracking, learned vs known parameter comparison).
  - Controlled experiments **A–F** (very small LR, reasonable LR, excessively large LR, different initialization, different dataset noise, different sample sizes) with recorded configs, train/val losses, saved loss curves, parameter comparisons, and explanations.
  - **From-scratch boundary:** no `sklearn` LinearRegression, no `torch.optim.*`, no `torch.nn` modules/losses/parameters, no `torch.nn.functional` loss/objective helpers, no `torch.utils.data` training abstractions.

- `notebooks/05_logistic_regression.ipynb` — **Sections 13–14**
  - From-scratch binary classification (2 features): synthetic data generation/visualization, train/validation split, raw trainable weights/bias, logits, sigmoid probabilities, numerically stable logits-based BCE (explicit tensor math), gradients, and manual GD updates.
  - Evaluation: validation predictions, accuracy/precision/recall/F1, confusion matrix counts, threshold experiments (0.3/0.5/0.7), decision boundary plot, and train vs validation performance comparison.
  - **From-scratch boundary:** no sklearn classifiers/metrics helpers, no `torch.optim.*`, no `torch.nn` modules/losses/parameters, no `torch.nn.functional` loss/objective helpers, no `torch.utils.data` training abstractions.

- `notebooks/06_gradient_verification.ipynb` — **Section 9**
  - Analytical derivatives, numerical finite-difference approximations, and PyTorch autograd comparisons for multiple functions; step-size and floating-point precision experiments.
  - Accidental gradient accumulation experiment with diagnosis/fix.
  - Includes `## Autograd One-Page Cheat Sheet (Section 9)` (Mermaid.js) as an identical copy of the authoritative notes source.

- `notebooks/07_linear_regression_fresh_assessment.ipynb` — **Section 15A (fresh notebook)**
  - Independent from-scratch linear regression implementation in a separate fresh notebook: generate data, initialize parameters, calculate predictions, calculate loss, backpropagate, update parameters, and evaluate the final model.
  - **From-scratch boundary:** no sklearn LinearRegression, no `torch.optim.*`, no `torch.nn` modules/losses/parameters, no `torch.nn.functional` loss/objective helpers, no `torch.utils.data` training abstractions.

- `notebooks/08_week1_assessment.ipynb` — **Sections 15B–D**
  - **Exercise B:** debugging challenge demos/fixes (forgetting to clear gradients, inappropriate learning rate, incompatible tensor shapes, tensors on different devices, accidentally disabled gradient tracking) with safe caught/diagnosed examples.
  - **Exercise C:** repository review checklist with recorded close-out evidence (aligned with notes checklist).
  - **Exercise D:** three one-page cheat sheets (Tensor operations and broadcasting; Autograd and backpropagation; Gradient descent and optimization), with autograd cheat sheet aligned to the authoritative notes/Section 9 source.

### Reports (generated artifacts)
- `reports/section10_linear_data.png`
- `reports/section11_linear_training_loss.png`
- `reports/section12_exp_01_loss.png` … `reports/section12_exp_06_loss.png`
- `reports/section13_classes.png`
- `reports/section13_training_loss.png`
- `reports/section14_decision_boundary.png`
- `reports/section15a_fresh_linear_regression.png`

> Note: no `experiments/` directory is present in this folder; image artifacts delivered with the curriculum are under `reports/`.

## Reproduction

### Environment
- Executed with Python 3.15.x, `numpy`, `torch`, `matplotlib`, `nbformat`, `nbconvert` (and Jupyter kernel available).
- Device selection in notebooks follows **CUDA → MPS → CPU** with robust checks; CPU-only environments are supported. Intentional error demonstrations (invalid reshape/broadcast, incompatible/unavailable device, autograd/debug cases) are wrapped in `try/except` and print predicted outcomes with exception type/message so notebooks run end-to-end.
- Reproducibility aids used where applicable: fixed seeds (`torch.manual_seed`, `np.random.seed`), recorded configs, and deterministic artifact paths under `reports/`. Notebook `04` also creates `../experiments` at runtime, but no `experiments/` artifacts are included in the delivered folder contents.

### Validate and execute notebooks
Run from the repository root (`mlops_study_main`).

```bash
# Optional: validate notebooks (nbformat v4)
python3 - <<'PY'
import glob, nbformat
for p in sorted(glob.glob('mlops_study_with_ai/week_01_pytorch_foundations/notebooks/*.ipynb')):
  nb = nbformat.read(p, as_version=4)
  nbformat.validate(nb)
  print('VALID', p)
PY
```

**Option A — Reproduce execution and persist outputs in-repo (audit/evidence state):**
```bash
# Execute notebooks in-place (01–07), then 08 last
for nb in \
  01_tensor_fundamentals.ipynb \
  02_tensor_manipulation.ipynb \
  03_autograd.ipynb \
  04_linear_regression.ipynb \
  05_logistic_regression.ipynb \
  06_gradient_verification.ipynb \
  07_linear_regression_fresh_assessment.ipynb
do
  python3 -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 \
    "mlops_study_with_ai/week_01_pytorch_foundations/notebooks/$nb"
done

python3 -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 \
  mlops_study_with_ai/week_01_pytorch_foundations/notebooks/08_week1_assessment.ipynb
```

**Option B — Execute to temporary copies (verification without modifying tracked notebooks):**
```bash
mkdir -p /tmp/opencode
for nb in \
  01_tensor_fundamentals.ipynb \
  02_tensor_manipulation.ipynb \
  03_autograd.ipynb \
  04_linear_regression.ipynb \
  05_logistic_regression.ipynb \
  06_gradient_verification.ipynb \
  07_linear_regression_fresh_assessment.ipynb \
  08_week1_assessment.ipynb
do
  python3 -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 \
    "mlops_study_with_ai/week_01_pytorch_foundations/notebooks/$nb" \
    --output "/tmp/opencode/${nb%.ipynb}.executed.ipynb"
done
```

Notes:
- The repository may contain persisted in-repo execution outputs/counts for auditability; use **Option A** to regenerate that state and **Option B** to verify execution without changing notebook files.
- Plot/image artifacts are written to `mlops_study_with_ai/week_01_pytorch_foundations/reports/` by the relevant notebooks.
- Parity checks (NumPy vs PyTorch) use explicit `np.allclose`/`torch.allclose` tolerances with visible confirmations.
- In CPU-only environments, accelerator-specific paths are guarded and unavailable-device demos are executed as caught, environment-aware checks (no notebook crash).
