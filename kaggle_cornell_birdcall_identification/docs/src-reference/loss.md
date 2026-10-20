# `src/loss/` — 16 files

Every criterion returns the same shape — **`(scalar_tensor, dict_of_scalars)`** —
and the dict's keys are published as `loss_keys` and turned into per-key
metrics by the engine.

| file | LOC | status | `loss_keys` |
|---|---|---|---|
| [`sed_scaled_pos_neg_focal_loss.py`](#sed_scaled_pos_neg_focal_losspy) | 48 | **LIVE** | `bce_loss, F_loss, FScaled_loss, F_loss_0, F_loss_1` |
| [`sed_scaled_pos_neg_focal_loss_augd.py`](#sed_scaled_pos_neg_focal_loss_augdpy) | 55 | **LIVE** | same 5 |
| [`bce_controlled_standard_loss.py`](#bce_controlled_standard_losspy) | 37 | **LIVE** | *(empty)* |
| [`focal_loss_standard.py`](#focal_loss_standardpy) | 34 | **LIVE** (no config uses it) | `bce_loss, F_loss` |
| [`sed_removed_focal_loss.py`](#sed_removed_focal_losspy) | 41 | **LIVE** (no config uses it) | `bce_loss, F_loss, FRemoved_loss` |
| [`sed_scaled_focal_loss.py`](#sed_scaled_focal_losspy) | 42 | **DEAD** | `bce_loss, F_loss, FScaled_loss` |
| [`bce_loss.py`](#bce_losspy) | 16 | LEGACY | *(none)* |
| [`smooth_bce_loss.py`](#smooth_bce_losspy) | 25 | LEGACY | *(none)* |
| [`bce_controlled_loss.py`](#bce_controlled_loss_losspy) | 49 | LEGACY | `primary_loss, bce_loss` |
| [`focal_controlled_standard_loss.py`](#focal_controlled_standard_losspy) | 45 | LEGACY | `bce_loss, F_loss` |
| [`focal_mse_loss.py`](#focal_mse_losspy) | 43 | LEGACY | `mse_loss, F_loss` |
| [`f1_controlled_loss.py`](#f1_controlled_losspy) | 37 | LEGACY | `precision, recall, f1` |
| [`noisy_loss.py`](#noisy_losspy) | 39 | LEGACY + **BROKEN** | declares 4, returns 2 |
| [`noisy_controlled_loss.py`](#noisy_controlled_losspy) | 47 | LEGACY | 4 (all returned) |
| [`custom_losses.py`](#custom_lossespy) | 79 | LEGACY | *(none)* |
| [`custom_losses_for_targets.py`](#custom_losses_for_targetspy) | 83 | LEGACY + **BROKEN** | *(none)* |

Only the **first five** are reachable from `SedEngine._init_criterion_function`
(`engine/sed_engine.py:87-102`).

---

## The contract

Every criterion:

1. takes the **already-sigmoided** prediction (the `AttBlock` applied the
   sigmoid) and a target;
2. returns `(scalar, dict)`;
3. publishes `loss_keys` — the dict's key names.

`engine/sed_engine.py:155-156` creates one `RunningAverage` per key, so the
dict is not decoration: it is the instrumentation contract.

---

## `sed_scaled_pos_neg_focal_loss.py` — the live criterion

**Status: LIVE.** Used by **19 of the 21** configs.

```python
class SedScaledPosNegFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, alpha_1=1.0, alpha_0=1.0, secondary_factor=0.1)
    def forward(self, y_pred, y_target) -> (scalar, dict)
```

**Behaviour.** `y_pred` is `y_target["clipwise_output"]`; the target is the
dict with `all_labels` and `secondary_labels` (L19-22).

```
bce_loss = BCE(y_pred_clamped, y_true)                  # reduction='none'
pt       = exp(-bce_loss)
F_loss_0 = alpha_0 * (1 - all_mask) * (1-pt)**gamma * bce_loss
F_loss_1 = alpha_1 * (all_mask)     * (1-pt)**gamma * bce_loss
F_loss   = F_loss_0 + F_loss_1
FScaled  = secondary_mask * F_loss        # mask = 1, or secondary_factor on secondary species
```

The masks are built under `torch.no_grad()` (L29-35). `y_pred` is clamped to
`[1e-5, 1-1e-5]` (L24) before the BCE.

**⚠️ At the shipped parameters this is exactly mean BCE.** `gamma=0` ⇒
`(1-pt)**0 = 1`; `alpha_0 = alpha_1 = 1` ⇒ the positive/negative split is a
no-op; `secondary_factor = 1.0` ⇒ the secondary mask is 1 everywhere. So
`FScaled_loss == bce_loss.mean()` bit for bit. The name promises a focal loss
that was never applied. (`docs/FINDINGS.md` calls this out; it is not a bug
that invalidates the result — the result was won with plain BCE.)

**Gotchas**

- ⚠️ `gamma=0` means `(1-pt)**0` — if `pt` were exactly 1 and the exponent
  were negative you would get `inf`; the shipped `gamma=0` is safe.
- ⚠️ The prediction must already be sigmoided. Applying a sigmoid here would
  double it.

---

## `sed_scaled_pos_neg_focal_loss_augd.py`

**Status: LIVE** — used by 2 configs (`nomix_fold2_augd`, `nomix_fold3_augd`).

```python
class SedScaledPosNegFocalLossAugd(nn.Module):
    def __init__(self, gamma=0.0, alpha_1=1.0, alpha_0=1.0,
                 secondary_factor=0.1, primary_drop_out_factor=0.3)
```

**Behaviour.** Everything the previous class does, then the **primary-species
positions are randomly zeroed** with probability `primary_drop_out_factor=0.3`
— i.e. **70 % of primary positions contribute nothing** to the loss.

`y_primary_true = y_true - y_sec_true`; a random mask keeps each primary
position with probability 0.3; `FScaled2 = (1 - primary_mask) * FScaled`.

**Gotchas**

- ⚠️ The random mask uses the **global** RNG and is not seeded per-epoch, so
  the run is not reproducible even with a fixed seed.
- ⚠️ The returned `FScaled_loss` in the dict is the **pre-drop** value; the
  dropped value is what the loss actually returns.
- The name `augd` is undocumented anywhere — the effect is invisible from the
  config.

---

## `bce_controlled_standard_loss.py`

**Status: LIVE.** The criterion of the **plain-BCE** models — the family the
winning ensemble came from.

```python
class BCEControlledStandardLoss(nn.Module):
    def __init__(self, beta=0.5)
    def forward(self, y_pred, target) -> (scalar, dict)
```

**Behaviour.** L-soft BCE on the already-sigmoided prediction:

```
y_true' = (beta * y_true + (1 - beta) * y_pred) * y_true
loss    = F.binary_cross_entropy(y_pred, y_true', reduction='none').mean(1).mean()
```

`target` is the **dict**; it reads `target["all_labels"]`.

**Gotchas**

- ⚠️ `loss_keys = []` — it contributes **zero** extra metrics. That is the
  winning criterion's actual behaviour; do not "fix" it.
- ⚠️ The prediction is **not** re-sigmoided. Passing logits would silently
  produce garbage.
- The `.mean(1).mean()` reduces over snippet axis then batch — a mean over
  `B·S·O` only because `binary_cross_entropy` is already elementwise.

---

## `focal_loss_standard.py`

**Status: LIVE (registered), but no shipped config selects it.**

```python
class FocalLossStandard(nn.Module):
    def __init__(self, gamma=0.0, alpha=1.0)
```

**Behaviour.** `F = alpha * (1 - exp(-bce))**gamma * bce`. Note `pt =
exp(-bce)` is a *proxy* for `p`, and the modulating base is `bce`, not
`log p` — so it is not textbook focal loss.

**⚠️ Inert at the defaults**: `gamma=0` ⇒ `F ≡ alpha * bce` ⇒ exactly BCE.

---

## `sed_removed_focal_loss.py`

**Status: LIVE (registered), but no shipped config selects it.**

```python
class SedRemovedFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, alpha=1.0)
```

**Behaviour.** Focal loss as above, then `FRemoved = (1 - mask) * F` where the
mask is 1 on **secondary** species — so secondary species contribute **zero**.

**Gotchas**

- ⚠️ The class name `SedRemovedFocalLoss` is **also used by
  `sed_scaled_focal_loss.py`**, which is dead. A careless
  `from loss.sed_scaled_focal_loss import SedRemovedFocalLoss` gets a
  *different* class with the same name.

---

## `sed_scaled_focal_loss.py`

**Status: DEAD** — imported by nothing in the repository (verified by
repo-wide grep).

```python
class SedRemovedFocalLoss(nn.Module):        # <- the same name as above
    def __init__(self, gamma=0.0, alpha=1.0, secondary_factor=0.1)
```

**Behaviour.** Focal loss with the secondary mask set to `secondary_factor`
(0.1) rather than 0 — so secondary species are **down-weighted**, not removed.
`loss_keys = ["bce_loss", "F_loss", "FScaled_loss"]`.

**Gotchas**

- ⚠️ The class name shadows `sed_removed_focal_loss.SedRemovedFocalLoss`.
- ⚠️ It is the only file in `src/loss/` that nothing imports — a candidate for
  deletion.

---

## `bce_loss.py`

**Status: LEGACY.** `class BCELoss(nn.Module)` — `__init__(self)`, no
`loss_keys`.

Applies a sigmoid, clamps to `[1e-5, 1-1e-5]`, and returns
`(nn.BCELoss()(y_pred, y_true).mean(), {})`.

**Gotcha** — ⚠️ `engine/sed_engine.py:155` reads `self.criterion.loss_keys`
**unguarded**, so selecting this criterion in a config would raise
`AttributeError` during metric setup. It is only reachable from
`main_engine*`.

---

## `smooth_bce_loss.py`

**Status: LEGACY.** `class SmoothBCELoss(nn.Module)` — `__init__(self,
smooth=0.1)`, no `loss_keys`.

BCE with none-reduction, then `loss[0, intersection] = 0` where `intersection`
is the columns that have both a zero target and a `pred <= smooth` entry —
**only row 0 is zeroed**.

**Gotcha** — ⚠️ `engine/main_engine.py:55` passes `smooth=hparams.smooth`,
which **no config defines** ⇒ `AttributeError`.

---

## `bce_controlled_loss.py`

**Status: LEGACY.** `class BCEControlledLoss(nn.Module)` — `__init__(self,
primary_loss_factor=1.0)`, `loss_keys = ["primary_loss", "bce_loss"]`.

L-soft BCE on a **tuple** target `(y_true, target_primary)`, plus a per-row
mean and a primary-class term scaled by `primary_loss_factor`. Also exports
`class LSoftLoss(nn.Module)` — `__init__(self, beta=0.5)`, masked variant.

Reached only from `main_engine_v4`.

---

## `focal_controlled_standard_loss.py`

**Status: LEGACY.** `class FocalControlledStandardLoss(nn.Module)` —
`__init__(self, gamma=0.0, alpha=1.0)`, `loss_keys = ["bce_loss", "F_loss"]`.

L-soft with `BCEWithLogitsLoss(reduction='none')` on **raw logits** (the
prediction is *not* pre-sigmoided here), on a tuple target.

**Gotchas** — ⚠️ `primary_beta` and `target_primary` are unpacked but never
used. ⚠️ Inert at `gamma=0` ⇒ BCEWithLogits on the L-soft target. Also exports
`LSoftLoss` (masked).

---

## `focal_mse_loss.py`

**Status: LEGACY.** `class FocalMSELoss(nn.Module)` — `__init__(self, gamma=0.0,
alpha=1.0)`, `loss_keys = ["mse_loss", "F_loss"]`.

L-soft with `MSELoss(reduction='none')`. Inert at `gamma=0` ⇒ MSE on the
L-soft target. Also exports `LSoftLoss` (unmasked).

---

## `f1_controlled_loss.py`

**Status: LEGACY.** `class F1ControlledLoss(nn.Module)` — `__init__(self)`,
`loss_keys = ["precision", "recall", "f1"]`.

A soft-F1 surrogate: `tp = (y_true*y_pred).sum(0)`, `fp = ((1-y_true)*y_pred).sum(0)`,
`fn = (y_true'*(1-y_pred)).sum(0)`, `p = tp/(tp+fp+ε)`, `r = tp/(tp+fn+ε)`,
`F1 = (2pr/(p+r+ε)).mean()`, `loss = 1 - F1`.

**Gotcha** — `primary_beta` is unpacked but never used.

---

## `noisy_loss.py`

**Status: LEGACY + BROKEN.**

```python
class NoisyLoss(nn.Module):
    def __init__(self)
    self.loss_keys = ["zeros_loss", "ones_loss", "primary_loss", "overall_target_loss"]
    def forward(self, output, target) -> (scalar, {"zeros_loss": ..., "ones_loss": ...})
```

**⚠️ Broken.** `loss_keys` declares **four** keys; `forward` returns **two**
(L40). The mismatch is **lazy** — the metric's `output_transform` is a lambda,
so the `KeyError: 'primary_loss_0'` fires at the **first metric update**, not
at construction.

The sibling `noisy_controlled_loss.py` returns all four — only this file is
inconsistent.

**Behaviour.** Unmasked L-soft: `ones = Σ(l·t)/Σt`, `zeros = Σ(l·(1-t))/Σ(1-t)`,
`loss = (zeros + ones) / 2`.

---

## `noisy_controlled_loss.py`

**Status: LEGACY.** `class NoisyControlledLoss(nn.Module)` — `__init__(self)`,
`loss_keys` = the same four, **all four returned** (L48).

Masked L-soft: `ones*4/5`, `zeros`, a `primary_loss` (selected class vs ones)
`*1/5`, `overall = ones + primary`, `loss = (zeros + overall)/2`. Also
exports `LSoftLoss` (masked).

---

## `custom_losses.py`

**Status: LEGACY.** `class NoisyCuratedLoss(nn.Module)` — `__init__(self,
noisy_type, beta=0.7, q=0.7)`, no `loss_keys`.

Splits rows by a `clean` flag: noisy rows get `LSoftLoss` (β) or `LqLoss`
(`(1-(p·t+ε)^q)/q`), curated rows get plain `nn.BCELoss()`; each is scaled by
`len/bs`, and `loss = 0.5*noisy + 0.5*curated`.

Also exports `LSoftLoss` (`__init__(self, beta=0.5)`, unmasked), `LqLoss`
(`__init__(self, q=0.5)`) and `CuratedLoss`.

**Gotcha** — ⚠️ the scaling uses `bs` = dim 0 of the 3-D target (**B**, not
`B·S`).

---

## `custom_losses_for_targets.py`

**Status: LEGACY + BROKEN.** `class NoisyCuratedLossTargetted(nn.Module)` —
`__init__(self, noisy_type, beta=0.7, q=0.7)`, no `loss_keys`.

Same split, but `LSoftLoss2` softens **only positive entries**
(`y_true == 1.0`) and **mutates `y_true` in place** — so the curated branch
must be computed *before* the noisy branch, which is why the order differs
from `custom_losses.py`.

**⚠️ Broken.** The `elif noisy_type == "lq":` branch (L43) is `pass` (L46), so
`self.noisy_loss` is **never assigned** → `AttributeError` on the first
forward.

Also exports `LSoftLoss2`, `LqLoss2` and `CuratedLoss2`.

---

## See also

- [modules/07 — Losses](../modules/07-losses.md) — the topic chapter
- [modules/11 — Legacy research paths](../modules/11-legacy-research-paths.md)
- [ports/code-issues-and-framework-wins.md](../ports/code-issues-and-framework-wins.md) §5 — why the focal loss is inert
