# `src/metrics/` — 7 files

Four are ignite `Metric` subclasses. The three live ones compute the numbers
the model is ranked by; the fourth writes the inference dump.

| file | LOC | status |
|---|---|---|
| [`sed_lraps.py`](#sed_lrapspy) | 33 | **LIVE** |
| [`sed_f1score_clip.py`](#sed_f1score_clippy) | 43 | **LIVE** |
| [`sed_f1score_frame.py`](#sed_f1score_framepy) | 45 | **LIVE** |
| [`sed_test.py`](#sed_testpy) | 82 | **LIVE** (conditional) + **BROKEN** (L59) |
| [`custom_f1score.py`](#custom_f1scorepy) | 42 | LEGACY |
| [`custom_lraps.py`](#custom_lrapspy) | 33 | LEGACY |
| [`custom_test.py`](#custom_testpy) | 79 | LEGACY |

All four live/legacy ignite metrics share the shape
`reset` / `update` / `compute`, with `@reinit__is_reduced` on the first two
and `@sync_all_reduce(...)` on the third. `update` is called per iteration
and `compute` once at the end, so the metric is a **mean of per-item
scores**, not a global score.

---

## `sed_lraps.py`

**Status: LIVE** — registered as `validation_metrics["lraps"]`
(`engine/sed_engine.py:161-162`).

```python
class SedLRAPS(Metric):
    def __init__(self, output_transform=lambda x: x)
    def reset(self) / update(self, output) / compute(self)
```

**Behaviour.** `pred = pred["clipwise_output"].max(axis=1)[0]` (L22) — the
**max** over the snippet axis — and `target = target[:,0,:]` (L24) — snippet 0's
labels. `label_ranking_average_precision_score([t], [p])` per item (L25-27),
averaged.

**Gotchas**

- ⚠️ **No sigmoid is applied** — the `AttBlock` already applied it. Each file
  says so in a comment.
- ⚠️ **No background branch.** A row with no positive label is scored by
  sklearn as a **perfect ranking (1.0)**. The validation set deliberately
  injects label-less no-bird audio, so `lraps` is inflated by exactly those
  rows. `SedF1ScoreClip` and `SedF1ScoreFrame` *do* have the branch and score
  them 0.0 when something fires — so the three metrics disagree about the
  same rows. `lraps` is not the tracked metric, so checkpoint ranking is
  unaffected, but the reported number is not a clean quality measure.
- `max(axis=1)[0]` takes the **values** of a torch `max` (which returns a
  named tuple).

---

## `sed_f1score_clip.py`

**Status: LIVE** — `validation_metrics["f1score_clip"]`
(`engine/sed_engine.py:165-166`). **This is the tracked metric** in the
`final_sed` family.

```python
class SedF1ScoreClip(Metric):
    def __init__(self, threshold=0.5, output_transform=lambda x: x)
```

**Behaviour.** `pred = pred["clipwise_output"].max(axis=1)[0]` (L26-27),
`target = target[:,0,:]` (L28). Per item:

```
if t.sum() == 0:                      # background row
    score = f1_score([1],[0]) = 0.0    #   if anything fired above threshold
          = f1_score([1],[1]) = 1.0    #   otherwise
else:
    score = f1_score([t], [p >= threshold], average='micro')
```

**Gotchas**

- ⚠️ No sigmoid is applied (the `AttBlock` did it).
- The background branch is the reason the injected no-bird audio is useful:
  a false positive on a background row scores **0.0**.
- It is the **mean of per-item micro-F1**, not a global micro-F1.

---

## `sed_f1score_frame.py`

**Status: LIVE** — `validation_metrics["f1score_frame"]`
(`engine/sed_engine.py:169-170`). The tracked metric in the **5-fold** family
(folds 1–4).

```python
class SedF1ScoreFrame(Metric):
    def __init__(self, threshold=0.5, output_transform=lambda x: x)
```

**Behaviour.** `pred_frame = pred["framewise_output"].max(axis=1)[0].max(axis=1)[0]`
(L26-27) — max over snippets, then max over the **entire time axis** — and
`pred_clip = pred["clipwise_output"].max(axis=1)[0]` (L30-31). Per item:

```
if t.sum() == 0:
    score = 0.0 if any((pf >= threshold) & (pc >= 0.3)) else 1.0
else:
    score = f1_score([t], [(pf >= threshold) & (pc >= 0.3)], average='micro')
```

**Gotchas**

- ⚠️ **The `0.3` clip threshold is a hard-coded literal** (L35, L40) — not a
  config value. It is the in-repo ancestor of the winning ensemble rule
  (`README.md:45`: 4 of 13 models, clip ≥ 0.3 **and** frame ≥ 0.3). A port
  must keep it, and must keep the two gates **independent**.
- ⚠️ The frame axis is max-pooled away, so this is a **file-level** metric
  despite the name.
- No sigmoid is applied.

---

## `sed_test.py`

**Status: BROKEN** — the only file in `src/` that does not compile.

```python
INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}     # defined, never used
class NumpyEncoder(json.JSONEncoder): default(self, obj)
class SedTest(Metric):
    def __init__(self, save_dir, combine_files=False, output_transform=lambda x: x)
    def reset(self) / update(self, output) / compute(self)
```

**Behaviour.** `update` (L41-72) writes one JSON per audio file:

| key | value |
|---|---|
| `filename` | the file name |
| `predicted_frames` | 264 frame scores, max-pooled over the time axis |
| `predicted_clip` | 264 clip scores, max-pooled over snippets |
| `targets` | the 264 label vector |
| `lraps_score_clip` | per-file LRAP on the clip head |
| `lraps_score_frames` | per-file LRAP on the frame head |

`compute` (L75-81) optionally merges every `save_dir/*.json` into
`save_dir/final/merged_results.json` by **text concatenation**
(`'[' + ','.join(file_texts) + ']'`), and returns the mean per-file frame LRAP.

**⚠️ Four independent blockers**

1. **L59** — a missing comma after `"filename": filename` ⇒
   `SyntaxError: invalid syntax` at L60. The file does not parse.
2. **L28** — `save_dir` must be a `pathlib.Path`; a `str` has no `.mkdir`.
3. `engine/sed_engine.py:174` reads `hparams.save_dir` and
   `hparams.combine_files`, **neither of which any config defines** ⇒ the
   branch is unreachable.
4. `engine/sed_engine.py:60` — the `mode='test'` branch of `prepare_batch`
   references an **undefined local `x`** ⇒ `NameError`.

**⚠️ A fifth, latent**

5. **L74** — `@sync_all_reduce("_score", "_num_items")` names an attribute
   that does not exist (the class has `_score_frame` / `_score_clip`).
   Ignite's decorator resolves a missing attribute with a `None` default, so
   under DDP the **denominator is globally reduced and the numerator is not** —
   a silently wrong metric.

**Why it cannot produce a submission**

The frame axis is max-pooled away (L50), so a 10-minute file yields **one**
record, not 120 five-second rows. There is no `row_id`, no `labels` string, no
`nocall` fallback, no site 1/2/3 handling and no CSV emission anywhere in the
repository. The vote lives in an external Kaggle kernel
(`README.md:130`).

---

## `custom_f1score.py`

**Status: LEGACY.** `class CustomF1Score(Metric)` — `__init__(self,
threshold=0.5, output_transform=lambda x: x)`.

**Behaviour.** `pred = torch.sigmoid(pred).max(axis=1)[0]` (L27) — the
**sigmoid IS applied** here — and `target = target[:,0,:]` (L28). Same
background branch as the SED version (L31-35). Mean over items.

**Axis-1 semantics.** The legacy model output is `(B, S, C)` = batch ×
**snippet** × class (see `models/densenet121.py:13-18`), so axis 1 is the
snippet axis: `max(axis=1)` takes, per class, the maximum sigmoid probability
across all snippets of the recording.

**Gotchas**

- ⚠️ `self.sigmoid = torch.nn.Sigmoid()` (L10) is created but **never used** —
  L27 calls `torch.sigmoid` directly.
- `NotComputableError` is imported and never raised; `compute()` divides by
  zero on an empty metric.
- The target is snippet 0 only, so a recording whose first snippet is
  background but whose later snippets carry the species is scored against an
  all-zero target.

---

## `custom_lraps.py`

**Status: LEGACY.** `class CustomLRAPS(Metric)` — `__init__(self,
output_transform=lambda x: x)`.

**Behaviour.** `pred = self.sigmoid(pred)` then `pred = pred.mean(axis=1)`
(L23-24) — **mean** pooling, not max. `target = target[:,0,:]` (L25).
`label_ranking_average_precision_score` per item.

**⚠️ The difference from `custom_f1score.py` is the pooling, not the
sigmoid** — both apply a sigmoid. F1 uses max, LRAPS uses mean.

**Gotchas**

- ⚠️ Mean-pooling the prediction while scoring against **snippet 0's** labels
  is an asymmetry: a species audible only in a later snippet gets a diluted
  score against a zero target.
- No `threshold` parameter, unlike `CustomF1Score`.
- No background branch.

---

## `custom_test.py`

**Status: LEGACY.** The predecessor of `sed_test.py`, and the one that
actually works.

```python
class NumpyEncoder(json.JSONEncoder): default(self, obj)
class CustomTest(Metric):
    def __init__(self, save_dir, combine_files=False, output_transform=lambda x: x)
```

**Behaviour.** Writes one JSON per file with the keys `predicted`, `targets`,
`filename` (the `.mp3` stripped), `lraps_score`, **`background`** (a
boolean, set when `t.sum() == 0`) and `f1_score_50` … `f1_score_90` (a
threshold sweep over `[0.5, 0.6, 0.7, 0.8, 0.9]`). `compute()` can merge
everything into `final/merged_results.json`, exactly as `SedTest` does.

**Gotchas**

- ⚠️ `NumpyEncoder` handles only `np.ndarray`. A **scalar** `np.float32` is
  not JSON-serialisable (`np.float32` is not a subclass of `float`), which
  bites whenever the prediction collapses to 1-D — i.e. the
  `x["y_pred"][0]` wiring in `main_engine_v2/v3/v4`.
- The merge opens every file **without closing it** (L77) and re-runs on every
  `compute()` call.
- `glob('*.json')` is non-recursive, so `final/merged_results.json` does not
  merge itself.

---

## See also

- [modules/10 — Metrics and tracking](../modules/10-metrics-and-tracking.md)
- [modules/08 — Inference and submission](../modules/08-inference-and-submission.md)
- [ports/code-issues-and-framework-wins.md](../ports/code-issues-and-framework-wins.md) §6, §7 — the LRAP and DDP-reduction defects
