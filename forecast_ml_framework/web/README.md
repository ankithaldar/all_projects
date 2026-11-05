# Forecast ML — Evaluation Dashboard (`web/`)

Single-page, offline dashboard (vanilla HTML/JS/CSS, no build step, no CDN).
Shows KPIs, per-bucket metrics, classification + regression charts, and SHAP bars
from the Phase 1 evaluation pipeline artifacts.

## How to open

```bash
cd forecast_ml_framework/web
python3 -m http.server 8000
# open http://localhost:8000/
```

`file://` note: `fetch()` of `metrics.json` is blocked under `file://` in most
browsers; the page shows a graceful fallback message telling you to use the
server above. No external requests are made in either mode.

## Files

| File | Purpose |
|---|---|
| `index.html` | Semantic page: header, KPIs, metrics table, charts gallery, SHAP, footer |
| `styles.css` | CSS variables, responsive grid, print stylesheet, `prefers-reduced-motion` |
| `app.js` | Fetches metrics, renders KPIs/table/charts/SHAP, sort + filter. Single `CONFIG` const at top |
| `sample/metrics.json` | Sample/synthetic data matching the evaluation output shape (below) |
| `sample/charts/*.png` | Sample matplotlib PNGs (ROC, PR, confusion, calibration, pred-vs-actual, residuals) |

## Data contract (for the evaluation pipeline)

Drop these into `web/sample/` (or point `CONFIG` at a real artifacts folder):

**`metrics.json`** — shape mirrors `pipelines/modeling/evaluate.py` +
`metrics.py` (`compute_metrics`), with web-friendly aliases:

```json
{
  "task": "classification",
  "meta": {
    "model": "xgboost (demo_model_v1)", "target_col": "target",
    "date": "2026-11-05", "env": "local", "random_state": 42,
    "reproducibility_hash": "sha256:…",
    "generated_by": "forecast_ml_framework pipelines/modeling evaluate_model"
  },
  "global": {"roc_auc": 0.912, "f1_weighted": 0.921, "ece": 0.031, "…": "…"},
  "buckets": {
    "segment": {"north": {"roc_auc": 0.931, "…": "…"}},
    "decile":  {"0": {"roc_auc": 0.999, "…": "…"}}
  },
  "shap_top": [{"feature": "income", "importance": 0.312}],
  "charts": [
    {"file": "charts/roc.png", "set": "classification", "name": "roc",
     "caption": "What this chart diagnoses…", "alt": "Alt text for screen readers"}
  ]
}
```

- `global` — output of `compute_metrics(y_true, y_pred, y_proba, task)`:
  classification keys include `accuracy, balanced_accuracy, precision_*,
  recall_*, f1_*, roc_auc, pr_auc, log_loss, brier, kappa, mcc, sensitivity,
  specificity, ks, ece, confusion_matrix, per_class_f1`; regression keys
  include `mae, medae, rmse, rmsle, mse, r2, adj_r2, mape, smape,
  explained_variance, max_error, bias`.
- `buckets.segment` = `evaluate_model`'s `by_bucket` (grouped by
  `params.bucket_segment_col`, e.g. `region`); `buckets.decile` = `by_decile`
  (prediction-decile groups). `by_bucket`/`by_decile` aliases are also accepted.
- `charts` — manifest; `file` is relative to `CONFIG.chartsBase`
  (default `sample/`), so `charts/roc.png` → `sample/charts/roc.png`.
  Chart names follow `charts.py::make_charts`: classification
  (`roc, pr, confusion_heatmap, calibration_curve, threshold_curve, lift_gain,
  ks_plot, prob_distribution, per_class_bars, cost_curve`) and regression
  (`pred_vs_actual, residuals_vs_fitted, residuals_vs_actual, residual_hist_qq,
  scale_location, error_vs_actual, error_by_bucket, segment_curves,
  feature_vs_residual`). Each entry needs `caption` (what issue it diagnoses:
  calibration, leakage, drift) and `alt`.
- `shap_top` — `[{feature, importance}]`, descending importance (page re-sorts).

The evaluation worker should write real `metrics.json` + PNGs with exactly this
shape; the page needs no other changes.

## Pointing at real artifacts

Edit the single `CONFIG` const at the top of `app.js`:

```js
const CONFIG = {
  metricsUrl: "../artifacts/web/metrics.json", // or "sample/metrics.json"
  chartsBase: "../artifacts/web/",              // folder containing metrics.json
  kpiKeys: ["roc_auc", "pr_auc", "f1_weighted", "accuracy", "ece"]
};
```

Keep `metrics.json` and its chart PNGs in the same folder (chart `file` paths
are resolved relative to `chartsBase`).

## Accessibility / design notes

- Landmarks (`header/nav/main/footer`), skip link, `label`led controls,
  keyboard-sortable table headers, `alt` text, lazy-loaded chart images.
- Responsive: KPI cards and chart gallery collapse to one column on narrow screens.
- `prefers-reduced-motion` disables transitions; `@media print` hides nav/controls.
- No external fonts, scripts, or CDN — works offline.

## Verification (no browser required)

```bash
node --check web/app.js
grep -RInE 'https?://' web/index.html web/app.js web/styles.css || true  # expect: no CDN refs
python3 -m http.server 8000   # in web/, then curl -s localhost:8000/ | grep -c '<section'
```
