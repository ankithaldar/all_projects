#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Charts node: matplotlib figures for both task types.

Existing evaluation charts: ROC, PR, confusion heatmap,
calibration, threshold, lift/gain, KS, probability
distribution, per-class F1; regression:
predicted-vs-actual, residuals vs fitted/actual,
residual hist+QQ, scale-location, error vs actual,
error by bucket, segment curves, feature vs residual.
(The placeholder cost-curve stub was deleted per F5:
a 2-point line is not a cost curve, and no caller or
test referenced the ``cost_curve`` key, so it was
removed instead of faked.)

Phase-3 additions:
I. Data understanding (make_data_charts): target
distribution, feature distributions, correlation
heatmap, pair plot, missing heatmap, outlier box,
PCA scatter.
II. Classification: DET curve, confusion wheel,
top-k accuracy, balanced accuracy, PR gain curve.
III. Regression: residual autocorrelation, Cook's
distance, REC curve, prediction-interval coverage,
residuals over time, forecast vs actual.
IV. Training (make_training_charts): learning curve,
validation curve, loss vs epoch, CV scores box,
overfitting gap, bias-variance.
V. Interpretability: SHAP dependence, SHAP waterfall,
SHAP force, partial dependence (1D), ICE, LIME bar,
decision boundary, tree plot.
VI. Comparison (make_comparison_charts): model
comparison box, Bland-Altman, ensemble diversity,
ROC overlay, PR overlay, error correlation heatmap.
VII. Specialized: quantile coverage, prediction
interval vs actual.

Out of scope for tabular data: image/text charts
(GradCAM, attention maps, word clouds) are not
applicable here and are intentionally omitted.

Every public chart function returns a Figure or
None on insufficient data, never raises, uses only
matplotlib (imported lazily on first chart call, V8 --
importing this module never requires the plotting
stack), is NaN-safe, and closes its figure in
a finally block so no figure slot leaks.
'''

import logging

import numpy as np

from forecast_ml_framework.pipelines.modeling.ntile import (
  normalize_task_kind,
)

LOG = logging.getLogger(__name__)


class _LazyPlt:
  '''Lazily import matplotlib.pyplot on first use (V8).

  The real ``matplotlib.pyplot`` module is imported
  the first time any attribute is accessed (i.e.
  inside the first chart call), with the Agg backend
  selected before pyplot loads. Module import alone
  therefore succeeds without the plotting stack.
  '''

  def __init__(self):
    '''Initialize the lazy pyplot holder.

    No import happens here; the real pyplot module
    is loaded on first attribute access.

    Returns:
      None.
    '''
    self._mod = None

  def _load(self):
    '''Import and cache the real pyplot module.'''
    if self._mod is None:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as mod
      self._mod = mod
    return self._mod

  def __getattr__(self, name):
    '''Forward attribute access to the real pyplot module.

    Args:
      name: Attribute name requested on pyplot.

    Returns:
      The attribute from the lazily loaded pyplot module.
    '''
    return getattr(self._load(), name)


plt = _LazyPlt()


def _pd_abs_err(y_true, y_pred):
  '''Return absolute errors as a pandas Series.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.

  Returns:
    Pandas Series of absolute errors aligned by position.
  '''
  import pandas as pd
  return pd.Series(np.abs(np.asarray(y_true, dtype=float)
                           - np.asarray(y_pred, dtype=float)))


def _new_fig(title):
  '''Create a figure with a title.'''
  fig, ax = plt.subplots()
  ax.set_title(title)
  return fig, ax


def _safe(figs, name, fn, *args):
  '''Add a figure if fn succeeds, else skip gracefully.

  Args:
    figs: Dict of chart name to figure, updated in place.
    name: Chart name to store the figure under.
    fn: Callable returning a matplotlib Figure.
    *args: Arguments forwarded to fn.

  Returns:
    None. Figures created by a failing fn are closed so
    no figure leaks.
  '''
  before = set(plt.get_fignums())
  try:
    fig = fn(*args)
    if fig is not None:
      figs[name] = fig
  except Exception as exc:  # noqa: BLE001 - charts skip on bad data
    LOG.debug('_safe: chart %s skipped (%s)', name, exc)
    for num in set(plt.get_fignums()) - before:
      plt.close(num)


def _roc(y_true, proba):
  '''Plot the ROC curve.'''
  from sklearn.metrics import roc_curve, roc_auc_score
  fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
  fig, ax = _new_fig('ROC curve')
  ax.plot(fpr, tpr, label=f'auc={roc_auc_score(y_true, proba[:, 1]):.3f}')
  ax.plot([0, 1], [0, 1], '--')
  ax.legend()
  return fig


def _pr(y_true, proba):
  '''Plot the precision-recall curve.'''
  from sklearn.metrics import precision_recall_curve
  prec, rec, _ = precision_recall_curve(y_true, proba[:, 1])
  fig, ax = _new_fig('PR curve')
  ax.plot(rec, prec)
  ax.set_xlabel('recall')
  ax.set_ylabel('precision')
  return fig


def _confusion(y_true, y_pred):
  '''Plot the confusion matrix heatmap.'''
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(y_true, y_pred)
  fig, ax = _new_fig('Confusion matrix')
  ax.imshow(cm, cmap='Blues')
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j, i, str(cm[i, j]), ha='center', va='center')
  return fig


def _calibration(y_true, proba):
  '''Plot the calibration curve.'''
  from sklearn.calibration import calibration_curve
  prob_true, prob_pred = calibration_curve(
      y_true, proba[:, 1], n_bins=5)
  fig, ax = _new_fig('Calibration curve')
  ax.plot(prob_pred, prob_true, marker='o')
  ax.plot([0, 1], [0, 1], '--')
  return fig


def _threshold(y_true, proba):
  '''Plot precision/recall vs threshold.'''
  from sklearn.metrics import precision_recall_curve
  prec, rec, thr = precision_recall_curve(y_true, proba[:, 1])
  fig, ax = _new_fig('Threshold curve')
  ax.plot(thr, prec[:-1], label='precision')
  ax.plot(thr, rec[:-1], label='recall')
  ax.legend()
  return fig


def _lift(y_true, proba):
  '''Plot lift and gain curves.'''
  from sklearn.metrics import roc_curve
  order = np.argsort(-proba[:, 1])
  y_sorted = np.asarray(y_true)[order]
  gain = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
  frac = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
  fig, ax = _new_fig('Lift / gain')
  ax.plot(frac, gain, label='gain')
  ax.plot(frac, gain / np.maximum(frac, 1e-9), label='lift')
  ax.legend()
  return fig


def _ks_plot(y_true, proba):
  '''Plot cumulative score distributions by class.'''
  pos = np.sort(proba[:, 1][np.asarray(y_true) == 1])
  neg = np.sort(proba[:, 1][np.asarray(y_true) == 0])
  fig, ax = _new_fig('KS plot')
  ax.plot(pos, np.arange(1, len(pos) + 1) / max(len(pos), 1),
          label='pos')
  ax.plot(neg, np.arange(1, len(neg) + 1) / max(len(neg), 1),
          label='neg')
  ax.legend()
  return fig


def _prob_dist(y_true, proba):
  '''Plot probability distributions by class.'''
  fig, ax = _new_fig('Probability distribution')
  for label in (0, 1):
    ax.hist(proba[:, 1][np.asarray(y_true) == label],
            bins=10, alpha=0.5, label=str(label))
  ax.legend()
  return fig


def _per_class_bars(y_true, y_pred):
  '''Plot per-class F1 bars.'''
  from sklearn.metrics import f1_score
  vals = f1_score(y_true, y_pred, average=None, zero_division=0)
  fig, ax = _new_fig('Per-class F1')
  ax.bar(range(len(vals)), vals)
  ax.set_xlabel('class')
  return fig


def _pred_vs_actual(y_true, y_pred):
  '''Plot predicted vs actual.'''
  fig, ax = _new_fig('Predicted vs actual')
  ax.scatter(y_true, y_pred, s=8)
  ax.plot([y_true.min(), y_true.max()],
          [y_true.min(), y_true.max()], '--')
  return fig


def _residuals_fitted(y_pred, resid):
  '''Plot residuals vs fitted.'''
  fig, ax = _new_fig('Residuals vs fitted')
  ax.scatter(y_pred, resid, s=8)
  ax.axhline(0, color='r')
  return fig


def _residuals_actual(y_true, resid):
  '''Plot residuals vs actual plus residual histogram.'''
  fig, ax = _new_fig('Residuals vs actual')
  ax.scatter(y_true, resid, s=8)
  ax.axhline(0, color='r')
  return fig


def _residual_hist_qq(y_true, y_pred):
  '''Plot residual histogram and QQ plot side by side.'''
  from scipy.stats import probplot
  resid = np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float)
  fig, axes = plt.subplots(1, 2)
  axes[0].hist(resid, bins=min(10, max(2, len(resid) // 2)))
  axes[0].set_title('Residual histogram')
  probplot(resid, plot=axes[1])
  axes[1].set_title('QQ plot')
  fig.suptitle('Residuals hist + QQ')
  return fig


def _scale_location(y_true, y_pred):
  '''Plot sqrt(|resid|) vs fitted (scale-location).'''
  resid = np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float)
  fig, ax = _new_fig('Scale-location')
  ax.scatter(y_pred, np.sqrt(np.abs(resid)), s=8)
  return fig


def _error_actual(y_true, y_pred):
  '''Plot absolute error vs actual.'''
  err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float))
  fig, ax = _new_fig('Error vs actual')
  ax.scatter(y_true, err, s=8)
  return fig


def _error_bucket(y_true, y_pred, x, params):
  '''Plot mean absolute error by bucket segment.'''
  col = params.get('bucket_segment_col')
  if not col or not hasattr(x, 'columns') or col not in x.columns:
    raise ValueError('no bucket column')
  err = _pd_abs_err(y_true, y_pred)
  means = err.groupby(x[col].values).mean()
  fig, ax = _new_fig('Error by bucket')
  ax.bar([str(k) for k in means.index], means.values)
  return fig


def _segment_curves(y_true, y_pred, x, params):
  '''Plot predicted vs actual per segment.'''
  col = params.get('bucket_segment_col')
  if not col or not hasattr(x, 'columns') or col not in x.columns:
    raise ValueError('no bucket column')
  fig, ax = _new_fig('Segment curves')
  for label in list(dict.fromkeys(x[col].astype(str))):
    mask = (x[col].astype(str) == label).values
    ax.scatter(np.asarray(y_true)[mask],
               np.asarray(y_pred)[mask], s=8, label=str(label))
  ax.legend()
  return fig


def _feature_residual(y_true, y_pred, x):
  '''Plot residuals vs the first numeric feature.'''
  resid = np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float)
  numeric = x.select_dtypes(include='number') if hasattr(
      x, 'select_dtypes') else None
  if numeric is None or numeric.shape[1] == 0:
    raise ValueError('no numeric feature')
  fig, ax = _new_fig('Feature vs residual')
  ax.scatter(numeric.iloc[:, 0], resid, s=8)
  ax.set_xlabel(str(numeric.columns[0]))
  return fig


def _safe_fig(fn):
  '''Wrap a chart fn: never raise, always close figure.

  Args:
    fn: Callable returning a Figure or None.

  Returns:
    Wrapper returning the (closed) Figure or None.
    The figure is closed in a finally block so
    plt.get_fignums() is unchanged for callers,
    while the Figure object remains inspectable.
  '''
  def wrapper(*args, **kwargs):
    '''Call the wrapped chart function safely.

    Args:
      *args: Positional arguments forwarded to fn.
      **kwargs: Keyword arguments forwarded to fn.

    Returns:
      The (closed) Figure from fn, or None on error.
    '''
    before = set(plt.get_fignums())
    fig = None
    try:
      fig = fn(*args, **kwargs)
      return fig
    except Exception as exc:  # noqa: BLE001 - charts skip bad data
      LOG.debug('_safe_fig: chart skipped (%s)', exc)
      for num in set(plt.get_fignums()) - before:
        plt.close(num)
      return None
    finally:
      if fig is not None:
        plt.close(fig)
  return wrapper


def _pairs(y_true, y_pred):
  '''Return finite (true, pred) float arrays or None.'''
  yt = np.asarray(y_true, dtype=float).ravel()
  yp = np.asarray(y_pred, dtype=float).ravel()
  if yt.size != yp.size or yt.size < 5:
    return None
  mask = np.isfinite(yt) & np.isfinite(yp)
  if mask.sum() < 5:
    return None
  return yt[mask], yp[mask]


def _proba_pos(proba):
  '''Return positive-class scores from proba, or None.'''
  arr = np.asarray(proba, dtype=float)
  if arr.ndim == 1:
    scores = arr
  elif arr.ndim == 2 and arr.shape[1] >= 2:
    scores = arr[:, 1]
  elif arr.ndim == 2 and arr.shape[1] == 1:
    scores = arr[:, 0]
  else:
    return None
  if scores.size < 5 or not np.isfinite(scores).any():
    return None
  return scores


def _numeric_df(df):
  '''Return df as a numeric DataFrame.'''
  import pandas as pd
  if hasattr(df, 'select_dtypes'):
    return df.select_dtypes(include='number')
  return pd.DataFrame(np.asarray(df, dtype=float))


def _merge(figs, extra):
  '''Merge extra chart dict into figs in place.'''
  for name, fig in dict(extra or {}).items():
    if fig is not None:
      figs[name] = fig


@_safe_fig
def target_distribution(df, target):
  '''Plot target distribution (hist or count bars).

  Args:
    df: DataFrame containing the target column.
    target: Target column name.

  Returns:
    Figure, or None if df/target is missing or tiny.
  '''
  import pandas as pd
  if df is None or not hasattr(df, 'columns'):
    return None
  if target not in df.columns or len(df) < 5:
    return None
  series = df[target].dropna()
  if len(series) < 5:
    return None
  fig, ax = _new_fig('Target distribution')
  if pd.api.types.is_numeric_dtype(series):
    ax.hist(series, bins=min(20, max(2, len(series) // 3)))
  else:
    counts = series.value_counts()
    ax.bar([str(k) for k in counts.index], counts.values)
  ax.set_xlabel(str(target))
  return fig


@_safe_fig
def feature_distributions(df, max_features=5):
  '''Plot box plots per numeric feature (capped).

  Args:
    df: Input DataFrame.
    max_features: Maximum numeric features to show.

  Returns:
    Figure, or None if insufficient numeric data.
  '''
  numeric = _numeric_df(df)
  if numeric.shape[1] == 0 or len(numeric) < 5:
    return None
  cols = list(numeric.columns)[:max_features]
  fig, ax = _new_fig('Feature distributions')
  ax.boxplot([numeric[c].dropna().values for c in cols],
             tick_labels=[str(c) for c in cols])
  return fig


@_safe_fig
def correlation_heatmap(df):
  '''Plot the numeric correlation heatmap.

  Args:
    df: Input DataFrame.

  Returns:
    Figure, or None if fewer than two numeric columns.
  '''
  numeric = _numeric_df(df)
  if numeric.shape[1] < 2 or len(numeric) < 5:
    return None
  corr = numeric.corr(numeric_only=True).fillna(0).values
  fig, ax = _new_fig('Correlation heatmap')
  ax.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
  ax.set_xticks(range(len(numeric.columns)))
  ax.set_xticklabels([str(c) for c in numeric.columns], rotation=45)
  ax.set_yticks(range(len(numeric.columns)))
  ax.set_yticklabels([str(c) for c in numeric.columns])
  fig.colorbar(ax.images[0], ax=ax)
  return fig


@_safe_fig
def pair_plot(df, target=None, max_cols=6):
  '''Plot pairwise scatters for capped numeric columns.

  Args:
    df: Input DataFrame.
    target: Optional column used to colour points.
    max_cols: Maximum numeric columns to use (<=6).

  Returns:
    Figure, or None if insufficient numeric data.
  '''
  numeric = _numeric_df(df)
  cols = list(numeric.columns)[:min(max_cols, 6)]
  if len(cols) < 2 or len(numeric) < 5:
    return None
  colours = None
  if target is not None and hasattr(df, 'columns') \
          and target in df.columns:
    colours = df[target].values
  fig, axes = plt.subplots(len(cols), len(cols))
  axes = np.atleast_2d(axes)
  for i, c1 in enumerate(cols):
    for j, c2 in enumerate(cols):
      ax = axes[i, j]
      if i == j:
        ax.hist(numeric[c1].dropna(), bins=10)
      else:
        ax.scatter(numeric[c2], numeric[c1], s=4, c=colours)
  fig.suptitle('Pair plot')
  return fig


@_safe_fig
def missing_heatmap(df):
  '''Plot the missing-value indicator heatmap.

  Args:
    df: Input DataFrame.

  Returns:
    Figure, or None if df has no rows/columns.
  '''
  import pandas as pd
  frame = df if hasattr(df, 'isna') else pd.DataFrame(df)
  if frame.size == 0 or len(frame) < 5 or bool(frame.isna().all().all()):
    return None
  fig, ax = _new_fig('Missing values')
  ax.imshow(frame.isna().values.astype(int), cmap='Reds',
            aspect='auto')
  ax.set_xlabel('column')
  ax.set_ylabel('row')
  return fig


@_safe_fig
def outlier_box(df, max_features=10):
  '''Plot box plots flagging z-score outliers (>|3|).

  Args:
    df: Input DataFrame.
    max_features: Maximum numeric features to show.

  Returns:
    Figure, or None if insufficient numeric data.
  '''
  numeric = _numeric_df(df)
  cols = list(numeric.columns)[:max_features]
  if not cols or len(numeric) < 5:
    return None
  data = []
  for col in cols:
    vals = numeric[col].dropna().values.astype(float)
    mu, sd = np.nanmean(vals), np.nanstd(vals) or 1.0
    data.append(vals[np.abs((vals - mu) / sd) <= 3])
  if not any(len(d) for d in data):
    return None
  fig, ax = _new_fig('Outliers (|z|>3 removed)')
  ax.boxplot(data, tick_labels=[str(c) for c in cols])
  return fig


@_safe_fig
def pca_scatter(df, target=None):
  '''Plot PCA 2D scatter coloured by target (sklearn lazy).

  Args:
    df: Input DataFrame.
    target: Optional colouring column name.

  Returns:
    Figure, or None if fewer than 3 numeric columns.
  '''
  from sklearn.decomposition import PCA
  numeric = _numeric_df(df).dropna()
  if numeric.shape[1] < 3 or len(numeric) < 5:
    return None
  coords = PCA(n_components=2).fit_transform(numeric.values)
  colours = None
  if target is not None and hasattr(df, 'columns') \
          and target in df.columns:
    colours = df.loc[numeric.index, target].values
  fig, ax = _new_fig('PCA scatter')
  ax.scatter(coords[:, 0], coords[:, 1], s=10, c=colours)
  ax.set_xlabel('PC1')
  ax.set_ylabel('PC2')
  return fig


def make_data_charts(df, target, params=None):
  '''Build all data-understanding charts.

  Args:
    df: Input DataFrame.
    target: Target column name.
    params: Optional params dict (unused, reserved).

  Returns:
    Dict of chart name to closed Figure.
  '''
  _ = params
  figs = {}
  try:
    _safe(figs, 'target_distribution', target_distribution, df, target)
    _safe(figs, 'feature_distributions', feature_distributions, df)
    _safe(figs, 'correlation_heatmap', correlation_heatmap, df)
    _safe(figs, 'pair_plot', pair_plot, df, target)
    _safe(figs, 'missing_heatmap', missing_heatmap, df)
    _safe(figs, 'outlier_box', outlier_box, df)
    _safe(figs, 'pca_scatter', pca_scatter, df, target)
    return figs
  finally:
    for fig in list(figs.values()):
      plt.close(fig)


@_safe_fig
def det_curve(y_true, proba):
  '''Plot the DET curve (FNR vs FPR).

  Args:
    y_true: True labels.
    proba: Predicted probabilities.

  Returns:
    Figure, or None on insufficient/degenerate data.
  '''
  from sklearn.metrics import det_curve as _det
  scores = _proba_pos(proba)
  yt = np.asarray(y_true).ravel()
  if scores is None or yt.size != scores.size or len(np.unique(yt)) < 2:
    return None
  fnr, fpr, _ = _det(yt, scores)
  fig, ax = _new_fig('DET curve')
  ax.plot(fpr, fnr, label='DET')
  ax.plot([0, 1], [1, 0], '--', label='chance')
  ax.legend()
  ax.set_xlabel('false positive rate')
  ax.set_ylabel('false negative rate')
  return fig


@_safe_fig
def confusion_wheel(y_true, y_pred):
  '''Plot per-class accuracy as a polar (wheel) bar chart.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.

  Returns:
    Figure, or None on insufficient data.
  '''
  from sklearn.metrics import confusion_matrix
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  labels, _ = np.unique(pairs[0], return_counts=True)
  cm = confusion_matrix(pairs[0], pairs[1]).astype(float)
  acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
  if len(acc) < 2:
    return None
  fig = plt.figure()
  ax = fig.add_subplot(111, polar=True)
  ax.set_title('Confusion wheel')
  angles = np.linspace(0, 2 * np.pi, len(acc), endpoint=False)
  ax.bar(angles, acc, width=0.8)
  ax.set_xticks(angles)
  ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v)
                       for v in labels])
  return fig


@_safe_fig
def topk_accuracy_plot(y_true, proba):
  '''Plot top-k accuracy vs k.

  Args:
    y_true: True labels.
    proba: Predicted probabilities (n, n_classes).

  Returns:
    Figure, or None on insufficient data.
  '''
  arr = np.asarray(proba, dtype=float)
  yt = np.asarray(y_true).ravel()
  if arr.ndim != 2 or arr.shape[0] != yt.size or yt.size < 5:
    return None
  order = np.argsort(-arr, axis=1)
  ks = list(range(1, arr.shape[1] + 1))
  accs = []
  for k in ks:
    top = order[:, :k]
    accs.append(np.mean([yt[i] in top[i] for i in range(yt.size)]))
  fig, ax = _new_fig('Top-k accuracy')
  ax.plot(ks, accs, marker='o')
  ax.set_xlabel('k')
  ax.set_ylabel('accuracy')
  return fig


@_safe_fig
def balanced_accuracy_plot(y_true, y_pred):
  '''Plot per-class recall (balanced accuracy bars).

  Args:
    y_true: True labels.
    y_pred: Predicted labels.

  Returns:
    Figure, or None on insufficient data.
  '''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  yt, yp = pairs
  labels = np.unique(yt)
  if len(labels) < 2:
    return None
  recalls = [np.mean(yp[yt == lab] == lab) for lab in labels]
  fig, ax = _new_fig('Balanced accuracy by class')
  ax.bar([str(int(v)) if float(v).is_integer() else str(v)
          for v in labels], recalls)
  ax.set_ylabel('recall per class')
  return fig


@_safe_fig
def pr_gain_curve(y_true, proba):
  '''Plot precision gain vs recall.

  Args:
    y_true: True labels.
    proba: Predicted probabilities.

  Returns:
    Figure, or None on insufficient data.
  '''
  from sklearn.metrics import precision_recall_curve
  scores = _proba_pos(proba)
  yt = np.asarray(y_true).ravel()
  if scores is None or yt.size != scores.size or len(np.unique(yt)) < 2:
    return None
  prec, rec, _ = precision_recall_curve(yt, scores)
  prevalence = max(np.mean(yt == 1), 1e-9)
  gain = prec / prevalence
  fig, ax = _new_fig('PR gain curve')
  ax.plot(rec, prec, label='precision')
  ax.plot(rec, gain, label='gain')
  ax.set_xlabel('recall')
  ax.set_ylabel('precision / gain')
  ax.legend()
  return fig


@_safe_fig
def residual_autocorr(y_true, y_pred, max_lag=5):
  '''Plot residual autocorrelation by lag.

  Args:
    y_true: True values.
    y_pred: Predicted values.
    max_lag: Maximum lag to compute.

  Returns:
    Figure, or None on insufficient data.
  '''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  resid = pairs[0] - pairs[1]
  resid = resid - resid.mean()
  denom = float(np.dot(resid, resid)) or 1.0
  lags = list(range(1, max(1, int(max_lag)) + 1))
  vals = [float(np.dot(resid[:-lag], resid[lag:]) / denom)
          for lag in lags if len(resid) > lag]
  if not vals:
    return None
  fig, ax = _new_fig('Residual autocorrelation')
  ax.bar([str(lag) for lag in lags[:len(vals)]], vals)
  ax.set_xlabel('lag')
  return fig


@_safe_fig
def cooks_distance(y_true, y_pred, x=None):
  '''Plot Cook's distance of regression residuals.

  Args:
    y_true: True values.
    y_pred: Predicted values.
    x: Optional design features used to build leverage.

  Returns:
    Figure, or None on insufficient data.
  '''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  yt, yp = pairs
  resid = yt - yp
  if x is not None:
    design = np.asarray(x, dtype=float)
    if design.ndim == 1:
      design = design.reshape(-1, 1)
    design = np.column_stack([np.ones(len(yt)), design])
  else:
    design = np.column_stack([np.ones(len(yt)), yp])
  p = design.shape[1]
  if len(yt) <= p + 1:
    return None
  hat = design @ np.linalg.pinv(design.T @ design) @ design.T
  leverage = np.clip(np.diag(hat), 0, 0.999)
  mse = float(np.mean(resid ** 2)) or 1.0
  cooks = (resid ** 2 / (p * mse)) * (leverage / (1 - leverage) ** 2)
  fig, ax = _new_fig('Cooks distance')
  ax.stem(cooks)
  ax.set_xlabel('observation')
  return fig


@_safe_fig
def rec_curve(y_true, y_pred):
  '''Plot the REC curve (accuracy within tolerance).

  Args:
    y_true: True values.
    y_pred: Predicted values.

  Returns:
    Figure, or None on insufficient data.
  '''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  err = np.abs(pairs[0] - pairs[1])
  tols = np.sort(err)
  acc = np.arange(1, len(err) + 1) / len(err)
  fig, ax = _new_fig('REC curve')
  ax.plot(tols, acc)
  ax.set_xlabel('tolerance')
  ax.set_ylabel('accuracy within tolerance')
  return fig


@_safe_fig
def prediction_interval_coverage(y_true, y_pred, coverage=0.95):
  '''Plot empirical coverage of residual-based intervals.

  Args:
    y_true: True values.
    y_pred: Predicted values.
    coverage: Nominal interval coverage level.

  Returns:
    Figure, or None on insufficient data.
  '''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  yt, yp = pairs
  try:
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + coverage / 2))
  except (ImportError, ValueError, TypeError, AttributeError) as exc:
    LOG.debug(
      'prediction_interval_coverage: fallback z (%s)', exc)
    z = 1.96
  resid = yt - yp
  half = z * float(np.std(resid) or 1.0)
  empirical = float(np.mean(np.abs(resid) <= half))
  fig, ax = _new_fig('Prediction interval coverage')
  ax.bar(['nominal', 'empirical'], [coverage, empirical])
  ax.set_ylim(0, 1)
  return fig


def _time_axis(dates, n):
  '''Return a plottable time axis from dates or index.'''
  if dates is not None and len(dates) == n:
    return dates
  return np.arange(n)


@_safe_fig
def residuals_over_time(y_true, y_pred, dates=None):
  '''Plot residuals over time (or sample order).

  Args:
    y_true: True values.
    y_pred: Predicted values.
    dates: Optional date/index axis.

  Returns:
    Figure, or None on insufficient data.
  '''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  fig, ax = _new_fig('Residuals over time')
  ax.plot(_time_axis(dates, len(pairs[0])), pairs[0] - pairs[1])
  ax.axhline(0, color='r')
  return fig


@_safe_fig
def forecast_vs_actual(y_true, y_pred, dates=None):
  '''Plot forecast vs actual over time (or sample order).

  Args:
    y_true: True values.
    y_pred: Predicted values.
    dates: Optional date/index axis.

  Returns:
    Figure, or None on insufficient data.
  '''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  axis = _time_axis(dates, len(pairs[0]))
  fig, ax = _new_fig('Forecast vs actual')
  ax.plot(axis, pairs[0], label='actual')
  ax.plot(axis, pairs[1], label='forecast')
  ax.legend()
  return fig


@_safe_fig
def learning_curve(train_scores, val_scores):
  '''Plot learning curves for train and validation scores.

  Args:
    train_scores: Training score per fold/epoch.
    val_scores: Validation score per fold/epoch.

  Returns:
    Figure, or None if fewer than two points.
  '''
  tr = np.asarray(train_scores, dtype=float).ravel()
  va = np.asarray(val_scores, dtype=float).ravel()
  if min(tr.size, va.size) < 2:
    return None
  n = min(tr.size, va.size)
  fig, ax = _new_fig('Learning curve')
  ax.plot(range(n), tr[:n], label='train')
  ax.plot(range(n), va[:n], label='validation')
  ax.legend()
  return fig


@_safe_fig
def validation_curve(param_values, train_scores, val_scores):
  '''Plot validation curve vs hyperparameter values.

  Args:
    param_values: Hyperparameter values.
    train_scores: Train scores per parameter value.
    val_scores: Validation scores per parameter value.

  Returns:
    Figure, or None if fewer than two points.
  '''
  pv = np.asarray(param_values, dtype=float).ravel()
  tr = np.asarray(train_scores, dtype=float).ravel()
  va = np.asarray(val_scores, dtype=float).ravel()
  if min(pv.size, tr.size, va.size) < 2:
    return None
  n = min(pv.size, tr.size, va.size)
  fig, ax = _new_fig('Validation curve')
  ax.plot(pv[:n], tr[:n], marker='o', label='train')
  ax.plot(pv[:n], va[:n], marker='s', label='validation')
  ax.legend()
  ax.set_xlabel('parameter')
  return fig


@_safe_fig
def loss_vs_epoch(train_loss, val_loss=None):
  '''Plot training (and validation) loss vs epoch.

  Args:
    train_loss: Loss per epoch.
    val_loss: Optional validation loss per epoch.

  Returns:
    Figure, or None if fewer than two epochs.
  '''
  tr = np.asarray(train_loss, dtype=float).ravel()
  if tr.size < 2:
    return None
  fig, ax = _new_fig('Loss vs epoch')
  ax.plot(tr, label='train')
  if val_loss is not None:
    va = np.asarray(val_loss, dtype=float).ravel()
    if va.size >= 2:
      ax.plot(va, label='validation')
  ax.legend()
  ax.set_xlabel('epoch')
  return fig


def _score_groups(cv_scores):
  '''Normalise CV scores into a name to scores dict.'''
  if isinstance(cv_scores, dict):
    out = {str(k): np.asarray(v, dtype=float).ravel()
           for k, v in cv_scores.items()}
    return {k: v for k, v in out.items() if v.size >= 2}
  arr = np.asarray(cv_scores, dtype=float).ravel()
  return {'scores': arr} if arr.size >= 2 else {}


@_safe_fig
def cv_scores_box(cv_scores):
  '''Plot box plots of cross-validation scores.

  Args:
    cv_scores: Dict of name to scores, or a score list.

  Returns:
    Figure, or None if no group has >=2 scores.
  '''
  groups = _score_groups(cv_scores)
  if not groups:
    return None
  fig, ax = _new_fig('CV scores')
  ax.boxplot(list(groups.values()),
             tick_labels=list(groups.keys()))
  ax.set_ylabel('score')
  return fig


@_safe_fig
def overfitting_gap(train_scores, val_scores):
  '''Plot the train-validation gap per point.

  Args:
    train_scores: Training scores.
    val_scores: Validation scores.

  Returns:
    Figure, or None if fewer than two points.
  '''
  tr = np.asarray(train_scores, dtype=float).ravel()
  va = np.asarray(val_scores, dtype=float).ravel()
  if min(tr.size, va.size) < 2:
    return None
  n = min(tr.size, va.size)
  gap = tr[:n] - va[:n]
  fig, ax = _new_fig('Overfitting gap')
  ax.bar(range(n), gap)
  ax.axhline(0, color='r')
  ax.set_xlabel('fold / epoch')
  return fig


@_safe_fig
def bias_variance(train_scores, val_scores):
  '''Plot bias-variance: mean scores with std bands.

  Args:
    train_scores: Training scores per complexity level.
    val_scores: Validation scores per complexity level.

  Returns:
    Figure, or None if fewer than two points.
  '''
  tr = np.asarray(train_scores, dtype=float).ravel()
  va = np.asarray(val_scores, dtype=float).ravel()
  if min(tr.size, va.size) < 2:
    return None
  n = min(tr.size, va.size)
  xs = np.arange(n)
  fig, ax = _new_fig('Bias-variance')
  ax.plot(xs, tr[:n], label='train')
  ax.fill_between(xs, tr[:n] - np.std(tr[:n]),
                  tr[:n] + np.std(tr[:n]), alpha=0.2)
  ax.plot(xs, va[:n], label='validation')
  ax.fill_between(xs, va[:n] - np.std(va[:n]),
                  va[:n] + np.std(va[:n]), alpha=0.2)
  ax.legend()
  return fig


def make_training_charts(history, params=None):
  '''Build all training-diagnostic charts.

  Args:
    history: Dict with train/val scores, losses,
      cv_scores and/or param_values entries.
    params: Optional params dict (fallback source).

  Returns:
    Dict of chart name to closed Figure.
  '''
  history = dict(history or {})
  params = dict(params or {})
  merged = dict(params, **history)
  figs = {}
  try:
    _safe(figs, 'learning_curve', learning_curve,
          merged.get('train_scores', []),
          merged.get('val_scores', []))
    _safe(figs, 'validation_curve', validation_curve,
          merged.get('param_values', []),
          merged.get('train_scores', []),
          merged.get('val_scores', []))
    _safe(figs, 'loss_vs_epoch', loss_vs_epoch,
          merged.get('train_loss', []), merged.get('val_loss'))
    _safe(figs, 'cv_scores_box', cv_scores_box,
          merged.get('cv_scores', {}))
    _safe(figs, 'overfitting_gap', overfitting_gap,
          merged.get('train_scores', []),
          merged.get('val_scores', []))
    _safe(figs, 'bias_variance', bias_variance,
          merged.get('train_scores', []),
          merged.get('val_scores', []))
    return figs
  finally:
    for fig in list(figs.values()):
      plt.close(fig)


def _shap_2d(shap_values):
  '''Return SHAP values as a finite 2D float array.'''
  arr = np.asarray(shap_values, dtype=float)
  if arr.ndim == 1:
    arr = arr.reshape(-1, 1)
  if arr.ndim != 2 or arr.shape[0] < 2:
    return None
  return arr


@_safe_fig
def shap_dependence(shap_values, x=None, feature_index=0):
  '''Plot SHAP dependence for one feature.

  Args:
    shap_values: 2D SHAP values (n, n_features).
    x: Optional feature matrix aligned with rows.
    feature_index: Feature column to display.

  Returns:
    Figure, or None on insufficient data.
  '''
  arr = _shap_2d(shap_values)
  if arr is None or not (0 <= feature_index < arr.shape[1]):
    return None
  vals = arr[:, feature_index]
  if x is not None:
    xs = np.asarray(x, dtype=float)
    if xs.ndim == 2 and xs.shape[0] == arr.shape[0] \
            and xs.shape[1] > feature_index:
      xs = xs[:, feature_index]
    else:
      xs = np.arange(arr.shape[0])
  else:
    xs = np.arange(arr.shape[0])
  fig, ax = _new_fig('SHAP dependence')
  ax.scatter(xs, vals, s=10, c=vals)
  ax.set_xlabel(f'feature {feature_index}')
  ax.set_ylabel('SHAP value')
  return fig


@_safe_fig
def shap_waterfall(shap_values, feature_names=None):
  '''Plot a waterfall-style bar chart for one row.

  Args:
    shap_values: 1D SHAP values for a single row.
    feature_names: Optional feature labels.

  Returns:
    Figure, or None on empty input.
  '''
  vals = np.asarray(shap_values, dtype=float).ravel()
  if vals.size < 1:
    return None
  order = np.argsort(-np.abs(vals))
  labels = list(feature_names) if feature_names is not None \
      else [f'f{i}' for i in range(vals.size)]
  labels = [str(labels[i]) if i < len(labels) else f'f{i}'
            for i in order]
  fig, ax = _new_fig('SHAP waterfall')
  ax.barh(labels, vals[order])
  ax.set_xlabel('SHAP value')
  return fig


@_safe_fig
def shap_force(shap_values, feature_names=None):
  '''Plot a simple SHAP force-style bar chart.

  Args:
    shap_values: 1D SHAP values for a single row.
    feature_names: Optional feature labels.

  Returns:
    Figure, or None on empty input.
  '''
  vals = np.asarray(shap_values, dtype=float).ravel()
  if vals.size < 1:
    return None
  labels = list(feature_names) if feature_names is not None \
      else [f'f{i}' for i in range(vals.size)]
  fig, ax = _new_fig('SHAP force')
  ax.bar([str(v) for v in labels[:vals.size]], vals)
  ax.set_ylabel('SHAP value')
  return fig


def _predict_scores(model, data):
  '''Return 1D scores from a fitted model.'''
  if hasattr(model, 'predict_proba'):
    proba = np.asarray(model.predict_proba(data), dtype=float)
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 \
        else proba.ravel()
  return np.asarray(model.predict(data), dtype=float).ravel()


@_safe_fig
def pdp_1d(model, x, feature_index=0):
  '''Plot 1D partial dependence for one feature.

  Args:
    model: Fitted model with predict/predict_proba.
    x: Feature matrix.
    feature_index: Feature to vary.

  Returns:
    Figure, or None on insufficient data.
  '''
  data = np.asarray(x, dtype=float)
  if data.ndim != 2 or data.shape[0] < 5:
    return None
  grid = np.quantile(data[:, feature_index],
                     np.linspace(0, 1, 20))
  means = []
  for value in np.unique(grid):
    copy = data.copy()
    copy[:, feature_index] = value
    means.append(float(np.mean(_predict_scores(model, copy))))
  fig, ax = _new_fig('Partial dependence')
  ax.plot(np.unique(grid), means)
  ax.set_xlabel(f'feature {feature_index}')
  return fig


@_safe_fig
def ice_plot(model, x, feature_index=0):
  '''Plot ICE curves for one feature (capped rows).

  Args:
    model: Fitted model with predict/predict_proba.
    x: Feature matrix.
    feature_index: Feature to vary.

  Returns:
    Figure, or None on insufficient data.
  '''
  data = np.asarray(x, dtype=float)
  if data.ndim != 2 or data.shape[0] < 5:
    return None
  grid = np.unique(np.quantile(data[:, feature_index],
                               np.linspace(0, 1, 10)))
  fig, ax = _new_fig('ICE plot')
  for row in data[:20]:
    preds = []
    for value in grid:
      copy = row.reshape(1, -1).copy()
      copy[0, feature_index] = value
      preds.append(float(_predict_scores(model, copy)[0]))
    ax.plot(grid, preds, alpha=0.4)
  ax.set_xlabel(f'feature {feature_index}')
  return fig


@_safe_fig
def lime_bar(importances, feature_names=None):
  '''Plot LIME-style feature importance bars.

  Args:
    importances: Dict of name to value, a LIME
      explainer exposing as_list(), or an array.
    feature_names: Optional labels for array input.

  Returns:
    Figure, or None on empty input.
  '''
  values = None
  labels = None
  if isinstance(importances, dict):
    labels = [str(k) for k in importances]
    values = np.asarray(list(importances.values()), dtype=float)
  elif hasattr(importances, 'as_list'):
    pairs = importances.as_list()
    labels = [str(k) for k, _ in pairs]
    values = np.asarray([v for _, v in pairs], dtype=float)
  else:
    values = np.asarray(importances, dtype=float).ravel()
    labels = list(feature_names) if feature_names is not None \
        else [f'f{i}' for i in range(values.size)]
  if values is None or values.size < 1:
    return None
  order = np.argsort(-np.abs(values))
  labels = [str(labels[i]) if i < len(labels) else f'f{i}'
            for i in order]
  fig, ax = _new_fig('LIME bar')
  ax.barh(labels, values[order])
  ax.set_xlabel('importance')
  return fig


@_safe_fig
def decision_boundary(x, y_true, model=None):
  '''Plot 2D decision boundary (top-2 features).

  Args:
    x: Feature matrix with at least 2 columns.
    y_true: True labels.
    model: Optional fitted model to draw contours.

  Returns:
    Figure, or None on insufficient data.
  '''
  data = np.asarray(x, dtype=float)
  yt = np.asarray(y_true).ravel()
  if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 5 \
          or data.shape[0] != yt.size:
    return None
  fig, ax = _new_fig('Decision boundary')
  if model is not None:
    mins = data[:, :2].min(axis=0)
    maxs = data[:, :2].max(axis=0)
    gx, gy = np.meshgrid(np.linspace(mins[0], maxs[0], 20),
                         np.linspace(mins[1], maxs[1], 20))
    try:
      grid = np.column_stack([gx.ravel(), gy.ravel()])
      scores = _predict_scores(model, grid)
    except Exception as exc:  # noqa: BLE001 - retry with full features
      LOG.debug('decision_boundary: full-feature retry (%s)', exc)
      full = np.tile(np.median(data, axis=0), (grid.shape[0], 1))
      full[:, :2] = np.column_stack([gx.ravel(), gy.ravel()])
      scores = _predict_scores(model, full)
    ax.contourf(gx, gy, scores.reshape(gx.shape), alpha=0.3)
  ax.scatter(data[:, 0], data[:, 1], s=8, c=yt)
  ax.set_xlabel('feature 0')
  ax.set_ylabel('feature 1')
  return fig


@_safe_fig
def tree_plot(model, feature_names=None):
  '''Plot a fitted sklearn decision tree.

  Args:
    model: Fitted tree estimator exposing tree_.
    feature_names: Optional feature labels.

  Returns:
    Figure, or None if the model has no tree_.
  '''
  from sklearn.tree import plot_tree
  if not hasattr(model, 'tree_'):
    return None
  fig, ax = plt.subplots()
  plot_tree(model, ax=ax, feature_names=feature_names,
            filled=True)
  ax.set_title('Decision tree')
  return fig


@_safe_fig
def model_comparison_box(cv_scores):
  '''Plot box plots comparing model CV scores.

  Args:
    cv_scores: Dict of model name to CV scores.

  Returns:
    Figure, or None if no group has >=2 scores.
  '''
  groups = _score_groups(cv_scores)
  if not groups:
    return None
  fig, ax = _new_fig('Model comparison')
  ax.boxplot(list(groups.values()),
             tick_labels=list(groups.keys()))
  ax.set_ylabel('CV score')
  return fig


@_safe_fig
def bland_altman(pred_a, pred_b):
  '''Plot a Bland-Altman agreement chart.

  Args:
    pred_a: First model predictions.
    pred_b: Second model predictions.

  Returns:
    Figure, or None on insufficient data.
  '''
  pairs = _pairs(pred_a, pred_b)
  if pairs is None:
    return None
  mean = (pairs[0] + pairs[1]) / 2
  diff = pairs[0] - pairs[1]
  bias, sd = float(np.mean(diff)), float(np.std(diff))
  fig, ax = _new_fig('Bland-Altman')
  ax.scatter(mean, diff, s=8)
  ax.axhline(bias, color='r', label='bias')
  ax.axhline(bias + 1.96 * sd, color='r', ls='--')
  ax.axhline(bias - 1.96 * sd, color='r', ls='--')
  ax.set_xlabel('mean of predictions')
  ax.set_ylabel('difference')
  ax.legend()
  return fig


def _pred_matrix(models_preds):
  '''Return aligned prediction matrix from a dict.'''
  if not isinstance(models_preds, dict) or len(models_preds) < 2:
    return None
  names = list(models_preds)
  arrays = [np.asarray(models_preds[n], dtype=float).ravel()
            for n in names]
  size = min(arr.size for arr in arrays)
  if size < 5:
    return None
  return names, np.column_stack([a[:size] for a in arrays])


@_safe_fig
def ensemble_diversity(models_preds):
  '''Plot the ensemble agreement/correlation heatmap.

  Args:
    models_preds: Dict of model name to predictions.

  Returns:
    Figure, or None if fewer than two models.
  '''
  built = _pred_matrix(models_preds)
  if built is None:
    return None
  names, matrix = built
  discrete = all(len(np.unique(matrix[:, i])) <= 10
                 for i in range(matrix.shape[1]))
  if discrete:
    sim = np.zeros((len(names), len(names)))
    for i in range(len(names)):
      for j in range(len(names)):
        sim[i, j] = np.mean(matrix[:, i] == matrix[:, j])
  else:
    sim = np.corrcoef(matrix.T)
  fig, ax = _new_fig('Ensemble diversity')
  ax.imshow(sim, cmap='Blues', vmin=0, vmax=1)
  ax.set_xticks(range(len(names)))
  ax.set_xticklabels(names, rotation=45)
  ax.set_yticks(range(len(names)))
  ax.set_yticklabels(names)
  return fig


@_safe_fig
def roc_overlay(y_true, probas):
  '''Plot overlaid ROC curves for multiple models.

  Args:
    y_true: True labels.
    probas: Dict of model name to probabilities.

  Returns:
    Figure, or None if fewer than two models.
  '''
  from sklearn.metrics import roc_auc_score, roc_curve
  if not isinstance(probas, dict) or len(probas) < 2:
    return None
  yt = np.asarray(y_true).ravel()
  if len(np.unique(yt)) < 2:
    return None
  fig, ax = _new_fig('ROC overlay')
  for name, proba in probas.items():
    scores = _proba_pos(proba)
    if scores is None or scores.size != yt.size:
      continue
    fpr, tpr, _ = roc_curve(yt, scores)
    ax.plot(fpr, tpr,
            label=f'{name} auc={roc_auc_score(yt, scores):.3f}')
  if not ax.lines:
    plt.close(fig)
    return None
  ax.plot([0, 1], [0, 1], '--')
  ax.legend()
  return fig


@_safe_fig
def pr_overlay(y_true, probas):
  '''Plot overlaid PR curves for multiple models.

  Args:
    y_true: True labels.
    probas: Dict of model name to probabilities.

  Returns:
    Figure, or None if fewer than two models.
  '''
  from sklearn.metrics import precision_recall_curve
  if not isinstance(probas, dict) or len(probas) < 2:
    return None
  yt = np.asarray(y_true).ravel()
  if len(np.unique(yt)) < 2:
    return None
  fig, ax = _new_fig('PR overlay')
  for name, proba in probas.items():
    scores = _proba_pos(proba)
    if scores is None or scores.size != yt.size:
      continue
    prec, rec, _ = precision_recall_curve(yt, scores)
    ax.plot(rec, prec, label=str(name))
  if not ax.lines:
    plt.close(fig)
    return None
  ax.set_xlabel('recall')
  ax.set_ylabel('precision')
  ax.legend()
  return fig


@_safe_fig
def error_correlation_heatmap(models_preds, y_true):
  '''Plot the correlation heatmap of model errors.

  Args:
    models_preds: Dict of model name to predictions.
    y_true: True values aligned with each prediction.

  Returns:
    Figure, or None if fewer than two models.
  '''
  built = _pred_matrix(models_preds)
  if built is None:
    return None
  names, matrix = built
  yt = np.asarray(y_true, dtype=float).ravel()[:matrix.shape[0]]
  if yt.size != matrix.shape[0]:
    return None
  errors = np.abs(matrix - yt.reshape(-1, 1))
  corr = np.corrcoef(errors.T)
  fig, ax = _new_fig('Error correlation')
  ax.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
  ax.set_xticks(range(len(names)))
  ax.set_xticklabels(names, rotation=45)
  ax.set_yticks(range(len(names)))
  ax.set_yticklabels(names)
  return fig


def make_comparison_charts(models_preds, y_true, params=None):
  '''Build all model-comparison charts.

  Args:
    models_preds: Dict of model name to predictions.
    y_true: True values/labels.
    params: Optional dict with probas and cv_scores.

  Returns:
    Dict of chart name to closed Figure.
  '''
  models_preds = models_preds or {}
  params = dict(params or {})
  probas = params.get('probas') or params.get('models_probas')
  if probas is None and models_preds:
    first = next(iter(models_preds.values()))
    if np.asarray(first).ndim == 2:
      probas = models_preds
  figs = {}
  try:
    _safe(figs, 'model_comparison_box', model_comparison_box,
          params.get('cv_scores', {}))
    names = list(models_preds)
    if len(names) >= 2:
      _safe(figs, 'bland_altman', bland_altman,
            models_preds[names[0]], models_preds[names[1]])
    _safe(figs, 'ensemble_diversity', ensemble_diversity, models_preds)
    if probas:
      _safe(figs, 'roc_overlay', roc_overlay, y_true, probas)
      _safe(figs, 'pr_overlay', pr_overlay, y_true, probas)
    _safe(figs, 'error_correlation_heatmap',
          error_correlation_heatmap, models_preds, y_true)
    return figs
  finally:
    for fig in list(figs.values()):
      plt.close(fig)


@_safe_fig
def quantile_coverage(y_true, quantile_preds):
  '''Plot empirical coverage per predicted quantile.

  Args:
    y_true: True values.
    quantile_preds: Dict of quantile level to preds.

  Returns:
    Figure, or None on insufficient data.
  '''
  if not isinstance(quantile_preds, dict) or not quantile_preds:
    return None
  yt = np.asarray(y_true, dtype=float).ravel()
  if yt.size < 5:
    return None
  levels, coverages = [], []
  for level, preds in sorted(quantile_preds.items()):
    arr = np.asarray(preds, dtype=float).ravel()
    if arr.size != yt.size:
      continue
    levels.append(float(level))
    coverages.append(float(np.mean(yt <= arr)))
  if len(levels) < 2:
    return None
  fig, ax = _new_fig('Quantile coverage')
  ax.plot(levels, coverages, marker='o')
  ax.plot([0, 1], [0, 1], '--')
  ax.set_xlabel('nominal quantile')
  ax.set_ylabel('empirical coverage')
  return fig


@_safe_fig
def prediction_interval_vs_actual(y_true, lower, upper, y_pred=None):
  '''Plot prediction intervals against actual values.

  Args:
    y_true: True values.
    lower: Lower interval bound per observation.
    upper: Upper interval bound per observation.
    y_pred: Optional point forecasts.

  Returns:
    Figure, or None on insufficient data.
  '''
  yt = np.asarray(y_true, dtype=float).ravel()
  lo = np.asarray(lower, dtype=float).ravel()
  hi = np.asarray(upper, dtype=float).ravel()
  if yt.size < 5 or lo.size != yt.size or hi.size != yt.size:
    return None
  xs = np.arange(yt.size)
  fig, ax = _new_fig('Prediction interval vs actual')
  ax.fill_between(xs, lo, hi, alpha=0.3, label='interval')
  ax.plot(xs, yt, label='actual')
  if y_pred is not None:
    ax.plot(xs, np.asarray(y_pred, dtype=float).ravel(),
            label='forecast')
  ax.legend()
  return fig


def _add_optional_groups(figs, y_true, y_pred, x_test, params):
  '''Add optional chart groups when params provide data.

  Args:
    figs: Dict of chart name to Figure, updated in place.
    y_true: True labels or values.
    y_pred: Predicted labels or values.
    x_test: Test features.
    params: modeling_params dict.

  Returns:
    None.
  '''
  history = params.get('training_history') or params.get('history')
  if history:
    _merge(figs, make_training_charts(history, params))
  target = params.get('target')
  frame = params.get('df', x_test)
  if target and hasattr(frame, 'columns') and target in frame.columns:
    _merge(figs, make_data_charts(frame, target, params))
  models_preds = params.get('models_preds')
  if models_preds:
    _merge(figs, make_comparison_charts(models_preds, y_true, params))
  shap_values = params.get('shap_values')
  if shap_values is not None:
    names = params.get('shap_feature_names')
    _safe(figs, 'shap_dependence', shap_dependence, shap_values,
          _numeric_df(x_test).values if x_test is not None else None)
    row = np.asarray(shap_values, dtype=float)
    row = row[0] if row.ndim == 2 and row.shape[0] else row
    _safe(figs, 'shap_waterfall', shap_waterfall, row, names)
    _safe(figs, 'shap_force', shap_force, row, names)
  model = params.get('model')
  if model is not None and x_test is not None:
    x_vals = _numeric_df(x_test).values
    _safe(figs, 'pdp_1d', pdp_1d, model, x_vals)
    _safe(figs, 'ice_plot', ice_plot, model, x_vals)
    _safe(figs, 'decision_boundary', decision_boundary,
          x_vals, y_true, model)
    _safe(figs, 'tree_plot', tree_plot, model,
          params.get('shap_feature_names'))
  if params.get('lime_importances') is not None:
    _safe(figs, 'lime_bar', lime_bar, params['lime_importances'])
  intervals = params.get('prediction_intervals')
  if isinstance(intervals, dict) and 'lower' in intervals \
          and 'upper' in intervals:
    _safe(figs, 'prediction_interval_vs_actual',
          prediction_interval_vs_actual, y_true,
          intervals['lower'], intervals['upper'], y_pred)
  if params.get('quantile_preds'):
    _safe(figs, 'quantile_coverage', quantile_coverage, y_true,
          params['quantile_preds'])


def make_charts(y_true, y_pred, y_proba, x_test, params):
  '''Build all charts for the task type.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.
    y_proba: Probabilities or None.
    x_test: Test features.
    params: modeling_params dict.

  Returns:
    Dict of chart name to matplotlib Figure. Charts whose
    data are insufficient are skipped gracefully. Every
    figure is closed before returning, so no figure leaks;
    callers that need the image must save it inside this
    call (figures are returned closed, not open).
  '''
  params = params or {}
  task = normalize_task_kind(params.get('task_type', 'classification'))
  figs = {}
  if task == 'classification':
    if y_proba is not None:
      proba = np.asarray(y_proba)
      _safe(figs, 'roc', _roc, y_true, proba)
      _safe(figs, 'pr', _pr, y_true, proba)
      _safe(figs, 'calibration_curve', _calibration, y_true, proba)
      _safe(figs, 'threshold_curve', _threshold, y_true, proba)
      _safe(figs, 'lift_gain', _lift, y_true, proba)
      _safe(figs, 'ks_plot', _ks_plot, y_true, proba)
      _safe(figs, 'prob_distribution', _prob_dist, y_true, proba)
      _safe(figs, 'det_curve', det_curve, y_true, proba)
      _safe(figs, 'topk_accuracy_plot', topk_accuracy_plot,
            y_true, proba)
      _safe(figs, 'pr_gain_curve', pr_gain_curve, y_true, proba)
    _safe(figs, 'confusion_heatmap', _confusion, y_true, y_pred)
    _safe(figs, 'per_class_bars', _per_class_bars, y_true, y_pred)
    _safe(figs, 'confusion_wheel', confusion_wheel, y_true, y_pred)
    _safe(figs, 'balanced_accuracy_plot', balanced_accuracy_plot,
          y_true, y_pred)
  else:
    resid = np.asarray(y_true, dtype=float) - np.asarray(
        y_pred, dtype=float)
    _safe(figs, 'pred_vs_actual', _pred_vs_actual, y_true, y_pred)
    _safe(figs, 'residuals_vs_fitted', _residuals_fitted,
          y_pred, resid)
    _safe(figs, 'residuals_vs_actual', _residuals_actual,
          y_true, resid)
    _safe(figs, 'residual_hist_qq', _residual_hist_qq,
          y_true, y_pred)
    _safe(figs, 'scale_location', _scale_location, y_true, y_pred)
    _safe(figs, 'error_vs_actual', _error_actual, y_true, y_pred)
    _safe(figs, 'error_by_bucket', _error_bucket,
          y_true, y_pred, x_test, params)
    _safe(figs, 'segment_curves', _segment_curves,
          y_true, y_pred, x_test, params)
    _safe(figs, 'feature_vs_residual', _feature_residual,
          y_true, y_pred, x_test)
    _safe(figs, 'residual_autocorr', residual_autocorr,
          y_true, y_pred)
    _safe(figs, 'cooks_distance', cooks_distance, y_true, y_pred)
    _safe(figs, 'rec_curve', rec_curve, y_true, y_pred)
    _safe(figs, 'prediction_interval_coverage',
          prediction_interval_coverage, y_true, y_pred)
    dates = params.get('dates')
    _safe(figs, 'residuals_over_time', residuals_over_time,
          y_true, y_pred, dates)
    _safe(figs, 'forecast_vs_actual', forecast_vs_actual,
          y_true, y_pred, dates)
  _add_optional_groups(figs, y_true, y_pred, x_test, params)
  try:
    return figs
  finally:
    for fig in list(figs.values()):
      plt.close(fig)
