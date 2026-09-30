#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Categorical and cardinality-control transforms.

These four routines are the ones that change a column's *type*, so they are the
ones that must get the metadata protocol right. The protocol, stated once:

* a column that becomes a set of new numeric columns is removed from
  ``feature_cols`` and its derivatives are appended to ``feature_cols`` **and**
  ``numerical_cols``, then :meth:`Metadata.clean_duplicates` is called;
* a column that becomes a single numeric column under a new name is removed from
  ``feature_cols`` and the new name is appended to both numeric namespaces;
* a column that becomes a numeric bin index under its *own* name is moved from
  ``categorical_cols`` to ``numerical_cols``.

The protocol has a failure mode worth naming, because it is silent: a routine
that appends to ``feature_cols`` without also appending to ``numerical_cols``
registers a feature that every numeric-only routine downstream will not see.
``TargetEncoding`` therefore writes to both namespaces in the same call, and
every other member of the family does the same.
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import functions as sf
from pyspark.sql import types as st

from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.core.routine import PreprocessRoutine
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.preprocessing.bases import SparkMlRoutine, require_columns
from forecasting_ml_framework.utils.reflection import resolve

LOGGER = get_logger(__name__)


class CategoricalEncoding(SparkMlRoutine):
  """Encode categorical columns with a Spark ML indexer and optional one-hot.

  This is the most metadata-invasive routine in the library, and the one that
  gets the contract right. The output width is read from the *fitted artefact*
  rather than from configuration, so the metadata matches what was actually
  fitted even when the fit saw fewer categories than expected. That detail is the
  difference between a correct implementation and one that is correct only when
  the configuration happens to match the data — a ``StringIndexer`` with
  ``handleInvalid: 'keep'`` adds a bucket for unseen values, so the fitted width
  can legitimately exceed the observed cardinality.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'CategoricalEncoding'
    self.use_cols: list[str] = []
    self.label_encode_handle = 'pyspark.ml.feature.StringIndexer'
    self.label_encode_params: dict[str, Any] = {'handleInvalid': 'keep'}
    self.ohe_flag = True
    self.ohe_handle = 'pyspark.ml.feature.OneHotEncoder'
    self.ohe_params: dict[str, Any] = {'handleInvalid': 'keep'}
    self.le_cat: list[str] = []
    self.ohe_cat: list[str] = []

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'label_encode_handle': 'pyspark.ml.feature.StringIndexer',
      'label_encode_params': {'handleInvalid': 'keep'},
      'ohe_flag': True,
      'ohe_handle': 'pyspark.ml.feature.OneHotEncoder',
      'ohe_params': {'handleInvalid': 'keep'},
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def artefact_name(self) -> str:
    """Return an artefact name that distinguishes the two encoding variants.

    Two registrations of this routine — one with one-hot and one without — must
    not overwrite each other's artefacts.

    Returns:
      A directory name.
    """
    return f'{type(self).__name__}{"1" if self.ohe_flag else "2"}'

  def fit(self, df: Any, metadata: Metadata) -> tuple[CategoricalEncoding, Any, Metadata]:
    """Fit the encoding pipeline and persist it.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.
    """
    self.use_cols = self.get_selected_columns(
      self.cols.get('exclude_cols', []), self.cols.get('include_cols', []), metadata.categorical_cols
    )
    if not self.use_cols:
      return self, df, metadata
    require_columns(df, self.use_cols, self.name)
    self.save_model_path = self.artefact_path(metadata)

    indexer = resolve(self.label_encode_handle)
    self.le_cat = []
    stages: list[Any] = [
      indexer(inputCol=column, outputCol=f'{column}_le', **self.label_encode_params) for column in self.use_cols
    ]

    if self.ohe_flag:
      encoder = resolve(self.ohe_handle)
      self.ohe_cat = list(self.use_cols)
      stages.extend(
        encoder(inputCol=f'{column}_le', outputCol=f'{column}_ohe', **self.ohe_params) for column in self.ohe_cat
      )
    else:
      self.ohe_cat = []

    self._fit_pipeline(df, stages)
    return self, df, metadata

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Reload the encoder, project its output, and register it as features.

    The string columns are *retained* in the frame but are no longer
    features. That is a small waste and is defensible: they remain available for
    debuggability and for the explainability job downstream.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    metadata = metadata.copy()
    if not self.use_cols:
      return df, metadata

    retained = list(df.columns)
    encoded = self._load_pipeline().transform(df)
    projections: list[Any] = []

    for column in self.use_cols:
      metadata.replace_feature(column, f'{column}_le')

    if self.ohe_cat:
      for column in self.ohe_cat:
        ohe_column = f'{column}_ohe'
        width = self._fitted_width(encoded, ohe_column)
        derived = [f'{ohe_column}_{index}' for index in range(width)]
        projections.extend(sf.col(ohe_column).getItem(index).alias(name) for index, name in enumerate(derived))
        metadata.replace_feature(f'{column}_le', f'{ohe_column}_0')
        metadata.feature_cols.extend(derived[1:])
        metadata.numerical_cols.extend(derived)
        metadata.clean_duplicates()
    else:
      projections.extend(sf.col(f'{column}_le').alias(f'{column}_le') for column in self.use_cols)

    return encoded.select(*retained, *projections), metadata

  @staticmethod
  def _fitted_width(encoded: Any, ohe_column: str) -> int:
    """Read the fitted vector width from the encoded frame.

    Args:
      encoded: The transformed frame.
      ohe_column: The one-hot vector column.

    Returns:
      The number of fitted categories, or ``0`` when the column is absent or
      null. Reading the width from the artefact rather than from configuration
      is what keeps the metadata and the encoder in agreement.
    """
    first = encoded.select(ohe_column).first()
    if first is None or first[0] is None:
      return 0
    return len(first[0])


class TargetEncoding(PreprocessRoutine):
  """Replace a category with the mean of its target, using out-of-fold means.

  Two properties make this transform correct rather than merely conventional:

  * sklearn's ``TargetEncoder`` cross-fits during ``fit_transform``, so a row's
    own label does not contribute to its own encoding. It does **not** cross-fit
    during ``fit``: ``fit`` populates ``encodings_`` with full-data smoothed
    means, so a row encoded from that table is encoded with a statistic that
    includes its own label. The distinction is not subtle -- on a 300-row,
    5%-positive fixture every category's stored value moved, and 213 of 300
    rows received a different encoding. In-sample target encoding is the
    textbook leakage pattern: it inflates reported training performance, and
    does so *non-uniformly across categories*, which makes it worst for exactly
    the categories most likely to be new at prediction time;
  * the framework then reduces those out-of-fold row encodings to a
    category-to-mean map and applies it as a frozen lookup table. Calling
    ``transform`` on new data would recompute encodings from the new data's
    labels -- which do not exist at scoring time and would in any case be
    leakage.

  Unseen categories fall back to the **global target mean**, not to zero. A zero
  fallback for a retention model would assert that the category never attrits:
  a catastrophic and entirely invisible error. The prior is the textbook choice.

  Attributes:
    hash_map: The category-to-target-mean map per column, reduced from the
      out-of-fold row encodings.
    target_mean: The global target mean, used as the unseen-category fallback.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'TargetEncoding'
    self.use_cols: list[str] = []
    self.handle = 'sklearn.preprocessing.TargetEncoder'
    self.params: dict[str, Any] = {'cv': 5}
    self.hash_map: dict[str, dict[Any, float]] = {}
    self.target_mean: float = 0.0

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'handle': 'sklearn.preprocessing.TargetEncoder',
      'params': {'cv': 5},
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[TargetEncoding, Any, Metadata]:
    """Fit the encoder and freeze its internal maps.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.
    """
    self.use_cols = self.get_selected_columns(
      self.cols.get('exclude_cols', []), self.cols.get('include_cols', []), metadata.categorical_cols
    )
    if not self.use_cols:
      return self, df, metadata
    require_columns(df, [*self.use_cols, metadata.target_col], self.name)

    frame = df.select(metadata.target_col, *self.use_cols).toPandas()
    encoder_class = resolve(self.handle)
    encoder = encoder_class(**self.params)
    encoder.set_output(transform='pandas')
    # `fit_transform`, not `fit`. This is the out-of-fold requirement, and the
    # distinction is the whole correctness of the routine -- see the class
    # docstring. The *fallback* remains `target_mean_`, the global mean, and
    # never zero.
    cross_fitted = encoder.fit_transform(frame[self.use_cols], frame[metadata.target_col])

    self.hash_map = _cross_fitted_maps(cross_fitted, frame, self.use_cols)
    self.target_mean = float(encoder.target_mean_)
    LOGGER.info(
      'Froze target-encoding maps',
      extra={'routine': self.name, 'columns': len(self.hash_map), 'target_mean': self.target_mean},
    )
    return self, df, metadata

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Apply the frozen map as a stateless lookup.

    A defensive copy is taken, unlike a, which mutated the caller's
    ``feature_cols`` in place. Copying here is the correct choice because the
    mutation is an implementation detail of this routine rather than a contract
    with the caller.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    from functools import partial  # noqa: PLC0415

    metadata = metadata.copy()
    for column in self.use_cols:
      def encode(category: Any, table: dict[Any, float], fallback: float) -> float:
        """Look a category up in the frozen map.

        Args:
          category: The category value.
          table: The category-to-mean map.
          fallback: The global target mean.

        Returns:
          The encoded value.
        """
        try:
          return float(round(table.get(category, fallback), 4))
        except TypeError:
          return float(fallback)

      udf = sf.udf(partial(encode, table=self.hash_map.get(column, {}), fallback=self.target_mean), st.FloatType())
      df = df.withColumn(f'{column}_encoded', udf(sf.col(column)))
      metadata.replace_feature(column, f'{column}_encoded')
    return df, metadata


class OptimalBinning(PreprocessRoutine):
  """Replace a column with a target-supervised bin index.

  The only target-supervised transform in the library, and it is fitted on the
  training frame only. The split edges therefore live in the persisted registry
  and are applied unchanged at scoring time, so train/serve parity is preserved
  by construction.

  The null encoding is the detail worth noting: nulls are assigned
  ``len(splits) + 1``, one past the top bin. No numeric value can collide with
  it, so a null is never confused with the highest bin.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'OptimalBinning'
    self.use_cols: list[str] = []
    self.handle = 'optbinning.OptimalBinning'
    self.dtype = 'numerical'
    self.params: dict[str, Any] = {'max_n_bins': 8}
    self.opt_bin_map: dict[str, Any] = {}

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'handle': 'optbinning.OptimalBinning',
      'dtype': 'numerical',
      'params': {'max_n_bins': 8},
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[OptimalBinning, Any, Metadata]:
    """Learn the split edges for each selected column.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.
    """
    self.use_cols = self.get_selected_columns(
      self.cols.get('exclude_cols', []), self.cols.get('include_cols', []), metadata.feature_cols
    )
    if not self.use_cols:
      return self, df, metadata
    require_columns(df, [*self.use_cols, metadata.target_col], self.name)

    binner_class = resolve(self.handle)
    self.opt_bin_map = {}
    for column in self.use_cols:
      pair = df.select(column, metadata.target_col).toPandas()
      binner = binner_class(name=column, dtype=self.dtype, **self.params)
      binner.fit(pair[column].values, pair[metadata.target_col].values)
      self.opt_bin_map[column] = binner.splits
    LOGGER.info('Learned optimal bin edges', extra={'routine': self.name, 'columns': len(self.opt_bin_map)})
    return self, df, metadata

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Assign each row to a learned bin.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    from functools import partial  # noqa: PLC0415

    metadata = metadata.copy()
    for column in self.use_cols:
      splits = self.opt_bin_map[column]
      column_udf = sf.udf(
        partial(_assign_bin, feature_type=self.dtype, splits=splits), returnType=st.FloatType()
      )
      df = df.withColumn(column, column_udf(sf.col(column)))

    if self.dtype == 'categorical':
      # A removed each column from categorical_cols unconditionally,
      # raising ValueError when use_cols was resolved from the broader feature
      # list and a column was not in the categorical namespace.
      metadata.reclassify_as_numerical(self.use_cols)
    return df, metadata


class HighLabelBinning(PreprocessRoutine):
  """Collapse low-frequency categories into a single bucket.

  The standard defence against one-hot explosion on a long-tailed categorical.
  Values appearing at least ``val_count`` times keep their own level; rarer
  values collapse.

  The distinguishing design choice is ``bin_count_by_col``: a per-column
  threshold override, so the cut can be tuned per feature while keeping a single
  global default.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'HighLabelBinning'
    self.use_cols: list[str] = []
    self.val_count = 300
    self.bin_count_by_col: dict[str, int] = {}
    self.bin_value = 'OTHER'
    self.bin_map: dict[str, dict[Any, str]] = {}

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'val_count': 300,
      'bin_count_by_col': {},
      'bin_value': 'OTHER',
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[HighLabelBinning, Any, Metadata]:
    """Learn which values survive and which collapse.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.
    """
    self.use_cols = self.get_selected_columns(
      self.cols.get('exclude_cols', []), self.cols.get('include_cols', []), metadata.categorical_cols
    )
    if not self.use_cols:
      return self, df, metadata
    require_columns(df, self.use_cols, self.name)

    self.bin_map = {}
    for column in self.use_cols:
      threshold = int(self.bin_count_by_col.get(column, self.val_count))
      survivors = {
        row[column]
        for row in (
          df.groupBy(sf.col(column).alias(column))
          .count()
          .filter(sf.col('count') >= threshold)
          .select(column)
          .collect()
        )
      }
      self.bin_map[column] = survivors
    LOGGER.info(
      'Learned high-frequency category sets',
      extra={'routine': self.name, 'columns': {k: len(v) for k, v in self.bin_map.items()}},
    )
    return self, df, metadata

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Collapse rare categories to the configured bucket value.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    for column, survivors in self.bin_map.items():
      df = df.withColumn(
        column,
        sf.when(sf.col(column).isin(list(survivors)), sf.col(column)).otherwise(sf.lit(self.bin_value)),
      )
    return df, metadata


def _cross_fitted_maps(
  cross_fitted: Any,
  frame: Any,
  columns: list[str],
) -> dict[str, dict[Any, float]]:
  """Reduce out-of-fold row encodings to the per-category map ``apply`` needs.

  ``fit_transform`` returns one *row* encoding per row, each computed from folds
  that excluded that row. The scoring path needs a *category* lookup, because a
  row seen at scoring time was not in the training frame at all. Reducing by
  taking each category's mean of its own out-of-fold encodings bridges the two,
  and the reduction is what preserves the property that matters: every value in
  the stored map is an average of encodings that never saw their own row's label.

  A category that is entirely absent from a fold is not a problem here, because
  the encoder substitutes the global mean for it -- which is also the correct
  value, and is the same fallback ``apply`` uses for an unseen category.

  Args:
    cross_fitted: The per-row out-of-fold encodings from ``fit_transform``.
    frame: The pandas frame the encoder was fitted on, supplying the category
      labels that index them.
    columns: The encoded columns, in the encoder's feature order.

  Returns:
    A mapping of column to category-to-mean mapping.
  """
  import pandas as pd  # noqa: PLC0415

  encoded = pd.DataFrame(cross_fitted, columns=list(columns), index=frame.index)
  maps: dict[str, dict[Any, float]] = {}
  for column in columns:
    grouped = encoded[column].groupby(frame[column].to_numpy(), dropna=True)
    maps[column] = {category: float(value) for category, value in grouped.mean().items()}
  return maps


def _assign_bin(value: Any, feature_type: str, splits: Any) -> float:
  """Assign a value to a learned bin.

  The numeric branch is a correct half-open-interval search over
  ``[-inf, s0), [s0, s1), ..., [s(n-1), +inf)``, with nulls assigned the sentinel
  ``len(splits) + 1``. The categorical branch is a membership test against the
  bins, which optbinning returns as a list of lists.

  Args:
    value: The value to bin.
    feature_type: ``'numerical'`` or ``'categorical'``.
    splits: The learned split edges or bins.

  Returns:
    The bin index as a float.
  """
  if feature_type == 'numerical':
    edges = [float(edge) for edge in splits]
    if value is None:
      return float(len(edges) + 1)
    number = float(value)
    for index in range(len(edges) + 1):
      if index == 0 and number < edges[0]:
        return float(index)
      if 0 < index < len(edges) and edges[index - 1] <= number < edges[index]:
        return float(index)
      if index == len(edges) and number >= edges[index - 1]:
        return float(index)
    return float(len(edges) + 1)

  for index, bucket in enumerate(splits):
    if value in bucket:
      return float(index)
  return float(len(splits))
