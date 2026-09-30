#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scalar and row-wise feature transforms.

Eleven transforms that operate on scalar columns. The shared members of the
template live on the base classes in :mod:`preprocessing.bases`; what remains
here is the genuinely per-routine logic.

Four decisions in this family are load-bearing and easy to get wrong:

* **``strategy: 'none'`` is a genuine no-op.** It is not an alias for "fill with
  zero". Zero-filling a *categorical string* column inserts the literal
  ``'0.0'`` as a new category downstream, which is both the opposite of what the
  strategy name says and invisible in every aggregate metric. A routine that
  falls through to a default bound would do exactly that, so the no-op branch is
  explicit.
* **The mode is deterministic.** Ties are broken by ascending value rather than
  by whatever order the aggregation happens to return. For a binary or one-hot
  column, where two values commonly share a count, a nondeterministic mode means
  two training runs of the same model can produce different models.
* **A transform that emits a new column registers it.** A routine that appends to
  a role list without emitting the column produces an invisible feature; one that
  emits without appending produces a column the trainer silently ignores. Both
  halves of the metadata protocol are therefore always performed together.
* **Scaling is applied in place.** The scaler operates on a ``Vector``, so the
  values are assembled, scaled, and exploded back into their named columns. The
  assembler keeps partially-null rows rather than skipping them, so a single
  null cannot null out an entire feature block.
"""

from __future__ import annotations

from typing import Any

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.core.routine import PreprocessRoutine
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.preprocessing.bases import SparkMlRoutine, require_columns
from forecasting_ml_framework.utils.reflection import resolve
from forecasting_ml_framework.utils.storage import load_pickle, save_pickle

LOGGER = get_logger(__name__)

#: Declared column role for each scalar routine. The third argument of
#: ``get_selected_columns`` is the semantically meaningful choice in the family,
#: so it is declared here once rather than repeated at each call site.
_SCOPE_FEATURES = 'feature_cols'
_SCOPE_NUMERICAL = 'numerical_cols'


class Imputations(PreprocessRoutine):
  """Fill missing values from a learned or configured bound.

  One aggregation pass computes every column's bound regardless of how many
  columns are selected, which is a deliberate avoidance of the per-column
  aggregation pattern.

  Attributes:
    bounds: The learned fill value per column. It is small and picklable, so it
      travels inside the persisted registry.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'Imputations'
    self.use_cols: list[str] = []
    self.bounds: dict[str, Any] = {}

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'strategy': 'median',
      'custom_bounds': {},
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[Imputations, Any, Metadata]:
    """Learn the fill bound for each selected column.

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
    require_columns(df, self.use_cols, self.name)

    self.bounds = self._learn_bounds(df, self.use_cols)
    if self.custom_bounds:
      # A configured bound fully overrides a computed one: the computed entry is
      # discarded rather than merged, so a sentinel is never averaged away.
      #
      # Eligibility is tested against `use_cols`, not against the bounds learned
      # so far. Testing against `self.bounds` silently discarded *every*
      # configured bound whenever the strategy computed nothing -- which is
      # exactly the `strategy: none` case that `custom_bounds` exists to serve,
      # because `none` returns an empty mapping. Both `Imputations` blocks in the
      # shipped parameters.yml use that combination, so the whole
      # `fill_missing_dict` in globals.yml was dead and every configured
      # categorical null survived into the encoder as a `__MISSING__` bucket.
      selected = set(self.use_cols)
      self.bounds.update({k: v for k, v in self.custom_bounds.items() if k in selected})
    LOGGER.info('Learned imputation bounds', extra={'routine': self.name, 'bounds': _summarise(self.bounds)})
    return self, df, metadata

  def _learn_bounds(self, df: Any, columns: list[str]) -> dict[str, Any]:
    """Compute the fill bound for each column.

    Args:
      df: The input frame.
      columns: The selected columns.

    Returns:
      A mapping of column to fill value. An explicit no-op strategy yields an
        empty mapping, so ``fillna`` leaves nulls untouched -- but
        ``custom_bounds`` is still applied, because that is the documented way
        to give individual columns a business-meaningful fill while leaving the
        rest of the selection alone.
    """
    strategy = (self.strategy or 'median').lower()
    if strategy == 'none':
      #
      # 0.0 for every column -- so a categorical column had its nulls replaced by
      # the string '0.0' and the routine silently did the opposite of what its
      # strategy name says.
      return {}

    if strategy == 'mode':
      return {column: self._learn_mode(df, column) for column in columns}

    aggregate = {
      'median': lambda c: sf.percentile_approx(sf.col(c), 0.5).alias(c),
      'mean': lambda c: sf.mean(sf.col(c)).alias(c),
      'min': lambda c: sf.min(sf.col(c)).alias(c),
      'max': lambda c: sf.max(sf.col(c)).alias(c),
    }.get(strategy)
    if aggregate is None:
      raise ValueError(
        f'Unknown imputation strategy: {self.strategy!r}. Supported: '
        "'median', 'mean', 'min', 'max', 'mode', or 'none' for no-op."
      )

    row = df.agg(*[aggregate(column) for column in columns]).collect()[0]
    return {column: row[column] for column in columns}

  @staticmethod
  def _learn_mode(df: Any, column: str) -> Any:
    """Return the most frequent value of a column, deterministically.

    a self-joined the count frame against its own maximum,
    which yields one row per tied value and then took ``[0][col]``. For a binary
    or one-hot column, where two values commonly share a count, the result was
    whichever Spark ordered first -- so two training runs of the same model could
    produce different fills. A single ordered ``limit`` is both deterministic and
    avoids the self-join's shuffle.

    Args:
      df: The input frame.
      column: The column to profile.

    Returns:
      The most frequent value, with ties broken by ascending value for
        determinism.
    """
    row = df.groupBy(sf.col(column).alias(column)).count().orderBy(sf.desc('count'), sf.col(column).asc()).limit(1)
    return row.collect()[0][column] if row.count() else None

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Fill nulls using the learned or configured bounds.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    if not self.bounds:
      LOGGER.info('Imputation is a no-op for this registration', extra={'routine': self.name})
      return df, metadata
    LOGGER.info('Applying imputation', extra={'routine': self.name, 'columns': len(self.bounds)})
    return df.fillna(self.bounds), metadata


class KNNImputation(SparkMlRoutine):
  """Impute missing values with a scikit-learn nearest-neighbour model.

  The fitted estimator is persisted externally rather than in the registry,
  because a ``KNNImputer`` holds the whole training matrix and is not
  appropriate to replicate per task.

  Scoped to the numeric roles only. A ``KNNImputer`` measures Euclidean
  distance, so a string column is not merely ignored -- it is a hard error. The
  routine therefore fits and predicts over the same selection, which is
  ``numerical_cols`` narrowed by the configured include/exclude lists.

  Attributes:
    use_cols: The selected numeric columns.
    fitted_columns: The exact columns the model was fitted on. It equals
      ``use_cols``; the UDF input width must match the fit width, so it is
      stored explicitly rather than assumed.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'KNNImputation'
    self.use_cols: list[str] = []
    self.fitted_columns: list[str] = []
    self.handle = 'sklearn.impute.KNNImputer'
    self.params: dict[str, Any] = {'n_neighbors': 5}

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'handle': 'sklearn.impute.KNNImputer',
      'params': {'n_neighbors': 5},
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[KNNImputation, Any, Metadata]:
    """Fit the imputer on the driver and persist it.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.
    """
    self.use_cols = self.get_selected_columns(
      self.cols.get('exclude_cols', []), self.cols.get('include_cols', []), metadata.numerical_cols
    )
    if not self.use_cols:
      return self, df, metadata

    # The fit width is the routine's own selection, not `metadata.feature_cols`.
    # A ``KNNImputer`` computes Euclidean distances, so every column it is given
    # must be numeric; `feature_cols` also contains the categorical roles, and
    # fitting on it raised `ValueError: could not convert string to float` for any
    # schema with a string feature. The routine was therefore unrunnable in every
    # realistic case, masked only by being disabled in the shipped configuration.
    #
    # Narrowing the fit width also removes the reason the two lists were kept
    # separate. The UDF must receive exactly the columns the model was fitted on,
    # and the imputer overwrites all of them, so `use_cols` and `fitted_columns`
    # are now the same list by construction rather than by a convention that
    # nothing enforced.
    self.fitted_columns = list(self.use_cols)
    require_columns(df, self.fitted_columns, self.name)
    frame = df.select(*self.fitted_columns).toPandas()

    estimator_class = resolve(self.handle)
    imputer = estimator_class(**self.params)
    imputer.set_output(transform='pandas')
    fitted = imputer.fit(frame[self.fitted_columns])

    self.save_model_path = f'{self.artefact_path(metadata)}/knn_imputer'
    save_pickle(fitted, self.save_model_path)
    LOGGER.info(
      'Fitted KNN imputer',
      extra={'routine': self.name, 'use_cols': len(self.use_cols), 'path': self.save_model_path},
    )
    return self, df, metadata

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Reload the imputer and apply it inside Spark.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    from functools import partial  # noqa: PLC0415

    from forecasting_ml_framework.platform.spark import get_spark_session  # noqa: PLC0415

    metadata = metadata.copy()
    if not self.use_cols:
      return df, metadata

    model = load_pickle(self.save_model_path)

    def impute_partition(partition: Any, use_cols: list[str], fitted_columns: list[str]) -> Any:
      """Apply the fitted imputer to one Arrow batch.

      Args:
        partition: The batch as a pandas frame.
        use_cols: The columns to overwrite.
        fitted_columns: The columns the model was fitted on.

      Returns:
        The imputed columns.
      """
      imputed = model.transform(partition[fitted_columns])
      return imputed[use_cols]

    udf = sf.pandas_udf(
      partial(impute_partition, use_cols=self.use_cols, fitted_columns=self.fitted_columns),
      st.StructType([st.StructField(column, st.FloatType(), True) for column in self.use_cols]),
    )
    get_spark_session()
    df = df.withColumn('__knn_struct', udf(sf.struct(*[sf.col(c) for c in self.fitted_columns])))
    for column in self.use_cols:
      df = df.withColumn(column, sf.col(f'__knn_struct.{column}'))
    return df.drop('__knn_struct'), metadata


class TreeImputation(PreprocessRoutine):
  """Impute each column from a predictive model fitted on the observed rows.

  For each selected column a *separate* model reconstructs that column from all
  the others, trained only on rows where the column is observed. This is the most
  sophisticated transform in the library and the one that most needs its
  broadcast design to be correct: a closure-captured ``RandomForest`` per column
  is replicated per task, whereas a broadcast variable is distributed once per
  executor.

  Attributes:
    save_imp_model_path: Per-column ``SimpleImputer`` paths.
    save_enc_model_path: Per-column ``OrdinalEncoder`` paths, for categoricals.
    save_model_path: Per-column estimator paths.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'TreeImputation'
    self.use_cols: list[str] = []
    self.fitted_columns: list[str] = []
    self.save_imp_model_path: dict[str, str] = {}
    self.save_enc_model_path: dict[str, str] = {}
    self.save_model_path: dict[str, str] = {}
    self.params: dict[str, Any] = {
      'regressor': 'sklearn.ensemble.RandomForestRegressor',
      'classifier': 'sklearn.ensemble.RandomForestClassifier',
      'initial_guess': 'mean',
    }

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'params': {
        'regressor': 'sklearn.ensemble.RandomForestRegressor',
        'classifier': 'sklearn.ensemble.RandomForestClassifier',
        'initial_guess': 'mean',
      },
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[TreeImputation, Any, Metadata]:
    """Fit the per-column reconstruction models.

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

    self.fitted_columns = list(metadata.feature_cols)
    require_columns(df, self.fitted_columns, self.name)
    frame = df.select(*self.fitted_columns).toPandas()
    observed = frame[frame.isna().sum(axis=1) == 0].index
    LOGGER.info(
      'Bootstrapping tree imputation',
      extra={'observed_rows': len(observed), 'total_rows': len(frame), 'use_cols': len(self.use_cols)},
    )
    if len(observed) < 2:
      raise ValueError(
        f'TreeImputation found only {len(observed)} fully-observed row(s). Per-column '
        'modelling needs enough complete rows to fit a regressor or classifier.'
      )

    #
    # a module-global of that name, so any categorical column raised NameError at
    # fit time.
    self._bootstrap(frame, metadata)

    for column in self.use_cols:
      is_categorical = column in self.save_enc_model_path
      estimator_class = resolve(self.params['classifier'] if is_categorical else self.params['regressor'])
      estimator = estimator_class()
      fitted = estimator.fit(frame.drop(columns=column).loc[observed], frame[column].loc[observed])
      path = f'{self._root(metadata)}/Estimator/{column}'
      save_pickle(fitted, path)
      self.save_model_path[column] = path
    return self, df, metadata

  def _root(self, metadata: Metadata) -> str:
    """Return the artefact root for this routine.

    Args:
      metadata: The schema contract.

    Returns:
      The artefact directory.
    """
    if not metadata.intermediate_file_path:
      raise ValueError('TreeImputation requires metadata.intermediate_file_path to persist its artefacts')
    return f'{metadata.intermediate_file_path.rstrip("/")}/TreeImputation'

  def _bootstrap(self, frame: Any, metadata: Metadata) -> None:
    """Fill and ordinal-encode the frame so every column is numeric.

    Args:
      frame: The driver-side feature frame, mutated in place.
      metadata: The schema contract, whose artefact root is used for the
        persisted encoders.

    Raises:
      ValueError: If the artefact root is not configured.
    """
    from sklearn.impute import SimpleImputer  # noqa: PLC0415
    from sklearn.preprocessing import OrdinalEncoder  # noqa: PLC0415

    root = self._root(metadata)
    for column in frame.columns:
      if pd_is_numeric(frame[column]):
        imputer = SimpleImputer(missing_values=float('nan'), strategy=self.params.get('initial_guess', 'mean'))
        imputer.set_output(transform='pandas')
        frame[column] = imputer.fit(frame[[column]]).transform(frame[[column]])
      else:
        imputer = SimpleImputer(missing_values=None, strategy='most_frequent')
        imputer.set_output(transform='pandas')
        frame[column] = imputer.fit(frame[[column]]).transform(frame[[column]]).astype(str)
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=int(frame[column].nunique()))
        encoder.set_output(transform='pandas')
        frame[column] = encoder.fit(frame[[column]]).transform(frame[[column]]).astype(int)
        path = f'{root}/OrdinalEncoder/{column}'
        save_pickle(encoder, path)
        self.save_enc_model_path[column] = path
      path = f'{root}/SimpleImputer/{column}'
      save_pickle(imputer, path)
      self.save_imp_model_path[column] = path

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Reload the models, broadcast them, and reconstruct in one UDF pass.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    from functools import partial  # noqa: PLC0415

    from forecasting_ml_framework.platform.spark import get_spark_session  # noqa: PLC0415

    metadata = metadata.copy()
    if not self.use_cols:
      return df, metadata

    #
    # scope, so apply raised NameError on every invocation.
    session, _ = get_spark_session()
    imputer_bc = session.sparkContext.broadcast(
      {column: load_pickle(path) for column, path in self.save_imp_model_path.items()}
    )
    encoder_bc = session.sparkContext.broadcast(
      {column: load_pickle(path) for column, path in self.save_enc_model_path.items()}
    )
    tree_bc = session.sparkContext.broadcast(
      {column: load_pickle(path) for column, path in self.save_model_path.items()}
    )

    def impute_partition(
      partition: Any,
      use_cols: list[str],
      imputer_models: dict[str, Any],
      encoder_models: dict[str, Any],
      tree_models: dict[str, Any],
    ) -> Any:
      """Reconstruct the selected columns for one Arrow batch.

      Args:
        partition: The batch as a pandas frame.
        use_cols: The columns to reconstruct.
        imputer_models: Per-column simple imputers.
        encoder_models: Per-column ordinal encoders.
        tree_models: Per-column reconstruction estimators.

      Returns:
        The reconstructed columns, with categoricals decoded back to strings.
      """
      missing = {column: partition.index[partition[column].isna()] for column in use_cols}
      for column, model in imputer_models.items():
        partition[column] = model.transform(partition[[column]])
      for column, model in encoder_models.items():
        partition[column] = model.transform(partition[[column]]).astype(int)
      for column in use_cols:
        rows = missing.get(column)
        if rows is not None and len(rows) > 0 and column in tree_models:
          prediction = tree_models[column].predict(partition.loc[rows].drop(columns=column))
          series = pd_series(prediction, index=rows)
          partition.loc[rows, column] = series
      for column in encoder_models:
        if column in use_cols:
          decoded = encoder_models[column].inverse_transform(partition[[column]].astype(float))
          partition[column] = decoded.reshape(-1)
          partition[column] = partition[column].astype(str)
      return partition[use_cols]

    schema = st.StructType(
      [
        st.StructField(column, st.StringType() if column in self.save_enc_model_path else st.FloatType(), True)
        for column in self.use_cols
      ]
    )
    udf = sf.pandas_udf(
      partial(
        impute_partition,
        use_cols=self.use_cols,
        imputer_models=imputer_bc,
        encoder_models=encoder_bc,
        tree_models=tree_bc,
      ),
      schema,
    )
    df = df.withColumn('__tree_struct', udf(sf.struct(*[sf.col(c) for c in self.fitted_columns])))
    for column in self.use_cols:
      df = df.withColumn(column, sf.col(f'__tree_struct.{column}'))
    return df.drop('__tree_struct'), metadata


class CapOutliers(PreprocessRoutine):
  """Clamp numeric columns to a learned fence.

  The transform *winsorises* rather than filters: values are clamped to the
  bounds and the row count is preserved. That matters because the identifiers
  must stay joinable and the row count is a downstream data-quality signal — but
  it also means the model is trained on synthetic values at the extremes. That is
  why tree models, which are insensitive to monotone transforms, are the reason
  this routine ships disabled.

  Attributes:
    bounds: The learned ``[lower, upper]`` fence per column.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'CapOutliers'
    self.use_cols: list[str] = []
    self.bounds: dict[str, list[float]] = {}

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'method': 'iqr',
      'custom_bounds': {},
      'cols': {'exclude_cols': [], 'include_cols': []},
      'std': {'lower': 3, 'upper': 3},
      'bound': {'lower': 0.01, 'upper': 0.99},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[CapOutliers, Any, Metadata]:
    """Learn the fence for each selected column in a single aggregation pass.

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
    require_columns(df, self.use_cols, self.name)

    method = (self.method or 'iqr').lower()
    if method == 'std':
      self.bounds = self._std_bounds(df, self.use_cols)
    elif method == 'iqr':
      self.bounds = self._iqr_bounds(df, self.use_cols)
    elif method == 'bound':
      quantiles = self.bound or {}
      self.bounds = self._quantile_bounds(
        df, self.use_cols, float(quantiles.get('lower', 0.01)), float(quantiles.get('upper', 0.99))
      )
    else:
      raise ValueError(f'Unknown capping method: {self.method!r}. Supported: std, iqr, bound.')

    if self.custom_bounds:
      self.bounds.update({k: v for k, v in self.custom_bounds.items() if k in self.bounds})
    LOGGER.info('Learned outlier fences', extra={'routine': self.name, 'method': method, 'columns': len(self.bounds)})
    return self, df, metadata

  @staticmethod
  def _iqr_bounds(df: Any, columns: list[str]) -> dict[str, list[float]]:
    """Compute the classical Tukey fence.

    Args:
      df: The input frame.
      columns: The selected columns.

    Returns:
      A mapping of column to ``[lower, upper]``.
    """
    row = df.agg(*[sf.percentile_approx(sf.col(c), [0.25, 0.75]).alias(c) for c in columns]).collect()[0]
    return {
      column: [float(row[column][0]) - 1.5 * (float(row[column][1]) - float(row[column][0])),
               float(row[column][1]) + 1.5 * (float(row[column][1]) - float(row[column][0]))]
      for column in columns
    }

  @staticmethod
  def _quantile_bounds(df: Any, columns: list[str], lower_pct: float, upper_pct: float) -> dict[str, list[float]]:
    """Compute an arbitrary quantile fence.

    Robust, because a percentile fence does not move when the values it is meant
    to constrain move. This is the strategy the configured model program uses.

    Args:
      df: The input frame.
      columns: The selected columns.
      lower_pct: The lower percentile, in ``[0, 1]``.
      upper_pct: The upper percentile, in ``[0, 1]``.

    Returns:
      A mapping of column to ``[lower, upper]``.
    """
    row = df.agg(*[sf.percentile_approx(sf.col(c), [lower_pct, upper_pct]).alias(c) for c in columns]).collect()[0]
    return {column: [float(row[column][0]), float(row[column][1])] for column in columns}

  def _std_bounds(self, df: Any, columns: list[str]) -> dict[str, list[float]]:
    """Compute a mean-plus/minus-standard-deviation fence.

    Not robust: the outliers the fence is meant to cap are themselves part of the
    mean and standard deviation, so a heavy tail drags the fence outward. It is
    offered because it is the conventional fence and because some domains
    legitimately want it.

    Args:
      df: The input frame.
      columns: The selected columns.

    Returns:
      A mapping of column to ``[lower, upper]``.
    """
    projections: list[Any] = []
    for column in columns:
      projections.append(sf.mean(sf.col(column)).alias(f'{column}__mean'))
      projections.append(sf.stddev(sf.col(column)).alias(f'{column}__std'))
    row = df.select(*projections).collect()[0]
    lower_sigma = float((self.std or {}).get('lower', 3))
    upper_sigma = float((self.std or {}).get('upper', 3))
    return {
      column: [
        float(row[f'{column}__mean']) - lower_sigma * float(row[f'{column}__std']),
        float(row[f'{column}__mean']) + upper_sigma * float(row[f'{column}__std']),
      ]
      for column in columns
    }

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Clamp each selected column to its fence.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    for column, fence in self.bounds.items():
      if not fence:
        continue
      lower, upper = fence[0], fence[1]
      if lower is None or upper is None:
        # A fence of [None, None] would replace the whole column with nulls.
        LOGGER.warning('Skipping capping for a column with an incomplete fence', extra={'column': column})
        continue
      if lower == upper:
        # A degenerate fence would replace the entire column with one constant.
        LOGGER.warning('Skipping capping for a degenerate fence', extra={'column': column, 'bound': lower})
        continue
      df = df.withColumn(
        column,
        sf.when(sf.col(column) > upper, sf.lit(upper))
        .when(sf.col(column) < lower, sf.lit(lower))
        .otherwise(sf.col(column)),
      )
    return df, metadata


class Normalizations(SparkMlRoutine):
  """Scale numeric columns with a Spark ML vector assembler and scaler.

  The scaler operates on a ``Vector``, so the routine assembles, scales, and
  then explodes the vector back into named columns. The round trip is the only
  way to restore a wide, named schema from a vectorised transform.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'Normalizations'
    self.use_cols: list[str] = []
    self.handle = 'pyspark.ml.feature.MinMaxScaler'
    self.params: dict[str, Any] = {}
    self.round_off = 6

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'handle': 'pyspark.ml.feature.MinMaxScaler',
      'params': {},
      'round_off': 6,
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[Normalizations, Any, Metadata]:
    """Fit the scaling pipeline and persist it.

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
    require_columns(df, self.use_cols, self.name)
    self.save_model_path = self.artefact_path(metadata)

    assembler = VectorAssembler(
      inputCols=self.use_cols,
      outputCol='__norm_features',
      handleInvalid='keep',
    )
    scaler = resolve(self.handle)(inputCol='__norm_features', outputCol='__norm_scaled', **self.params)
    self._fit_pipeline(df, [assembler, scaler])
    return self, df, metadata

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Reload the scaler and write the scaled values back in place.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    if not self.use_cols:
      return df, metadata
    other_columns = [column for column in df.columns if column not in self.use_cols]
    scaled = self._load_pipeline().transform(df)

    def to_array(vector: Any) -> list[float]:
      """Explode a scaled vector back into an array of doubles.

      Args:
        vector: The scaled vector.

      Returns:
        The vector's elements.
      """
      return None if vector is None else vector.toArray().tolist()

    scaled = scaled.withColumn('__norm_scaled', sf.udf(to_array, st.ArrayType(st.DoubleType()))('__norm_scaled'))
    projections = [
      sf.round(sf.col('__norm_scaled')[index], self.round_off).alias(self.use_cols[index])
      for index in range(len(self.use_cols))
    ]
    return scaled.select(*other_columns, *projections), metadata


class QuantileDiscretizer(SparkMlRoutine):
  """Discretise numeric columns into quantile buckets.

  Registering the output is the whole difficulty, and it is silent when it goes
  wrong. The transform writes to ``col + '_qd'``, so unless that new name is also
  registered the trainer's feature resolution -- which gates on both the feature
  list and the numeric list -- excludes the column that was just computed. The
  frame changes and the model does not, and nothing reports a discrepancy.

  So the discretised column is registered in both namespaces in one call, and it
  replaces the source in the feature list: a bucket index is an order, and
  keeping the continuous column alongside it would present the estimator with two
  representations of the same ordering.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'QuantileDiscretizer'
    self.use_cols: list[str] = []
    self.handle = 'pyspark.ml.feature.QuantileDiscretizer'
    self.params: dict[str, Any] = {'numBuckets': 10}

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'handle': 'pyspark.ml.feature.QuantileDiscretizer',
      'params': {'numBuckets': 10},
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def artefact_name(self) -> str:
    """Return an artefact name that varies with the bucket count.

    Returns:
      A directory name such as ``QuantileDiscretizer10``.
    """
    return f'{type(self).__name__}{self.params.get("numBuckets", 10)}'

  def fit(self, df: Any, metadata: Metadata) -> tuple[QuantileDiscretizer, Any, Metadata]:
    """Fit one discretiser per selected column.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.

    Raises:
      ValueError: If a selected column is constant. The discretiser itself would
        fail with a far less clear message, so the guard trades an expensive
        per-column distinct count for a legible failure.
    """
    self.use_cols = self.get_selected_columns(
      self.cols.get('exclude_cols', []), self.cols.get('include_cols', []), metadata.numerical_cols
    )
    if not self.use_cols:
      return self, df, metadata
    require_columns(df, self.use_cols, self.name)
    self.save_model_path = self.artefact_path(metadata)

    stage_class = resolve(self.handle)
    stages: list[Any] = []
    for column in self.use_cols:
      if df.select(column).distinct().limit(2).count() < 2:
        raise ValueError(
          f'QuantileDiscretizer expects at least 2 distinct values in {column!r}, which is constant. '
          'Remove it from cols.include_cols or exclude it from the role list.'
        )
      stages.append(stage_class(inputCol=column, outputCol=f'{column}_qd', **self.params))
    self._fit_pipeline(df, stages)
    return self, df, metadata

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Reload the discretisers and register their output as features.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    metadata = metadata.copy()
    if not self.use_cols:
      return df, metadata
    other_columns = [column for column in df.columns if column not in self.use_cols]
    discretised = self._load_pipeline().transform(df)
    projections = [sf.col(f'{column}_qd').alias(f'{column}_qd') for column in self.use_cols]

    for column in self.use_cols:
      metadata.replace_feature(column, f'{column}_qd')
    return discretised.select(*other_columns, *projections), metadata


class DomainNormalizations(PreprocessRoutine):
  """Divide numeric columns by a reference column, so the feature is a rate.

  The telecom motivation is immediate: usage normalised by tenure, spend
  normalised by plan count. The result measures a *rate* rather than a *level*,
  removing a strong proxy for customer age.

  The transform is not idempotent — dividing twice by the reference compounds —
  and it emits no null-handling branch, so a null in either operand yields null.
  Both are acceptable for a domain-specific transform and are documented here so
  they are not discovered by surprise.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    self.load_params(global_params, params)
    self.name = 'DomainNormalizations'
    self.use_cols: list[str] = []
    self.normalize_by_col: str = ''
    self.round_off = 4

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'normalize_by_col': '',
      'round_off': 4,
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[DomainNormalizations, Any, Metadata]:
    """Resolve the target columns. No state is learned.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.
    """
    self.use_cols = self.get_selected_columns(
      self.cols.get('exclude_cols', []), self.cols.get('include_cols', []), metadata.feature_cols
    )
    return self, df, metadata

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Divide each selected column by the reference column.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.

    Raises:
      ValueError: If no reference column is configured.
    """
    if not self.normalize_by_col:
      raise ValueError(
        f'{self.name} requires normalize_by_col to be set; it is the reference column each '
        'selected column is divided by.'
      )
    require_columns(df, [*self.use_cols, self.normalize_by_col], self.name)
    for column in self.use_cols:
      df = df.withColumn(column, sf.round(sf.col(column) / sf.col(self.normalize_by_col), self.round_off))
    return df, metadata


def _summarise(bounds: dict[str, Any]) -> dict[str, Any]:
  """Render a bounds dictionary compactly for the run log.

  Args:
    bounds: The learned bounds.

  Returns:
    A mapping of column to its bound, unchanged.
  """
  return bounds


def pd_is_numeric(series: Any) -> bool:
  """Report whether a pandas series holds a numeric dtype.

  Args:
    series: The series to inspect.

  Returns:
    ``True`` for a numeric dtype.
  """
  return str(series.dtype).startswith(('int', 'uint', 'float'))


def pd_series(values: Any, index: Any) -> Any:
  """Construct a pandas series with an explicit index.

  Args:
    values: The values.
    index: The index.

  Returns:
    A pandas ``Series``.
  """
  import pandas as pd  # noqa: PLC0415

  return pd.Series(values, index=index)
