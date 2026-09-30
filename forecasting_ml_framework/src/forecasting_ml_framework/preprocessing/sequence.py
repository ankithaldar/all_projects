#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequence and behavioural-array transforms.

The family splits into two groups, and the split determines correctness.

**Summarising** routines write a *new*, suffixed column and register it in both
``feature_cols`` and ``numerical_cols``. The source array therefore survives, so
several summaries of the same column coexist — mean usage, peak usage and trend
can all be derived from one array — and every later numeric routine sees the
summary, so ``SequenceAverage`` followed by ``CapOutliers`` caps the summary as
intended.

**In-place** routines overwrite the source array, which is also correct because
the array type is unchanged.

Three properties every member of the family holds to, and which together are what
make the family composable:

* **A summariser always registers its output** in ``feature_cols`` *and*
  ``numerical_cols``, in one call, so the new column is visible both to the
  trainer and to every later numeric-only routine. A summariser that computes a
  value without registering it produces a column the model silently ignores --
  modelling-inert, and invisible in every aggregate metric.
* **Nothing destructive.** No member overwrites its source array. A summariser
  that replaced the sequence it read would destroy the sequence, and because both
  the input and the output are array-typed, Spark would accept the schema change
  without complaint.
* **Windows are normalised to equal length.** Any routine pairing two slices of
  the same array normalises both first. Below that, the arithmetic raises -- and
  it raises hardest on low-activity customers, who are exactly the population a
  retention model most needs to score.
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import functions as sf
from pyspark.sql import types as st

from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.preprocessing.bases import UdfRoutine, require_columns
from forecasting_ml_framework.preprocessing.sequence_ops import (
  change_ratio,
  coefficient_of_variation,
  cos_sim,
  domain_normalization,
  get_fixed_length,
  get_slope_func,
  normalise_window,
  seq_average,
  seq_date_delta,
  seq_delta,
  seq_last,
  seq_max,
  seq_min,
  seq_normalize,
  seq_recency,
  seq_sum,
)

LOGGER = get_logger(__name__)


class SequenceTransform(UdfRoutine):
  """Base for the sequence family: array in, scalar summary out.

  Attributes:
    use_cols: The selected array columns. Sequence routines scope themselves to
      ``feature_cols`` rather than to ``seq_cols``, which works because sequence
      columns are also declared as features — but they must not be *only*
      declared as features, or the trainer would receive a raw array. Naming the
      array columns in ``cols.include_cols`` is therefore the expected pattern.
    last_n_val: The recency window. Almost every routine summarises only the
      most recent ``k`` elements: recent behaviour is more predictive than
      lifetime behaviour, and truncating to ``k`` makes the summary robust to
      array-length variation across customers.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__()
    # `use_cols` is resolved state, not configuration, so it is initialised
    # before the load. Every other attribute this family declares arrives from
    # `get_default_dict` via the base constructor and is then overlaid by
    # `load_params`. Re-assigning a configured key *after* the load discards it,
    # which is what this constructor used to do to `last_n_val` and `cols` --
    # silently reducing all sixteen routines to their defaults and discarding the
    # `include_cols` that names the array columns.
    self.use_cols: list[str] = []
    self.load_params(global_params, params)

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'last_n_val': 3,
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[SequenceTransform, Any, Metadata]:
    """Resolve the target array columns. No state is learned.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.
    """
    self.use_cols = self.get_selected_columns(
      self.cols.get('exclude_cols', []), self.cols.get('include_cols', []), metadata.feature_cols
    )
    if self.use_cols:
      require_columns(df, self.use_cols, self.name)
    LOGGER.debug('Resolved sequence columns', extra={'routine': self.name, 'columns': self.use_cols})
    return self, df, metadata

  def helper_kwargs(self) -> dict[str, Any]:
    """Return the recency window bound into the helper.

    Returns:
      A mapping of helper keyword argument to value.
    """
    return {'last_n_val': self.last_n_val}


class SequenceInPlace(SequenceTransform):
  """Base for the four routines that rewrite an array in place.

  No metadata action is needed: the shape is preserved, so the column remains an
  ``array<float>`` in both the frame and the sequence namespace.
  """

  output_suffix = None
  return_type: Any = None


class FixedLength(SequenceInPlace):
  """Pad or truncate an array to a fixed length.

  It pads *and* slices, because a rectangular array is the entire purpose: the
  downstream tensor is dense and the model reshapes to a fixed width, so a batch
  of mixed widths cannot be built. Padding alone would leave the longest-history
  customers -- the highest-value ones -- producing the ragged batch that fails
  the job. Left-padding preserves recency ordering with the newest event last, and
  truncation therefore takes from the front.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'FixedLength'
    # `fixed_length` is a configuration key declared in `get_default_dict`; it
    # must not be re-assigned here or the configured window is discarded.
    self.return_type = st.ArrayType(st.FloatType())
    self.helper = get_fixed_length

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {'skip': True, 'fixed_length': 6, 'cols': {'exclude_cols': [], 'include_cols': []}}

  def helper_kwargs(self) -> dict[str, Any]:
    """Return the target length bound into the helper.

    Returns:
      A mapping of helper keyword argument to value.
    """
    return {'fixed_length': self.fixed_length}


class SequenceFillNa(SequenceInPlace):
  """Replace a null array with a fixed-length zero array.

  The most efficient routine in the library and the best template for writing
  new ones: if a sequence transform can be expressed with Spark's array
  functions, it should be, because that executes entirely in the JVM and avoids
  the per-partition Arrow transfer entirely.

  It pairs with :class:`FixedLength`: this one manufactures a fixed-length array
  for customers with no history, and that one normalises everyone else to the
  same length. Together they give a dense, rectangular sequence matrix.
  """

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceFillNa'

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {'skip': True, 'fixed_length': 6, 'cols': {'exclude_cols': [], 'include_cols': []}}

  def helper(self) -> Any:  # type: ignore[override]
    """No helper is used; the transform is pure Spark.

    Raises:
      NotImplementedError: Always. The base class's UDF path is bypassed.
    """
    raise NotImplementedError('SequenceFillNa is expressed with native Spark array functions')

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Replace null arrays with zeros, entirely in the JVM.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    null_value = sf.array(*([sf.lit(0.0)] * int(self.fixed_length)))
    for column in self.use_cols:
      df = df.withColumn(column, sf.when(sf.col(column).isNull(), null_value).otherwise(sf.col(column)))
    return df, metadata


class SequenceNormalize(SequenceInPlace):
  """Express each element as a share of the array total."""

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceNormalize'
    self.return_type = st.ArrayType(st.FloatType())
    self.helper = seq_normalize

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {'skip': True, 'round_off': 4, 'cols': {'exclude_cols': [], 'include_cols': []}}

  def helper_kwargs(self) -> dict[str, Any]:
    """Return the rounding bound into the helper.

    Returns:
      A mapping of helper keyword argument to value.
    """
    return {'round_off': self.round_off}


class SequenceDomainNormalizations(SequenceInPlace):
  """Divide each element of an array by a reference value."""

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceDomainNormalizations'
    self.return_type = st.ArrayType(st.FloatType())
    self.helper = domain_normalization

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'normalize_by_val': 1.0,
      'round_off': 4,
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def helper_kwargs(self) -> dict[str, Any]:
    """Return the reference value bound into the helper.

    Returns:
      A mapping of helper keyword argument to value.
    """
    return {'normalize_by_val': self.normalize_by_val, 'round_off': self.round_off}


class SequenceAverage(SequenceTransform):
  """Summarise an array as the mean of its most recent values."""

  output_suffix = '_avg'
  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceAverage'
    self.helper = seq_average


class SequenceSum(SequenceTransform):
  """Summarise an array as the sum of its most recent values."""

  output_suffix = '_sum'
  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceSum'
    self.helper = seq_sum


class SequenceMin(SequenceTransform):
  """Summarise an array as the minimum of its most recent values."""

  output_suffix = '_min'
  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceMin'
    self.helper = seq_min


class SequenceMax(SequenceTransform):
  """Summarise an array as the maximum of its most recent values."""

  output_suffix = '_max'
  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceMax'
    self.helper = seq_max


class SequenceLast(SequenceTransform):
  """Summarise an array as its final element."""

  output_suffix = '_last'
  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceLast'
    self.helper = seq_last

  def helper_kwargs(self) -> dict[str, Any]:
    """This helper takes no window.

    Returns:
      An empty mapping.
    """
    return {}


class SequenceDelta(SequenceTransform):
  """Summarise an array as its total change across the recency window.

  a helper returned a *list* of consecutive differences,
  a ragged array of length ``k-1`` that no estimator consumes, and the routine
  registered nothing, so the transform is modelling-inert. The helper is a
  scalar and the routine registers it like its siblings.
  """

  output_suffix = '_delta'
  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceDelta'
    self.helper = seq_delta


class SequenceDateDelta(SequenceTransform):
  """Summarise a date array as its mean inter-event gap, in days.

  a overwrote the source column with a list of day-gaps
  and registered nothing, destroying the event dates and producing a feature no
  estimator accepts. It writes a suffixed scalar
  column and registers it.
  """

  output_suffix = '_date_delta'
  return_type = st.IntegerType()
  output_is_numerical = True

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceDateDelta'
    self.helper = seq_date_delta


class SequenceRecency(SequenceTransform):
  """Summarise a date array as its age in days at the run date.

  The *absence* of a recent event is itself a signal, so an empty sequence must
  be represented by a value rather than a null: a null would propagate into
  imputation and destroy that signal. ``fill_na_val`` exists for exactly that
  case, and the helper now actually applies it.

  This is also the one routine in the family that is a function of two columns:
  the array *and* the reference date. The base class passes only the array, so
  ``extra_udf_columns`` supplies the reference. Without it the helper's
  ``run_dt_val`` stayed ``None``, the reference test failed, and the routine
  emitted the sentinel for **every** row -- a registered numerical feature that
  was constant, and therefore invisible in every aggregate metric.
  """

  output_suffix = '_recency'
  return_type = st.IntegerType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'SequenceRecency'
    self.helper = seq_recency

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'run_date_col': 'run_date',
      'fill_na_val': 9_999_999,
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def fit(self, df: Any, metadata: Metadata) -> tuple[SequenceTransform, Any, Metadata]:
    """Resolve the array columns and assert the reference date is present.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.

    Raises:
      DataContractError: If the configured reference-date column is absent. The
        reference is load-bearing, and a missing column would otherwise surface
        as a frame of sentinels rather than as an error.
    """
    # `self` is rebound by the base class's contract, so the local is named
    # `routine` rather than shadowing the instance -- an assignment to `self`
    # inside a method silently discards the object the method belongs to.
    routine, frame, contract = super().fit(df, metadata)
    if not routine.use_cols:
      return routine, frame, contract
    require_columns(frame, [routine.run_date_col], f'{routine.name}(run_date_col)')
    return routine, frame, contract

  def extra_udf_columns(self, column: str) -> list[Any]:
    """Return the reference-date column as the helper's second argument.

    Args:
      column: The array column being summarised. Unused.

    Returns:
      A single-element list holding the reference-date column expression.
    """
    del column
    return [sf.col(self.run_date_col)]

  def helper_kwargs(self) -> dict[str, Any]:
    """Return the sentinel bound into the helper. No window is used.

    Returns:
      A mapping of helper keyword argument to value.
    """
    return {'fill_na_val': self.fill_na_val}


class ChangeRatio(SequenceTransform):
  """Summarise an array as the most recent value relative to its baseline."""

  output_suffix = '_cr'
  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'ChangeRatio'
    self.helper = change_ratio


class CoefficientOfVariation(SequenceTransform):
  """Summarise an array as sigma over mu over the recency window."""

  output_suffix = '_cv'
  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'CoefficientOfVariation'
    self.helper = coefficient_of_variation


class Slope(SequenceTransform):
  """Summarise an array as the least-squares slope of its recent values.

  The configurable ``suffix`` is the cleanest use of a configuration key in the
  library: it lets an ``order: 1`` and an ``order: 2`` slope of the same column
  coexist, both be registered, and both reach the model.
  """

  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'Slope'
    self.helper = get_slope_func

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'order': 1,
      'suffix': 'slope',
      'last_n_val': 2,
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  @property
  def output_suffix(self) -> str:  # type: ignore[override]
    """Return the configured output suffix.

    Returns:
      The suffix appended to the source column name.
    """
    return f'_{self.suffix}'

  def derived_name(self, column: str) -> str:
    """Return the derived column name.

    Args:
      column: The source column name.

    Returns:
      ``{column}_{suffix}``.
    """
    return f'{column}_{self.suffix}'

  def helper_kwargs(self) -> dict[str, Any]:
    """Return the polynomial order and window bound into the helper.

    Returns:
      A mapping of helper keyword argument to value.
    """
    return {'order': int(self.order), 'last_n_val': self.last_n_val}


class Similarity(SequenceTransform):
  """Score how closely a behavioural profile matches a reference archetype.

  A *reference-pattern* feature: "how much does this customer's usage shape
  resemble the archetype of a high-value user, or of an at-risk one?" The
  archetype vectors are supplied in configuration, so domain knowledge can be
  injected without a code change. This is the most domain-specific routine in the
  library and a good example of what the configuration-driven design buys.

  a paired two *disjoint* windows of the same array, which
  have equal length only when the array holds at least ``2k`` elements. Below
  that, ``np.dot`` on vectors of different lengths raises inside the UDF, failing
  the whole job. Since the default reference map is empty, *every* column took
  that path, so the affected population was every customer with fewer than
  ``2k`` events — precisely the low-activity customers a retention model most
  needs to score. Both windows are normalised to the same length.
  """

  output_suffix = '_similarity'
  return_type = st.FloatType()

  def __init__(self, global_params: dict[str, Any] | None, params: dict[str, Any] | None) -> None:
    """Build the routine from configuration.

    Args:
      global_params: The shared parameter bag.
      params: The routine's own configuration block.
    """
    super().__init__(global_params, params)
    self.name = 'Similarity'
    self.helper = cos_sim

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the parameter surface.

    Returns:
      The default configuration.
    """
    return {
      'skip': True,
      'rational_vector_map': {},
      'last_n_val': 2,
      'cols': {'exclude_cols': [], 'include_cols': []},
    }

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Compute the cosine similarity against each column's archetype.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    from functools import partial  # noqa: PLC0415

    metadata = metadata.copy()
    window = int(self.last_n_val)

    slice_udf = sf.udf(partial(get_fixed_length, fixed_length=window), st.ArrayType(st.FloatType()))
    baseline_udf = sf.udf(
      partial(normalise_window, start=-2 * window, end=-window, fixed_length=window),
      st.ArrayType(st.FloatType()),
    )
    similarity_udf = sf.udf(cos_sim, st.FloatType())

    intermediates: list[str] = []
    for column in self.use_cols:
      derived = self.derived_name(column)
      archetype = self.rational_vector_map.get(column)
      if archetype is None:
        # No archetype configured: compare the two most recent equal-length
        # windows, which measures short-term stability.
        #
        # BOTH sides go through `get_fixed_length`, which pads and truncates to
        # exactly `window`. Only the recent window did so before: the baseline was
        # a raw Python slice, so it was shorter than `window` for any history
        # shorter than `2 * window`. `cos_sim` then rejected the mismatched pair
        # and returned 0.0 -- so the feature was 0 for every customer with fewer
        # than `2k` events, and 0 means "maximally dissimilar", the opposite
        # reading. Since the reference map is empty by default, that was the
        # branch every column took, and the affected population was exactly the
        # low-activity customers a retention model most needs to score.
        recent = f'__sim_recent_{column}'
        baseline = f'__sim_base_{column}'
        df = df.withColumn(recent, slice_udf(sf.col(column)))
        df = df.withColumn(baseline, baseline_udf(sf.col(column)))
        intermediates.extend([recent, baseline])
        left, right = recent, baseline
      else:
        reference = f'__sim_ref_{column}'
        # The archetype is normalised with the same helper as the recent window,
        # so a configured vector shorter than `window` is padded rather than
        # compared against a longer vector.
        df = df.withColumn(
          reference,
          sf.udf(partial(get_fixed_length, fixed_length=window), st.ArrayType(st.FloatType()))(
            sf.array(*[sf.lit(float(value)) for value in archetype])
          ),
        )
        intermediates.append(reference)
        left, right = f'__sim_recent_{column}', reference
        df = df.withColumn(left, slice_udf(sf.col(column)))
        intermediates.append(left)

      df = df.withColumn(derived, similarity_udf(sf.col(left), sf.col(right)))
      metadata.register_derived_feature(derived)

    if intermediates:
      df = df.drop(*intermediates)
    return df, metadata
