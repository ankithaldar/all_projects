#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exception taxonomy.

A small, explicit hierarchy, so that a failure is classifiable both
programmatically and in the run log. That distinction is the point: a bare
traceback from a deeply nested frame tells an operator that something broke
somewhere, and a decision the framework makes on their behalf -- accept or reject
a model -- has to be reportable as an outcome rather than as a stack trace.

Two properties apply to every type here:

* **Structured context.** ``FrameworkError`` takes arbitrary keyword detail and
  renders it into the message, so a failure names the run, the model and the
  stage without anyone having to read the traceback.
* **Semantic types, not categories.** A caller that must distinguish "the
  configuration is wrong" from "the warehouse rejected the query" catches two
  different types. A single catch-all would force that caller to parse messages.
"""

from __future__ import annotations


class FrameworkError(Exception):
  """Base class for every error raised by the forecasting ML framework.

  Attributes:
    context: A mapping of diagnostic detail attached to the failure. It is
      rendered into the log message so an operator can identify the run, the
      model and the stage without reading the traceback.
  """

  def __init__(self, message: str, **context: object) -> None:
    """Build the error.

    Args:
      message: A human-readable description of the failure.
      **context: Additional diagnostic key/value pairs.
    """
    self.message = message
    self.context = dict(context)
    super().__init__(self._render())

  def _render(self) -> str:
    """Render the message together with its diagnostic context.

    Returns:
      The formatted error string.
    """
    if not self.context:
      return self.message
    detail = ', '.join(f'{key}={value!r}' for key, value in sorted(self.context.items()))
    return f'{self.message} [{detail}]'


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(FrameworkError):
  """Raised when configuration is missing, malformed or contradictory."""


class MissingParameterError(ConfigurationError):
  """Raised when a mandatory runtime parameter is absent.

  A mandatory parameter is looked up through this rather than by a bare
  ``params['key']`` subscript. A ``KeyError`` names the key but carries no
  configuration context, so the operator learns *what* is missing and nothing
  about where it should have come from.
  """


class ConfigurationRewriteError(ConfigurationError):
  """Raised when the bootstrap configuration rewriter fails.

  The rewriter must not swallow its own failures behind a bare
  ``except Exception: print(...)``, so a rewrite failure surfaced minutes later
  as an unresolvable interpolation error. It fails fast and loudly.
  """


# --------------------------------------------------------------------------- #
# Data contract
# --------------------------------------------------------------------------- #
class DataContractError(FrameworkError):
  """Raised when a dataframe violates the declared schema contract."""


class TargetCardinalityError(DataContractError):
  """Raised when the target column has fewer than two distinct values.

  Every downstream metric (AUC, lift, calibration, threshold selection) is
  undefined for a single-class frame, so this is a legitimate hard failure.
  """


class ColumnRoleConflictError(DataContractError):
  """Raised when a column is declared in two mutually exclusive role lists.

  array (sequence) columns must not also be declared numerical,
  categorical or indicator columns, otherwise the estimator receives an
  ``array<float>`` feature it cannot fit. A code had no validation
  for this invariant.
  """


class RowCountAssertionError(DataContractError):
  """Raised when an intermediate frame's row count drifts beyond tolerance.

  the framework observed every count and asserted none. A routine
  that silently dropped 40 % of a frame produced a complete, plausible and
  entirely wrong model with no alert.
  """


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class RoutineError(FrameworkError):
  """Base class for feature-transform failures."""


class RoutineNotFoundError(RoutineError):
  """Raised when a configured routine name cannot be resolved.

  a resolver hard-coded a single module prefix, so a routine
  living in any other module was unreachable and a typo produced a
  ``ModuleNotFoundError`` at pipeline-execution time.
  """


class InvalidSearchSpaceError(ConfigurationError):
  """Raised when a search-space expression cannot be resolved safely.

  The framework used to evaluate the expression by handing a parsed tree to
  ``eval()``, behind guards that checked the node types. The replacement is a
  closed registry: the callee must be one of the declared distribution
  constructors and the arguments must be literals, so there is no interpreter in
  the path at all.
  """


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
class ModelError(FrameworkError):
  """Base class for modelling-tier failures."""


class HoldoutRequiredError(ModelError):
  """Raised when an operation needs a validation split and none was created.

  Replaces an uncaught ``ValueError`` raised from deep inside ``train``.
  """


class MetricConfigurationError(ModelError):
  """Raised when the configured promotion metric is not supported.

  An unrecognised name is an error, never a silent substitution. Falling back to
  ``LIFT_DECILE_10`` would let a typo in ``eval_metric`` decide a production
  promotion on a metric the operator did not choose, with the only evidence a
  line in a cluster log. Silently substituting the metric that drives a promotion
  is a governance failure rather than a usability convenience, so unless
  ``strict=False`` is passed the promotion fails instead.
  """


class MetricBandEmptyError(ModelError):
  """Raised when a requested score band contains no rows.

  An empty band is reported, never substituted with a neighbouring one. Reporting
  decile 10's lift under a request for ``LIFT_DECILE_7`` would present a
  materially different quantity as a successful measurement, and the difference
  between changing whether a task runs and changing a reported number must never
  be silent.
  """


class PromotionError(FrameworkError):
  """Base class for champion/challenger promotion failures."""


class RegistryEntryMissingError(PromotionError):
  """Raised when no current registry row exists for the model key.

  The framework cannot bootstrap its own registry; onboarding a new ``model_key``
  requires an out-of-band insert. That dependency is stated explicitly in the
  message rather than implicit in a bare ``ValueError``.
  """


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - kedro is a hard dependency; the guard aids tooling
  from kedro.io.core import DatasetError as _KedroDatasetError
except Exception:  # noqa: BLE001 - allow import without kedro installed
  _KedroDatasetError = Exception  # type: ignore[assignment,misc]


class DatasetError(FrameworkError, _KedroDatasetError):  # type: ignore[misc,valid-type]
  """Raised when a dataset is misconfigured or an I/O operation fails.

  It subclasses both the framework's error base and Kedro's own ``DatasetError``,
  so a consumer catching either sees it -- which matters because the catalog
  itself catches Kedro's type when reporting a dataset failure.
  """


class InvalidWriteModeError(DatasetError):
  """Raised when a dataset is configured with an unsupported write mode."""


class PartitionValueMissingError(DatasetError):
  """Raised when a partitioned dataset is read or written without a value."""


# --------------------------------------------------------------------------- #
# Platform
# --------------------------------------------------------------------------- #
class SparkSessionError(FrameworkError):
  """Raised when a Spark session cannot be constructed."""


class SparkTuningConfigNotFoundError(SparkSessionError):
  """Raised when no Spark tuning document exists for the environment."""


class PipelineTopologyError(FrameworkError):
  """Raised when the pipeline registry cannot build a valid graph."""
