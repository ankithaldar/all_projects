#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Trainer classes: the estimator lifecycle.

Three trainers behind one informal interface, dispatched by a single
configuration string. Dispatch lives in the node layer; each trainer owns its own
lifecycle.

:class:`ModelTraining`
  The primary abstraction, and the one that matters. A pandas/scikit-learn
  trainer whose instance **is** the model artefact: after ``train`` it carries
  the ordered feature vector, the resolved target, the label set, both
  operating-point thresholds and both rate statistics. Pickling that instance
  yields a self-describing model — loading it is sufficient to score, with no
  external feature list, no separate scaler and no separate threshold.

:class:`SparkModelTraining`
  A Spark ML trainer. Its ``train`` returns the same five-tuple as the other two
  tiers, which is what lets the node layer dispatch on tier without a special
  case: a trainer that returned a different arity would be wired and dispatched
  to, and yet never successfully return.

:class:`TransformerTraining`
  The sequence tier, over parquet directories on object storage. It requires the
  optional deep-learning extra, so it is opt-in through configuration and the
  framework runs without it until the tier is enabled.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd

from forecasting_ml_framework import constants
from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.exceptions import HoldoutRequiredError
from forecasting_ml_framework.modeling.calibration import calibrate_probability, select_threshold
from forecasting_ml_framework.modeling.scoring import (
  positive_probability,
  project_score_frame,
  score_frame,
  score_regression_frame,
)
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.utils.reflection import coerce_search_space, resolve
from forecasting_ml_framework.utils.storage import save_pickle

LOGGER = get_logger(__name__)

#: The five-tuple every trainer's ``train`` returns. Declared once so the node
#: layer and the trainers cannot drift apart.
TRAIN_RETURN = 'self, fitted_model, calibrated_model, train_frame, holdout_frame'

#: Estimator classes that take a per-class weight vector rather than a scalar
#: positive-class weight. Membership is decided by walking the class's MRO rather
#: than by comparing ``__name__`` to a literal: name-based detection is a string
#: comparison wearing a type check's clothes, and a subclass or a wrapper falls
#: through to the scalar branch, where it is handed a parameter its constructor
#: rejects. The MRO check follows the hierarchy that actually determines the
#: estimator's behaviour.
_CLASS_WEIGHT_ESTIMATORS = ('CatBoostClassifier',)

#: The two spellings of the search-space constructor argument. A grid searcher
#: takes ``param_grid``; a randomised or successive-halving one takes
#: ``param_distributions``. Both are accepted in configuration under either
#: spelling, and the one the resolved searcher actually declares is the one used.
_SEARCH_SPACE_ARGUMENTS = ('param_grid', 'param_distributions')

#: Searchers that require scikit-learn's experimental opt-in before they can be
#: imported. Membership is by bare class name, which is what ``__name__`` holds.
HALVING_SEARCHERS = ('HalvingGridSearchCV', 'HalvingRandomSearchCV')

#: Above this many distinct values the target is treated as continuous rather than
#: categorical. The classification tier's threshold, stratification and hard-prediction
#: steps are all defined over a small label set, and each of them is either
#: meaningless or fatal against a continuous one -- so the limit is what keeps the
#: two tiers from borrowing each other's assumptions.
CONTINUOUS_TARGET_DISTINCT_LIMIT = 10


def _is_continuous_target(handle: Any) -> bool:
  """Report whether a resolved model handle denotes a regressor.

  The question is answered by scikit-learn's own predicate rather than by a
  name convention, because "does this class predict a continuous quantity" is a
  property of the estimator and not of how it happens to be spelled. A configured
  ``XGBRegressor`` and a ``RandomForestRegressor`` are both regressors and neither
  is a classifier; ``XGBClassifier`` is neither continuous nor absent.

  The name check is the last resort for a handle that scikit-learn cannot
  classify -- a third-party estimator that does not inherit from its base classes.
  Returning ``False`` there means the classification path is taken, which is the
  historical behaviour and therefore the safe default.

  Args:
    handle: The resolved estimator class, or an instance of one.

  Returns:
    ``True`` when the handle denotes a regressor.
  """
  if handle is None:
    return False
  try:
    from sklearn.base import is_classifier, is_regressor  # noqa: PLC0415

    if is_classifier(handle):
      return False
    if is_regressor(handle):
      return True
  except (ImportError, TypeError, ValueError):  # pragma: no cover - defensive
    pass
  name = getattr(handle, '__name__', type(handle).__name__).lower()
  return 'regress' in name


def _suggest(trial: Any, name: str, value: Any) -> Any:
  """Resolve one search-space value for a trial.

  The three forms are distinguished explicitly rather than by a nested conditional
  expression: a distribution object is *sampled* (which is what makes it a search
  over a continuous range), a sequence of candidates is *offered* to the trial so it
  can prune branches, and anything else is a fixed value passed straight through.

  Args:
    trial: The Optuna trial.
    name: The hyper-parameter name.
    value: The configured candidate or distribution.

  Returns:
    The value to fit with.
  """
  if hasattr(value, 'rvs'):
    return value.rvs()
  if isinstance(value, (list, tuple)):
    return trial.suggest_categorical(name, value)
  return value


def _score(scorer: Any, labels: Any, predictions: Any) -> float:
  """Score a fitted model with a scikit-learn scorer.

  ``_score_func`` is the scorer's own scoring callable. A ``_PredictScorer`` wraps a
  metric with the estimator's prediction method, so the metric is invoked through the
  wrapper's public ``__call__`` where one exists, and through the scoring callable
  only where the wrapper exposes it.

  Args:
    scorer: The scorer.
    labels: The true labels.
    predictions: The model's predictions.

  Returns:
    The score.
  """
  metric = getattr(scorer, '_score_func', None)
  if callable(metric):
    return float(metric(labels, predictions))
  return float(scorer(labels, predictions))


def _accepts(searcher_class: Any, argument: str) -> bool:
  """Report whether a searcher class declares a constructor argument.

  Args:
    searcher_class: The resolved searcher class.
    argument: The constructor argument name to test.

  Returns:
    ``True`` when the argument appears in the constructor signature. A class whose
    signature cannot be inspected is assumed to accept the argument, which keeps an
    exotic searcher usable at the cost of a possible ``TypeError`` from the searcher
    itself -- a clearer error than one this function would have to invent.
  """
  try:
    parameters = inspect.signature(searcher_class).parameters
  except (TypeError, ValueError):
    return True
  if argument in parameters:
    return True
  return any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())


def _bind_search_space(searcher_class: Any, tuning_params: dict[str, Any]) -> dict[str, Any]:
  """Bind the configured search space to the spelling the searcher accepts.

  The configuration vocabulary is deliberately forgiving -- ``param_grid`` and
  ``param_distributions`` are interchangeable there -- but the searcher classes
  are not: passing ``param_grid`` to a randomised searcher raises, and passing
  ``param_distributions`` to a grid searcher raises. Forwarding whichever key the
  operator happened to type therefore breaks every searcher but the one matching
  their spelling. This resolves the vocabulary to the constructor rather than
  guessing from the class name.

  Args:
    searcher_class: The resolved searcher class.
    tuning_params: The configured tuning parameters.

  Returns:
    A new parameter mapping carrying the coerced search space under the accepted
    name, with the other spelling removed.
  """
  declared = next((name for name in _SEARCH_SPACE_ARGUMENTS if name in tuning_params), None)
  if declared is None:
    return dict(tuning_params)

  space = coerce_search_space(tuning_params[declared])
  accepted = declared if _accepts(searcher_class, declared) else next(
    name for name in _SEARCH_SPACE_ARGUMENTS if name != declared
  )
  resolved = {key: value for key, value in tuning_params.items() if key not in _SEARCH_SPACE_ARGUMENTS}
  resolved[accepted] = space
  LOGGER.debug('Bound the search space', extra={'declared': declared, 'accepted': accepted})
  return resolved


class ModelTraining:
  """Fit, calibrate, threshold and score with a pandas-native estimator.

  Attributes:
    features_to_use: The exact, ordered model feature vector, resolved once
      during ``train`` and reused verbatim at scoring time.
    optimal_model_threshold: The decision threshold on the raw score.
    optimal_calib_threshold: The decision threshold on the calibrated score.
    real_rate: ``P(y = 1)`` in the population.
    sample_rate: ``P(y = 1)`` in the effective training distribution.
    best_params: The search outcome, when a search was configured.
    best_score: The search outcome score, when a search was configured.
    calibrated_model: The fitted calibrator, or ``None`` when prior-shift
      correction is used. a initialised this to ``{}`` and
      never reassigned it, so a configured ``BetaCalibration`` or
      ``IsotonicRegression`` was fitted, used for threshold selection, and then
      discarded — meaning production calibration silently used a different method
      from the one the configuration named, and the published score and the
      calibrated threshold were derived from different curves.
  """

  def __init__(
    self,
    model_handle: Any,
    model_params: dict[str, Any] | None = None,
    model_initial_params: dict[str, Any] | None = None,
    tuning_handle: Any = None,
    tuning_params: dict[str, Any] | None = None,
    fit_params: dict[str, Any] | None = None,
    calibration_handle: Any = None,
    calibration_params: dict[str, Any] | None = None,
    calibration_real_rate: float = 0.0,
    threshold: float | None = None,
    calib_threshold: float | None = None,
    threshold_selection_method: str | None = None,
    include_cols: list[str] | None = None,
    exclude_cols: list[str] | None = None,
    balance_class_weight: bool = False,
    eval_size: float | None = None,
    validation_flag: bool = False,
    val_set_col: str = 'eval_set',
    prediction_col: str | None = None,
    prediction_probability_col: str | None = None,
    save_model_path: str | None = None,
    save_calib_model_path: str | None = None,
  ) -> None:
    """Resolve every string handle and normalise every optional parameter.

    Args:
      model_handle: A dotted path to the estimator class, or the class itself.
      model_params: The estimator's kwargs when no search is configured.
      model_initial_params: The base estimator's kwargs when a search *is*
        configured. The two-parameter distinction is non-obvious and a recurring
        source of confusion: **initial** is the base the search explores around;
        **model** is the estimator used when no search is configured.
      tuning_handle: A dotted path to a searcher class, or to a module exposing
        ``create_study`` for Optuna.
      tuning_params: The searcher's kwargs, including ``param_grid`` or
        ``param_distributions``.
      fit_params: Extra keyword arguments splatted into ``fit``. This is how a
        library-native evaluation slot receives the holdout without the framework
        knowing anything about the estimator.
      calibration_handle: A dotted path to a calibrator class.
      calibration_params: The calibrator's kwargs.
      calibration_real_rate: An explicit prior, when the sampling stage did not
        supply one.
      threshold: An explicit decision threshold, used when no selection method
        is configured.
      calib_threshold: An explicit calibrated decision threshold.
      threshold_selection_method: One of the criteria in the calibration module.
      include_cols: A literal feature allow-list.
      exclude_cols: Regular expressions matching excluded feature names.
      balance_class_weight: Whether to inject an imbalance weight.
      eval_size: The holdout fraction. ``0.0`` trains on everything.
      validation_flag: Whether to route the holdout into ``val_set_col``.
      val_set_col: The name of the estimator's evaluation slot.
      prediction_col: The hard prediction column name.
      prediction_probability_col: The probability column name.
      save_model_path: Where to persist the fitted estimator.
      save_calib_model_path: Where to persist the fitted calibrator.
    """
    self.model_handle = resolve(model_handle)
    self.model_params = dict(model_params or {})
    self.model_initial_params = dict(model_initial_params or self.model_params)
    self.tuning_handle = resolve(tuning_handle) if tuning_handle is not None else None
    self.tuning_params = dict(tuning_params or {})
    self.fit_params = dict(fit_params or {})
    self.calibration_handle = resolve(calibration_handle) if calibration_handle else None
    self.calibration_params = dict(calibration_params or {})
    self.calibration_real_rate = float(calibration_real_rate or 0.0)
    self.threshold = threshold if threshold is not None else 0.5
    self.calib_threshold = calib_threshold
    self.threshold_selection_method = threshold_selection_method
    self.include_cols = list(include_cols or [])
    self.exclude_cols = list(exclude_cols or [])
    self.balance_class_weight = balance_class_weight
    self.eval_size = 0.15 if eval_size is None else float(eval_size)
    self.validation_flag = validation_flag
    self.val_set_col = val_set_col
    self.prediction_col = prediction_col or constants.DEFAULT_PREDICTION_COLUMN
    self.prediction_probability_col = (
      prediction_probability_col or constants.DEFAULT_PREDICTION_PROBABILITY_COLUMN
    )
    self.save_model_path = save_model_path
    self.save_calib_model_path = save_calib_model_path

    # Resolved contract, populated by train().
    self.features_to_use: list[str] = []
    self.target_col: str | None = None
    self.unique_label: list[Any] = []
    self.multi_class_flag = False
    self.real_rate: float = 0.0
    self.sample_rate: float = 0.0
    self.optimal_model_threshold: float = self.threshold
    self.optimal_calib_threshold: float = self.calib_threshold if self.calib_threshold is not None else 0.5
    self.best_params: dict[str, Any] | None = None
    self.best_score: float | None = None
    self.calibrated_model: Any = None
    #: Transient. Set only for the duration of the calibrated-column computation in
    #: :meth:`_select_thresholds`, because a wrapper calibrator (``CalibratedClassifierCV``)
    #: scores from the feature matrix rather than from the raw score. It is cleared
    #: in a ``finally`` block: the trainer itself is pickled into the model handle and
    #: shipped to the scoring tier, so a training frame left attached here would be
    #: serialised into an artefact every consumer unpickles.
    self._threshold_probe_frame: pd.DataFrame | None = None
    #: Whether the configured handle denotes a regressor. Resolved once, from the
    #: handle, so every stage that needs to know the task's shape reads one answer
    #: instead of each re-deriving it -- and so the answer cannot disagree between
    #: the split, the calibration and the threshold stages.
    self.is_continuous_target: bool = _is_continuous_target(self.model_handle)
    #: Per-feature fill statistics recorded from the training partition, keyed by
    #: column and then by strategy. This is the *only* usable source for a non-zero
    #: drift fill: a column the current pipeline no longer produces is by definition
    #: absent from the holdout and from the challenger's scored frame, so a fill
    #: read from either of them is unavailable. The incumbent's own training
    #: statistics are the correct reference, and they travel with the incumbent
    #: because the trainer is pickled into the model handle.
    self._feature_fill_stats: dict[str, dict[str, float]] = {}

  # ---------------------------------------------------------------------- #
  # Training
  # ---------------------------------------------------------------------- #
  def train(
    self, frame: pd.DataFrame, metadata: Metadata
  ) -> tuple[ModelTraining, Any, Any, pd.DataFrame, pd.DataFrame]:
    """Fit the model, calibrate it, and select the operating point.

    Args:
      frame: The preprocessed training frame.
      metadata: The schema contract.

    Returns:
      The five-tuple ``(self, fitted_model, calibrated_model, train_frame,
      holdout_frame)``.

    Raises:
      HoldoutRequiredError: If a holdout-requiring path was requested without
        one.
    """
    frame = self._resolve_features(frame, metadata)
    train_frame, holdout_frame = self._split(frame, metadata)

    x_train = train_frame[self.features_to_use]
    y_train = train_frame[self.target_col]
    self._inject_imbalance_weight(train_frame, metadata)

    if self.tuning_params:
      fitted = self._search(x_train, y_train, holdout_frame)
    else:
      fitted = self._direct_fit(x_train, y_train, holdout_frame)

    self._record_rates(train_frame, metadata)
    self._record_fill_stats(train_frame)
    self._is_continuous_flag = self._detect_continuous(train_frame)
    # The calibrator is fitted on the training partition and the operating point is
    # selected on the holdout. Both steps previously read the *same* slice -- the
    # holdout when one existed -- so the calibrator was applied to the very rows it
    # was fitted on and the threshold was then chosen on those in-sample
    # calibrated scores. In-sample isotonic output is a staircase that attains the
    # training positive rate on each step, which places the F1/MCC/KS optimum far
    # lower than the out-of-sample optimum; the published operating point was
    # therefore systematically too permissive, with no log line distinguishing the
    # two. The holdout exists precisely to be data neither of those steps has seen.
    self.calibrated_model = self._fit_calibrator(fitted, train_frame, holdout_frame)
    # The label set is recorded *before* the operating point is selected, because
    # selecting it needs to know whether the target has a single positive class.
    # Recording it afterwards left `multi_class_flag` at its constructor default of
    # False while the selection ran, so the multiclass guard could never fire.
    self._finalise_labels(train_frame)
    self._select_thresholds(fitted, train_frame, holdout_frame)
    self._record_importance(fitted)
    self._persist(fitted)

    return self, fitted, self.calibrated_model or {}, train_frame, holdout_frame

  def _resolve_features(self, frame: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
    """Resolve the feature vector and coerce the target to float.

    The trainer always requests the full feature list, narrowed by
    ``exclude_cols`` (regular expressions) and ``include_cols`` (a literal
    allow-list). The resolved order is baked into the persisted handle, so
    reordering the configuration silently changes the meaning of every existing
    model.

    Args:
      frame: The preprocessed training frame.
      metadata: The schema contract.

    Returns:
      The frame with a float target.
    """
    self.target_col = metadata.target_col
    self.features_to_use = metadata.feature_resolution(
      args=('feature_cols',), feature_exclusions=self.exclude_cols, feature_inclusions=self.include_cols
    )
    if not self.features_to_use:
      raise ValueError(
        'No features were resolved for the model. The trainer requests metadata.feature_cols, so '
        'either the load node declared no features or every one was excluded.'
      )
    frame[self.target_col] = frame[self.target_col].astype('float')
    LOGGER.info('Resolved the model feature vector', extra={'features': self.features_to_use})
    return frame

  def _split(self, frame: pd.DataFrame, metadata: Metadata) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the in-memory holdout.

    Args:
      frame: The preprocessed training frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the training frame and the holdout. With ``eval_size`` at
      zero the holdout is an empty frame and the model is fitted on everything,
      which makes every reported threshold optimistic.
    """
    LOGGER.info(
      'Training frame received',
      extra={'shape': frame.shape, 'target_distribution': frame[metadata.target_col].value_counts().to_dict()},
    )
    if self.eval_size == 0.0:
      LOGGER.warning(
        'eval_size is 0.0: the model is fitted on the whole frame and every threshold is selected '
        'in-sample, so the reported operating point will be optimistic'
      )
      return frame, pd.DataFrame()

    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    # Stratification is a property of a *classification* split. Applying it to a
    # continuous target asks scikit-learn to reproduce an exact value distribution
    # it cannot, so the guard is on the target's cardinality rather than on a
    # family flag: a continuous target has more distinct values than the limit, and
    # is therefore never stratified.
    stratify = (
      frame[metadata.target_col]
      if (not self.is_continuous_target and frame[metadata.target_col].nunique() <= 10)
      else None
    )
    train_frame, holdout_frame = train_test_split(
      frame, test_size=self.eval_size, random_state=42, stratify=stratify
    )
    # Release the full frame so the two slices do not coexist in driver memory.
    del frame
    LOGGER.info(
      'Created the holdout split',
      extra={
        'train_shape': train_frame.shape,
        'holdout_shape': holdout_frame.shape,
        'train_target': train_frame[metadata.target_col].value_counts().to_dict(),
      },
    )
    return train_frame, holdout_frame

  def _inject_imbalance_weight(self, train_frame: pd.DataFrame, metadata: Metadata) -> None:
    """Inject the estimator's imbalance parameter.

    Two different mechanisms are selected by the estimator's class: a per-class
    weight vector for the gradient-boosting libraries that take one, and a
    scalar positive-class weight for the rest. The values are injected into
    *both* parameter dictionaries, because one builds the estimator for the
    search and the other is used in the non-tuning path — and only this block
    guarantees they agree.

    Args:
      train_frame: The training frame.
      metadata: The schema contract, which carries the sampling stage's weights.
    """
    if not self.balance_class_weight:
      return
    labels = train_frame[self.target_col]
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())

    if _takes_class_weights(type(self.model_handle)):
      weights = metadata.class_weights
      if not weights:
        total = positives + negatives
        weights = [
            np.round(total / (2 * negatives), 4) if negatives else 1.0,
            np.round(total / (2 * positives), 4) if positives else 1.0,
        ]
      weight_value: Any = np.asarray(weights, dtype=float)
      LOGGER.info('Injecting per-class weights', extra={'class_weights': weight_value.tolist()})
    else:
      weight = metadata.scale_pos_weight
      if not weight:
        if not positives:
          raise ValueError('Cannot compute scale_pos_weight: the training frame has no positive rows')
        weight = np.round(negatives / positives, 4)
      weight_value = weight
      LOGGER.info('Injecting a positive-class weight', extra={'scale_pos_weight': weight_value})

    weight_key = 'scale_pos_weight' if not isinstance(weight_value, np.ndarray) else 'class_weights'
    self.model_params[weight_key] = weight_value
    self.model_initial_params[weight_key] = weight_value

  def _search(self, x_train: pd.DataFrame, y_train: pd.Series, holdout: pd.DataFrame) -> Any:
    """Run the configured hyperparameter search and return the fitted estimator.

    The searcher is invoked as ``tuning_handle(**tuning_params)`` with
    ``estimator`` injected, so anything honouring that constructor contract
    works: ``GridSearchCV``, ``RandomizedSearchCV``, ``HalvingGridSearchCV`` and
    ``OptunaSearchCV`` all work without a framework change.

    Args:
      x_train: The training feature matrix.
      y_train: The training labels.
      holdout: The holdout frame, required by the Optuna path.

    Returns:
      The fitted estimator.

    Raises:
      HoldoutRequiredError: If the Optuna path is configured without a holdout.
    """
    LOGGER.info('Running hyperparameter search', extra={'searcher': getattr(self.tuning_handle, '__name__', None)})

    if getattr(self.tuning_handle, '__name__', '') == 'optuna.study':
      if holdout.empty:
        raise HoldoutRequiredError(
          'Optuna tuning needs a fixed evaluation set. Set eval_size (for example eval_size: 0.2) '
          'under the modeling block in parameters.yml.'
        )
      return self._search_with_optuna(x_train, y_train, holdout[self.features_to_use], holdout[self.target_col])

    # The halving searchers are absent from ``sklearn.model_selection`` until this
    # opt-in runs, so the handle must be enabled before it is constructed. The
    # comparison is on the bare class name: ``__name__`` is never the dotted path,
    # so a comparison against a dotted path never fires.
    if getattr(self.tuning_handle, '__name__', '') in HALVING_SEARCHERS:
      from sklearn.experimental import enable_halving_search_cv  # noqa: F401, PLC0415 - opt-in by import

      del enable_halving_search_cv

    params = _bind_search_space(self.tuning_handle, self.tuning_params)
    params['estimator'] = self.model_handle(**self.model_initial_params)

    searcher = self.tuning_handle(**params)
    searcher.fit(x_train, y_train)
    self.best_score = float(searcher.best_score_)
    self.best_params = dict(searcher.best_params_)
    LOGGER.info(
      'Hyperparameter search complete',
      extra={
        'best_score': self.best_score,
        'best_params': self.best_params,
        'candidates': len(searcher.cv_results_.get('params', [])),
      },
    )
    return searcher.best_estimator_

  def _search_with_optuna(
    self,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
  ) -> Any:
    """Run an Optuna study and return a refitted estimator.

    The tuning handle is used as a *module*: the reflective resolver imports
    ``optuna.study`` and ``create_study`` is called on it, so the framework never
    imports Optuna itself and a run without it installed is unaffected.

    The objective deliberately scores on the holdout, which is a documented
    weakness: the same holdout is then used to fit the decision threshold and to
    score the challenger in the promotion gate, so the headline metric is
    selected on the data it is reported on. The conventional remedy is a
    three-way split, and the framework supports it by adding a second split
    level.

    Args:
      x_train: The training feature matrix.
      y_train: The training labels.
      x_holdout: The holdout feature matrix.
      y_holdout: The holdout labels.

    Returns:
      The refitted estimator.
    """
    import optuna  # noqa: PLC0415 - resolved through the module handle, imported here for typing

    del optuna
    study_module = self.tuning_handle
    sampler = self._build_optional(self.tuning_params.get('sampler_handle'), self.tuning_params.get('sampler_params'))
    pruner = self._build_optional(self.tuning_params.get('pruner_handle'), self.tuning_params.get('pruner_params'))
    space = coerce_search_space(self.tuning_params.get('param_grid', {}))
    scoring = self.tuning_params.get('scoring', 'roc_auc')

    from sklearn.metrics import get_scorer  # noqa: PLC0415

    scorer_names = [scoring] if isinstance(scoring, str) else list(scoring)
    scorers = [get_scorer(name) for name in scorer_names]

    def objective(trial: Any) -> tuple[float, ...]:
      """Score one trial.

      Args:
        trial: The Optuna trial.

      Returns:
        A tuple of scores, one per configured metric.
      """
      params = {name: _suggest(trial, name, value) for name, value in space.items()}
      model = self.model_handle(**self.model_initial_params, **params)
      model.fit(x_train, y_train, **self.fit_params)
      train_scores = [_score(scorer, y_train, model.predict(x_train)) for scorer in scorers]
      holdout_scores = [_score(scorer, y_holdout, model.predict(x_holdout)) for scorer in scorers]
      LOGGER.info(
        'Optuna trial',
        extra={'trial': trial.number, 'params': params, 'train': train_scores, 'holdout': holdout_scores},
      )
      return tuple(holdout_scores)

    direction = self.tuning_params.get('direction', 'maximize')
    trials = self.tuning_params.get('n_trials', 20)
    if isinstance(direction, str):
      study = study_module.create_study(sampler=sampler, pruner=pruner, direction=direction)
      study.optimize(objective, n_trials=trials, n_jobs=self.tuning_params.get('n_jobs', 1))
      best_params = study.best_params
      self.best_score = float(study.best_value)
    else:
      directions = list(direction)
      study = study_module.create_study(sampler=sampler, pruner=pruner, directions=directions)
      study.optimize(objective, n_trials=trials, n_jobs=self.tuning_params.get('n_jobs', 1))
      # Multi-objective resolution filters on the first direction only, so the
      # remaining objectives are reporting rather than optimising. This is stated
      # rather than hidden.
      chooser = max if directions[0] == 'maximize' else min
      best_params = chooser(study.best_trials, key=lambda t: t.values[0]).params

    self.best_params = dict(best_params)
    fitted = self.model_handle(**self.model_initial_params, **best_params)
    fitted.fit(x_train, y_train, **self.fit_params)
    # Do not leave a module reference on the persisted handle.
    self.tuning_handle = None
    return fitted

  def _build_optional(self, handle: Any, params: Any) -> Any:
    """Build an optional Optuna component.

    Args:
      handle: A dotted path, or ``None``.
      params: The component's kwargs.

    Returns:
      The constructed component, or ``None``.
    """
    if not handle:
      return None
    factory = resolve(handle)
    return factory(**params) if params else factory()

  def _direct_fit(self, x_train: pd.DataFrame, y_train: pd.Series, holdout: pd.DataFrame) -> Any:
    """Fit the estimator directly, optionally with a library-native eval slot.

    Routing the holdout into ``val_set_col`` gives early stopping and
    per-iteration logging without the framework knowing anything about the
    estimator: the same three lines serve LightGBM's ``eval_set``, XGBoost's
    ``eval_set`` and CatBoost's ``eval_set``.

    Args:
      x_train: The training feature matrix.
      y_train: The training labels.
      holdout: The holdout frame.

    Returns:
      The fitted estimator.

    Raises:
      HoldoutRequiredError: If cross-validation was requested without a holdout.
    """
    fit_params = dict(self.fit_params)
    if self.validation_flag:
      if holdout.empty:
        raise HoldoutRequiredError(
          'Cross-validation was requested (validation_flag: True) but no holdout exists. Set '
          'eval_size to a non-zero value, e.g. eval_size: 0.2. The framework raised a '
          'bare ValueError from inside train(), which named neither the parameter nor its owner.'
        )
      fit_params[self.val_set_col] = (holdout[self.features_to_use], holdout[self.target_col])
      LOGGER.info('Routed the holdout into the estimator evaluation slot', extra={'slot': self.val_set_col})

    LOGGER.info('Fitting estimator', extra={'handle': self.model_handle.__name__, 'model_params': self.model_params})
    estimator = self.model_handle(**self.model_params)
    fitted = estimator.fit(x_train, y_train, **fit_params)
    return fitted

  def _record_rates(self, train_frame: pd.DataFrame, metadata: Metadata) -> None:
    """Record the population and training positive rates.

    The two rates are the sole inputs to prior-shift calibration, and their
    ratio is the exact factor needed to undo the training-time reweighting. The
    three-way branch mirrors the three-way branch in the weight injection, so
    the two stay consistent.

    Args:
      train_frame: The training frame.
      metadata: The schema contract.
    """
    labels = train_frame[self.target_col]
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())

    if metadata.real_rate:
      self.real_rate = float(metadata.real_rate)
    else:
      self.real_rate = _safe_rate(positives, positives + negatives)

    if 'class_weights' in self.model_params:
      pos_weight = float(np.asarray(self.model_params['class_weights'])[-1])
      neg_weight = float(np.asarray(self.model_params['class_weights'])[0])
    elif 'scale_pos_weight' in self.model_params:
      pos_weight = float(self.model_params['scale_pos_weight'])
      neg_weight = 1.0
    else:
      pos_weight = 1.0
      neg_weight = 1.0

    self.sample_rate = _safe_rate(positives * pos_weight, positives * pos_weight + negatives * neg_weight)
    LOGGER.info('Recorded the class rates', extra={'real_rate': self.real_rate, 'sample_rate': self.sample_rate})

  def _record_fill_stats(self, frame: pd.DataFrame) -> None:
    """Record per-feature fill statistics from the training partition.

    Recorded during training so that the promotion gate can fill a feature the
    current pipeline no longer produces with the *incumbent's own* training
    statistic rather than a fabricated zero. Zero is a poor surrogate: for a
    standardised or encoded feature it is a meaningful, adversarial value rather
    than a neutral one, so a zero-filled incumbent is scored by a materially
    different model than the one in production, on the exact comparison that
    decides a promotion. A statistic that cannot be computed is simply not
    recorded, and the gate falls back to zero for that feature alone.

    Args:
      frame: The training frame.
    """
    stats: dict[str, dict[str, float]] = {}
    for column in self.features_to_use:
      if column not in frame.columns:
        continue
      series = frame[column]
      if not pd.api.types.is_numeric_dtype(series):
        continue
      entry: dict[str, float] = {}
      for name, value in (('mean', series.mean()), ('median', series.median())):
        try:
          number = float(value)
        except (TypeError, ValueError):
          continue
        if np.isfinite(number):
          entry[name] = number
      if entry:
        stats[column] = entry
    self._feature_fill_stats = stats
    LOGGER.info('Recorded fill statistics for the drift gate', extra={'columns': len(stats)})

  def _fit_calibrator(
    self,
    fitted: Any,
    train_frame: pd.DataFrame,
    holdout: pd.DataFrame,
  ) -> Any:
    """Fit the configured calibrator and compute the calibrated operating column.

    Five strategies, distinguished by input signature: a wrapper calibrator
    consumes the feature matrix, the other three consume the scalar score, and
    the fallback is the framework's own prior-shift correction.

    a fitted the calibrator into a function-local variable
    and returned nothing, so the persisted calibrated-model dataset always held
    an empty dict. The fitted calibrator is stored on the trainer and
    returned, which is what makes the configured strategy actually reach
    production.

    Args:
      fitted: The fitted estimator.
      train_frame: The training partition the calibrator is fitted on.
      holdout: The holdout. Retained for signature stability with the call site and
        deliberately *not* used: fitting here would contaminate the slice the
        operating point is selected on.

    Returns:
      The fitted calibrator, or ``None`` when prior-shift correction is used.
    """
    # The calibrator is fitted on the TRAINING partition, never on the holdout.
    # The holdout is the only slice the operating point is selected on, and a
    # calibrator fitted on it would make the selected threshold an in-sample
    # quantity. When no holdout exists the training partition is the only data
    # available, and the framework already treats that case as in-sample: the
    # evaluation nodes degrade to no-ops and promotion is unavailable.
    #
    # The holdout is dropped from the signature *after* this choice is recorded, so
    # a future edit that consults it cannot fail with an unbound name -- which is
    # what a bare `del holdout` above this line would have permitted.
    working = train_frame if not train_frame.empty else holdout
    del holdout
    x_score = working[self.features_to_use]
    y_score = working[self.target_col]
    raw = pd.Series(positive_probability(fitted, x_score), index=working.index, name='raw_score')

    if not self.calibration_handle:
      return None

    name = self.calibration_handle.__name__
    reshaped = raw.to_numpy().reshape(-1, 1)
    if name == 'CalibratedClassifierCV':
      calibrator = self.calibration_handle(fitted, **self.calibration_params)
      calibrator.fit(x_score, y_score)
    elif name in ('LogisticRegression', 'BetaCalibration', 'IsotonicRegression'):
      calibrator = self.calibration_handle(**self.calibration_params)
      calibrator.fit(reshaped, y_score)
    else:
      LOGGER.warning(
        'Unrecognised calibration handle; prior-shift correction will be used instead',
        extra={
          'handle': name,
          'supported': ['CalibratedClassifierCV', 'LogisticRegression', 'BetaCalibration', 'IsotonicRegression'],
        },
      )
      return None
    LOGGER.info('Fitted the probability calibrator', extra={'handle': name, 'rows': len(working)})
    return calibrator

  def _detect_continuous(self, frame: pd.DataFrame) -> bool:
    """Decide whether the target is continuous rather than categorical.

    The decision is made once, during training, from the target's own cardinality
    against a fixed limit. Deciding it per call site is what let the classification
    path's assumptions -- stratify on the label, threshold the score, binarise the
    prediction -- leak into the regression path, where each of them is either
    meaningless or fatal.

    The limit is deliberately generous for a classifier and deliberately low for a
    regressor: a classification target is a handful of classes, and a continuous
    target has far more distinct values than any categorical one has classes. A
    target with a large number of *integer* classes is therefore treated as
    continuous, which is the safe direction -- the regression path degrades to
    skipping the threshold rather than the classification path attempting a
    precision-recall curve over a continuous label.

    Args:
      frame: The training frame.

    Returns:
      Whether the target is continuous.
    """
    distinct = int(frame[self.target_col].nunique())
    continuous = distinct > CONTINUOUS_TARGET_DISTINCT_LIMIT
    LOGGER.info(
      'Classified the target',
      extra={'target_col': self.target_col, 'distinct_values': distinct, 'continuous': continuous},
    )
    return continuous

  def _is_continuous(self) -> bool:
    """Return the recorded target classification.

    Returns:
      Whether the target is continuous, defaulting to false before training runs.
    """
    return bool(getattr(self, '_is_continuous_flag', False))

  def _select_thresholds(
    self,
    fitted: Any,
    train_frame: pd.DataFrame,
    holdout: pd.DataFrame,
  ) -> None:
    """Select both decision thresholds, on the configured criterion.

    Two thresholds are always both computed: one gates the raw score and one
    gates the calibrated score. They are stored on the trainer, which is what
    makes the persisted handle self-sufficient.

    A **continuous target** has no decision to make, so neither threshold is
    selected and the configured values are left in place. Every criterion in
    :data:`THRESHOLD_METHODS` is defined over a binary label, so running one
    against a continuous target raised ``ValueError: continuous format is not
    supported`` from inside scikit-learn -- on every regression run, after the
    model had already been fitted. Skipping the selection is not a silent
    degradation: the regression tier never applies a threshold, and it says so.

    Args:
      fitted: The fitted estimator.
      train_frame: The training frame.
      holdout: The holdout frame.
    """
    if self.is_continuous_target:
      LOGGER.info(
        'Threshold selection skipped: a continuous target has no decision to threshold. The '
        'regression tier compares predictions to the target on the target scale.',
        extra={'target_col': self.target_col, 'threshold': float(self.threshold)},
      )
      self.optimal_model_threshold = float(self.threshold)
      self.optimal_calib_threshold = float(self.calib_threshold if self.calib_threshold is not None else self.threshold)
      return

    # A second, data-derived condition on the same guard. `is_continuous_target`
    # describes the *estimator class*, resolved at construction; the target's own
    # cardinality is a property of the data and is only known after the frame is
    # loaded. It is recorded by `_detect_continuous` during training and by
    # `unique_label` immediately after, and both were computed and then never
    # read -- so a target that is not continuous but has more than two classes
    # reached a criterion that scikit-learn defines only over a binary label, and
    # the run died with `ValueError: multiclass format is not supported` after
    # the model had been fitted. The unconfigured path is affected too, because
    # it defaults to `pr_curve`.
    #
    # There is no single positive class to gate on with K > 2, so the learned
    # thresholds have no meaning and are left at their configured values. This is
    # the same reasoning the regression tier uses, applied for the same reason.
    if self._is_continuous() or self.multi_class_flag:
      LOGGER.info(
        'Threshold selection skipped: the target has more than two classes, so there is no single '
        'positive class for a decision threshold to gate on. The configured thresholds are left in '
        'place and the hard prediction is the estimator argmax. Ranking metrics such as AUC remain '
        'valid, but a binary operating point does not.',
        extra={
          'target_col': self.target_col,
          'threshold': float(self.threshold),
          'multiclass': bool(self.multi_class_flag),
          'continuous': bool(self._is_continuous()),
        },
      )
      self.optimal_model_threshold = float(self.threshold)
      self.optimal_calib_threshold = float(self.calib_threshold if self.calib_threshold is not None else self.threshold)
      return

    working = holdout if not holdout.empty else train_frame
    probe = pd.DataFrame(index=working.index)
    probe[self.target_col] = working[self.target_col]
    probe['raw_score'] = positive_probability(fitted, working[self.features_to_use])
    # The frame is published only for the duration of the calibrated-column
    # computation, because that is the only step that needs it. Holding a
    # training frame on the instance would drag it into the pickled model handle,
    # which is persisted, shipped to scoring, and unpickled there.
    self._threshold_probe_frame = working
    try:
      probe['calib_score'] = self._calibrated_column(probe['raw_score'])
    finally:
      self._threshold_probe_frame = None

    if not self.threshold_selection_method:
      self.optimal_model_threshold = float(self.threshold)
      self.optimal_calib_threshold = (
        float(self.calib_threshold)
        if self.calib_threshold is not None
        else select_threshold(probe, 'pr_curve', self.target_col, 'calib_score')
      )
    else:
      self.optimal_model_threshold = select_threshold(
        probe, self.threshold_selection_method, self.target_col, 'raw_score'
      )
      self.optimal_calib_threshold = select_threshold(
        probe, self.threshold_selection_method, self.target_col, 'calib_score'
      )
    self._align_estimator_threshold(fitted)
    LOGGER.info(
      'Selected the operating point',
      extra={
        'method': self.threshold_selection_method or 'configured',
        'model_threshold': self.optimal_model_threshold,
        'calib_threshold': self.optimal_calib_threshold,
      },
    )

  def _align_estimator_threshold(self, fitted: Any) -> None:
    """Align an estimator's own internal decision cut-off with the framework's.

    CatBoost carries a private ``predict`` cut-off (``set_probability_threshold``)
    which defaults to 0.5. Leaving it there makes the persisted estimator disagree
    with the framework about what a positive prediction is: the score table's
    ``prediction`` column is thresholded by the framework, while
    ``model.predict()`` -- and therefore ``model.score()``, and therefore any
    accuracy logged from a diagnostic -- still uses 0.5. Two numbers describing
    the same model in the same run then describe two different classifiers.

    The alignment used to sit in the fit path, where the selected threshold did not
    yet exist and the constructor default was applied, so it could never be
    correct. It is applied here, after selection, where the value is known.

    Estimators that carry no such cut-off are left untouched, which is the
    majority: the capability is probed rather than assumed, because naming a
    concrete third-party class here would couple the framework to one estimator
    family for no benefit.

    Args:
      fitted: The fitted estimator.
    """
    setter = getattr(fitted, 'set_probability_threshold', None)
    if not callable(setter):
      return
    setter(self.optimal_model_threshold)
    LOGGER.info(
      'Aligned the estimator internal decision threshold with the framework operating point',
      extra={'estimator': type(fitted).__name__, 'threshold': self.optimal_model_threshold},
    )

  def _calibrated_column(self, raw: pd.Series) -> pd.Series:
    """Produce the calibrated score for a raw score series.

    The calibrated column must be produced by *exactly* the transform that
    scoring time will apply to it, because the calibrated decision threshold is
    selected on this column and then gates that same column in the published
    score. A wrapper calibrator was the gap: :meth:`_fit_calibrator` supports
    ``CalibratedClassifierCV`` -- it is the one branch that consumes the feature
    matrix rather than the scalar score -- but this method had no case for it, so
    it fell through to prior-shift correction. The threshold was then chosen on a
    prior-shift scale and applied at scoring time to an isotonic or Platt output
    (``modeling.scoring`` dispatches on the calibrator's real type), so the
    published ``calib_`` flag gated a column it had never been selected on. With a
    moderately skewed raw score that is the difference between a targeting
    operating point and flagging most of the population.

    The wrapper needs the feature matrix, so it is passed in. It cannot be
    reconstructed from the raw score, and reconstructing it from "every column that
    is not a framework output column" -- the scoring-side fallback -- produces the
    target column and the identifiers too, in *frame* order rather than the
    resolved feature order that is baked into the handle.

    Args:
      raw: The raw score.

    Returns:
      The calibrated score, on the same scale the scoring path will produce.
    """
    if self.calibrated_model is not None:
      name = type(self.calibrated_model).__name__
      reshaped = raw.to_numpy().reshape(-1, 1)
      if name in ('BetaCalibration', 'IsotonicRegression'):
        calibrated = pd.Series(self.calibrated_model.predict(reshaped), index=raw.index)
        return calibrated.fillna(0.0)
      if name == 'LogisticRegression':
        return pd.Series(self.calibrated_model.predict_proba(reshaped)[:, 1], index=raw.index)
      if name == 'CalibratedClassifierCV' and self._threshold_probe_frame is not None:
        # The wrapper re-scores from the feature matrix, so the training-time
        # frame is required. It is supplied by `_select_thresholds` for exactly
        # this call and cleared immediately afterwards, so it never reaches the
        # persisted trainer and never inflates the pickled handle.
        per_class = self.calibrated_model.predict_proba(
          self._threshold_probe_frame[self.features_to_use]
        )
        return pd.Series(per_class[:, -1], index=raw.index)
      if name == 'CalibratedClassifierCV':
        LOGGER.warning(
          'The configured wrapper calibrator needs the feature matrix, which is unavailable here, '
          'so the calibrated threshold is selected on the prior-shift column instead. The published '
          'calibrated flag may not correspond to the intended operating point.',
          extra={'handle': name},
        )
    return calibrate_probability(raw, self.real_rate, self.sample_rate)

  def _record_importance(self, fitted: Any) -> None:
    """Log feature importance without failing when the estimator has none.

    reading ``fitted_model.feature_importances_``
    unconditionally *after* fitting, so any estimator without the attribute —
    ``LogisticRegression`` with the default L2 penalty, ``SVC(probability=True)``,
    ``MLPClassifier`` — raised ``AttributeError`` and the run failed on a purely
    cosmetic feature. That excluded the entire non-ensemble estimator family for
    no benefit. The attribute is read defensively, with a coefficient
    fallback, and its absence is logged rather than raised.
    """
    importance = getattr(fitted, 'feature_importances_', None)
    source = 'feature_importances_'
    if importance is None:
      coefficients = getattr(fitted, 'coef_', None)
      if coefficients is not None:
        importance = np.abs(np.ravel(coefficients))
        source = 'coef_'
    if importance is None:
      LOGGER.info(
        'Estimator exposes neither feature_importances_ nor coef_; importance was not recorded',
        extra={'handle': type(fitted).__name__},
      )
      return
    pairs = dict(zip(self.features_to_use, np.ravel(importance), strict=False))
    LOGGER.info(
      'Feature importance',
      extra={'source': source, 'ranked': sorted(pairs.items(), key=lambda item: -item[1])},
    )

  def _persist(self, fitted: Any) -> None:
    """Persist the fitted estimator and the fitted calibrator.

    What is written is the **fitted object**, not a description of how to build
    one. The previous version wrote ``{'estimator': <class name>, 'params': <ctor
    kwargs>}`` to the model path while writing the real fitted calibrator to the
    calibration path one line below -- so the two halves of one method disagreed
    about what "persist" meant, and the model artefact was unusable: loading it
    yields a dict with no ``predict_proba``, and the search outcome (``best_params``)
    was discarded, making a 200-candidate search unrecoverable. A description of a
    class is a recipe, not an artefact; the fitted estimator *is* the artefact, and
    it is also the only thing that carries the tuned parameters.

    Args:
      fitted: The fitted estimator.
    """
    if self.save_model_path:
      save_pickle(fitted, self.save_model_path)
    if self.save_calib_model_path and self.calibrated_model is not None:
      save_pickle(self.calibrated_model, self.save_calib_model_path)

  # ---------------------------------------------------------------------- #
  # Scoring
  # ---------------------------------------------------------------------- #
  def score(
    self,
    fitted: Any,
    calibrated: Any,
    frame: pd.DataFrame,
    metadata: Metadata,
    data_type: str = 'score',
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a frame and return the uncalibrated and calibrated projections.

    Args:
      fitted: The fitted estimator.
      calibrated: The fitted calibrator, or ``{}``.
      frame: The preprocessed scoring frame.
      metadata: The schema contract.
      data_type: The slice label, used only for the run log.

    Returns:
      A two-tuple of the raw-scored frame and the calibrated-scored frame.
    """
    LOGGER.info('Scoring a frame', extra={'slice': data_type, 'shape': frame.shape, 'ids': metadata.id_cols})
    if not (hasattr(fitted, 'predict_proba') or hasattr(fitted, 'decision_function')):
      # Dispatch on the estimator's capability rather than on a configured flag.
      # A regressor has no probability column and no positive class, so the
      # classification projection cannot describe it; routing it through anyway
      # raised `AttributeError: 'XGBRegressor' object has no attribute
      # 'predict_proba'` on the first frame, which is why the shipped
      # `modeling_reg_params` estimator could never score. The flag would also be
      # a second source of truth that could disagree with the fitted object.
      LOGGER.info(
        'Estimator exposes neither predict_proba nor decision_function; using the regression projection'
      )
      scored = score_regression_frame(
        model=fitted,
        frame=frame,
        features_to_use=self.features_to_use,
        prediction_col=self.prediction_col,
        run_date=getattr(metadata, 'run_date', None),
      )
      raw = project_score_frame(
        scored, metadata.id_cols, metadata.target_col, self.prediction_col,
        self.prediction_probability_col, metadata.model_key, calibrated=False,
      )
      calibrated_projection = project_score_frame(
        scored, metadata.id_cols, metadata.target_col, self.prediction_col,
        self.prediction_probability_col, metadata.model_key, calibrated=True,
      )
      return raw, calibrated_projection

    scored = score_frame(
      model=fitted,
      calibrated_model=calibrated or self.calibrated_model,
      frame=frame,
      features_to_use=self.features_to_use,
      target_col=metadata.target_col,
      prediction_col=self.prediction_col,
      prediction_probability_col=self.prediction_probability_col,
      model_threshold=self.optimal_model_threshold,
      calib_threshold=self.optimal_calib_threshold,
      labels=self.unique_label or [0, 1],
      multi_class=self.multi_class_flag,
      real_rate=self.real_rate,
      sample_rate=self.sample_rate,
    )
    raw = project_score_frame(
      scored, metadata.id_cols, metadata.target_col, self.prediction_col,
      self.prediction_probability_col, metadata.model_key, calibrated=False,
    )
    calibrated_projection = project_score_frame(
      scored, metadata.id_cols, metadata.target_col, self.prediction_col,
      self.prediction_probability_col, metadata.model_key, calibrated=True,
    )
    return raw, calibrated_projection

  def _finalise_labels(self, train_frame: pd.DataFrame) -> None:
    """Record the label set and the multiclass flag from the training frame.

    Args:
      train_frame: The training frame.
    """
    self.unique_label = sorted(train_frame[self.target_col].unique().tolist())
    self.multi_class_flag = len(self.unique_label) > 2
    LOGGER.info('Recorded the label set', extra={'labels': self.unique_label, 'multiclass': self.multi_class_flag})


class SparkModelTraining:
  """Fit and score with a Spark ML pipeline.

  a's ``train`` returned a one-tuple while the node unpacked a
  five-tuple, so the Spark tier was dispatched to but unreachable — a runtime
  failure waiting for anyone who sets ``model_type: pyspark``. It returns the
  same five-tuple as the pandas trainer, so both tiers are interchangeable.
  """

  def __init__(
    self,
    model_handle: Any,
    model_params: dict[str, Any] | None = None,
    tuning_handle: Any = None,
    tuning_params: dict[str, Any] | None = None,
    evaluator_handle: Any = None,
    evaluator_params: dict[str, Any] | None = None,
    prediction_col: str | None = None,
    save_model_path: str | None = None,
  ) -> None:
    """Resolve handles and normalise parameters.

    Args:
      model_handle: A dotted path to the estimator class.
      model_params: The estimator's kwargs.
      tuning_handle: A dotted path to a cross-validator class.
      tuning_params: The cross-validator's kwargs, including ``param_grid``.
      evaluator_handle: A dotted path to an evaluator class.
      evaluator_params: The evaluator's kwargs, including a ``labelCol``.
      prediction_col: The prediction column name.
      save_model_path: Where to persist the fitted pipeline.
    """
    self.model_handle = resolve(model_handle)
    self.model_params = dict(model_params or {})
    self.tuning_handle = resolve(tuning_handle) if tuning_handle else None
    self.tuning_params = dict(tuning_params or {})
    self.evaluator_handle = resolve(evaluator_handle) if evaluator_handle else None
    self.evaluator_params = dict(evaluator_params or {})
    self.prediction_col = prediction_col or constants.DEFAULT_PREDICTION_COLUMN
    self.save_model_path = save_model_path
    self.features_to_use: list[str] = []
    self.target_col: str | None = None

  def train(
    self, frame: Any, metadata: Metadata
  ) -> tuple[SparkModelTraining, Any, dict[str, Any], Any, Any]:
    """Fit a Spark ML pipeline, optionally through a cross-validator.

    Args:
      frame: The preprocessed Spark training frame.
      metadata: The schema contract.

    Returns:
      The five-tuple, matching :class:`ModelTraining`.
    """
    from pyspark.ml import Pipeline  # noqa: PLC0415
    from pyspark.ml.feature import VectorAssembler  # noqa: PLC0415

    self.target_col = metadata.target_col
    self.features_to_use = metadata.feature_resolution(args=('numerical_cols', 'categorical_cols', 'feature_cols'))
    assembler = VectorAssembler(
      inputCols=self.features_to_use,
      outputCol='__features',
      handleInvalid='keep',
    )
    estimator = self.model_handle(**self.model_params, featuresCol='__features')

    if self.tuning_handle and self.tuning_params:
      from pyspark.ml.tuning import ParamGridBuilder  # noqa: PLC0415

      grid = ParamGridBuilder()
      for name, values in (self.tuning_params.get('param_grid') or {}).items():
        builder = grid.addGrid
        for value in values if isinstance(values, (list, tuple)) else [values]:
          builder(getattr(estimator, name), value)
      pipeline = Pipeline(stages=[assembler, estimator])
      validator = self.tuning_handle(
        estimator=pipeline,
        estimatorParamMaps=grid.build(),
        numFolds=self.tuning_params.get('num_folds', 3),
      )
      fitted = validator.fit(frame)
      self.best_model = fitted.bestModel
    else:
      fitted = Pipeline(stages=[assembler, estimator]).fit(frame)
      self.best_model = fitted

    if self.save_model_path:
      self.best_model.write().overwrite().save(self.save_model_path)
    LOGGER.info(
      'Fitted a Spark ML pipeline', extra={'features': len(self.features_to_use), 'tuned': bool(self.tuning_params)}
    )
    return self, self.best_model, {}, frame, frame

  def score(
    self,
    fitted: Any,
    calibrated: dict[str, Any],
    frame: Any,
    metadata: Metadata,  # noqa: ARG001 - part of the scoring contract shared with the other tiers
    data_type: str = 'score',
  ) -> tuple[Any, Any]:
    """Score a Spark frame.

    Args:
      fitted: The fitted pipeline model.
      calibrated: Unused; the Spark tier has no calibration dimension.
      frame: The preprocessed Spark scoring frame.
      metadata: The schema contract.
      data_type: The slice label, used only for the run log.

    Returns:
      A two-tuple of the scored frame and itself, mirroring the pandas trainer's
      two-projection contract.
    """
    del calibrated
    scored = fitted.transform(frame).withColumn(
      constants.ORIG_SCORE_VALUE,
      _probability_column(fitted, self.prediction_col),
    )
    LOGGER.info('Scored a Spark frame', extra={'slice': data_type})
    return scored, scored


class TransformerTraining:
  """Train a transformer over behavioural sequences.

  Opt-in: fully implemented, referenced by no registered pipeline, and gated on
  the optional deep-learning extra. The model checkpoint is a *complete*
  artefact rather than a bare state dict, and three properties of it are
  load-bearing:

  * it carries the architecture and the data-derived vocabulary sizes, not just
    the weights. Vocabulary sizes come from the training data, so a state dict
    alone cannot be reconstructed in a later process, and the tier would not
    round-trip through object storage the way every other model artefact must;
  * prediction frames are concatenated rather than appended, because
    ``DataFrame.append`` is removed in the pinned pandas 2.x;
  * the Lightning 2.x epoch hooks are used, because the 1.x hook names are
    removed in the pinned Lightning 2.x.

  The tier remains opt-in: ``torch`` and ``pytorch-lightning`` are an optional
  dependency and are imported lazily, so the framework runs without them.
  """

  def __init__(self, config: dict[str, Any] | None = None) -> None:
    """Build the trainer from a configuration block.

    Args:
      config: The transformer configuration. Torch and Lightning are imported
        here, so constructing this trainer is the point at which the optional
        dependency becomes required.
    """
    self.config = dict(config or {})
    self.max_value_dict: dict[str, int] = {}
    self.model: Any = None

  def train_and_score(self, metadata: Metadata) -> TransformerTraining:
    """Train over a sequence parquet directory and write predictions.

    Args:
      metadata: The schema contract, whose sequence and identifier columns and
        intermediate path drive the training set.

    Returns:
      ``self``, mirroring the other trainers' return arity at the node layer.
    """
    from forecasting_ml_framework.modeling.transformer import (  # noqa: PLC0415
      CustomWriter,
      PositionalEncoding,
      SequenceModel,
      TransformerRunner,
    )

    params = self.config
    root = params.get('intermediate_file_path') or metadata.intermediate_file_path
    label_col = metadata.target_col
    sequence_cols = list(metadata.seq_cols)
    key_cols = list(metadata.id_cols)

    runner = TransformerRunner(
      model_class=SequenceModel,
      writer_class=CustomWriter,
      positional_encoding_class=PositionalEncoding,
      params=params,
    )
    runner.run(
      train_dir=f'{root}/train',
      test_dir=f'{root}/test',
      output_dir=f'{root}/predictions',
      label_col=label_col,
      key_cols=key_cols,
      sequence_cols=sequence_cols,
      model_save_path=f'{root}/transformer.pt',
    )
    self.model = runner.model
    self.max_value_dict = runner.max_value_dict
    return self


def _probability_column(fitted: Any, prediction_col: str) -> Any:
  """Build an expression extracting the positive-class probability from a Spark
  prediction vector.

  The Spark ML estimators name their probability column after the prediction column
  they were given, so the name is derived from ``prediction_col`` rather than read
  from the fitted pipeline. Reading it off the fitted model instead would require
  the pipeline to have been fitted, which defeats the point of an expression that
  is built *before* the write.

  Args:
    fitted: Unused. Present so the helper can be swapped for a fitted-model reader
      without changing any call site.
    prediction_col: The prediction column name.

  Returns:
    A Spark column expression.
  """
  from pyspark.ml.functions import vector_to_array  # noqa: PLC0415
  from pyspark.sql import functions as sf  # noqa: PLC0415

  probability_col = getattr(fitted, 'probabilityCol', None) or 'probability'
  return sf.element_at(vector_to_array(sf.col(probability_col)), 2).cast('double')


def _takes_class_weights(estimator_class: type) -> bool:
  """Report whether an estimator takes a per-class weight vector.

  Detection is by the class's method-resolution order rather than by comparing
  ``__name__`` to a literal, so a subclass is handled the same way as its base —
  which a's string comparison did not do.

  Args:
    estimator_class: The estimator class.

  Returns:
    ``True`` when the estimator expects ``class_weights``.
  """
  return any(base.__name__ in _CLASS_WEIGHT_ESTIMATORS for base in estimator_class.__mro__)


def _safe_rate(numerator: float, denominator: float) -> float:
  """Compute a rate, recovering precision when the rate rounds to zero.

  A rare-event rate of 0.00043 rounds to ``0.0`` at two decimal places, which
  would silently disable calibration through the degenerate branch. The
  precision guard is a targeted defence against exactly that, and it is applied
  at both rate-computation sites.

  Args:
    numerator: The numerator.
    denominator: The denominator.

  Returns:
    The rate, rounded to two places, or to four when the rate is below 0.1.
  """
  if not denominator:
    return 0.0
  rate = float(np.round(numerator / denominator, 2))
  if rate < 0.1:
    rate = float(np.round(numerator / denominator, 4))
  return rate
