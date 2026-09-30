#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reflective resolution of dotted paths.

The reflective resolver is the single mechanism behind the framework's entire
extensibility surface. It is used for estimators, hyperparameter searchers,
calibrators, Optuna samplers and pruners, Spark ML transformer stages,
persisted-dataset implementations and preprocessing routines.

Two properties are preserved from the design and are relied upon
everywhere:

1. Configuration carries a *string*; the constructor accepts either a string or
   an already-resolved object. The ``isinstance`` guard is a testable seam.
2. Resolution is deferred to first use, so the framework does not need XGBoost
   installed to run a logistic-regression model.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import types
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy import random as np_random
from scipy import stats as scipy_stats

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import (
  ConfigurationError,
  InvalidSearchSpaceError,
  RoutineNotFoundError,
)

#: Module prefixes that mark a configuration string as a distribution expression
#: rather than a plain value, and so route it to the registry below.
#:
#: Calling bare ``eval()`` on a configuration string is a code-execution surface,
#: because configuration is data and data must never be able to name an arbitrary
#: callable. The registry below is the replacement: a search space may only
#: construct a distribution this module declares, by name, with literal arguments.
#: Nothing in this module evaluates a configuration-supplied expression; the
#: arguments are read with ``ast.literal_eval``, which can only produce literals.
#:
#: The prefixes are matched before the registry lookup rather than derived from it,
#: so they are a *pre-filter* and not a second source of truth: a name that passes
#: this test but is absent from the registry is still refused, with a message naming
#: the supported set. Adding a distribution therefore needs an entry in
#: :data:`DISTRIBUTIONS` only, and the two lists cannot disagree about what is
#: constructible.
_DISTRIBUTION_PREFIXES = ('scipy.', 'numpy.', 'np.')

#: Top-level modules a configuration string is permitted to name.
#:
#: The general resolver is the framework's single extension mechanism: a model
#: handle, a tuning handle, a calibration handle, an evaluator handle, a Spark ML
#: stage, a binner, and every routine's ``handle`` are all dotted strings read
#: from ``parameters.yml`` and resolved here. That is the design (P1 -- behaviour
#: is configuration), and it is also, without a constraint, arbitrary code
#: execution driven by a configuration value.
#:
#: The documented compensating control is an allow-list, and it is a real control
#: only if the list is enforced here rather than trusted to code review. An
#: unbounded ``getattr`` walk from any top-level package resolves
#: ``os.system``, ``builtins.eval``, ``subprocess.Popen`` and ``pickle.loads`` --
#: all of which this module's own distribution registry exists to avoid. Adding a
#: resolvable library therefore requires an entry here, in review, rather than
#: becoming reachable by a configuration edit.
#:
#: The rule is *top-level package* granularity, not a full path list, so a new
#: class in an already-allowed library is reachable without touching this map --
#: which is what keeps the allow-list maintainable. It is also the right
#: granularity for the threat: the risk is naming a package that hands over the
#: interpreter (``os``, ``builtins``, ``subprocess``, ``importlib``), not naming
#: an estimator class inside a package whose whole purpose is to provide them.
_ALLOWED_RESOLUTION_ROOTS = frozenset(
  {
    # --- the framework itself ------------------------------------------------
    'forecasting_ml_framework',
    # --- orchestration and its datasets --------------------------------------
    'kedro',
    'kedro_datasets',
    # --- distributed compute --------------------------------------------------
    'pyspark',
    # --- data frames and numerics ---------------------------------------------
    'numpy',
    'np',
    'scipy',
    'pandas',
    # --- machine learning -----------------------------------------------------
    'sklearn',
    'xgboost',
    'lightgbm',
    'catboost',
    'optuna',
    'optbinning',
    'betacal',
    'mlflow',
    'torch',
    'pytorch_lightning',
    'torchmetrics',
    'skopt',
    'hyperopt',
  }
)

#: Every distribution a search space may name, keyed by its dotted path.
#:
#: This mapping is the framework's whole search-space vocabulary, and it is
#: deliberately a *closed* set. The alternative -- parsing a configuration-supplied
#: expression and evaluating the resulting tree -- makes arbitrary code execution a
#: property of the interpreter being reachable from configuration, and no amount of
#: guarding an interpreter compares to not having one. Declaring the constructors
#: here makes the supported set readable in one place and removes the evaluation
#: surface entirely.
#:
#: A caller needing a distribution outside this set adds it here, in review, rather
#: than discovering at run time that it is unreachable.
#:
#: Each lambda mirrors the wrapped constructor's own signature, **including its
#: defaults**, so omitting an optional argument behaves as it would have under a
#: direct call -- ``scipy.stats.uniform(0.01)`` means ``loc=0.01, scale=1.0``.
#: The lambdas exist only to keep the registry a mapping of *name to callable*; they
#: deliberately add no validation of their own, so the wrapped library remains the
#: single authority on what a valid distribution is.
DISTRIBUTIONS: dict[str, Callable[..., Any]] = {
  # --- scipy: continuous distributions -------------------------------------
  'scipy.stats.uniform': lambda loc=0.0, scale=1.0, **kw: scipy_stats.uniform(loc, scale, **kw),
  'scipy.stats.loguniform': lambda a=0.0, b=1.0, **kw: scipy_stats.loguniform(a, b, **kw),
  'scipy.stats.expon': lambda scale=1.0, **kw: scipy_stats.expon(scale, **kw),
  'scipy.stats.exponnorm': lambda k=1.0, loc=0.0, scale=1.0, **kw: scipy_stats.exponnorm(
    k, loc, scale, **kw
  ),
  'scipy.stats.gamma': lambda a=1.0, loc=0.0, scale=1.0, **kw: scipy_stats.gamma(
    a, loc, scale, **kw
  ),
  'scipy.stats.beta': lambda a=1.0, b=1.0, loc=0.0, scale=1.0, **kw: scipy_stats.beta(
    a, b, loc, scale, **kw
  ),
  'scipy.stats.triang': lambda c=None, loc=0.0, scale=1.0, **kw: scipy_stats.triang(
    c, loc, scale, **kw
  ),
  'scipy.stats.norm': lambda loc=0.0, scale=1.0, **kw: scipy_stats.norm(loc, scale, **kw),
  'scipy.stats.lognorm': lambda s=1.0, loc=0.0, scale=1.0, **kw: scipy_stats.lognorm(
    s, loc, scale, **kw
  ),
  'scipy.stats.randint': lambda low, high=None, **kw: scipy_stats.randint(low, high, **kw),
  # --- numpy: integer and categorical draws -------------------------------
  'numpy.random.randint': lambda low, high=None, **kw: np_random.randint(low, high, **kw),
  'numpy.random.uniform': lambda low=0.0, high=1.0, **kw: np_random.uniform(low, high, **kw),
  'numpy.random.normal': lambda loc=0.0, scale=1.0, **kw: np_random.normal(loc, scale, **kw),
  'numpy.random.exponential': lambda scale=1.0, **kw: np_random.exponential(scale, **kw),
  'numpy.random.choice': lambda a, **kw: np_random.choice(a, **kw),
  'numpy.random.loguniform': lambda low=0.0, high=1.0, **kw: np.exp(
    np_random.uniform(np.log(low), np.log(high), **kw)
  ),
}

# Aliases so a configuration may use the short spellings the libraries themselves
# accept. They resolve to the same constructors rather than being a second
# vocabulary, so there is still exactly one place that decides what is callable.
DISTRIBUTIONS.update(
  {
    'scipy.uniform': DISTRIBUTIONS['scipy.stats.uniform'],
    'scipy.loguniform': DISTRIBUTIONS['scipy.stats.loguniform'],
    'scipy.randint': DISTRIBUTIONS['scipy.stats.randint'],
    'np.random.randint': DISTRIBUTIONS['numpy.random.randint'],
    'np.random.uniform': DISTRIBUTIONS['numpy.random.uniform'],
    'np.random.normal': DISTRIBUTIONS['numpy.random.normal'],
    'np.random.choice': DISTRIBUTIONS['numpy.random.choice'],
    'np.random.exponential': DISTRIBUTIONS['numpy.random.exponential'],
    'np.random.loguniform': DISTRIBUTIONS['numpy.random.loguniform'],
  }
)


def resolve(path: str | Callable[..., Any]) -> Any:
  """Import and return the object named by a dotted path.

  Args:
    path: A dotted path such as ``'xgboost.XGBClassifier'``. Callables are
      returned unchanged, which makes the function safe to call unconditionally.

  Returns:
    The resolved attribute.

  Raises:
    ImportError: If the module cannot be imported.
    ConfigurationError: If the path names a package outside
      :data:`_ALLOWED_RESOLUTION_ROOTS`.
    AttributeError: If the module does not expose the requested attribute.
  """
  if not isinstance(path, str):
    return path
  parts = path.split('.')
  if len(parts) < 2:
    raise ImportError(
      f'Reflective path must be fully qualified, e.g. "package.module.Name"; got {path!r}',
      path=path,
    )
  _assert_resolution_allowed(parts[0], path)
  # Import the full module path first, so every parent link in the chain is
  # bound, then start the walk from the TOP-LEVEL package. Starting from the
  # submodule instead would re-walk the module's own name -- `getattr` on
  # `sklearn.tree` looking for `tree` -- which is the failure this avoids.
  importlib.import_module('.'.join(parts[:-1]))
  resolved: Any = importlib.import_module(parts[0])
  for index, component in enumerate(parts[1:], start=1):
    resolved = _walk(resolved, component, '.'.join(parts[: index + 1]))
  return resolved


def _assert_resolution_allowed(root: str, path: str) -> None:
  """Refuse to resolve a dotted path outside the allow-list.

  Args:
    root: The path's top-level package.
    path: The full dotted path, for the error message.

  Raises:
    ConfigurationError: If the package is not allow-listed. The message names both
      the offending path and the permitted set, so the fix for an operator is
      visible in the failure rather than requiring them to find this module.
  """
  if root in _ALLOWED_RESOLUTION_ROOTS:
    return
  raise ConfigurationError(
    f'Reflective path {path!r} names the package {root!r}, which is not on the resolution '
    f'allow-list. A configuration document is data, and data must never be able to name an '
    f'arbitrary callable: an unbounded walk from the top-level package resolves '
    f'os.system, builtins.eval, subprocess.Popen and pickle.loads. Allowed packages: '
    f'{", ".join(sorted(_ALLOWED_RESOLUTION_ROOTS))}. To use another library, add its top-level '
    f'package to _ALLOWED_RESOLUTION_ROOTS in review.',
    path=path,
    root=root,
  )


def _walk(current: Any, component: str, partial: str) -> Any:
  """Resolve one component of a dotted path.

  Two failure modes are handled here, and both are real:

  * **An unbound submodule.** ``importlib.import_module('sklearn.tree')`` puts the
    submodule in ``sys.modules`` but does not reliably bind it as an attribute of
    its parent, so a bare ``getattr`` on the parent raises. Re-importing the full
    partial path resolves the attribute as a side effect.
  * **A submodule shadowing an attribute.** When a package exposes both a
    submodule and a class of the same name, the attribute is preferred, because a
    dotted path in configuration is naming a class or function.

  Args:
    current: The object resolved so far.
    component: The next path component.
    partial: The path resolved so far, for error messages and re-import.

  Returns:
    The resolved attribute.

  Raises:
    AttributeError: If the component cannot be resolved either way.
  """
  try:
    found = getattr(current, component)
  except AttributeError:
    found = None
  else:
    if not isinstance(found, types.ModuleType) or not hasattr(found, '__name__'):
      return found

  if isinstance(current, types.ModuleType):
    try:
      reimported = importlib.import_module(partial)
    except ImportError:
      reimported = None
    if reimported is not None:
      found = getattr(reimported, component, found)
  if found is None:
    raise AttributeError(
      f'Could not resolve {partial!r}: {getattr(current, "__name__", current)!r} has no attribute '
      f'{component!r}. Check the dotted path in the configuration document.'
    )
  return found


def try_resolve(path: str) -> tuple[Any, Exception | None]:
  """Attempt to resolve a dotted path without raising.

  A rejected path is a *resolution failure*, not a crash, so the allow-list
  rejection is caught here alongside the ordinary import and attribute errors.
  The caller decides what to do with it: :func:`resolve_routine` folds it into
  its "searched these paths" diagnostic, which is where an operator looking at a
  mistyped ``modules:`` entry needs to see it.

  Args:
    path: A dotted path.

  Returns:
    A two-tuple of the resolved object and the exception that occurred, where
    the object is ``None`` on failure.
  """
  try:
    return resolve(path), None
  except (ImportError, AttributeError, ValueError, ConfigurationError) as error:
    return None, error


def resolve_routine(short_name: str, extra_modules: list[str] | None = None) -> type:
  """Resolve a preprocessing routine class by its short configuration name.

  A single hard-coded module prefix would close the extension point to new
  modules. The prefix is instead the first candidate in a fallback chain, so a
  routine may live in any importable module supplied in configuration.

  Args:
    short_name: The routine class name as it appears in ``data_prep_params``.
    extra_modules: Additional dotted module paths to search, in order.

  Returns:
    The routine class.

  Raises:
    RoutineNotFoundError: If the routine cannot be resolved from any candidate
      module. The error carries the routine name and every path searched, so the
      message a caller sees is the same diagnostic a test can assert on.
  """
  candidates = [f'{constants.ROUTINES_MODULE}.{short_name}']
  candidates.extend(f'{module}.{short_name}' for module in extra_modules or [])

  errors: list[str] = []
  for path in candidates:
    resolved, error = try_resolve(path)
    if error is None:
      return resolved
    errors.append(f'  - {path}: {error}')

  raise RoutineNotFoundError(
    f'Preprocessing routine {short_name!r} could not be resolved. Searched:\n' + '\n'.join(errors)
    + '\nAdd the defining module to the routine block\'s "modules" key, or check the spelling.'
    + f'\nNote that the framework module prefix is "{constants.ROUTINES_MODULE}".',
    routine=short_name,
    candidates=candidates,
  )


def looks_like_distribution(value: object) -> bool:
  """Report whether a configuration value must be evaluated as an expression.

  Args:
    value: A raw configuration value.

  Returns:
    ``True`` when the value is a string that looks like a distribution
    expression, ``False`` otherwise.
  """
  return isinstance(value, str) and value.startswith(_DISTRIBUTION_PREFIXES)


def evaluate_distribution(expression: str) -> Any:
  """Construct a search-space distribution by resolving its name in a registry.

  The expression is *parsed*, never executed. The callee's dotted name is looked up
  in :data:`DISTRIBUTIONS` and called with literal arguments, so the only code that
  can run is the constructor this module names. That is a categorically stronger
  guarantee than the previous arrangement, which parsed the expression, checked
  that it was an attribute call with literal arguments, and then handed the parsed
  tree to ``eval``. The guards were sound, but the surface they defended was still
  an interpreter, and a guard on an interpreter is one edit away from a hole.

  Two properties follow from the registry that did not hold before:

  * **The supported set is enumerable.** :data:`DISTRIBUTIONS` is the complete list
    of what may appear in a search space, so a reader can discover it by reading one
    constant rather than by inferring a policy from a parser.
  * **A typo fails as a typo.** An unregistered name raises
    :class:`InvalidSearchSpaceError` naming the supported set, instead of raising
    ``AttributeError`` from inside a namespace built for this one call.

  Args:
    expression: A distribution expression such as
      ``'scipy.stats.uniform(0.01, 0.1)'``.

  Returns:
    The constructed distribution object.

  Raises:
    InvalidSearchSpaceError: If the expression is not a single registered call with
      literal arguments.
  """
  try:
    tree = ast.parse(expression, mode='eval')
  except SyntaxError as error:
    raise InvalidSearchSpaceError(
      f'Search-space expression is not parseable: {expression!r}', expression=expression
    ) from error

  node = tree.body
  if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
    raise InvalidSearchSpaceError(
      'Search-space expression must be a single registered call such as '
      f'"scipy.stats.uniform(0.01, 0.1)"; got {expression!r}',
      expression=expression,
    )

  dotted = _dotted_name(node.func)
  if dotted is None:
    raise InvalidSearchSpaceError(
      f'Search-space callee must be a dotted name; got {expression!r}', expression=expression
    )

  # The arguments must be literals so that no argument position can carry an
  # expression of its own. The check covers positional *and* keyword arguments, and
  # rejects ``**kwargs`` (``keyword.arg is None``), because a comprehension or an
  # attribute chain passed as a keyword value would otherwise be evaluated during
  # the call.
  #
  # The test is `_is_literal`, not `isinstance(arg, ast.Constant)`. A bare
  # `Constant` test rejects a list or tuple of literals, and those are how a
  # categorical search space is written: `numpy.random.choice([1, 2, 3])` and
  # `numpy.random.uniform(0, 1, size=(3,))` are both legitimate, and both were
  # refused. `_is_literal` was written to admit them and then never called, so the
  # capability its own docstring describes was not actually reachable.
  if any(not _is_literal(arg) for arg in node.args) or any(
    keyword.arg is None or not _is_literal(keyword.value) for keyword in node.keywords
  ):
    raise InvalidSearchSpaceError(
      'Search-space arguments must be literals so that a configuration document '
      f'cannot execute arbitrary code; got {expression!r}',
      expression=expression,
    )

  constructor = DISTRIBUTIONS.get(dotted)
  if constructor is None:
    raise InvalidSearchSpaceError(
      f'{dotted!r} is not a registered distribution. Supported distributions: '
      f'{sorted(DISTRIBUTIONS)}',
      expression=expression,
      supported=sorted(DISTRIBUTIONS),
    )

  # ``**kwargs`` was rejected above, so the keyword channel carries named literals
  # only and can be passed straight through.
  arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
  try:
    return constructor(*(ast.literal_eval(arg) for arg in node.args), **arguments)
  except InvalidSearchSpaceError:
    raise
  except Exception as error:  # noqa: BLE001 - every failure is reported as a search-space error
    raise InvalidSearchSpaceError(
      f'Could not construct {dotted!r} from {expression!r}: {error}', expression=expression
    ) from error


def _is_literal(node: ast.AST) -> bool:
  """Report whether a node is a literal, or a literal container of literals.

  A configuration value has to be able to express a *list of candidates* -- that is
  what a categorical search space is -- so a list or tuple of literals is accepted
  alongside a bare literal. Nesting is not: a list containing a comprehension is a
  comprehension, and ``ast.literal_eval`` would refuse it, but accepting it here
  would move the failure rather than remove it.

  Args:
    node: The argument node.

  Returns:
    Whether the node is a literal.
  """
  if isinstance(node, ast.Constant):
    return True
  if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
    return all(_is_literal(element) for element in node.elts)
  return False


def _dotted_name(node: ast.Attribute) -> str | None:
  """Return the dotted name of an attribute chain, or ``None`` if not pure.

  Args:
    node: The AST node to flatten.

  Returns:
    The dotted name, or ``None`` when the chain contains a call or subscript.
  """
  parts: list[str] = []
  current: ast.AST = node
  while isinstance(current, ast.Attribute):
    parts.append(current.attr)
    current = current.value
  if not isinstance(current, ast.Name):
    return None
  parts.append(current.id)
  return '.'.join(reversed(parts))


def coerce_search_space(param_grid: dict[str, Any] | None) -> dict[str, Any]:
  """Coerce distribution strings inside a search space to real objects.

  Args:
    param_grid: A mapping of hyper-parameter name to candidate value(s). Values
      that look like distribution expressions are replaced by the constructed
      object; a one-element list containing such a string is unwrapped so that
      ``'scipy.stats.uniform(...)'`` and ``['scipy.stats.uniform(...)']`` behave
      identically.

  Returns:
    A new mapping with expressions evaluated. Non-dict input is returned
    unchanged.
  """
  if not isinstance(param_grid, dict):
    return param_grid
  return {key: _coerce_search_value(value) for key, value in param_grid.items()}


def _coerce_search_value(value: Any) -> Any:
  """Coerce a single search-space value.

  Args:
    value: A raw configuration value.

  Returns:
    The coerced value.
  """
  if looks_like_distribution(value):
    return evaluate_distribution(value)  # type: ignore[arg-type]
  if isinstance(value, list) and len(value) == 1 and looks_like_distribution(value[0]):
    return evaluate_distribution(value[0])
  return value


def assert_no_builtin_namespace() -> None:
  """Assert that the search-space registry cannot name a Python builtin.

  The registry replaced an interpreter, so the question this asked changed. It is
  no longer "does the evaluation namespace expose a dangerous builtin" -- there is
  no namespace and no evaluation -- but "can a configuration value name arbitrary
  code", which is now answered by the registry being a closed set. A builtin
  appearing in it would reintroduce exactly the surface the registry removed, so
  the assertion is worth keeping, retargeted at the registry.

  It exists purely so a future refactor cannot silently reintroduce an unrestricted
  ``eval`` or extend the registry with something that should not be callable.

  Raises:
    InvalidSearchSpaceError: If the registry names a builtin or a dunder.
  """
  offenders = sorted(
    key
    for key, constructor in DISTRIBUTIONS.items()
    if getattr(constructor, '__name__', '') in dir(builtins) or '__' in key.split('.')[-1]
  )
  if offenders:  # pragma: no cover - the registry is static and clean
    raise InvalidSearchSpaceError(
      f'The search-space registry must not name builtins or dunders: {offenders}',
      offenders=offenders,
    )
