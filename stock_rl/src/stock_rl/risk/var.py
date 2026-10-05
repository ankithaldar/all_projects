#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Parametric and historical VaR and CVaR, in the standard library only.

**VaR is not on the NSE pre-trade list.** The research corpus is
explicit: NSE Detailed Operational Modalities para 11.1 lists sixteen
pre-trade RMS checks and VaR is not among them. The sixteen are in
:mod:`stock_rl.risk.checks`, and the fourteen that are implemented there
are the compliance surface. This module is a risk-management nicety used
for sizing and reporting, and saying so here matters more than having the
function, because a control that nobody believes is a compliance
requirement gets waived the first time it blocks a trade.

Two estimators, because they fail in opposite directions:

  * **historical** makes no distributional assumption at all. It reads
    the quantile off the sample, so a fat tail present in the data is
    priced correctly and a regime change is only picked up once it has
    happened. Its weakness is sample size: a 95 percent VaR from 60 daily
    observations is a statement about the worst three observations you
    have seen.
  * **parametric** (variance-covariance) assumes joint normality, reads
    the covariance matrix, and therefore *extrapolates* the tails. That
    is precisely its weakness. Real returns are fat-tailed and
    asymmetric; the excess kurtosis of Indian single-stock daily returns
    is not near zero, and a Gaussian tail estimate at 99 percent is
    understated by a wide margin.

That gap is not an academic complaint. It is why the Deflated Sharpe's
denominator carries the Mertens correction ``(kurtosis - 1) / 4 * SR^2``:
the correction exists *because* the variance of an estimated Sharpe is
larger under real kurtosis than a normal model predicts, so any risk
statistic built on a Gaussian assumption is optimistic in the same
direction. If the Mertens term is load-bearing in this project's own
Sharpe arithmetic, a Gaussian 99 percent VaR is not a conservative
estimate of anything.

**Sign convention.** Every function returns a positive number for a
loss and a negative number for a gain. A VaR of ``0.02`` means "the
worst 5 percent outcome lost 2 percent"; a negative VaR means the 95th
percentile outcome was a gain, which is a real answer and not an error.

**Zero volatility never produces NaN.** A degenerate input is the case
that silently poisons a downstream risk report, because NaN compares
false against every limit and therefore passes every check. Neither
parametric function can return it: the volatility is validated before the
division, and a zero volatility is divided only after being caught.

The two of them answer a zero-volatility input differently, and the
difference is deliberate:

- :func:`parametric_var` returns ``-mean``. A series with no dispersion
  is deterministic, so a mean of ``+0.01`` means every period returned
  exactly ``+0.01`` and the worst 5 percent outcome was a gain. Reporting
  that as ``-0.01`` is informative; flooring it to ``0.0`` would claim
  the worst outcome broke even, which is false.
- :func:`portfolio_var` returns ``0.0``. Its input is a covariance
  matrix, so a zero variance means the weighted exposure cancels exactly.
  There is no per-period mean to report against it, and a negative VaR
  from a fully hedged book would be read as free money.
'''

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist, fmean

__all__ = [
  'VarEstimate',
  'covariance_matrix',
  'historical_cvar',
  'historical_var',
  'min_confidence',
  'normal_quantile',
  'parametric_cvar',
  'parametric_var',
  'portfolio_var',
  'portfolio_variance',
  'sample_volatility',
  'var_from_returns',
  'z_score',
]

#: Lowest confidence a risk statistic may be evaluated at. A VaR at 50
#: percent is the median and calling it a risk measure is meaningless;
#: below this the function refuses rather than returning a number that
#: reads as safety.
min_confidence = 0.5

#: Highest confidence, inclusive. Kept below 1.0 because a quantile at
#: exactly 1.0 is the sample maximum, which makes the statistic a
#: one-observation estimate wearing a confidence level as a hat.
max_confidence = 0.999

#: Relative slack allowed on a Cholesky pivot before it is read as
#: negative. Set well above double-precision rounding on a scale-1
#: matrix and well below the negative pivots a genuinely indefinite
#: matrix produces. Without it, a singular sample covariance -- two
#: identical return series give an exactly rank-1 matrix -- would be
#: refused for a pivot that is zero rather than negative.
pivot_tolerance = 1e-12

#: Normal distribution reused for the quantile function. ``NormalDist``
#: is the standard library's rational approximation, accurate to about
#: 1e-15, and it means this module does not carry its own inverse normal
#: function.
_normal = NormalDist()


def z_score(confidence: float) -> float:
  '''Return the standard normal quantile for a confidence level.

  Args:
    confidence: Tail confidence in ``[min_confidence, max_confidence]``.

  Returns:
    The quantile, so 0.95 gives about 1.6449.

  Raises:
    ValueError: If the confidence is outside the supported range. A
      confidence of 0 or 1 would make the quantile infinite or undefined,
      and a NaN that flows into a limit comparison passes it.
  '''
  _require_confidence(confidence)
  return _normal.inv_cdf(confidence)


def normal_quantile(confidence: float) -> float:
  '''Alias of :func:`z_score`, named for what it returns.

  Args:
    confidence: Tail confidence.

  Returns:
    The standard normal quantile.
  '''
  return z_score(confidence)


def _require_confidence(confidence: float) -> None:
  '''Validate a confidence level.

  Args:
    confidence: The confidence to check.

  Raises:
    ValueError: If it is outside
      ``[min_confidence, max_confidence]`` or is not a real number.
  '''
  if not isinstance(confidence, (int, float)) or \
      isinstance(confidence, bool):
    raise ValueError(
      f'confidence must be a real number, got {confidence!r}')
  if not min_confidence <= confidence <= max_confidence:
    raise ValueError(
      f'confidence must be in [{min_confidence}, {max_confidence}], got '
      f'{confidence}')


def _quantile(values: Sequence[float], level: float) -> float:
  '''Return a sample quantile by linear interpolation.

  ``statistics.quantiles`` is not used because it refuses a two-point
  sample and defaults to an exclusive method whose behaviour at the tail
  is easy to misremember. A risk statistic whose tail estimator is a
  method flag is a risk statistic nobody can review.

  Args:
    values: Sample, in any order.
    level: Quantile level in ``[0, 1]``.

  Returns:
    The interpolated quantile.
  '''
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  position = level * (len(ordered) - 1)
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[int(position)]
  weight = position - lower
  return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def historical_var(returns: Sequence[float],
                   confidence: float = 0.95) -> float:
  '''Return the historical value at risk of a return sample.

  The loss at the ``1 - confidence`` quantile. Non-parametric: nothing
  is assumed about the distribution, which is the whole point, and also
  the whole limitation -- the tail is whatever the sample happened to
  contain.

  Args:
    returns: Simple periodic returns, one per period.
    confidence: Tail confidence.

  Returns:
    Loss as a positive fraction. ``0.0`` for an empty sample, which is a
    documented answer rather than an accident: an account with no
    observed period has no observed loss, and the caller should treat it
    as "no evidence" rather than "no risk".

  Raises:
    ValueError: If the confidence is unsupported.
  '''
  _require_confidence(confidence)
  if not returns:
    return 0.0
  return -_quantile(returns, 1.0 - confidence) + 0.0


def historical_cvar(returns: Sequence[float],
                    confidence: float = 0.95) -> float:
  '''Return the historical conditional value at risk (expected shortfall).

  The mean of the worst ``1 - confidence`` of the sample, so it is always
  at least as large as :func:`historical_var`. That ordering is a
  mathematical property here, not an assertion: the tail mean is at most
  the tail quantile, and negating both preserves the order.

  **Related to but not shared with** :func:`stock_rl.rl.hedge_env.cvar`.
  That function is the same functional form with a floor of 0.85 on the
  confidence, imposed because Francois et al. (2025) found the
  deep-hedging objective degenerating at a low tail. That floor is a
  property of the *hedging objective*, not of the risk measure, and
  importing it here would make a 90 percent VaR report refuse to compute.

  Args:
    returns: Simple periodic returns, one per period.
    confidence: Tail confidence.

  Returns:
    Mean tail loss as a positive fraction, ``0.0`` for an empty sample.

  Raises:
    ValueError: If the confidence is unsupported.
  '''
  _require_confidence(confidence)
  if not returns:
    return 0.0
  tail = 1.0 - confidence
  count = max(1, math.ceil(tail * len(returns)))
  worst = sorted(returns)[:count]
  return -fmean(worst) + 0.0


def parametric_var(mean: float, volatility: float,
                   confidence: float = 0.95) -> float:
  '''Return the variance-covariance value at risk of one return series.

  ``VaR = -(mu - z * sigma)``, positive for a loss. The lower tail is
  ``mu - z * sigma`` and the loss is its negation, so the minus on ``mu``
  sits *inside* the bracket. Getting it outside produces a VaR smaller
  than the CVaR of the same fit, which is the signature of the error and
  is why :meth:`VarEstimate.coherent` exists.

  **The normality assumption is the whole content of this function.**
  It treats the tail as a Gaussian tail, and Gaussian tails are thin,
  so at 99 percent this understates a fat-tailed loss by a wide margin.
  It is implemented because it is the estimator a portfolio risk report
  is expected to quote, not because it is the better one.

  Args:
    mean: Mean periodic return.
    volatility: Periodic standard deviation. Zero is accepted and
      returns ``-mean`` normalised to avoid a negative zero; it does not
      raise, because a zero-volatility series is a real input and the
      answer ``0.0`` is correct, whereas a NaN would pass every
      downstream limit comparison.
    confidence: Tail confidence.

  Returns:
    Loss as a positive fraction.

  Raises:
    ValueError: If the confidence is unsupported, or the volatility is
      negative.
  '''
  _require_confidence(confidence)
  if volatility < 0.0:
    raise ValueError(f'volatility must be >= 0, got {volatility}')
  value = -(mean - z_score(confidence) * volatility)
  return value + 0.0


def parametric_cvar(mean: float, volatility: float,
                    confidence: float = 0.95) -> float:
  '''Return the parametric conditional value at risk of a normal return.

  Under the normal assumption the tail is closed-form, so this is the
  exact CVaR of the fitted model rather than a sample statistic. It is
  therefore only as trustworthy as the normality assumption, and the
  closed form is exactly what makes it look trustworthy.

  Args:
    mean: Mean periodic return.
    volatility: Periodic standard deviation.
    confidence: Tail confidence.

  Returns:
    Mean tail loss as a positive fraction.

  Raises:
    ValueError: If the confidence is unsupported, or the volatility is
      negative.
  '''
  _require_confidence(confidence)
  if volatility < 0.0:
    raise ValueError(f'volatility must be >= 0, got {volatility}')
  tail = z_score(confidence)
  value = -(mean - volatility * _normal.pdf(tail) / (1.0 - confidence))
  return value + 0.0


def sample_volatility(values: Sequence[float]) -> float:
  '''Return the sample standard deviation of a return series.

  Args:
    values: Sample.

  Returns:
    Sample standard deviation with the ``n - 1`` denominator, or ``0.0``
      for fewer than two observations.
  '''
  if len(values) < 2:
    return 0.0
  mean = fmean(values)
  variance = fmean([(value - mean) ** 2 for value in values])
  return math.sqrt(variance * len(values) / (len(values) - 1))


def var_from_returns(returns: Sequence[float],
                     confidence: float = 0.95) -> VarEstimate:
  '''Return both estimators for one return series, side by side.

  Reporting only one is how a book ends up with an undisclosed choice of
  estimator. Zernikov's point applies directly: two books holding the
  same positions and quoting different VaRs will have chosen different
  tails and different means without either knowing.

  Args:
    returns: Simple periodic returns.
    confidence: Tail confidence.

  Returns:
    A :class:`VarEstimate` carrying the historical and parametric
    figures and the gap between them. The gap is the fat-tail evidence:
    a large gap means the Gaussian fit is extrapolating, and the
    parametric number should not be the one acted on.

  Raises:
    ValueError: If the confidence is unsupported.
  '''
  _require_confidence(confidence)
  if not returns:
    return VarEstimate(0.0, 0.0, 0.0, 0.0, 0.0, confidence)
  mean = fmean(returns)
  volatility = sample_volatility(returns)
  return VarEstimate(
    historical_var(returns, confidence),
    historical_cvar(returns, confidence),
    parametric_var(mean, volatility, confidence),
    parametric_cvar(mean, volatility, confidence),
    volatility,
    confidence,
  )


@dataclass(frozen=True, slots=True)
class VarEstimate:
  '''Both estimators over one sample, plus the disagreement between them.

  Attributes:
    historical: Historical VaR, positive for a loss.
    historical_tail: Historical CVaR.
    parametric: Variance-covariance VaR.
    parametric_tail: Variance-covariance CVaR.
    volatility: Sample standard deviation of the sample.
    confidence: Tail confidence both estimators were evaluated at.
  '''

  historical: float
  historical_tail: float
  parametric: float
  parametric_tail: float
  volatility: float
  confidence: float = 0.95

  @property
  def tail_gap(self) -> float:
    '''Return how far the Gaussian tail sits below the empirical one.

    A large positive gap is the fat tail showing up as a number: the
    sample contains losses the Gaussian fit does not consider possible.
    A negative gap means the Gaussian fit is the more conservative of
    the two, which happens when the sample's worst outcomes are mild
    relative to its dispersion.
    '''
    return self.historical_tail - self.parametric_tail

  def coherent(self) -> bool:
    '''Return whether both estimators satisfy ``CVaR >= VaR``.

    Coherence of the tail ordering is a mathematical property of both
    estimators, so a violation means a bug rather than bad luck. It is
    checked rather than assumed because it is cheap and because the
    alternative -- a risk report whose CVaR is smaller than its VaR --
    is the kind of number that gets quoted in a report without anybody
    noticing.

    Returns:
      True when both orderings hold.
    '''
    return (self.historical_tail >= self.historical
            and self.parametric_tail >= self.parametric)


def covariance_matrix(returns: Sequence[Sequence[float]]) -> tuple[
  tuple[float, ...], ...]:
  '''Return the sample covariance matrix of aligned return series.

  Takes a plain nested sequence rather than a matrix type, because there
  is no matrix type in the standard library and taking one would mean a
  runtime dependency. The performance ceiling is the point: a
  30-symbol book over 500 bars is a 30x30 covariance, which is a rounding
  error, and the thing that would actually be fast enough for this
  project is not the linear algebra.

  Args:
    returns: One sequence per asset, each of equal length, aligned by
      period.

  Returns:
    Symmetric covariance matrix as nested tuples.

  Raises:
    ValueError: If there are no series, the series are of unequal
      length, or there is a single observation (a covariance needs two).
  '''
  series = [list(values) for values in returns]
  if not series:
    raise ValueError('at least one return series is required')
  length = len(series[0])
  if length < 2:
    raise ValueError(
      f'a covariance needs at least two periods, got {length}')
  for values in series:
    if len(values) != length:
      raise ValueError(
        f'return series must be aligned; got lengths '
        f'{[len(item) for item in series]}')
  means = [fmean(values) for values in series]
  count = length - 1
  matrix: list[tuple[float, ...]] = []
  for i in range(len(series)):
    row: list[float] = []
    for j in range(len(series)):
      total = sum(
        (series[i][k] - means[i]) * (series[j][k] - means[j])
        for k in range(length))
      row.append(total / count)
    matrix.append(tuple(row))
  return tuple(matrix)


def portfolio_variance(weights: Sequence[float],
                      covariance: Sequence[Sequence[float]]) -> float:
  '''Return portfolio variance from weights and a covariance matrix.

  ``w' Sigma w``, computed as a double sum. With a diagonal-dominant,
  shrunk matrix -- the only kind worth using -- this is well behaved; the
  eigenvector shortcut is a performance trick this project does not need.

  **A negative quadratic form raises. It is not floored to zero.**

  Flooring was how the "never negative" promise was kept, and it is the
  worst possible answer in a variance estimator. The way a shrinkage or
  Ledoit-Wolf step goes wrong in practice is by producing a matrix that
  is not positive semi-definite; for a weight vector that exposes the
  indefiniteness, ``w' Sigma w`` is negative, and ``max(0.0, total)``
  answers that with a portfolio carrying *no risk at all*. No error, no
  warning, and a book that reports zero volatility while the estimator
  feeding it is broken. :func:`var_from_returns` then quotes
  ``tail_gap=0`` and ``coherent()=True``, because those are properties of
  the numbers it was handed.

  Raising names the failure where it can still be traced to the estimator
  that produced it. A caller that genuinely wants a non-negative floor --
  for instance to survive a near-singular matrix whose smallest
  eigenvalue is negative by rounding -- can apply it at the call site,
  where it is visible. A floor applied here is not.

  **Checking the quadratic form alone is not enough**, which is worth
  stating because it looks sufficient. An indefinite matrix has *some*
  weight vector giving a negative variance, not all of them: the test
  matrix ``[[0.03, 0.05], [0.05, -0.03]]`` is indefinite and returns
  ``+0.025`` at ``[0.5, 0.5]``, because that direction happens to align
  with the positive eigenvector. Refusing only negative forms therefore
  lets half of every indefinite matrix straight through, and the half it
  lets through is the half that reports no risk. So the matrix itself is
  checked, via :func:`_is_positive_semidefinite`.

  Args:
    weights: Portfolio weight per asset, aligned with the matrix.
    covariance: Covariance matrix as nested sequences.

  Returns:
    Portfolio variance, never negative and never floored: a covariance
    that is not positive semi-definite, or that produces a negative
    variance here, is refused.

  Raises:
    ValueError: If the shapes disagree, the matrix is not symmetric, the
      matrix is not positive semi-definite, or the quadratic form is
      negative for the given weights.
  '''
  size = len(weights)
  if len(covariance) != size or any(
      len(row) != size for row in covariance):
    raise ValueError(
      f'weights of {size} do not match a {len(covariance)}x'
      f'{len(covariance[0]) if covariance else 0} covariance')
  if not _is_positive_semidefinite(covariance):
    raise ValueError(
      f'the {size}x{size} covariance matrix is not positive '
      'semi-definite, so it has a direction in which the portfolio '
      'variance is negative. A variance estimator that answers this with '
      'zero reports a book with no risk at all, so the matrix is refused '
      'and named rather than clamped')
  total = 0.0
  for i in range(size):
    for j in range(size):
      total += weights[i] * covariance[i][j] * weights[j]
  if total < 0.0:
    raise ValueError(
      f'portfolio variance is {total:.6g}, which is negative. A '
      'positive semi-definite matrix cannot produce that, so the input '
      'is not a covariance matrix at these weights')
  return total


def _is_positive_semidefinite(matrix: Sequence[Sequence[float]]) -> bool:
  '''Return whether a square matrix is positive semi-definite.

  Cholesky without pivoting, which is exact for a symmetric
  positive-semi-definite matrix: every pivot is a Schur complement and is
  therefore non-negative. A pivot that comes out negative means the
  matrix has a negative eigenvalue, and the function returns False at
  that point rather than finishing a factorisation that does not exist.

  A pivot within :data:`pivot_tolerance` of zero is a direction the matrix
  is flat in, which is legal for a PSD matrix and is what a singular
  sample covariance almost always is -- two assets with identical return
  series give a rank-1 matrix whose second pivot is exactly zero. It is
  clamped to zero and the factorisation continues past it.

  PONYTAIL: this is O(n cubed) per call and recomputed on every
  :func:`portfolio_variance`, which is fine for a 30-name book sized a
  handful of times a session and wrong for anything per-bar. It is also a
  ~30 line reimplementation of LAPACK's ``dpstrf``, because the project
  takes no runtime dependencies. Ceiling: it allocates two n-by-n
  working copies, it is unpivoted, and it verifies symmetry rather than
  repairing an asymmetric input. Upgrade path: factor once in the
  estimator that owns the matrix (:func:`covariance_matrix`, or a
  shrinkage step upstream) and pass the factorisation in, or adopt numpy
  when the no-dependency rule is retired.

  Args:
    matrix: Square matrix as nested sequences.

  Returns:
    True when the matrix is symmetric and every pivot is non-negative
    within tolerance, False otherwise. An empty matrix is trivially
    positive semi-definite.
  '''
  size = len(matrix)
  if size == 0:
    return True
  original = [[float(value) for value in row] for row in matrix]
  # L, the Cholesky factor under construction. Kept apart from
  # ``original`` rather than written in place: the factorisation reads
  # the *original* Schur complements, so overwriting the input with the
  # factor is the classic way to get a silently wrong pivot sequence.
  factor = [[0.0] * size for _ in range(size)]
  scale = max((abs(original[i][j])
               for i in range(size) for j in range(size)), default=0.0)
  tolerance = pivot_tolerance * max(1.0, scale)
  for i in range(size):
    for j in range(i + 1, size):
      if abs(original[i][j] - original[j][i]) > tolerance:
        return False
  for k in range(size):
    # The pivot is the Schur complement: the original diagonal minus
    # everything the already-factored columns of this row have claimed.
    # ``factor[k][j]`` holds L[k][j] for j < k, kept separately from
    # ``working`` so the two are never confused.
    pivot = original[k][k] - sum(
      factor[k][j] ** 2 for j in range(k))
    if pivot < -tolerance:
      return False
    if pivot <= tolerance:
      # A flat direction, which a singular PSD matrix always has. Zero
      # the row of L and continue against the projection; the remaining
      # pivots are then computed for the subspace orthogonal to it.
      for i in range(k + 1, size):
        factor[i][k] = 0.0
      factor[k][k] = 0.0
      continue
    root = math.sqrt(pivot)
    factor[k][k] = root
    for i in range(k + 1, size):
      offset = original[i][k] - sum(
        factor[i][j] * factor[k][j] for j in range(k))
      factor[i][k] = offset / root
  return True


def portfolio_var(
  weights: Sequence[float],
  covariance: Sequence[Sequence[float]],
  mean: Sequence[float] = (),
  confidence: float = 0.95,
) -> float:
  '''Return the parametric VaR of a weighted portfolio.

  The variance-covariance method proper: the portfolio variance is built
  from the full covariance matrix, so correlations are honoured, and the
  marginal variances are not simply added. That is the difference between
  this and summing per-symbol VaRs, and the difference is the entire
  diversification benefit -- adding VaRs is a strictly conservative
  answer that reports no benefit from holding uncorrelated assets.

  Args:
    weights: Portfolio weight per asset.
    covariance: Covariance matrix as nested sequences.
    mean: Mean periodic return per asset, one per weight. Defaults to
      empty, which is zero drift, and is the conservative choice:
      assuming a positive mean would shrink every VaR by a number the
      sample may not support.
    confidence: Tail confidence.

  Returns:
    Loss as a positive fraction, ``0.0`` on a zero-variance portfolio.

  Raises:
    ValueError: If the shapes disagree, or the confidence is
      unsupported.
  '''
  _require_confidence(confidence)
  if mean and len(mean) != len(weights):
    raise ValueError(
      f'mean has {len(mean)} entries for {len(weights)} weights: the '
      'drift vector is read one per asset, so a short vector raises an '
      'IndexError from the summation several lines below and a long one '
      'is silently truncated. Both hide a shape disagreement that this '
      'call site is the right place to name')
  variance = portfolio_variance(weights, covariance)
  if variance <= 0.0:
    return 0.0
  drift = 0.0 if not mean else sum(
    weights[i] * float(mean[i]) for i in range(len(weights)))
  return parametric_var(drift, math.sqrt(variance), confidence)
