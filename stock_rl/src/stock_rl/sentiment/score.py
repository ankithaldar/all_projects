#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Sentiment as an injected input, never as a model call.

The design doc builds a Sentiment Engine: FinBERT plus Llama 3.1 8B on a
GPU, Qdrant embeddings, a 5-minute cron, and an LLM judge that turns
headline scores into a per-ticker verdict. The research review says
build none of that, and says it in three specific ways:

1. **FinBERT-class sentiment inverted as a standalone signal.**
   Johnsen & Shasharina (Feb 2026) found FinBERT-positive days averaged
   **-0.37%** the next day, spread **-0.03%**. Its defensible role is
   **triage** -- deciding what is worth reading -- not signal generation.
2. **The profitable leg is the short one.** Stambaugh, Yu & Yuan (2012):
   **78%** of sentiment-anomaly profits come from the short leg, which
   is expensive to borrow. Muravyev, Pearson & Pollet (2025) find the
   long-short anomaly return going from +0.14%/month to **-0.01%** after
   borrowing fees. A long-only NSE book cannot collect most of this.
3. **Attention is not information.** Ben-Rephael, Da & Israelsen (2017):
   what matters is whether attention carries information, not how much of
   it there is. NVDA had the second-most coverage in the replication
   dataset and sentiment had **r = -0.03**. Nifty-50 names are precisely
   the large, well-covered stocks where this dilution is worst.

So this module does the only defensible part: it defines the shape of a
sentiment reading, it aggregates readings that are **already computed**
and handed to it, and it enforces the two rules that make the number
mean anything:

* ``available_from`` is when the information became **publicly**
  available, never when it was fetched. A reading stamped with a fetch
  time is a look-ahead leak wearing a timestamp, and it is the single
  most effective way to manufacture a fake Sharpe.
* A single source is not a reading, it is an opinion. The aggregate
  carries a **source-count minimum** and returns ``usable = False``
  below it.

**Disagreement is returned, not averaged away.** The review is explicit
that multi-agent debate's only statistically significant gain came from
an anti-convergence guard, and that three different providers produced
byte-identical portfolios on the same scenario. Identical independent
extractors are the pathology, so cross-source spread is reported as its
own number rather than folded into the mean where nothing can see it.

The anti-convergence guard itself is :func:`converged`, and
:func:`aggregate` calls it on every set of per-source scores it
publishes, so the answer travels on
:attr:`SentimentAggregate.converged` rather than waiting for a caller
to remember. It is reported, not enforced, and the docstring there says
why refusing on it would be wrong.

An LLM judge is deliberately absent. The review's comparison found a
plain equal-weight mean competitive with a judge, and a mean is
reproducible, free and auditable.
'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import combinations
from statistics import fmean, pstdev

__all__ = [
  'LookAheadError',
  'Rejection',
  'SentimentAggregate',
  'SentimentReading',
  'aggregate',
  'converged',
  'convergence_tolerance',
  'default_min_sources',
  'disagreement',
  'require_visible',
  'visible_readings',
]

#: Minimum number of **distinct** sources before an aggregate is usable.
#: Two, not one, and not three. Johnsen & Shasharina's inverted FinBERT
#: result is the reason one is too few: a single classifier output used
#: as a signal had the wrong sign. Three is not required either,
#: because a hard floor on corroboration in a thin news day would
#: discard the reading exactly when the tape is interesting.
default_min_sources = 2

#: Smallest gap between two scores at or below which those two sources
#: count as having converged. The Jensen-Shannon divergence guard was the
#: only significant intervention in the Stanford study (Sharpe +0.14,
#: p = 0.028), and its point was that independent extractors which
#: produce identical output have stopped being independent. This is the
#: standard-library stand-in for that idea, on a different scale: it is
#: NOT the JSD statistic and must not be reported as if it were.
convergence_tolerance = 0.05

#: Bucket key for readings that name no source. All of them collapse
#: into this one group, because an unlabelled feed cannot be shown to be
#: independent of another unlabelled feed.
_unnamed = '<unnamed>'


class Rejection(StrEnum):
  '''Why an aggregate is not usable.

  Attributes:
    NO_READINGS: Nothing was supplied for the symbol.
    LOOK_AHEAD: At least one reading was public **after** the decision
      bar. The reading is dropped AND the aggregate is refused, so
      ``usable`` is False. The refusal is the point: a leak means the
      inputs to this average are not all knowable at the bar, and there
      is no way to subtract one from a mean afterwards. Nothing in the
      package reads ``rejected``, so recording the leak without refusing
      it would have been a note to self that no code acted on -- which
      is exactly what happened before.
    SINGLE_SOURCE: Fewer distinct sources than the minimum. The reading
      is an unverified opinion, and a FinBERT-class signal used alone
      was measured with the wrong sign.
  '''

  NO_READINGS = 'no_readings'
  LOOK_AHEAD = 'look_ahead'
  SINGLE_SOURCE = 'single_source'


class LookAheadError(ValueError):
  '''Raised when a reading was not yet public at the decision bar.

  Subclasses :class:`ValueError` so a caller who catches the ordinary
  case still catches this, while a pipeline that genuinely must fail
  loudly can single out the leak.
  '''


@dataclass(frozen=True, slots=True)
class SentimentReading:
  '''One already-computed sentiment observation about one symbol.

  There is no model in this file and no way to call one. A reading
  arrives from outside -- a classifier, a feed, a human -- and this
  module's only job is to decide whether it may be used at a given bar.

  Attributes:
    symbol: Canonical NSE symbol. Free text belongs in
      :mod:`stock_rl.sentiment.entity_link` first; resolving it here
      would put two decisions in one place.
    score: Directional score in ``[-1, 1]``, negative is bearish.
    event_type: Free-form label for the kind of event, e.g.
      ``'earnings'`` or ``'regulatory'``. Deliberately a string: an enum
      here would imply a taxonomy nobody has validated.
    intensity: Strength of the event in ``[0, 1]``. Kept **separate**
      from ``score``, because "this was a big event" and "this was good
      news" are different facts and collapsing them loses one.
    available_from: When the information became **publicly** available.
      Never the fetch time. A reading whose ``available_from`` is after
      the decision bar is a look-ahead leak, and
      :func:`aggregate` refuses it.
    source: Identifier of the source that produced this reading, e.g.
      ``'reuters'`` or ``'finbert_local'``. Two items from one source are
      one source, which is the whole point of the minimum.
    source_count: How many independent sources this reading claims to
      stand behind. Recorded, and deliberately **not** used to satisfy
      the source minimum: a wire story syndicated to forty outlets is
      still one editorial source, and counting its republication as
      corroboration would make the minimum a formality.
  '''

  symbol: str
  score: float
  event_type: str
  intensity: float
  available_from: datetime
  source: str = ''
  source_count: int = 1

  def __post_init__(self) -> None:
    '''Validate the reading at construction time.

    Raises:
      ValueError: If the symbol is blank, the score or intensity is out
        of range, ``source_count`` is below 1, or ``available_from`` is
        naive. Timezone-awareness is required rather than optional: a
        naive timestamp compared with an aware decision bar raises a
        ``TypeError`` from deep inside a backtest, and an IST-vs-UTC
        slip on the one field that defines look-ahead is not a
        recoverable class of bug.
    '''
    if not self.symbol.strip():
      raise ValueError('sentiment reading needs a symbol')
    if not -1.0 <= self.score <= 1.0:
      raise ValueError(f'score must be in [-1, 1], got {self.score}')
    if not 0.0 <= self.intensity <= 1.0:
      raise ValueError(f'intensity must be in [0, 1], got {self.intensity}')
    if self.source_count < 1:
      raise ValueError(
        f'source_count must be >= 1, got {self.source_count}')
    if self.available_from.tzinfo is None:
      raise ValueError(
        f'{self.symbol}: available_from must be timezone-aware; a naive '
        f'timestamp on the look-ahead field is not acceptable')

  def is_public_at(self, decision_bar: datetime) -> bool:
    '''Return True when this reading was public at the decision bar.

    Args:
      decision_bar: The bar the decision is taken at.

    Returns:
      True when ``available_from <= decision_bar``.

    Raises:
      ValueError: If ``decision_bar`` is naive.
    '''
    _require_aware(decision_bar, 'decision_bar')
    return self.available_from <= decision_bar


@dataclass(frozen=True, slots=True)
class SentimentAggregate:
  '''Equal-weight aggregate of the usable readings for one symbol.

  Attributes:
    symbol: Symbol the aggregate describes.
    score: Equal-weight mean of the per-source scores, in ``[-1, 1]``.
      0.0 when unusable, because a leaked or single-source number must
      not leak into a state vector wearing a real-looking value.
    disagreement: Population standard deviation of the per-source
      scores. **This is a feature, not noise.** High disagreement means
      the sources do not agree on the sign, which is information about
      uncertainty that an average destroys.
    converged: True when the **closest pair** of per-source scores sits
      within :data:`convergence_tolerance` of each other, i.e. when at
      least two sources agree closely enough that averaging them is
      averaging a voice with itself. Computed by :func:`converged` on
      every aggregate, so the guard is on the production path rather
      than exported and never called.

      **Reported, not enforced.** :func:`aggregate` still publishes the
      mean, because a genuine news event really can produce unanimous
      scores and a hard refusal would throw away the reading exactly
      when every desk saw the same thing. Refusing is therefore the
      caller's decision, on a flag this module computes and hands over.
      What it does *not* do is what the old spread-about-the-mean
      statistic did: with two cloned extractors and one honest dissenter
      it reported False, so the flag was silent in the case the review
      says it exists for. The default is ``False`` and the field is
      declared last so a hand-built aggregate cannot be accused of a
      convergence it never measured.
    source_count: Number of **distinct** sources behind the score. This
      is what the minimum is applied to.
    reading_count: Number of readings collapsed into those sources.
    corroboration: Sum of the ``source_count`` each source declared.
      Reported so a caller can see how much republication is claiming
      to be behind a single editorial source.
    intensity: Equal-weight mean of reading intensities.
    usable: False when the aggregate must not enter a state vector. Note
    that the score is **zeroed** in that case, not merely flagged: a
    number that must not be used should not be able to reach a caller
    that forgets to check the flag. ``source_scores`` still carries the
    raw per-source values for an audit trail.
    rejected: Every :class:`Rejection` that applied, so the reason is
      auditable rather than inferred.
    source_scores: The per-source means, in first-seen source order.
  '''

  symbol: str
  score: float
  disagreement: float
  source_count: int
  reading_count: int
  corroboration: int
  intensity: float
  usable: bool
  rejected: tuple[str, ...]
  source_scores: tuple[float, ...]
  # Defaulted, and last, so the positional order of the fields above is
  # unchanged for any caller that builds an aggregate by hand.
  converged: bool = False


def visible_readings(
  readings: list[SentimentReading],
  decision_bar: datetime,
) -> tuple[SentimentReading, ...]:
  '''Return the readings that were public at ``decision_bar``.

  Args:
    readings: Readings to filter. Order is preserved.
    decision_bar: The bar the decision is taken at.

  Returns:
    Tuple of readings with ``available_from <= decision_bar``.

  Raises:
    ValueError: If ``decision_bar`` is naive.
  '''
  _require_aware(decision_bar, 'decision_bar')
  return tuple(reading for reading in readings
               if reading.available_from <= decision_bar)


def require_visible(
  readings: list[SentimentReading],
  decision_bar: datetime,
) -> tuple[SentimentReading, ...]:
  '''Return the public readings, or refuse loudly about the leaks.

  The strict counterpart to :func:`visible_readings`. A backtest that
  silently drops a leaked reading still reports a number, and that number
  is built on data a trader could not have had. Use this where silence
  would hide a bug: the ingest path, the end-of-day rebuild, and any
  test.

  Args:
    readings: Readings to check.
    decision_bar: The bar the decision is taken at.

  Returns:
    The public readings, unchanged.

  Raises:
    LookAheadError: If any reading became public after the decision bar.
    ValueError: If ``decision_bar`` is naive.
  '''
  _require_aware(decision_bar, 'decision_bar')
  leaked = [reading for reading in readings
            if reading.available_from > decision_bar]
  if leaked:
    listed = ', '.join(
      f'{reading.symbol}@{reading.available_from.isoformat()}'
      for reading in leaked)
    raise LookAheadError(
      f'{len(leaked)} reading(s) were not public at '
      f'{decision_bar.isoformat()}: {listed}')
  return tuple(readings)


def aggregate(
  readings: list[SentimentReading],
  symbol: str,
  decision_bar: datetime,
  min_sources: int = default_min_sources,
) -> SentimentAggregate:
  '''Aggregate readings for ``symbol`` into one equal-weight number.

  Steps, in order, and each step is a reason a caller can audit:

  1. Filter to the symbol.
  2. Drop anything not yet public at ``decision_bar``, recording
     :attr:`Rejection.LOOK_AHEAD` if any were dropped.

     ``decision_bar`` is REQUIRED and has no default. It used to default
     to ``None``, which skipped the filter entirely rather than applying
     it, so the permissive default was the one that let the future in:
     four readings dated a year after the bar reached a context as usable
     sentiment at full strength, with ``rejected`` empty and no record
     that anything had been dropped. In a backtesting library the
     unattributed call is the dangerous one, so the signature now forces
     the caller to say which bar it is standing on.
  3. Collapse readings to one score per **distinct source**, so one
     prolific wire feed cannot masquerade as three corroborating voices.
     Readings with no ``source`` collapse into a single unlabelled
     bucket, because the minimum exists to count independent voices and
     an unlabelled feed cannot be shown to be independent of itself.
  4. Refuse below ``min_sources`` distinct sources.
  5. Average with equal weights and report the cross-source spread
     alongside it.
  6. Run :func:`converged` over the per-source scores and publish the
     answer on :attr:`SentimentAggregate.converged`.

  Step 6 is the anti-convergence guard, and it is here rather than
  nowhere because :func:`converged` used to be called by nothing in the
  package: a documented, tested and completely unreachable guard. It is
  reported and not enforced, because a unanimous score really can mean a
  single shared event, and the caller is the only thing that knows
  whether it does.

  No LLM, no judge, no weighting scheme. The review found a plain mean
  competitive with an LLM judge, and a weighted mean would be a model
  with no out-of-sample evidence behind it.

  Args:
    readings: All readings for the decision, of any symbol.
    symbol: Symbol to aggregate.
    decision_bar: Bar the decision is taken at, timezone-aware. Required
      in a backtest; ``None`` means "everything handed over is already
      public", which is only true live.
    min_sources: Minimum distinct sources for a usable aggregate.

  Returns:
    The aggregate, usable or not, always with the reason attached.

  Raises:
    ValueError: If ``symbol`` is blank, ``decision_bar`` is naive, or
      ``min_sources`` is below 1.
  '''
  if not symbol.strip():
    raise ValueError('aggregate needs a symbol')
  if min_sources < 1:
    raise ValueError(f'min_sources must be >= 1, got {min_sources}')
  if decision_bar is not None:
    _require_aware(decision_bar, 'decision_bar')
  mine = [reading for reading in readings if reading.symbol == symbol]
  rejections: list[str] = []
  if decision_bar is not None:
    public = [reading for reading in mine
              if reading.available_from <= decision_bar]
    if len(public) != len(mine):
      rejections.append(Rejection.LOOK_AHEAD.value)
    mine = public
  if not mine:
    rejections.append(Rejection.NO_READINGS.value)
    return SentimentAggregate(
      symbol=symbol,
      score=0.0,
      disagreement=0.0,
      source_count=0,
      reading_count=0,
      corroboration=0,
      intensity=0.0,
      usable=False,
      rejected=tuple(rejections),
      source_scores=(),
      converged=False,
    )
  per_source: dict[str, list[SentimentReading]] = {}
  for reading in mine:
    per_source.setdefault(reading.source or _unnamed, []).append(reading)
  source_scores = tuple(fmean([reading.score for reading in group])
                        for group in per_source.values())
  corroboration = sum(max(reading.source_count for reading in group)
                      for group in per_source.values())
  usable = len(source_scores) >= min_sources
  if not usable:
    rejections.append(Rejection.SINGLE_SOURCE.value)
  # A leak refuses the aggregate outright, whatever the source count says.
  # The docstring for Rejection.LOOK_AHEAD has always promised this; the
  # code dropped the offending reading and published the average of the
  # survivors with usable=True. Nothing in the package reads `rejected`,
  # so the leak was recorded and then ignored, and the surviving score
  # reached a state vector at full strength.
  if Rejection.LOOK_AHEAD.value in rejections:
    usable = False
  return SentimentAggregate(
    symbol=symbol,
    score=fmean(source_scores) if usable else 0.0,
    disagreement=disagreement(source_scores),
    source_count=len(source_scores),
    reading_count=len(mine),
    corroboration=corroboration,
    intensity=fmean([reading.intensity for reading in mine]),
    usable=usable,
    rejected=tuple(rejections),
    source_scores=source_scores,
    converged=converged(source_scores),
  )


def disagreement(scores: list[float] | tuple[float, ...]) -> float:
  '''Return the population standard deviation of source scores.

  Args:
    scores: Per-source scores, each in ``[-1, 1]``.

  Returns:
    Standard deviation in ``[0, 1]``, which is 0.0 for a single value or
    for perfect agreement and at most 1.0 when half the sources say
    ``+1`` and half say ``-1``.

  Raises:
    ValueError: If any score is outside ``[-1, 1]``.
  '''
  if not scores:
    return 0.0
  for score in scores:
    if not -1.0 <= score <= 1.0:
      raise ValueError(f'score must be in [-1, 1], got {score}')
  return pstdev(scores)


def converged(
  scores: list[float] | tuple[float, ...],
  tolerance: float = convergence_tolerance,
) -> bool:
  '''Return True when any two sources agree suspiciously well.

  The Stanford study's mechanism was sycophantic convergence: three
  different providers produced byte-identical portfolios, and cash
  allocations converged from 10-50% at proposal to 55-61% by round five.
  Independent inputs that agree to within noise are no longer
  independent, and averaging them then buys nothing but an amplified
  error. This is the cheap check for that on the sentiment side.

  **The statistic is the CLOSEST PAIR, not the spread.** It used to be
  the mean absolute deviation about the mean, which cannot detect the
  case it exists for: one dissenting outlier inflates that average past
  any tolerance, so two cloned extractors plus one honest source came
  back ``converged=False``, and only the case where every source agreed
  was ever caught. Measured on the old code::

      converged([1.0, 1.0, 0.0])    -> False   # the pathology, missed
      converged([0.4, 0.4, -0.4])   -> False   # ditto
      converged([1.0, 1.0])         -> True    # only all-identical

  The minimum pairwise gap has the property the guard needs: the number
  of dissenting sources is irrelevant, because averaging a distance over
  sources is exactly what let the dissenter speak over the clones.

  **This is not the Jensen-Shannon divergence the review reports.** It
  is a pairwise distance threshold on scores, it says nothing about *why*
  the sources agree, and a genuine news event really can produce
  unanimous scores. Treat a True here as "these inputs are not independent
  evidence", not as a bug.

  Args:
    scores: Per-source scores.
    tolerance: Largest gap between two scores at or below which those two
      sources count as converged.

  Returns:
    True when the **closest pair** of scores is within ``tolerance`` of
    each other, which is what "at least two scores agree to within
    tolerance" means. Fewer than two scores is False, because one source
    cannot disagree with itself.

  Raises:
    ValueError: If ``tolerance`` is negative.
  '''
  if tolerance < 0.0:
    raise ValueError(f'tolerance must be >= 0, got {tolerance}')
  if len(scores) < 2:
    return False
  closest = min(abs(left - right) for left, right in combinations(scores, 2))
  return closest <= tolerance


def _require_aware(moment: datetime, name: str) -> None:
  '''Reject a naive datetime on a look-ahead field.

  Args:
    moment: Timestamp to check.
    name: Argument name, for the error message.

  Raises:
    ValueError: If ``moment`` has no timezone.
  '''
  if moment.tzinfo is None:
    raise ValueError(f'{name} must be timezone-aware, got {moment!r}')
