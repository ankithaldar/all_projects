#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for sentiment scoring and symbol resolution.

Three things here are load-bearing and must never be relaxed:

* **Look-ahead.** A reading stamped with when it was *fetched* rather
  than when it became *public* manufactures a Sharpe out of nothing, and
  the leak is invisible in the equity curve. The rejection tests are the
  most important in this file.
* **The source minimum.** A single classifier output is an opinion, and
  the one FinBERT-class result in the literature was found **inverted**.
* **Refusal over guessing.** ``'RELIANCE'`` must never resolve to
  ``RELIANCEINFRA``, and an ambiguous prefix must return both candidates
  and choose neither.
'''

from datetime import datetime, timedelta, timezone

import pytest

from stock_rl.sentiment import (
  LinkStatus,
  LookAheadError,
  Rejection,
  SentimentReading,
  SymbolLinker,
  SymbolRecord,
  aggregate,
  converged,
  convergence_tolerance,
  default_min_sources,
  disagreement,
  min_prefix_chars,
  require_visible,
  resolve,
  visible_readings,
)

IST = timezone(timedelta(hours=5, minutes=30))
BAR = datetime(2026, 1, 2, 15, 30, tzinfo=IST)
BEFORE = BAR - timedelta(hours=6)
AFTER = BAR + timedelta(days=1)


def reading(
  score: float = 0.5,
  source: str = 'reuters',
  event_type: str = 'earnings',
  intensity: float = 0.5,
  available_from: datetime = BEFORE,
  source_count: int = 1,
  symbol: str = 'RELIANCE',
) -> SentimentReading:
  '''Return a valid reading, so each test varies one field only.

  Args:
    score: Directional score in ``[-1, 1]``.
    source: Source identifier.
    event_type: Event label.
    intensity: Event strength in ``[0, 1]``.
    available_from: Public availability timestamp.
    source_count: Declared corroborating sources.
    symbol: Canonical symbol.

  Returns:
    A constructed reading.
  '''
  return SentimentReading(
    symbol=symbol,
    score=score,
    event_type=event_type,
    intensity=intensity,
    available_from=available_from,
    source=source,
    source_count=source_count,
  )


# --- reading validation ---------------------------------------------------

def test_reading_rejects_out_of_range_and_naive_timestamps() -> None:
  '''Ranges are checked, and a naive timestamp is never accepted.'''
  with pytest.raises(ValueError, match='score'):
    reading(score=1.5)
  with pytest.raises(ValueError, match='intensity'):
    reading(intensity=-0.1)
  with pytest.raises(ValueError, match='source_count'):
    reading(source_count=0)
  with pytest.raises(ValueError, match='symbol'):
    reading(symbol=' ')
  with pytest.raises(ValueError, match='timezone-aware'):
    reading(available_from=datetime(2026, 1, 2, 15, 30))


def test_reading_validates_the_decision_bar_timezone() -> None:
  '''A naive decision bar is refused at the comparison, not below it.'''
  with pytest.raises(ValueError, match='timezone-aware'):
    reading().is_public_at(datetime(2026, 1, 2, 15, 30))


# --- look-ahead -----------------------------------------------------------

def test_reading_public_after_the_bar_is_rejected_as_look_ahead() -> None:
  '''A reading public after the decision bar is refused, not averaged in.'''
  readings = [reading(source='reuters', available_from=BEFORE),
              reading(score=0.9, source='bloomberg',
                      available_from=AFTER)]
  result = aggregate(readings, 'RELIANCE', BAR)
  assert not result.usable
  assert Rejection.LOOK_AHEAD.value in result.rejected
  assert result.score == 0.0
  assert result.source_count == 1


def test_look_ahead_reading_is_never_used_even_alongside_good_ones() -> None:
  '''One leak poisons the average; there is no way to subtract it.'''
  readings = [reading(score=0.4, source='reuters'),
              reading(score=0.6, source='bloomberg'),
              reading(score=-0.9, source='cnbc',
                      available_from=BAR + timedelta(minutes=1))]
  result = aggregate(readings, 'RELIANCE', BAR)
  assert result.source_count == 2
  assert result.score == pytest.approx(0.5)
  assert Rejection.LOOK_AHEAD.value in result.rejected


def test_available_from_is_public_time_not_fetch_time() -> None:
  '''Fetch time and public time differ; only public time is load-bearing.'''
  late = reading(source='reuters', available_from=BAR - timedelta(minutes=1))
  assert late.is_public_at(BAR)
  assert not late.is_public_at(BAR - timedelta(hours=1))


def test_visible_and_required_filters_agree_on_what_is_usable() -> None:
  '''The lenient filter drops leaks; the strict one names them.'''
  readings = [reading(source='reuters'), reading(source='cnbc',
                                                  available_from=AFTER)]
  assert len(visible_readings(readings, BAR)) == 1
  with pytest.raises(LookAheadError, match='RELIANCE'):
    require_visible(readings, BAR)
  assert len(require_visible(readings[:1], BAR)) == 1
  assert not require_visible([], BAR)
  with pytest.raises(ValueError, match='timezone-aware'):
    visible_readings(readings, datetime(2026, 1, 2, 15, 30))
  with pytest.raises(ValueError, match='timezone-aware'):
    require_visible(readings, datetime(2026, 1, 2, 15, 30))


# --- aggregation ----------------------------------------------------------

def test_aggregate_without_a_decision_bar_uses_every_reading() -> None:
  '''Live use has no bar to leak past; the caller takes responsibility.'''
  readings = [reading(source='reuters'), reading(source='cnbc')]
  result = aggregate(readings, 'RELIANCE')
  assert result.usable
  assert result.source_count == 2
  assert Rejection.LOOK_AHEAD.value not in result.rejected


def test_aggregate_ignores_other_symbols() -> None:
  '''A reading about another name is not evidence about this one.'''
  readings = [reading(source='reuters'), reading(source='bloomberg'),
              reading(source='cnbc', symbol='TCS')]
  result = aggregate(readings, 'RELIANCE', BAR)
  assert result.source_count == 2
  assert result.score == pytest.approx(0.5)


def test_source_minimum_rejects_a_single_source() -> None:
  '''One unverified opinion is not a reading; FinBERT alone inverted.'''
  result = aggregate([reading(source='finbert_local')], 'RELIANCE', BAR)
  assert not result.usable
  assert result.score == 0.0
  assert result.disagreement == 0.0
  assert Rejection.SINGLE_SOURCE.value in result.rejected
  assert default_min_sources == 2


def test_one_source_with_many_readings_is_still_one_source() -> None:
  '''Repetition is not corroboration: a prolific feed stays one voice.'''
  readings = [reading(source='reuters'),
              reading(source='reuters', available_from=BAR - timedelta(days=1)),
              reading(source='reuters', available_from=BAR - timedelta(days=2))]
  result = aggregate(readings, 'RELIANCE', BAR)
  assert result.source_count == 1
  assert not result.usable
  assert result.reading_count == 3


def test_unlabelled_readings_collapse_into_one_bucket() -> None:
  '''An unnamed feed cannot be shown to be independent of another.'''
  readings = [reading(source=''), reading(source='')]
  result = aggregate(readings, 'RELIANCE', BAR)
  assert result.source_count == 1
  assert not result.usable


def test_declared_corroboration_does_not_satisfy_the_minimum() -> None:
  '''Syndication is not independence, so source_count is only reported.'''
  result = aggregate([reading(source='reuters', source_count=40)],
                     'RELIANCE', BAR)
  assert not result.usable
  assert result.source_count == 1
  assert result.corroboration == 40


def test_minimum_is_callable_and_validated() -> None:
  '''A caller can demand three sources; zero is nonsense.'''
  readings = [reading(source='reuters'), reading(source='cnbc')]
  assert aggregate(readings, 'RELIANCE', BAR, 3).usable is False
  assert aggregate(readings, 'RELIANCE', BAR, 1).usable
  with pytest.raises(ValueError, match='min_sources'):
    aggregate(readings, 'RELIANCE', BAR, 0)
  with pytest.raises(ValueError, match='symbol'):
    aggregate(readings, ' ', BAR)


def test_no_readings_is_reported_as_such() -> None:
  '''Silence is not a neutral score; it is a missing one.'''
  result = aggregate([], 'RELIANCE', BAR)
  assert result.source_count == 0
  assert result.reading_count == 0
  assert not result.source_scores
  assert not result.usable
  assert Rejection.NO_READINGS.value in result.rejected


# --- disagreement is a feature -------------------------------------------

def test_disagreement_is_preserved_not_averaged_away() -> None:
  '''Two sources that disagree must not collapse into one quiet number.'''
  readings = [reading(score=0.8, source='reuters'),
              reading(score=-0.8, source='cnbc')]
  result = aggregate(readings, 'RELIANCE', BAR)
  assert result.usable
  assert result.score == pytest.approx(0.0)
  assert result.disagreement == pytest.approx(0.8)
    # A mean of zero with a real spread is the honest output, and the
    # spread is the part that survives into the context vector.
  assert result.source_scores == (0.8, -0.8)


def test_disagreement_helper_bounds() -> None:
  '''The measure is bounded and rejects impossible inputs.'''
  assert disagreement([]) == 0.0
  assert disagreement([0.4]) == 0.0
  assert disagreement([1.0, -1.0]) == pytest.approx(1.0)
  with pytest.raises(ValueError, match='score'):
    disagreement([2.0])


def test_convergence_guard_flags_identical_extractors() -> None:
  '''Independent sources that agree to nothing are not independent.'''
  assert converged([0.5, 0.51]) is True
  assert converged([0.5, -0.5]) is False
  assert converged([0.5]) is False
  assert convergence_tolerance == 0.05
  with pytest.raises(ValueError, match='tolerance'):
    converged([0.5, 0.5], tolerance=-1.0)


def test_unusable_aggregate_still_reports_its_disagreement() -> None:
  '''One source has no spread to report, and the reason is single_source.'''
  result = aggregate([reading(score=0.9)], 'RELIANCE', BAR)
  assert result.disagreement == 0.0
  assert result.rejected == (Rejection.SINGLE_SOURCE.value,)


# --- symbol resolution ----------------------------------------------------

def test_reliance_resolves_to_reliance_and_not_reliance_infra() -> None:
  '''The doc's example. RELIANCEINFRA is a real, separate NSE symbol.'''
  result = resolve('RELIANCE')
  assert result.status is LinkStatus.RESOLVED
  assert result.symbol == 'RELIANCE'
  assert 'RELIANCEINFRA' not in result.candidates


def test_ambiguous_prefix_returns_ambiguous_and_picks_nothing() -> None:
  '''Two prefix matches, zero chosen. A guess would be silent corruption.'''
  result = resolve('RELI')
  assert result.status is LinkStatus.AMBIGUOUS
  assert result.symbol is None
  assert result.candidates == ('RELIANCE', 'RELIANCEINFRA')
  assert result.matched_by == 'prefix'


def test_full_name_of_the_other_company_still_resolves() -> None:
  '''Refusing an ambiguity must not make the company unnameable.'''
  result = resolve('reliance infrastructure ltd')
  assert result.status is LinkStatus.RESOLVED
  assert result.symbol == 'RELIANCEINFRA'


@pytest.mark.parametrize('text,expected', [
  ('ril', 'RELIANCE'),
  ('Reliance Industries', 'RELIANCE'),
  ('reliance infra', 'RELIANCEINFRA'),
  ('ASIANPAINT', 'ASIANPAINT'),
  ('asian paint', 'ASIANPAINT'),
  ('Infosys Ltd', 'INFY'),
  ('infy', 'INFY'),
  ('HDFC Bank', 'HDFCBANK'),
  ('hdfc', 'HDFCBANK'),
  ('sbi', 'SBIN'),
  ('dr reddy', 'DRREDDY'),
  ('sun pharma', 'SUNPHARMA'),
  ('ongc oil', 'ONGC'),
])
def test_alias_symbol_and_prefix_layers(text: str, expected: str) -> None:
  '''Each resolution layer works, and none of them needs a model.

  Args:
    text: Free text as a news feed would present it.
    expected: Canonical symbol it must resolve to.
  '''
  result = resolve(text)
  assert result.status is LinkStatus.RESOLVED
  assert result.symbol == expected


@pytest.mark.parametrize('text', ['', '   ', 'MRF', 'ZZZZZZ', 'ON'])
def test_unknown_text_is_unknown_not_a_nearest_match(text: str) -> None:
  '''No fuzzy fallback: an unknown name stays unknown.

  Args:
    text: Free text with no defensible resolution.
  '''
  result = resolve(text)
  assert result.status is LinkStatus.UNKNOWN
  assert result.symbol is None
  assert not result.candidates
  assert result.reason


def test_short_prefix_is_refused_before_matching() -> None:
  '''Two letters match half the universe, so they are not a match.'''
  result = resolve('RE')
  assert result.status is LinkStatus.UNKNOWN
  assert 'shorter' in result.reason
  assert min_prefix_chars == 3


def test_linker_registry_is_configurable_and_validated() -> None:
  '''A vendor symbol file plugs into the same constructor.'''
  linker = SymbolLinker((SymbolRecord('AAA', 'Alpha Corp', ('Alpha',)),
                         SymbolRecord('AAB', 'Alpha Beta Ltd')))
  assert linker.symbols == ('AAA', 'AAB')
  assert linker.link('Alpha').symbol == 'AAA'
  assert linker.link('alpha bet').symbol == 'AAB'
  with pytest.raises(ValueError, match='duplicate symbol'):
    SymbolLinker((SymbolRecord('AAA', 'One'), SymbolRecord('AAA', 'Two')))
  with pytest.raises(ValueError, match='min_prefix'):
    SymbolLinker((), min_prefix=1)


def test_corporate_suffixes_are_dropped_from_both_sides() -> None:
  '''"Limited" on one side only must not break a name match.'''
  assert resolve('Sun Pharmaceutical Ltd').symbol == 'SUNPHARMA'
  assert resolve('the Tata Consultancy Services').symbol == 'TCS'
  assert resolve('Tata Consultancy').symbol == 'TCS'


def test_symbol_matching_splits_a_hyphenated_exchange_symbol() -> None:
  '''A hyphenated symbol is matched on its parts, not its raw text.'''
  linker = SymbolLinker((SymbolRecord('BAJAJ-AUTO', 'Bajaj Auto'),
                         SymbolRecord('HDFCBANK', 'HDFC Bank')))
  assert linker.link('bajaj').symbol == 'BAJAJ-AUTO'
  assert linker.link('BAJAJAUTO').symbol == 'BAJAJ-AUTO'
  assert linker.link('hdfc').symbol == 'HDFCBANK'


def test_two_symbols_claiming_one_alias_fails_at_construction() -> None:
  '''A silent alias overwrite would be a silent wrong entity.'''
  with pytest.raises(ValueError, match='claimed by both'):
    SymbolLinker((SymbolRecord('AAA', 'Same Name'),
                  SymbolRecord('BBB', 'Same Name')))


def test_custom_prefix_length_is_honoured() -> None:
  '''The prefix floor is a constructor argument, not a constant.'''
  linker = SymbolLinker((SymbolRecord('TITAN', 'Titan Company'),))
  assert linker.link('TI').status is LinkStatus.UNKNOWN
  assert linker.min_prefix == 3
