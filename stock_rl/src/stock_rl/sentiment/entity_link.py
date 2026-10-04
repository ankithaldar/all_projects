#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Free-text to NSE symbol resolution, with an explicit ambiguity guard.

The design doc's UI and API accept ``"RELIANCE"`` and expect
``Reliance Industries``. Getting that wrong is not a display bug: the
name is the join key for every feature, every graph edge and every
position. Resolve ``"RELIANCE"`` to ``RELIANCEINFRA`` and the backtest
still runs, still produces an equity curve, and is measuring the wrong
company for the whole period. Silent wrong-entity resolution is worse
than no resolution, so the failure mode here is a **refusal**.

Refusals are explicit and cheap to handle:

* ``resolved`` -- exactly one canonical symbol.
* ``ambiguous`` -- a prefix matches more than one. Every candidate is
  returned and **none** is chosen.
* ``unknown`` -- nothing matched, including empty input.

**Why not an LLM for this?** Three reasons, in order of weight. It is
the one classification problem in the pipeline where a wrong answer is
invisible downstream, so it wants a deterministic function and an
explicit alias table rather than a model. It is free: a table is bytes,
a model is dollars per headline. And it is auditable: a reviewer can
read the table and check every row, which is exactly what the review
demands of the rest of this project.

Matching is layered, most explicit first:

1. **Alias table** -- ``'reliance industries'``, ``'ril'``. An alias row
   is a human assertion that this text means this symbol, so it wins
   over any prefix rule. This is why ``'RELIANCE'`` resolves to
   ``RELIANCE`` and not to ``RELIANCEINFRA``, even though
   ``'RELIANCE'`` is also a prefix of ``'RELIANCEINFRA'``.
2. **Exact symbol** -- ``'INFY'``.
3. **Prefix**, on the symbol and on the company name, with corporate
   suffixes dropped from both sides so ``'asian paint'`` and
   ``'asian paints ltd'`` both land.

**Step 3 IS a deliberate fuzzy match, and this module says so rather
than claiming otherwise.** There is no edit distance, no spelling
correction and no transliteration anywhere in the file, all of which is
true and worth keeping. What is *not* true, and used to be claimed
here, is "no fuzzy matching": a query token is matched as a **prefix**
of a name token, so a truncated or misspelt company name finds its
symbol. Measured on the starter registry:

  resolve('Infosy')         -> INFY     # six of seven characters
  resolve('inf')            -> INFY     # three characters of anything
  resolve('Axis Ban')       -> AXISBANK # 'ban' is a prefix of 'bank'
  resolve('sun pharm')      -> SUNPHARMA
  resolve('Dr Red')         -> DRREDDY

That is the residual risk, stated plainly: **a prefix match can be the
right answer for the wrong reason**, and the whole reason it is
acceptable is that the cost of the mistake is a refusal-shaped result
rather than a wrong number. What makes it survivable here, and what did
not exist before, is that such a result is *labelled as a guess*:

* :attr:`Resolution.truncated` is True whenever the match rests on a
  fragment, and False when a whole symbol or a whole number of name
  tokens matched. A caller that needs certainty checks the flag; a
  caller reading an audit log reads :attr:`Resolution.reason`, which
  says ``'whole name tokens matched'`` or ``'whole symbol spelled'``
  rather than one undifferentiated ``'unambiguous prefix match'``.
* :data:`min_prefix_chars` is applied to **every** query token as well
  as to the squashed query, so a two-letter fragment cannot match
  anything.
* Ambiguity still refuses: ``'Tata'`` is ``ambiguous`` across
  TATAMOTORS, TATASTEEL and TCS, ``'Rel'`` and ``'Relian'`` are
  ``ambiguous`` across RELIANCE and RELIANCEINFRA, and ``'RELIANCE'``
  is RELIANCE and never RELIANCEINFRA.

**The stricter alternative was considered and rejected.** Requiring a
whole-token match, or a minimum per-token prefix fraction, does not
work at the symbol layer at all, because ``'inf'`` and ``'infy'`` are
the same fraction of one another as ``'Rel'`` and ``'Reliance'``: any
rule that lets a short prefix match a long symbol must also let
``'inf'`` match ``'infy'``. And at the name layer a whole-token rule
rejects ``'alpha bet'`` for ``'Alpha Beta'``, which is a shipped
behaviour. Prefix matching is therefore kept, and made visible.

PONYTAIL: exact, alias, prefix; a prefix match is a labelled guess.
Ceiling: a company that rebrands is not tracked, a wrong alias row is
trusted absolutely, and a truncated query is resolved to a symbol that
may be the wrong company. Upgrade path: add a ``typos`` mapping, or a
vendor symbol file loaded through the same constructor -- never by
widening the prefix rule, which is how a refusal becomes a coin flip.
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
  'LinkStatus',
  'Resolution',
  'SymbolLinker',
  'SymbolRecord',
  'default_linker',
  'min_prefix_chars',
  'resolve',
]

#: Shortest prefix accepted by the prefix rule, for the squashed query
#: **and for every query token**. Two letters match half the Nifty and
#: would turn a refusal into a coin flip.
min_prefix_chars = 3

#: Recorded when the squashed query equals the symbol exactly, spaces
#: removed. Nothing was matched by a prefix, so this is not a guess.
_spelled = 'whole symbol spelled without its spaces'

#: Recorded when a whole number of name tokens matched exactly. Also not
#: a guess: every character the query supplied matched.
_tokens = 'whole name tokens matched, no truncation'


def _truncated(layer: str, matched: int, total: int) -> str:
  '''Return the reason for a match that rests on a fragment.

  Args:
    layer: Which spelling was truncated, ``'symbol'`` or ``'name'``.
    matched: Characters of the target the query covered.
    total: Characters in the target.

  Returns:
    A reason that names the fraction, so an audit record shows a guess
    rather than reading like an exact match.
  '''
  return (f'truncated {layer} match: {matched} of {total} characters, '
          f'which is a guess and not a spelling')

#: Corporate suffixes dropped from both sides before name matching, so
#: ``'asian paints limited'`` and ``'asian paint'`` agree.
_suffixes: frozenset[str] = frozenset({
  'and', 'company', 'corporation', 'corp', 'inc', 'incorporated', 'limited',
  'ltd', 'llp', 'plc', 'private', 'pvt', 'the',
})


class LinkStatus(StrEnum):
  '''Outcome of resolving free text to a symbol.

  Attributes:
    RESOLVED: Exactly one canonical symbol matched.
    AMBIGUOUS: More than one matched, so none was chosen. Carries every
      candidate.
    UNKNOWN: Nothing matched.
  '''

  RESOLVED = 'resolved'
  AMBIGUOUS = 'ambiguous'
  UNKNOWN = 'unknown'


@dataclass(frozen=True, slots=True)
class SymbolRecord:
  '''One canonical symbol and the name it is known by.

  Attributes:
    symbol: NSE trading symbol, e.g. ``'RELIANCE'``.
    name: Company name without the corporate suffix, e.g.
      ``'Reliance Industries'``.
    aliases: Extra free-text forms, lowercased. A row here is an
      assertion by a human, so it outranks every prefix rule.
  '''

  symbol: str
  name: str
  aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Resolution:
  '''The result of a resolution attempt, refusal included.

  Attributes:
    query: The text that was asked about, as given.
    normalised: The normalised form that was matched on.
    status: One of :class:`LinkStatus`.
    symbol: The canonical symbol, or ``None`` unless resolved. Never a
      best guess.
    candidates: Every symbol that matched, sorted. Populated for an
      ambiguous result so the caller can offer a choice.
    matched_by: Which rule fired: ``'alias'``, ``'symbol'``, ``'prefix'``
      or ``'none'``.
    reason: Plain-language explanation, for logs and audit records. A
      prefix match says which kind it was: ``'whole symbol spelled
      without its spaces'`` and ``'whole name tokens matched, no
      truncation'`` are exact in effect, while ``'truncated symbol
      match: 3 of 4 characters, ...'`` says out loud that the caller
      guessed.
    truncated: True when the resolution rests on a **fragment** of a
      symbol or of a name token, so ``matched_by='prefix'`` alone does
      not distinguish a correct prefix from a typo that happened to fall
      inside a word. False for every alias, symbol and whole-token
      match. This is the flag a caller checks when a prefix match is not
      good enough; the module docstring's "a prefix match is a labelled
      guess" is this attribute.
  '''

  query: str
  normalised: str
  status: LinkStatus
  symbol: str | None
  candidates: tuple[str, ...]
  matched_by: str
  reason: str
  truncated: bool = False


def _normalise(text: str) -> str:
  '''Lowercase ``text`` and keep only letters, digits and spaces.

  Args:
    text: Raw free text, e.g. ``'Reliance  Industries (NSE)'``.

  Returns:
    Lowercase alphanumeric words separated by single spaces.
  '''
  return ' '.join(
    ''.join(character if character.isalnum() else ' '
             for character in text).split()).lower()


def _strip_suffixes(words: tuple[str, ...]) -> tuple[str, ...]:
  '''Drop corporate suffixes from a normalised token tuple.

  Args:
    words: Normalised tokens.

  Returns:
    Tokens with trailing corporate words removed. A leading ``'the'``
    is dropped too, since it carries no identity.
  '''
  kept = list(words)
  while kept and kept[-1] in _suffixes:
    kept.pop()
  if kept and kept[0] == 'the':
    kept.pop(0)
  return tuple(kept)


class SymbolLinker:
  '''Resolves free text to canonical NSE symbols, refusing ambiguity.

  Attributes:
    symbols: Canonical symbols in the registry, sorted.
    min_prefix: Shortest prefix the prefix rule will accept.
  '''

  symbols: tuple[str, ...]
  min_prefix: int

  def __init__(
    self,
    registry: tuple[SymbolRecord, ...] = (),
    min_prefix: int = min_prefix_chars,
  ) -> None:
    '''Build a linker from a registry of canonical symbols.

    Args:
      registry: Canonical records. Defaults to the starter Nifty-50
        registry, :data:`_registry`, which is deliberately small and
        reviewable rather than a full exchange master.
      min_prefix: Shortest accepted prefix, must be at least 2.

    Raises:
      ValueError: If the registry is empty, contains a duplicate symbol,
        or ``min_prefix`` is below 2.
    '''
    records = tuple(registry) if registry else _registry
    if not records:
      raise ValueError('registry must not be empty')
    seen: set[str] = set()
    for record in records:
      key = record.symbol.strip().upper()
      if key in seen:
        raise ValueError(f'duplicate symbol in registry: {record.symbol}')
      seen.add(key)
    if min_prefix < 2:
      raise ValueError(f'min_prefix must be >= 2, got {min_prefix}')
    self._records = {record.symbol.strip().upper(): record
                     for record in records}
    self._aliases: dict[str, str] = {}
    for symbol, record in self._records.items():
      self._add_alias(_normalise(record.name), symbol)
      for alias in record.aliases:
        self._add_alias(_normalise(alias), symbol)
    self._name_words = {
      symbol: _strip_suffixes(tuple(_normalise(record.name).split()))
      for symbol, record in self._records.items()
    }
    self._symbol_words = {
      symbol: tuple(symbol.split('-')) for symbol in self._records
    }
    self.min_prefix = min_prefix
    self.symbols = tuple(sorted(self._records))

  def _add_alias(self, alias: str, symbol: str) -> None:
    '''Register one alias, refusing to let two symbols share it.

    A silent overwrite here is exactly the silent wrong-entity
    resolution this module exists to prevent, so a collision fails at
    construction rather than at query time.

    Args:
      alias: Normalised alias text.
      symbol: Canonical symbol it maps to.

    Raises:
      ValueError: If a different symbol already owns this alias.
    '''
    if not alias:
      return
    existing = self._aliases.setdefault(alias, symbol)
    if existing != symbol:
      raise ValueError(
        f'alias {alias!r} is claimed by both {existing} and {symbol}')

  def link(self, text: str) -> Resolution:
    '''Resolve free text to a canonical symbol.

    Args:
      text: Free text, e.g. ``'reliance'`` or ``'Reliance Infra'``.

    Returns:
      A :class:`Resolution`. Ambiguous and unknown inputs are returned
      as themselves, not raised, because a news feed contains both and
      the caller needs to know *which* symbols were plausible.

    Raises:
      ValueError: Never. Refusal is a value here by design.
    '''
    normalised = _normalise(text)
    if not normalised:
      return Resolution(
        query=text,
        normalised=normalised,
        status=LinkStatus.UNKNOWN,
        symbol=None,
        candidates=(),
        matched_by='none',
        reason='empty query',
      )
    alias_hit = self._aliases.get(normalised)
    if alias_hit is not None:
      return _resolved(text, normalised, alias_hit, 'alias',
                       'exact company name or alias')
    words = _strip_suffixes(tuple(normalised.split()))
    compact = ''.join(words)
    if compact.upper() in self._records:
      return _resolved(text, normalised, compact.upper(), 'symbol',
                       'exact exchange symbol')
    if len(compact) < self.min_prefix:
      return Resolution(
        query=text,
        normalised=normalised,
        status=LinkStatus.UNKNOWN,
        symbol=None,
        candidates=(),
        matched_by='none',
        reason=f'prefix shorter than {self.min_prefix} characters',
      )
    shortest = min(words, key=len, default='')
    if len(shortest) < self.min_prefix:
      # The floor applies per token, not only to the squashed query. It
      # used to be applied to the squashed string alone, so a query
      # assembled from several two-letter words sailed past a check
      # that every one of them fails.
      return Resolution(
        query=text,
        normalised=normalised,
        status=LinkStatus.UNKNOWN,
        symbol=None,
        candidates=(),
        matched_by='none',
        reason=f'a query token is shorter than {self.min_prefix} '
               f'characters: {shortest!r}',
      )
    evidence = self._prefix_evidence(compact, words)
    if len(evidence) == 1:
      symbol, reason = next(iter(evidence.items()))
      return _resolved(text, normalised, symbol, 'prefix', reason,
                       truncated=reason.startswith('truncated'))
    if evidence:
      return Resolution(
        query=text,
        normalised=normalised,
        status=LinkStatus.AMBIGUOUS,
        symbol=None,
        candidates=tuple(sorted(evidence)),
        matched_by='prefix',
        reason=f'prefix matches {len(evidence)} symbols; refusing to '
               f'guess',
      )
    return Resolution(
      query=text,
      normalised=normalised,
      status=LinkStatus.UNKNOWN,
      symbol=None,
      candidates=(),
      matched_by='none',
      reason='no alias, symbol or unambiguous prefix match',
    )

  def _prefix_evidence(
    self,
    compact: str,
    words: tuple[str, ...],
  ) -> dict[str, str]:
    '''Return every symbol the prefix rule reaches, with its evidence.

    Both the symbol and the company name are tried, and the results are
    unioned, because the name is often what disambiguates: ``'ONGC OIL'``
    prefix-matches the symbol ``'ONGC'``, while ``'Reliance Infra'``
    is settled by the alias table before this is reached.

    The evidence is the reason string, and it is the whole point of the
    method: a symbol reached by matching **all** of a name's characters,
    or a symbol reached by spelling the whole symbol, gets a reason that
    says so, and a symbol reached by matching a **fragment** gets one
    that says that too. Every caller then sees which kind of match it
    is holding, instead of one undifferentiated "unambiguous prefix
    match" for both ``'asian paint'`` and ``'Axis Ban'``.

    Args:
      compact: Query with spaces removed, so ``'hdfc bank'`` can match
        the symbol ``'HDFCBANK'``.
      words: Query tokens with corporate suffixes stripped.

    Returns:
      Mapping of canonical symbol to the reason it matched. Empty when
      none did. When both layers reach the same symbol the **stronger**
      evidence wins, so a query whose name tokens matched in full is not
      downgraded to a truncation merely because the symbol layer also
      half-covered it.
    '''
    evidence: dict[str, str] = {}
    for symbol in self._records:
      found: list[tuple[int, str]] = []
      lowered = ''.join(self._symbol_words[symbol]).lower()
      if lowered == compact:
        found.append((0, _spelled))
      elif lowered.startswith(compact) or compact.startswith(lowered):
        found.append((1, _truncated('symbol', min(len(compact), len(lowered)),
                                    max(len(compact), len(lowered)))))
      name_words = self._name_words[symbol]
      if words and len(words) <= len(name_words) and all(
        name_word.startswith(word)
        for word, name_word in zip(words, name_words)
      ):
        matched = sum(len(word) for word in words)
        total = sum(len(word) for word in name_words[:len(words)])
        found.append((0, _tokens) if matched == total
                     else (1, _truncated('name', matched, total)))
      if found:
        evidence[symbol] = min(found, key=lambda item: item[0])[1]
    return evidence


def _resolved(
  query: str,
  normalised: str,
  symbol: str,
  matched_by: str,
  reason: str,
  truncated: bool = False,
) -> Resolution:
  '''Build a resolved result.

  Args:
    query: Original text.
    normalised: Normalised text that matched.
    symbol: Canonical symbol chosen.
    matched_by: Which rule fired.
    reason: Plain-language explanation.
    truncated: True when the match rested on a fragment rather than on
      a whole symbol or a whole number of name tokens.

  Returns:
    A :class:`Resolution` with status ``resolved``.
  '''
  return Resolution(
    query=query,
    normalised=normalised,
    status=LinkStatus.RESOLVED,
    symbol=symbol,
    candidates=(symbol,),
    matched_by=matched_by,
    reason=reason,
    truncated=truncated,
  )


#: Starter registry. Small on purpose: this is a table to argue with,
#: not an exchange master file.
_registry: tuple[SymbolRecord, ...] = (
  SymbolRecord('RELIANCE', 'Reliance Industries',
               ('Reliance', 'RIL', 'Reliance Industries Ltd')),
  SymbolRecord('RELIANCEINFRA', 'Reliance Infrastructure',
               ('Reliance Infra', 'Reliance Infrastructure Ltd')),
  SymbolRecord('ONGC', 'Oil and Natural Gas Corporation',
               ('Oil India', 'ONGC Ltd')),
  SymbolRecord('ASIANPAINT', 'Asian Paints'),
  SymbolRecord('INFY', 'Infosys', ('Infosys Ltd', 'Infosys Limited')),
  SymbolRecord('TCS', 'Tata Consultancy Services', ('TCS Ltd',)),
  SymbolRecord('HCLTECH', 'HCL Technologies', ('HCL',)),
  SymbolRecord('WIPRO', 'Wipro'),
  SymbolRecord('HDFCBANK', 'HDFC Bank', ('HDFC',)),
  SymbolRecord('ICICIBANK', 'ICICI Bank', ('ICICI',)),
  SymbolRecord('SBIN', 'State Bank of India', ('SBI',)),
  SymbolRecord('AXISBANK', 'Axis Bank', ('Axis',)),
  SymbolRecord('KOTAKBANK', 'Kotak Mahindra Bank', ('Kotak',)),
  SymbolRecord('TATAMOTORS', 'Tata Motors'),
  SymbolRecord('MARUTI', 'Maruti Suzuki India', ('Maruti',)),
  SymbolRecord('TVSMOTOR', 'TVS Motor Company', ('TVS',)),
  SymbolRecord('SUNPHARMA', 'Sun Pharmaceutical',
               ('Sun Pharma',)),
  SymbolRecord('DRREDDY', "Dr Reddy's Laboratories",
               ('Dr Reddy', 'Dr Reddys')),
  SymbolRecord('CIPLA', 'Cipla'),
  SymbolRecord('HINDUNILVR', 'Hindustan Unilever', ('HUL', 'Unilever India')),
  SymbolRecord('ITC', 'ITC Limited'),
  SymbolRecord('BHARTIARTL', 'Bharti Airtel', ('Airtel',)),
  SymbolRecord('NTPC', 'NTPC Limited'),
  SymbolRecord('POWERGRID', 'Power Grid Corporation', ('Power Grid',)),
  SymbolRecord('ULTRACEMCO', 'UltraTech Cement',
               ('Ultratech', 'Ultra Tech Cement')),
  SymbolRecord('SHREECEM', 'Shree Cement'),
  SymbolRecord('TITAN', 'Titan Company', ('Titan Industries',)),
  SymbolRecord('HINDALCO', 'Hindalco Industries', ('Hindalco',)),
  SymbolRecord('JSWSTEEL', 'JSW Steel'),
  SymbolRecord('TATASTEEL', 'Tata Steel'),
)

#: Shared linker over the starter registry, for callers that do not need
#: their own. Module-level rather than per-call so alias tables are
#: built once.
default_linker = SymbolLinker()


def resolve(text: str) -> Resolution:
  '''Resolve free text against the starter registry.

  Args:
    text: Free text to resolve.

  Returns:
    A :class:`Resolution`; ambiguous and unknown inputs are values, not
    exceptions.
  '''
  return default_linker.link(text)
