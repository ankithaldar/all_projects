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
   ``'asian paints ltd'`` both land. The symbol layer is **one
   directional**: a query may be *shorter* than the symbol, never
   *longer*. The longer case needs the company's own name to vouch for
   the characters beyond the symbol, and the rule that does it is stated
   in full below rather than left for a reader to infer.

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

**A query that EXTENDS a symbol is not a prefix of it, and is refused
unless the company corroborates the characters beyond the symbol.**
``'ITC-INFRA'`` squashed is ``'itcinfra'``: it *contains* the symbol
``'itc'`` plus the word ``'infra'``, which belongs to Reliance
Infrastructure. Answering it with ``ITC`` names one company and means
another, which is the worst outcome this module has, and a test written
as ``lowered.startswith(compact)`` cannot catch it because the
containment ran the other way round.

The symbol layer's inverse case therefore fires only on
**corroboration**, and the corroboration has to come from the name
layer, which is the one other opinion about what the words mean:

* the squashed symbol must be spelled by a **whole run of query
  tokens**. ``'itcxyz'`` and ``'AXISBANKING'`` are refused because no
  whole token spells ``'itc'`` or ``'axisbank'``: there the symbol
  survives only as a *fragment* of the query, which is the fragment a
  prefix match is allowed to fire on and nothing more.
* every **remaining** query token must be a prefix of a **distinct**
  word of that same company's name, paired one word to one, so the
  leftovers cannot all lean on the same word.

``'ongc oil'`` passes both: ``'ongc'`` is a whole token, and ``'oil'``
is ONGC's own first name word. ``'Reliance Industr'`` passes too,
``'industr'`` being a prefix of ``'industries'``, so it resolves to
``RELIANCE`` and is still **labelled a truncation**, which is what it
is. ``'ITC-INFRA'`` fails the second clause: ``'infra'`` is a word of
nobody called ITC.

**This is what makes the longer symbol win.** ``'reliance infra'``
corroborates ``RELIANCEINFRA`` -- the run ``'reliance infra'`` spells
its symbol whole and there are no leftovers -- and it also reaches the
inverse clause for the shorter ``RELIANCE``, where the leftover
``'infra'`` is *not* a word of Reliance, so ``RELIANCE`` is refused and
``RELIANCEINFRA`` is the only candidate left. Nothing needed to compare
lengths: the rule that refuses the wrong company is the same rule that
keeps the right one.

The asymmetry is deliberate. A query *shorter* than the symbol,
``'inf'`` for ``'infy'``, is a truncated query naming a longer name,
which is precisely what a prefix match is for. A query *longer* than the
symbol is naming **more company than it named**, and that is a refusal.

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

PONYTAIL: exact, alias, prefix; a prefix match is a labelled guess; a
superstring of a symbol needs that company's own name to corroborate it.
Ceiling: a company that rebrands is not tracked, a wrong alias row is
trusted absolutely, and a *truncated* query -- one shorter than the
symbol -- is resolved to a symbol that may be the wrong company. Upgrade
path: add a ``typos`` mapping, or a vendor symbol file loaded through the
same constructor -- never by widening the prefix rule, which is how a
refusal becomes a coin flip.

**Nothing in this package acts on :attr:`Resolution.truncated`.** It is a
label carried into ``reason`` for a human, not a gate. That is stated
here because the same shape has been a real defect twice elsewhere in
this repository: ``Rejection.LOOK_AHEAD`` recorded a leaked reading that
no consumer ever read, and the dashboard referenced two assets that no
route served. A flag with no consumer is a note to self, so a caller who
wants truncation refused must check it explicitly. Until one does, the
guarantee is "a typo is labelled", not "a typo cannot resolve".
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


def _symbol_run(
  lowered: str,
  words: tuple[str, ...],
) -> tuple[str, ...] | None:
  '''Return the query tokens left over once a run spells ``lowered``.

  The symbol layer is one directional: it fires on a query *shorter*
  than the symbol, which is a truncated query naming a longer name. When
  the query is *longer*, the symbol is only a fragment of it, and a
  fragment on its own is not enough -- but a whole **run** of tokens that
  spells the symbol is, because then the characters before the symbol
  are a prefix of nothing and the ones after it have to be accounted for
  by :func:`_name_corroborates`.

  The run must start at the first token. A run further in cannot spell a
  symbol that ``compact`` *starts* with, since ``compact`` is the tokens
  concatenated in order, so searching anywhere else would only admit a
  query whose leading token is not part of the symbol at all.

  Args:
    lowered: The squashed symbol, lowercased, e.g. ``'ongc'``.
    words: Query tokens with corporate suffixes stripped.

  Returns:
    The tokens after the run, or None when no leading run spells
    ``lowered``. An empty tuple means the whole query spelled the symbol,
    which the exact-symbol layer has already answered by then.
  '''
  joined = ''
  for count, word in enumerate(words, start=1):
    joined += word
    if joined == lowered:
      return words[count:]
    if len(joined) >= len(lowered):
      break
  return None


def _name_corroborates(
  words: tuple[str, ...],
  name_words: tuple[str, ...],
) -> bool:
  '''Return whether the company's own name accounts for every word.

  This is the second opinion the symbol layer needs before it will fire
  on a query longer than the symbol. Each leftover query token must be a
  prefix of a name word, and the pairing must be **one to one**: a
  distinct name word per token, because one word vouching for three
  leftovers is not corroboration, it is a coincidence counted three
  times.

  Matching is any-order, not positional. ONGC's name words start with
  ``'oil'`` while the query tokens are ``('ongc', 'oil')``, so a
  positional pairing would test ``'ongc'`` against ``'oil'`` and fail on
  the very case this rule exists to admit. Order carries no evidence
  about identity, and pairing one to one already stops a single name
  word from being reused.

  The name layer's own *evidence* still pairs positionally with
  :func:`zip`, and must. Any-order there made its truncation ratio
  compare a quantity with itself, which silently stopped reporting
  truncations; order-free matching is confined to this yes/no coverage
  question, where no ratio is derived from it.

  Args:
    words: Leftover query tokens, i.e. what the symbol run did not spell.
    name_words: The company's name tokens with corporate suffixes
      stripped, e.g. ``('oil', 'and', 'natural', 'gas')`` for ONGC.

  Returns:
    True when every token can be paired with a distinct name word it is
    a prefix of. True for an empty tuple of leftovers, which is the case
    the symbol run alone explains.
  '''
  paired: dict[int, int] = {}

  def place(word: int, seen: set[int]) -> bool:
    '''Assign one query token to a name word, displacing if needed.

    Args:
      word: Index into ``words`` of the token to place.
      seen: Name-word indices already tried on this search path, so a
        displacement cannot loop.

    Returns:
      True when the token found a name word, having moved whoever held
      it to a different one.
    '''
    for index, name_word in enumerate(name_words):
      if index in seen or not name_word.startswith(words[word]):
        continue
      seen.add(index)
      if index not in paired or place(paired[index], seen):
        paired[index] = word
        return True
    return False

  return all(place(word, set()) for word in range(len(words)))

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
    reaches the symbol ``'ONGC'`` and its own name word ``'oil'``, while
    ``'Reliance Infra'`` is settled by the alias table before this is
    reached.

    The symbol layer is **one directional**. A query shorter than the
    symbol fires on the fragment; a query longer than it fires only on
    corroboration from the name layer, via :func:`_symbol_run` and
    :func:`_name_corroborates`. Both halves of that are load bearing and
    neither is decorative:

    * :func:`_symbol_run` requires a whole run of tokens to spell the
      symbol, which is what refuses ``'itcxyz'``, where only a
      *fragment* of the query spells ``'itc'``;
    * :func:`_name_corroborates` requires the leftover tokens to be
      words of this same company, which is what refuses ``'ITC-INFRA'``,
      where ``'infra'`` is Reliance Infrastructure's word.

    Note that the name layer's own *evidence* still pairs
    **positionally** with :func:`zip`, and must. Any-order pairing there
    made its truncation ratio compare a quantity with itself and
    silently stopped reporting truncations; order-free matching is
    confined to the corroboration question, where it is a yes/no about
    coverage and no ratio is derived from it.

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
      name_words = self._name_words[symbol]
      if lowered == compact:
        found.append((0, _spelled))
      elif lowered.startswith(compact):
        found.append((1, _truncated('symbol', len(compact), len(lowered))))
      elif compact.startswith(lowered):
        # The query EXTENDS the symbol, so this is the inverse of a
        # prefix test and the direction that used to answer 'ITC-INFRA'
        # with ITC. It fires only when a whole run of tokens spells the
        # symbol and the company's own name accounts for the leftovers.
        # Both halves are load bearing and neither is decorative: without
        # the run, 'itcxyz' matches on the fragment 'itc'; without the
        # name, 'ITC-INFRA' matches on 'itc' with 'infra' left over,
        # which is Reliance Infrastructure's word and not this
        # company's. Nothing here compares lengths, which is why the
        # longer symbol wins without a tiebreak: 'reliance infra'
        # corroborates RELIANCEINFRA and fails for RELIANCE.
        rest = _symbol_run(lowered, words)
        if rest is not None and _name_corroborates(rest, name_words):
          found.append((1, _truncated('symbol', len(lowered), len(compact))))
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

