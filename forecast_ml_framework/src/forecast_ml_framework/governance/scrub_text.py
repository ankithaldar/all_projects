#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Text and composite secret scrubbing.

Free-text scanning, span/alignment hybrids, and
composite marker detection live here, built on
``scrub_keys`` (markers, confusables, key
normalization, safe helpers). The decoding leaf
for bytes lives here because it delegates to
``_scrub_text``. ``scrub`` re-exports every name
so the public leaf API is unchanged.
'''

import itertools
import re
import sys

from forecast_ml_framework.governance.scrub_keys import (
    _AWS_ACCESS_KEY_RE,
    _BARE_SCHEME_TEXT_RE,
    _BEARER_TEXT_RE,
    _CONFUSABLE_TABLE_ALT,
    _GLUED_PAREN_DIGIT_RE,
    _MARKER_RE,
    _MAX_ITEMS,
    _MAX_KEY_LEN,
    _MAX_TEXT_LEN,
    _NORMALIZED_MARKERS,
    _PEM_BLOCK_RE,
    _QUOTE_PAIRS,
    _SK_KEY_RE,
    _TERMINATORS,
    _TRUNCATED_NOTE,
    _URL_CRED_RE,
    REDACTED,
    _hybrid_contains,
    _hybrid_exact,
    _hybrid_key_reading,
    _hybrid_units_from_original,
    _is_ly_adverb,
    _is_secret_key,
    _normalize_key,
    _normalize_unicode,
    _safe_list,
    _safe_repr,
    _strip_glued_filler,
)

def _scrub_type_name(name):
  '''Scrub a type name for ``{'type': ...}`` sinks.

  Loop17 P1/H3/F1: a dynamic type name is
  attacker-controlled. Marker patterns are scrubbed
  via ``_scrub_text`` (``password=***REDACTED***``).
  F1: ``_scrub_text`` returns the normalized name
  even with no redaction, so compare against the
  normalized form -- if ``REDACTED`` is present
  return the scrubbed name, else run the token-like
  checks on the normalized name regardless of a
  normalization-only difference (``áZq...``,
  fullwidth, combining). H3: only a CamelCase
  allowlist name (``_*[A-Z][A-Za-z0-9_]*``) is
  preserved; a lowercase-start name (``zqsentinel``)
  or a symbol name is ``REDACTED`` whole. A long
  all-caps run (``SENTINEL``) or a digit in an
  all-lower/all-upper token is token-like, so it is
  redacted whole. Plain CamelCase names, including
  digit-bearing ones such as ``_Loop8EvilKeyPw``,
  are preserved.
  '''
  scrubbed = _scrub_text(name)
  if REDACTED in scrubbed:
    return scrubbed
  normalized = _normalize_unicode(name)
  if re.fullmatch(r'_*[A-Z][A-Za-z0-9_]*', normalized) is None:
    return REDACTED
  if re.search(r'[A-Z]{5,}', normalized):
    return REDACTED
  if re.search(r'[0-9]', normalized) and re.fullmatch(
        r'[a-z0-9]+|[A-Z0-9]+', normalized):
    return REDACTED
  return normalized


def _unknown_key(key):
  '''Return ``(placeholder, secret)`` for an unknown key.

  Unknown keys always become a type-only placeholder
  ``<TypeName>``, never their repr (like unknown values):
  keeping a scrubbed repr leaks when normalization alone
  changes a character without redacting (``áZq``,
  fullwidth, zero-width, combining) -- the normalized
  bare secret is not a marker redaction. ``secret`` is
  decided by whether scrubbing redacted beyond
  normalization (scrubbed differs from the normalized
  repr) or the repr is marker-bearing, so the associated
  value is redacted. The type name itself is scrubbed
  (loop17 P1) and a scrubbed type name is secret-bearing.
  Unrepresentable keys stay ``<unrepresentable>`` and are
  secret-bearing.
  '''
  safe = _safe_repr(key)
  if safe == '<unrepresentable>':
    return '<unrepresentable>', True
  scrubbed = _scrub_text(safe)
  normalized = _normalize_unicode(safe)
  type_name = type(key).__name__
  scrubbed_name = _scrub_type_name(type_name)
  secret = (
      scrubbed != normalized or REDACTED in scrubbed
      or _is_secret_key(safe) or scrubbed_name != type_name
      or _is_secret_key(type_name))
  return f'<{scrubbed_name}>', secret


def _scan_quoted(text, start):
  '''Return ``(end, closed)`` for a quoted string at ``start``.

  Handles escaped and doubled quotes inside ``"..."`` /
  ``'...'`` / backtick values; curly quotes have no doubling.
  An unterminated quote captures to end of string.
  '''
  opening = text[start]
  closing = _QUOTE_PAIRS[opening]
  index = start + 1
  while index < len(text):
    char = text[index]
    if char == '\\' and opening in ('"', '\'', '`'):
      index += 2
      continue
    if (char == opening and opening in ('"', '\'', '`')
        and index + 1 < len(text) and text[index + 1] == opening):
      index += 2
      continue
    if char == closing:
      return index + 1, True
    index += 1
  return len(text), False


_NEW_PAIR_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-.]*[=:][A-Za-z0-9]')

_FILLER_WORDS = frozenset({
    'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'definitely', 'very', 'really', 'absolutely', 'truly',
    'completely', 'utterly', 'totally', 'certainly', 'precisely',
    'exactly', 'way', 'well', 'well-known', 'wellknown',
    'just', 'actually', 'basically', 'one', 'two', 'three',
    'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
    'a', 'b', 'c', 'd', 'e', 'f', 'filler', 'very2', '123',
    'api', 'key', 'access', 'private', 'pass', 'word', 'aws',
    'encryption', 'master', 'signing', 'gcp', 'otp',
    'pin', 'number', 'code', 'phrase',
    'hash', 'hashes', 'value', 'values', 'id', 'ld', 'lds',
    # Loop14 F3: plural marker parts (``api keys``,
    # ``pass words``, ``aws access key ids``) are filler
    # like their singular forms, so the secret after a
    # plural phrase is the value, not the plural word.
    'keys', 'words', 'ids', 'tokens', 'secrets', 'passwords',
    'credentials', 'passphrases',
    # Loop16: ``well-known`` scans as ``well`` + ``known``
    # once tokens split on any non-alphanumeric, so
    # ``known`` is filler like its ``well-known`` whole.
    'known',
    # Loop17 P3: common degree adverbs (``password pretty
    # Z``, ``password is pretty much Z``) are filler like
    # ``very``/``really``.
    'pretty', 'quite', 'so', 'too', 'somewhat', 'extremely',
    'rather', 'super', 'much',
    # Loop17-arch H1: ``tremendously``/``literally``/
    # ``honestly`` (and the ``literaly`` spelling) are
    # filler like ``very``; any other ``-ly`` adverb is
    # filler via ``_is_ly_adverb`` in ``_is_filler``.
    # Loop18 P3: ``highly``/``incredibly``/``fairly``/
    # ``particularly``/``especially`` are filler too.
    'tremendously', 'literally', 'literaly', 'honestly',
    'highly', 'incredibly', 'fairly', 'particularly',
    'especially',
    # Cycle13 S14: HTTP auth schemes are filler between a
    # marker (``Authorization``) and its credential, so the
    # token after the scheme is the value (``Authorization:
    # Basic Z`` redacts ``Z``). Only in marker-scan context:
    # prose without a marker never reaches ``_is_filler``.
    'basic', 'digest', 'negotiate'})

# Loop15 B1: parts of a glued marker phrase (``api*keys``,
# ``aws*access*key``). A word whose alnum runs are all in
# this set and whose joining contains a normalized marker
# is a marker phrase, not a secret value.
_MARKER_PHRASE_PARTS = frozenset({
    'api', 'aws', 'pass', 'access', 'private', 'key', 'keys',
    'hash', 'hashes', 'value', 'values', 'token', 'tokens',
    'secret', 'secrets', 'password', 'passwords', 'passwd',
    'pwd', 'passphrase', 'passphrases', 'word', 'words',
    'encryption', 'master', 'signing', 'gcp', 'otp',
    'pin', 'number',
    'phrase', 'code',
    'id', 'ids', 'ld', 'lds', 'auth', 'bearer', 'cookie',
    'credential', 'credentials', 'authorization'})


def _is_marker_phrase_word(bare):
  '''Return True when ``bare`` is a glued marker phrase.

  Linear scan: alnum runs in ``bare`` must all be marker
  parts (or filler, H2) and their joining must contain
  a normalized marker, so a glued value merely
  containing a marker substring (``secretHUNTER2``)
  is not a marker phrase.
  '''
  parts = re.findall(r'[A-Za-z0-9]+', bare.lower())
  if not parts:
    return False
  if any(part not in _MARKER_PHRASE_PARTS
         and part not in _FILLER_WORDS
         and not _is_ly_adverb(part) for part in parts):
    return False
  joined = _strip_glued_filler(''.join(parts))
  return any(marker in joined for marker in _NORMALIZED_MARKERS)


def _is_filler(word):
  '''Return True when ``word`` is filler between marker/value.

  Loop16: the separator between filler parts is any
  non-alphanumeric (``[^A-Za-z0-9]``), mirroring ``_MARKER``
  (loop15 B1), so ``very*really`` / ``is=was`` are filler
  chains, not single tokens. ``word`` is split on that
  separator -- the same stripping as ``_MARKER`` and
  ``_is_marker_phrase_word`` -- and is filler only when
  every part is a known filler word. Loop17-arch H1:
  a ``-ly`` adverb part (``tremendously``, ``absurdly``)
  is filler even outside the vocab; this stays
  best-effort for non-``ly`` fillers in ``_scan_value``.
  '''
  normalized = word.lower().strip(
      ' \t\r\n"\'`“”‘’,.;:!?()[]{}=<>-_/\\|~#')
  if normalized in _FILLER_WORDS or _is_ly_adverb(normalized):
    return True
  parts = re.findall(r'[a-z0-9]+', word.lower())
  return bool(parts) and all(
      part in _FILLER_WORDS or _is_ly_adverb(part)
      for part in parts)


def _scan_value(text, start, pos=None):
  '''Scan forward from ``start`` for the secret value token.

  Deterministic linear scan: separators (whitespace,
  ``:=``/``--``/``-``, any symbolic run such as ``>>``/``~``/
  ``#``, underscores) and known filler words (copulas,
  adverbs, marker parts, filler-chain words) are consumed
  in order until the first value token. The value is that
  first non-filler word/quoted token -- not the last token
  scanned -- so trailing prose after the value (``password
  HUNTER2 end``) and a trailing pair (``password=QZ
  other=1``) are preserved. A quote adjacent to the
  marker/another token is a closing quote of a key name
  (``{"password": ...}``), not an opening quote, and is
  skipped as a separator. An empty or whitespace-only
  quoted span is likewise a separator, not a value, so
  scanning continues for the first non-empty value token
  (``password "" HUNTER2`` redacts ``HUNTER2``). An
  inserted ``REDACTED`` placeholder (from an earlier
  sub in this pass, e.g. Bearer consuming a compound
  marker word; ``pos`` is ``None`` for its chars) is
  likewise a separator, not a value, so scanning
  continues for the real secret after it. A
  user-typed ``***REDACTED***`` (``pos`` set) is a
  normal value token. A filler token carrying its own
  marker or a following ``key=value`` pair starts a new
  secret and returns ``None`` here, so it is handled by
  its own marker. Loop17-arch H1 best-effort: the
  filler vocab (plus ``-ly`` adverbs) is finite, so a
  non-``ly`` filler outside it is the first value
  token and is redacted itself.

  Returns:
    ``(repl_start, repl_end, replacement, token_end)`` or
    ``None`` when no value token follows.
  '''
  size = len(text)
  index = start
  last_end = start
  seen_separator = False
  while index < size:
    char = text[index]
    if char in _TERMINATORS:
      # Loop14 F2: a terminator glued to the marker end
      # (or with only spaces in between, as in the
      # compound marker ``Bearer token, Z`` where the
      # marker scan starts at ``,``) is a separator
      # before the value, not the end of the scan: skip
      # it and keep looking for the value. A terminator
      # after other content (e.g. a closing quote in
      # ``{"password", "user"}``) still ends the scan.
      if not text[start:index].strip():
        seen_separator = True
        index += 1
        continue
      break
    if char in _QUOTE_PAIRS:
      kind, payload = _scan_quoted_branch(
        text, index, last_end, seen_separator)
      if kind == 'continue':
        index = payload
        last_end = index
        seen_separator = True
        continue
      return payload
    if char.isspace() or char == '_':
      seen_separator = True
      index += 1
      continue
    if not char.isalnum():
      seen_separator = True
      index = _skip_symbol_run(text, index)
      continue
    if not seen_separator:
      kind, payload = _scan_glued_branch(text, index)
      if kind == 'continue':
        # The glued suffix is a maximal alnum run, so
        # the next token cannot be glued to it: it is
        # separated (space, quote, terminator, ...).
        index = payload
        last_end = index
        seen_separator = True
        continue
      return payload
    end = _scan_token_end(text, index)
    # Punctuation attached to a filler word (``very,``)
    # belongs to that word, not a terminator.
    while end < size and text[end] in ',.;!?':
      end += 1
    kind, payload = _scan_word_branch(text, index, end, pos)
    if kind == 'continue':
      index = payload
      last_end = index
      continue
    return payload
  return None


def _scan_token_end(text, index):
  '''Return the end of the token starting at ``index``.

  Loop16: the token is one alphanumeric run. Any
  non-alphanumeric separator (``[^A-Za-z0-9]``, e.g.
  ``*``/``|``/``=``/``_``/``-``) ends the token, mirroring
  the intra-marker separator in ``_MARKER`` (loop15 B1),
  so a filler chain glued by a symbol (``very*really``,
  ``is=was``) scans as separate filler words instead of
  one token that fails the filler check and is mistaken
  for the secret value.
  '''
  size = len(text)
  end = index
  while end < size and text[end].isalnum():
    end += 1
  return end


def _scan_full_token_end(text, index):
  '''Return the end of the whole glued token at ``index``.

  Loop16: this is the pre-loop16 ``_scan_token_end``
  boundary (whitespace, terminator or quote). Filler and
  new-pair checks use the split alphanumeric run from
  ``_scan_token_end``, but a token decided to be the
  value itself is redacted whole (``tok-123``,
  ``Zq...|MORE``), so splitting cannot leave a glued
  value suffix behind.
  '''
  size = len(text)
  end = index
  while (end < size and not text[end].isspace()
      and text[end] not in _TERMINATORS
      and text[end] not in _QUOTE_PAIRS):
    end += 1
  return end


def _skip_symbol_run(text, index):
  '''Skip a run of symbolic separator chars at ``index``.'''
  size = len(text)
  while (index < size and not text[index].isalnum()
      and not text[index].isspace()
      and text[index] not in _TERMINATORS
      and text[index] not in _QUOTE_PAIRS
      and text[index] != '_'):
    index += 1
  return index


def _trim_value_end(text, index, end):
  '''Trim trailing ``,.;!?`` from a scanned value end.'''
  value_end = end
  while value_end > index and text[value_end - 1] in ',.;!?':
    value_end -= 1
  return value_end


def _extend_past_terminators(text, end):
  '''Extend ``end`` past spaced terminators.

  Loop14 F2: after a skipped filler/marker word, a
  terminator with only spaces in between (``(token)
  , Z``) is likewise a separator before the value.
  Spaces not followed by a terminator are left for
  the main scan loop.
  '''
  size = len(text)
  while end < size:
    look = end
    while look < size and text[look].isspace():
      look += 1
    if look < size and text[look] in ')]};,':
      while look < size and text[look] in ')]};,':
        look += 1
      end = look
    else:
      break
  return end


def _scan_quoted_branch(text, index, last_end, seen_separator):
  '''Handle a quote at ``index`` (loop14 split).

  Returns ``('continue', new_index)`` when the quote
  is a separator, else ``('return', result)``.
  '''
  if index == last_end:
    return ('continue', index + 1)
  if not seen_separator:
    return ('return', None)
  end, closed = _scan_quoted(text, index)
  inner_end = end - 1 if closed else end
  if not text[index + 1:inner_end].strip():
    return ('continue', end)
  # Loop14 F1: a quoted marker (``Bearer "token" Z``)
  # is filler, not the value: redacting it would resume
  # the search after it and leak the real secret. Skip
  # it like a bare marker word and keep scanning. The
  # check is exact (filler, normalized marker, or a
  # whole marker match), so a quoted value merely
  # containing a marker substring still redacts.
  inner = text[index + 1:inner_end].strip()
  if (_is_filler(inner)
      or _normalize_key(inner) in _NORMALIZED_MARKERS
      or _MARKER_RE.fullmatch(inner)):
    return ('continue', _extend_past_terminators(text, end))
  opening = text[index]
  if closed and end - index >= 2:
    replacement = opening + REDACTED + _QUOTE_PAIRS[opening]
  else:
    replacement = opening + REDACTED
  return ('return', (index, end, replacement, end))


def _short_glued_continue_end(text, suffix_end):
  '''Return a continue end for a short glued suffix.

  Loop21 F1: a short suffix (<= 3 chars: ``X``/``1``,
  ``Hu``/``AB``, ``ab1``) is a marker extension, not
  the value, so scanning continues for the real value
  after a separator (space, quote, terminator, ``_``,
  or a symbol run such as ``:`` in ``passwordX: Z``).
  A symbol run landing directly on an alnum token
  (``passwordX=Z``) is a glued ``key=value`` token:
  return ``None`` so the caller redacts the full
  token instead.
  '''
  size = len(text)
  if suffix_end >= size:
    return suffix_end
  char = text[suffix_end]
  if (char.isspace() or char in _QUOTE_PAIRS
          or char in _TERMINATORS or char == '_'):
    return _extend_past_terminators(text, suffix_end)
  after_symbols = _skip_symbol_run(text, suffix_end)
  if after_symbols < size and text[after_symbols].isalnum():
    return None
  return _extend_past_terminators(text, after_symbols)


def _scan_glued_branch(text, index):
  '''Handle a glued suffix at ``index`` (loop14 split).

  A glued suffix directly after the marker (no
  separator, e.g. ``passwordZq9SENTINEL``) is the
  value: redact the suffix. Known marker extensions
  (``Id`` in ``accessKeyId``; ``ld`` is its alt
  capital-I-to-l reading), filler words glued to the
  marker, and a single Capitalized word (the ``Thing``
  in a type name such as ``SecretThing``) are part of
  the marker phrase, not a value, so consume them and
  keep scanning. Loop21 F1: a short suffix (<= 3
  chars, see ``_short_glued_continue_end``) is such
  an extension too, except in a glued ``key=value``
  token. Returns ``('continue', suffix_end)`` or
  ``('return', result)``.
  '''
  size = len(text)
  suffix_end = index
  while suffix_end < size and text[suffix_end].isalnum():
    suffix_end += 1
  suffix = text[index:suffix_end]
  if suffix.lower() in ('id', 'ids', 's', 'ld', 'lds'):
    return ('continue', _extend_past_terminators(
        text, suffix_end))
  if _is_filler(suffix):
    return ('continue', _extend_past_terminators(
        text, suffix_end))
  if re.fullmatch(r'[A-Z][a-z]{2,}', suffix or ''):
    return ('continue', _extend_past_terminators(
        text, suffix_end))
  if len(suffix) <= 3:
    continue_end = _short_glued_continue_end(text, suffix_end)
    if continue_end is not None:
      return ('continue', continue_end)
  end = _scan_full_token_end(text, index)
  value_end = _trim_value_end(text, index, end)
  return ('return', (index, value_end, REDACTED, value_end))


def _scan_word_branch(text, index, end, pos):
  '''Handle a separated word at ``index`` (loop14 split).

  Returns ``('continue', end)`` for filler/inserted
  placeholders, else ``('return', result)``.
  Loop14 F2: filler and exact-marker words continue
  past a closing bracket/terminator glued to them
  (``)``/``]``/``}``/``;``/``,``), so a bracketed or
  terminated marker (``(token)``, ``token,``) is
  skipped as filler and scanning reaches the real
  secret after it instead of stopping at the
  terminator in the inner marker scan.
  '''
  skip_end = _extend_past_terminators(text, end)
  word = text[index:end]
  if (pos is not None
      and word.strip('*.,;!?') == 'REDACTED'
      and all(pos[idx] is None for idx in range(index, end))):
    # Loop17 P4: an inserted placeholder likewise continues
    # past a terminator, so a Bearer-redacted filler before
    # a bracket does not stop the scan before the secret.
    return ('continue', skip_end)
  if _is_filler(word):
    return ('continue', skip_end)
  # A token carrying its own marker or a following
  # ``key=value`` / ``key:value`` pair starts a new
  # secret: no value for this marker (``other=1`` after
  # ``password=QZ`` is never reached, as QZ returns first).
  # Only a token with an inline value after ``=``/``:``
  # (``other=1``, ``other:1``) counts, so filler words
  # glued to a separator (``definitely: hunter2``) are
  # not mistaken for a new key -- those passed _is_filler.
  # A glued value whose word merely contains a marker
  # substring (``secretHUNTER2`` after ``password=``) is
  # the value itself: redact the whole glued word, or
  # the secret leaks. A new ``key=value`` pair still
  # starts a new secret; an exact marker word is
  # skipped as filler (see below).
  # Loop16: the pair test uses the whole glued token,
  # but a filler chain (``is=was``) has already passed
  # ``_is_filler`` above, so ``_NEW_PAIR_RE`` cannot
  # swallow filler+sep+filler as a new secret.
  full_end = _scan_full_token_end(text, index)
  if _NEW_PAIR_RE.match(text[index:full_end]):
    # Cycle1 W1 F1: defer only when the new pair key is
    # itself secret-bearing; a non-secret key (``session``,
    # ``user``, ``data``, ``foo``) is this marker's value.
    pair_token = text[index:full_end]
    pair_key = re.split(r'[=:]', pair_token, maxsplit=1)[0]
    if (_is_secret_key(pair_key)
        or _MARKER_RE.search(pair_key)):
      return ('return', None)
    value_end = _trim_value_end(text, index, full_end)
    return ('return', (index, value_end, REDACTED, value_end))
  marker_match = _MARKER_RE.search(word)
  if marker_match is not None:
    bare = word.strip(' \t\r\n"\'`“”‘’,.;:!?()[]{}=<>-_/\\|~#')
    if marker_match.group(0).lower() == bare.lower():
      # Exact marker word: skip it as filler (F2)
      # rather than deferring to its own scan, which
      # stops at a following terminator/bracket.
      return ('continue', skip_end)
    if _is_marker_phrase_word(bare):
      # Loop15 B1: glued plural/aws marker phrase
      # (``api*keys``, ``aws*access*key``) is likewise
      # a marker, not this marker's value.
      return ('continue', skip_end)
    # Glued marker substring: fall through and redact
    # the whole word as this marker's value.
  value_end = _trim_value_end(text, index, full_end)
  return ('return', (index, value_end, REDACTED, value_end))


def _is_base64ish_token(token):
  '''Return True for a bare-scheme credential-shaped token.

  Cycle13 S14: a bare ``Basic``/``Digest``/``Negotiate``
  redacts only a token of length >= 8 carrying a letter
  and a digit or ``+/=`` (base64-ish), so ordinary prose
  (``basic math``, ``Basic research``) is preserved. The
  ``Authorization`` context is checked in the marker scan
  instead and redacts any token after the scheme.
  '''
  return (len(token) >= 8
          and any(char.isalpha() for char in token)
          and any(char.isdigit() or char in '+/=' for char in token))


def _sub_spans(text, pos, redacted, pattern, mode):
  '''Apply one regex, recording its core span in ``redacted``.

  ``pos`` maps each char in ``text`` to its index in the
  original input (``None`` for inserted replacement
  chars), so a span recorded at any stage is in original
  coordinates even though earlier replacements shift
  lengths (``ß``/``þ``/Roman/squared expansions and
  ``REDACTED`` itself). ``mode`` is ``'whole'``,
  ``'url'``, ``'bearer'`` or ``'scheme'``: the core is
  the whole match, URL group 2, Bearer group 2, or a
  scheme token passing ``_is_base64ish_token`` (a scheme
  match failing that shape is kept verbatim), and
  preserved prefix/suffix chars keep their original
  positions.
  Replacement text and output are byte-identical to the
  equivalent ``re.sub`` (except a kept scheme match).
  '''
  parts = []
  new_pos = []
  last = 0
  for match in pattern.finditer(text):
    parts.append(text[last:match.start()])
    new_pos.extend(pos[last:match.start()])
    if mode == 'whole':
      core = (match.start(), match.end())
    elif mode == 'url':
      parts.append(text[match.start(1):match.end(1)])
      new_pos.extend(pos[match.start(1):match.end(1)])
      parts.append(REDACTED)
      new_pos.extend([None] * len(REDACTED))
      parts.append(text[match.start(3):match.end(3)])
      new_pos.extend(pos[match.start(3):match.end(3)])
      core = match.span(2)
    elif mode == 'scheme' and not _is_base64ish_token(
        match.group(2)):
      parts.append(text[match.start():match.end()])
      new_pos.extend(pos[match.start():match.end()])
      core = None
    else:
      parts.append(text[match.start(1):match.end(1)])
      new_pos.extend(pos[match.start(1):match.end(1)])
      parts.append(REDACTED)
      new_pos.extend([None] * len(REDACTED))
      core = match.span(2)
    if mode == 'whole':
      parts.append(REDACTED)
      new_pos.extend([None] * len(REDACTED))
    if core is not None:
      for idx in range(core[0], core[1]):
        if pos[idx] is not None:
          redacted.add(pos[idx])
    last = match.end()
  parts.append(text[last:])
  new_pos.extend(pos[last:])
  return ''.join(parts), new_pos


def _spans_from_set(redacted):
  '''Return sorted contiguous ``(start, end)`` spans.'''
  spans = []
  for idx in sorted(redacted):
    if spans and idx == spans[-1][1]:
      spans[-1] = (spans[-1][0], idx + 1)
    else:
      spans.append((idx, idx + 1))
  return spans


_TOKEN_RE = re.compile(r'[A-Za-z0-9]+')

# Loop20 two-pass: compound markers whose first part is
# not a single marker (``api``/``aws``/``access``/
# ``private``/``pass``). Longest first. The regex
# ``_MARKER`` handles these for ordinary filler, but
# its atomic filler traps ``-ly`` tokens starting
# a-f (``a`` matches one letter and commits) and, in
# general, a deterministic token scan is the linear
# fallback for arbitrarily many filler tokens and
# ``-ly`` tokens of any length (whole-token
# ``_is_filler``/``_is_ly_adverb``).
_COMPOUND_PATTERNS = (
    ('aws', 'secret', 'access', 'key'),
    ('aws', 'access', 'key'),
    ('secret', 'access', 'key'),
    ('encryption', 'key'),
    ('signing', 'key'),
    ('access', 'key'),
    ('private', 'key'),
    ('master', 'key'),
    ('api', 'key'),
    ('aws', 'key'),
    ('gcp', 'key'),
    ('pass', 'word'),
    ('pass', 'phrase'),
    ('pass', 'code'),
    ('pin', 'code'),
    ('pin', 'number'),
)


def _compound_tokens(text):
  '''Return ``(start, end, lower)`` alnum tokens.'''
  return [
      (match.start(), match.end(), match.group().lower())
      for match in _TOKEN_RE.finditer(text)]


def _match_compound_at(tokens, index):
  '''Return marker end for a compound at ``index``.'''
  for pattern in _COMPOUND_PATTERNS:
    if pattern[0] != tokens[index][2]:
      continue
    pos_index = index
    marker_end = None
    matched = True
    for part_index, part in enumerate(pattern):
      if part_index:
        while (pos_index < len(tokens)
               and tokens[pos_index][2] != part
               and _is_filler(tokens[pos_index][2])):
          pos_index += 1
      if (pos_index >= len(tokens)
          or tokens[pos_index][2] != part):
        matched = False
        break
      marker_end = tokens[pos_index][1]
      pos_index += 1
    if matched:
      return marker_end
  return None


def _find_compound_marker(text, tokens, search_from, start_index=0):
  '''Return ``(start, end)`` of next compound, or None.

  Args:
    text: Scrubbed text (unused, kept for call-site symmetry).
    tokens: Alnum tokens from ``_compound_tokens``.
    search_from: Only compounds at or after here count.
    start_index: Resume token index; tokens before it are
      never rescanned, so marker-dense text stays linear.

  Returns:
    ``(start, end)`` of the next compound, or ``None`` when
    the scan reaches the end of ``tokens``.
  '''
  for index in range(start_index, len(tokens)):
    token = tokens[index]
    if token[1] <= search_from or token[0] < search_from:
      continue
    marker_end = _match_compound_at(tokens, index)
    if marker_end is not None:
      return (token[0], marker_end)
  return None


def _scrub_text_inner_spans(text):
  '''Scrub ``text`` and return ``(scrubbed, core_spans)``.

  Core spans are the original-input ranges replaced by
  ``REDACTED`` (quote/prefix/suffix chars preserved by a
  replacement are outside the core), so redactions from
  two skeletons of the same input can be unioned.
  '''
  pos = list(range(len(text)))
  redacted = set()
  text, pos = _sub_spans(
      text, pos, redacted, _PEM_BLOCK_RE, 'whole')
  text, pos = _sub_spans(
      text, pos, redacted, _AWS_ACCESS_KEY_RE, 'whole')
  text, pos = _sub_spans(
      text, pos, redacted, _SK_KEY_RE, 'whole')
  text, pos = _sub_spans(
      text, pos, redacted, _URL_CRED_RE, 'url')
  text, pos = _sub_spans(
      text, pos, redacted, _BEARER_TEXT_RE, 'bearer')
  text, pos = _sub_spans(
      text, pos, redacted, _BARE_SCHEME_TEXT_RE, 'scheme')
  parts = []
  cursor = 0
  search_from = 0
  tokens = _compound_tokens(text)
  token_index = 0
  while True:
    match = _MARKER_RE.search(text, search_from)
    compound = _find_compound_marker(
        text, tokens, search_from, token_index)
    if compound is None:
      # The scan reached the end: no compound exists at or
      # after search_from, and search_from only grows, so
      # later calls resume at the end instead of rescanning.
      token_index = len(tokens)
    else:
      while (token_index < len(tokens)
          and tokens[token_index][0] < compound[0]):
        token_index += 1
    if match is None and compound is None:
      break
    if compound is not None and (
        match is None or compound[0] < match.start()):
      marker_end = compound[1]
    else:
      marker_end = match.end()
    found = _scan_value(text, marker_end, pos)
    if found is None:
      search_from = marker_end
      continue
    repl_start, repl_end, replacement, token_end = found
    core_start, core_end = repl_start, repl_end
    if replacement != REDACTED:
      core_start = repl_start + 1
      if len(replacement) > len(REDACTED) + 1:
        core_end = repl_end - 1
    for idx in range(core_start, core_end):
      if pos[idx] is not None:
        redacted.add(pos[idx])
    parts.append(text[cursor:repl_start])
    parts.append(replacement)
    cursor = repl_end
    search_from = token_end
  parts.append(text[cursor:])
  return ''.join(parts), _spans_from_set(redacted)


_ALIGN_CANDIDATES = (
    (1, 1), (2, 1), (1, 2), (1, 0), (0, 1), (2, 2),
    (2, 0), (0, 2), (3, 1), (1, 3), (3, 2), (2, 3),
    (3, 0), (0, 3), (3, 3), (4, 1), (1, 4), (4, 0),
    (0, 4))


def _align_opcodes(primary, alt):
  '''Align two skeletons of one input, locally greedy.

  Both skeletons differ only at ambiguous glyphs
  (1-1, 2-1 or 1-2 substitutions), so a greedy walk
  that resynchronizes at the next equal char within a
  few positions aligns them positionally. A global
  matcher can instead jump to a repeated later token
  (``password`` appearing twice) and misalign spans.
  Returns ``(tag, i1, i2, j1, j2)`` opcodes.
  '''
  opcodes = []
  size_p = len(primary)
  size_a = len(alt)
  index_p = 0
  index_a = 0
  while index_p < size_p or index_a < size_a:
    run = _equal_run(primary, alt, index_p, index_a)
    if run is not None:
      start_p, index_p, start_a, index_a = run
      opcodes.append(
          ('equal', start_p, index_p, start_a, index_a))
      continue
    step_p, step_a = _choose_align_step(
      primary, alt, index_p, index_a, size_p, size_a)
    tag = ('replace' if step_p and step_a
           else 'delete' if step_a == 0 else 'insert')
    opcodes.append((tag, index_p, index_p + step_p,
                    index_a, index_a + step_a))
    index_p += step_p
    index_a += step_a
  return opcodes


def _equal_run(primary, alt, index_p, index_a):
  '''Return the equal run at the cursors, or None.

  Split from ``_align_opcodes`` (loop14). Returns
  ``(start_p, new_p, start_a, new_a)``.
  '''
  if not (index_p < len(primary) and index_a < len(alt)
          and primary[index_p] == alt[index_a]):
    return None
  start_p = index_p
  start_a = index_a
  while (index_p < len(primary) and index_a < len(alt)
         and primary[index_p] == alt[index_a]):
    index_p += 1
    index_a += 1
  return (start_p, index_p, start_a, index_a)


def _choose_align_step(primary, alt, index_p, index_a, size_p, size_a):
  '''Choose the next greedy align step (loop14 split).'''
  chosen = None
  for step_p, step_a in _ALIGN_CANDIDATES:
    next_p = index_p + step_p
    next_a = index_a + step_a
    if next_p > size_p or next_a > size_a:
      continue
    if next_p == size_p and next_a == size_a:
      chosen = (step_p, step_a)
      break
    if (next_p < size_p and next_a < size_a
        and primary[next_p] == alt[next_a]):
      chosen = (step_p, step_a)
      break
  if chosen is None:
    chosen = (1 if index_p < size_p else 0,
              1 if index_a < size_a else 0)
  return chosen


def _map_alt_span(primary, opcodes, span):
  '''Map an alt ``span`` to primary coordinates.

  Equal blocks map 1-1, replace blocks map as a whole
  (``VI``/``V``, ``r``/``a``), and deleted primary chars
  between mapped blocks are bridged by taking the min
  start and max end, so 1->2 expansions are included.
  A pure-insert span (no primary counterpart) expands to
  the next primary char.
  '''
  alt_start, alt_end = span
  mapped_start = None
  mapped_end = None
  for tag, i1, i2, j1, j2 in opcodes:
    block = _map_alt_span_block(
      tag, i1, i2, j1, j2, alt_start, alt_end)
    if block is None:
      continue
    start, end = block
    mapped_start = (
        start if mapped_start is None
        else min(mapped_start, start))
    mapped_end = (
        end if mapped_end is None
        else max(mapped_end, end))
  return _finalize_mapped_span(primary, mapped_start, mapped_end)


def _map_alt_span_block(tag, i1, i2, j1, j2, alt_start, alt_end):
  '''Map one opcode block to primary coords, or None.

  Split from ``_map_alt_span`` (loop14). Equal blocks
  map 1-1, replace blocks map as a whole, insert
  blocks map to their primary position, and delete
  blocks never map.
  '''
  if tag == 'delete':
    return None
  if tag == 'equal':
    low = max(alt_start, j1)
    high = min(alt_end, j2)
    if low < high:
      return (i1 + (low - j1), i1 + (high - j1))
    return None
  if tag == 'replace':
    if alt_start < j2 and j1 < alt_end:
      return (i1, i2)
    return None
  if alt_start < j2 and j1 < alt_end:
    return (i1, i1)
  return None


def _finalize_mapped_span(primary, mapped_start, mapped_end):
  '''Expand a pure-insert mapped span, or None.'''
  if mapped_start is None:
    return None
  if mapped_start == mapped_end:
    if mapped_start < len(primary):
      mapped_end = mapped_start + 1
    elif mapped_start > 0:
      mapped_start -= 1
    else:
      return None
  return (mapped_start, mapped_end)


def _union_spans(primary, alt, primary_spans, alt_spans):
  '''Union primary spans with alt spans mapped to primary.

  An alt span overlapping a primary span is skipped:
  both skeletons redacted the same secret and the
  primary span is the precise one (an alt ``pass``
  substring marker inside a primary ``password`` must
  not mangle the marker name).
  '''
  if not alt_spans:
    return list(primary_spans)
  opcodes = _align_opcodes(primary, alt)
  combined = list(primary_spans)
  for span in alt_spans:
    mapped = _map_alt_span(primary, opcodes, span)
    # An unmappable alt span must not leak: fall back to
    # redacting the whole primary text.
    if mapped is None:
      combined.append((0, len(primary)))
      continue
    if any(mapped[0] < end and start < mapped[1]
           for start, end in primary_spans):
      continue
    combined.append(mapped)
  merged = []
  for start, end in sorted(combined):
    if merged and start <= merged[-1][1]:
      merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    else:
      merged.append((start, end))
  return merged


def _apply_spans(text, spans):
  '''Return ``text`` with each span replaced by REDACTED.'''
  parts = []
  cursor = 0
  for start, end in spans:
    parts.append(text[cursor:start])
    parts.append(REDACTED)
    cursor = end
  parts.append(text[cursor:])
  return ''.join(parts)


def _build_hybrid_text(text):
  '''Return a hybrid skeleton for original ``text``.

  Per original character (``_char_hybrid_segments``):
  each alphanumeric token with ambiguous units is
  resolved to a marker-part reading when one exists
  (exact), otherwise to a marker-containing reading
  when one exists, otherwise kept as primary. Separator
  units keep their primary form. The result differs
  from primary only at ambiguous glyphs, so existing
  span alignment/union applies to it. An empty primary
  segment (zero-width, combining, format) does not
  break a token, and a unit whose alt segment is
  alphanumeric tokenises with its token too (``|`` is
  ``|`` in primary but ``l`` in alt): flushing at ``|``
  would split ``credentI<ya>|`` so no hybrid reading
  could form ``credential``. A glued parenthesized
  digit (``credent(1)al``) is folded to its digit first
  (loop14 F5, mirroring ``_normalize_unicode``), so
  the brackets do not split the token either.
  '''
  text = _GLUED_PAREN_DIGIT_RE.sub(r'\1', text)
  units = _hybrid_units_from_original(text)
  if all(primary_seg == alt_seg for primary_seg, alt_seg in units):
    return _normalize_unicode(text)
  hybrid = []
  token = []

  def flush_token():
    '''Flush the pending token into ``hybrid``.

    Resolves an ambiguous token to a marker-part or
    marker-containing reading when one exists, else
    keeps the primary form.

    Returns:
      None.
    '''
    if not token:
      return
    primary_token = ''.join(seg for seg, _ in token)
    if any(seg != alt_seg for seg, alt_seg in token):
      lowered = [
        (seg.lower(), alt_seg.lower())
        for seg, alt_seg in token]
      reading = None
      # Loop20 P4: skip targets the token cannot
      # produce (alphabet filter, see _hybrid_key_reading).
      possible = set()
      for seg, alt_seg in lowered:
        possible.update(seg)
        possible.update(alt_seg)
      for part in sorted(_MARKER_PARTS):
        if not set(part) <= possible:
          continue
        reading = _hybrid_exact(lowered, part)
        if reading is not None:
          break
      if reading is None:
        for marker in _NORMALIZED_MARKERS:
          if not set(marker) <= possible:
            continue
          reading = _hybrid_contains(lowered, marker)
          if reading is not None:
            break
      hybrid.append(
        reading if reading is not None else primary_token)
    else:
      hybrid.append(primary_token)
    token.clear()

  for primary_seg, alt_seg in units:
    if primary_seg.isalnum() or alt_seg.isalnum():
      token.append((primary_seg, alt_seg))
    elif not primary_seg and not alt_seg:
      continue
    else:
      flush_token()
      hybrid.append(primary_seg)
  flush_token()
  return ''.join(hybrid)


def _build_hybrid_text_via_facade(text):
  '''Build hybrid text via the facade override if any.

  Args:
    text: Raw text to build a hybrid skeleton from.

  Returns:
    A hybrid skeleton string. When the public
    ``scrub`` facade carries a patched
    ``_build_hybrid_text`` (tests spy on the
    facade name), that override is called;
    otherwise the local builder is called. The
    deferred facade lookup avoids a load-time
    cycle (``scrub`` imports this module).
  '''
  facade = sys.modules.get(
    'forecast_ml_framework.governance.scrub')
  builder = getattr(facade, '_build_hybrid_text', None)
  if builder is not None and builder is not _build_hybrid_text:
    return builder(text)
  return _build_hybrid_text(text)


def _scrub_text(text):
  '''Redact secret values in a free-text string.

  Args:
    text: String that may embed ``key=value`` secret patterns
      (or a bare ``Bearer``/HTTP scheme ``<token>``).

  Returns:
    The string with secret values replaced by ``REDACTED``.
    Strings longer than ``_MAX_TEXT_LEN`` are truncated
    in raw form before normalization (cycle1 W1 F5)
    and a ``[TRUNCATED]`` remainder note is appended.
    Both the primary and alt skeletons are
    scrubbed and their redaction spans are unioned (alt
    spans mapped to primary coordinates), so a string
    mixing a primary-only marker and an alt-only marker
    redacts both secrets instead of returning after the
    first skeleton that redacts anything. A hybrid
    skeleton (loop12 F1) is scrubbed too when it differs
    from both: per-token per-letter primary/alt choices
    (``_build_hybrid_text``) form markers that neither
    global skeleton forms, and its spans are unioned
    in the same way.
  '''
  if not isinstance(text, str):
    return text
  truncated = len(text) > _MAX_TEXT_LEN
  raw_text = text[:_MAX_TEXT_LEN] if truncated else text
  normalized = _normalize_unicode(raw_text)
  alt = _normalize_unicode(raw_text, _CONFUSABLE_TABLE_ALT)
  primary_in, alt_in, suffix = _scrub_text_inputs(
      normalized, alt, truncated)
  scrubbed, spans = _scrub_text_inner_spans(primary_in)
  if alt_in == primary_in:
    return scrubbed + suffix
  return _scrub_text_skeletons(
    raw_text, primary_in, alt_in, suffix, scrubbed, spans)


def _scrub_text_inputs(normalized, alt, truncated=False):
  '''Truncate skeletons to the text cap (loop14 split).

  The raw text is already truncated before normalization
  (cycle1 W1 F5), so a normalized skeleton can exceed
  the cap only when normalization expands characters;
  the suffix records either form of truncation.
  '''
  if len(normalized) > _MAX_TEXT_LEN:
    return (
      normalized[:_MAX_TEXT_LEN], alt[:_MAX_TEXT_LEN],
      REDACTED + _TRUNCATED_NOTE)
  if truncated:
    return (normalized, alt, REDACTED + _TRUNCATED_NOTE)
  return (normalized, alt, '')


def _scrub_text_hybrid(text, suffix, primary_in, alt_in):
  '''Scrub the hybrid skeleton; return (hit, spans, in).'''
  # Build the hybrid skeleton from truncated text when the
  # input is truncated: building it from the full text is
  # unbounded work (per-token DP over an over-long token)
  # before the _MAX_TEXT_LEN truncate below. When the input
  # is not truncated its normalized length is within the
  # cap, so the token DP is bounded by that length.
  hybrid_full = _build_hybrid_text_via_facade(
    text[:_MAX_TEXT_LEN] if suffix else text)
  hybrid_in = hybrid_full[:_MAX_TEXT_LEN] if suffix else hybrid_full
  if hybrid_in in (primary_in, alt_in):
    return (False, [], hybrid_in)
  hybrid_scrubbed, hybrid_spans = _scrub_text_inner_spans(hybrid_in)
  hybrid_hit = bool(hybrid_spans) or hybrid_scrubbed != hybrid_in
  return (hybrid_hit, hybrid_spans, hybrid_in)


def _combine_skeleton_results(
    primary_in, alt_in, suffix, scrubbed, spans, alt_scrubbed,
    alt_spans, primary_hit, alt_hit, hybrid_hit, hybrid_spans,
    hybrid_in):
  '''Union skeleton redactions and apply them (loop14).'''
  if hybrid_hit:
    union = list(spans)
    if alt_hit:
      union = _union_spans(primary_in, alt_in, union, alt_spans)
    union = _union_spans(
      primary_in, hybrid_in, union, hybrid_spans)
    return _apply_spans(primary_in, union) + suffix
  if primary_hit and alt_hit:
    union = _union_spans(primary_in, alt_in, spans, alt_spans)
    return _apply_spans(primary_in, union) + suffix
  if primary_hit:
    return scrubbed + suffix
  if alt_hit:
    return alt_scrubbed + suffix
  return scrubbed + suffix


def _scrub_text_skeletons(
    text, primary_in, alt_in, suffix, scrubbed, spans):
  '''Scrub alt/hybrid skeletons and combine (loop14).'''
  alt_scrubbed, alt_spans = _scrub_text_inner_spans(alt_in)
  primary_hit = bool(spans) or scrubbed != primary_in
  alt_hit = bool(alt_spans) or alt_scrubbed != alt_in
  hybrid_hit, hybrid_spans, hybrid_in = _scrub_text_hybrid(
    text, suffix, primary_in, alt_in)
  return _combine_skeleton_results(
    primary_in, alt_in, suffix, scrubbed, spans, alt_scrubbed,
    alt_spans, primary_hit, alt_hit, hybrid_hit, hybrid_spans,
    hybrid_in)


_MARKER_PARTS = frozenset({
    'api', 'key', 'access', 'private', 'pass', 'word', 'aws',
    'secret', 'token', 'password', 'passwd', 'pwd', 'passphrase',
    'phrase', 'code', 'hash', 'value',
    'encryption', 'master', 'signing', 'gcp', 'otp',
    'pin', 'number',
    'bearer', 'auth', 'cookie', 'credential', 'authorization'})

# Depth cap for composite flattening / marker detection, checked
# before recursing so a deeply nested or cyclic raw container
# cannot exhaust the call stack (mirrors the depth cap applied
# by ``telemetry.exceptions._coerce``).
_MAX_COMPOSITE_DEPTH = 10


def _is_marker_element(element):
  '''Return True when a key element is itself a marker word.'''
  return (isinstance(element, (str, bytes, bytearray))
          and _is_secret_key(element))


def _marker_part(element):
  '''Return the marker part ``element`` names, or ``None``.

  Both skeletons are tried (primary first): a part that
  only joins a marker in the alt skeleton (``\\u044fpi``
  -> ``api``) counts, matching ``_is_marker_element``.
  A per-letter hybrid reading is tried last (loop12 F1:
  ``ya-pI`` -> ``api``), matching ``_is_secret_key``.
  '''
  if not isinstance(element, (str, bytes, bytearray)):
    return None
  normalized = _normalize_key(element)
  if normalized in _MARKER_PARTS:
    return normalized
  alt = _normalize_key(element, _CONFUSABLE_TABLE_ALT)
  if alt in _MARKER_PARTS:
    return alt
  if isinstance(element, (bytes, bytearray)):
    text = bytes(element).decode('utf-8', errors='replace')
  else:
    text = element
  return _hybrid_key_reading(
    text, sorted(_MARKER_PARTS), exact=True)


def _flatten_composite(items, _seen=None, _depth=0):
  '''Yield str/bytes leaves of nested composites, in order.

  Recurses into tuple/list/frozenset/set at any depth so a
  marker nested inside a tuple (``(b'password',
  b'HUNTER2')`` inside another tuple) is seen by composite
  marker detection. Dicts are not traversed here: a dict
  with its own marker key (``{'token': S}``) already
  redacts its own value and must not taint sibling
  elements; dict marker-*parts* split with an outer
  composite (``('api', {'key': S})``) are collected by
  ``_ordered_parts`` instead. Cyclic containers yield
  ``'<cyclic>'`` once and are not re-entered (``_seen``
  holds ids on the current path, seeded by callers that
  already track a path); recursion stops at
  ``_MAX_COMPOSITE_DEPTH`` with ``'<max-depth>'`` before
  the call stack can be exhausted by a deeply nested raw
  container.
  '''
  if _seen is None:
    _seen = frozenset()
  if _depth > _MAX_COMPOSITE_DEPTH:
    yield '<max-depth>'
    return
  items_list = _safe_list(items)
  if items_list is None:
    yield '<unrepresentable>'
    return
  for item in items_list:
    if isinstance(item, (str, bytes, bytearray)):
      yield item
    elif isinstance(item, (tuple, list, frozenset, set)):
      if id(item) in _seen:
        yield '<cyclic>'
        continue
      nested = _safe_list(item)
      if nested is None:
        yield '<unrepresentable>'
        continue
      yield from _flatten_composite(
          nested, _seen | {id(item)}, _depth + 1)


def _ordered_parts(items, _seen=None, _depth=0):
  '''Yield marker parts in order, including dict keys.

  Direct tuple/list/set elements yield their part, dict
  keys yield their part only when it is not itself a full
  marker (a full-marker dict key redacts its own value
  and must not taint siblings), and containers nested in
  dict values are recursed into so a part key at any
  depth (``('api', {'outer': {'key': S}})``) is seen in
  traversal order. Plain string dict values are payload,
  never parts.
  '''
  if _seen is None:
    _seen = frozenset()
  if _depth > _MAX_COMPOSITE_DEPTH:
    return
  items_list = _safe_list(items)
  if items_list is None:
    return
  for item in items_list:
    if isinstance(item, (str, bytes, bytearray)):
      part = _marker_part(item)
      if part is not None:
        yield part
    elif isinstance(item, dict):
      yield from _ordered_parts_dict(item, _seen, _depth)
    elif isinstance(item, (tuple, list, frozenset, set)):
      if id(item) in _seen:
        continue
      nested = _safe_list(item)
      if nested is None:
        continue
      yield from _ordered_parts(
          nested, _seen | {id(item)}, _depth + 1)


def _ordered_parts_dict(item, _seen, _depth):
  '''Yield marker parts from a dict item (loop14 split).'''
  if id(item) in _seen:
    return
  try:
    pairs = list(itertools.islice(item.items(), _MAX_ITEMS + 1))
  except Exception:
    return
  try:
    pairs = [(key, value) for key, value in pairs]
  except (TypeError, ValueError):
    return
  seen_next = _seen | {id(item)}
  for key, value in pairs:
    if isinstance(key, (str, bytes, bytearray)):
      part = _marker_part(key)
      if part is not None and not _is_marker_element(key):
        yield part
    elif isinstance(key, (dict, list, tuple, frozenset, set)):
      yield from _ordered_parts([key], seen_next, _depth + 1)
    if isinstance(value, (dict, list, tuple, frozenset, set)):
      yield from _ordered_parts([value], seen_next, _depth + 1)


def _composite_marker_bearing(items, ordered, _seen=None, _depth=0):
  '''Return True when composite key items carry a marker.

  A composite is marker-bearing when any direct element is
  a marker element, when its joined normalized form
  contains a marker, or when marker-parts split across
  elements join to a marker (``('api', 'key')``,
  ``('api', 'HUNTER2', 'key')``, ``('HUNTER2', 'api',
  'key')``). Detection recurses into nested tuples/
  frozensets/sets/lists at any depth (flattened first, in
  order for ordered composites) and collects non-marker
  marker-part dict keys at any depth via
  ``_ordered_parts`` (``('api', {'key': S})``,
  ``({'api': S}, 'key')``), so a marker nested in a
  tuple cannot hide from the outer composite, and vice
  versa. For unordered composites (frozenset/set) the
  marker-parts are tried in every order (up to five parts);
  for ordered tuples only the given order counts.
  '''
  items_list = _safe_list(items)
  if items_list is None:
    return False
  flat = list(_flatten_composite(items_list, _seen, _depth))
  if any(_is_marker_element(item) for item in flat):
    return True
  if _is_secret_key(flat):
    return True
  parts = list(_ordered_parts(items_list, _seen, _depth))
  if not parts:
    return False
  if ordered:
    joined = ''.join(parts)
    return any(marker in joined for marker in _NORMALIZED_MARKERS)
  if len(parts) <= 5:
    for perm in itertools.permutations(parts):
      joined = ''.join(perm)
      if any(marker in joined for marker in _NORMALIZED_MARKERS):
        return True
    return False
  part_set = set(parts)
  # Every segmentation of a normalized marker into marker
  # parts: the four pair combos, the four-part
  # awssecretaccesskey combo, and single-part markers (a
  # part that is itself a marker, e.g. via a tuple dict
  # key, is bearing on its own, as in the ordered and
  # <=5 permutation cases above).
  combos = (
    ('api', 'key'), ('access', 'key'), ('private', 'key'),
    ('pass', 'word'), ('aws', 'secret', 'access', 'key'),
    ('password',), ('passwd',), ('pwd',), ('pass',),
    ('secret',), ('token',), ('authorization',),
    ('credential',), ('passphrase',), ('bearer',),
    ('auth',), ('cookie',))
  return any(set(combo) <= part_set for combo in combos)


def _composite_has_over_cap_leaf(items, _seen=None, _depth=0):
  '''Return True when any flattened leaf exceeds the key cap.

  Cycle8 Q5 (C8-01): an over-cap leaf (``len > _MAX_KEY_LEN``)
  is unverifiable -- a filler/leet/hybrid/fullwidth/separator
  marker in its middle gap is not detector-visible (``_is_secret_key``
  is False by Q2 design, with no blanket fail-closed there),
  so sibling finalization cannot trust that the leaf carries
  no marker. Flattening mirrors ``_composite_marker_bearing``
  so nested tuple/frozenset leaves count for the outer
  composite too. This is a ``len`` check only: bounded and
  linear, no normalization or hybrid work on the leaf.
  '''
  return any(
      isinstance(leaf, (str, bytes, bytearray))
      and len(leaf) > _MAX_KEY_LEN
      for leaf in _flatten_composite(items, _seen, _depth))




def _decode_bytes_scrubbed(value):
  '''Return scrubbed decoded bytes, or REDACTED if hostile.

  A hostile ``decode`` that raises or returns non-str
  becomes ``REDACTED``, never a crash or a leak. Shared
  by ``scrub_details`` and ``audit._clean_text``.
  '''
  try:
    decoded = value.decode('utf-8', errors='replace')
  except Exception:
    return REDACTED
  if not isinstance(decoded, str):
    return REDACTED
  return _scrub_text(decoded)
