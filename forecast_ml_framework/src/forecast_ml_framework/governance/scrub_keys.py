#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Secret markers, confusables, and key normalization.

Markers, filler vocabularies, glue tables,
unicode/confusable/split tables, caps, and the
safe leaf helpers plus key normalization live
here. This module imports no governance or
telemetry code, so it is the lowest scrub
layer; text/composite scrubbing in
``scrub_text`` and the facade ``scrub`` build
on it without cycles.
'''

import itertools
import re
import unicodedata

REDACTED = '***REDACTED***'

_SECRET_MARKERS = ('password', 'passwd', 'pwd', 'pass', 'secret',
                   'token',
                   'api_key', 'apikey', 'api-key', 'access_key',
                   'accesskey', 'private_key', 'privatekey',
                   'authorization', 'credential', 'passphrase',
                   'bearer', 'auth', 'cookie', 'awsaccesskeyid',
                   'awssecretaccesskey', 'encryption_key',
                   'master_key', 'signing_key', 'aws_key',
                   'gcp_key', 'otp', 'pin_code', 'pin_number')

# Loop17 P2/P3: non-marker filler words (copulas and degree
# adverbs). These may sit between marker parts (``api very
# key``) and between marker and value, and are stripped from
# normalized keys. Marker parts (``api``/``key``/...) are
# excluded so stripping never eats the marker itself.
# Loop17-arch F2: ``known``/``wellknown``/``well-known``
# were in ``_FILLER_WORDS`` but missing here, so
# ``api well-known key`` leaked; the chain words
# (``one``..``twenty``, ``a``..``f``, ``filler``,
# ``very2``, ``123``) are audited in for the same
# reason. H1: ``tremendously``/``literally``/
# ``literaly``/``honestly`` are degree adverbs like
# ``very``; any other ``-ly`` adverb is also filler
# via ``_is_ly_adverb`` (see ``_INTRA_FILLER``).
_INTRA_FILLER_WORDS = (
    'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'definitely', 'very', 'really', 'absolutely', 'truly',
    'completely', 'utterly', 'totally', 'certainly', 'precisely',
    'exactly', 'way', 'well', 'just', 'actually', 'basically',
    'pretty', 'quite', 'so', 'too', 'somewhat', 'extremely',
    'rather', 'super', 'much', 'tremendously', 'literally',
    'literaly', 'honestly', 'highly', 'incredibly', 'fairly',
    'particularly', 'especially', 'known', 'wellknown',
    'well-known',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
    'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
    'nineteen', 'twenty', 'a', 'b', 'c', 'd', 'e', 'f',
    'filler', 'very2', '123',
    # Cycle13 S14: HTTP auth schemes, filler like ``is``/
    # ``very`` in both filler vocabularies (see
    # ``_FILLER_WORDS``), so ``api basic key`` and a marker
    # chain (``Authorization: Basic Authorization: ...``)
    # scan as one marker instead of rescanning per marker.
    'basic', 'digest', 'negotiate')
_INTRA_FILLER_SET = frozenset(_INTRA_FILLER_WORDS)
_INTRA_FILLER = (
    '|'.join(sorted(_INTRA_FILLER_WORDS, key=len, reverse=True))
    # Loop20 P3: ``-ly`` stem is ``{3,}`` (total >= 5),
    # aligning with ``_is_ly_adverb`` (unbounded). The
    # stem stays last after the explicit vocab and the
    # filler is atomic, so a long ``a`` token matches the
    # single-letter ``a`` first and the stem is not
    # rescanned per suffix; a long non-filler token
    # tries the stem once, in C, over that token.
    + r'|[A-Za-z]{3,}ly')
# Separator between compound marker parts: any non-alnum run,
# optionally carrying filler words (``api very key``,
# ``api*very*key``). Loop17-arch H2: filler may also be
# glued (``apiverykey``), so no non-alnum is required
# after a filler word; a glued filler counts only
# between marker parts here and, in ``_normalize_key``,
# only when flanked by alphanumerics.
# Loop20 P1: filler count is uncapped (``*``). The
# 128 cap leaked ``api very*129 key``. Separator runs
# stay possessive (``*+``) and each filler stays
# atomic (``(?>...)``), so the regex stays linear:
# filler consumption is one pass in C and backtracking
# over filler count is linear in filler tokens, not
# quadratic (the ``-ly`` stem is last, see above, so
# a long ``a`` token consumes single letters without
# rescanning the stem per suffix). A deterministic
# token scan (see ``_find_compound_marker``) handles
# the a-f ``-ly`` trap the atomic regex cannot.
_SEP = (
    r'(?:[^A-Za-z0-9]*+'
    r'(?:(?>' + _INTRA_FILLER + r')[^A-Za-z0-9]*+)*)')


def _is_ly_adverb(word):
  '''Return True for a plain ``-ly`` adverb word.

  H1 class close: degree adverbs (``tremendously``,
  ``literally``, ``honestly``, ``absurdly``, ...) are
  filler even when not in the explicit vocab. Only a
  lowercased alpha word of length >= 5 ending in
  ``ly`` counts, so short words (``only``) and tokens
  with digits do not. Best-effort: a non-``ly`` filler
  outside the vocab is still treated as the value.
  '''
  return (len(word) >= 5 and word.isalpha()
          and word.endswith('ly'))


_GLUE_FLANK_PARTS = frozenset({
    'api', 'access', 'private', 'pass', 'aws', 'secret',
    'key', 'word', 'phrase', 'code', 'token', 'password',
    'passwd', 'pwd', 'passphrase', 'bearer', 'auth',
    'cookie', 'credential', 'authorization', 'hash',
    'encryption', 'master', 'signing', 'gcp', 'otp',
    'pin', 'number',
    'value', 'id', 'ids', 'keys', 'words', 'tokens',
    'secrets', 'passwords', 'credentials', 'passphrases'})


# Loop21 F2: glued filler words in scope are the
# alpha intra-filler entries of length >= 2 (single
# letters and digit-bearing entries are skipped, as
# before). Tokenizing an alnum run in one pass with
# the longest flank/filler word first (so ``bearer``
# is one flank token, not ``be`` + ``are``) is
# linear: each character is consumed once, unlike
# rescanning the whole run for every filler word.
_GLUE_FILLER_WORDS = frozenset(
    word for word in _INTRA_FILLER_SET
    if len(word) >= 2 and word.isalpha())
_GLUE_TOKEN_WORDS = tuple(sorted(
    _GLUE_FLANK_PARTS | _GLUE_FILLER_WORDS,
    key=len, reverse=True))
_GLUE_TOKEN_RE = re.compile(
    r'(?:' + '|'.join(
        re.escape(word) for word in _GLUE_TOKEN_WORDS)
    + r'|.)')
_GLUE_RUN_RE = re.compile(r'[a-z0-9]+')


def _strip_glued_filler_run(run):
  '''Strip chained glued filler tokens from one run.

  Filler tokens are dropped only in a contiguous
  run between two flank tokens (``api`` + ``very`` +
  ``very`` + ``key`` -> ``apikey``), so a filler
  inside a single marker word is kept (the whole
  word tokenizes as one flank token) and a leading
  or trailing filler is kept.
  '''
  tokens = [
      match.group() for match in _GLUE_TOKEN_RE.finditer(run)]
  kept = []
  pending = []
  prev_flank = False
  for token in tokens:
    if token in _GLUE_FILLER_WORDS:
      pending.append(token)
      continue
    if not (prev_flank and token in _GLUE_FLANK_PARTS):
      kept.extend(pending)
    pending.clear()
    kept.append(token)
    prev_flank = token in _GLUE_FLANK_PARTS
  kept.extend(pending)
  return ''.join(kept)


def _strip_glued_filler(text):
  '''Strip glued filler between marker parts.

  H2: ``apiverykey`` normalizes to ``apikey`` by
  removing ``very`` glued between ``api`` and
  ``key``. Loop21 F2: each ``[a-z0-9]+`` run is
  tokenized in a single pass (see
  ``_strip_glued_filler_run``), so 2+ chained
  fillers (``apiveryverykey``) are stripped too and
  a long chained key stays linear instead of
  rescanning the run for every filler word.
  '''
  return _GLUE_RUN_RE.sub(
      lambda match: _strip_glued_filler_run(match.group()),
      text)

# Loop18 P2: ``pass phrase``/``pass code`` are markers
# in free text exactly as in dict keys (where the
# ``pass`` substring already blocks them).
_MARKER = (
    r'(?:aws' + _SEP + r'access' + _SEP + r'key' + _SEP + r'id|'
    r'aws' + _SEP + r'secret' + _SEP + r'access' + _SEP + r'key|'
    r'secret' + _SEP + r'access' + _SEP + r'key|'
    r'access' + _SEP + r'key' + _SEP + r'id|'
    r'password|passwd|passphrase|pass' + _SEP + r'word|secret|token|'
    r'pass' + _SEP + r'phrase|pass' + _SEP + r'code|'
    r'api' + _SEP + r'key|access' + _SEP + r'key|'
    r'private' + _SEP + r'key|'
    r'encryption' + _SEP + r'key|master' + _SEP + r'key|'
    r'signing' + _SEP + r'key|aws' + _SEP + r'key|'
    r'gcp' + _SEP + r'key|otp|'
    r'pin' + _SEP + r'code|pin' + _SEP + r'number|'
    r'authorization|credential|bearer|cookie|auth|pwd|pass)'
    + r'(?:' + _SEP + r'(?:key|keys|hash|hashes|value|values|'
    r'token|tokens|secret|secrets|credential|credentials|auth|'
    r'access|private|password|passwd|pwd|passphrase|bearer|'
    r'cookie|authorization|word|id|ids|phrase|code)'
    r'(?![A-Za-z0-9]))*')
# Marker-only search pattern: literal marker words (longest
# first so ``password`` wins over ``pass``). Searching for a
# marker is linear -- no lazy prefix/suffix or bridge is
# involved -- so a long token without a marker cannot cause
# quadratic backtracking. The bridge/value parse that follows
# a marker is a deterministic forward scan (see
# ``_scan_value``), never a backtracking regex, and filler
# words between marker and value are unbounded. Loop15 B1:
# the intra-marker separator is any non-alnum
# (``[^A-Za-z0-9]``), matching ``_normalize_key`` which
# strips any non-alnum, so ``api|key``/``api*key``/
# ``api·key`` are markers exactly as dict keys are.
_MARKER_RE = re.compile(_MARKER, re.IGNORECASE)
# Inputs longer than this are truncated in raw form
# before normalization and scrubbing; the remainder is
# dropped and a ``[TRUNCATED]`` note (with ``REDACTED``)
# is appended, so over-long text is bounded work and a
# secret past the cutoff cannot leak.
_MAX_TEXT_LEN = 10000
# Loop20 P4: dict keys are capped at 8192 (telemetry
# ``_MAX_STRING_LEN``) before ``_is_secret_key`` /
# hybrid checks, so an unbounded key cannot cause
# unbounded hybrid-DP work in ``scrub_details``/audit.
_MAX_KEY_LEN = 8192
_MAX_ITEMS = 1000
_TRUNCATED_NOTE = '[TRUNCATED]'
_QUOTE_PAIRS = {'"': '"', '\'': '\'', '`': '`', '“': '”', '‘': '’'}
_TERMINATORS = frozenset({',', ';', '}', ']', ')'})
_AWS_ACCESS_KEY_RE = re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b')
_SK_KEY_RE = re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b')
_PEM_BLOCK_RE = re.compile(
    r'-----BEGIN [A-Z ]*?PRIVATE KEY-----.*?'
    r'(?:-----END [A-Z ]*?PRIVATE KEY-----|$)',
    re.DOTALL)
# Bearer group 2 must not consume a marker word itself
# (``Bearer token Zq...``): redacting ``token`` would leave
# ``Bearer ***REDACTED*** Zq...`` and the marker scan would
# treat the placeholder as the value and stop, leaking the
# real secret. The negative lookahead skips a whole marker
# word (longest compounds first, e.g. ``api_key`` before
# ``api``) followed by a non-token char, so the marker scan
# redacts the following secret instead. A glued form
# (``token=Zq...``) is not skipped: Bearer consumes and
# redacts the whole glued token. Loop14 F2/F3/F1: the
# lookahead also skips a bracketed/quoted marker
# (``(token)``, ``[token]``, ``"token"``) and plural
# marker words (``keys``/``words``/``ids``/...).
# Loop15 B1: its intra-marker separator is likewise any
# non-alnum, and ``aws`` is a skipped first word so
# ``Bearer aws*access*key`` is not consumed as a token.
# Loop17 P4: the Bearer separator class also includes the
# closing brackets )]} so ``Bearer very)really Z`` and
# ``Bearer is] Z`` consume the filler chain and redact the
# token after it, instead of redacting the filler itself.
_BEARER_TEXT_RE = re.compile(
    r'\b(Bearer(?:[\s,;:=)\]}]+(?:' + _INTRA_FILLER + r'))*'
    r'[\s,;:=)\]}]+)'
    r'(?!(?:[\(\[\{\x27"`]\s*)?(?:aws' + _SEP
    + r'access' + _SEP + r'key' + _SEP + r'id|'
    r'aws' + _SEP + r'secret' + _SEP + r'access' + _SEP + r'key|'
    r'secret' + _SEP + r'access' + _SEP + r'key|'
    r'access' + _SEP + r'key' + _SEP + r'id|'
    r'password|passwd|passphrase|pass' + _SEP + r'word|secret|'
    r'pass' + _SEP + r'phrase|pass' + _SEP + r'code|'
    r'token|api' + _SEP + r'key|access' + _SEP + r'key|'
    r'private' + _SEP + r'key|encryption' + _SEP + r'key|'
    r'master' + _SEP + r'key|signing' + _SEP + r'key|'
    r'aws' + _SEP + r'key|gcp' + _SEP + r'key|otp|'
    r'pin' + _SEP + r'code|pin' + _SEP + r'number|'
    r'authorization|credential|'
    r'bearer|cookie|auth|pwd|pass|aws|api|access|private|'
    r'encryption|master|signing|gcp|otp|pin|number|'
    r'key|keys|words|ids|tokens|secrets|passwords|credentials|'
    r'passphrases|word|phrase|code)(?:\s*[\)\]\}\x27"`]|'
    r'(?![A-Za-z0-9_.\-+/\=])))'
    r'([A-Za-z0-9_.\-+/\=]+)', re.IGNORECASE)
# Cycle13 S14: bare HTTP auth schemes (``Basic``/``Digest``/
# ``Negotiate``) mirror ``Bearer`` but stay FP-guarded: a bare
# scheme redacts only a base64-ish token (see
# ``_is_base64ish_token``), so ordinary prose such as ``This
# is basic math`` or ``Basic research ...`` is kept. The shape
# check lives in ``_sub_spans`` (mode ``'scheme'``), not in
# regex lookaheads: two ``[charset]*`` lookaheads backtrack
# quadratically on a long token that fails the shape.
# The ``Authorization`` context needs no such shape check --
# scheme words are filler between that marker and its value
# (see ``_FILLER_WORDS``), so any token after the scheme, even
# a short one, redacts there. A marker word after a bare
# scheme (``Basic token Z``) needs no skip here: it carries
# no digit/``+/=``, so the marker scan redacts ``Z`` instead.
_BARE_SCHEME_TEXT_RE = re.compile(
    r'\b((?:Basic|Digest|Negotiate)'
    r'(?:[\s,;:=)\]}]+(?:' + _INTRA_FILLER + r'))*'
    r'[\s,;:=)\]}]+)'
    r'([A-Za-z0-9_.\-+/\=]+)', re.IGNORECASE)
_URL_CRED_RE = re.compile(
    r'(://[^/\s:@]+:)([^\s/]+)(@[^/\s@]+)', re.IGNORECASE)


_ZERO_WIDTH = '\u200b\u200c\u200d\u2060\ufeff\u00ad'

# Cyrillic (and Greek look-alike) confusables that NFKC does
# not fold to Latin: maps each look-alike to its Latin twin so
# marker matching cannot be evaded by homoglyphs. Accented
# Latin (``á`` U+00E1 -> ``a``), IPA alpha (``ɑ`` U+0251 ->
# ``a``), combining acute (U+0301, deleted; also stripped
# generically in ``_normalize_unicode``) and Cyrillic ya
# (``я`` U+044F -> ``r``, reversed-R look-alike) are included
# because NFKC folds none of them to ASCII.
# Digraph folds: sharp S (``ß`` U+00DF / ``ẞ`` U+1E9E ->
# ``ss``/``SS``) and thorn (``þ`` U+00FE / ``Þ`` U+00DE ->
# ``th``/``TH``) are single glyphs transliterating to two
# ASCII letters that can spell a marker bigram (``ss`` in
# password/passwd/passphrase/access_key, ``th`` in
# auth/authorization). A systematic scan of all codepoints
# (NFKC expansion, casefold to two letters, and known
# linguistic digraphs) found no other single glyph whose
# two-letter form can spell a marker: ``ae``/``oe``/``ij``/
# ``sh``/``zh``/``ng``/``dz``/``ff``/``fi``/``fl``/``st``
# occur in no normalized marker (password, passwd,
# passphrase, secret, token, api_key, access_key,
# private_key, authorization, credential, bearer, auth,
# cookie), and NFKC ligatures/square units/Roman numerals
# already expand to ASCII before ``translate`` runs.
# Folding ``æ``->``a`` (vs ``ae``) or ``œ``->``o`` (vs
# ``oe``) would be ambiguous and could false-positive, so
# ``æ``/``œ`` (and the other excluded digraphs) are not
# folded.
# UTS39 letter confusables (confusables.txt, Unicode 18):
# every missing letter source (category ``L*``, not ASCII,
# not already present, single ASCII-letter target, no
# duplicate keys) maps to its ASCII twin, covering
# Coptic, Cherokee, small-caps/modifiers, Armenian,
# Lisu, Canadian Aboriginal and other scripts.
# Existing digraph entries (``ß``/``þ``) win over the
# UTS39 pair for the same source. ``\u2c63`` (P with
# stroke) maps to ``p`` per the adversarial PoC. The
# alternate table below inherits all of these entries.
# Loop12 F2/F3: UTS39 non-L single-letter sources (110
# glyphs: digits/symbols in non-ASCII scripts, symbols
# like U+237A/U+212E) and digraph sources (13 glyphs
# spelling oo/au/co/ie/th in markers) are added below,
# except ASCII ``0``/``1``/``|``, which go to the alt
# table only (see below) so legit ASCII numbers and
# timestamps keep their primary form while leet
# spellings (``t0ken``, ``credentia1``) are still
# detected via the alt skeleton. U+FE31 also sits in
# the pre-NFKC split table (NFKC would turn it into an
# em dash before the confusable translate). Trade-off:
# a non-ASCII digit/symbol glyph in ordinary text folds
# to its
# letter in normalized output, as do the L* glyphs.
_CONFUSABLES = {
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0445': 'x', '\u0443': 'y', '\u043a': 'k',
    '\u043d': 'h', '\u0442': 't', '\u0432': 'b', '\u0455': 's',
    '\u0456': 'i', '\u0458': 'j', '\u04bb': 'h', '\u043f': 'n',
    '\u043c': 'm', '\u0434': 'd', '\u049b': 'k',
    '\u044f': 'r', '\u0433': 'r', '\u0451': 'e', '\u04af': 'y',
    '\u0410': 'A', '\u0412': 'B', '\u0415': 'E', '\u041a': 'K',
    '\u041c': 'M', '\u041d': 'H', '\u041e': 'O', '\u0420': 'P',
    '\u0421': 'C', '\u0422': 'T', '\u0425': 'X', '\u0423': 'Y',
    '\u0405': 'S', '\u0406': 'I', '\u0408': 'J', '\u04ba': 'H',
    '\u042f': 'R', '\u0413': 'R', '\u0401': 'E', '\u04ae': 'Y',
    '\u03b1': 'a', '\u03b5': 'e', '\u03bf': 'o', '\u03c1': 'p',
    '\u03c4': 't', '\u03c5': 'u', '\u03c7': 'x', '\u03ba': 'k',
    '\u03bd': 'v', '\u03b9': 'i', '\u03c3': 'o', '\u03c2': 's',
    '\u03bc': 'u', '\u03b2': 'b', '\u03b3': 'y', '\u03b4': 'd',
    '\u03b7': 'n', '\u03c0': 'n', '\u03b8': 'o', '\u03c6': 'o',
    '\u03b6': 'z', '\u03c9': 'w', '\u0391': 'A', '\u0395': 'E',
    '\u039f': 'O', '\u03a1': 'P', '\u03a4': 'T', '\u03a5': 'Y',
    '\u03a7': 'X', '\u039a': 'K', '\u039d': 'N', '\u0399': 'I',
    '\u0392': 'B', '\u0397': 'H', '\u0396': 'Z', '\u03a9': 'W',
    '\u03ac': 'a', '\u03ad': 'e', '\u03ae': 'n', '\u03af': 'i',
    '\u03cc': 'o', '\u03cd': 'u', '\u0386': 'A', '\u0388': 'E',
    '\u0389': 'I', '\u038c': 'O',
    '\u00e1': 'a', '\u00e0': 'a', '\u00e2': 'a', '\u00e3': 'a',
    '\u00e4': 'a', '\u00e5': 'a', '\u0101': 'a', '\u0103': 'a',
    '\u0105': 'a', '\u0251': 'a', '\u0250': 'a',
    '\u00c1': 'A', '\u00c0': 'A', '\u00c2': 'A', '\u00c3': 'A',
    '\u00c4': 'A', '\u00c5': 'A', '\u0100': 'A', '\u0102': 'A',
    '\u0104': 'A',
    '\u00e9': 'e', '\u00e8': 'e', '\u00ea': 'e', '\u00eb': 'e',
    '\u0113': 'e', '\u0115': 'e', '\u0117': 'e', '\u0119': 'e',
    '\u011b': 'e', '\u025b': 'e',
    '\u00c9': 'E', '\u00c8': 'E', '\u00ca': 'E', '\u00cb': 'E',
    '\u0112': 'E', '\u0114': 'E', '\u0116': 'E', '\u0118': 'E',
    '\u011a': 'E',
    '\u00ed': 'i', '\u00ec': 'i', '\u00ee': 'i', '\u00ef': 'i',
    '\u0129': 'i', '\u012b': 'i', '\u012d': 'i', '\u012f': 'i',
    '\u0131': 'i',
    '\u00cd': 'I', '\u00cc': 'I', '\u00ce': 'I', '\u00cf': 'I',
    '\u0128': 'I', '\u012a': 'I', '\u012c': 'I', '\u012e': 'I',
    '\u0130': 'I',
    '\u00f3': 'o', '\u00f2': 'o', '\u00f4': 'o', '\u00f5': 'o',
    '\u00f6': 'o', '\u00f8': 'o', '\u014d': 'o', '\u014f': 'o',
    '\u0151': 'o',
    '\u00d3': 'O', '\u00d2': 'O', '\u00d4': 'O', '\u00d5': 'O',
    '\u00d6': 'O', '\u00d8': 'O', '\u014c': 'O', '\u014e': 'O',
    '\u0150': 'O',
    '\u00fa': 'u', '\u00f9': 'u', '\u00fb': 'u', '\u00fc': 'u',
    '\u0169': 'u', '\u016b': 'u', '\u016d': 'u', '\u016f': 'u',
    '\u0171': 'u', '\u0173': 'u',
    '\u00da': 'U', '\u00d9': 'U', '\u00db': 'U', '\u00dc': 'U',
    '\u0168': 'U', '\u016a': 'U', '\u016c': 'U', '\u016e': 'U',
    '\u0170': 'U', '\u0172': 'U',
    '\u00fd': 'y', '\u00ff': 'y', '\u0177': 'y',
    '\u00dd': 'Y', '\u0176': 'Y',
    '\u00e7': 'c', '\u0107': 'c', '\u010b': 'c', '\u010d': 'c',
    '\u00c7': 'C', '\u0106': 'C', '\u010a': 'C', '\u010c': 'C',
    '\u00f1': 'n', '\u0144': 'n', '\u0146': 'n',
    '\u00d1': 'N', '\u0143': 'N', '\u0145': 'N',
    '\u011f': 'g', '\u0121': 'g', '\u0123': 'g', '\u011c': 'G',
    '\u011e': 'G', '\u0120': 'G', '\u0122': 'G',
    '\u0125': 'h', '\u0127': 'h', '\u0124': 'H', '\u0126': 'H',
    '\u010f': 'd', '\u0111': 'd', '\u00f0': 'd',
    '\u010e': 'D', '\u0110': 'D', '\u00d0': 'D',
    '\u00df': 'ss', '\u1e9e': 'SS',
    '\u00fe': 'th', '\u00de': 'TH',
    '\u015b': 's', '\u015d': 's', '\u015f': 's', '\u0161': 's',
    '\u017f': 's', '\u015a': 'S', '\u015c': 'S', '\u015e': 'S',
    '\u0160': 'S',
    '\u0165': 't', '\u0163': 't', '\u0167': 't',
    '\u0164': 'T', '\u0162': 'T', '\u0166': 'T',
    '\u0137': 'k', '\u0136': 'K',
    '\u013a': 'l', '\u013c': 'l', '\u013e': 'l', '\u0142': 'l',
    '\u0139': 'L', '\u013b': 'L', '\u013d': 'L', '\u0141': 'L',
    '\u0155': 'r', '\u0157': 'r', '\u0159': 'r', '\u027f': 'r',
    '\u1d19': 'r', '\u0154': 'R', '\u0156': 'R', '\u0158': 'R',
    '\u017a': 'z', '\u017c': 'z', '\u017e': 'z',
    '\u0179': 'Z', '\u017b': 'Z', '\u017d': 'Z',
    '\u0300': '', '\u0301': '', '\u0302': '', '\u0303': '',
    '\u0304': '', '\u0305': '', '\u0306': '', '\u0307': '',
    '\u0308': '', '\u030b': '', '\u030c': '', '\u0327': '',
    '\u0328': '', '\u0331': '', '\u0332': '', '\u0342': '',
    '\u0344': '', '\u0345': '',
    '\u0184': 'b', '\u018d': 'g', '\u0192': 'f', '\u0196': 'l',
    '\u01a6': 'R', '\u01bd': 's', '\u01bf': 'p', '\u01c0': 'l',
    '\u0237': 'j', '\u024c': 'R', '\u0261': 'g', '\u0263': 'y',
    '\u0269': 'i', '\u026a': 'i', '\u026f': 'w', '\u0284': 'f',
    '\u028b': 'u', '\u028f': 'y', '\u037f': 'J', '\u039c': 'M',
    '\u03dc': 'F', '\u03ed': 'o', '\u03f3': 'j',
    '\u03f8': 'p', '\u03fa': 'M', '\u042c': 'b',
    '\u0448': 'w', '\u0461': 'w', '\u0474': 'V', '\u0475': 'v',
    '\u04bd': 'e', '\u04c0': 'l', '\u04cf': 'l', '\u0501': 'd',
    '\u050c': 'G', '\u051a': 'Q', '\u051b': 'q', '\u051c': 'W',
    '\u051d': 'w', '\u054d': 'U', '\u054f': 'S', '\u0555': 'O',
    '\u0561': 'w', '\u0563': 'q', '\u0566': 'q', '\u0570': 'h',
    '\u0575': 'j', '\u0578': 'n', '\u057c': 'n', '\u057d': 'u',
    '\u0581': 'g', '\u0582': 'i', '\u0584': 'f', '\u0585': 'o',
    '\u05d5': 'l', '\u05d8': 'v', '\u05df': 'l', '\u05e1': 'o',
    '\u0627': 'l', '\u0647': 'o', '\u06be': 'o', '\u06c1': 'o',
    '\u06d5': 'o', '\u07ca': 'l', '\u07cb': 'o', '\u07cc': 'Y',
    '\u07d3': 'F', '\u07d5': 'b', '\u07e0': 'T', '\u0840': 'o',
    '\u0b20': 'O', '\u0d1f': 's', '\u0d20': 'o', '\u1004': 'c',
    '\u101d': 'o', '\u105a': 'c', '\u10b9': 'h', '\u10bd': 'S',
    '\u10cd': 'Z', '\u10e7': 'y', '\u10fd': 'S', '\u10ff': 'o',
    '\u110b': 'o', '\u11bc': 'o', '\u1200': 'U', '\u12d0': 'O',
    '\u13a0': 'D', '\u13a1': 'R', '\u13a2': 'T', '\u13a5': 'i',
    '\u13a9': 'Y', '\u13aa': 'A', '\u13ab': 'J', '\u13ac': 'E',
    '\u13b3': 'W', '\u13b7': 'M', '\u13bb': 'H', '\u13bd': 'Y',
    '\u13c0': 'G', '\u13c2': 'h', '\u13c3': 'Z', '\u13cf': 'b',
    '\u13d2': 'R', '\u13d4': 'W', '\u13d5': 'S', '\u13d9': 'V',
    '\u13da': 'S', '\u13de': 'L', '\u13df': 'C', '\u13e2': 'P',
    '\u13e6': 'K', '\u13e7': 'd', '\u13f3': 'G', '\u13f4': 'B',
    '\u142f': 'V', '\u144c': 'U', '\u146d': 'P', '\u146f': 'd',
    '\u1472': 'b', '\u148d': 'J', '\u14aa': 'L', '\u1541': 'x',
    '\u157c': 'H', '\u157d': 'x', '\u1587': 'R', '\u15af': 'b',
    '\u15b4': 'F', '\u15c5': 'A', '\u15de': 'D', '\u15ea': 'D',
    '\u15f0': 'M', '\u15f7': 'B', '\u16b7': 'X', '\u16c1': 'l',
    '\u16d0': 'l', '\u16d5': 'K', '\u16d6': 'M', '\u1763': 'x',
    '\u1a45': 'o', '\u1c82': 'o', '\u1c83': 'c', '\u1cbd': 'S',
    '\u1cbf': 'O', '\u1cf5': 'X', '\u1d04': 'c', '\u1d0f': 'o',
    '\u1d11': 'o', '\u1d1c': 'u', '\u1d20': 'v', '\u1d21': 'w',
    '\u1d22': 'z', '\u1d26': 'r', '\u1d83': 'g', '\u1d8c': 'y',
    '\u1e9d': 'f', '\u1eff': 'y',
    '\u2c63': 'p', '\u2c6b': 'Z', '\u2c6c': 'z', '\u2c82': 'B',
    '\u2c85': 'r', '\u2c8c': 'Z', '\u2c8d': 'z', '\u2c8e': 'H',
    '\u2c92': 'l', '\u2c93': 'i', '\u2c94': 'K', '\u2c98': 'M',
    '\u2c9a': 'N', '\u2c9e': 'O', '\u2c9f': 'o', '\u2ca2': 'P',
    '\u2ca3': 'p', '\u2ca4': 'C', '\u2ca5': 'c', '\u2ca6': 'T',
    '\u2ca8': 'Y', '\u2ca9': 'y', '\u2cac': 'X', '\u2cbd': 'w',
    '\u2cce': 'P', '\u2ccf': 'p', '\u2cd0': 'L', '\u2d2d': 'z',
    '\u2d38': 'V', '\u2d39': 'E', '\u2d4a': 'l', '\u2d4f': 'l',
    '\u2d54': 'O', '\u2d55': 'Q', '\u2d5d': 'X', '\u3112': 'T',
    '\u311a': 'Y', '\u3147': 'o', '\u4e05': 'T', '\u4e2b': 'Y',
    '\ua4d0': 'B', '\ua4d1': 'P', '\ua4d2': 'd', '\ua4d3': 'D',
    '\ua4d4': 'T', '\ua4d6': 'G', '\ua4d7': 'K', '\ua4d9': 'J',
    '\ua4da': 'C', '\ua4dc': 'Z', '\ua4dd': 'F', '\ua4df': 'M',
    '\ua4e0': 'N', '\ua4e1': 'L', '\ua4e2': 'S', '\ua4e3': 'R',
    '\ua4e6': 'V', '\ua4e7': 'H', '\ua4ea': 'W', '\ua4eb': 'X',
    '\ua4ec': 'Y', '\ua4ee': 'A', '\ua4f0': 'E', '\ua4f2': 'l',
    '\ua4f3': 'O', '\ua4f4': 'U', '\ua50b': 'T', '\ua557': 'B',
    '\ua56f': 'l', '\ua576': 'S', '\ua5cb': 'E', '\ua647': 'i',
    '\ua6c9': 'Z', '\ua6df': 'V', '\ua731': 's', '\ua781': 'l',
    '\ua798': 'F', '\ua799': 'f', '\ua79f': 'u', '\ua7ae': 'l',
    '\ua7b2': 'J', '\ua7b3': 'X', '\ua7b4': 'B', '\ua7fa': 'w',
    '\ua7fe': 'l', '\uab32': 'e', '\uab35': 'f', '\uab3d': 'o',
    '\uab47': 'r', '\uab48': 'r', '\uab4e': 'u', '\uab52': 'u',
    '\uab5a': 'y', '\uab64': 'a', '\uab75': 'i', '\uab81': 'r',
    '\uab83': 'w', '\uab93': 'z', '\uaba4': 'w', '\uaba9': 'v',
    '\uabaa': 's', '\uabaf': 'c', '\ufba6': 'o', '\ufba7': 'o',
    '\ufba8': 'o', '\ufba9': 'o', '\ufbaa': 'o', '\ufbab': 'o',
    '\ufbac': 'o', '\ufbad': 'o', '\ufe8d': 'l', '\ufe8e': 'l',
    '\ufee9': 'o', '\ufeea': 'o', '\ufeeb': 'o', '\ufeec': 'o',
    '\uffb7': 'o', '\U00010282': 'B', '\U00010286': 'E',
    '\U00010287': 'F', '\U0001028a': 'l', '\U00010290': 'X', '\U00010292': 'O',
    '\U00010295': 'P', '\U00010296': 'S', '\U00010297': 'T', '\U000102a0': 'A',
    '\U000102a1': 'B', '\U000102a2': 'C', '\U000102a5': 'F', '\U000102ab': 'O',
    '\U000102b0': 'M', '\U000102b1': 'T', '\U000102b2': 'Y', '\U000102b4': 'X',
    '\U000102cf': 'H', '\U00010301': 'B', '\U00010302': 'C', '\U00010309': 'l',
    '\U0001030f': 'O', '\U00010311': 'M', '\U00010315': 'T', '\U00010317': 'X',
    '\U0001031c': 'b', '\U00010404': 'O', '\U00010415': 'C', '\U0001041b': 'L',
    '\U00010420': 'S', '\U0001042c': 'o', '\U0001043d': 'c', '\U00010448': 's',
    '\U000104b4': 'R', '\U000104c2': 'O', '\U000104ce': 'U', '\U000104ea': 'o',
    '\U000104f6': 'u', '\U00010507': 'Z', '\U0001050e': 'l', '\U00010513': 'N',
    '\U00010516': 'O', '\U00010518': 'K', '\U0001051b': 'C', '\U0001051d': 'V',
    '\U00010525': 'F', '\U00010526': 'L', '\U00010527': 'X', '\U00010926': 'l',
    '\U0001092c': 'o', '\U00010c13': 'X', '\U00010c17': 'O', '\U00010c1f': 'V',
    '\U00010c20': 'Y', '\U00010c21': 'M', '\U00010c3e': 'l', '\U00010c82': 'X',
    '\U00010ca5': 'l', '\U00010cc2': 'x', '\U00010d07': 'o', '\U00011124': 'o',
    '\U00011706': 'v', '\U0001170a': 'w', '\U0001170e': 'w', '\U0001170f': 'w',
    '\U000118a0': 'V', '\U000118a2': 'F', '\U000118a3': 'L', '\U000118a4': 'Y',
    '\U000118a6': 'E', '\U000118a9': 'Z', '\U000118ae': 'E', '\U000118b2': 'L',
    '\U000118b5': 'O', '\U000118b8': 'U', '\U000118bc': 'T', '\U000118c0': 'v',
    '\U000118c1': 's', '\U000118c2': 'F', '\U000118c3': 'i', '\U000118c4': 'y',
    '\U000118c8': 'o', '\U000118d7': 'o', '\U000118d8': 'u', '\U000118dc': 'y',
    '\U00011abc': 'Z', '\U00011abe': 'N', '\U00016a19': 'r', '\U00016ad6': 'S',
    '\U00016ae9': 'O', '\U00016d63': 'l', '\U00016f08': 'V', '\U00016f0a': 'T',
    '\U00016f16': 'L', '\U00016f28': 'l', '\U00016f35': 'R', '\U00016f3a': 'S',
    '\U00016f40': 'A', '\U00016f42': 'U', '\U00016f43': 'Y',
    '\U0001d6a5': 'j',
    '\U0001d6b3': 'M', '\U0001d6ed': 'M',
    '\U0001d727': 'M', '\U0001d761': 'M',
    '\U0001d79b': 'M', '\U0001d7ca': 'F', '\U0001ee00': 'l', '\U0001ee24': 'o',
    '\U0001ee80': 'l', '\U0001ee84': 'o',
    # Cn in Unicode 16 (assigned in Unicode 18): confusable
    # letters Python 3.14 leaves unassigned, so NFKC keeps
    # them and the table must fold them before marker search.
    '\U0001df5a': 'a', '\U0001df6a': 'A', '\U00016eb6': 'b',
    '\U0001df81': 'E', '\U00011de1': 'l', '\U00011dda': 'l',
    '\U00016eaa': 'l', '\U00011de0': 'O', '\U0001cefc': 'V',
    '\U0001df7d': 'w',
    # Loop12 F2: UTS39 non-L single-letter confusables
    # (non-ASCII; ASCII 0/1/| are alt-only, below).
    '\u00a1': 'i', '\u05c0': 'l', '\u0661': 'l',
    '\u0665': 'o', '\u0667': 'V', '\u06f1': 'l',
    '\u06f5': 'o', '\u06f7': 'V', '\u07c0': 'O',
    '\u0964': 'l', '\u0966': 'o', '\u09e6': 'o',
    '\u0a66': 'o', '\u0ae6': 'o', '\u0b66': 'o',
    '\u0be6': 'o', '\u0c02': 'o', '\u0c66': 'o',
    '\u0c82': 'o', '\u0ce6': 'O', '\u0d02': 'o',
    '\u0d66': 'o', '\u0d82': 'o', '\u0e50': 'o',
    '\u0ed0': 'o', '\u1040': 'o', '\u104a': 'l',
    '\u17e0': 'o', '\u1a80': 'o', '\u1a90': 'o',
    '\u1bea': 'o', '\u212e': 'e', '\u21bf': 'l',
    '\u2223': 'l', '\u2228': 'v', '\u222a': 'U',
    '\u22a4': 'T', '\u22c1': 'v', '\u22c3': 'U',
    '\u22ff': 'E', '\u2373': 'i', '\u2374': 'p',
    '\u237a': 'a', '\u23fd': 'l', '\u2502': 'l',
    '\u2503': 'l', '\u27d9': 'T', '\u29e2': 'w',
    '\u3007': 'O', '\ua830': 'l', '\ua8ce': 'l',
    '\uaa5d': 'l', '\ufe31': 'l', '\uff10': 'O',
    '\uff11': 'l', '\uff5c': 'l', '\uffe8': 'l',
    '\U0001018b': 'd', '\U00010320': 'l',
    '\U00010cfa': 'l', '\U00011047': 'l',
    '\U000110c0': 'l', '\U00011141': 'l',
    '\U000111c5': 'l', '\U00011302': 'o',
    '\U000113d4': 'l', '\U0001144b': 'l',
    '\U000114d0': 'o', '\U000115c5': 'l',
    '\U00011641': 'l', '\U000118e0': 'O',
    '\U000118e6': 'W', '\U000118e9': 'C',
    '\U000118ef': 'W', '\U000118f2': 'C',
    '\U00011c41': 'l', '\U0001ccf0': 'O',
    '\U0001ccf1': 'l', '\U0001d100': 'l',
    '\U0001d134': 'c', '\U0001d207': 'b',
    '\U0001d20c': 'W', '\U0001d20d': 'V',
    '\U0001d216': 'R', '\U0001d22a': 'L',
    '\U0001d373': 'T', '\U0001d377': 'l',
    '\U0001d7ce': 'O', '\U0001d7cf': 'l',
    '\U0001d7d8': 'O', '\U0001d7d9': 'l',
    '\U0001d7e2': 'O', '\U0001d7e3': 'l',
    '\U0001d7ec': 'O', '\U0001d7ed': 'l',
    '\U0001d7f6': 'O', '\U0001d7f7': 'l',
    '\U0001e140': 'O', '\U0001e141': 'l',
    '\U0001e145': 'V', '\U0001e2f0': 'O',
    '\U0001e8c7': 'l', '\U0001ed01': 'l',
    '\U0001f74c': 'C', '\U0001f768': 'T',
    '\U0001fbf0': 'O', '\U0001fbf1': 'l',
    # Loop12 F3: UTS39 digraph confusables spelling
    # oo/au/co/ie/th in markers (multi-char values).
    '\u1147': 'oo', '\u11ee': 'oo', '\u3180': 'oo',
    '\ua698': 'OO', '\ua699': 'oo', '\ua736': 'AU',
    '\ua737': 'au', '\ua74e': 'OO', '\ua74f': 'oo',
    '\uab43': 'co', '\u221e': 'oo',
    '\U0001df5d': 'ie', '\U0001df64': 'th',
}
_CONFUSABLE_TABLE = str.maketrans(_CONFUSABLES)
# Alternate skeleton: Cyrillic ya (U+044F) is a reversed-R
# look-alike (primary ``r``), but some homoglyph generators
# also use it for ``a``. Detection tries both skeletons so
# neither spelling leaks. Capital I (ASCII U+0049, Greek
# Iota U+0399, Cyrillic U+0406) looks like ``l`` in
# ``credential`` (alt ``l``, primary ``I``), and thorn
# (U+00FE/U+00DE) is ``th`` in ``authorization`` (primary)
# but ``p`` in ``password`` (alt ``p``/``P``): the alt
# skeleton carries those readings so neither leaks.
_CONFUSABLES_ALT = dict(_CONFUSABLES)
_CONFUSABLES_ALT['\u044f'] = 'a'
_CONFUSABLES_ALT['\u042f'] = 'A'
_CONFUSABLES_ALT['I'] = 'l'
_CONFUSABLES_ALT['\u0399'] = 'l'
_CONFUSABLES_ALT['\u0406'] = 'l'
_CONFUSABLES_ALT['\u00fe'] = 'p'
_CONFUSABLES_ALT['\u00de'] = 'P'
# Loop12 F2 ASCII leet sources are alt-only: folding
# ``0``/``1``/``|`` in the primary skeleton would mangle
# legit ASCII numbers, timestamps and pipes in normalized
# output; the alt skeleton still detects ``t0ken``,
# ``credentia1`` and ``credentia|``, and span union
# redacts the value in primary coordinates.
# Cycle1 W1 F3: ``4``/``3``/``5``/``7``/``@``/``$``/``!``
# are alt-only leet readings the same way, so
# ``p4ssword``, ``p455word``, ``p@ssword``, ``$ecret``
# and ``5ecret`` redact while legit digits keep their
# primary form (``epoch 2026-01-02`` is untouched).
_CONFUSABLES_ALT['0'] = 'O'
_CONFUSABLES_ALT['1'] = 'l'
_CONFUSABLES_ALT['|'] = 'l'
_CONFUSABLES_ALT['4'] = 'a'
_CONFUSABLES_ALT['3'] = 'e'
_CONFUSABLES_ALT['5'] = 's'
_CONFUSABLES_ALT['7'] = 't'
_CONFUSABLES_ALT['@'] = 'a'
_CONFUSABLES_ALT['$'] = 's'
_CONFUSABLES_ALT['!'] = 'i'
_CONFUSABLE_TABLE_ALT = str.maketrans(_CONFUSABLES_ALT)

# Single glyphs whose NFKC decomposition is a separator plus
# a single ASCII letter: parenthesized letters (``⒟`` ->
# ``(d)``), tortoise-shell ``S``, degree ``C``/``F`` and
# ``Ŀ``/``ŀ`` (``L·``/``l·``). Collapsing them to the bare
# letter before NFKC keeps marker detection from being split
# by the introduced brackets/dot, while ASCII ``(a)`` typed
# as three characters (spaced or not) is left untouched.
# Enclosed negative circled/squared capital letters
# (U+1F150-169, U+1F170-189), crossed negative squared P
# (U+1F18A), negative squared abbreviations (U+1F18B-18F:
# IC/PA/SA/AB/WC), squared ASCII words/small D (U+1F191-
# 1F19A, U+1F1A5-1F1AC) and non-NFKC Roman forms
# (U+2180-2188: CD/V/X/C/c/VI/L) likewise have no NFKC
# ASCII form, so they fold to their letter/word here
# before NFKC. Regional indicators (U+1F1E6-1F1FF: A-Z,
# no NFKC form, same enclosed class) likewise fold to
# their letter here. Squared forms already folded by NFKC
# (U+1F130-149, U+1F14A-14F, U+1F16A-16C, U+2160-217F,
# U+1F190) are left to NFKC. Enclosed Hangul ieung
# (U+3207, NFKC ``(o)``) folds to ``o`` here so
# ``passw<3207>rd`` joins ``password``. Parenthesized
# digits (U+2474-2487, NFKC ``(1)``..``(20)``) likewise
# fold to their digit(s) here, mirroring the parenthesized
# letters: ``credentia<2474>`` then reads ``credentia1``
# in primary and ``credential`` in alt (``1`` -> ``l``).
# Loop14 F4: the 39 enclosed digits with no NFKC digit
# form (double circled 1-9 U+24F5-24FD, negative circled
# 0 U+24FF, dingbat negative 1-9 U+2776-277E, dingbat
# circled sans 1-9 U+2780-2788, dingbat negative circled
# sans 1-9 U+278A-2792, dingbat circled/negative sans 0
# U+1F10B-1F10C) likewise fold to their digit here, so
# ``credentia<2776>`` reads ``credentia1``/``credential``
# and ``t<24FF>ken`` reads ``t0ken``/``tOken``. Their
# enclosed-number companions in the same ranges (10 and
# 11-20) fold to their number likewise.
_SPLIT_LETTERS = {
    '\u3207': 'o', '\ufe31': 'l',
    '\u013f': 'L', '\u0140': 'l', '\u2103': 'C',
    '\u2109': 'F', '\U0001f12a': 'S',
    '\U0001f18a': 'P', '\U0001f18b': 'IC', '\U0001f18c': 'PA',
    '\U0001f18d': 'SA', '\U0001f18e': 'AB', '\U0001f18f': 'WC',
    '\U0001f191': 'CL', '\U0001f192': 'COOL',
    '\U0001f193': 'FREE', '\U0001f194': 'ID',
    '\U0001f195': 'NEW', '\U0001f196': 'NG',
    '\U0001f197': 'OK', '\U0001f198': 'SOS',
    '\U0001f199': 'UP', '\U0001f19a': 'VS',
    '\U0001f1a5': 'd', '\U0001f1a6': 'HC',
    '\U0001f1a7': 'HDR', '\U0001f1a9': 'LOSSLESS',
    '\U0001f1aa': 'SHV', '\U0001f1ab': 'UHD',
    '\U0001f1ac': 'VOD',
    '\u2180': 'CD', '\u2181': 'V', '\u2182': 'X',
    '\u2183': 'C', '\u2184': 'c', '\u2185': 'VI',
    '\u2186': 'L', '\u2187': 'L', '\u2188': 'C',
}
for _offset in range(26):
  _SPLIT_LETTERS[chr(0x249C + _offset)] = chr(ord('a') + _offset)
  _SPLIT_LETTERS[chr(0x1F110 + _offset)] = chr(ord('A') + _offset)
  _SPLIT_LETTERS[chr(0x1F150 + _offset)] = chr(ord('A') + _offset)
  _SPLIT_LETTERS[chr(0x1F170 + _offset)] = chr(ord('A') + _offset)
  _SPLIT_LETTERS[chr(0x1F1E6 + _offset)] = chr(ord('A') + _offset)
for _number in range(1, 21):
  _SPLIT_LETTERS[chr(0x2473 + _number)] = str(_number)
for _number in range(1, 10):
  _SPLIT_LETTERS[chr(0x24F4 + _number)] = str(_number)
  _SPLIT_LETTERS[chr(0x2775 + _number)] = str(_number)
  _SPLIT_LETTERS[chr(0x277F + _number)] = str(_number)
  _SPLIT_LETTERS[chr(0x2789 + _number)] = str(_number)
for _number in range(11, 21):
  _SPLIT_LETTERS[chr(0x24E0 + _number)] = str(_number)
_SPLIT_LETTERS[chr(0x24FE)] = '10'
_SPLIT_LETTERS[chr(0x24FF)] = '0'
_SPLIT_LETTERS[chr(0x277F)] = '10'
_SPLIT_LETTERS[chr(0x2789)] = '10'
_SPLIT_LETTERS[chr(0x2793)] = '10'
_SPLIT_LETTERS[chr(0x1F10B)] = '0'
_SPLIT_LETTERS[chr(0x1F10C)] = '0'
_SPLIT_LETTER_TABLE = str.maketrans(_SPLIT_LETTERS)
del _offset, _number
# Alternate split skeleton: late Roman VI (U+2185) is
# used both as the digraph ``VI`` (``pr`` + VI + ``ate``
# -> ``private``) and as a single ``V`` (``pri`` + VI +
# ``ate`` -> ``private``). Detection tries both the
# primary ``VI`` fold and this alternate ``V`` fold so
# neither spelling leaks (mirrors the Cyrillic ya
# alternate above).
_SPLIT_LETTERS_ALT = dict(_SPLIT_LETTERS)
_SPLIT_LETTERS_ALT['\u2185'] = 'V'
# Alternate folds for UTS39 sources whose NFKC form is a
# different letter (script/math/fullwidth ``I`` -> ``I``,
# lunate sigma -> ``s``/sigma): the primary NFKC fold
# covers the ``I``/``s`` use, this alternate covers the
# confusable ``l``/``c`` use, so both spellings redact.
_SPLIT_LETTERS_ALT.update({
    '\u03f2': 'c', '\u03f9': 'C', '\u2110': 'l',
    '\u2111': 'l', '\uff29': 'l', '\U0001d408': 'l',
    '\U0001d43c': 'l', '\U0001d470': 'l', '\U0001d4d8': 'l',
    '\U0001d540': 'l', '\U0001d574': 'l', '\U0001d5a8': 'l',
    '\U0001d5dc': 'l', '\U0001d610': 'l', '\U0001d644': 'l',
    '\U0001d678': 'l', '\U0001d6b0': 'l', '\U0001d6ea': 'l',
    '\U0001d724': 'l', '\U0001d75e': 'l', '\U0001d798': 'l',
})
_SPLIT_LETTER_TABLE_ALT = str.maketrans(_SPLIT_LETTERS_ALT)

# Single glyphs whose NFKC decomposition is a space plus
# non-spacing marks (space+Mn): dropping the marks after
# NFKC leaves a spurious space splitting a marker
# (``p¨assword`` -> ``p assword``). Dropping the source
# glyph before NFKC joins the marker, while surrounding
# spaces in spaced prose are preserved.
_SPACE_MARK_CPS = (
  0x00A8, 0x00AF, 0x00B4, 0x00B8, 0x02D8, 0x02D9, 0x02DA,
  0x02DB, 0x02DC, 0x02DD, 0x037A, 0x0384, 0x0385, 0x1FBD,
  0x1FBF, 0x1FC0, 0x1FC1, 0x1FCD, 0x1FCE, 0x1FCF, 0x1FDD,
  0x1FDE, 0x1FDF, 0x1FED, 0x1FEE, 0x1FFD, 0x1FFE, 0x2017,
  0x203E, 0x309B, 0x309C, 0xFC5E, 0xFC5F, 0xFC60, 0xFC61,
  0xFC62, 0xFC63, 0xFE49, 0xFE4A, 0xFE4B, 0xFE4C, 0xFE70,
  0xFE72, 0xFE74, 0xFE76, 0xFE78, 0xFE7A, 0xFE7C, 0xFE7E,
  0xFFE3,
)
_SPACE_MARK_TABLE = str.maketrans(
    {chr(cp): '' for cp in _SPACE_MARK_CPS})

# Loop14 F5: a glued ASCII parenthesized digit inside a
# word (``credentia(1)``, ``t(0)ken``) folds to the digit,
# mirroring the single-glyph parenthesized digits above:
# primary reads ``credentia1``/``t0ken`` and alt reads
# ``credential``/``tOken``. The lookbehind requires a
# directly preceding alphanumeric, so spaced ``(1)`` /
# ``(a)`` / ``(world)`` in prose are preserved.
_GLUED_PAREN_DIGIT_RE = re.compile(r'(?<=[A-Za-z0-9])\((\d+)\)')


def _normalize_unicode(text, table=None):
  '''NFKC-normalize ``text`` and strip zero-width characters.

  Fullwidth homoglyphs (``ｐａｓｓｗｏｒｄ``) fold to ASCII,
  Cyrillic/Greek confusables (``pаssword`` with U+0430) are
  mapped to their Latin twins, and zero-width joiners/spaces
  inserted inside a marker (``pass\\u200bword``) are removed,
  so marker matching cannot be evaded by unicode lookalikes.
  Accented Latin (``á``), IPA alpha (``ɑ``), Cyrillic ya
  (``я``) and combining marks (``a`` + U+0301, which NFKC
  composes to ``á`` and the table then folds to ``a``, plus
  any surviving non-spacing mark on a non-composable base)
  are folded the same way. Format characters (category
  ``Cf``, e.g. LRM U+200E, RLM U+200F, bidi U+202A-E,
  isolates U+2066-69, U+2061-64) are dropped so marker
  matching cannot be evaded by invisible bidi/format
  characters. Modifier letters (category ``Lm``, e.g.
  apostrophe U+02BC from ``ŉ`` U+0149 -> ``ʼn`` and right
  half ring U+02BE from ``ẚ`` U+1E9A -> ``aʾ``) and
  non-ASCII modifier symbols (category ``Sk`` above
  ASCII, e.g. U+02C2) are dropped after NFKC/translate,
  before marker search, so a separator introduced by
  NFKC cannot split a marker (``tokeŉ`` -> ``token``).
  Single glyphs whose NFKC is a space plus non-spacing
  marks (see ``_SPACE_MARK_CPS``) are dropped before
  NFKC, so the spurious space cannot split a marker
  either; spaces actually typed in prose are kept.
  Trade-off:
  legitimate modifier letters/symbols (IPA, tone marks,
  Arabic tatweel, Japanese prolonged sound mark, emoji
  skin-tone modifiers) are dropped from normalized
  output, as are single-glyph parenthesized/degree
  letters folded to the bare letter; ASCII ``^``/`` ` ``,
  spaced parentheses and other punctuation are kept.
  '''
  if table is None:
    table = _CONFUSABLE_TABLE
  split_table = (
      _SPLIT_LETTER_TABLE_ALT
      if table is _CONFUSABLE_TABLE_ALT
      else _SPLIT_LETTER_TABLE)
  text = text.translate(split_table)
  text = _GLUED_PAREN_DIGIT_RE.sub(r'\1', text)
  text = text.translate(_SPACE_MARK_TABLE)
  normalized = unicodedata.normalize('NFKC', text)
  normalized = normalized.translate(table)
  normalized = _drop_nonspacing_marks(normalized)
  return _drop_format_and_modifier_chars(normalized)


def _drop_nonspacing_marks(normalized):
  '''NFD-decompose and drop non-spacing marks, if needed.

  A combining acute on a base NFKC composes to a
  precomposed letter outside the table (``p`` + U+0301
  -> ``ṕ`` U+1E55) is folded back to its base letter,
  as are any surviving combining marks on
  non-composable bases. Split from
  ``_normalize_unicode`` (loop14).
  '''
  if any(unicodedata.category(char) == 'Mn'
         or unicodedata.combining(char)
         or ord(char) > 127 for char in normalized):
    normalized = ''.join(
        char for char in unicodedata.normalize('NFD', normalized)
        if unicodedata.category(char) != 'Mn')
  return normalized


def _drop_format_and_modifier_chars(normalized):
  '''Drop zero-width, format, modifier chars.

  Post-NFKC cleanup: zero-width chars, format
  characters (``Cf``), modifier letters (``Lm``) and
  non-ASCII modifier symbols (``Sk``) are dropped so
  an NFKC-introduced (or directly inserted) separator
  cannot split a marker. Split from
  ``_normalize_unicode`` (loop14).
  '''
  for char in _ZERO_WIDTH:
    normalized = normalized.replace(char, '')
  normalized = ''.join(
      char for char in normalized
      if unicodedata.category(char) != 'Cf')
  normalized = ''.join(
      char for char in normalized
      if unicodedata.category(char) != 'Lm'
      and not (unicodedata.category(char) == 'Sk'
               and ord(char) > 127))
  return normalized


def _normalize_key(key, table=None):
  '''Return ``key`` lowercased with non-alphanumerics stripped.

  Loop17 P2: intra-marker filler tokens (``very``/``is``/...)
  are dropped, so ``api very key`` normalizes to ``apikey``
  exactly as ``api_key`` does. Loop17-arch H1/H2/F2:
  ``-ly`` adverbs (``tremendously``) and glued filler
  substrings flanked by alphanumerics (``apiverykey`` ->
  ``apikey`` via ``_strip_glued_filler``) are dropped
  too, so a glued compound is a marker exactly as a
  spaced one is.
  '''
  if isinstance(key, (bytes, bytearray)):
    key = bytes(key).decode('utf-8', errors='replace')
  try:
    text = str(key)
  except Exception:
    # Hostile __str__/__repr__ (e.g. a set element reached
    # via ``str(list)``) must not crash marker detection.
    return ''
  # Loop20 P4: cap before unicode/hybrid work.
  if len(text) > _MAX_KEY_LEN:
    text = text[:_MAX_KEY_LEN]
  normalized = _normalize_unicode(text, table).lower()
  tokens = re.findall(r'[a-z0-9]+', normalized)
  kept = [tok for tok in tokens
          if tok not in _INTRA_FILLER_SET
          and not _is_ly_adverb(tok)]
  return _strip_glued_filler(''.join(kept))


_NORMALIZED_MARKERS = tuple(dict.fromkeys(
    _normalize_key(marker) for marker in _SECRET_MARKERS))


def _safe_str(text):
  '''Return ``str(text)`` or ``'<unrepresentable>'``.'''
  try:
    return str(text)
  except Exception:
    return '<unrepresentable>'


def _safe_repr(value):
  '''Return ``repr(value)`` or ``'<unrepresentable>'``.

  Never falls back to ``str(value)``: a hostile ``__repr__``
  that raises must not cause a second, secret-leaking
  ``__str__`` execution downstream.
  '''
  try:
    return repr(value)
  except Exception:
    return '<unrepresentable>'


def _hashable(value):
  '''Return ``value`` if hashable, else its safe repr string.'''
  try:
    hash(value)
  except TypeError:
    return _safe_repr(value)
  return value


def _key_as_text(key):
  '''Return ``key`` as text, or ``None`` if unrepresentable.

  Bytes decode UTF-8 with replacement, anything else
  goes through ``str``; a hostile ``__str__`` that
  raises yields ``None`` so callers fall through to the
  capped checks (which handle it as before).
  '''
  if isinstance(key, (bytes, bytearray)):
    return bytes(key).decode('utf-8', errors='replace')
  try:
    return str(key)
  except Exception:
    return None


_NON_ALNUM_RE = re.compile(r'[^a-z0-9]+')


def _key_middle_has_marker(text):
  '''Return True when a plain marker sits mid-key.

  Over-cap keys are checked on head/tail windows by
  ``_is_secret_key``; a plain (contiguous or
  separator-split) marker in the middle would fall
  between the windows. This scan is C-level only
  (lower, strip non-alphanumerics, substring) so it
  stays linear and fast on huge keys without the
  unicode/hybrid work the windows cap. Filler/unicode/
  hybrid forms in the middle are covered at the value
  layer by the truncated-key fail-closed rule in
  ``_scrub_str_key``/``_scrub_bytes_key``.
  '''
  stripped = _NON_ALNUM_RE.sub('', text.lower())
  return any(marker in stripped for marker in _NORMALIZED_MARKERS)


def _is_secret_key(key):
  '''Return True when ``key`` looks secret-bearing.

  Checks primary and alt skeletons first, then a
  per-letter hybrid reading (loop12 F1): a key mixing
  primary and alt readings in one word (``credentIaI``,
  ``ya-pI_key``) matches neither global skeleton, but a
  hybrid reading does, via ``_hybrid_contains``.
  Cycle7 Q2 (F-SCRUB-LONGKEY): an over-cap key is
  checked on its head and tail windows (plus a plain
  mid-key scan), so a marker past the 8192 cap cannot
  hide by truncation; each window stays within the cap,
  keeping detection bounded work on huge keys.
  '''
  text = _key_as_text(key)
  if text is not None and len(text) > _MAX_KEY_LEN:
    if _is_secret_key(text[:_MAX_KEY_LEN]):
      return True
    if _is_secret_key(text[-_MAX_KEY_LEN:]):
      return True
    return _key_middle_has_marker(text)
  normalized = _normalize_key(key)
  if any(marker in normalized for marker in _NORMALIZED_MARKERS):
    return True
  alt = _normalize_key(key, _CONFUSABLE_TABLE_ALT)
  if any(marker in alt for marker in _NORMALIZED_MARKERS):
    return True
  if isinstance(key, (bytes, bytearray)):
    text = bytes(key).decode('utf-8', errors='replace')
  else:
    try:
      text = str(key)
    except Exception:
      return False
  # Loop20 P4: cap before hybrid DP.
  if len(text) > _MAX_KEY_LEN:
    text = text[:_MAX_KEY_LEN]
  return _hybrid_key_reading(
    text, _NORMALIZED_MARKERS) is not None


def _safe_list(container):
  '''Return ``list(container)`` capped, or ``None`` on failure.

  Iteration is capped at ``_MAX_ITEMS`` via ``islice`` so a
  hostile/infinite container (tuple/list subclass whose
  ``__iter__`` never ends) cannot hang scrubbing. A hostile
  container whose ``__iter__`` raises must not crash
  scrubbing: callers fall back to a safe coercion on
  ``None``.
  '''
  try:
    return list(itertools.islice(container, _MAX_ITEMS))
  except Exception:
    return None


_HYBRID_CHAR_CACHE = {}


def _char_hybrid_segments(char):
  '''Return ``(primary, alt)`` normalized segments for a char.

  Per original character, so adjacent ambiguous glyphs
  stay separate units (``_align_opcodes`` would merge
  them into one replace block and lose per-letter
  mixing). Cached: the same glyph recurs often.
  Loop14: a segment that is digit ``1`` in primary and
  ``l`` in alt (ASCII ``1``, enclosed/parenthesized
  ``1``) is also read as ``I`` in the hybrid layer, so
  ``1`` standing for ``i`` (``credent1al``) and for
  ``l`` (``credentia1``) both form ``credential``.
  '''
  cached = _HYBRID_CHAR_CACHE.get(char)
  if cached is None:
    cached = (
      _normalize_unicode(char),
      _normalize_unicode(char, _CONFUSABLE_TABLE_ALT))
    if cached == ('1', 'l'):
      cached = ('I', 'l')
    _HYBRID_CHAR_CACHE[char] = cached
  return cached


def _hybrid_units_from_original(text):
  '''Return per-char ``(primary, alt)`` units for ``text``.'''
  return [_char_hybrid_segments(char) for char in text]


def _hybrid_exact(units, target):
  '''Return a hybrid reading equal to ``target``, or None.

  DP over units and target index; each unit contributes
  its primary or alt segment, backtracked to a concrete
  reading. Segment lengths may differ (``th``/``p``).
  '''
  layers = [{0: None}]
  for primary_seg, alt_seg in units:
    current = layers[-1]
    nxt = {}
    for prev_index in current:
      for seg in (primary_seg, alt_seg):
        if target[prev_index:prev_index + len(seg)] != seg:
          continue
        new_index = prev_index + len(seg)
        if new_index not in nxt:
          nxt[new_index] = (prev_index, seg)
    if not nxt:
      return None
    layers.append(nxt)
  if len(target) not in layers[-1]:
    return None
  parts = []
  index = len(target)
  for layer in reversed(layers[1:]):
    prev_index, seg = layer[index]
    parts.append(seg)
    index = prev_index
  return ''.join(reversed(parts))


def _hybrid_contains(units, target):
  '''Return a hybrid reading containing ``target``, or None.

  DP over units and marker progress (0..len(target), len
  is absorbing); each unit's chosen segment advances the
  progress character by character, starting a new match
  at any character. Backtracked to a concrete reading.
  '''
  size = len(target)
  layers = [{0: None}]
  for primary_seg, alt_seg in units:
    current = layers[-1]
    nxt = {}
    for prev_prog in current:
      for seg in (primary_seg, alt_seg):
        prog = prev_prog
        for char in seg:
          if prog >= size:
            prog = size
          elif prog > 0 and char == target[prog]:
            prog += 1
          elif char == target[0]:
            prog = 1
          else:
            prog = 0
        if prog not in nxt:
          nxt[prog] = (prev_prog, seg)
    if not nxt:
      return None
    layers.append(nxt)
  if size not in layers[-1]:
    return None
  parts = []
  prog = size
  for layer in reversed(layers[1:]):
    prev_prog, seg = layer[prog]
    parts.append(seg)
    prog = prev_prog
  return ''.join(reversed(parts))


def _hybrid_try_targets(units, targets, exact):
  '''Try hybrid targets, skipping impossible ones.

  Loop20 P4: a target using a char the key cannot
  produce can never form; skip its DP entirely.
  Split from ``_hybrid_key_reading`` (complexity).
  '''
  possible = set()
  for primary_seg, alt_seg in units:
    possible.update(primary_seg)
    possible.update(alt_seg)
  for target in targets:
    if not set(target) <= possible:
      continue
    if exact:
      reading = _hybrid_exact(units, target)
    else:
      reading = _hybrid_contains(units, target)
    if reading is not None:
      return reading
  return None


def _hybrid_key_reading(text, targets, exact=False):
  '''Return a hybrid key reading for original ``text``.

  Per original character, lowercased and stripped to
  alphanumeric (mirroring ``_normalize_key``), so a
  mixed-reading key word is a hybrid reading. Returns
  None when primary and alt skeletons already agree
  (caller checked them) or no target matches.
  '''
  if not isinstance(text, str):
    return None
  # Loop20 P4 + cycle7 Q2 (F-SCRUB-LONGKEY): cap
  # before hybrid DP, and check the tail window too so
  # a hybrid marker past the cap cannot hide.
  if len(text) > _MAX_KEY_LEN:
    reading = _hybrid_key_reading(
        text[:_MAX_KEY_LEN], targets, exact)
    if reading is not None:
      return reading
    return _hybrid_key_reading(
        text[-_MAX_KEY_LEN:], targets, exact)
  units = []
  ambiguous = False
  for char in text:
    primary_seg, alt_seg = _char_hybrid_segments(char)
    primary_seg = ''.join(
      c for c in primary_seg.lower() if c.isalnum())
    alt_seg = ''.join(
      c for c in alt_seg.lower() if c.isalnum())
    if primary_seg != alt_seg:
      ambiguous = True
    if primary_seg or alt_seg:
      units.append((primary_seg, alt_seg))
  if not ambiguous:
    return None
  return _hybrid_try_targets(units, targets, exact)


