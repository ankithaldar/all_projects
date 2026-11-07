#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Secret scrubbing leaf: stdlib-only, no telemetry or
governance imports.

Deep-scrubs secret-like keys (password/secret/token/api_key
and similar, matched on the normalized key name -- lowercased
with all non-alphanumerics stripped, so ``accessKey``,
``access_key`` and ``AWSAccessKeyId`` all match -- recursively)
so secrets never reach audit trails or telemetry contexts.
Free-text strings and string detail values are scrubbed for
``key=value`` secret patterns, space-separated
``password <value>`` forms (including an optional
``is``/``was`` copula and ``api key`` space/hyphen
variants), symbolic separators (any run of non-alphanumeric
characters or underscores, e.g. ``>>``, ``~``, ``#``,
``password_hunter2``), bare AWS access keys
(``AKIA...``/``ASIA...``), bare ``sk-...`` keys, and PEM
private-key blocks as well. String dict keys are scrubbed
the same way, and tuple/frozenset dict keys are scrubbed
recursively with their structure preserved, so secrets
carried in a key never reach the trail either.
Loop17 P1: unknown objects become ``{'type': name}``
only for CamelCase type name values (``_*[A-Z]...``);
lowercase-start, symbol or token-like names are
``REDACTED`` whole. P2: intra-marker filler includes
``well-known``/``wellknown``/``known`` and the filler
chain words, glued filler (``apiverykey``) is stripped
after normalization. P3: filler vocab includes
``tremendously``/``literally``/``honestly`` and any
``-ly`` adverb (``_is_ly_adverb``); this is
best-effort -- a non-``ly`` filler outside the vocab
is treated as the value itself (first token redacted).
P4: Bearer filler may carry a closing bracket.
Loop21: a short glued suffix (<= 3 chars) is a
marker extension, not the value, and chained glued
filler (``apiveryverykey``) is stripped in one
linear tokenization pass.
Cycle1 W1: bare ``key``/``hash`` are deliberately
not secret markers, as substrings or on their own.
The framework itself uses ``key_cols_list`` and
similar ordinary field names, so a blanket
``key``/``hash`` substring would redact legitimate
configuration; only unambiguous compounds such as
``api_key``, ``encryption_key`` and ``pin_code``
redact. Unambiguous compounds added in cycle1 are
``encryption_key``, ``master_key``, ``signing_key``,
``aws_key``, ``gcp_key``, ``otp`` and ``pin_code``/
``pin_number`` (``private_key`` already redacted).

This leaf exists so ``telemetry.exceptions`` can scrub
without importing ``governance.audit`` (governance code
may raise ``FrameworkError``; a leaf import cannot cycle).
Cross-layer callers use the public aliases at the end
of this module (``safe_str``, ``scrub_text``, ...);
the ``_`` names are kept as identical aliases.
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
    'filler', 'very2', '123')
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
_QUOTE_PAIRS = {'"': '"', "'": "'", '`': '`', '“': '”', '‘': '’'}
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


def _is_secret_key(key):
  '''Return True when ``key`` looks secret-bearing.

  Checks primary and alt skeletons first, then a
  per-letter hybrid reading (loop12 F1): a key mixing
  primary and alt readings in one word (``credentIaI``,
  ``ya-pI_key``) matches neither global skeleton, but a
  hybrid reading does, via ``_hybrid_contains``.
  '''
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
  return f"<{scrubbed_name}>", secret


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
    if char == '\\' and opening in ('"', "'", '`'):
      index += 2
      continue
    if (char == opening and opening in ('"', "'", '`')
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
    'especially'})

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


def _sub_spans(text, pos, redacted, pattern, mode):
  '''Apply one regex, recording its core span in ``redacted``.

  ``pos`` maps each char in ``text`` to its index in the
  original input (``None`` for inserted replacement
  chars), so a span recorded at any stage is in original
  coordinates even though earlier replacements shift
  lengths (``ß``/``þ``/Roman/squared expansions and
  ``REDACTED`` itself). ``mode`` is ``'whole'``,
  ``'url'`` or ``'bearer'``: the core is the whole
  match, URL group 2, or Bearer group 2, and preserved
  prefix/suffix chars keep their original positions.
  Replacement text and output are byte-identical to the
  equivalent ``re.sub``.
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
    else:
      parts.append(text[match.start(1):match.end(1)])
      new_pos.extend(pos[match.start(1):match.end(1)])
      parts.append(REDACTED)
      new_pos.extend([None] * len(REDACTED))
      core = match.span(2)
    if mode == 'whole':
      parts.append(REDACTED)
      new_pos.extend([None] * len(REDACTED))
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
  # Loop20 P4: cap before hybrid DP.
  if len(text) > _MAX_KEY_LEN:
    text = text[:_MAX_KEY_LEN]
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


def _scrub_text(text):
  '''Redact secret values in a free-text string.

  Args:
    text: String that may embed ``key=value`` secret patterns
      (or a bare ``Bearer <token>``).

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
  hybrid_full = _build_hybrid_text(
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


def _scrub_key_items(items, marker_bearing, seen_next, depth):
  '''Scrub each composite item, returning ``(items, secret)``.'''
  scrubbed_items = []
  secret = False
  for item in items:
    scrubbed_item, item_secret = _scrub_key(
        item, marker_bearing, seen_next, depth + 1)
    scrubbed_items.append(scrubbed_item)
    secret = secret or item_secret
  return scrubbed_items, secret


def _scrub_key(key, _inherited_marker=False, _seen=None, _depth=0):
  '''Return ``(scrubbed_key, secret_key)`` for any hashable key.

  String/bytes keys are scrubbed as text. Tuple/frozenset
  keys are scrubbed recursively, preserving type, length and
  (for tuples) order: elements that are bare secrets are
  redacted in place, and when the composite is
  marker-bearing (a marker element anywhere, or marker-
  parts joining to a marker across elements -- including
  interleaved or preceding data elements such as
  ``('HUNTER2', 'password')`` or ``('api', 'HUNTER2',
  'key')``) every non-marker, non-marker-part string/
  bytes element is redacted regardless of position,
  since its position relative to a marker cannot be
  trusted to be the value. Marker and marker-part
  elements are kept. ``secret_key`` is True when the key
  is marker-bearing or scrubbing changed it, so the
  associated value is redacted like a string key.
  ``_inherited_marker`` carries an outer composite's
  marker-bearing context into nested tuples, so a secret
  nested at any depth under a marker is redacted.
  Cyclic/deep keys yield ``'<cyclic>'``/``'<max-depth>'``
  with ``secret_key`` True, mirroring ``scrub_details``.
  '''
  if isinstance(key, str):
    return _scrub_str_key(key)
  if isinstance(key, (bytes, bytearray)):
    return _scrub_bytes_key(key)
  if isinstance(key, tuple):
    return _scrub_tuple_key(key, _inherited_marker, _seen, _depth)
  if isinstance(key, (frozenset, set)):
    return _scrub_set_key(key, _inherited_marker, _seen, _depth)
  if key is None or isinstance(key, (bool, int, float, complex)):
    return key, False
  # Unknown / hostile key: never keep the original object (its
  # ``__str__``/``__repr__`` would execute and could leak or
  # raise downstream). See ``_unknown_key``: always a
  # type-only placeholder; scrubbing decides the secret flag.
  return _unknown_key(key)


def _scrub_str_key(key):
  '''Scrub a str key (loop14 split).'''
  # Loop20 P4: cap key length before scrub/secret check.
  if len(key) > _MAX_KEY_LEN:
    key = key[:_MAX_KEY_LEN]
  scrubbed = _scrub_text(key)
  return scrubbed, _is_secret_key(key) or scrubbed != key


def _scrub_bytes_key(key):
  '''Scrub a bytes key (loop14 split).'''
  # Loop20 P4: cap decoded bytes key length.
  if len(key) > _MAX_KEY_LEN:
    key = bytes(key)[:_MAX_KEY_LEN]
  decoded = bytes(key).decode('utf-8', errors='replace')
  scrubbed = _scrub_text(decoded)
  return scrubbed, _is_secret_key(key) or scrubbed != decoded


def _finalize_tuple_key(items, scrubbed_items, marker_bearing):
  '''Redact non-marker tuple elements (loop14 split).'''
  final = []
  for idx, item in enumerate(items):
    scrubbed_item = scrubbed_items[idx]
    if (marker_bearing
        and isinstance(item, (str, bytes, bytearray))
        and _marker_part(item) is None
        and not _is_marker_element(item)):
      final.append(REDACTED)
    else:
      final.append(scrubbed_item)
  return final


def _scrub_tuple_key(key, _inherited_marker, _seen, _depth):
  '''Scrub a tuple key (loop14 split).'''
  if _seen is None:
    _seen = frozenset()
  if _depth > _MAX_COMPOSITE_DEPTH:
    return '<max-depth>', True
  if id(key) in _seen:
    return '<cyclic>', True
  seen_next = _seen | {id(key)}
  items = _safe_list(key)
  if items is None:
    return _unknown_key(key)
  marker_bearing = (
      _inherited_marker
      or _composite_marker_bearing(
          items, True, _seen=_seen, _depth=_depth))
  scrubbed_items, secret = _scrub_key_items(
      items, marker_bearing, seen_next, _depth)
  scrubbed_key = tuple(
    _finalize_tuple_key(items, scrubbed_items, marker_bearing))
  try:
    changed = scrubbed_key != tuple(items)
  except Exception:
    changed = True
  return scrubbed_key, marker_bearing or secret or changed


def _finalize_set_key(items, scrubbed_items, marker_bearing):
  '''Redact non-marker set elements (loop14 split).'''
  final = set()
  for item, scrubbed_item in zip(items, scrubbed_items):
    if (isinstance(item, (str, bytes, bytearray))
        and not _is_marker_element(item)
        and _marker_part(item) is None
        and marker_bearing):
      final.add(REDACTED)
    else:
      final.add(_hashable(scrubbed_item))
  return final


def _scrub_set_key(key, _inherited_marker, _seen, _depth):
  '''Scrub a frozenset/set key (loop14 split).'''
  if _seen is None:
    _seen = frozenset()
  if _depth > _MAX_COMPOSITE_DEPTH:
    return '<max-depth>', True
  if id(key) in _seen:
    return '<cyclic>', True
  seen_next = _seen | {id(key)}
  items = _safe_list(key)
  if items is None:
    return _unknown_key(key)
  marker_bearing = (
      _inherited_marker
      or _composite_marker_bearing(
          items, False, _seen=_seen, _depth=_depth))
  scrubbed_items, secret = _scrub_key_items(
      items, marker_bearing, seen_next, _depth)
  final = _finalize_set_key(items, scrubbed_items, marker_bearing)
  scrubbed_key = (
      frozenset(final) if isinstance(key, frozenset) else final)
  try:
    original = (
        frozenset(items) if isinstance(key, frozenset)
        else set(items))
    changed = scrubbed_key != original
  except Exception:
    changed = True
  return scrubbed_key, marker_bearing or secret or changed


def _scrub_container_values(
    container, ordered, _inherited_marker=False,
    _seen=None, _depth=0):
  '''Return scrubbed items for a list/tuple/set/frozenset value.

  Same marker-bearing rule as ``_scrub_key`` for keys: when
  the container is marker-bearing (a marker element anywhere,
  or marker-parts joining to a marker across elements), every
  non-marker, non-marker-part string/bytes element is
  redacted, regardless of position, so a composite secret
  carried in a value (``('password', 'HUNTER2')``,
  ``('api', 'HUNTER2', 'key')``) never reaches the audit
  trail. ``_inherited_marker`` carries an outer composite's
  marker-bearing context into nested containers at any
  depth. Other elements are recursed via ``scrub_details``,
  so nested mappings/containers and bare secrets inside
  elements are still scrubbed. ``_seen`` holds ids on the
  current path and ``_depth`` caps recursion at
  ``_MAX_COMPOSITE_DEPTH``: a cyclic element yields
  ``'<cyclic>'`` and a too-deep element ``'<max-depth>'``
  (same placeholders as ``telemetry.exceptions._coerce``)
  instead of raising ``RecursionError``.
  '''
  # Probe one past the cap so a truncated sentinel already
  # present (e.g. from ``telemetry.exceptions._coerce``) at
  # index ``_MAX_ITEMS`` is preserved, while hostile /
  # infinite iteration is still bounded at ``_MAX_ITEMS + 1``.
  if _seen is None:
    _seen = frozenset()
  seen_next = _seen | {id(container)}
  try:
    items = list(
        itertools.islice(container, _MAX_ITEMS + 1))
  except Exception:
    return [{'type': _scrub_type_name(type(container).__name__)}]
  marker_bearing = (
      _inherited_marker
      or _composite_marker_bearing(
          items, ordered, _seen=_seen, _depth=_depth))
  return [
    _scrub_container_item(item, marker_bearing, seen_next, _depth)
    for item in items]


def _scrub_container_item(item, marker_bearing, seen_next, _depth):
  '''Scrub one container item (loop14 split).'''
  if (marker_bearing
      and isinstance(item, (str, bytes, bytearray))
      and _marker_part(item) is None
      and not _is_marker_element(item)):
    return REDACTED
  if isinstance(item, (list, tuple)):
    if id(item) in seen_next:
      return '<cyclic>'
    if _depth + 1 > _MAX_COMPOSITE_DEPTH:
      return '<max-depth>'
    if isinstance(item, tuple):
      return tuple(_scrub_container_values(
        item, True, marker_bearing, seen_next, _depth + 1))
    return _scrub_container_values(
      item, True, marker_bearing, seen_next, _depth + 1)
  if isinstance(item, (frozenset, set)):
    return _scrub_container_set_item(
      item, marker_bearing, seen_next, _depth)
  if isinstance(item, dict):
    return scrub_details(item, marker_bearing, seen_next, _depth + 1)
  return scrub_details(item, _seen=seen_next, _depth=_depth + 1)


def _scrub_container_set_item(item, marker_bearing, seen_next, _depth):
  '''Scrub a set/frozenset item, keeping it hashable.'''
  if id(item) in seen_next:
    return '<cyclic>'
  if _depth + 1 > _MAX_COMPOSITE_DEPTH:
    return '<max-depth>'
  # Set elements must stay hashable: an unknown element
  # scrubs to ``{'type': name}`` (a dict), so wrap every
  # scrubbed element with ``_hashable`` (safe repr).
  scrubbed_set = [
    _hashable(v) for v in _scrub_container_values(
      item, False, marker_bearing, seen_next, _depth + 1)]
  return (frozenset(scrubbed_set)
          if isinstance(item, frozenset) else set(scrubbed_set))


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


def scrub_details(
    details, _inherited_marker=False, _seen=None, _depth=0):
  '''Return a deep copy of ``details`` with secrets redacted.

  Args:
    details: Mapping to scrub; nested mappings and lists are
      scrubbed recursively. String values (and a bare string
      ``details``) are scrubbed for ``key=value`` secret
      patterns; scalar values are returned as-is.

  Returns:
    A new object; the input is never mutated. Only
    dict/list/tuple/set/frozenset/scalars/str/bytes are
    traversed: unknown objects are never deep-copied (which
    could execute ``__deepcopy__``/``__reduce__`` and fail on
    file handles) and are replaced by ``{'type': name}``.
    Cyclic containers (a dict/list/tuple/set referring to an
    ancestor on the current path) yield the string
    ``'<cyclic>'`` for the repeat, and nesting past
    ``_MAX_COMPOSITE_DEPTH`` yields ``'<max-depth>'`` --
    the same placeholders as ``_flatten_composite`` and
    ``telemetry.exceptions._coerce`` -- instead of raising
    ``RecursionError``. ``_seen``/``_depth`` carry that
    path state through dict/list/tuple/set/frozenset
    recursion.
  '''
  if _seen is None:
    _seen = frozenset()
  if isinstance(details, (dict, list, tuple, frozenset, set)):
    if _depth > _MAX_COMPOSITE_DEPTH:
      return '<max-depth>'
    if id(details) in _seen:
      return '<cyclic>'
    _seen = _seen | {id(details)}
  if isinstance(details, dict):
    return _scrub_dict_details(
      details, _inherited_marker, _seen, _depth)
  if isinstance(details, (list, tuple, frozenset, set)):
    return _scrub_container_details(
      details, _inherited_marker, _seen, _depth)
  return _scrub_leaf_details(details, _inherited_marker)


def _scrub_dict_details(details, _inherited_marker, _seen, _depth):
  '''Scrub a dict ``details`` (loop14 split).'''
  try:
    # ``+ 1`` preserves a truncated sentinel key already
    # present at index ``_MAX_ITEMS`` (see above).
    items_list = list(
        itertools.islice(details.items(), _MAX_ITEMS + 1))
  except Exception:
    return {'type': _scrub_type_name(type(details).__name__)}
  try:
    items_list = [(key, value) for key, value in items_list]
  except (TypeError, ValueError):
    return {'type': _scrub_type_name(type(details).__name__)}
  scrubbed = {}
  for key, value in items_list:
    # Keys may themselves carry secrets (bare ``AKIA...`` /
    # ``sk-...`` / PEM, ``key=value`` or ``password <value>``
    # forms, including inside tuple/frozenset keys): scrub
    # the key recursively too, and redact its value when the
    # key was secret-bearing. Under an inherited marker (a
    # dict nested in a marker-bearing container) the marker
    # is propagated into key scrubbing, so a tuple key's
    # non-marker elements are redacted, and a plain
    # non-marker string/bytes key -- itself secret payload
    # under the marker -- is replaced by ``REDACTED``
    # instead of leaking. Marker/marker-part key names are
    # kept; every value leaf is likewise redacted,
    # recursively.
    scrubbed_key, secret_key = _scrub_key(
        key, _inherited_marker, _seen, _depth)
    if (_inherited_marker
        and isinstance(key, (str, bytes, bytearray))
        and not _is_marker_element(key)
        and _marker_part(key) is None):
      scrubbed[REDACTED] = scrub_details(
          value, True, _seen, _depth + 1)
    elif secret_key:
      scrubbed[_hashable(scrubbed_key)] = REDACTED
    elif _inherited_marker:
      scrubbed[_hashable(scrubbed_key)] = scrub_details(
          value, True, _seen, _depth + 1)
    else:
      scrubbed[_hashable(scrubbed_key)] = scrub_details(
          value, False, _seen, _depth + 1)
  return scrubbed


def _scrub_container_details(
    details, _inherited_marker, _seen, _depth):
  '''Scrub a list/tuple/set/frozenset (loop14 split).'''
  if isinstance(details, list):
    return _scrub_container_values(
        details, True, _inherited_marker, _seen, _depth)
  if isinstance(details, tuple):
    return tuple(_scrub_container_values(
        details, True, _inherited_marker, _seen, _depth))
  if isinstance(details, frozenset):
    return frozenset(
        _hashable(v) for v in _scrub_container_values(
            details, False, _inherited_marker, _seen, _depth))
  return set(
      _hashable(v) for v in _scrub_container_values(
          details, False, _inherited_marker, _seen, _depth))


def _scrub_leaf_details(details, _inherited_marker):
  '''Scrub a scalar/bytes/str leaf (loop14 split).'''
  if isinstance(details, (bytes, bytearray)):
    if _inherited_marker:
      return REDACTED
    return _decode_bytes_scrubbed(details)
  if isinstance(details, str):
    return REDACTED if _inherited_marker else _scrub_text(details)
  if details is None or isinstance(
      details, (bool, int, float, complex)):
    return details
  return {'type': _scrub_type_name(type(details).__name__)}


# Public aliases for cross-layer callers (loop17-arch
# medium): ``governance.audit`` and ``telemetry``
# import these names, never the ``_`` private twins.
# The ``_`` names are kept as identical aliases so
# existing imports keep working.
safe_str = _safe_str
safe_repr = _safe_repr
safe_list = _safe_list
scrub_text = _scrub_text
scrub_type_name = _scrub_type_name
unknown_key = _unknown_key
is_secret_key = _is_secret_key
normalize_key = _normalize_key
normalize_unicode = _normalize_unicode
decode_bytes_scrubbed = _decode_bytes_scrubbed
composite_marker_bearing = _composite_marker_bearing
marker_part = _marker_part
is_marker_element = _is_marker_element
flatten_composite = _flatten_composite
MAX_ITEMS = _MAX_ITEMS
MAX_COMPOSITE_DEPTH = _MAX_COMPOSITE_DEPTH
