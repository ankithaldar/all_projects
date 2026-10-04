#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''NSE order tagging: the 15-digit NNF ID and its 13th-digit algo flag.

This is the field that makes an order attributable, so it is the field
that has to be right. Two things about it are worth stating before the
structure:

  * It is **NSE's** structure, not SEBI's. SEBI says only that a unique
    identifier is provided by the stock exchange. NSE implements it as
    a 15-digit NNF ID plus a separate ALGO ID field, and the pre-trade
    validation below is NSE/FAOP/69296, not a SEBI circular.
  * Digits 1-12 identify the platform, digit 13 is the algo flag, and
    digits 14 and 15 are not decoded here at all. A 15-digit field with
    three undocumented positions is a good reason to validate strictly
    and refuse to guess.

What is verified, and what is not:

  * Digits 1-12: four sentinel platforms are documented. Client Direct
    API is ``444444444444``, which is the case this project runs in when
    it places orders through a broker's API, and it is the case where
    the 13th digit **must** always be an algo flag.
  * Digit 13: 0, 1, 2, 3, 7 and 8 were confirmed against exchange
    documents in this project's research pass. Digits 4, 5 and 6 appear
    in the NNF field documentation with meanings -- inter-exchange algo,
    RMS square-off, after-market -- and one research pass marked them
    only partially documented, so they are accepted but flagged here
    rather than presented as settled.
  * Digit 9 has no documented meaning and is **rejected**. An
    undocumented digit is not a digit to guess at: the whole cost of this
    module is a mis-tagged order becoming unattributable in an audit
    trail, and refusing is cheaper than being wrong.

So :data:`documented_flag_digits` is what this module accepts,
:data:`verified_flag_digits` is the subset confirmed against a primary
exchange document, and :data:`unverified_flag_digits` is rejected with a
distinct error message. No full digit-to-meaning table is hardcoded,
because the project's own research says not to without the current NNF
protocol CD.
'''

from __future__ import annotations

from stock_rl.compliance.retention import non_algo_id

__all__ = [
  'AlgoTagError',
  'algo_flag_digit',
  'algo_flag_digits',
  'algo_flag_position',
  'allows_algo',
  'build_nnf_id',
  'client_direct_api_platform',
  'documented_flag_digits',
  'is_algo_flag',
  'known_platform_algo_allowed',
  'nnf_length',
  'non_algo_flag_digits',
  'platform_of',
  'unverified_flag_digits',
  'validate_algo_tag',
  'verified_flag_digits',
]

#: Width of an NNF ID, in digits. Part of the field width rather than a
#: free choice, so it is named once and used everywhere.
nnf_length = 15

#: Position of the algo flag within the NNF ID, one-based. Counting from
#: one because that is how the exchange documentation counts, and an
#: off-by-one here would mis-tag every order.
algo_flag_position = 13

#: The Client Direct API platform, i.e. this project trading through a
#: broker's API on its own account. Algo trading is permitted, and the
#: 13th digit must always be an algo flag.
client_direct_api_platform = '444444444444'

#: Sentinel platforms documented by NSE, mapped to whether algo orders
#: are permitted on them. CTCL is absent because it has no sentinel:
#: its first six digits are the client PIN, so there is nothing to match.
known_platform_algo_allowed: dict[str, bool] = {
  '111111111111': False,  # IBT
  '222222222222': True,   # DMA
  '333333333333': False,  # STWT
  client_direct_api_platform: True,
}

#: Flag digits confirmed against a primary exchange document.
verified_flag_digits = frozenset({0, 1, 2, 3, 7, 8})

#: Flag digits this module accepts: the verified set plus 4, 5 and 6,
#: which are described in the NNF field documentation but were flagged
#: as only partially documented by one research pass.
documented_flag_digits = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8})

#: Digits with no documented meaning, rejected rather than guessed at.
unverified_flag_digits = frozenset(range(10)) - documented_flag_digits

#: Flag digits that mark an order as algo, per NSE/FAOP/69296. A 0 is a
#: plain algo order, 2 is an algo routed via SOR and 4 is an
#: inter-exchange algo.
algo_flag_digits = frozenset({0, 2, 4})

#: Flag digits that mark an order as NOT algo: 1 non-algo, 3 non-algo
#: via SOR, 5 RMS square-off, 6 after-market, 7 basket, 8 batch upload.
#: Each of these must carry Algo ID "0", which is why they are a
#: different validation rule and not merely a different label.
non_algo_flag_digits = documented_flag_digits - algo_flag_digits


class AlgoTagError(ValueError):
  '''Raised when an NNF ID or its pairing with an Algo ID is invalid.

  A ValueError subclass because the inputs are strings, with its own
  type so that an order-routing layer can reject the order and log the
  reason without matching on message text.
  '''


def platform_of(nnf_id: str) -> str:
  '''Return the 12-digit platform prefix of an NNF ID.

  Args:
    nnf_id: A 15-digit NNF ID.

  Returns:
    Digits 1 to 12 as a string, so a leading zero survives.

  Raises:
    AlgoTagError: If the NNF ID is not 15 decimal digits.
  '''
  _require_nnf(nnf_id)
  return nnf_id[:12]


def algo_flag_digit(nnf_id: str) -> int:
  '''Return the 13th digit of an NNF ID, the algo flag.

  Args:
    nnf_id: A 15-digit NNF ID.

  Returns:
    The flag as an int, from 0 to 9. The digit is returned whether or not
    it is documented, so a caller can log what was actually sent; use
    :func:`validate_algo_tag` to require a documented value.

  Raises:
    AlgoTagError: If the NNF ID is not 15 decimal digits.
  '''
  _require_nnf(nnf_id)
  return int(nnf_id[algo_flag_position - 1])


def allows_algo(platform: str) -> bool | None:
  '''Return whether a platform prefix may carry algo orders.

  Args:
    platform: The 12-digit platform prefix from :func:`platform_of`.

  Returns:
    True or False for a documented sentinel platform, and None for a
    prefix that is not one. None rather than False, because the
    documented cases include CTCL, whose first six digits are a client
    PIN and which therefore has no sentinel to match; returning False
    there would refuse algo orders on a platform the exchange permits
    them on.

  Raises:
    AlgoTagError: If the prefix is not 12 decimal digits.
  '''
  if len(platform) != 12 or not platform.isdecimal():
    raise AlgoTagError(
      f'platform prefix must be 12 digits, got {platform!r}')
  return known_platform_algo_allowed.get(platform)


def is_algo_flag(flag: int) -> bool:
  '''Return True if a flag digit marks the order as an algo order.

  Args:
    flag: The 13th digit of an NNF ID.

  Returns:
    True for 0, 2 and 4. False for everything else, including the
    undocumented digits, which is the safe direction: an unknown flag is
    not treated as an algo, so it also cannot silently satisfy the algo
    half of the pairing rule.
  '''
  return flag in algo_flag_digits


def build_nnf_id(platform: str, flag: int, tail: str = '00') -> str:
  '''Assemble a 15-digit NNF ID from its parts.

  Exposed so that the validation above can be tested against IDs this
  project would actually construct, rather than against hand-typed
  strings that might not be well formed. Building an ID is not the same
  as holding one the exchange will accept: the Algo ID field is separate
  and cannot be derived here.

  Args:
    platform: 12-digit platform prefix, typically
      :data:`client_direct_api_platform`.
    flag: The algo flag digit, 0 to 9.
    tail: The two trailing digits. They are passed through unchanged
      because they are not decoded by this module, and inventing a
      meaning for them is what the module docstring warns against.

  Returns:
    The assembled 15-digit NNF ID.

  Raises:
    AlgoTagError: If the platform is not 12 digits, the flag is not a
      single decimal digit, or the tail is not two digits.
  '''
  if len(platform) != 12 or not platform.isdecimal():
    raise AlgoTagError(
      f'platform prefix must be 12 digits, got {platform!r}')
  if not 0 <= flag <= 9:
    raise AlgoTagError(f'algo flag must be a digit 0-9, got {flag}')
  if len(tail) != 2 or not tail.isdecimal():
    raise AlgoTagError(f'tail must be two digits, got {tail!r}')
  return f'{platform}{flag}{tail}'


def validate_algo_tag(nnf_id: str, algo_id: str) -> str:
  '''Validate an NNF ID against the Algo ID it is paired with.

  The rule, from NSE/FAOP/69296:

    * a flag in {0, 2, 4} -- algo, algo via SOR, inter-exchange algo --
      requires an Algo ID that is not the non-algo sentinel, because an
      algo order must name the algo that produced it;
    * a flag in {1, 3, 5, 6, 7, 8} requires Algo ID ``0``, because a
      non-algo order has no algo to name.

  Both halves are rejections. Sending algo ID ``0`` with an algo flag
  produces an order the audit trail cannot attribute, which is the exact
  failure the tagging requirement exists to prevent; sending a real Algo
  ID with a non-algo flag claims an algo ran when none did, which is
  worse because it is a false record.

  A flag of 9, or any other undocumented digit, is rejected before
  either rule is applied. There is no third option here: the choice is
  between refusing the order and mis-describing it.

  Args:
    nnf_id: The 15-digit NNF ID to be sent.
    algo_id: The Algo ID field value, as validated by
      ``validate_algo_id`` in this package.

  Returns:
    The NNF ID, unchanged, so a caller can write
    ``nnf_id = validate_algo_tag(nnf_id, algo_id)``.

  Raises:
    AlgoTagError: If the NNF ID is malformed, the flag digit is
      undocumented, or the flag and Algo ID are inconsistent.
  '''
  _require_nnf(nnf_id)
  flag = algo_flag_digit(nnf_id)
  # unverified_flag_digits is defined as the complement of
  # documented_flag_digits, so this one check rejects every undocumented
  # digit and no second membership test is needed.
  if flag in unverified_flag_digits:
    raise AlgoTagError(
      f'NNF algo flag {flag} has no documented meaning; refusing rather '
      'than mis-tagging the order')
  claims_algo = is_algo_flag(flag)
  claims_algo_id = algo_id != non_algo_id
  if claims_algo and not claims_algo_id:
    raise AlgoTagError(
      f'NNF algo flag {flag} marks an algo order but Algo ID is the '
      f'non-algo sentinel {non_algo_id!r}; the audit trail cannot '
      'attribute this order')
  if not claims_algo and claims_algo_id:
    raise AlgoTagError(
      f'NNF algo flag {flag} marks a non-algo order but Algo ID is '
      f'{algo_id!r}; that records an algo run that did not happen')
  return nnf_id


def _require_nnf(nnf_id: str) -> None:
  '''Raise unless ``nnf_id`` is exactly 15 decimal digits.

  Args:
    nnf_id: The value to check.

  Raises:
    AlgoTagError: If the value is not a string of 15 decimal digits.
  '''
  if not isinstance(nnf_id, str):
    raise AlgoTagError(
      f'NNF ID must be a string, got {type(nnf_id).__name__}')
  if len(nnf_id) != nnf_length:
    raise AlgoTagError(
      f'NNF ID must be {nnf_length} digits, got {len(nnf_id)}: {nnf_id!r}')
  if not nnf_id.isdecimal():
    raise AlgoTagError(f'NNF ID must be all digits, got {nnf_id!r}')
