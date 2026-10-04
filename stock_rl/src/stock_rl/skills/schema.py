#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''What a skill file is, and what it must declare before it may run.

An *agent* in this project is a markdown file. Adding one means dropping a
``.md`` file into ``skills/skills/``: no Python to write, no wiring to
redeploy. That only works if the file itself carries every fact the
runtime needs, which is what this module specifies.

The shape is deliberately narrow. A file opens with a ``---`` fence, then
frontmatter carrying a **fixed key set**, then a markdown body::

    ---
    name: momentum
    version: 1
    enabled: true
    evidence_tier: supported
    data_sources:
      - nse_bhavcopy_eod
    kill_criteria:
      - id: beat_equal_weight
        statement: Diebold-Mariano on realised equity curves
        threshold: p <= 0.10 and |t| > 3.0 in at least 4 of 5 folds
    ---
    ## Purpose
    ...

``evidence_tier`` is the point of the exercise, and the tiers are ordered
weakest evidence first: ``rejected``, ``experimental``, ``supported``.
Most shipped skills sit at the bottom two, because that is where the
research puts them -- see
``docs/research/agents/llm-agents-in-finance.md``. A tier is a claim about
the *evidence*, not about the code. Only ``supported`` may ever reach a
live order path, and the registry in :mod:`stock_rl.skills.registry` gates
even that on the kill criteria declared here.

Validation is loud by design. A malformed skill file raises
:class:`SkillFormatError` at load instead of being skipped, because a
silently ignored skill file is a bug you discover in production, when
somebody asks why their agent stopped contributing.

``version`` is an integer, not a date. The fingerprint below is a
compliance control rather than an explainability toy: NSE operational
modalities 9.1 and 9.9 require a fresh exchange re-registration for any
change to the decision logic of a black-box algo, so "is the running
process the registered one?" has to be answerable. It is answerable here
because the fingerprint covers the name, the version and the entire body:
change one word of the reasoning and the registered fingerprint no longer
matches, which is exactly the event the exchange wants told.
'''

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from stock_rl.rl.policy import policy_fingerprint

__all__ = [
  'EvidenceTier',
  'KillCriterion',
  'Skill',
  'SkillDigest',
  'SkillError',
  'SkillFormatError',
  'document_sections',
  'rejected_sections',
  'required_keys',
  'required_sections',
  'tiers',
]

#: The three evidence tiers, ordered weakest evidence first. The order is
#: the argument: a tier is only as strong as the research behind it, and
#: two of the six shipped skills are ``rejected`` outright.
EvidenceTier = Literal['rejected', 'experimental', 'supported']

tiers: tuple[EvidenceTier, ...] = ('rejected', 'experimental',
                                   'supported')

#: Frontmatter keys every skill file must declare. The set is closed: an
#: unknown key is a typo or an unagreed extension, and both are errors.
required_keys = ('data_sources', 'enabled', 'evidence_tier', 'kill_criteria',
                 'name', 'version')

#: Level-2 headings every skill body must contain. Enforced rather than
#: requested, because a skill that does not separate its claim from the
#: evidence for the claim has not made an argument at all.
required_sections = ('data sources', 'kill criteria', 'purpose',
                     'what it claims', 'what the evidence actually says')

#: Extra headings a ``rejected`` skill must carry. 'Document WHY' is only
#: meaningful if the document is checked, so it is a schema requirement.
rejected_sections = ('why this is rejected',)

#: Skill names are lowercase with underscores, like the module constants
#: in this package, and because the name becomes an audit-log token.
name_pattern = re.compile(r'^[a-z][a-z0-9_]*$')

heading_pattern = re.compile(r'^##\s+(?P<title>.*?)\s*$')


class SkillError(Exception):
  '''Base class for every skill-file failure.

  Separate from ``ValueError`` so a caller can distinguish "this skill
  file is wrong" from "this argument is wrong", which matters because
  the first is an operator-facing deployment failure and the second is a
  caller bug.
  '''


class SkillFormatError(SkillError):
  '''Raised when a skill file cannot be parsed or fails the schema.

  Carries the file name in its message. A malformed file is rejected
  loudly at load rather than skipped: the alternative is a skill that
  vanishes from the system with no trace, and nobody notices until a
  decision nobody can explain changes.
  '''


@dataclass(frozen=True, slots=True)
class SkillDigest:
  '''The three fields whose content defines a skill's identity.

  Deliberately narrower than :class:`Skill`. The file's path is excluded
  so relocating a skill does not look like editing it, and the frontmatter
  is included rather than excluded so that flipping ``enabled`` or
  downgrading an ``evidence_tier`` **is** a detectable change -- both are
  decision-logic changes as far as the exchange is concerned.

  Frozen and slot-bearing because this object is fed straight to
  :func:`stock_rl.rl.policy.policy_fingerprint`, which serialises it to
  canonical JSON and digests the result.

  Attributes:
    name: Skill name, lowercase with underscores.
    version: Monotonic integer version of the file.
    body: Canonical markdown body, i.e. the frontmatter-free document.
    enabled: Whether the file declares itself enabled.
    evidence_tier: The declared evidence tier.
    data_sources: Declared data sources, in file order.
  '''

  name: str
  version: int
  body: str
  enabled: bool = True
  evidence_tier: EvidenceTier = 'rejected'
  data_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KillCriterion:
  '''One pre-registered kill criterion.

  Pre-registered means declared **before** the experiment runs, in the
  file, in version control. That is the only property that makes a kill
  criterion worth anything: a threshold chosen after seeing the result is
  a description, not a test.

  Attributes:
    id: Stable identifier used to look the result up in a
      :class:`stock_rl.skills.registry.Gate`.
    statement: What is being measured, in plain words.
    threshold: The numeric or boolean bar the result must clear.
  '''

  id: str
  statement: str
  threshold: str = ''


def document_sections(body: str) -> tuple[str, ...]:
  '''Return the normalised level-2 headings of a markdown body.

  Only ``##`` headings count, because ``#`` is the document title and
  ``###`` is a detail under a section this schema does not require.
  Titles are case-folded and stripped of a trailing colon so that
  ``## Purpose:`` and ``## purpose`` satisfy the same requirement.

  Args:
    body: Markdown body below the frontmatter fence.

  Returns:
    Heading titles, normalised, in document order.
  '''
  found: list[str] = []
  for line in body.splitlines():
    match = heading_pattern.match(line.rstrip())
    if match is not None:
      title = match.group('title').strip().rstrip(':.').strip()
      found.append(title.casefold())
  return tuple(found)


def _label(source: str) -> str:
  '''Return a readable prefix for schema error messages.

  Args:
    source: File name of the skill, or an empty string.

  Returns:
    The name followed by a colon, or a placeholder when the name is
    unknown.
  '''
  return f'{source}: ' if source else ''


def _text_field(fields: Mapping[str, object], key: str,
                label: str) -> str:
  '''Return a required non-empty string field.

  Args:
    fields: Parsed frontmatter mapping.
    key: Field name.
    label: Human-readable prefix for error messages.

  Returns:
    The stripped field value.

  Raises:
    SkillFormatError: If the field is missing, not a string, or blank.
  '''
  if key not in fields:
    raise SkillFormatError(f'{label}missing required key: {key}')
  value = fields[key]
  if not isinstance(value, str) or not value.strip():
    raise SkillFormatError(
      f'{label}{key} must be a non-empty string, got {value!r}')
  return value.strip()


def _flag_field(fields: Mapping[str, object], key: str,
                label: str) -> bool:
  '''Return a required boolean field.

  Truthiness is not accepted. ``enabled: yes`` in a file a human wrote
  is not the same declaration as ``enabled: true``, and coercing one into
  the other is how a typo becomes an enabled live agent.

  Args:
    fields: Parsed frontmatter mapping.
    key: Field name.
    label: Human-readable prefix for error messages.

  Returns:
    The declared boolean.

  Raises:
    SkillFormatError: If the field is missing or not a real boolean.
  '''
  value = fields.get(key)
  if not isinstance(value, bool):
    raise SkillFormatError(
      f'{label}{key} must be true or false, got {value!r}')
  return value


def _version_field(fields: Mapping[str, object], key: str,
                   label: str) -> int:
  '''Return a required positive integer version field.

  Args:
    fields: Parsed frontmatter mapping.
    key: Field name.
    label: Human-readable prefix for error messages.

  Returns:
    The declared version, at least 1.

  Raises:
    SkillFormatError: If the field is missing, not an integer, a
      boolean, or below 1.
  '''
  value = fields.get(key)
  if isinstance(value, bool) or not isinstance(value, int):
    raise SkillFormatError(
      f'{label}{key} must be a whole number, got {value!r}')
  if value < 1:
    raise SkillFormatError(
      f'{label}{key} must be 1 or greater, got {value}')
  return value


def _tier_field(fields: Mapping[str, object], key: str,
                label: str) -> EvidenceTier:
  '''Return a required evidence tier field.

  Args:
    fields: Parsed frontmatter mapping.
    key: Field name.
    label: Human-readable prefix for error messages.

  Returns:
    The declared tier.

  Raises:
    SkillFormatError: If the field is missing or not a known tier. The
      message lists the valid tiers because an agent that guessed a tier
      should be corrected, not rejected with no guidance.
  '''
  value = fields.get(key)
  if value not in tiers:
    raise SkillFormatError(
      f'{label}{key} must be one of {', '.join(tiers)}, got {value!r}')
  return value  # type: ignore[return-value]


def _source_list(fields: Mapping[str, object], key: str,
                 label: str) -> tuple[str, ...]:
  '''Return a required list of data-source names.

  Args:
    fields: Parsed frontmatter mapping.
    key: Field name.
    label: Human-readable prefix for error messages.

  Returns:
    The declared source names, stripped, in file order.

  Raises:
    SkillFormatError: If the field is not a list, is empty, or holds
      anything other than non-empty strings. A list is mandatory because
      'where does this number come from' is the first question of every
      later audit.
  '''
  value = fields.get(key)
  if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
    raise SkillFormatError(f'{label}{key} must be a list, got {value!r}')
  entries: list[str] = []
  for item in value:
    if not isinstance(item, str) or not item.strip():
      raise SkillFormatError(
        f'{label}{key} entries must be non-empty strings, got {item!r}')
    entries.append(item.strip())
  if not entries:
    raise SkillFormatError(f'{label}{key} must list at least one source')
  return tuple(entries)


def _criteria_list(fields: Mapping[str, object], key: str,
                   label: str) -> tuple[KillCriterion, ...]:
  '''Return the required, pre-registered kill criteria.

  Args:
    fields: Parsed frontmatter mapping.
    key: Field name.
    label: Human-readable prefix for error messages.

  Returns:
    The criteria in file order.

  Raises:
    SkillFormatError: If the field is not a list, is empty, holds a
      non-mapping, omits ``id`` or ``statement``, repeats an id, or
      declares a blank field. Empty is refused because a skill with no
      kill criteria has no way to ever be stopped, which for a
      ``supported`` skill is precisely the failure this package exists to
      prevent.
  '''
  value = fields.get(key)
  if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
    raise SkillFormatError(f'{label}{key} must be a list, got {value!r}')
  criteria: list[KillCriterion] = []
  seen: set[str] = set()
  for item in value:
    if not isinstance(item, Mapping):
      raise SkillFormatError(
        f'{label}each {key} entry must declare an id and a statement, '
        f'got {item!r}')
    if 'id' not in item or 'statement' not in item:
      raise SkillFormatError(
        f'{label}each {key} entry needs both id and statement, '
        f'got keys {sorted(str(k) for k in item)}')
    criterion_id = _text_field(item, 'id', f'{label}{key} entry: ')
    statement = _text_field(item, 'statement', f'{label}{key} entry: ')
    if criterion_id in seen:
      raise SkillFormatError(
        f'{label}duplicate kill criterion id: {criterion_id!r}')
    seen.add(criterion_id)
    threshold = item.get('threshold', '')
    if not isinstance(threshold, str):
      raise SkillFormatError(
        f'{label}{key} entry {criterion_id!r} threshold must be a '
        f'string, got {threshold!r}')
    criteria.append(
      KillCriterion(criterion_id, statement, threshold.strip()))
  if not criteria:
    raise SkillFormatError(
      f'{label}{key} must pre-register at least one criterion; a skill '
      'with no kill criterion can never be stopped')
  return tuple(criteria)


@dataclass(frozen=True, slots=True)
class Skill:
  '''One loaded, schema-valid skill file.

  Frozen because a skill is a declaration, not a configuration knob: if a
  running system can mutate a skill's tier or criteria at runtime then
  the fingerprint in :mod:`stock_rl.rl.policy` no longer describes the
  decision logic the exchange was told about, and the whole re-
  registration story collapses.

  Attributes:
    name: Skill name, lowercase with underscores.
    version: Monotonic integer version of the file.
    enabled: The file's own switch. ``false`` means this skill must never
      run whatever any other flag says.
    evidence_tier: Declared evidence tier.
    data_sources: Declared data-source names, in file order.
    kill_criteria: Pre-registered kill criteria, in file order.
    body: Canonical markdown body.
    source: File name the skill was loaded from, for error messages and
      the audit trail.
  '''

  name: str
  version: int
  enabled: bool
  evidence_tier: EvidenceTier
  data_sources: tuple[str, ...]
  kill_criteria: tuple[KillCriterion, ...]
  body: str
  source: str = ''

  @property
  def digest(self) -> SkillDigest:
    '''Return the identity payload the fingerprint is computed over.

    Exposed so a caller can inspect exactly what was hashed rather than
    trusting that nothing important was left out.
    '''
    return SkillDigest(
      self.name, self.version, self.body, self.enabled, self.evidence_tier,
      self.data_sources)

  @property
  def fingerprint(self) -> str:
    '''Return the deployment fingerprint of this skill file.

    Delegates to :func:`stock_rl.rl.policy.policy_fingerprint`, so the
    scheme tag, the canonicalisation and the JSON payload are exactly the
    ones the exchange registration record already uses for a learned
    policy. One hashing convention for both means a registration covering
    policies and skills is one string to compare.
    '''
    return policy_fingerprint(self.digest)

  @property
  def criteria_ids(self) -> tuple[str, ...]:
    '''Return the ids of the pre-registered kill criteria.'''
    return tuple(criterion.id for criterion in self.kill_criteria)

  def has_section(self, section: str) -> bool:
    '''Return whether the body carries a given level-2 heading.

    Args:
      section: Heading title, matched case-insensitively after
        normalising a trailing colon or full stop.

    Returns:
      True when the body contains the heading.
    '''
    wanted = section.strip().rstrip(':.').strip().casefold()
    return wanted in document_sections(self.body)

  @classmethod
  def from_mapping(cls, frontmatter: Mapping[str, object], body: str,
                   source: str = '') -> Skill:
    '''Validate parsed frontmatter and build a Skill.

    This is the single place the schema is enforced. It refuses rather
    than repairs: there is no default tier, no default version and no
    optional criterion list, because a default supplied here would be a
    default the file author never agreed to.

    Args:
      frontmatter: Frontmatter mapping as produced by
        :func:`stock_rl.skills.loader.parse_frontmatter`.
      body: Markdown body below the frontmatter fence.
      source: File name, used only in error messages.

    Returns:
      The validated, frozen skill.

    Raises:
      SkillFormatError: If ``frontmatter`` is not a mapping, omits a
        required key, carries an unknown key, or gives a field the wrong
        shape; or if the body omits a required heading, or omits
        ``Why this is rejected`` on a ``rejected`` skill.
    '''
    label = _label(source)
    if not isinstance(frontmatter, Mapping):
      raise SkillFormatError(
        f'{label}frontmatter must be a key and value block, '
        f'got {type(frontmatter).__name__}')
    missing = [key for key in required_keys if key not in frontmatter]
    if missing:
      raise SkillFormatError(
        f'{label}missing required key(s): {', '.join(missing)}; '
        f'required keys are {', '.join(required_keys)}')
    unknown = sorted(str(key) for key in set(frontmatter)
                     - set(required_keys))
    if unknown:
      raise SkillFormatError(
        f'{label}unknown key(s): {', '.join(unknown)}; the key set is '
        f'closed at {', '.join(required_keys)}')

    name = _text_field(frontmatter, 'name', label)
    if name_pattern.match(name) is None:
      raise SkillFormatError(
        f'{label}name must be lowercase_with_underscores, got {name!r}')
    tier = _tier_field(frontmatter, 'evidence_tier', label)
    skill = cls(
      name=name,
      version=_version_field(frontmatter, 'version', label),
      enabled=_flag_field(frontmatter, 'enabled', label),
      evidence_tier=tier,
      data_sources=_source_list(frontmatter, 'data_sources', label),
      kill_criteria=_criteria_list(frontmatter, 'kill_criteria', label),
      body=body,
      source=source,
    )
    skill._check_body(label)
    return skill

  def _check_body(self, label: str) -> None:
    '''Verify the body carries every heading the tier requires.

    Args:
      label: Human-readable prefix for error messages.

    Raises:
      SkillFormatError: If a required heading is absent, listing all of
        them at once so a broken file takes one edit to fix rather than
        one edit per heading.
    '''
    needed = list(required_sections)
    if self.evidence_tier == 'rejected':
      needed.extend(rejected_sections)
    present = document_sections(self.body)
    absent = [section for section in needed if section not in present]
    if absent:
      raise SkillFormatError(
        f'{label}body is missing heading(s): '
        f'{' | '.join('## ' + item.capitalize() for item in absent)}')
