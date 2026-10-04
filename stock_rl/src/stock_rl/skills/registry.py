#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Load, validate and gate the markdown skill files.

Three objects, in the order they matter.

:class:`Gate`
  The results of the pre-registered kill criteria. Immutable: recording a
  result returns a new gate, because a gate that can be quietly mutated
  is not a pre-registration, it is a suggestion.

:class:`SkillRegistry`
  The loaded skills, and the only thing allowed to decide whether one of
  them runs. The rule it enforces is short enough to state in one line
  and is the reason this package exists:

      A skill runs only if its own file declares ``enabled: true``, the
      operator lists it in the enable set, **and** its evidence tier's
      gate allows it.

:class:`SkillManifest`
  The registered identity of every loaded skill, and one fingerprint over
  the lot. That fingerprint is the NSE re-registration hook: operational
  modalities 9.1 and 9.9 require a fresh exchange registration for a
  change to the decision logic of a black-box algo, and a skill file is
  decision logic. Change a criterion, downgrade a tier, edit a sentence
  of reasoning, and the manifest fingerprint moves.

The tier gate, which is the whole design:

  ``rejected``
    Never runs. No flag combination reaches it, and the status line says
    so in words rather than returning a silent false. Two shipped skills
    are here, and both are here for structural reasons rather than
    empirical ones -- see the body of each file.

  ``experimental``
    Runs in experiment mode only, and is marked ``experiment_only`` in
    every status it returns so a caller cannot mistake an experiment for
    a live input. One shipped skill sits here: plausible mechanism, no
    evidence it adds alpha on Nifty-50 names.

  ``supported``
    Runs live only on two independent gates: an explicit enable flag and
    a **passed kill criterion**. This is the mechanism that stops a team
    building ten agents that each quietly lose to 1/N. A supported skill
    with no result for one of its criteria is refused, and the reason
    names the criteria that are missing or failed.

Nothing is cached. ``status`` recomputes on every call, so a criterion
that passed when a skill was switched on and failed last week leaves the
skill inactive on the next call with no restart and no cache to clear.
That is the difference between a kill criterion and a comment.
'''

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from stock_rl.rl.policy import policy_fingerprint
from stock_rl.skills.loader import load_skills
from stock_rl.skills.schema import (
  EvidenceTier,
  KillCriterion,
  Skill,
  SkillError,
)

__all__ = [
  'Gate',
  'KillResult',
  'RegistrationRecord',
  'SkillError',
  'SkillManifest',
  'SkillRegistry',
  'SkillStatus',
]


@dataclass(frozen=True, slots=True)
class KillResult:
  '''The outcome of one pre-registered kill criterion.

  ``note`` carries the evidence -- the p-value, the fold count, the date
  of the run -- because a bare boolean is not auditable and this project
  keeps an audit trail for eight years whether it likes it or not.

  Attributes:
    criterion: The :attr:`KillCriterion.id` this result answers.
    passed: Whether the criterion cleared its threshold.
    note: Evidence backing the verdict, recorded by the experiment.
  '''

  criterion: str
  passed: bool
  note: str = ''


class Gate:
  '''Immutable set of kill-criterion results.

  Built from the experiment log and handed to the registry. There is no
  mutating ``set``-style method: :meth:`record` returns a **new** gate, so
  a caller holding an old reference cannot have it edited underneath
  them, and an activation decision can always be reproduced from the gate
  value that produced it.

  Results may arrive in any order and may include criteria belonging to
  skills this gate is not asked about; a gate is not scoped to one skill
  because one experiment usually answers criteria for several at once.
  '''

  def __init__(self, results: Iterable[KillResult] = ()) -> None:
    '''Build a gate from a sequence of results.

    Args:
      results: Kill-criterion results. Later entries for the same
        criterion are refused rather than silently winning, because a
        duplicated criterion id means two runs disagree about what was
        measured.

    Raises:
      TypeError: If an item is not a :class:`KillResult`.
      ValueError: If a criterion id is blank or repeated.
    '''
    collected: dict[str, KillResult] = {}
    for item in results:
      if not isinstance(item, KillResult):
        raise TypeError(
          f'expected a KillResult, got {type(item).__name__}')
      if not item.criterion.strip():
        raise ValueError('a kill result needs a criterion id')
      if item.criterion in collected:
        raise ValueError(
          f'duplicate kill result for criterion {item.criterion!r}')
      collected[item.criterion] = item
    self._results = collected

  def __len__(self) -> int:
    '''Return the number of recorded results.'''
    return len(self._results)

  def __iter__(self) -> Iterator[KillResult]:
    '''Iterate over the recorded results, in insertion order.'''
    return iter(self._results.values())

  def __eq__(self, other: object) -> bool:
    '''Return whether another gate holds the same results.

    Args:
      other: Candidate gate.

    Returns:
      True when both gates hold identical results.
    '''
    if not isinstance(other, Gate):
      return NotImplemented
    return self._results == other._results

  def __repr__(self) -> str:
    '''Return a short debugging representation.'''
    passed = sum(1 for item in self._results.values() if item.passed)
    return f'Gate({passed}/{len(self._results)} passed)'

  def record(self, criterion: str, passed: bool,
             note: str = '') -> Gate:
    '''Return a new gate with one result added.

    Args:
      criterion: Criterion id being answered.
      passed: Whether it cleared its threshold.
      note: Evidence backing the verdict.

    Returns:
      A new gate holding every previous result plus this one.

    Raises:
      ValueError: If the criterion id is blank.
    '''
    return Gate((*self._results.values(),
                 KillResult(criterion, passed, note)))

  def result(self, criterion: str) -> KillResult | None:
    '''Return the recorded result for a criterion, if any.

    Args:
      criterion: Criterion id to look up.

    Returns:
      The result, or None when the criterion has not been run.
    '''
    return self._results.get(criterion)

  def outcome(self, criterion: str) -> bool | None:
    '''Return whether a criterion passed, with None for 'not run'.

    Args:
      criterion: Criterion id to look up.

    Returns:
      True if it passed, False if it failed, None if there is no result.
    '''
    found = self._results.get(criterion)
    return None if found is None else found.passed

  def missing(self, criteria: Iterable[KillCriterion]) -> tuple[str, ...]:
    '''Return the criteria of a skill that have no result at all.

    Args:
      criteria: Criteria declared by the skill being gated.

    Returns:
      Their ids, in declaration order.
    '''
    return tuple(item.id for item in criteria
                 if item.id not in self._results)

  def failed(self, criteria: Iterable[KillCriterion]) -> tuple[str, ...]:
    '''Return the criteria of a skill that were run and did not pass.

    Args:
      criteria: Criteria declared by the skill being gated.

    Returns:
      Their ids, in declaration order.
    '''
    return tuple(item.id for item in criteria
                 if self._results.get(item.id) is not None
                 and not self._results[item.id].passed)

  def all_passed(self, criteria: Iterable[KillCriterion]) -> bool:
    '''Return whether every one of these criteria has passed.

    A criterion with no result has **not** passed. Absence of evidence is
    not evidence, and a supported skill must clear every bar it set for
    itself rather than the subset somebody happened to run.

    Args:
      criteria: Criteria declared by the skill being gated.

    Returns:
      True when there is at least one criterion and all of them passed.
    '''
    declared = tuple(criteria)
    if not declared:
      return False
    return all(self.outcome(item.id) is True for item in declared)

  def as_dict(self) -> dict[str, bool]:
    '''Return the results as a plain id-to-verdict mapping.

    Returns:
      Copy of the recorded verdicts, in insertion order.
    '''
    return {key: item.passed for key, item in self._results.items()}


@dataclass(frozen=True, slots=True)
class SkillStatus:
  '''Why one skill did or did not run.

  Returned rather than a bare boolean because the reason is the useful
  part. A caller that gets ``False`` for an experimental skill in live
  mode and a caller that gets ``False`` for a supported skill whose kill
  criteria failed have different things to do about it, and only one of
  them can go and fix something.

  Attributes:
    name: Skill name.
    version: Version of the loaded file.
    evidence_tier: Declared tier.
    active: Whether the skill runs under the flags it was asked about.
    experiment_only: Whether the skill is confined to experiment mode.
    reason: Prose explanation of ``active``.
    fingerprint: Fingerprint of the skill file, for the audit trail.
  '''

  name: str
  version: int
  evidence_tier: EvidenceTier
  active: bool
  experiment_only: bool
  reason: str
  fingerprint: str

  @property
  def allowed_live(self) -> bool:
    '''Return whether the skill may influence a live order path.

    True only for an active skill that is not experiment-only. An
    experimental skill is genuinely running in experiment mode and must
    still answer False here, which is what stops an experiment result
    being wired into production by a caller that only checked
    ``active``.
    '''
    return self.active and not self.experiment_only


@dataclass(frozen=True, slots=True)
class RegistrationRecord:
  '''One skill's identity as registered with the exchange.

  Attributes:
    name: Skill name.
    version: Version of the loaded file.
    evidence_tier: Declared tier, because a downgrade is a
      decision-logic change and must move the registration fingerprint.
    fingerprint: Fingerprint of the file's content.
  '''

  name: str
  version: int
  evidence_tier: EvidenceTier
  fingerprint: str


@dataclass(frozen=True, slots=True)
class SkillManifest:
  '''The registered identity of every loaded skill, as one hashable thing.

  A frozen dataclass so it can be handed straight to
  :func:`stock_rl.rl.policy.policy_fingerprint`, which is what makes the
  skill set and a learned policy share one registration record and one
  hashing convention.

  Attributes:
    skills: Per-skill records, sorted by name so the manifest does not
      depend on directory iteration order.
  '''

  skills: tuple[RegistrationRecord, ...]

  def fingerprint(self) -> str:
    '''Return one fingerprint covering every registered skill.

    Returns:
      A stable fingerprint string, tagged with the policy fingerprint
      schema because it is the same convention.
    '''
    return policy_fingerprint(self)

  def names(self) -> tuple[str, ...]:
    '''Return the registered skill names, in manifest order.'''
    return tuple(record.name for record in self.skills)


class SkillRegistry:
  '''Every loaded skill, and the only authority on whether one runs.

  Immutable after construction. Names are unique and checked on the way
  in: two files declaring the same name would make the enable set
  ambiguous, and an ambiguous enable set is not a control.
  '''

  def __init__(self, skills: Iterable[Skill] = ()) -> None:
    '''Build a registry from validated skills.

    Args:
      skills: Skills to register, typically from
        :func:`stock_rl.skills.loader.load_skills`.

    Raises:
      TypeError: If an item is not a :class:`~stock_rl.skills.schema.Skill`.
      ValueError: If two skills declare the same name.
    '''
    collected: dict[str, Skill] = {}
    for skill in skills:
      if not isinstance(skill, Skill):
        raise TypeError(
          f'expected a Skill, got {type(skill).__name__}')
      if skill.name in collected:
        raise ValueError(f'duplicate skill name: {skill.name!r}')
      collected[skill.name] = skill
    self._skills = collected

  def __len__(self) -> int:
    '''Return the number of registered skills.'''
    return len(self._skills)

  def __iter__(self) -> Iterator[Skill]:
    '''Iterate over the skills, in name order.'''
    return iter(self._values())

  def __contains__(self, name: object) -> bool:
    '''Return whether a skill name is registered.

    Args:
      name: Candidate skill name.

    Returns:
      True when the name is registered.
    '''
    return str(name) in self._skills

  def __repr__(self) -> str:
    '''Return a short debugging representation.'''
    return f'SkillRegistry({len(self._skills)} skills)'

  def _values(self) -> tuple[Skill, ...]:
    '''Return every skill, in name order.

    Returns:
      The registered skills sorted by name.
    '''
    return tuple(self._skills[name] for name in sorted(self._skills))

  @classmethod
  def load(cls, directory: Path | str | None = None) -> SkillRegistry:
    '''Load every skill file in a directory.

    Args:
      directory: Directory to scan, or None for the packaged directory.

    Returns:
      A registry over every validated skill in the directory.

    Raises:
      SkillError: If any file fails to parse or validate. Nothing is
        skipped silently.
    '''
    return cls(load_skills(directory))

  @property
  def names(self) -> tuple[str, ...]:
    '''Return every registered skill name, in name order.'''
    return tuple(sorted(self._skills))

  def get(self, name: str) -> Skill:
    '''Return one registered skill.

    Args:
      name: Skill name.

    Returns:
      The skill.

    Raises:
      KeyError: If no skill carries that name.
    '''
    try:
      return self._skills[name]
    except KeyError:
      raise KeyError(
        f'unknown skill {name!r}; registered: '
        f'{', '.join(self.names) or '(none)'}') from None

  def manifest(self) -> SkillManifest:
    '''Return the registered identity of every skill.

    Returns:
      A manifest sorted by name.
    '''
    return SkillManifest(tuple(
      RegistrationRecord(skill.name, skill.version, skill.evidence_tier,
                         skill.fingerprint)
      for skill in self._values()))

  def registration_fingerprint(self) -> str:
    '''Return one fingerprint covering the whole registered skill set.

    This is the string a registration record should hold. Editing one
    character of one skill body changes it, which is exactly the change
    NSE operational modalities 9.1 and 9.9 want told about.

    Returns:
      A stable fingerprint string.
    '''
    return self.manifest().fingerprint()

  def status(self, name: str, gate: Gate | None = None,
             enabled: Iterable[str] = (),
             experiment_mode: bool = False) -> SkillStatus:
    '''Decide whether one skill runs, and say why.

    Nothing is cached, so a criterion that has since failed switches the
    skill off on the next call.

    Args:
      name: Skill to evaluate.
      gate: Kill-criterion results. None is an empty gate, which refuses
        every ``supported`` skill -- an absent result is not a pass.
      enabled: Names the operator has switched on. Independent of the
        file's own ``enabled`` key: both must agree.
      experiment_mode: Whether the caller is running experiments rather
        than trading.

    Returns:
      The full status, including the reason.

    Raises:
      KeyError: If no skill carries that name.
    '''
    skill = self.get(name)
    active, experiment_only, reason = self._decide(
      skill, gate or Gate(), frozenset(str(item) for item in enabled),
      experiment_mode)
    return SkillStatus(
      name=skill.name,
      version=skill.version,
      evidence_tier=skill.evidence_tier,
      active=active,
      experiment_only=experiment_only,
      reason=reason,
      fingerprint=skill.fingerprint,
    )

  def statuses(self, gate: Gate | None = None,
               enabled: Iterable[str] = (),
               experiment_mode: bool = False) -> tuple[SkillStatus, ...]:
    '''Decide whether every registered skill runs.

    Args:
      gate: Kill-criterion results shared across skills.
      enabled: Names the operator has switched on.
      experiment_mode: Whether the caller is running experiments rather
        than trading.

    Returns:
      One status per skill, in name order.
    '''
    permitted = frozenset(str(item) for item in enabled)
    return tuple(
      self.status(skill.name, gate, permitted, experiment_mode)
      for skill in self._values())

  def active(self, gate: Gate | None = None, enabled: Iterable[str] = (),
             experiment_mode: bool = False) -> tuple[Skill, ...]:
    '''Return the skills that run under the given flags.

    Args:
      gate: Kill-criterion results shared across skills.
      enabled: Names the operator has switched on.
      experiment_mode: Whether the caller is running experiments rather
        than trading.

    Returns:
      The active skills, in name order.
    '''
    return tuple(
      skill for skill, status in zip(
        self._values(),
        self.statuses(gate, enabled, experiment_mode))
      if status.active)

  def _decide(self, skill: Skill, gate: Gate,
              enabled: frozenset[str],
              experiment_mode: bool) -> tuple[bool, bool, str]:
    '''Apply the tier gate and the two enable flags.

    The order is deliberate. The tier is checked first so that a
    ``rejected`` skill reports the evidence verdict as its reason, which
    is the only fact about it anybody needs.

    Args:
      skill: The skill to evaluate.
      gate: Kill-criterion results.
      enabled: Names the operator has switched on.
      experiment_mode: Whether the caller is running experiments.

    Returns:
      Tuple of (active, experiment_only, reason).
    '''
    if skill.evidence_tier == 'rejected':
      return (False, False,
              'evidence tier is rejected: this skill must never run, and '
              'no flag combination reaches it')
    if not skill.enabled:
      return (False, False,
              'the skill file declares enabled: false')
    if skill.name not in enabled:
      return (False, False, 'not in the operator enable set')
    if skill.evidence_tier == 'experimental':
      if not experiment_mode:
        return (False, False,
                'experimental skills run in experiment mode only')
      return (True, True,
              'experimental: allowed in experiment mode, never live')
    unmeasured = gate.missing(skill.kill_criteria)
    failed = gate.failed(skill.kill_criteria)
    if unmeasured or failed:
      parts = []
      if unmeasured:
        parts.append(f'no result for {', '.join(unmeasured)}')
      if failed:
        parts.append(f'failed {', '.join(failed)}')
      return (False, False,
              f'supported skill refused: kill criteria not passed '
              f'({' and '.join(parts)})')
    return (True, False,
            'supported: kill criteria passed and the skill is enabled')
