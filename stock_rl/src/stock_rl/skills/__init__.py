#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Agents as markdown, gated by evidence.

This package exists because of one sentence in
``docs/research/agents/llm-agents-in-finance.md``: the only defensible
architecture from the evidence is independent feature extractors averaged
with equal weights, and the decomposition itself is not the point. A skill
file is exactly that -- one independent extractor, written down, with its
evidence and its kill criteria in the file rather than in somebody's
memory. Ten of them averaged is the design; ten of them debating is the
thing the research measured to lose to equal-weight two thirds of the
time.

Three modules:

``schema``
  The file format, the evidence tiers and the validation. Fixed key set,
  closed, and every failure raises rather than being skipped.

``loader``
  Discovery and a small hand-rolled frontmatter parser. No YAML library,
  because the project carries zero runtime dependencies and the supported
  grammar is a strict subset of a fixed key set.

``registry``
  The gate. A ``rejected`` skill never runs, an ``experimental`` skill runs
  only in experiment mode, and a ``supported`` skill needs both an
  explicit enable flag and a **passed kill criterion**.

Adding an agent is adding a ``.md`` file to ``skills/skills/``. No Python,
no redeploy. It will be picked up on the next
:meth:`SkillRegistry.load`, validated loudly if it is wrong, and refused
if its evidence tier does not earn it.

The tiers are the content of this package as much as the code is. Two of
the six shipped skills are ``rejected`` for structural reasons that no
experiment can fix, one is ``experimental`` because the mechanism is
plausible and the evidence is not there, and three are ``supported`` --
and one of those three is supported only as a **sizing** input, never as a
direction.
'''

from __future__ import annotations

from stock_rl.skills.loader import (
  discover_skills,
  load_skill,
  load_skills,
  parse_frontmatter,
  read_document,
  skills_directory,
  split_document,
)
from stock_rl.skills.registry import (
  Gate,
  KillResult,
  RegistrationRecord,
  SkillManifest,
  SkillRegistry,
  SkillStatus,
)
from stock_rl.skills.schema import (
  EvidenceTier,
  KillCriterion,
  Skill,
  SkillDigest,
  SkillError,
  SkillFormatError,
  document_sections,
  rejected_sections,
  required_keys,
  required_sections,
  tiers,
)

__all__ = [
  'EvidenceTier',
  'Gate',
  'KillCriterion',
  'KillResult',
  'RegistrationRecord',
  'Skill',
  'SkillDigest',
  'SkillError',
  'SkillFormatError',
  'SkillManifest',
  'SkillRegistry',
  'SkillStatus',
  'discover_skills',
  'document_sections',
  'load_skill',
  'load_skills',
  'parse_frontmatter',
  'read_document',
  'rejected_sections',
  'required_keys',
  'required_sections',
  'skills_directory',
  'split_document',
  'tiers',
]
