#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Agents as markdown files, gated by an evidence tier.

There is no ``test_skills.py`` anywhere else in this repository, which
means the package whose whole claim is that "adding an agent is adding a
``.md`` file" is the one nobody has pinned down. This file is the
regression suite for that claim, and it is written against the shipped
files rather than against fixtures, because the failure mode being guarded
is exactly the one a fixture cannot see: a markdown file that loads, is
discovered, and quietly asserts something nobody measured.

Three properties are load-bearing:

**Loudness.** A malformed skill file raises at load. It is never skipped,
and :func:`load_skills` collects every bad file and raises once, so a
typo in one skill cannot take the others down and cannot vanish
silently. A skill that silently stops loading is a decision that changes
with no audit trail.

**The tier gate.** A ``rejected`` skill is unreachable by any flag
combination, an ``experimental`` skill runs only in experiment mode and is
labelled ``experiment_only`` wherever it appears, and a ``supported``
skill stays off until every criterion it pre-registered has passed. The
gate is recomputed on every call, so a criterion that passed when the
skill was switched on and failed last week switches it off with no
restart and no cache to clear.

**No unearned performance.** A skill file may not assert a Sharpe or a
return figure it cannot support. Figures are allowed only where they are
attributed to somebody else's sample or explicitly marked as not this
project's claim, and :func:`unsupported_claims` is the scanner that
decides. It is tested against a deliberately bad body as well as against
the shipped ones, so a scanner that has quietly stopped scanning cannot
pass this file.
'''

from __future__ import annotations

import itertools
import pathlib
import re
from collections.abc import Iterable

import pytest

from stock_rl.skills import (
  EvidenceTier,
  Gate,
  KillCriterion,
  KillResult,
  RegistrationRecord,
  Skill,
  SkillDigest,
  SkillError,
  SkillFormatError,
  SkillManifest,
  SkillRegistry,
  SkillStatus,
  discover_skills,
  document_sections,
  load_skill,
  load_skills,
  parse_frontmatter,
  read_document,
  required_keys,
  required_sections,
  skills_directory,
  split_document,
  tiers,
)
from stock_rl.skills import loader, schema

#: A body carrying every heading the schema requires, used as the base for
#: the malformed-file cases so that each one breaks exactly one rule.
body_sections = (
  '## Purpose\n'
  '\n'
  'Do the thing.\n'
  '\n'
  '## Data sources\n'
  '\n'
  '- a free feed\n'
  '\n'
  '## What it claims\n'
  '\n'
  'That it might work.\n'
  '\n'
  '## What the evidence actually says\n'
  '\n'
  'Nothing has been run.\n'
  '\n'
  '## Why this is rejected\n'
  '\n'
  'Because the criteria cannot be met.\n'
  '\n'
  '## Kill criteria\n'
  '\n'
  '1. beat_equal_weight, pre-registered above.\n'
)

rejected_body = body_sections


def frontmatter_text(name='demo', version=1, enabled=True,
                     tier='experimental', sources=('nse_eod_bhavcopy',),
                     criteria=None, extra=''):
  '''Render a frontmatter block for one skill file.

  Args:
    name: Skill name.
    version: Declared version.
    enabled: Declared enable flag.
    tier: Declared evidence tier.
    sources: Declared data sources.
    criteria: Kill criteria as ``(id, statement, threshold)`` triples.
      Defaults to one passable criterion.
    extra: Raw text appended inside the fence, for the malformed cases.

  Returns:
    The frontmatter text without its fences.
  '''
  pairs = criteria if criteria is not None else (
    ('beat_equal_weight', 'beats equal weight on a p-value', 'p <= 0.10'),
  )
  lines = [
    f'name: {name}',
    f'version: {version}',
    f'enabled: {str(enabled).lower()}',
    f'evidence_tier: {tier}',
    'data_sources:',
  ]
  lines.extend(f'  - {source}' for source in sources)
  lines.append('kill_criteria:')
  for criterion_id, statement, threshold in pairs:
    lines.append(f'  - id: {criterion_id}')
    lines.append(f'    statement: {statement}')
    lines.append(f'    threshold: {threshold}')
  if extra:
    lines.append(extra)
  return '\n'.join(lines)


def write_skill(directory, name='demo', body=body_sections, **kwargs):
  '''Write one skill file into a directory and return its path.

  Args:
    directory: Target directory.
    name: Skill name, which is also the file stem.
    body: Markdown body below the fence.
    kwargs: Forwarded to :func:`frontmatter_text`.

  Returns:
    Path of the written file.
  '''
  path = pathlib.Path(directory) / f'{name}.md'
  path.write_text(
    f'---\n{frontmatter_text(name=name, **kwargs)}\n---\n\n{body}',
    encoding='utf-8')
  return path


def frontmatter(name='demo', version=1, enabled=True,
                tier='experimental', sources=('nse_eod_bhavcopy',),
                criteria=None):
  '''Return a frontmatter mapping for schema-level tests.

  Built directly rather than parsed, because the frontmatter reader
  converts what it reads: a file declaring ``version: 1`` can never carry
  a string version, so the schema's type checks are only reachable
  through a hand-built mapping.

  Args:
    name: Skill name.
    version: Declared version, of whatever type.
    enabled: Declared enable flag, of whatever type.
    tier: Declared evidence tier.
    sources: Declared data sources.
    criteria: Kill criteria as ``(id, statement, threshold)`` triples.

  Returns:
    A mapping that satisfies the schema unless a field is wrong.
  '''
  pairs = criteria if criteria is not None else (
    ('beat_equal_weight', 'beats equal weight on a p-value', 'p <= 0.10'),
  )
  return {
    'name': name,
    'version': version,
    'enabled': enabled,
    'evidence_tier': tier,
    'data_sources': list(sources),
    'kill_criteria': [
      {'id': item[0], 'statement': item[1], 'threshold': item[2]}
      for item in pairs
    ],
  }


def make_skill(name='demo', tier='supported', enabled=True, body=None,
               criteria=None, version=1, sources=('nse_eod_bhavcopy',)):
  '''Build a validated skill without touching the filesystem.

  Args:
    name: Skill name.
    tier: Declared evidence tier.
    enabled: Declared enable flag.
    body: Markdown body, defaulting to a schema-complete body.
    criteria: Kill criteria as ``(id, statement, threshold)`` triples.
    version: Declared version.
    sources: Declared data sources.

  Returns:
    A validated ``Skill``.
  '''
  pairs = criteria if criteria is not None else (
    ('beat_equal_weight', 'beats equal weight on a p-value', 'p <= 0.10'),
  )
  return Skill.from_mapping(
    frontmatter(name=name, version=version, enabled=enabled, tier=tier,
                sources=sources, criteria=pairs),
    body if body is not None else body_sections,
    source=f'{name}.md')


def passing_gate(*skills):
  '''Return a gate in which every declared criterion has passed.

  Criterion ids are de-duplicated, because two skills routinely
  pre-register the same control arm (``beat_equal_weight``) and a gate is
  not scoped to one skill: one experiment usually answers criteria for
  several at once. Recording the same id twice is a disagreement about
  what was measured, and the gate refuses it.

  Args:
    skills: ``Skill`` objects, or skill names, whose criteria passed.

  Returns:
    A ``Gate`` carrying one passed result per distinct criterion id.
  '''
  wanted = [
    make_skill(name=name) if isinstance(name, str) else name
    for name in skills
  ]
  ids: list[str] = []
  for skill in wanted:
    for criterion in skill.criteria_ids:
      if criterion not in ids:
        ids.append(criterion)
  return Gate([KillResult(name, True, 'run on the full panel')
               for name in ids])


# ---------------------------------------------------------------------------
# The shipped files.
# ---------------------------------------------------------------------------

class TestShippedSkillsLoad:
  '''Every ``.md`` in the packaged directory loads and validates.'''

  def test_the_directory_ships_files(self):
    paths = discover_skills()
    assert paths, 'the packaged skills directory is empty'
    assert skills_directory().is_dir()
    assert all(path.suffix == '.md' for path in paths)

  def test_every_shipped_file_loads(self):
    skills = load_skills()
    assert len(skills) == len(discover_skills()), (
      'load_skills returned fewer skills than there are files')

  @pytest.mark.parametrize('path', discover_skills(),
                           ids=lambda path: path.name)
  def test_one_file_validates(self, path):
    skill = load_skill(path)
    assert skill.name == path.stem, (
      f'{path.name} declares name {skill.name!r}; the file name is part of '
      'the audit trail so the two must agree')
    assert skill.version >= 1
    assert skill.evidence_tier in tiers
    assert skill.data_sources
    assert all(source.strip() for source in skill.data_sources)
    assert skill.criteria_ids, 'a skill with no kill criterion can never '
    assert len(set(skill.criteria_ids)) == len(skill.criteria_ids)
    assert skill.source == path.name

  @pytest.mark.parametrize('path', discover_skills(),
                           ids=lambda path: path.name)
  def test_one_file_carries_every_required_heading(self, path):
    skill = load_skill(path)
    for section in required_sections:
      found = document_sections(skill.body)
      assert skill.has_section(section), (
        f'{path.name} has no {section!r} heading: {found}')
    if skill.evidence_tier == 'rejected':
      assert skill.has_section('why this is rejected'), (
        f'{path.name} is rejected but does not say why')

  def test_the_tiers_are_all_exercised(self):
    # Two rejected, one experimental, three supported. A file that moved
    # tier without the count moving would mean the evidence changed
    # silently, which is the event the registration fingerprint exists for.
    found = {skill.evidence_tier for skill in load_skills()}
    assert found == set(tiers), f'tiers present: {sorted(found)}'

  def test_the_package_exports_what_it_documents(self):
    exported = schema.__all__
    assert exported == sorted(set(exported), key=exported.index), (
      'every name in __all__ must exist and be listed once')
    assert schema.tiers == ('rejected', 'experimental', 'supported')
    assert set(required_keys) == {
      'data_sources', 'enabled', 'evidence_tier', 'kill_criteria', 'name',
      'version'}
    assert 'why this is rejected' in schema.rejected_sections

  def test_load_from_the_default_directory_matches_load_skills(self):
    assert len(load_skills(None)) == len(load_skills())
    assert SkillRegistry.load().names == tuple(
      sorted(skill.name for skill in load_skills()))


class TestManifestAndFingerprint:
  '''The fingerprint is a compliance control, so it has to be exact.'''

  def test_fingerprint_is_stable_for_identical_content(self):
    first = load_skill(discover_skills()[0])
    second = load_skill(discover_skills()[0])
    assert first.fingerprint == second.fingerprint
    scheme, _, digest = first.fingerprint.rpartition(':')
    assert scheme == 'stock_rl.policy.fingerprint/1', (
      'the skill fingerprint shares the policy registration scheme, so one '
      'record covers both')
    assert len(digest) == 64 and digest.isalnum(), 'a sha256 hex digest'

  def test_fingerprint_is_stable_across_directories(self, tmp_path):
    original = discover_skills()[0]
    relocated = tmp_path / 'relocated.md'
    relocated.write_text(read_document(original), encoding='utf-8')
    # The digest deliberately excludes the path: moving a skill is not
    # editing it, and a relocation that moved the registration fingerprint
    # would be a false re-registration event every deploy.
    assert load_skill(relocated).fingerprint == load_skill(original).fingerprint

  @pytest.mark.parametrize('change,other', [
    ('body', body_sections + '\nOne more sentence of reasoning.\n'),
    ('version', 2),
    ('tier', 'experimental'),
    ('enabled', False),
    ('sources', ('nse_eod_bhavcopy', 'nse_fii_dii_activity')),
  ], ids=['body', 'version', 'tier', 'enabled', 'sources'])
  def test_fingerprint_moves_when_the_decision_logic_moves(self, change,
                                                             other):
    kwargs = {
      'name': 'demo',
      'body': body_sections,
      'version': 1,
      'tier': 'supported',
      'enabled': True,
      'sources': ('nse_eod_bhavcopy',),
    }
    kwargs[change] = other
    assert make_skill().fingerprint != make_skill(**kwargs).fingerprint, (
      f'{change} is decision logic and must move the fingerprint: NSE '
      'operational modalities 9.1 and 9.9 require a fresh registration for '
      'a change to it')

  def test_fingerprint_is_not_the_plain_body_hash(self):
    skill = make_skill()
    assert skill.fingerprint != skill.body
    assert skill.digest == SkillDigest(
      skill.name, skill.version, skill.body, skill.enabled,
      skill.evidence_tier, skill.data_sources)

  def test_registry_manifest_covers_every_skill(self):
    registry = SkillRegistry.load()
    manifest = registry.manifest()
    assert isinstance(manifest, SkillManifest)
    assert manifest.names() == registry.names
    assert all(isinstance(record, RegistrationRecord)
               for record in manifest.skills)
    expected = {skill.name: skill.fingerprint for skill in registry}
    assert {record.name: record.fingerprint
            for record in manifest.skills} == expected
    assert registry.registration_fingerprint() == manifest.fingerprint()

  def test_manifest_fingerprint_moves_when_a_skill_is_added(self, tmp_path):
    before = SkillRegistry.load().registration_fingerprint()
    write_skill(tmp_path)
    assert SkillRegistry.load(tmp_path).registration_fingerprint() != before


# ---------------------------------------------------------------------------
# Loudness.
# ---------------------------------------------------------------------------

class TestMalformedFilesAreRejectedLoudly:
  '''A bad skill file raises. It is never skipped.'''

  def test_a_broken_file_does_not_stop_the_others_being_found(self, tmp_path):
    write_skill(tmp_path, name='good_one', tier='experimental')
    (tmp_path / 'broken.md').write_text('not a skill at all\n',
                                        encoding='utf-8')
    with pytest.raises(SkillFormatError) as caught:
      load_skills(tmp_path)
    message = str(caught.value)
    assert 'broken.md' in message
    assert '1 of 2' in message, message
    assert 'good_one' not in message, (
      'the message names rejected files only; a valid file has nothing to '
      'report')

  def test_load_skills_never_returns_a_subset(self, tmp_path):
    for name in ('one', 'two', 'three'):
      write_skill(tmp_path, name=name)
    (tmp_path / 'four.md').write_text('---\nname: four\n---\n',
                                      encoding='utf-8')
    with pytest.raises(SkillFormatError):
      load_skills(tmp_path)
    # Nothing was returned, so nothing downstream can run a subset.
    assert sorted(path.name for path in discover_skills(tmp_path)) == [
      'four.md', 'one.md', 'three.md', 'two.md']

  @pytest.mark.parametrize('text,expected', [
    ('no fence at all\n', 'fence'),
    ('---\nname: demo\n', 'never closed'),
  ], ids=['missing_fence', 'unclosed_fence'])
  def test_fence_faults(self, tmp_path, text, expected):
    (tmp_path / 'demo.md').write_text(text, encoding='utf-8')
    with pytest.raises(SkillFormatError, match=expected):
      load_skills(tmp_path)

  def test_a_non_markdown_file_in_the_directory_is_an_error(self, tmp_path):
    write_skill(tmp_path)
    (tmp_path / 'notes.txt').write_text('a note, not a skill\n',
                                        encoding='utf-8')
    with pytest.raises(SkillFormatError, match='expected a .md skill file'):
      load_skills(tmp_path)

  def test_a_subdirectory_is_an_error(self, tmp_path):
    write_skill(tmp_path)
    (tmp_path / 'nested').mkdir()
    with pytest.raises(SkillFormatError, match='directories are not skills'):
      load_skills(tmp_path)

  def test_editor_droppings_are_skipped(self, tmp_path):
    write_skill(tmp_path)
    (tmp_path / '.DS_Store').write_text('junk', encoding='utf-8')
    (tmp_path / '_draft.md').write_text('---\nname: draft\n---\n',
                                        encoding='utf-8')
    assert len(load_skills(tmp_path)) == 1

  def test_a_missing_directory_raises(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      discover_skills(tmp_path / 'absent')
    with pytest.raises(FileNotFoundError):
      load_skills(tmp_path / 'absent')

  @pytest.mark.parametrize('text,expected', [
    ('name: demo\nversion: 1\n\tname: x\n', 'tab'),
    ('   name: demo\n', 'indentation must be 0'),
    ('name demo\n', 'expected key: value'),
    ('name: demo\nname: again\n', 'duplicate key'),
    ('- orphan\n', 'no key above it'),
    ('name: demo\ndata_sources:\n  one\n', 'must use'),
    ('  stray: x\nname: demo\n', 'no key to attach to'),
    ('name: demo\nkill_criteria:\n    id: x\n', 'list above it is empty'),
    ('name: demo\nkill_criteria:\n  - a bare scalar\n    id: y\n',
     'no parent item'),
    ('name: demo\nkill_criteria:\n  - id: x\n    id: y\n',
     'duplicate key'),
    ('name: demo\nname: demo\n', 'duplicate key'),
  ], ids=[
    'tab', 'wrong_indent', 'no_colon', 'duplicate_top_key', 'orphan_item',
    'missing_dash', 'orphan_subkey', 'subkey_over_empty_list',
    'subkey_after_scalar_item', 'duplicate_subkey', 'duplicate_key',
  ])
  def test_frontmatter_grammar_faults(self, text, expected):
    with pytest.raises(SkillFormatError, match=expected):
      parse_frontmatter(text)

  def test_a_slow_path_fault_is_not_an_index_error(self):
    # A sub-key line under an empty list used to index an empty list, so
    # the caller saw an IndexError from inside the parser instead of the
    # SkillFormatError that names the line and the fault.
    with pytest.raises(SkillFormatError, match='line 3'):
      parse_frontmatter('kill_criteria:\n    id: x\n')

  def test_an_empty_key_is_refused(self):
    with pytest.raises(SkillFormatError, match='key is empty'):
      parse_frontmatter(': orphan value\n')

  def test_an_unreadable_file_is_reported_not_skipped(self, tmp_path,
                                                       monkeypatch):
    write_skill(tmp_path)

    def deny(path):
      del path
      raise PermissionError('permission denied')

    monkeypatch.setattr(loader, 'load_skill', deny)
    with pytest.raises(SkillFormatError) as caught:
      load_skills(tmp_path)
    message = str(caught.value)
    assert 'unreadable' in message
    assert 'permission denied' in message, (
      'an unreadable file is a licence or permissions problem, and the '
      'reason has to survive into the aggregate error')

  def test_a_comment_and_a_blank_line_are_skipped(self):
    fields = parse_frontmatter(
      '# a comment\n\nname: demo\n\n  # indented comment\nversion: 2\n')
    assert fields == {'name': 'demo', 'version': 2}

  def test_scalar_conversion_is_explicit(self):
    fields = parse_frontmatter(
      'truthy: true\nfalsy: false\nnothing: null\ntilde: ~\n'
      'count: 3\nratio: 1.5\nthousand: 1_000\ninfinite: inf\n'
      'not_a_number: nan\ntext: hello world\nquoted: "0.5"\n'
      "single: 'true'\n")
    assert fields['truthy'] is True
    assert fields['falsy'] is False
    assert fields['nothing'] is None
    assert fields['tilde'] is None
    assert fields['count'] == 3
    assert fields['ratio'] == 1.5
    assert fields['thousand'] == '1_000', 'python spellings stay text'
    assert fields['infinite'] == 'inf'
    assert fields['not_a_number'] == 'nan'
    assert fields['text'] == 'hello world'
    assert fields['quoted'] == '0.5', 'a quoted scalar is text'
    assert fields['single'] == 'true'

  def test_a_url_in_a_list_item_stays_a_scalar(self):
    fields = parse_frontmatter('sources:\n  - https://example.invalid/x\n')
    assert fields['sources'] == ['https://example.invalid/x']

  @pytest.mark.parametrize('override,expected', [
    ({'tier': 'vibes'}, 'must be one of'),
    ({'name': 'Mixed Case'}, 'lowercase_with_underscores'),
    ({'version': 0}, '1 or greater'),
  ], ids=['unknown_tier', 'bad_name', 'version_zero'])
  def test_schema_faults_reachable_through_a_file(self, tmp_path, override,
                                                   expected):
    kwargs = {
      'name': 'demo',
      'tier': 'experimental',
      'version': 1,
    }
    kwargs.update(override)
    path = write_skill(tmp_path, **kwargs)
    with pytest.raises(SkillFormatError, match=expected) as caught:
      load_skill(path)
    assert path.name in str(caught.value), (
      'the message must name the file, not just the fault')

  @pytest.mark.parametrize('fields,expected', [
    ({'version': '1'}, 'whole number'),
    ({'version': 1.0}, 'whole number'),
    ({'version': True}, 'whole number'),
    ({'version': None}, 'whole number'),
    ({'enabled': 'yes'}, 'must be true or false'),
    ({'enabled': 1}, 'must be true or false'),
    ({'enabled': None}, 'must be true or false'),
    ({'name': ''}, 'non-empty string'),
    ({'name': 7}, 'non-empty string'),
    ({'name': '9leading'}, 'lowercase_with_underscores'),
    ({'evidence_tier': None}, 'must be one of'),
    ({'evidence_tier': 7}, 'must be one of'),
    ({'data_sources': []}, 'must list at least one source'),
    ({'data_sources': ['ok', '  ']}, 'non-empty strings'),
    ({'data_sources': ['ok', 3]}, 'non-empty strings'),
    ({'data_sources': 'nse_eod_bhavcopy'}, 'must be a list'),
    ({'data_sources': {'nse': 1}}, 'must be a list'),
    ({'kill_criteria': []}, 'must pre-register at least one criterion'),
    ({'kill_criteria': ['id']}, 'must declare an id'),
    ({'kill_criteria': [{'id': 'a'}]}, 'both id and statement'),
    ({'kill_criteria': [{'id': '', 'statement': 'x'}]},
     'non-empty string'),
    ({'kill_criteria': [{'id': 'a', 'statement': 'x', 'threshold': 0.1}]},
     'threshold must be a'),
  ], ids=[
    'version_string', 'version_float', 'version_bool', 'version_none',
    'enabled_string', 'enabled_int', 'enabled_none', 'blank_name',
    'non_string_name', 'name_starting_with_a_digit', 'tier_none',
    'tier_number', 'empty_sources', 'blank_source', 'non_string_source',
    'source_string', 'source_mapping', 'empty_criteria',
    'criteria_scalar_item', 'criteria_missing_statement',
    'criteria_blank_id', 'criteria_float_threshold',
  ])
  def test_schema_faults_only_a_mapping_can_carry(self, fields, expected):
    mapping = frontmatter()
    mapping.update(fields)
    with pytest.raises(SkillFormatError, match=expected):
      Skill.from_mapping(mapping, body_sections)

  def test_missing_and_unknown_keys_are_both_refused(self, tmp_path):
    path = tmp_path / 'demo.md'
    path.write_text(
      '---\nname: demo\nversion: 1\nextra_key: 1\n---\n\n'
      + body_sections, encoding='utf-8')
    with pytest.raises(SkillFormatError, match='missing required key'):
      load_skill(path)
    complete = frontmatter_text(name='demo')
    complete += '\nmystery: 1'
    path.write_text(f'---\n{complete}\n---\n\n{body_sections}',
                    encoding='utf-8')
    with pytest.raises(SkillFormatError, match='unknown key'):
      load_skill(path)

  def test_a_criterion_needs_an_id_and_a_statement(self, tmp_path):
    path = tmp_path / 'demo.md'
    text = frontmatter_text(name='demo', extra='  - statement: only this')
    path.write_text(f'---\n{text}\n---\n\n{body_sections}', encoding='utf-8')
    with pytest.raises(SkillFormatError, match='both id and statement'):
      load_skill(path)
    other = tmp_path / 'other.md'
    text = frontmatter_text(name='other') + '\n  - a bare scalar item'
    other.write_text(f'---\n{text}\n---\n\n{body_sections}',
                     encoding='utf-8')
    with pytest.raises(SkillFormatError, match='must declare an id'):
      load_skill(other)

  def test_a_repeated_criterion_id_is_refused(self):
    mapping = frontmatter(criteria=(('beat_equal_weight', 'one', 'p'),
                                    ('beat_equal_weight', 'two', 'p')))
    with pytest.raises(SkillFormatError, match='duplicate kill criterion'):
      Skill.from_mapping(mapping, body_sections)

  def test_a_criterion_entry_without_a_key_is_refused(self, tmp_path):
    path = tmp_path / 'demo.md'
    text = frontmatter_text(name='demo', criteria=())
    path.write_text(f'---\n{text}\n---\n\n{body_sections}', encoding='utf-8')
    with pytest.raises(SkillFormatError, match='must be a list'):
      load_skill(path)

  def test_a_threshold_must_be_text(self, tmp_path):
    path = tmp_path / 'demo.md'
    text = frontmatter_text(name='demo', criteria=())
    text += '\n  - id: demo\n    statement: a bar\n    threshold: 0.10\n'
    path.write_text(f'---\n{text}\n---\n\n{body_sections}', encoding='utf-8')
    with pytest.raises(SkillFormatError, match='threshold must be a'):
      load_skill(path)

  def test_a_missing_heading_names_all_of_them(self, tmp_path):
    thin = '## Purpose\n\nDo the thing.\n'
    path = write_skill(tmp_path, body=thin)
    with pytest.raises(SkillFormatError) as caught:
      load_skill(path)
    message = str(caught.value).casefold()
    for heading in ('data sources', 'what it claims', 'kill criteria',
                    'what the evidence actually says'):
      assert heading in message, (
        f'{heading!r} is missing and must be named: one broken file takes '
        'one edit, not one edit per heading')

  def test_a_rejected_skill_must_say_why(self, tmp_path):
    without = body_sections.replace(
      '## Why this is rejected\n\nBecause the criteria cannot be met.\n\n',
      '')
    path = write_skill(tmp_path, tier='rejected', body=without)
    with pytest.raises(SkillFormatError) as caught:
      load_skill(path)
    assert 'why this is rejected' in str(caught.value).casefold()
    assert load_skill(write_skill(tmp_path, name='kept', tier='rejected',
                                  body=body_sections))

  def test_headings_are_matched_case_and_trailing_punctuation_insensitively(
      self):
    shouted = ('## PURPOSE:\n\n## Data Sources\n\n## What it Claims:\n\n'
               '## WHAT THE EVIDENCE ACTUALLY SAYS\n\n## Kill Criteria.\n')
    skill = make_skill(body=shouted)
    assert skill.has_section('purpose')
    assert skill.has_section('what it claims')
    assert not skill.has_section('why this is rejected')
    assert 'purpose' in document_sections(shouted)

  def test_from_mapping_refuses_a_non_mapping(self):
    with pytest.raises(SkillFormatError, match='key and value block'):
      Skill.from_mapping(['not', 'a', 'mapping'], body_sections)

  def test_split_document_returns_both_halves(self):
    text = f'---\n{frontmatter_text()}\n---\n\n{body_sections}'
    front, body = split_document(text)
    assert 'name: demo' in front
    assert body.strip() == body_sections.strip()

  def test_error_types_are_distinguishable(self):
    assert issubclass(SkillFormatError, SkillError)
    assert not issubclass(SkillError, ValueError), (
      'a caller must be able to tell a deployment failure from a bug in '
      'their own arguments')


# ---------------------------------------------------------------------------
# The tier gate.
# ---------------------------------------------------------------------------

class TestRejectedTierIsUnreachable:
  '''No flag combination reaches a rejected skill.'''

  def test_the_gate_is_checked_before_the_flags(self):
    registry = SkillRegistry([make_skill(name='dead', tier='rejected')])
    status = registry.status('dead', passing_gate('dead'), ['dead'], True)
    assert status.active is False
    assert status.experiment_only is False
    assert status.allowed_live is False
    assert 'rejected' in status.reason
    assert 'no flag combination' in status.reason

  def test_no_flag_combination_activates_it(self):
    skill = make_skill(name='dead', tier='rejected')
    registry = SkillRegistry([skill])
    gates = [
      None,
      Gate(),
      passing_gate('dead'),
      Gate().record('beat_equal_weight', True),
    ]
    enables: list[Iterable[str]] = [(), ('dead',), ('dead', 'other'),
                                     tuple(registry.names)]
    for gate, enabled, experiment in itertools.product(
        gates, enables, (False, True)):
      status = registry.status('dead', gate, enabled, experiment)
      assert status.active is False, (
        f'gate={gate!r} enabled={enabled} experiment={experiment} '
        f'activated a rejected skill')
      assert status.experiment_only is False
      assert status.allowed_live is False
    assert not registry.active(passing_gate('dead'), ['dead'], True)

  def test_a_rejected_skill_declares_enabled_false_is_still_refused(self):
    registry = SkillRegistry([
      make_skill(name='dead', tier='rejected', enabled=False)])
    status = registry.status('dead', passing_gate('dead'), ['dead'], True)
    assert status.active is False
    assert 'rejected' in status.reason

  def test_every_shipped_rejected_skill_is_unreachable(self):
    registry = SkillRegistry.load()
    rejected = [skill for skill in registry
                if skill.evidence_tier == 'rejected']
    assert rejected, 'no shipped skill sits at the rejected tier'
    everything = passing_gate(*registry.names)
    for skill in rejected:
      status = registry.status(skill.name, everything, registry.names, True)
      assert status.active is False
      assert 'rejected' in status.reason


class TestSupportedTierNeedsItsKillCriteria:
  '''A supported skill refuses until every criterion has passed.'''

  def test_an_empty_gate_refuses(self):
    registry = SkillRegistry([make_skill(name='live', tier='supported')])
    status = registry.status('live', Gate(), ['live'])
    assert status.active is False
    assert 'no result for' in status.reason
    assert 'beat_equal_weight' in status.reason

  def test_a_missing_gate_argument_is_an_empty_gate(self):
    registry = SkillRegistry([make_skill(name='live', tier='supported')])
    assert registry.status('live', None, ['live']).active is False

  def test_a_failed_criterion_refuses(self):
    registry = SkillRegistry([make_skill(name='live', tier='supported')])
    gate = Gate([KillResult('beat_equal_weight', False, 'p = 0.61')])
    status = registry.status('live', gate, ['live'])
    assert status.active is False
    assert 'failed' in status.reason

  def test_partial_evidence_is_not_a_pass(self):
    skill = make_skill(name='live', tier='supported', criteria=(
      ('first', 'one bar', 'p <= 0.10'),
      ('second', 'another bar', 'p <= 0.10'),
    ))
    registry = SkillRegistry([skill])
    gate = Gate([KillResult('first', True)])
    status = registry.status('live', gate, ['live'])
    assert status.active is False, (
      'one passed criterion out of two is not a pass; a skill clears every '
      'bar it set for itself rather than the subset somebody ran')
    assert 'second' in status.reason
    full = Gate([KillResult('first', True), KillResult('second', True)])
    assert registry.status('live', full, ['live']).active is True

  def test_both_flags_are_needed(self):
    registry = SkillRegistry([make_skill(name='live', tier='supported')])
    gate = passing_gate('live')
    assert registry.status('live', gate, ()).active is False
    assert 'enable set' in registry.status('live', gate, ()).reason
    off = SkillRegistry([
      make_skill(name='live', tier='supported', enabled=False)])
    assert off.status('live', gate, ['live']).active is False
    assert 'enabled: false' in off.status('live', gate, ['live']).reason

  def test_a_passed_criterion_activates_it(self):
    registry = SkillRegistry([make_skill(name='live', tier='supported')])
    status = registry.status('live', passing_gate('live'), ['live'])
    assert status.active is True
    assert status.experiment_only is False
    assert status.allowed_live is True
    assert 'supported' in status.reason

  def test_the_status_carries_the_fingerprint_and_version(self):
    skill = make_skill(name='live', tier='supported')
    status = SkillRegistry([skill]).status('live')
    assert isinstance(status, SkillStatus)
    assert status.name == 'live'
    assert status.version == skill.version
    assert status.evidence_tier == 'supported'
    assert status.fingerprint == skill.fingerprint

  def test_an_unknown_name_raises_rather_than_returning_false(self):
    registry = SkillRegistry([make_skill(name='live', tier='supported')])
    with pytest.raises(KeyError, match='unknown skill'):
      registry.status('absent')
    with pytest.raises(KeyError, match='registered: live'):
      registry.get('absent')

  def test_every_shipped_supported_skill_refuses_without_a_gate(self):
    registry = SkillRegistry.load()
    supported = [skill for skill in registry
                 if skill.evidence_tier == 'supported']
    assert supported
    for skill in supported:
      status = registry.status(skill.name, Gate(), registry.names)
      assert status.active is False, (
        f'{skill.name} activated with no criterion result at all')
      live = registry.status(skill.name, passing_gate(skill),
                             [skill.name])
      assert live.active is True, (
        f'{skill.name} stayed off with every criterion passed')


class TestExperimentalTierIsLabelled:
  '''An experiment may run; it may not be mistaken for a live input.'''

  def test_experiment_mode_activates_and_labels(self):
    registry = SkillRegistry([
      make_skill(name='try_me', tier='experimental')])
    status = registry.status('try_me', Gate(), ['try_me'], True)
    assert status.active is True
    assert status.experiment_only is True
    assert status.allowed_live is False, (
      'an active experimental skill must still refuse the live order path')
    assert 'experiment mode' in status.reason

  def test_live_mode_refuses(self):
    registry = SkillRegistry([
      make_skill(name='try_me', tier='experimental')])
    status = registry.status('try_me', Gate(), ['try_me'], False)
    assert status.active is False
    assert status.experiment_only is False
    assert 'experiment mode only' in status.reason

  def test_the_flags_still_apply(self):
    registry = SkillRegistry([
      make_skill(name='try_me', tier='experimental')])
    assert registry.status('try_me', Gate(), (), True).active is False
    off = SkillRegistry([
      make_skill(name='try_me', tier='experimental', enabled=False)])
    assert off.status('try_me', Gate(), ['try_me'], True).active is False

  def test_no_gate_is_needed_because_nothing_is_claimed(self):
    # The kill criteria are still pre-registered, and the registry refuses
    # to read them as a pass, but an experimental run is a measurement of
    # the criteria rather than a claim that cleared them.
    registry = SkillRegistry([
      make_skill(name='try_me', tier='experimental')])
    assert registry.status('try_me', Gate(), ['try_me'], True).active is True

  def test_active_and_statuses_agree(self):
    registry = SkillRegistry([
      make_skill(name='one', tier='experimental'),
      make_skill(name='two', tier='supported'),
      make_skill(name='three', tier='rejected'),
    ])
    gate = passing_gate('two')
    statuses = registry.statuses(gate, ['one', 'two', 'three'], True)
    assert [status.name for status in statuses] == ['one', 'three', 'two']
    active = registry.active(gate, ['one', 'two', 'three'], True)
    assert {skill.name for skill in active} == {'one', 'two'}
    assert all(status.active == (status.name in {'one', 'two'})
               for status in statuses)
    assert all(status.experiment_only
               for status in statuses if status.name == 'one')

  def test_every_shipped_experimental_skill_is_experiment_only(self):
    registry = SkillRegistry.load()
    experimental = [skill for skill in registry
                    if skill.evidence_tier == 'experimental']
    assert experimental
    for skill in experimental:
      live = registry.status(skill.name, passing_gate(skill.name),
                             [skill.name], False)
      assert live.active is False
      assert live.experiment_only is False
      run = registry.status(skill.name, Gate(), [skill.name], True)
      assert run.active is True
      assert run.experiment_only is True
      assert run.allowed_live is False


class TestDeactivationIsNotCached:
  '''A criterion that failed later switches the skill off immediately.'''

  def test_a_criterion_that_later_failed_deactivates(self):
    skill = make_skill(name='live', tier='supported')
    registry = SkillRegistry([skill])
    gate = passing_gate(skill)
    assert registry.status('live', gate, ['live']).active is True
    later = Gate([KillResult('beat_equal_weight', False,
                             're-run on 2026 data: p = 0.61')])
    status = registry.status('live', later, ['live'])
    assert status.active is False, (
      'a kill criterion that passes once and fails later must stop the '
      'skill on the next call, with no restart and no cache to clear')
    assert 'failed' in status.reason
    assert status.fingerprint == skill.fingerprint

  def test_a_gate_is_not_scoped_to_one_skill(self):
    # One experiment usually answers criteria for several skills at once,
    # so recording a result for a skill the gate is not asked about is
    # normal rather than an error.
    shared = passing_gate(make_skill(name='one', tier='supported'),
                          make_skill(name='two', tier='supported'))
    assert shared.result('beat_equal_weight').passed is True
    assert len(shared) == 1

  def test_record_refuses_to_overwrite_an_existing_verdict(self):
    gate = Gate([KillResult('a', True, 'first run')])
    with pytest.raises(ValueError, match='duplicate kill result'):
      gate.record('a', False, 'second run')

  def test_a_gate_record_returns_a_new_gate(self):
    first = Gate([KillResult('a', True)])
    second = first.record('b', False, 'note')
    assert len(first) == 1
    assert len(second) == 2
    assert first != second
    assert second.result('b') == KillResult('b', False, 'note')
    assert second.outcome('b') is False
    assert second.outcome('a') is True
    assert second.outcome('absent') is None
    assert second.as_dict() == {'a': True, 'b': False}
    assert [item.criterion for item in second] == ['a', 'b']
    assert 'Gate(1/2 passed)' == repr(second)

  def test_gate_helpers_report_per_criterion_state(self):
    skill = make_skill(name='live', criteria=(
      ('passed', 'cleared', 'p <= 0.10'),
      ('failed', 'not cleared', 'p <= 0.10'),
      ('missing', 'never run', 'p <= 0.10'),
    ))
    gate = Gate([KillResult('passed', True), KillResult('failed', False)])
    assert gate.missing(skill.kill_criteria) == ('missing',)
    assert gate.failed(skill.kill_criteria) == ('failed',)
    assert gate.all_passed(skill.kill_criteria) is False
    assert gate.all_passed(()) is False, (
      'no criteria is not a pass; there is nothing to have passed')

  @pytest.mark.parametrize('results,error', [
    ([KillResult('', True)], ValueError),
    ([KillResult('a', True), KillResult('a', False)], ValueError),
    (['not a result'], TypeError),
  ], ids=['blank_criterion', 'duplicate_criterion', 'wrong_type'])
  def test_gate_rejects_inconsistent_input(self, results, error):
    with pytest.raises(error):
      Gate(results)

  def test_gates_compare_by_value(self):
    assert Gate([KillResult('a', True)]) == Gate([KillResult('a', True)])
    assert Gate([KillResult('a', True)]) != Gate([KillResult('a', False)])
    assert Gate() != Gate([KillResult('a', True)])
    # Comparing against a non-gate returns NotImplemented so Python falls
    # back to identity, rather than claiming two unrelated objects differ.
    assert (Gate() == object()) is False
    assert len(Gate()) == 0

  def test_registry_rejects_duplicate_names_and_wrong_types(self):
    skill = make_skill(name='twice', tier='experimental')
    with pytest.raises(ValueError, match='duplicate skill name'):
      SkillRegistry([skill, skill])
    with pytest.raises(TypeError, match='expected a Skill'):
      SkillRegistry(['not a skill'])
    assert 'SkillRegistry(0 skills)' == repr(SkillRegistry())

  def test_registry_iteration_is_name_ordered(self):
    registry = SkillRegistry([
      make_skill(name='zebra', tier='experimental'),
      make_skill(name='alpha', tier='experimental'),
    ])
    assert [skill.name for skill in registry] == ['alpha', 'zebra']
    assert 'alpha' in registry
    assert 'zebra' in registry
    assert 'absent' not in registry
    assert len(registry) == 2


# ---------------------------------------------------------------------------
# Invariant: no unearned performance claims.
# ---------------------------------------------------------------------------

#: Words that make a numeric figure a claim about PERFORMANCE rather than
#: about a cost, a data count or a threshold. A figure next to one of
#: these is the kind that ends up in a pitch deck.
performance_words = (
  'sharpe', 'return', 'alpha', 'yield', 'cagr', 'excess', 'performance',
  'drawdown', 'profit',
)

#: A numeric figure carrying a unit, or a bare numeric range. Plain
#: integers are excluded: a criterion threshold of "at least 4 of 5 folds"
#: is a bar to clear, not a result.
figure_pattern = re.compile(
  r'\d[\d.,]*\s*(?:percent|percentage points|bps|%|times)\b'
  r'|\b\d+(?:\.\d+)?\s*(?:to|-|–)\s*\d+(?:\.\d+)?\b',
  re.IGNORECASE)

#: Markers that a figure belongs to somebody else. A working paper, a
#: named sample, a year in parentheses: the figure is a citation.
attribution_words = (
  'report', 'find', 'found', 'study', 'studies', 'paper', 'papers',
  'review', 'et al', 'jfqa', 'ssrn', 'author', 'authors', 'sample',
  'samples', 'universe', 'cite', 'citation', 'doi', 'published',
)

#: Markers that a figure is explicitly NOT this project's own result.
unverified_words = (
  'artefact', 'artifact', 'unverified', 'not harvestable', 'reported figures',
  'void', 'none is claimed', 'no return figure', 'not claimed', 'may not',
  'must not', 'prohibit', 'decayed', 'decay', 'bias', 'inflate', 'failed',
  'inferior', 'required', 'threshold', 'criterion', 'criteria', 'break',
  'against', 'premium', 'drift', 'drifts',
)

#: First-person and project-claiming constructions. A figure next to one of
#: these is a claim about THIS system, which is exactly what no skill file
#: here is allowed to make.
claim_pattern = re.compile(
  r'\b(?:we|our|ours|here|this project|this package|this skill|this file)\b'
  r'[^.!?]{0,80}?\b(?:sharpe|return|alpha|yield|performance|drawdown|profit|'
  r'percent|bps)\b', re.IGNORECASE)

#: A prohibition or an explicit denial. "May not claim alpha" is the file
#: forbidding a figure, which is the opposite of asserting one, so a
#: sentence carrying one of these is never treated as a claim.
denial_words = (
  'may not', 'must not', 'cannot', 'can not', 'never', 'nothing', 'none',
  'no ', 'not ', 'without', 'void', 'forbid', 'prohibit', 'refuse',
)

#: Sentence boundaries, for the same reason: a claim is made per sentence,
#: not per file.
sentence_pattern = re.compile(r'[^.!?\n]+[.!?]?')


def paragraphs(body):
  '''Split a markdown body into prose blocks.

  Args:
    body: Markdown body below the frontmatter fence.

  Returns:
    Non-empty blocks with surrounding whitespace stripped, in document
    order.
  '''
  blocks = []
  for block in re.split(r'\n\s*\n', body):
    text = block.strip()
    if text:
      blocks.append(text)
  return blocks


def _section_body(body, title):
  '''Return the text of one level-2 section.

  Args:
    body: Markdown body below the frontmatter fence.
    title: Heading title, matched case-insensitively.

  Returns:
    The section's text with surrounding blank lines stripped, or an empty
    string when the section is absent.
  '''
  blocks = re.split(r'\n(?=##\s)', body)
  for block in blocks:
    heading, _, text = block.partition('\n')
    wanted = title.strip().rstrip(':.').casefold()
    if heading.strip().lstrip('#').strip().rstrip(':.').casefold() == wanted:
      return text.strip()
  return ''


def unsupported_claims(body):
  '''Return the prose blocks that assert performance nobody measured.

  Two faults, checked separately because they are different mistakes.

  The first is a block that carries both a performance word (Sharpe,
  return, alpha, yield, excess, performance, drawdown, profit) and a
  numeric figure with a unit or a range, while naming neither an external
  source nor an explicit statement that the figure is not this project's
  own claim. That block is quoting somebody else's measurement, so it is
  allowed; anything else is a number this project has not produced.

  The second is a first-person performance claim, which is never
  acceptable: a sentence that says "this skill" and then names a return
  or a Sharpe is asserting a result, whatever else the sentence says.

  Scope: performance words, not every number in the file. A kill
  criterion's threshold of "at least 4 of 5 folds" is a bar to clear and
  a cost constant of "about 22 bps" is an input; neither is a result, and
  flagging them would train people to ignore the scanner.

  Args:
    body: Markdown body below the frontmatter fence.

  Returns:
    Human-readable descriptions of the offending text, in document order.
    Empty when the body supports everything it asserts.
  '''
  offenders = []
  for block in paragraphs(body):
    lowered = block.casefold()
    if not any(word in lowered for word in performance_words):
      continue
    if figure_pattern.search(block) is None:
      continue
    attributed = any(word in lowered for word in attribution_words)
    marked = any(word in lowered for word in unverified_words)
    if not (attributed or marked):
      offenders.append(f'unsupported figure: {block[:120]}')
  for sentence in sentence_pattern.findall(body):
    lowered = sentence.casefold()
    if any(word in lowered for word in denial_words):
      continue
    match = claim_pattern.search(sentence)
    if match is not None:
      offenders.append(
        f'first-person performance claim: {match.group(0)!r}')
  return offenders


class TestNoUnearnedPerformanceClaims:
  '''No shipped markdown may assert a Sharpe or a return it cannot back.'''

  def test_the_scanner_detects_a_body_that_claims_a_result(self):
    claiming = ('## What it claims\n\n'
                'This skill earns a Sharpe of 1.8 on Nifty-50 names.\n\n')
    assert unsupported_claims(claiming), (
      'the scanner must fail on a body that simply asserts a Sharpe')

  def test_the_scanner_detects_an_unattributed_percentage(self):
    claiming = ('## What the evidence actually says\n\n'
                'The rule delivers a 14 percent return per year after '
                'costs.\n\n')
    assert unsupported_claims(claiming), (
      'a return figure with no source and no disclaimer is a claim')

  def test_the_scanner_flags_a_first_person_claim_even_when_attributed(self):
    claiming = ('## What it claims\n\n'
                'We measured a Sharpe of 1.8 on our own panel.\n\n')
    offenders = unsupported_claims(claiming)
    assert any('first-person' in item for item in offenders)

  def test_the_scanner_does_not_flag_a_prohibition(self):
    forbidden = ('## What this skill may not do\n\n'
                 'This skill may not be described as producing alpha of 4 '
                 'percent a year.\n\n')
    assert not unsupported_claims(forbidden)

  def test_the_scanner_allows_a_cited_figure(self):
    cited = ('## What the evidence actually says\n\n'
             'Agarwalla, Jacob and Varma (2014) report a low-volatility '
             'excess return of 10.62 percent against 4.82 percent on a '
             'beta sort.\n\n')
    assert not unsupported_claims(cited)

  def test_the_scanner_allows_a_figure_marked_as_an_artefact(self):
    marked = ('## What the evidence actually says\n\n'
              'Reported figures of 6 to 9 on winner-minus-loser '
              'long-short are artefacts of markets where shorting was '
              'banned, and they are not harvestable here.\n\n')
    assert not unsupported_claims(marked)

  def test_the_scanner_allows_a_threshold_that_is_not_a_result(self):
    threshold = ('## Kill criteria\n\n'
                 'The delivery round trip is about 22 bps, so a breakeven '
                 'below roughly 30 bps is dead on arrival.\n\n')
    assert not unsupported_claims(threshold)

  @pytest.mark.parametrize('path', discover_skills(),
                           ids=lambda path: path.name)
  def test_no_shipped_file_claims_performance_it_cannot_support(self, path):
    offenders = unsupported_claims(read_document(path))
    assert not offenders, f'{path.name}: {offenders}'

  @pytest.mark.parametrize('path', discover_skills(),
                           ids=lambda path: path.name)
  def test_no_shipped_file_asserts_a_first_person_performance_figure(self,
                                                                     path):
    offenders = [
      item for item in unsupported_claims(read_document(path))
      if item.startswith('first-person')
    ]
    assert not offenders, f'{path.name}: {offenders}'

  def test_no_shipped_file_states_a_target_return(self):
    # Three of the six files forbid this in their own words, and the rule
    # is enforced rather than requested: a skill file that quotes a target
    # return is quoting a number nobody measured on this system. Where the
    # phrase appears it must be inside a prohibition.
    pattern = re.compile(r'(?:target|expected)\s+(?:annual\s+)?return',
                         re.IGNORECASE)
    for path in discover_skills():
      for sentence in sentence_pattern.findall(read_document(path)):
        if pattern.search(sentence) is None:
          continue
        lowered = sentence.casefold()
        assert any(word in lowered for word in denial_words), (
          f'{path.name} states {sentence.strip()!r}')

  @pytest.mark.parametrize('path', discover_skills(),
                           ids=lambda path: path.name)
  def test_every_file_states_what_it_may_not_do(self, path):
    skill = load_skill(path)
    assert skill.has_section('what this skill may not do'), (
      f'{path.name} states no prohibitions, so nothing bounds it')

  @pytest.mark.parametrize('path', discover_skills(),
                           ids=lambda path: path.name)
  def test_kill_criteria_appear_in_the_body_as_well_as_the_frontmatter(
      self, path):
    skill = load_skill(path)
    body = skill.body.casefold()
    for criterion in skill.kill_criteria:
      assert criterion.statement, f'{path.name}: blank statement'
      assert criterion.id.casefold() in body, (
        f'{path.name}: criterion {criterion.id!r} is pre-registered but '
        'never explained in the body, so nobody can check what it means')

  @pytest.mark.parametrize('path', discover_skills(),
                           ids=lambda path: path.name)
  def test_thresholds_are_text_and_therefore_unambiguous(self, path):
    skill = load_skill(path)
    assert all(isinstance(criterion.threshold, str)
               for criterion in skill.kill_criteria)
    assert all(criterion.threshold.strip()
               for criterion in skill.kill_criteria), (
      f'{path.name} has a criterion with no threshold, which is a criterion '
      'nobody can fail')

  @pytest.mark.parametrize('path', discover_skills(),
                           ids=lambda path: path.name)
  def test_the_declared_tier_matches_the_prose(self, path):
    skill = load_skill(path)
    claims = _section_body(skill.body, 'what it claims').casefold()
    if skill.evidence_tier == 'rejected':
      assert claims.startswith('nothing'), (
        f'{path.name} is rejected but its claims section opens with '
        f'{claims[:60]!r}; a rejected file has to say it claims nothing')
    else:
      assert not claims.startswith('nothing'), (
        f'{path.name} is {skill.evidence_tier} yet disclaims every claim')

  @pytest.mark.parametrize('path', discover_skills(),
                           ids=lambda path: path.name)
  def test_data_sources_are_described_in_the_body(self, path):
    skill = load_skill(path)
    described = _section_body(skill.body, 'data sources')
    bullets = [line for line in described.splitlines()
               if line.strip().startswith('- ')]
    assert len(bullets) >= len(skill.data_sources), (
      f'{path.name} declares {len(skill.data_sources)} sources and lists '
      f'{len(bullets)} in prose; a source nobody describes cannot be '
      'reconciled against a contract')

  def test_tier_counts_match_what_the_package_documents(self):
    counts = {'rejected': 0, 'experimental': 0, 'supported': 0}
    for skill in load_skills():
      counts[skill.evidence_tier] += 1
    assert counts == {'rejected': 2, 'experimental': 1, 'supported': 3}, (
      f'the shipped tiers are {counts}; the package docstring quotes two '
      'rejected, five experimental and three supported, so a change here '
      'means a change to that prose')

  def test_every_tier_is_exercised_by_the_registry(self):
    registry = SkillRegistry.load()
    everything = passing_gate(*registry.names)
    seen = {status.evidence_tier for status
            in registry.statuses(everything, registry.names, True)}
    assert seen == set(tiers)

  def test_no_third_party_import_in_the_package(self):
    # The package is loaded from markdown at runtime, so a stray import
    # would put a dependency between a skill file and the interpreter.
    import ast  # noqa: PLC0415  pylint: disable=import-outside-toplevel
    root = pathlib.Path(loader.__file__).resolve().parent
    for path in sorted(root.glob('*.py')):
      tree = ast.parse(path.read_text(encoding='utf-8'))
      for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
          modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
          modules = [node.module or '']
        for module in modules:
          root_module = module.split('.')[0]
          assert root_module in ('stock_rl', 'ast', 'collections', 'abc',
                                 'dataclasses', 'datetime', 'enum',
                                 'itertools', 'math', 'pathlib', 'random',
                                 're', 'statistics', 'typing',
                                 '__future__'), (
            f'{path.name} imports {module}')

  def test_skill_names_are_unique_across_the_directory(self):
    names = [skill.name for skill in load_skills()]
    assert len(set(names)) == len(names)

  def test_the_registry_survives_a_reload(self):
    first = SkillRegistry.load()
    second = SkillRegistry.load()
    assert first.registration_fingerprint() == second.registration_fingerprint()

  def test_no_skill_file_is_empty(self):
    for path in discover_skills():
      assert len(read_document(path)) > 500, (
        f'{path.name} is a stub; an agent with no prose is not an agent')

  def test_schema_helpers_are_public_and_documented(self):
    assert callable(schema.document_sections)
    assert callable(loader.describe_sources)
    sources = loader.describe_sources(load_skills())
    assert sources == tuple(sorted(set(sources)))
    assert 'nse_eod_bhavcopy' in sources

  def test_kill_criterion_and_digest_are_frozen(self):
    criterion = KillCriterion('a', 'b', 'c')
    with pytest.raises(AttributeError):
      criterion.id = 'other'
    digest = SkillDigest('demo', 1, 'body')
    with pytest.raises(AttributeError):
      digest.version = 2

  def test_tier_type_is_a_literal_of_the_three_tiers(self):
    assert isinstance(tiers, tuple)
    assert all(isinstance(tier, str) for tier in tiers)
    assert EvidenceTier is not None

  def test_required_sections_are_lowercase_and_ordered(self):
    assert required_sections == tuple(sorted(required_sections))
    assert schema.rejected_sections == ('why this is rejected',)

  def test_a_shipped_skill_survives_a_round_trip_through_text(self, tmp_path):
    original = discover_skills()[0]
    copy = tmp_path / original.name
    copy.write_text(read_document(original), encoding='utf-8')
    assert load_skill(copy).digest == load_skill(original).digest

  def test_every_shipped_file_uses_single_quote_fences_only(self):
    # The frontmatter reader understands one shape. A file that opens with
    # a different delimiter is a file this parser cannot read, and the
    # failure would surface as a skipped agent rather than as an error.
    for path in discover_skills():
      lines = read_document(path).splitlines()
      assert lines[0].strip() == loader.fence, path.name
      assert lines.count(loader.fence) >= 2, path.name

  def test_shipped_files_declare_a_version_and_are_all_version_one(self):
    versions = {skill.version for skill in load_skills()}
    assert versions == {1}, (
      f'the shipped versions are {versions}; the first release is version 1 '
      'and a bump is a registration event')

  def test_a_string_is_refused_where_a_list_is_required(self):
    # The schema accepts any sequence, and a string is one, so it is
    # refused explicitly before it is iterated one character at a time.
    mapping = frontmatter()
    mapping['data_sources'] = 'nse_eod_bhavcopy'
    with pytest.raises(SkillFormatError, match='must be a list'):
      Skill.from_mapping(mapping, body_sections)
    mapping = frontmatter()
    mapping['kill_criteria'] = 'beat_equal_weight'
    with pytest.raises(SkillFormatError, match='must be a list'):
      Skill.from_mapping(mapping, body_sections)
