#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''File discovery and frontmatter parsing for markdown skill files.

This module is the reason 'adding an agent' is a file copy. It walks a
directory, reads every ``.md`` file, splits the ``---`` fence, and parses
a **fixed key set** of frontmatter with a hand-rolled reader.

No YAML library, deliberately, for three reasons. The project carries
zero runtime dependencies and a parser for eight keys and two list shapes
is about sixty lines. The supported grammar is a strict subset of YAML,
so adopting the format costs nothing later if the key set ever needs to
grow. And a general parser would accept nesting, anchors, multi-line
scalars and tags that nothing here validates, which is how a frontmatter
block acquires five ways to mean the same thing.

The grammar, in full:

  * A blank line, or a line whose first non-space character is ``#``, is
    a comment and is skipped.
  * ``key: value`` at column 0 is a scalar entry. Bare ``true``, ``false``,
    ``null``, integers and decimals are converted; everything else,
    including quoted strings, is text.
  * ``key:`` with nothing after the colon opens a **list**, whose items are
    ``- item`` lines indented by exactly two spaces.
  * A list item may itself be ``- key: value``, which makes it a mapping.
    Further ``key: value`` lines indented by exactly four spaces fill in
    that mapping. This is how ``kill_criteria`` carries an id, a statement
    and a threshold. A sub-key under an empty list is an error rather
    than an ``IndexError``, and a ``- scheme://host`` item is **not** a
    mapping: the scheme reads as a bare identifier, so a bare URL would
    otherwise be silently reshaped into ``{'https': '//host'}``.
  * Anything else -- an unclosed fence, a tab, the wrong indent, a
    duplicate key, a list item with no key -- raises
    :class:`~stock_rl.skills.schema.SkillFormatError`.

The strictness is the feature. Discovery is equally strict: a file in the
skills directory that is not a ``.md`` file is an error rather than
skipped, because the only reason to put a file there is to have it loaded.
'''

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from stock_rl.skills.schema import (
  Skill,
  SkillFormatError,
)

__all__ = [
  'discover_skills',
  'load_skill',
  'load_skills',
  'parse_frontmatter',
  'read_document',
  'skills_directory',
  'split_document',
]

#: The fence that opens and closes frontmatter. Also the reason
#: ``split_document`` can be written without a state machine.
fence = '---'

#: Directory holding the skill files shipped with the package.
skills_subdirectory = 'skills'

#: Extension every discoverable skill file must carry.
skill_suffix = '.md'

#: Exact indentation allowed for a list item, and for a sub-key inside
#: one. Fixed rather than 'any deeper indent' so a stray space produces
#: an error instead of a differently shaped value.
list_indent = 2
subkey_indent = 4

#: Marker that opens a list item, dash and one space. A named constant
#: because three places depend on it and one of them quotes it in an
#: error message, which is where an inline literal would need its own
#: quoting and stop being readable.
item_marker = '- '

#: Separator between a frontmatter key and its value.
key_separator = ':'

#: A ``- key: value`` list item becomes a mapping only when the part
#: before the colon is a bare identifier AND what follows the colon is not
#: a URL scheme separator. Without the second test a data source written
#: as a bare URL becomes ``{'https': '//example.invalid/x'}``, which is a
#: mapping one function deeper with no error anywhere: exactly the
#: silently reshaped value this module exists to refuse.
item_key_pattern = re.compile(
  r'^(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?!//)(?P<value>.*)$')

#: Prefixes whose file names are skipped during discovery: editor
#: droppings and Python bytecode directories.
ignored_prefixes = ('.', '_')


def skills_directory() -> Path:
  '''Return the packaged directory holding the shipped skill files.

  A function rather than a module constant so that a relocated
  installation -- an editable checkout, a zipped egg, a wheel installed
  somewhere unexpected -- still finds its own files instead of trusting
  a path baked in at import time.

  Returns:
    Absolute path of the skills directory.
  '''
  return Path(__file__).resolve().parent / skills_subdirectory


def discover_skills(directory: Path | str | None = None) -> tuple[Path, ...]:
  '''Return every skill file in a directory, in stable name order.

  Order is by file name so that a fingerprint manifest and any log line
  listing loaded skills are reproducible run to run.

  Dotfiles and underscore-prefixed names are skipped as tooling
  droppings. Everything else must be a ``.md`` file: a stray ``.txt`` or
  a subdirectory is refused, because the sole reason to place a file in
  this directory is to have it read as a skill, and quietly ignoring one
  is how a renamed agent disappears.

  Args:
    directory: Directory to scan, or None for the packaged directory.

  Returns:
    Skill file paths, sorted by name.

  Raises:
    FileNotFoundError: If the directory does not exist or is not a
      directory.
    SkillFormatError: If the directory holds a file that is not a
      markdown skill file.
  '''
  root = Path(directory) if directory is not None else skills_directory()
  if not root.is_dir():
    raise FileNotFoundError(f'no skill directory at {root}')
  found: list[Path] = []
  for entry in sorted(root.iterdir(), key=lambda item: item.name):
    if entry.name.startswith(ignored_prefixes):
      continue
    if entry.is_dir():
      raise SkillFormatError(
        f'{entry.name}: directories are not skills; every file in the '
        'skills directory is loaded, so nesting one here would hide it')
    if entry.suffix != skill_suffix:
      raise SkillFormatError(
        f'{entry.name}: expected a {skill_suffix} skill file; every file '
        'in the skills directory is loaded, so an unrecognised extension '
        'is a mistake rather than a note')
    found.append(entry)
  return tuple(found)


def read_document(path: Path | str) -> str:
  '''Read one skill file as text.

  Args:
    path: Path to a markdown skill file.

  Returns:
    The full file contents.

  Raises:
    OSError: If the file cannot be read.
  '''
  return Path(path).read_text(encoding='utf-8')


def split_document(text: str) -> tuple[str, str]:
  '''Split raw file text into its frontmatter and its body.

  The opening fence must be the very first line. Tolerating a blank line
  or a title before it would mean two conventions per file, and the second
  one is always the one somebody forgets.

  Args:
    text: Full contents of a skill file.

  Returns:
    Tuple of (frontmatter text without its fences, markdown body).

  Raises:
    SkillFormatError: If the document does not open with a fence, or the
      frontmatter is never closed.
  '''
  lines = text.splitlines()
  if not lines or lines[0].strip() != fence:
    raise SkillFormatError(
      f'file must open with a {fence!r} frontmatter fence')
  for index in range(1, len(lines)):
    if lines[index].strip() == fence:
      return '\n'.join(lines[1:index]), '\n'.join(lines[index + 1:])
  raise SkillFormatError('frontmatter is never closed by a fence')


def _scalar(text: str) -> str | bool | int | float | None:
  '''Convert one frontmatter scalar to a Python value.

  Only an unambiguous plain number is converted. ``int`` and ``float``
  both accept Python spellings such as ``1_000``, ``inf`` and ``nan``, so
  the digits are checked first and anything else stays text. A threshold
  written as ``1.0`` then reads as a string rather than as a float the
  caller has to remember to normalise.

  Args:
    text: Scalar text with surrounding whitespace already stripped.

  Returns:
    A bool, int, float, None or str.
  '''
  if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
    return text[1:-1]
  lowered = text.casefold()
  if lowered in ('true', 'false'):
    return lowered == 'true'
  if lowered in ('null', 'none', '~'):
    return None
  digits = text[1:] if text[:1] in ('+', '-') else text
  if digits.isascii() and digits.replace('.', '', 1).isdigit():
    return float(text) if '.' in text else int(text)
  return text


def _entry_key(content: str, number: int) -> str:
  '''Return the key of a ``key: value`` frontmatter line.

  Args:
    content: Line content with indentation stripped.
    number: Line number in the file, for error messages.

  Returns:
    The stripped key.

  Raises:
    SkillFormatError: If the line carries no colon separator, or the key
      is empty.
  '''
  key, separator, _ = content.partition(key_separator)
  if not separator:
    raise SkillFormatError(
      f'frontmatter line {number}: expected key: value, '
      f'got {content!r}')
  stripped = key.strip()
  if not stripped:
    raise SkillFormatError(
      f'frontmatter line {number}: key is empty')
  return stripped


def _scalar_value(content: str, number: int) -> str:
  '''Return the value half of a ``key: value`` frontmatter line.

  Args:
    content: Line content with indentation stripped.
    number: Line number in the file, for error messages.

  Returns:
    The stripped value text.

  Raises:
    SkillFormatError: If the line carries no colon separator.
  '''
  _, separator, value = content.partition(key_separator)
  if not separator:
    raise SkillFormatError(
      f'frontmatter line {number}: expected key: value, '
      f'got {content!r}')
  return value.strip()


def _append_item(bucket: list[object], text: str) -> None:
  '''Append one list item, as a mapping when it declares sub-keys.

  The mapping or scalar decision lives here and nowhere else, so the
  shape of every list in every skill file is decided by one test.

  Args:
    bucket: List the item is appended to, mutated in place.
    text: Item text with the ``- `` marker removed.
  '''
  match = item_key_pattern.match(text)
  if match is None:
    bucket.append(_scalar(text))
    return
  bucket.append({match.group('key'): _scalar(match.group('value').strip())})


def _list_bucket(fields: dict[str, object], key: str | None,
                 number: int) -> list[object]:
  '''Return the list a ``key:`` line has opened, creating it if needed.

  Args:
    fields: Frontmatter mapping under construction.
    key: Key whose value is a list, or None when the line is orphaned.
    number: Line number in the file, for error messages.

  Returns:
    The mutable list stored under ``key``.

  Raises:
    SkillFormatError: If ``key`` is None, i.e. the list line has no key
      to attach to.
  '''
  if key is None:
    raise SkillFormatError(
      f'frontmatter line {number}: list item has no key to attach to')
  value = fields.get(key)
  if value is None:
    value = []
    fields[key] = value
  if not isinstance(value, list):
    raise SkillFormatError(
      f'frontmatter line {number}: {key} already has a scalar value, so '
      'it cannot also take a list')
  return value


def parse_frontmatter(frontmatter: str) -> dict[str, object]:
  '''Parse a frontmatter block into a mapping.

  Implements exactly the grammar in this module's docstring. Returns a
  plain dict rather than a typed object because the key set is validated
  in :meth:`stock_rl.skills.schema.Skill.from_mapping`, and validating in
  one place is what keeps 'what does the format allow' and 'what does the
  schema require' from drifting apart.

  Args:
    frontmatter: Frontmatter text, without its ``---`` fences.

  Returns:
    Mapping of key to scalar, list of scalars, or list of mappings.

  Raises:
    SkillFormatError: On a tab, an unexpected indent, a missing colon, a
      duplicate key, a list item with no key, or a sub-key line that
      does not follow a mapping item.
  '''
  fields: dict[str, object] = {}
  pending: str | None = None
  for number, raw in enumerate(frontmatter.splitlines(), start=2):
    line = raw.rstrip()
    if not line.strip() or line.lstrip().startswith('#'):
      continue
    if '\t' in line:
      raise SkillFormatError(
        f'frontmatter line {number}: tab indentation is not supported; '
        'use spaces so the shape of the file is unambiguous')
    indent = len(line) - len(line.lstrip(' '))
    content = line.strip()
    if indent == 0:
      if content.startswith(item_marker):
        raise SkillFormatError(
          f'frontmatter line {number}: list item with no key above it')
      key = _entry_key(content, number)
      if key in fields:
        raise SkillFormatError(
          f'frontmatter line {number}: duplicate key {key!r}')
      value = _scalar_value(content, number)
      if value:
        fields[key] = _scalar(value)
        pending = None
      else:
        fields[key] = None
        pending = key
      continue
    if indent not in (list_indent, subkey_indent):
      raise SkillFormatError(
        f'frontmatter line {number}: indentation must be 0, '
        f'{list_indent} or {subkey_indent} spaces, got {indent}')
    bucket = _list_bucket(fields, pending, number)
    if indent == list_indent:
      if not content.startswith(item_marker):
        raise SkillFormatError(
          f'frontmatter line {number}: a list value must use '
          f'{item_marker!r} items, got {content!r}')
      _append_item(bucket, content[len(item_marker):].strip())
      continue
    if not bucket:
      raise SkillFormatError(
        f'frontmatter line {number}: {content!r} has no parent item; a '
        'sub-key must follow a key: value item, and the list above it is '
        'empty')
    item = bucket[-1]
    if not isinstance(item, dict):
      lead = content.partition(key_separator)[0].strip()
      raise SkillFormatError(
        f'frontmatter line {number}: {lead!r} has no parent item; a '
        'sub-key must follow a key: value item')
    subkey = _entry_key(content, number)
    if subkey in item:
      raise SkillFormatError(
        f'frontmatter line {number}: duplicate key {subkey!r} in list item')
    item[subkey] = _scalar(_scalar_value(content, number))
  return fields


def load_skill(path: Path | str) -> Skill:
  '''Read, parse and validate one skill file.

  Args:
    path: Path to a markdown skill file.

  Returns:
    The validated, frozen skill.

  Raises:
    SkillFormatError: If the fence is missing, the frontmatter is
      malformed, or the result violates the schema.
    OSError: If the file cannot be read.
  '''
  target = Path(path)
  frontmatter, body = split_document(read_document(target))
  fields = parse_frontmatter(frontmatter)
  return Skill.from_mapping(fields, body, source=target.name)


def load_skills(directory: Path | str | None = None) -> tuple[Skill, ...]:
  '''Load every skill file in a directory, rejecting the lot on any fault.

  The aggregation is the important part. Loading the good files and
  reporting the bad ones later would leave the system running a subset
  it was never tested with, which is how a rejected tier quietly becomes
  an enabled one. Every failure is collected and raised as one error, so
  one fix makes the directory load.

  Args:
    directory: Directory to scan, or None for the packaged directory.

  Returns:
    The validated skills, ordered by file name.

  Raises:
    SkillFormatError: If any file fails to parse or validate. The message
      names every rejected file and its reason.
    FileNotFoundError: If the directory does not exist.
  '''
  paths = discover_skills(directory)
  skills: list[Skill] = []
  problems: list[str] = []
  for path in paths:
    try:
      skills.append(load_skill(path))
    except SkillFormatError as error:
      problems.append(f'  {path.name}: {error}')
    except OSError as error:
      problems.append(f'  {path.name}: unreadable ({error})')
  if problems:
    raise SkillFormatError(
      f'{len(problems)} of {len(paths)} skill files were rejected. A '
      'skill file that cannot be loaded is a bug to fix, never a file to '
      'skip, because a skipped skill fails silently in production:\n'
      + '\n'.join(problems))
  return tuple(skills)


def describe_sources(skills: Iterable[Skill]) -> tuple[str, ...]:
  '''Return the distinct data sources declared across skills.

  One helper, because the answer is a governance question -- which
  exchanges, endpoints and vendors does the running system depend on --
  and the answer should not require walking the registry by hand.

  Args:
    skills: Skills to inspect.

  Returns:
    Source names, de-duplicated, sorted.
  '''
  names = {source for skill in skills for source in skill.data_sources}
  return tuple(sorted(names))
