#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for git-coordinate discovery and clone bootstrap helpers."""

# White-box tests exercise internal helpers directly.
# pylint: disable=protected-access

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
  'kaggle_cell', Path(__file__).resolve().parents[1] / 'kaggle_cell.py'
)
kc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kc)


def write(path: Path, content: str) -> None:
  """Create parent dirs and write text.

  Args:
      path: Destination file.
      content: Full file content.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(content, encoding='utf-8')


class TestReadRepoMeta:
  """Coordinate discovery across repository layouts."""

  def test_plain_repo_via_files(self, tmp_path):
    write(tmp_path / '.git' / 'HEAD', 'ref: refs/heads/feature/x\n')
    write(
      tmp_path / '.git' / 'config',
      '[remote "origin"]\n\turl = https://github.com/o/r.git\n'
      '\tfetch = +refs/heads/*:refs/remotes/origin/*\n',
    )
    meta = kc.read_repo_meta(tmp_path)
    assert meta == {
      'url': 'https://github.com/o/r.git',
      'branch': 'feature/x',
      'remote': 'origin',
    }

  def test_worktree_pointer_resolves_common_config(self, tmp_path):
    # Mirrors real git: <main>/.git is the common dir; the checkout's .git
    # is a pointer file into .git/worktrees/<name> carrying commondir.
    common = tmp_path / 'main' / '.git'
    write(
      common / 'config',
      '[remote "origin"]\n\turl = https://github.com/o/big.git\n',
    )
    linked = tmp_path / 'main' / '.git' / 'worktrees' / 'knee'
    write(linked / 'HEAD', 'ref: refs/heads/competitions/knee\n')
    write(linked / 'commondir', '../..')
    dot_git_file = tmp_path / 'checkout' / '.git'
    write(dot_git_file, f'gitdir: {linked}\n')
    meta = kc.read_repo_meta(dot_git_file.parent)
    assert meta == {
      'url': 'https://github.com/o/big.git',
      'branch': 'competitions/knee',
      'remote': 'origin',
    }

  def test_resolve_git_dirs_shapes(self, tmp_path):
    plain = tmp_path / 'plain' / '.git'
    plain.mkdir(parents=True)
    linked, common = kc._resolve_git_dirs(plain)
    assert linked == common == plain.resolve() or (
      linked == plain and common == plain
    )
    pointer = tmp_path / 'checkout' / '.git'
    write(pointer, 'gitdir: /nowhere/wt\n')
    assert kc._resolve_git_dirs(pointer)[0] == Path('/nowhere/wt')
    write(pointer, 'not a pointer\n')
    assert kc._resolve_git_dirs(pointer) == (None, None)

  def test_missing_everything_returns_none(self, tmp_path):
    assert kc.read_repo_meta(tmp_path) is None


class TestMetaPersistence:
  """load_or_refresh_meta refresh-on-live / load-from-json semantics."""

  def test_refreshes_json_inside_checkout(self, tmp_path):
    write(tmp_path / '.git' / 'HEAD', 'ref: refs/heads/main\n')
    write(
      tmp_path / '.git' / 'config',
      '[remote "origin"]\n\turl = https://github.com/o/r.git\n',
    )
    meta, in_repo = kc.load_or_refresh_meta(tmp_path)
    assert in_repo and meta['branch'] == 'main'
    stored = json.loads((tmp_path / kc.META_FILENAME).read_text())
    assert stored['url'].endswith('r.git')

  def test_loads_committed_json_outside_checkout(self, tmp_path):
    write(
      tmp_path / kc.META_FILENAME,
      json.dumps(
        {
          'url': 'https://github.com/o/r.git',
          'branch': 'b1',
          'remote': 'origin',
        }
      ),
    )
    meta, in_repo = kc.load_or_refresh_meta(tmp_path)
    assert not in_repo and meta['branch'] == 'b1'


class TestAuthenticatedUrl:
  """Token injection policy."""

  def test_injects_token_into_https(self, monkeypatch):
    monkeypatch.setenv('GIT_TOKEN', 't0k3n')
    url = kc._authenticated_url('https://github.com/o/r.git')
    assert url == 'https://t0k3n@github.com/o/r.git'

  def test_leaves_ssh_and_tokenless_untouched(self, monkeypatch):
    monkeypatch.delenv('GIT_TOKEN', raising=False)
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    monkeypatch.delenv('GH_TOKEN', raising=False)
    ssh = 'git@github.com:o/r.git'
    https = 'https://github.com/o/r.git'
    assert kc._authenticated_url(ssh) == ssh
    assert kc._authenticated_url(https) == https


@pytest.mark.parametrize(
  'line,expected_linked',
  [
    ('gitdir: /nowhere/wt', Path('/nowhere/wt')),
    ('garbage without prefix', None),
  ],
)
def test_resolve_git_dirs_pointer_parsing(line, expected_linked, tmp_path):
  """Pointer files route to the linked dir; junk yields (None, None)."""
  pointer = tmp_path / '.git'
  pointer.write_text(f'{line}\n', encoding='utf-8')
  linked, common = kc._resolve_git_dirs(pointer)
  if expected_linked is None:
    assert linked is None and common is None
  else:
    assert linked == expected_linked
