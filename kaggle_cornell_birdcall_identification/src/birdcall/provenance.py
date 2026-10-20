#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The run manifest: what a run was, and what it was made of.

A result without provenance is an anecdote.  This module writes the
document that turns "a run happened" into "a run can be repeated": the
resolved configuration, the seeds, the configuration file it came from,
the git state, the package versions, and the SHA-256 of every input and
every output.

Two properties are load bearing:

* The **inputs are hashed, not just named**.  A path is not an identity:
  a file can change under a stable name, and a "reproduced" run that
  trained on different bytes is not a reproduction.
* The **git state records the dirty diff**.  A commit alone does not
  identify the code that ran when the working tree was dirty, so the
  diff is hashed too.

The module is **shared**: it imports neither torch nor tensorflow, so
both backends write the same manifest format and a replay does not
depend on the backend that produced it.
'''

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from datetime import timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np

# The packages whose version can change a number.
PACKAGE_NAMES = (
    'torch',
    'torchvision',
    'lightning',
    'tensorflow',
    'keras',
    'numpy',
    'pydantic',
    'soundfile',
    'scikit-learn',
    'pandas',
    'pyyaml',
    'wandb',
)

# The digest of nothing.  It is what a clean git diff hashes to.
EMPTY_DIGEST = hashlib.sha256(b'').hexdigest()

# The file is read in chunks, so a 76 MB recording never has to fit in
# memory.
CHUNK_SIZE = 1 << 20


def sha256_bytes(data: bytes) -> str:
  '''Hashes a byte string.

  Args:
    data: The bytes to hash.

  Returns:
    The lowercase hex digest.
  '''
  return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
  '''Hashes a file.

  The file is read in chunks, so a 76 MB recording does not have to fit
  in memory.

  Args:
    path: The path of the file.

  Returns:
    The lowercase hex digest.

  Raises:
    FileNotFoundError: If the file does not exist.
  '''
  digest = hashlib.sha256()
  with Path(path).open('rb') as handle:
    for chunk in iter(lambda: handle.read(CHUNK_SIZE), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> Optional[str]:
  '''Runs one git command inside a directory.

  Args:
    repo: The directory the command runs in.
    *arguments: The git arguments.

  Returns:
    The command's standard output, or ``None`` when the command failed.
  '''
  try:
    completed = subprocess.run(['git', *arguments],
                               cwd=str(repo),
                               capture_output=True,
                               text=True,
                               check=False)
  except OSError:
    return None
  if completed.returncode != 0:
    return None
  return completed.stdout


def git_state(repo: Path) -> dict:
  '''Describes the git state of a directory.

  The dirty diff is **hashed**, so a non-clean tree is visible instead
  of being assumed clean.  The digest of an empty diff is the digest of
  the empty string, which is what a clean tree produces.

  Args:
    repo: The directory of the repository.

  Returns:
    A mapping with the keys ``commit``, ``branch`` and ``dirty``.  When
    the directory is not a repository, all three are ``None``.
  '''
  repo = Path(repo)
  if _git(repo, 'rev-parse', '--git-dir') is None:
    return {'commit': None, 'branch': None, 'dirty': None}
  commit = _git(repo, 'rev-parse', 'HEAD')
  branch = _git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')
  diff = _git(repo, 'diff', 'HEAD') or ''
  status = _git(repo, 'status', '--porcelain') or ''
  return {
      'commit': commit.strip() if commit else None,
      'branch': branch.strip() if branch else None,
      'dirty': sha256_bytes((diff + status).encode('utf-8')),
  }


def package_versions() -> dict[str, str]:
  '''Returns the versions of the packages that matter.

  A package that is not installed is simply absent from the mapping:
  the manifest records what was there, not what was wished for.

  Returns:
    The package name to version mapping, plus the Python version under
    the key ``python``.
  '''
  versions = {'python': platform.python_version()}
  for name in PACKAGE_NAMES:
    try:
      versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
      continue
  return versions


def _jsonable(value: Any) -> Any:
  '''Converts a value into something JSON can hold.

  Args:
    value: The value to convert.

  Returns:
    The converted value.
  '''
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple, set)):
    return [_jsonable(item) for item in value]
  return value


def _file_entry(path: Path) -> dict[str, Optional[str]]:
  '''Describes one file by its path and its digest.

  A file that does not exist is described with a ``None`` digest and
  ``missing`` set, rather than raising.  The manifest is written at the
  END of a run: losing it because one input was absent would be the
  worst possible failure mode, and the missing file is exactly what
  ``reproduce`` needs to report.

  Args:
    path: The path of the file.

  Returns:
    A mapping with the keys ``path``, ``sha256`` and ``missing``.
  '''
  path = Path(path)
  if not path.exists():
    return {'path': str(path), 'sha256': None, 'missing': True}
  return {'path': str(path), 'sha256': sha256_file(path), 'missing': False}


def _file_entries(paths: Optional[dict[str, Path]]) -> dict:
  '''Describes a group of files by their paths and their digests.

  Args:
    paths: The files, by name.

  Returns:
    A mapping of name to the file's entry.
  '''
  if not paths:
    return {}
  return {name: _file_entry(path) for name, path in paths.items()}


def resolved_config(parameters) -> dict[str, Any]:
  '''Flattens a configuration into a JSON-serialisable mapping.

  Args:
    parameters: The validated configuration.

  Returns:
    The flat configuration, with the nested blocks that matter added
    under their own names.
  '''
  config = dict(parameters.flat_params())
  config['model_config'] = dict(vars(parameters.model_config))
  config['criterion_params'] = dict(parameters.criterion_params)
  config['run_params'] = dict(vars(parameters.run_params))
  config['checkpoint_params'] = dict(vars(parameters.checkpoint_params))
  return _jsonable(config)


def build_manifest(parameters, config_path: Optional[Path] = None,
                   inputs: Optional[dict[str, Path]] = None,
                   outputs: Optional[dict[str, Path]] = None,
                   extra: Optional[dict] = None) -> dict:
  '''Builds the manifest of one run.

  Args:
    parameters: The validated configuration the run used.
    config_path: The configuration file the run was loaded from.
    inputs: The input files by name: the fold CSVs, the audio files, the
      label space and the mel matrix.
    outputs: The output files by name: the score artefacts and the
      submission.
    extra: Extra top level entries.  The tracker record and the command
      the run was launched with go here.

  Returns:
    The manifest document.  Every value in it is JSON-serialisable.
  '''
  manifest: dict[str, Any] = {
      'config': resolved_config(parameters),
      'seeds': {
          'seed': int(parameters.seed),
          'mixup_seed': int(parameters.mixup_seed),
          'deterministic': bool(parameters.deterministic),
      },
      'config_file': (_file_entry(config_path) if config_path else None),
      'git': git_state(Path.cwd()),
      'packages': package_versions(),
      'inputs': _file_entries(inputs),
      'outputs': _file_entries(outputs),
      'tracker': None,
      'command': [],
      'timestamp': datetime.now(timezone.utc).isoformat(),
  }
  if extra:
    manifest.update(_jsonable(extra))
  return manifest


def write_run_manifest(parameters, config_path: Optional[Path] = None,
                       mode: str = 'train',
                       extra: Optional[dict] = None) -> Path:
  '''Builds and writes the manifest of a run beside its checkpoints.

  The inputs are the files the run actually read: the three fold CSVs,
  the label space and the mel matrix.  The audio is not hashed here —
  a training run touches every file in the corpus, and hashing all of
  them belongs to the data pipeline, not to the manifest of one run.

  Args:
    parameters: The validated configuration the run used.
    config_path: The configuration file the run was loaded from.
    mode: The mode the run was in: ``train``, ``infer`` or ``vote``.
    extra: Extra top level entries for the manifest.

  Returns:
    The path the manifest was written to.
  '''
  from birdcall.config_params import configs  # pylint: disable=import-outside-toplevel
  from birdcall.helpers import mel_matrix  # pylint: disable=import-outside-toplevel

  inputs = {
      'train_csv': parameters.train_ds_params.csv_dir,
      'valid_csv': parameters.valid_ds_params.csv_dir,
      'test_csv': parameters.test_ds_params.csv_dir,
      'label_space': configs.LABEL_SPACE_PATH,
      'mel_matrix': mel_matrix.mel_matrix_path(
          parameters.model_config.sample_rate,
          parameters.model_config.window_size,
          parameters.model_config.mel_bins,
          parameters.model_config.fmin,
          parameters.model_config.fmax),
  }
  payload = {'mode': mode}
  if extra:
    payload.update(extra)
  manifest = build_manifest(parameters, config_path=config_path,
                            inputs=inputs, extra=payload)
  path = Path(parameters.checkpoint_params.save_dir) / 'manifest.json'
  write_manifest(manifest, path)
  return path


def write_manifest(manifest: dict, path: Path) -> Path:
  '''Writes a manifest as indented JSON.

  Args:
    manifest: The manifest document.
    path: The path of the file.

  Returns:
    The path the manifest was written to.
  '''
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                  encoding='utf-8')
  return path


__all__ = [
    'EMPTY_DIGEST',
    'PACKAGE_NAMES',
    'build_manifest',
    'git_state',
    'package_versions',
    'resolved_config',
    'sha256_bytes',
    'sha256_file',
    'write_manifest',
]
