#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The vote entry point: read the ensemble manifest, vote, write the CSV.

The manifest names the thirteen models of the ensemble, the experiment
configuration each of them was trained with, and the directory their
score artefacts live in.  Every artefact is the output of
:func:`birdcall.metrics.scoring.save_score_artefact` for **one** model
and **one** audio file, so the vote runs over the members of the
ensemble for one file at a time.

The command is::

    birdcall-vote --ensemble ensemble_13.yaml --scores scores/ \\
        --output submission.csv

The thresholds and the quorum come from the manifest and can be
overridden on the command line, which is how the other rows of the
original's leaderboard table are reproduced.
'''

import argparse
import logging
from pathlib import Path

from birdcall.config_params.loader import read_yaml
from birdcall.voting.vote import DEFAULT_CLIP_THRESHOLD
from birdcall.voting.vote import DEFAULT_FRAME_THRESHOLD
from birdcall.voting.vote import DEFAULT_VOTES
from birdcall.voting.vote import FRAMES_PER_ROW
from birdcall.voting.vote import FRAME_RATE
from birdcall.voting.vote import vote_file
from birdcall.voting.vote import write_submission

PACKAGE_ROOT = Path(__file__).resolve().parent
ENSEMBLE_DIR = PACKAGE_ROOT / 'config_params' / 'ensembles'
DEFAULT_ENSEMBLE = ENSEMBLE_DIR / 'ensemble_13.yaml'

LOGGER = logging.getLogger(__name__)


def read_ensemble(path: Path) -> dict:
  '''Reads an ensemble manifest.

  Args:
    path: The path of the manifest.

  Returns:
    The manifest document.

  Raises:
    FileNotFoundError: If the manifest does not exist.
    ValueError: If the manifest is not a mapping, or lists no member.
  '''
  path = Path(path)
  document = read_yaml(path)
  members = document.get('members')
  if not isinstance(members, list) or not members:
    raise ValueError(f'Ensemble {path} lists no members.')
  return document


def resolve_experiment_path(value: str) -> Path:
  '''Resolves the path of an experiment configuration.

  A relative path is looked for inside the package first, so a manifest
  that ships with the package names its experiments the same way the
  package does, and is found wherever the package is installed.

  Args:
    value: The path as the manifest spells it.

  Returns:
    The resolved path.
  '''
  path = Path(value)
  if path.is_absolute():
    return path
  packaged = PACKAGE_ROOT / path
  if packaged.exists():
    return packaged
  return path


def member_score_directory(member: dict, scores_root: Path) -> Path:
  '''Returns the directory holding one member's score artefacts.

  Args:
    member: The member of the ensemble.
    scores_root: The root of the score directories.

  Returns:
    The directory of the member.

  Raises:
    ValueError: If the member names no score directory.
  '''
  value = member.get('scores')
  if not value:
    raise ValueError(f'The ensemble member {member.get("name")!r} names no '
                     'score directory.')
  path = Path(value)
  if path.is_absolute():
    return path
  return Path(scores_root) / path


def discover_audio_ids(members: list[dict], scores_root: Path) -> list[str]:
  '''Finds the audio files every member of the ensemble scored.

  Args:
    members: The members of the ensemble.
    scores_root: The root of the score directories.

  Returns:
    The identifiers of the audio files, sorted.

  Raises:
    FileNotFoundError: If a member's score directory does not exist.
  '''
  common: set[str] | None = None
  for member in members:
    directory = member_score_directory(member, scores_root)
    if not directory.is_dir():
      raise FileNotFoundError(
          f'The score directory {directory} of the ensemble member '
          f'{member.get("name")!r} does not exist.')
    identifiers = {path.stem for path in directory.glob('*.npz')}
    common = (identifiers if common is None
              else common & identifiers)
  return sorted(common or set())


def audio_plan(manifest: dict, members: list[dict],
               scores_root: Path) -> list[tuple[str, int]]:
  '''Decides which audio files are voted over, and at which site.

  The manifest pins the site of a file when it lists one.  Otherwise the
  audio files are discovered from the score artefacts and every one of
  them is treated as a site 1 file, which is the common case.

  Args:
    manifest: The manifest document.
    members: The members of the ensemble.
    scores_root: The root of the score directories.

  Returns:
    One ``(audio id, site)`` pair per audio file.

  Raises:
    ValueError: If the manifest lists an audio file without an id.
  '''
  default_site = int(manifest.get('default_site', 1))
  listed = manifest.get('audio') or []
  if listed:
    plan = []
    for entry in listed:
      audio_id = entry.get('audio_id')
      if not audio_id:
        raise ValueError(f'The ensemble lists an audio file without an id: '
                         f'{entry!r}.')
      plan.append((str(audio_id), int(entry.get('site', default_site))))
    return plan
  return [(audio_id, default_site)
          for audio_id in discover_audio_ids(members, scores_root)]


def score_paths(members: list[dict], scores_root: Path,
                audio_id: str) -> list[Path]:
  '''Builds the score artefact path of every member for one file.

  Args:
    members: The members of the ensemble.
    scores_root: The root of the score directories.
    audio_id: The identifier of the audio file.

  Returns:
    One path per member.

  Raises:
    FileNotFoundError: If a member did not score this audio file.
  '''
  paths = []
  for member in members:
    path = member_score_directory(member, scores_root) / f'{audio_id}.npz'
    if not path.is_file():
      raise FileNotFoundError(
          f'The ensemble member {member.get("name")!r} has no score '
          f'artefact for {audio_id!r} at {path}.')
    paths.append(path)
  return paths


def vote_ensemble(ensemble_path: Path, scores_root: Path, output: Path,
                  votes: int | None = None,
                  clip_threshold: float | None = None,
                  frame_threshold: float | None = None) -> Path:
  '''Runs the vote of an ensemble and writes the submission.

  Args:
    ensemble_path: The path of the ensemble manifest.
    scores_root: The root of the score directories.
    output: The path of the submission CSV.
    votes: The quorum, overriding the manifest.
    clip_threshold: The gate on the clip score, overriding the manifest.
    frame_threshold: The gate on the frame score, overriding the manifest.

  Returns:
    The path of the submission CSV.

  Raises:
    FileNotFoundError: If the manifest or a score artefact is missing.
    ValueError: If the manifest is inconsistent.
  '''
  manifest = read_ensemble(Path(ensemble_path))
  members = manifest['members']
  frame_rate = float(manifest.get('frame_rate', FRAME_RATE))
  if frame_rate != FRAME_RATE:
    raise ValueError(
        f'The ensemble {ensemble_path} declares a frame rate of '
        f'{frame_rate}, but the row id format is tied to {FRAME_RATE} '
        'frames per second.')
  votes_required = (DEFAULT_VOTES if votes is None
                    else int(votes))
  clip = (DEFAULT_CLIP_THRESHOLD if clip_threshold is None
          else float(clip_threshold))
  frame = (DEFAULT_FRAME_THRESHOLD if frame_threshold is None
           else float(frame_threshold))
  rows = []
  for audio_id, site in audio_plan(manifest, members, Path(scores_root)):
    rows.extend(vote_file(audio_id,
                          score_paths(members, Path(scores_root), audio_id),
                          site=site,
                          votes_required=votes_required,
                          clip_threshold=clip,
                          frame_threshold=frame,
                          frames_per_row=FRAMES_PER_ROW))
  target = write_submission(rows, Path(output))
  LOGGER.info('Wrote %d rows to %s', len(rows), target)
  return target


def build_parser() -> argparse.ArgumentParser:
  '''Builds the command line parser.

  Returns:
    The parser.
  '''
  parser = argparse.ArgumentParser(
      description='Vote an ensemble of models and write a submission CSV.')
  parser.add_argument('--ensemble', type=Path, default=DEFAULT_ENSEMBLE,
                      help='The ensemble manifest. Defaults to the packaged '
                           f'ensemble: {DEFAULT_ENSEMBLE}.')
  parser.add_argument('--scores', type=Path, required=True,
                      help='The directory holding the per model score '
                           'artefacts.')
  parser.add_argument('--output', type=Path, default=Path('submission.csv'),
                      help='The path of the submission CSV.')
  parser.add_argument('--votes', type=int, default=None,
                      help='The number of models that have to agree. '
                           'Overrides the manifest.')
  parser.add_argument('--clip-threshold', type=float, default=None,
                      help='The gate on the clip score. Overrides the '
                           'manifest.')
  parser.add_argument('--frame-threshold', type=float, default=None,
                      help='The gate on the frame score. Overrides the '
                           'manifest.')
  return parser


def main(argv: list[str] | None = None) -> int:
  '''Runs the vote.

  Args:
    argv: The command line arguments, or ``None`` for the real ones.

  Returns:
    The exit code.
  '''
  logging.basicConfig(level=logging.INFO, format='%(message)s')
  args = build_parser().parse_args(argv)
  vote_ensemble(args.ensemble, args.scores, args.output,
                votes=args.votes,
                clip_threshold=args.clip_threshold,
                frame_threshold=args.frame_threshold)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
