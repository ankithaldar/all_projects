#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The 13-model vote and the submission writer.

The winning submission was a vote over thirteen models: the four-fold
no-mixup family, the five-fold no-mixup family and the four-fold mixup
family.  The rule is the original author's, from the original README
(line 45): a species is accepted for a row when at least **four** of
the thirteen models predicted it, with a threshold of ``0.3`` applied to
the clip head and to the frame head.

The two thresholds are **independent**, and that is the in-repo
ancestor of the rule.  The original's frame metric wrote::

    score = f1_score([t], [(pf >= self.threshold) & (pc >= 0.3)], ...)

with the ``0.3`` on the clip side a hard-coded literal.  A model that
clears one gate and misses the other does not vote at all.

The frame axis survives all the way to the row.  A row is five seconds,
which is 500 frames at 100 frames per second, and the last, partial row
of a file is dropped: a row shorter than five seconds is not a row.
This is the opposite of the original's inference metric, which
max-pooled the whole time axis away and therefore could only ever have
produced one row per file.

The module is **shared**: it imports neither torch nor tensorflow, so
the PyTorch Lightning port and the Keras port vote on the same score
artefacts and produce the same submission.
'''

import csv
from pathlib import Path

import numpy as np

from birdcall.config_params.configs import get_inv_bird_code
from birdcall.metrics.scoring import load_score_artefact

DEFAULT_VOTES = 4
DEFAULT_CLIP_THRESHOLD = 0.3
DEFAULT_FRAME_THRESHOLD = 0.3
FRAMES_PER_ROW = 500
DEFAULT_MODEL_COUNT = 13

# The frame period of the score artefacts, in frames per second.  The row
# id carries the second a time window ends at, so this constant is part
# of the row id format.
FRAME_RATE = 100.0

# The label a row with no accepted species carries.
NOCALL = 'nocall'

# The recording sites of the test set.
VALID_SITES = (1, 2, 3)


def model_fires(clip: np.ndarray, frames: np.ndarray,
                clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
                frame_threshold: float = DEFAULT_FRAME_THRESHOLD) -> np.ndarray:
  '''Says at which frames each species fires for one model.

  A species fires at a frame when the frame score is at least
  ``frame_threshold`` **and** the clip score is at least
  ``clip_threshold``.  The two gates are independent, exactly as the
  original's frame metric had them.

  Args:
    clip: The clip level scores, shape ``(num_classes,)``.
    frames: The frame level scores, shape ``(num_frames, num_classes)``.
      The time axis must be preserved.
    clip_threshold: The gate on the clip score.
    frame_threshold: The gate on the frame score.

  Returns:
    The boolean mask of the frames at which each species fires, shape
    ``(num_frames, num_classes)``.

  Raises:
    ValueError: If the shapes do not agree, or if a threshold is not a
      finite number.
  '''
  clip = np.asarray(clip, dtype=np.float32)
  frames = np.asarray(frames, dtype=np.float32)
  if clip.ndim != 1:
    raise ValueError(f'The clip scores must be one dimensional, got shape '
                     f'{clip.shape}.')
  if frames.ndim != 2:
    raise ValueError(f'The frame scores must be two dimensional, got shape '
                     f'{frames.shape}.')
  if frames.shape[1] != clip.shape[0]:
    raise ValueError(f'The frame scores have {frames.shape[1]} classes and '
                     f'the clip scores have {clip.shape[0]}.')
  if not np.isfinite(clip_threshold) or not np.isfinite(frame_threshold):
    raise ValueError('The thresholds must be finite numbers.')
  gate = clip >= clip_threshold
  return (frames >= frame_threshold) & gate[None, :]


def slice_rows(frames: np.ndarray,
               frames_per_row: int = FRAMES_PER_ROW) -> np.ndarray:
  '''Splits a frame sequence into five second rows.

  The last, partial row is dropped, because a row shorter than five
  seconds is not a row.

  Args:
    frames: The frame scores, shape ``(num_frames, num_classes)``.
    frames_per_row: The number of frames in one row.

  Returns:
    The rows, shape ``(num_rows, frames_per_row, num_classes)``.

  Raises:
    ValueError: If the row length is not positive.
  '''
  frames = np.asarray(frames)
  if frames_per_row <= 0:
    raise ValueError(f'frames_per_row must be positive, got {frames_per_row}.')
  num_rows = frames.shape[0] // frames_per_row
  trimmed = frames[:num_rows * frames_per_row]
  return trimmed.reshape((num_rows, frames_per_row) + frames.shape[1:])


def _model_masks(score_paths: list[Path], clip_threshold: float,
                 frame_threshold: float) -> list[np.ndarray]:
  '''Loads the fire mask of every model.

  Args:
    score_paths: The score artefact of every model, all of them for the
      same audio file.
    clip_threshold: The gate on the clip score.
    frame_threshold: The gate on the frame score.

  Returns:
    One boolean mask per model, shape ``(num_frames, num_classes)``.

  Raises:
    ValueError: If no artefact was given.
  '''
  masks = []
  for path in score_paths:
    data = load_score_artefact(Path(path))
    masks.append(model_fires(data['clip'], data['frames'], clip_threshold,
                             frame_threshold))
  if not masks:
    raise ValueError('The vote needs at least one score artefact, none '
                     'was given.')
  return masks


def _row_votes(masks: list[np.ndarray], frames_per_row: int) -> np.ndarray:
  '''Counts the votes of every model, row by row.

  A model votes for a species in a row when it fires for that species
  at **any** frame of the row.

  Args:
    masks: The fire mask of every model.
    frames_per_row: The number of frames in one row.

  Returns:
    The vote count, shape ``(num_rows, num_classes)``, integer.

  Raises:
    ValueError: If the models disagree about the number of classes or
      about the number of rows.
  '''
  rows = [slice_rows(mask, frames_per_row) for mask in masks]
  shape = rows[0].shape
  for other in rows[1:]:
    if other.shape[2] != shape[2]:
      raise ValueError(f'The models disagree about the number of classes: '
                       f'{shape[2]} and {other.shape[2]}.')
    if other.shape[0] != shape[0]:
      raise ValueError(f'The models disagree about the number of five '
                       f'second rows: {shape[0]} and {other.shape[0]}.')
  votes = np.zeros((shape[0], shape[2]), dtype=np.int64)
  for row in rows:
    votes += row.any(axis=1)
  return votes


def _file_votes(masks: list[np.ndarray]) -> np.ndarray:
  '''Counts the votes of every model over a whole file.

  Site 3 is labelled per file, so a model votes for a species when it
  fires for that species at any frame of the file.

  Args:
    masks: The fire mask of every model.

  Returns:
    The vote count, shape ``(1, num_classes)``, integer.

  Raises:
    ValueError: If the models disagree about the number of classes.
  '''
  num_classes = masks[0].shape[1]
  for mask in masks[1:]:
    if mask.shape[1] != num_classes:
      raise ValueError(f'The models disagree about the number of classes: '
                       f'{num_classes} and {mask.shape[1]}.')
  votes = np.zeros((1, num_classes), dtype=np.int64)
  for mask in masks:
    votes += mask.any(axis=0)
  return votes


def count_votes(score_paths: list[Path],
                frames_per_row: int = FRAMES_PER_ROW,
                clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
                frame_threshold: float = DEFAULT_FRAME_THRESHOLD) -> np.ndarray:
  '''Counts the votes of every model, row by row.

  Every path is one model's score artefact for the **same** audio file,
  written by :func:`birdcall.metrics.scoring.save_score_artefact`.

  Args:
    score_paths: The score artefact of every model.
    frames_per_row: The number of frames in one row.
    clip_threshold: The gate on the clip score.
    frame_threshold: The gate on the frame score.

  Returns:
    The vote count, shape ``(num_rows, num_classes)``, integer.

  Raises:
    ValueError: If no artefact was given, or if the models disagree
      about the number of classes or the number of rows.
  '''
  masks = _model_masks(score_paths, clip_threshold, frame_threshold)
  return _row_votes(masks, frames_per_row)


def accepted_species(votes: np.ndarray,
                     votes_required: int = DEFAULT_VOTES) -> np.ndarray:
  '''Says which species reached the quorum.

  Args:
    votes: The vote count, shape ``(num_rows, num_classes)``.
    votes_required: The number of models that have to agree.

  Returns:
    The boolean mask of the species that reached the quorum, shape
    ``(num_rows, num_classes)``.

  Raises:
    ValueError: If the quorum is not a positive number of votes.
  '''
  votes = np.asarray(votes)
  if votes_required < 1:
    raise ValueError(f'votes_required must be at least 1, got '
                     f'{votes_required}.')
  return votes >= votes_required


def build_row_id(site: int, audio_id: str, row_index: int,
                 frames_per_row: int = FRAMES_PER_ROW) -> str:
  '''Builds the competition row id.

  The row id encodes the site, the audio file and the second the time
  window **ends** at::

      site_1_0a997dff022e3ad9744d4e7bbf923288_5
      site_3_0a997dff022e3ad9744d4e7bbf923288

  Sites 1 and 2 are labelled in five second increments, so every window
  has an end second.  Site 3 is labelled per file, so its single row
  carries no seconds at all and every row index maps to the same id.

  The end second is derived from the row index and the frame period,
  which is why ``frames_per_row`` is a parameter: the module constant
  :data:`FRAME_RATE` fixes the length of a frame.

  The exact string is external knowledge -- the original repository has
  no ``row_id`` anywhere -- so it lives in this one function.  Change it
  here and every emitted row id changes with it.

  Args:
    site: The recording site, 1, 2 or 3.
    audio_id: The identifier of the audio file.
    row_index: The zero based index of the time window.
    frames_per_row: The number of frames in one row.

  Returns:
    The row id.

  Raises:
    ValueError: If the site is not one of 1, 2 or 3, if the row index is
      negative, if the row length is not positive, or if a row is not a
      whole number of seconds long.
  '''
  if site not in VALID_SITES:
    raise ValueError(f'Unknown site {site!r}, expected one of '
                     f'{VALID_SITES}.')
  if row_index < 0:
    raise ValueError(f'The row index must not be negative, got {row_index}.')
  if frames_per_row <= 0:
    raise ValueError(f'frames_per_row must be positive, got {frames_per_row}.')
  if site == 3:
    return f'site_{site}_{audio_id}'
  seconds = (row_index + 1) * frames_per_row / FRAME_RATE
  if not float(seconds).is_integer():
    raise ValueError(f'A row of {frames_per_row} frames is {seconds} '
                     'seconds long, which is not a whole number of '
                     'seconds.')
  return f'site_{site}_{audio_id}_{int(seconds)}'


def vote_file(
    audio_id: str,
    score_paths: list[Path],
    site: int = 1,
    votes_required: int = DEFAULT_VOTES,
    clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    frames_per_row: int = FRAMES_PER_ROW,
) -> list[tuple[str, list[str]]]:
  '''Runs the vote for one audio file and returns the rows.

  Sites 1 and 2 are labelled in five second increments, so they get one
  row per time window.  Site 3 is only labelled at the file level, so it
  gets **one** row for the whole file, holding every species that
  reached the quorum somewhere in that file.

  Args:
    audio_id: The identifier of the audio file.
    score_paths: The score artefact of every model, all of them for this
      audio file.
    site: The recording site, 1, 2 or 3.
    votes_required: The number of models that have to agree.
    clip_threshold: The gate on the clip score.
    frame_threshold: The gate on the frame score.
    frames_per_row: The number of frames in one row.

  Returns:
    One ``(row_id, species codes)`` pair per row.

  Raises:
    ValueError: If the site is unknown, if no artefact was given, or if
      the models disagree.
  '''
  if site not in VALID_SITES:
    raise ValueError(f'Unknown site {site!r}, expected one of '
                     f'{VALID_SITES}.')
  codes = get_inv_bird_code()
  masks = _model_masks(score_paths, clip_threshold, frame_threshold)
  if site == 3:
    votes = _file_votes(masks)
  else:
    votes = _row_votes(masks, frames_per_row)
  accepted = accepted_species(votes, votes_required)
  rows = []
  for index in range(accepted.shape[0]):
    species = [codes[position]
               for position in np.flatnonzero(accepted[index])]
    rows.append((build_row_id(site, audio_id, index, frames_per_row),
                 species))
  return rows


def _species_codes(species: list[str], codes: dict[int, str]) -> list[str]:
  '''Turns the species of a row into their codes.

  Args:
    species: The species of the row, as codes or as class indices.
    codes: The class index to species code map.

  Returns:
    The species codes.

  Raises:
    ValueError: If a class index has no code.
  '''
  resolved = []
  for item in species:
    if isinstance(item, str):
      resolved.append(item)
      continue
    index = int(item)
    if index not in codes:
      raise ValueError(f'Class index {index} has no species code.')
    resolved.append(codes[index])
  return resolved


def write_submission(rows: list[tuple[str, list[str]]], output: Path,
                     bird_codes: dict[int, str] | None = None) -> Path:
  '''Writes the submission CSV.

  The columns are ``row_id`` and ``labels``.  A row with no accepted
  species gets the string ``nocall`` and several species are separated
  by a space.

  The competition's own sample submission calls the second column
  ``birds``; this project writes ``labels``, so rename the column
  before uploading the file to Kaggle.

  A species may be given as its code or as its class index; the
  ``bird_codes`` map turns an index into a code and defaults to the
  packaged label space.

  Args:
    rows: The ``(row_id, species)`` pairs to write.
    output: The path of the CSV file.
    bird_codes: The class index to species code map.

  Returns:
    The path of the CSV file.
  '''
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  codes = get_inv_bird_code() if bird_codes is None else bird_codes
  with output.open('w', encoding='utf-8', newline='') as handle:
    writer = csv.writer(handle, lineterminator='\n')
    writer.writerow(['row_id', 'labels'])
    for row_id, species in rows:
      labels = ' '.join(_species_codes(species, codes)) or NOCALL
      writer.writerow([row_id, labels])
  return output
