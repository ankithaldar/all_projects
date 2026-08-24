#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Weak labels from multilingual reports via the OpenRouter API.

Alternative to ``train_text_teacher.py`` when GPU hours are better
spent on the image student: a flash-class LLM reads every training
report (any language) zero-shot and returns the 12 finding
probabilities; :class:`knee.engines.weak_label_builder.WeakLabelBuilder`
then fuses gold > rule seeds > LLM exactly as for the XLM-R teacher.

Resumability (12 h kernel wall): per-report results persist to a cache
parquet keyed by report hash -- re-running only queries uncached
reports, and partial progress survives kills via periodic flushes.

Setup: Add-ons -> Secrets -> ``OPENROUTER_API_KEY`` (or env/.env).
Cost guide: one chat call per study at ~0.5-1.5k tokens; flash-class
models label full train sets in single-digit dollars.

Usage:
    python scripts/build_weak_labels_llm.py \
        --config configs/labeling/text_teacher.yaml \
        [--model openai/gpt-4o-mini] [--concurrency 8] [--limit 100]

Outputs:
    <output_dir>/weak_labels.parquet  (+ llm_label_cache.parquet)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from knee.config_params.schema import TARGETS
from knee.engines.llm_labeler import OpenRouterLabeler, label_many
from knee.engines.rule_labeler import RuleBasedLabeler
from knee.engines.text_teacher_lit import TextTeacherConfig
from knee.engines.weak_label_builder import WeakLabelBuilder
from knee.helpers.env import get_secret
from knee.helpers.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with config, model, concurrency, limit, cache and
      fusion overrides.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', default='configs/labeling/text_teacher.yaml')
  parser.add_argument(
    '--model',
    default='openai/gpt-4o-mini',
    help='OpenRouter model slug.',
  )
  parser.add_argument(
    '--concurrency', type=int, default=8, help='Parallel API calls.'
  )
  parser.add_argument(
    '--limit',
    type=int,
    default=0,
    help='Debug cap on studies labeled (>0 limits).',
  )
  parser.add_argument(
    '--out',
    default=None,
    help='Output parquet (default <output_dir>/weak_labels.parquet). '
    'Pass $WORK/weak_labels.parquet on Kaggle so it lands in the '
    'published dataset.',
  )
  parser.add_argument(
    '--cache',
    default=None,
    help='Resume cache parquet (default <output_dir>/llm_label_cache.parquet).',
  )
  parser.add_argument(
    '--teacher-weight',
    type=float,
    default=None,
    help='Override fusion.teacher_weight from the config.',
  )
  return parser.parse_args()


def _rule_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Compute Tier-1 rule seeds for every study.

  Args:
      df: Study frame with Report and StudyInstanceUID columns.

  Returns:
      (rule_probs, rule_mask) frames keyed by StudyInstanceUID.
  """
  labeler = RuleBasedLabeler()
  seeds = [labeler.seed_labels(text or '') for text in df['Report'].fillna('')]
  probs = pd.DataFrame([s[0] for s in seeds], columns=list(TARGETS))
  mask = pd.DataFrame([s[1] for s in seeds], columns=list(TARGETS))
  probs.insert(0, 'StudyInstanceUID', df['StudyInstanceUID'].to_numpy())
  mask.insert(0, 'StudyInstanceUID', df['StudyInstanceUID'].to_numpy())
  return probs, mask


def main() -> None:
  """Label reports with OpenRouter and fuse into weak_labels.parquet."""

  args = parse_args()
  log = get_logger('build_weak_labels_llm')
  cfg = TextTeacherConfig.model_validate(
    OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
  )
  out_dir = Path(cfg.data.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  train_csv = str(Path(cfg.data.train_csv))
  df = pd.read_csv(train_csv)
  if args.limit > 0:
    df = df.head(args.limit)
    log.info('debug limit active: %d studies', len(df))

  cache_path = args.cache or str(out_dir / 'llm_label_cache.parquet')
  api_key = get_secret('OPENROUTER_API_KEY') or ''
  pending_probe = df['Report'].fillna('').astype(str)
  if not api_key and pending_probe.str.strip().any():
    raise SystemExit(
      'OPENROUTER_API_KEY missing: add it to Kaggle Secrets / env'
    )
  labeler = OpenRouterLabeler(api_key=api_key, model=args.model)
  llm = label_many(
    df['StudyInstanceUID'],
    df['Report'].fillna('').astype(str),
    labeler.label_report,
    cache_path=cache_path,
    concurrency=args.concurrency,
  )

  rule_probs, rule_mask = _rule_frames(df)
  teacher_oof = llm[['StudyInstanceUID', *TARGETS]]
  fusion = cfg.fusion
  builder = WeakLabelBuilder(
    rule_confidence_floor=fusion.rule_confidence_floor,
    teacher_weight=(
      args.teacher_weight
      if args.teacher_weight is not None
      else fusion.teacher_weight
    ),
    min_positive_prob=fusion.min_positive_prob,
  )
  weak, stats = builder.build(
    train_csv=train_csv,
    rule_probs=rule_probs,
    rule_mask=rule_mask,
    rule_precision=None,
    teacher_oof=teacher_oof,
  )
  out = Path(args.out) if args.out else out_dir / 'weak_labels.parquet'
  out.parent.mkdir(parents=True, exist_ok=True)
  weak.to_parquet(out)
  known = int((~np.isclose(teacher_oof[list(TARGETS)], 0.5)).any(axis=1).sum())
  log.info(
    'round labels -> %s | %s | %d/%d studies LLM-labeled',
    out,
    stats,
    known,
    len(df),
  )


if __name__ == '__main__':
  main()
