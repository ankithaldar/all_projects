#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the rule-based pseudo-labeler (negation, uncertainty, OA)."""

import pandas as pd
import pytest

from knee.helpers.nlp_labeling import (
  NEGATED,
  POSITIVE,
  UNKNOWN,
  RuleBasedLabeler,
  build_pseudo_labels,
)

LABELER = RuleBasedLabeler()


@pytest.mark.parametrize(
  'report,target,expected',
  [
    # Affirmative mentions.
    ('Full-thickness tear of the ACL.', 'ACL', POSITIVE),
    ('There is a Baker cyst posteriorly.', "Baker's", POSITIVE),
    ('Joint effusion is present.', 'Effusion', POSITIVE),
    ('Non-displaced patellar fracture.', 'Fracture', POSITIVE),
    # Negated mentions within the window.
    ('The ACL is intact.', 'ACL', NEGATED),
    ('No evidence of fracture.', 'Fracture', NEGATED),
    ('There is no joint effusion.', 'Effusion', NEGATED),
    # Uncertainty sentinels.
    ('Possible tear of the medial meniscus.', 'Medial Meniscus', UNKNOWN),
    ('Cannot exclude an ACL injury.', 'ACL', UNKNOWN),
  ],
)
def test_simple_targets(report: str, target: str, expected: int):
  assert LABELER.label_report(report)[target] == expected


@pytest.mark.parametrize(
  'report,target,expected',
  [
    ('Moderate medial compartment osteoarthritis.', 'Medial OA', POSITIVE),
    (
      'Osteoarthritis predominantly affecting the lateral compartment.',
      'Lateral OA',
      POSITIVE,
    ),
    (
      'Patellofemoral cartilage loss with joint space narrowing.',
      'PF OA',
      POSITIVE,
    ),
    ('Tricompartmental degenerative joint disease.', 'Medial OA', POSITIVE),
    ('No significant medial compartment osteoarthritis.', 'Medial OA', NEGATED),
  ],
)
def test_oa_cooccurrence(report: str, target: str, expected: int):
  assert LABELER.label_report(report)[target] == expected


def test_unmentioned_target_is_unknown():
  labels = LABELER.label_report('Normal study of the right knee.')
  assert all(value == UNKNOWN for value in labels.values())


def test_negation_beyond_window_is_ignored():
  # 'no' sits in the previous sentence -> sentence scoping must block it.
  report = (
    'There was no trauma and the examination was of excellent quality. '
    'ACL appears disrupted.'
  )
  assert LABELER.label_report(report)['ACL'] == POSITIVE


def test_affirmation_outweighs_distant_negation():
  report = 'No effusion. A small suprapatellar effusion is also seen.'
  assert LABELER.label_report(report)['Effusion'] == POSITIVE


def test_build_pseudo_labels_prefers_gold():
  frame = pd.DataFrame(
    {
      'StudyInstanceUID': ['s1', 's2'],
      'Report': ['Intact ACL.', 'ACL tear.'],
      'ACL': [1.0, float('nan')],
    }
  )
  out = build_pseudo_labels(frame, 'StudyInstanceUID', target_columns=['ACL'])
  assert out.loc[0, 'ACL'] == 1  # gold preserved
  assert out.loc[0, 'source'] == 'gold'
  assert out.loc[1, 'ACL'] == POSITIVE  # rule-derived
  assert out.loc[1, 'source'] == 'rules'
