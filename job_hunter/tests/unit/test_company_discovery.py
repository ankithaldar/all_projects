#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for company normalization and vertical rule classification.'''


from __future__ import annotations

from pathlib import Path

import yaml
from job_hunter.db.repositories.companies import normalize_name
from job_hunter.services.vertical_classifier import VerticalClassifier


def test_normalize_name_strips_suffixes() -> None:
  '''Suffix stripping produces a stable key.'''
  assert normalize_name('Flipkart Internet Private Limited') == 'flipkart internet'
  assert normalize_name('  Razorpay   Pvt. Ltd. ') == 'razorpay'


def test_classifier_rules(tmp_path: Path) -> None:
  '''Keyword rules classify text and miss cleanly.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  taxonomy = {
    'verticals': {
      'fintech': {'keywords': ['payments', 'lending', 'upi']},
      'saas': {'keywords': ['saas', 'crm']},
    },
    'default_confidence': {'keyword_hit': 0.7},
  }
  path = tmp_path / 'verticals.yaml'
  path.write_text(yaml.safe_dump(taxonomy), encoding='utf-8')
  classifier = VerticalClassifier(path)
  vertical, confidence = classifier.classify_rules(
    'We build payments and lending products for India',
  )
  assert vertical == 'fintech'
  assert confidence == 0.7
  assert classifier.classify_rules('we sell vegetables') == (None, 0.0)
