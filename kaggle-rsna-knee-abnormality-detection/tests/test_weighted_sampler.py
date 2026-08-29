#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for imbalanced-target study sampling.

Covers prevalence math, weight tempering/aggregation, the torch sampler
alignment and draw skew, the YAML factory, and the DataModule wiring
(train-only sampler; validation stays uniform).
"""

# pytest fixtures + duck-typed stubs mirror the other suites.
# pylint: disable=redefined-outer-name,invalid-name

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import (
  RandomSampler,
  SequentialSampler,
  WeightedRandomSampler,
)

from knee.datamodules.study_datamodule import StudyDataModule
from knee.engines.assembly import build_datamodule
from knee.samplers.factory import build_train_sampler
from knee.samplers.weighted import (
  StudyWeightedRandomSampler,
  positive_prevalences,
  study_weights,
)

TARGETS = ['acl', 'fracture']


def _labels_frame():
  """Deterministic labels frame: 5 studies, skewed fracture prevalence.

  Returns:
      DataFrame with StudyInstanceUID + TARGETS columns.
  """
  return pd.DataFrame(
    {
      'StudyInstanceUID': [f's{i}' for i in range(5)],
      'acl': [1.0, 0.0, 1.0, 0.0, -1.0],
      'fracture': [0.0, 1.0, -1.0, 0.0, -1.0],
    }
  )


class TestPositivePrevalences:
  """Prevalence over unmasked entries only."""

  def test_ignores_unknown(self):
    matrix = np.array(
      [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
      ]
    )
    prevalence = positive_prevalences(matrix)
    # acl: 2/3 positives over 3 unmasked; fracture: 1/2 over 2.
    assert prevalence[0] == pytest.approx(2.0 / 3.0)
    assert prevalence[1] == pytest.approx(0.5)

  def test_all_unknown_target_is_zero(self):
    matrix = np.array([[1.0, -1.0], [0.0, -1.0]])
    prevalence = positive_prevalences(matrix)
    assert prevalence[0] == pytest.approx(0.5)
    assert prevalence[1] == 0.0

  def test_nan_counts_as_unknown(self):
    matrix = np.array([[1.0, np.nan], [0.0, np.nan]])
    prevalence = positive_prevalences(matrix)
    assert prevalence[1] == 0.0


class TestStudyWeights:
  """Tempering, aggregation, and baseline behavior."""

  def test_zero_tempering_is_uniform(self):
    weights = study_weights(_labels_frame()[TARGETS].to_numpy(), tempering=0.0)
    assert np.allclose(weights, 1.0)

  def test_inverse_frequency_math(self):
    frame = _labels_frame()
    matrix = frame[TARGETS].to_numpy()
    # acl prevalence 2/4 (row 4 unknown) -> raw weight 2.0
    # fracture prevalence 1/3 -> raw weight 3.0; mean = 2.5
    weights = study_weights(matrix, tempering=1.0, aggregation='max')
    # normalization runs over KNOWN targets only: a row whose second
    # target is entirely unknown anchors that target's weight at 1.0.
    acl_only = study_weights(
      np.array([[1.0, 0.0]]), tempering=1.0, aggregation='max'
    )
    assert acl_only[0] == pytest.approx(1.0)
    # s1 is the only fracture positive -> full 3.0/2.5 = 1.2
    assert weights[1] == pytest.approx(3.0 / 2.5)
    # s4 fully unknown -> baseline
    assert weights[4] == pytest.approx(1.0)
    # s0 and s2 are acl-only positives -> 2.0/2.5 = 0.8, below s1
    assert weights[0] == pytest.approx(2.0 / 2.5)
    assert weights[2] == pytest.approx(2.0 / 2.5)
    assert weights[1] > weights[0]

  def test_mean_aggregation_blends(self):
    # 3/5 acl prevalence vs 2/5 fracture -> normalized weights ~0.8/1.2.
    matrix = np.array(
      [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
      ]
    )
    max_w = study_weights(matrix, tempering=1.0, aggregation='max')[2]
    mean_w = study_weights(matrix, tempering=1.0, aggregation='mean')[2]
    assert max_w > mean_w
    assert mean_w == pytest.approx((0.8 + 1.2) / 2)

  def test_negative_only_study_gets_baseline(self):
    weights = study_weights(np.array([[0.0, 0.0]]), tempering=1.0)
    assert weights[0] == pytest.approx(1.0)

  def test_invalid_aggregation_raises(self):
    with pytest.raises(ValueError):
      study_weights(np.array([[1.0]]), aggregation='sum')

  def test_negative_tempering_raises(self):
    with pytest.raises(ValueError):
      study_weights(np.array([[1.0]]), tempering=-0.1)


class TestStudyWeightedRandomSampler:
  """Torch sampler alignment and statistical skew."""

  def _sampler(self, tempering=1.0, **kwargs):
    return StudyWeightedRandomSampler(
      study_ids=[f's{i}' for i in range(5)],
      label_frame=_labels_frame(),
      target_columns=TARGETS,
      tempering=tempering,
      **kwargs,
    )

  def test_num_samples_defaults_to_dataset_size(self):
    assert self._sampler().num_samples == 5

  def test_num_samples_override(self):
    assert self._sampler(num_samples=17).num_samples == 17

  def test_weights_aligned_to_study_order(self):
    sampler = self._sampler()
    reordered = StudyWeightedRandomSampler(
      study_ids=['s1', 's0', 's2', 's3', 's4'],
      label_frame=_labels_frame(),
      target_columns=TARGETS,
      tempering=1.0,
    )
    assert sampler.weights[1] == pytest.approx(reordered.weights[0])
    assert sampler.weights[0] == pytest.approx(reordered.weights[1])

  def test_missing_study_gets_baseline_weight(self):
    sampler = StudyWeightedRandomSampler(
      study_ids=['s0', 'ghost'],
      label_frame=_labels_frame(),
      target_columns=TARGETS,
      tempering=1.0,
    )
    assert sampler.weights[1] == pytest.approx(1.0)

  def test_draws_skew_toward_rare_positives(self):
    sampler = StudyWeightedRandomSampler(
      study_ids=[f's{i}' for i in range(5)],
      label_frame=_labels_frame(),
      target_columns=TARGETS,
      tempering=1.0,
      num_samples=4000,
    )
    torch.manual_seed(7)
    counts = np.bincount(list(sampler), minlength=5)
    # s1 carries the only fracture positive (rarest) -> most drawn.
    assert counts[1] == counts.max()
    # uniform expectation is 800 draws; the fracture study must clear it.
    assert counts[1] > 800
    # neutral studies (s2/s3) must stay near or below expectation.
    assert counts[2] <= 800 * 1.5

  def test_zero_tempering_matches_uniform_expectation(self):
    sampler = StudyWeightedRandomSampler(
      study_ids=[f's{i}' for i in range(5)],
      label_frame=_labels_frame(),
      target_columns=TARGETS,
      tempering=0.0,
      num_samples=4000,
    )
    torch.manual_seed(11)
    counts = np.bincount(list(sampler), minlength=5)
    assert np.all(counts > 600)
    assert np.all(counts < 1000)


class TestBuildTrainSampler:
  """YAML spec factory."""

  def _dataset_stub(self, with_labels=True):
    class Stub:
      study_ids = [f's{i}' for i in range(5)]
      labels_df = _labels_frame() if with_labels else None
      target_columns = TARGETS

    return Stub()

  def test_none_spec_returns_none(self):
    assert build_train_sampler(None, self._dataset_stub()) is None

  def test_no_labels_returns_none(self):
    assert (
      build_train_sampler({'tempering': 0.5}, self._dataset_stub(False)) is None
    )

  def test_default_class_builds_configured_sampler(self):
    sampler = build_train_sampler(
      {'init_params': {'tempering': 0.25, 'aggregation': 'mean'}},
      self._dataset_stub(),
    )
    assert isinstance(sampler, WeightedRandomSampler)
    assert isinstance(sampler, StudyWeightedRandomSampler)
    expected = study_weights(
      _labels_frame()[TARGETS].to_numpy(),
      tempering=0.25,
      aggregation='mean',
    )
    assert np.allclose(sampler.weights.numpy(), expected)

  def test_explicit_class_path_respected(self):
    sampler = build_train_sampler(
      {
        'class_path': 'knee.samplers.weighted.StudyWeightedRandomSampler',
        'init_params': {'tempering': 0.0},
      },
      self._dataset_stub(),
    )
    assert np.allclose(sampler.weights.numpy(), 1.0)

  def test_bad_class_path_raises(self):
    with pytest.raises(ImportError):
      build_train_sampler(
        {'class_path': 'knee.samplers.weighted.NoSuchSampler'},
        self._dataset_stub(),
      )


def _item(study_uid, n_targets=2):
  """Minimal collate-compatible study item.

  Args:
      study_uid: Identifier string for the item.
      n_targets: Label width.

  Returns:
      Dictionary matching collate_studies' expectations.
  """
  return {
    'slices': torch.zeros(1, 3, 8, 8),
    'slice_counts': torch.tensor([1, 0]),
    'series_meta': torch.zeros(2, 3),
    'metadata': torch.zeros(4),
    'label': torch.zeros(n_targets),
    'study_uid': study_uid,
  }


class TestDataModuleWiring:
  """Sampler engages on train only; validation stays uniform."""

  def _module(self, train_sampler_cfg):
    module = StudyDataModule(
      batch_size=2,
      num_workers=0,
      prefetch_factor=2,
      persistent_workers=False,
      pin_memory=False,
    )
    module.train_sampler_cfg = train_sampler_cfg

    class Stub:
      study_ids = [f's{i}' for i in range(5)]
      labels_df = _labels_frame()
      target_columns = TARGETS

      def __len__(self):
        return len(self.study_ids)

      def __getitem__(self, idx):
        return _item(self.study_ids[idx])

    stub = Stub()
    module.attach(stub, stub)
    return module

  def test_train_loader_uses_sampler_when_configured(self):
    loader = self._module({'tempering': 0.5}).train_dataloader()
    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert not isinstance(loader.sampler, RandomSampler)

  def test_train_loader_uniform_without_spec(self):
    loader = self._module(None).train_dataloader()
    assert isinstance(loader.sampler, RandomSampler)

  def test_val_loader_stays_uniform(self):
    loader = self._module({'tempering': 0.5}).val_dataloader()
    assert isinstance(loader.sampler, SequentialSampler)

  def test_batches_iterate(self):
    loader = self._module({'tempering': 0.5}).train_dataloader()
    batch = next(iter(loader))
    assert batch['slices'].shape == (2, 3, 8, 8)
    assert len(batch['study_uid']) == 2


class TestAssemblyInjection:
  """build_datamodule passes the sibling spec through."""

  def test_sibling_spec_reaches_module(self):
    config = {
      'datamodule': {
        'class_path': 'knee.datamodules.study_datamodule.StudyDataModule',
        'init_params': {
          'batch_size': 2,
          'num_workers': 0,
          'prefetch_factor': 2,
          'persistent_workers': False,
          'pin_memory': False,
        },
        'train_sampler': {'tempering': 0.5},
      }
    }

    class Stub:
      study_ids = [f's{i}' for i in range(5)]
      labels_df = _labels_frame()
      target_columns = TARGETS

      def __len__(self):
        return len(self.study_ids)

      def __getitem__(self, idx):
        return _item(self.study_ids[idx])

    module = build_datamodule(config, Stub(), Stub())
    assert module.train_sampler_cfg == {'tempering': 0.5}
    assert isinstance(module.train_dataloader().sampler, WeightedRandomSampler)
