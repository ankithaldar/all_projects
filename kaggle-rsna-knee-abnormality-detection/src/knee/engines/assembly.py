#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Object assembly bridging YAML configuration and runtime components.

Every builder here consumes the composed experiment dictionary (see
``config_params.loader.load_experiment``) so no tunable value leaks into
code. ``main.py`` orchestrates; this module constructs.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from knee.config_params.loader import instantiate, load_experiment
from knee.datasets.series_dataset import SeriesReader
from knee.datasets.study_dataset import StudyDataset
from knee.helpers.dicom_io import DecoderRegistry
from knee.helpers.utils import seed_everything

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture',
]

PLANE_ORDER = ['Sagittal', 'Coronal', 'Axial']
SEX_ORDER = ['M', 'F', 'O']


def compose_experiment(path: str, overrides: list[str] | None = None) -> dict:
    """Load and seed an experiment configuration.

    Args:
        path: Experiment YAML path under configs/experiments/.
        overrides: Optional CLI dot-path overrides.

    Returns:
        Composed configuration dictionary.
    """
    config = load_experiment(path, overrides)
    seed_everything(int(config['experiment']['seed']))
    return config


def build_reader(config: dict, dicom_root: str) -> SeriesReader:
    """Create the streaming series reader.

    Args:
        config: Composed experiment configuration.
        dicom_root: DICOM root to read from (train or test mount).

    Returns:
        Configured SeriesReader.
    """
    data = config['data']
    registry = DecoderRegistry(data['decode_backend_order'])
    return SeriesReader(
        dicom_root=dicom_root,
        registry=registry,
        n_slices=int(data['n_slices']),
        percentiles=tuple(data['normalize_percentiles']),
        autocrop_margin=float(data['autocrop_margin']),
    )


def _augmentation_pipelines(config: dict):
    """Build train/valid augmentation pipelines.

    Args:
        config: Composed experiment configuration.

    Returns:
        Tuple ``(train_compose, valid_compose)``.
    """
    from knee.augmentations.factory import build_compose

    img_size = int(config['data']['img_size'])
    normalize_output = config['data']['normalize_output']
    specs = config.get('augmentations', {})
    return (
        build_compose(specs.get('train', []), img_size, normalize_output),
        build_compose(specs.get('valid', []), img_size, normalize_output),
    )


def build_datasets(
    config: dict,
    index_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    valid_study_ids: list[str],
    train_study_ids: list[str] | None,
):
    """Construct fold-split study datasets.

    Args:
        config: Composed experiment configuration.
        index_df: Header-scan index (sop lists already exploded).
        labels_df: Labels frame or None at inference.
        valid_study_ids: Studies held out for validation.
        train_study_ids: Training studies; None builds validation-only pairs.

    Returns:
        Tuple ``(train_dataset_or_None, valid_dataset)``.
    """
    data = config['data']
    dicom_root = config['paths']['train_dicom_dir']
    reader = build_reader(config, dicom_root)
    augment_train, augment_valid = _augmentation_pipelines(config)

    def make(study_ids, augment):
        return StudyDataset(
            index_df=index_df,
            labels_df=labels_df,
            study_ids=list(study_ids),
            reader=reader,
            augmentations=augment,
            img_size=int(data['img_size']),
            n_slices=int(data['n_slices']),
            n_series_tokens_max=int(config['model']['init_params']['n_series_tokens_max']),
            series_selection=data['series_selection'],
            metadata_features=data['metadata_features'],
            normalize_output=data['normalize_output'],
            target_columns=TARGET_COLUMNS,
        )

    valid_dataset = make(valid_study_ids, augment_valid)
    train_dataset = (
        make(train_study_ids, augment_train) if train_study_ids is not None else None
    )
    return train_dataset, valid_dataset


def build_datamodule(config: dict, train_dataset, valid_dataset):
    """Instantiate and attach the Lightning DataModule.

    Args:
        config: Composed experiment configuration.
        train_dataset: Training dataset (may be None at inference).
        valid_dataset: Validation/inference dataset.

    Returns:
        Attached StudyDataModule.
    """
    module = instantiate(config['datamodule'])
    if train_dataset is not None:
        module.attach(train_dataset, valid_dataset)
    else:
        module.attach(valid_dataset, valid_dataset)
    return module


def build_model(config: dict):
    """Instantiate KneeNet from the model section.

    Args:
        config: Composed experiment configuration.

    Returns:
        Unwrapped KneeNet module.
    """
    model_cfg = dict(config['model'])
    model_cfg['init_params'] = dict(model_cfg.get('init_params', {}))
    model_cfg['init_params'].setdefault('pretrained_cfg', {})
    return instantiate(model_cfg)


def find_resume_checkpoint(checkpoint_dir: str, fold_id: int) -> str | None:
    """Locate an existing last-checkpoint for a fold.

    Args:
        checkpoint_dir: Root checkpoint directory.
        fold_id: Fold number searched.

    Returns:
        Path to ``last.ckpt`` when present, else None.
    """
    pattern = os.path.join(checkpoint_dir, f'fold{fold_id}', 'last.ckpt')
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def fold_done(checkpoint_dir: str, fold_id: int) -> bool:
    """Check whether a fold finished in any previous session.

    Args:
        checkpoint_dir: Root checkpoint directory.
        fold_id: Fold number checked.

    Returns:
        True when the ``done`` marker exists.
    """
    return os.path.exists(os.path.join(checkpoint_dir, f'fold{fold_id}', 'done'))
