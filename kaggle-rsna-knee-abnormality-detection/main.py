#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI entrypoint orchestrating the RSNA knee MVP pipeline.

Subcommands map one-to-one onto BLUEPRINT phases:

* ``build-index``: header-only DICOM scan -> index.parquet
* ``build-labels``: rule-based pseudo-labels -> labels_pseudo.csv
* ``build-folds``: grouped stratified folds -> folds.csv
* ``train``: resume-aware fold training with session budget + pushes
* ``infer``: fold-ensemble prediction -> submission.csv

Example:
    python main.py train --experiment configs/experiments/mvp_efnv2s_384_k24_5f.yaml \
        --fold 0 --override data.n_slices=16
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import pytorch_lightning as pl

from knee.callbacks.session import PeriodicPushCallback, TimeBudgetCallback
from knee.config_params.loader import dump_config, instantiate
from knee.engines.assembly import (
    TARGET_COLUMNS,
    build_datamodule,
    build_datasets,
    build_model,
    find_resume_checkpoint,
    compose_experiment,
    fold_done,
)
from knee.engines.inferencer import predict_studies, write_submission
from knee.engines.train_module import KneeModule
from knee.helpers.folds import make_folds, resolve_group_column
from knee.helpers.header_scan import build_index, explode_sop_uids
from knee.helpers.kaggle_io import CredentialResolver, KaggleDatasetClient
from knee.helpers.nlp_labeling import RuleBasedLabeler, build_pseudo_labels
from knee.helpers.utils import get_logger
from knee.loggers.csv_logger import build_csv_logger

_LOGGER = get_logger('main')


def _parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        ArgumentParser with subcommands and shared flags.
    """
    parser = argparse.ArgumentParser(description='RSNA knee MVP pipeline')
    sub = parser.add_subparsers(dest='command', required=True)

    def add_common(target):
        """Attach shared flags to a subparser.

        Args:
            target: Subparser being configured.
        """
        target.add_argument('--experiment', required=True)
        target.add_argument('--override', nargs='*', default=[])

    for name in ['build-index', 'build-labels', 'build-folds', 'train', 'infer']:
        add_common(sub.add_parser(name))
    sub.choices['train'].add_argument('--fold', type=int, default=None)
    return parser


def _client(config: dict) -> KaggleDatasetClient | None:
    """Build the Kaggle client when resume is enabled.

    Args:
        config: Composed experiment configuration.

    Returns:
        Client instance or None when resume is disabled.
    """
    if not config['resume']['enabled']:
        return None
    secrets = config['kaggle_secrets']
    return KaggleDatasetClient(
        CredentialResolver(secrets['username_key'], secrets['token_key'])
    )


def cmd_build_index(config: dict) -> None:
    """Scan DICOM headers and persist the merged series index.

    Args:
        config: Composed experiment configuration.
    """
    frame = build_index(
        config['paths']['train_dicom_dir'],
        workers=int(config['data'].get('scan_workers', 4)),
    )
    series_meta = pd.read_csv(config['paths']['train_series_csv'])
    merged = frame.merge(
        series_meta.rename(columns={
            'SeriesInstanceUID': 'series',
            'StudyInstanceUID': 'study',
            'Anatomical_Plane': 'plane',
        })[['series', 'plane', 'Fluid_Sensitive', 'Fat_Suppression']],
        on='series',
        how='left',
    ).rename(columns={
        'Fluid_Sensitive': 'fluid_sensitive',
        'Fat_Suppression': 'fat_suppression',
    })
    out_path = config['paths']['index_parquet']
    merged.to_parquet(out_path, index=False)
    _LOGGER.info('Index written: %s (%d series)', out_path, len(merged))


def cmd_build_labels(config: dict) -> None:
    """Derive rule-based pseudo-labels from reports.

    Args:
        config: Composed experiment configuration.
    """
    train_df = pd.read_csv(config['paths']['train_csv'])
    labeled = build_pseudo_labels(
        train_df,
        study_column='StudyInstanceUID',
        target_columns=TARGET_COLUMNS,
        labeler=RuleBasedLabeler(),
    )
    out_path = config['paths']['labels_csv']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labeled.to_csv(out_path, index=False)
    _LOGGER.info('Labels written: %s (%d rows)', out_path, len(labeled))


def cmd_build_folds(config: dict) -> None:
    """Assign CV folds and persist the mapping.

    Args:
        config: Composed experiment configuration.
    """
    labels = pd.read_csv(config['paths']['labels_csv'])
    splitter = instantiate(config['folds'])
    stratify = config.get('stratify', {})
    fold_series = make_folds(
        labels,
        splitter,
        rare_targets=stratify.get('rare_targets', []),
        anchor_targets=stratify.get('anchor_targets', []),
    )
    group_column, _ = resolve_group_column(labels)
    out_frame = pd.DataFrame({
        group_column: labels[group_column].values,
        'fold': fold_series.values,
    })
    out_path = config['paths']['folds_csv']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_frame.to_csv(out_path, index=False)
    _LOGGER.info('Folds written: %s (groups=%s)', out_path, group_column)


def cmd_train(config: dict, fold_id: int | None) -> None:
    """Run resume-aware fold training within one kernel session.

    Args:
        config: Composed experiment configuration.
        fold_id: Optional single-fold override of run.folds.
    """
    client = _client(config)
    checkpoint_dir = config['paths']['checkpoint_dir']
    if client is not None:
        client.pull_latest(config['resume']['checkpoint_dataset_slug'], checkpoint_dir)
    folds = [fold_id] if fold_id is not None else list(config['run']['folds'])

    index_df = explode_sop_uids(pd.read_parquet(config['paths']['index_parquet']))
    labels = pd.read_csv(config['paths']['labels_csv'])
    folds_df = pd.read_csv(config['paths']['folds_csv'])
    labels = labels.merge(folds_df[['StudyInstanceUID', 'fold']], on='StudyInstanceUID')
    label_lookup = labels.set_index('StudyInstanceUID')

    trainer_cfg = {
        key: value for key, value in config['trainer']['init_params'].items()
    }
    trainer_cfg.setdefault('gradient_clip_val', config['optimizer'].get('gradient_clip_val'))
    total_epochs = int(trainer_cfg.get('max_epochs', 1))

    for current_fold in folds:
        if fold_done(checkpoint_dir, current_fold):
            _LOGGER.info('Fold %d already done; skipping', current_fold)
            continue
        valid_ids = label_lookup.index[label_lookup['fold'] == current_fold].tolist()
        train_ids = label_lookup.index[label_lookup['fold'] != current_fold].tolist()
        train_ds, valid_ds = build_datasets(config, index_df, labels, valid_ids, train_ids)
        module = KneeModule(
            model=build_model(config),
            criterion=instantiate(config['loss']),
            optimizer_cfg=config['optimizer']['optimizer'],
            scheduler_cfg=config['optimizer'].get('scheduler'),
            warmup_epochs=int(config['optimizer'].get('warmup_epochs', 0)),
            backbone_lr_scale=float(config['optimizer']['backbone_lr_scale']),
            total_epochs=total_epochs,
            target_columns=TARGET_COLUMNS,
            oof_dir=config['paths']['oof_dir'],
            fold_id=current_fold,
        )
        callbacks = [
            TimeBudgetCallback(
                session_time_budget_h=float(config['session_time_budget_h']),
                time_margin_min=float(config.get('time_margin_min', 30.0)),
            ),
            PeriodicPushCallback(
                checkpoint_dir=checkpoint_dir,
                fold_id=current_fold,
                push_every_n_epochs=int(config['checkpoint_every_n_epochs']),
                client=client,
                push_slug=(
                    config['resume']['checkpoint_dataset_slug'] if client else None
                ),
            ),
        ]
        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=build_csv_logger(
                config['experiment']['output_dir'],
                config['experiment']['name'],
                current_fold,
            ),
            **trainer_cfg,
        )
        resume_path = find_resume_checkpoint(checkpoint_dir, current_fold)
        _LOGGER.info('Fold %d fit start (resume=%s)', current_fold, resume_path)
        trainer.fit(
            module,
            datamodule=build_datamodule(config, train_ds, valid_ds),
            ckpt_path=resume_path,
        )


def _collect_fold_checkpoints(config: dict) -> list[str]:
    """List completed-fold checkpoints honoring infer.yaml's fold selection.

    Args:
        config: Composed experiment configuration.

    Returns:
        Checkpoint paths for folds carrying both ``done`` and ``last.ckpt``.
    """
    checkpoint_dir = config['paths']['checkpoint_dir']
    requested = config['infer']['folds']
    keep = (
        {f'fold{int(f)}' for f in requested} if requested != 'all' else None
    )
    paths = []
    for ckpt in sorted(glob.glob(os.path.join(checkpoint_dir, 'fold*', 'last.ckpt'))):
        fold_name = os.path.basename(os.path.dirname(ckpt))
        done_marker = os.path.join(os.path.dirname(ckpt), 'done')
        if not os.path.exists(done_marker):
            continue
        if keep is not None and fold_name not in keep:
            continue
        paths.append(ckpt)
    return paths


def cmd_infer(config: dict) -> None:
    """Ensemble fold checkpoints and emit submission.csv.

    Args:
        config: Composed experiment configuration.
    """
    client = _client(config)
    checkpoint_dir = config['paths']['checkpoint_dir']
    if client is not None:
        client.pull_latest(config['resume']['checkpoint_dataset_slug'], checkpoint_dir)
    fold_paths = _collect_fold_checkpoints(config)
    assert fold_paths, 'No completed fold checkpoints found for inference'

    test_csv = pd.read_csv(config['paths']['test_csv'])
    test_index_path = config['paths']['index_parquet'].replace('.parquet', '_test.parquet')
    test_index = explode_sop_uids(pd.read_parquet(test_index_path))
    config['paths']['train_dicom_dir'] = config['paths']['test_dicom_dir']

    predictions = predict_studies(
        config,
        test_index,
        test_csv['StudyInstanceUID'].tolist(),
        fold_paths,
    )
    write_submission(
        predictions,
        config['infer']['submission_path'],
        expected_uids=set(test_csv['StudyInstanceUID']),
    )


def main() -> None:
    """Parse arguments and dispatch the selected command."""
    args = _parser().parse_args()
    config = compose_experiment(args.experiment, args.override or None)
    dump_config(
        config,
        os.path.join(
            config['paths']['artifact_dir'],
            f'resolved_{config["experiment"]["name"]}.yaml',
        ),
    )
    handlers = {
        'build-index': lambda: cmd_build_index(config),
        'build-labels': lambda: cmd_build_labels(config),
        'build-folds': lambda: cmd_build_folds(config),
        'train': lambda: cmd_train(config, args.fold),
        'infer': lambda: cmd_infer(config),
    }
    handlers[args.command]()


if __name__ == '__main__':
    main()
