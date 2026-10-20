#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The Lightning module of the SED pipeline.

The module owns the model, the criterion and the mixup augmenter, and it
replaces the original's three Ignite engines with the three Lightning
steps.  The mapping is one to one:

============================  ===================================
Ignite                        Lightning
============================  ===================================
``train_step``                ``training_step``
``eval_step``                 ``validation_step``
``test_step``                 ``test_step``
``prepare_batch``             ``prepare_batch``
``_init_optimizer``           ``configure_optimizers``
============================  ===================================

Two details of the original are load bearing and are reproduced exactly:

* the loss consumes **only** the clip head, never the frame head, which
  is unsupervised by design;
* the mixup halves the batch in ``prepare_batch`` and the model halves it
  again, so the label tensors and the waveforms stay in step.

The validation metrics are epoch-wise accumulators, so they are
**updated** in ``validation_step`` and reduced in
``on_validation_epoch_end``.  Their names carry the ``valid_`` prefix, and
the checkpoint monitor is built from the same prefix, so a monitor that is
never logged cannot happen.
'''

import lightning
import torch

from birdcall.augmentations.mixup import Mixup
from birdcall.augmentations.mixup import do_mixup
from birdcall.config_params.parameters import Parameters
from birdcall.losses.pl_registry import build_criterion
from birdcall.metrics.pl_registry import build_metric
from birdcall.models.pl_registry import build_model
from birdcall.optimizers.pl_optimizer_adamw import build_optimizer
from birdcall.optimizers.pl_optimizer_adamw import build_scheduler

TRAIN_PREFIX = 'train_'
VALID_PREFIX = 'valid_'
METRIC_THRESHOLD = 0.5
'''The detection threshold of the two F1 metrics, as in the original.'''


class SedLightningModule(lightning.LightningModule):
  '''The SED model, criterion and training steps.

  Attributes:
    config: The validated configuration of the run.
    model: The model, built by the model registry.
    criterion: The criterion, built by the criterion registry.
    mixup_augmenter: The mixup coefficient generator, or ``None`` when
      the run does not mix.
    loss_keys: The names of the criterion's scalar components.
    validation_metrics: The epoch-wise validation metric accumulators,
      keyed by the bare metric name.
    epoch_length: The number of iterations in one epoch, zero when it is
      not known yet.
  '''

  def __init__(self, parameters: Parameters,
               epoch_length: int | None = None) -> None:
    '''Builds the module of a run.

    Args:
      parameters: The validated configuration.
      epoch_length: The number of iterations in one epoch.  It is the
        length of the training dataloader when the configuration does
        not fix it.
    '''
    super().__init__()
    self.config = parameters
    self.epoch_length = int(epoch_length or
                            parameters.run_params.epoch_length or 0)
    self.model = build_model(parameters.model_name,
                             **vars(parameters.model_config))
    self.criterion = self._build_criterion(parameters)
    self.mixup_augmenter = (
        Mixup(mixup_alpha=1.0, random_seed=parameters.mixup_seed)
        if parameters.apply_mixup else None)
    self.loss_keys = list(getattr(self.criterion, 'loss_keys', []))
    self.validation_metrics = {
        name: self._build_metric(name) for name in parameters.metrics}

  def _build_criterion(self, parameters: Parameters):
    '''Builds the criterion of the run.

    Args:
      parameters: The validated configuration.

    Returns:
      The criterion.

    Raises:
      ValueError: If the configuration selects an unknown criterion.
    '''
    try:
      criterion_params = parameters.resolved_criterion_params()
    except KeyError as error:
      raise ValueError(
          f'Unknown criterion {parameters.criterion_name!r}.') from error
    return build_criterion(parameters.criterion_name, **criterion_params)

  def _build_metric(self, name: str):
    '''Builds one validation metric.

    Args:
      name: The name of the metric.

    Returns:
      The metric accumulator.
    '''
    if name == 'lraps':
      return build_metric(name)
    return build_metric(name, threshold=METRIC_THRESHOLD)

  def prepare_batch(self, batch: dict, mode: str = 'valid'):
    '''Turns a batch into the inputs of the model and of the criterion.

    The mixup is applied to the **labels** here and to the spectrogram
    inside the model, which is why the batch is halved twice: once for
    the labels and once for the waveforms.

    Args:
      batch: The batch of the dataloader.
      mode: One of ``train``, ``valid`` or ``test``.

    Returns:
      A tuple ``((waveforms, mixup_lambda), labels)``.  The
      ``mixup_lambda`` is ``None`` outside the training mode, and the
      label dict holds ``all_labels``, ``primary_labels`` and
      ``secondary_labels``, plus the file name in the test mode.

    Raises:
      ValueError: If the mode is not one of the three.
    '''
    if mode not in ('train', 'valid', 'test'):
      raise ValueError(
          f'Unknown mode {mode!r}. Valid modes: '
          f'{["train", "valid", "test"]}.')
    waveforms = batch['waveforms']
    labels = {
        'all_labels': batch['all_labels'],
        'primary_labels': batch['primary_labels'],
        'secondary_labels': batch['secondary_labels'],
    }
    if mode == 'test':
      labels['filename'] = batch['filename']
    if mode != 'train' or self.mixup_augmenter is None:
      return (waveforms, None), labels
    batch_size, snippets = waveforms.shape[0], waveforms.shape[1]
    wave_rows = batch_size * snippets
    label_rows = labels['all_labels'].shape[0] * labels['all_labels'].shape[1]
    if label_rows != wave_rows:
      raise ValueError(
          f'The labels hold {label_rows} rows and the waveforms hold '
          f'{wave_rows}. The mixup would mix the wrong rows.')
    mixup_lambda = self.mixup_augmenter.get_lambda(
        batch_size=wave_rows, device=waveforms.device)
    for key in ('all_labels', 'primary_labels', 'secondary_labels'):
      rows = labels[key].shape[0] * labels[key].shape[1]
      mixed = do_mixup(labels[key].reshape(rows, -1), mixup_lambda)
      labels[key] = mixed.reshape((rows) // 2, 1, -1)
    return (waveforms, mixup_lambda), labels

  def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
    '''Runs one training step.

    Args:
      batch: The batch of the dataloader.
      batch_idx: The index of the batch.

    Returns:
      The scalar loss of the batch.
    '''
    inputs, labels = self.prepare_batch(batch, mode='train')
    y_pred = self.model(inputs)
    loss, dict_loss = self.criterion(y_pred['clipwise_output'], labels)
    self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log_dict({f'{TRAIN_PREFIX}{key}': dict_loss[key]
                   for key in self.loss_keys},
                  on_step=False, on_epoch=True)
    return loss

  def validation_step(self, batch: dict, batch_idx: int) -> None:
    '''Runs one validation step.

    The validation metrics are epoch-wise accumulators, so they are
    updated here and reduced at the end of the epoch.

    Args:
      batch: The batch of the dataloader.
      batch_idx: The index of the batch.
    '''
    inputs, labels = self.prepare_batch(batch, mode='valid')
    y_pred = self.model(inputs)
    loss, dict_loss = self.criterion(y_pred['clipwise_output'], labels)
    self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log_dict({f'{VALID_PREFIX}{key}': dict_loss[key]
                   for key in self.loss_keys},
                  on_step=False, on_epoch=True)
    output = {'y_pred': y_pred, 'y': labels}
    for metric in self.validation_metrics.values():
      metric.update(output)

  def test_step(self, batch: dict, batch_idx: int) -> dict:
    '''Runs one test step.

    No loss is computed: the original's test step returned the
    predictions, the waveforms and the labels, and the dump callback
    consumed them.

    Args:
      batch: The batch of the dataloader.
      batch_idx: The index of the batch.

    Returns:
      The dict with the keys ``y_pred``, ``x`` and ``y``.
    '''
    inputs, labels = self.prepare_batch(batch, mode='test')
    y_pred = self.model(inputs)
    return {'y_pred': y_pred, 'x': inputs, 'y': labels}

  def on_validation_epoch_start(self) -> None:
    '''Zeroes the validation metric accumulators.'''
    for metric in self.validation_metrics.values():
      metric.reset()

  def on_validation_epoch_end(self) -> None:
    '''Reduces the validation metrics and logs them.

    The names are the bare metric names with the ``valid_`` prefix,
    which is the name the checkpoint monitor looks up.
    '''
    for name, metric in self.validation_metrics.items():
      self.log(f'{VALID_PREFIX}{name}', metric.compute(),
               on_step=False, on_epoch=True, prog_bar=True)

  def on_train_epoch_start(self) -> None:
    '''Advances the epoch of the sampler and of the dataset.

    The sampler draws its indices and the dataset draws its windows
    from ``(seed, epoch, index)``, so both have to know the epoch.
    '''
    loader = self._train_loader()
    if loader is None:
      return
    sampler = getattr(loader, 'sampler', None)
    if hasattr(sampler, 'set_epoch'):
      sampler.set_epoch(self.current_epoch)
    dataset = getattr(loader, 'dataset', None)
    if hasattr(dataset, 'set_epoch'):
      dataset.set_epoch(self.current_epoch)

  def _train_loader(self):
    '''Returns the training dataloader, if there is one.

    Returns:
      The training dataloader of the trainer, or ``None`` when the
      module is not attached to a trainer.
    '''
    trainer = self._trainer
    if trainer is None:
      return None
    try:
      return trainer.train_dataloader
    except (RuntimeError, TypeError):
      return None

  def _epoch_length(self) -> int:
    '''Returns the number of iterations in one epoch.

    Returns:
      The configured epoch length, or the length of the training
      dataloader, or one when neither is known.
    '''
    if self.epoch_length:
      return self.epoch_length
    loader = self._train_loader()
    if loader is not None:
      try:
        return len(loader)
      except TypeError:
        return 1
    return 1

  def configure_optimizers(self) -> dict:
    '''Builds the optimiser and the learning rate schedule.

    Returns:
      The Lightning optimiser dictionary.  The schedule is stepped with
      every optimiser step, which is the Lightning spelling of Ignite's
      per-iteration stepping.
    '''
    optimizer = build_optimizer(self.config, self.model)
    scheduler = build_scheduler(self.config, optimizer, self._epoch_length())
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        },
    }
