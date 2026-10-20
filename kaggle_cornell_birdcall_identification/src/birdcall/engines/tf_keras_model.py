#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The training wrapper of the Keras backend.

The Keras port has no ``SedEngine``: the framework replaces the control
plane.  What it does not replace is the **mixup halving**, because the
batch axis changes its meaning in the middle of the forward pass: the
model halves the batch, so the labels must be halved with it, and only
the training step knows both.

Three things are therefore overridden here:

* ``train_step`` -- the mixup coefficients are drawn, the labels are
  mixed and halved, the forward pass is run, and the loss is computed
  and minimised;
* ``compute_loss`` -- the argument order is inverted in Keras, and the
  criterion takes the label dict and the model output dict;
* ``test_step`` -- it returns the same dict the torch engine's
  ``output_transform`` returns, so the two backends can be compared.

The criterion's ``loss_keys`` are turned into one running mean each, so
the five extra metrics the original logged are logged here too.
'''


import numpy as np
import tensorflow as tf

from birdcall.augmentations.mixup import Mixup


def _is_scalar(value) -> bool:
  '''Whether a value is a single number.

  Args:
    value: The value.

  Returns:
    True when the value is a number or a zero dimensional tensor.
  '''
  if isinstance(value, (int, float, bool)):
    return True
  ndim = getattr(value, 'ndim', None)
  if ndim is None:
    return False
  return int(ndim) == 0

@tf.keras.utils.register_keras_serializable(package='birdcall',
                                           name='SedKerasModel')
class SedKerasModel(tf.keras.models.Model):
  '''The training model of the Keras backend.

  Attributes:
    backbone: The model that turns a waveform into the clip and the
      frame heads.
    criterion: The criterion, a ``keras.losses.Loss``.
    apply_mixup: Whether the mixup runs.
    mixup: The mixup coefficient generator.  Its seed is the original's
      hard coded 1234, so the sequence of coefficients is the same one
      the original drew.
    loss_metrics: One running mean per loss key.
  '''

  def __init__(self, backbone, criterion=None, apply_mixup: bool = False,
               mixup_alpha: float = 1.0, mixup_seed: int = 1234,
               name: str | None = None) -> None:
    '''Builds the model.

    Args:
      backbone: The model that turns a waveform into the clip and the
        frame heads.
      criterion: The criterion, a ``keras.losses.Loss``.
      apply_mixup: Whether the mixup runs.
      mixup_alpha: The shape parameter of the Beta distribution of the
        mixup coefficients.
      mixup_seed: The seed of the mixup coefficient sequence.
      name: The model name.
    '''
    super().__init__(name=name)
    self.backbone = backbone
    self.criterion = criterion
    self.apply_mixup = apply_mixup
    self.mixup_seed = mixup_seed
    self.mixup = Mixup(mixup_alpha, random_seed=mixup_seed)
    self.loss_metrics = {
        key: tf.keras.metrics.Mean(name=f'loss_{key}')
        for key in getattr(criterion, 'loss_keys', [])
    }
    self.validation_metrics: dict = {}
    '''The validation metrics of the run, keyed by their name.'''
    self.last_components: dict = {}
    '''The loss components of the last batch, for the loggers.'''
    self.last_labels: dict = {}
    '''The labels of the last batch, mixed and halved if the mixup ran.'''
    self.last_output: dict = {}
    '''The output of the last evaluation step.'''

  def add_validation_metric(self, name: str, metric) -> None:
    '''Adds a validation metric to the model.

    The metrics are **not** compiled: the model returns a dict of two
    heads, and Keras insists that the compiled metrics have the same
    structure as the output, which would tie every metric to one head.
    The metrics are owned by the model instead, and they are updated by
    the evaluation step.

    Args:
      name: The name of the metric.
      metric: The metric.
    '''
    self.validation_metrics[name] = metric
    self._metrics.append(metric)

  def get_config(self) -> dict:
    '''Returns the serialisable configuration of the model.

    Returns:
      The configuration dictionary.
    '''
    config = super().get_config()
    config.update({
        'backbone': tf.keras.utils.serialize_keras_object(self.backbone),
        'criterion': (tf.keras.utils.serialize_keras_object(self.criterion)
                       if self.criterion is not None else None),
        'apply_mixup': self.apply_mixup,
        'mixup_alpha': self.mixup.mixup_alpha,
        'mixup_seed': self.mixup_seed,
    })
    return config

  @classmethod
  def from_config(cls, config):
    '''Rebuilds a model from its configuration.

    Args:
      config: The configuration dictionary.

    Returns:
      The model.
    '''
    config = dict(config)
    fields = ('backbone', 'criterion', 'apply_mixup', 'mixup_alpha',
              'mixup_seed', 'name')
    kwargs = {key: config[key] for key in fields if key in config}
    kwargs['backbone'] = tf.keras.utils.deserialize_keras_object(
        kwargs['backbone'])
    criterion = kwargs.get('criterion')
    kwargs['criterion'] = (tf.keras.utils.deserialize_keras_object(criterion)
                            if criterion is not None else None)
    return cls(**kwargs)

  def call(self, inputs, training: bool = False):
    '''Runs the backbone.

    Args:
      inputs: The waveform, or the tuple ``(waveform, mixup_lambda)``.
      training: Whether the model is training.

    Returns:
      The dict of the two heads.
    '''
    return self.backbone(inputs, training=training)

  @staticmethod
  def _waveforms(x):
    '''Extracts the waveform of a batch.

    Args:
      x: The model input, a waveform tensor or a dict holding one.

    Returns:
      The waveform tensor.
    '''
    if isinstance(x, dict):
      return x['waveforms']
    return x

  def _draw_mixup_lambda(self, count: int):
    '''Draws the mixup coefficients of one batch.

    One ``Beta(alpha, alpha)`` is drawn per pair and emitted as
    ``[lam, 1 - lam]``, so adjacent coefficients sum to one, exactly as
    the original's ``Mixup`` did.

    Args:
      count: The number of rows of the batch.

    Returns:
      A float32 tensor of ``count`` coefficients.
    '''
    lambdas = []
    for _ in range(0, count, 2):
      lam = self.mixup.random_state.beta(self.mixup.mixup_alpha,
                                         self.mixup.mixup_alpha, 1)[0]
      lambdas.append(lam)
      lambdas.append(1.0 - lam)
    return tf.constant(lambdas, dtype=tf.float32)

  def _mix_labels(self, y, mixup_lambda):
    '''Mixes and halves the label tensors.

    Args:
      y: The label dict.
      mixup_lambda: The mixup coefficients.

    Returns:
      The mixed and halved label dict.
    '''
    return {key: self.backbone.mix_labels(value, mixup_lambda)
            for key, value in y.items()}

  def get_metrics_result(self) -> dict:
    '''Returns the results of the metrics that have been updated.

    Keras raises when a metric has never been updated, and the metrics
    of this model are the **validation** metrics: they are built by the
    first validation, which happens after the first epoch.  The metrics
    that are not built yet are simply not reported.

    Returns:
      The results of the built tf.keras.metrics.
    '''
    results = {}
    for metric in self.metrics:
      try:
        result = metric.result()
      except ValueError:
        continue
      if isinstance(result, dict):
        results.update(result)
      else:
        results[metric.name] = result
    return results

  def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None,
                   training: bool = True):
    '''Computes the loss of one batch.

    Args:
      x: The model input.
      y: The label dict.
      y_pred: The model output dict.
      sample_weight: The sample weights.
      training: Whether the model is training.

    Returns:
      The scalar loss.

    Raises:
      ValueError: If the model has no criterion.
    '''
    if self.criterion is None:
      raise ValueError('The model has no criterion to compute a loss with.')
    return self.criterion.compute_loss(x=x, y=y, y_pred=y_pred,
                                        sample_weight=sample_weight,
                                        training=training)

  def update_loss_metrics(self) -> None:
    '''Feeds the criterion's components into the running means.'''
    components = getattr(self.criterion, 'components', {})
    for key, metric in self.loss_metrics.items():
      if key in components:
        metric.update_state(components[key])

  def reset_loss_metrics(self) -> None:
    '''Zeroes the running means of the loss keys.'''
    for metric in self.loss_metrics.values():
      metric.reset_state()

  def train_step(self, data):
    '''Runs one training step.

    Args:
      data: The ``(x, y)`` pair of the batch.

    Returns:
      A dict with the loss of the batch.  Only scalars are returned:
      Keras aggregates the value a training step returns over the batch,
      and the prediction and the file names are not aggregatable.  The
      loss components are in ``last_components``.
    '''
    x, y = data
    waveforms = self._waveforms(x)
    shape = tf.shape(waveforms)
    mixup_lambda = None
    if self.apply_mixup and getattr(self.backbone, 'apply_aug', False):
      mixup_lambda = self._draw_mixup_lambda(int(shape[0]) * int(shape[1]))
      y = self._mix_labels(y, mixup_lambda)
    with tf.GradientTape() as tape:
      y_pred = self.backbone((waveforms, mixup_lambda), training=True)
      loss = self.compute_loss(x, y, y_pred, training=True)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply(gradients, self.trainable_variables)
    if self._loss_tracker is not None:
      # The running average of the epoch's training loss.
      self._loss_tracker.update_state(loss)
    self.update_loss_metrics()
    self.last_components = dict(getattr(self.criterion, 'components', {}))
    self.last_labels = y
    return {'loss': loss}

  def test_step(self, data):
    '''Runs one evaluation step.

    Args:
      data: The ``(x, y)`` pair of the batch.

    Returns:
      The same dict the torch engine's ``output_transform`` returns, plus
      the value of every validation metric.
    '''
    x, y = data
    waveforms = self._waveforms(x)
    y_pred = self.backbone(waveforms, training=False)
    loss = self.compute_loss(x, y, y_pred, training=False)
    self.update_loss_metrics()
    # The validation metrics are the ones the original computed in its
    # evaluator.
    result = {
        'loss': loss,
        'x': x,
        'y_pred': y_pred,
        'y': y,
        'dict_loss': getattr(self.criterion, 'components', {}),
    }
    for name, metric in self.validation_metrics.items():
      metric.update_state(y, y_pred)
      result[name] = metric.result()
    self.last_output = result
    return result

  def make_test_function(self, force: bool = False):
    '''Builds the evaluation function.

    The evaluation step returns the dict the torch engine returns, and
    Keras aggregates the values of an evaluation log: only the scalars
    may go into it, or the progress bar tries to average the file
    names.

    Args:
      force: Whether the function is rebuilt.

    Returns:
      The evaluation function.
    '''
    if self.test_function is not None and not force:
      return self.test_function
    super().make_test_function(force=force)
    inner = self.test_function

    def scalars(*args, **kwargs):
      '''Keeps the scalar entries of the evaluation log.

      Args:
        *args: The arguments of the evaluation step.
        **kwargs: The keyword arguments of the evaluation step.

      Returns:
        The scalar entries of the evaluation log.
      '''
      out = inner(*args, **kwargs)
      return {key: value for key, value in out.items()
              if _is_scalar(value)}

    self.test_function = scalars
    return self.test_function
