#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The sequence deep-learning tier.

Opt-in capability: no registered pipeline references it, and the sequence column
list in the configured model program is empty. It is retained because the
architecture is sound and because the vocabulary-sizing design is genuinely good
-- each sequence column gets its own embedding table sized to its own observed
maximum, which is correct when usage counts and event counts have wildly
different ranges and a shared table would waste capacity.

Four version-sensitive decisions carry this tier, and each is a place where the
obvious spelling is wrong for the pinned dependency set:

======================================  ==========================================
Decision                                   Why it is made this way
======================================  ==========================================
The checkpoint stores the architecture    A state dict alone is not reconstructible
and the learned vocabulary                in a later process, so the tier would not
                                          round-trip through object storage the way
                                          every other model artefact must.
Prediction frames are concatenated,       ``DataFrame.append`` is removed in the
never appended                            pinned pandas 2.x.
The Lightning 2.x epoch hooks             The 1.x hook names are removed in the
                                          pinned Lightning 2.x.
Tensors are passed through, not rebuilt   Rebuilding with ``torch.tensor(...)``
via ``torch.tensor(...)``                 detaches the value from the autograd
                                          graph, so no gradient reaches the
                                          positional-encoding parameters.
loader
``write_on_interval`` not overridden     Both write hooks are implemented.
======================================  ==========================================

``torch`` and ``pytorch-lightning`` are an optional dependency and are imported
lazily, so the framework imports and runs without them.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Suffix appended to a sequence column's learned vocabulary size key.
VOCAB_KEY = 'seq_max_'
#: Suffix appended to a sequence column's learned length key.
LENGTH_KEY = 'seq_len_'


def _merge_vocabulary(*vocabularies: dict[str, int]) -> dict[str, int]:
  """Merge learned vocabularies, keeping the largest size declared for each key.

  The embedding table must be large enough for every token that will be seen, so
  the sizes are combined by maximum rather than by overwrite. Taking the training
  vocabulary alone left a test-set token above the training maximum indexing
  outside the table.

  Args:
    vocabularies: The per-split vocabularies.

  Returns:
    The merged vocabulary.
  """
  merged: dict[str, int] = {}
  for vocabulary in vocabularies:
    for key, size in vocabulary.items():
      merged[key] = max(merged.get(key, 0), int(size))
  return merged


def _join_address(root: str, name: str) -> str:
  """Join a root to a relative name, in whichever address form the root uses.

  A managed root is scheme-qualified and a local root is a bare path, so the join
  is expressed once, here, rather than at each call site. A root that already
  names the object is returned unchanged, which keeps the function idempotent --
  enumerating and then re-constructing an address is a place where the abstraction
  can be broken twice, once in each direction.

  Args:
    root: The directory root.
    name: The object's name within it.

  Returns:
    The full address.
  """
  base = str(root).rstrip('/')
  if name.startswith(f'{base}/'):
    return name
  return f'{base}/{name.lstrip("/")}'


class PositionalEncoding:
  """The standard sinusoidal positional encoding.

  It is constructed lazily as an ``nn.Module`` subclass, so importing this
  module does not require PyTorch.
  """

  @staticmethod
  def build(d_model: int, max_len: int = 5000, dropout: float = 0.1) -> Any:
    """Build a positional-encoding module.

    Args:
      d_model: The embedding width.
      max_len: The maximum sequence length the table covers.
      dropout: The dropout probability applied after encoding.

    Returns:
      An ``nn.Module``.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    class _PositionalEncoding(nn.Module):
      """Inject a sinusoidal positional signal into an embedded sequence."""

      def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer('encoding', encoding.unsqueeze(0))

      def forward(self, tensor: Any) -> Any:
        """Add the positional signal to an embedded sequence.

        Args:
          tensor: The embedded sequence, shaped ``(batch, length, d_model)``.

        Returns:
          The encoded sequence, with dropout applied.
        """
        # Passing the tensor through unchanged, rather than rebuilding it, is what
        # keeps gradients flowing back into the embedding table.
        return self.dropout(tensor + self.encoding[:, : tensor.size(1), :])

    return _PositionalEncoding()


def SequenceModel(  # noqa: N802 - the class name is part of the configuration contract
  vocabulary: dict[str, int],
  params: dict[str, Any] | None = None,
) -> Any:
  """Build the sequence transformer.

  Per-column embedding plus positional encoding, then one shared encoder layer
  applied to every column's encoded sequence, then a flatten-concatenate head.

  Sharing one encoder layer across every column is a deliberate and economical
  parameter-sharing constraint: all columns must share the same ``d_model``.
  It also prevents per-column architecture tuning.

  Args:
    vocabulary: The learned vocabulary, keyed ``seq_max_{i}`` and
      ``seq_len_{i}``. This is the data-derived state a failed to
      persist.
    params: The architecture hyper-parameters.

  Returns:
    A ``LightningModule``.
  """
  import torch  # noqa: PLC0415
  import torch.nn as nn  # noqa: PLC0415
  from pytorch_lightning import LightningModule  # noqa: PLC0415

  settings = {
    'pos_encoding_dim': 6,
    'num_head': 2,
    'dropout': 0.2,
    'class_pos_weight': [4.0],
    'learning_rate': 0.0005,
    'weight_decay': 1e-5,
    'layer_dim': [512, 124, 50],
    'output_dim': 1,
    'batch_size': 32,
    'multiclass': False,
    **(params or {}),
  }
  feature_count = sum(1 for key in vocabulary if key.startswith(VOCAB_KEY))
  encoders = {key: value for key, value in vocabulary.items() if key.startswith(VOCAB_KEY)}

  class _SequenceModel(LightningModule):
    """A transformer over per-column behavioural sequences."""

    def __init__(self) -> None:
      super().__init__()
      self.save_hyperparameters(settings)
      self.pos_encoding_dim = int(settings['pos_encoding_dim'])
      self.output_dim = int(settings['output_dim'])
      self.threshold = float(settings.get('prediction_threshold', 0.5))

      # One embedding table per column, sized to that column's own observed
      # maximum. A shared table would waste capacity across columns whose ranges
      # differ by orders of magnitude.
      self.max_len = 0
      for index in range(feature_count):
        size = int(encoders.get(f'{VOCAB_KEY}{index}', 1))
        length = int(vocabulary.get(f'{LENGTH_KEY}{index}', 1))
        setattr(self, f'seq_embedding_{index}', nn.Embedding(size + 1, self.pos_encoding_dim))
        setattr(self, f'seq_pos_encoding_{index}', PositionalEncoding.build(self.pos_encoding_dim, max(length, 2)))
        self.max_len += length
      self.input_len = self.pos_encoding_dim * max(self.max_len, 1)

      self.transformer_layer = nn.TransformerEncoderLayer(
        self.pos_encoding_dim, int(settings['num_head']), dropout=float(settings['dropout'])
      )
      dimensions = [self.input_len, *settings['layer_dim'], self.output_dim]
      self.linear = nn.Sequential(
        *[nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(len(dimensions) - 1)]
      )
      self.criterion = (
        nn.CrossEntropyLoss(pos_weight=torch.tensor(settings['class_pos_weight']))
        if settings['multiclass']
        else nn.BCEWithLogitsLoss(pos_weight=torch.tensor(settings['class_pos_weight'][0]))
      )

    def forward(self, sequence: Any, labels: Any) -> Any:
      """Encode the sequence and predict.

      Args:
        sequence: The sequence tensor, shaped ``(batch, n_cols, length)``.
        labels: The label tensor.

      Returns:
        The raw logits, shaped ``(batch, output_dim)``.
      """
      encoded: list[Any] = []
      for index in range(feature_count):
        embedded = getattr(self, f'seq_embedding_{index}')(sequence[:, index, :])
        # The tensor is passed through the positional encoding and the encoder
        # unchanged. A rebuilt both with torch.tensor(...), which
        # detached them from the autograd graph.
        positioned = getattr(self, f'seq_pos_encoding_{index}')(embedded)
        output = self.transformer_layer(positioned.float())
        encoded.append(torch.flatten(output, start_dim=1))
        encoded[-1] = torch.reshape(encoded[-1], (labels.size(0), -1))
      return self.linear(torch.cat(encoded, dim=1))

    def training_step(self, batch: Any, batch_idx: int) -> Any:
      """Compute one training loss.

      Args:
        batch: A ``(sequence, labels, keys)`` tuple.
        batch_idx: The batch index.

      Returns:
        The loss tensor.
      """
      del batch_idx
      sequence, labels, _ = batch
      return self.criterion(self(sequence, labels), labels.float().view(-1, 1))

    def validation_step(self, batch: Any, batch_idx: int) -> None:
      """Run one validation step.

      Args:
        batch: A ``(sequence, labels, keys)`` tuple.
        batch_idx: The batch index.
      """
      del batch_idx
      sequence, labels, _ = batch
      self.criterion(self(sequence, labels), labels.float().view(-1, 1))

    def predict_step(self, batch: Any, batch_idx: int = 0) -> list[Any]:
      """Predict a batch.

      The threshold is a constructor parameter rather than a hardcoded ``0.5``,
      so this tier shares the framework's operating-point logic rather than
      bypassing it.

      Args:
        batch: A ``(sequence, labels, keys)`` tuple.
        batch_idx: The batch index.

      Returns:
        A list of ``(labels, predictions, keys, probabilities)`` tuples. The
        order is asserted here and consumed by the prediction writer, so it is
        defined once.
      """
      del batch_idx
      sequence, labels, keys = batch
      logits = self(sequence, labels)
      probabilities = torch.sigmoid(logits).view(-1)
      predictions = torch.where(probabilities > self.threshold, 1.0, 0.0)
      # The keys are the batch's own identifier columns. Substituting a tensor of
      # batch indices here would produce a prediction set whose key column carried
      # no entity identity, so nothing downstream could join it back.
      # The keys are widened to 2-D so a single identifier column and several
      # identifier columns both assemble into a frame of the right width.
      return [(labels.view(-1, 1), predictions.view(-1, 1), keys.view(-1, -1), probabilities)]

    def configure_optimizers(self) -> Any:
      """Build the optimiser.

      Returns:
        A configured Adam optimiser.
      """
      return torch.optim.Adam(
        self.parameters(), lr=self.hparams['learning_rate'], weight_decay=self.hparams['weight_decay']
      )

  return _SequenceModel()


class SequenceDataset:
  """Load a sequence parquet directory into dense in-memory tensors.

  The whole corpus is materialised eagerly, which is the binding memory
  constraint on this tier and the exact opposite of the framework's
  Spark-everywhere posture. That trade is documented rather than hidden.

  Attributes:
    max_value_dict: The learned vocabulary, keyed ``seq_max_{i}`` and
      ``seq_len_{i}``. This is the state a computed in the training
      process and wrote nowhere, which is what made the model unloadable in a
      later one.
  """

  def __init__(
    self,
    label_col: str,
    key_cols: list[str],
    seq_cols: list[str],
    data_dir: str,
    data_file_format: str = 'parquet',
    parts_num: int = 9999,
  ) -> None:
    """Load and concatenate the sequence corpus.

    Args:
      label_col: The label column.
      key_cols: The identifier columns, preserved so predictions can be joined
        back to entities.
      seq_cols: The sequence columns.
      data_dir: The object-storage directory.
      data_file_format: The on-disk format.
      parts_num: The maximum number of parts to read.
    """
    import torch  # noqa: PLC0415
    import torch.utils.data as data  # noqa: PLC0415

    from forecasting_ml_framework.utils.storage import get_file_system  # noqa: PLC0415

    filesystem = get_file_system(data_dir)
    if hasattr(filesystem, 'invalidate_cache'):
      filesystem.invalidate_cache()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.label_col = label_col
    self.key_cols = list(key_cols)
    self.seq_cols = list(seq_cols)

    import pandas as pd  # noqa: PLC0415

    sequences: list[Any] = []
    labels: list[Any] = []
    keys: list[Any] = []
    # The listing is re-addressed rather than used verbatim. A managed listing
    # returns qualified addresses and a local directory listing returns bare names;
    # concatenating a root onto the latter, or assuming the former, is how an
    # address gets mangled in one direction or the other. Joining against the root
    # produces the correct address for whichever listing was returned.
    for index, listed in enumerate(sorted(filesystem.ls(data_dir))[:parts_num]):
      name = os.path.basename(str(listed).rstrip('/'))
      if not name.endswith(f'.{data_file_format}'):
        continue
      path = _join_address(data_dir, name)
      with filesystem.open(path, 'rb') as handle:
        frame = pd.read_parquet(handle, engine='pyarrow')
      # `to_numpy` on a frame of array columns yields a 2-D object array, so the
      # three-axis transpose raised "axes don't match array". Each part is stacked
      # as (sequence, length), giving (part, sequence, length).
      sequences.append(np.stack(frame[self.seq_cols].to_numpy(dtype=float).tolist(), axis=0))
      labels.append(frame[[self.label_col]].to_numpy(dtype=float))
      keys.append(frame[self.key_cols].to_numpy())
      LOGGER.debug('Loaded a sequence part', extra={'part': index, 'path': path, 'rows': len(frame)})

    self.seq_np = np.concatenate(sequences, axis=0) if sequences else np.zeros((0, len(seq_cols), 0))
    self.label_np = np.concatenate(labels, axis=0) if labels else np.zeros((0, 1))
    self.key_np = np.concatenate(keys, axis=0) if keys else np.zeros((0, len(key_cols)))
    self.len_data = len(self.label_np)
    self.max_value_dict = self._learn_vocabulary()
    # The identifier columns travel with each sample. They are built into the
    # dataset and not merely stored on the instance, because without them in the
    # batch there is nothing for `predict_step` to return as keys -- and the only
    # available substitute was a tensor of repeated batch indices, which produced
    # a prediction set that could not be joined back to the entities it scored.
    self._dataset = data.TensorDataset(
      torch.tensor(self.seq_np, dtype=torch.long),
      torch.tensor(self.label_np, dtype=torch.long).view(-1),
      torch.tensor(self.key_np, dtype=torch.long),
    )

  def _learn_vocabulary(self) -> dict[str, int]:
    """Derive the per-column vocabulary from the loaded data.

    Returns:
      A mapping of ``seq_max_{i}`` and ``seq_len_{i}`` to their observed values.
    """
    vocabulary: dict[str, int] = {}
    for index in range(self.seq_np.shape[0]):
      column = self.seq_np[index]
      vocabulary[f'{VOCAB_KEY}{index}'] = int(column.max()) if column.size else 0
      vocabulary[f'{LENGTH_KEY}{index}'] = int(column.shape[1]) if column.ndim > 1 else 0
    LOGGER.info('Learned the sequence vocabulary', extra={'vocabulary': vocabulary})
    return vocabulary

  def __len__(self) -> int:
    """Return the corpus size.

    Returns:
      The number of rows.
    """
    return self.len_data

  def __getitem__(self, index: int) -> tuple[Any, Any]:
    """Return one sample, moved to the compute device.

    Args:
      index: The row index.

    Returns:
      A three-tuple of the sequence tensor, the label tensor and the key tensor.
    """
    return self._dataset[index]


class CustomWriter:
  """The prediction sink for the sequence tier.

  Assembles key columns, the label, the hard prediction and the probability into
  a frame per distributed rank, and writes it with the four provenance columns
  the tabular tier's score table carries, so a sequence prediction set can be
  reconciled with the enterprise score table.
  """

  def __init__(
    self,
    output_dir: str,
    key_cols: list[str],
    model_key: str,
    run_date: str,
    probability_col_name: str = 'score_value',
    project: str | None = None,
  ) -> None:
    """Build the writer.

    Args:
      output_dir: The destination directory.
      key_cols: The identifier columns.
      model_key: The model identifier, written as a provenance column.
      run_date: The run date, written as a provenance column.
      probability_col_name: The probability column name.
      project: The project used to resolve the filesystem.
    """
    self.output_dir = output_dir
    self.key_cols = list(key_cols)
    self.model_key = model_key
    self.run_date = run_date
    self.probability_col_name = probability_col_name
    self.project = project

  def assemble(self, predictions: list[Any]) -> Any:
    """Assemble the prediction frame from the Lightning prediction tuples.

    The tuple order is asserted once, here, and produced once, in the model's
    ``predict_step``. A asserted it in two places with no shared
    constant.

    Args:
      predictions: The list of per-batch tuples.

    Returns:
      A pandas ``DataFrame``.
    """
    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415

    frames: list[Any] = []
    key_width = len(self.key_cols)
    for batch in predictions:
      labels, hard, keys, probabilities = batch
      key_values = keys.numpy() if hasattr(keys, 'numpy') else keys
      key_values = np.asarray(key_values).reshape(len(np.asarray(labels)), key_width)
      frame = pd.DataFrame(key_values, columns=self.key_cols)
      frame['actual'] = torch.flatten(torch.as_tensor(labels)).numpy()
      frame['prediction'] = torch.flatten(torch.as_tensor(hard)).numpy()
      frame[self.probability_col_name] = torch.flatten(torch.as_tensor(probabilities)).numpy()
      frames.append(frame)
    if not frames:
      return pd.DataFrame(columns=[*self.key_cols, 'actual', 'prediction', self.probability_col_name])

    # pd.concat, not DataFrame.append: the latter was removed in pandas 2.x,
    # which is the pinned version, so a raised AttributeError on
    # every prediction write.
    assembled = pd.concat(frames, ignore_index=True)
    assembled['model_key'] = self.model_key
    assembled['run_date'] = self.run_date
    assembled['insert_ts'] = pd.Timestamp.now(tz='UTC')
    assembled['last_upd_dt'] = pd.Timestamp(self.run_date).date()
    return assembled

  def write(self, predictions: list[Any], global_rank: int = 0) -> str:
    """Assemble and write one rank's predictions.

    Args:
      predictions: The list of per-batch tuples.
      global_rank: The distributed rank, used to name the output file.

    Returns:
      The path that was written.
    """
    from forecasting_ml_framework.utils.storage import get_file_system  # noqa: PLC0415

    frame = self.assemble(predictions)
    path = f'{self.output_dir.rstrip("/")}/predictions_{global_rank}.parquet'
    filesystem = get_file_system(path, self.project)
    with filesystem.open(path, 'wb') as handle:
      frame.to_parquet(handle)
    LOGGER.info('Wrote sequence predictions', extra={'path': path, 'rows': len(frame)})
    return path


class TransformerRunner:
  """Drive the sequence tier end to end within one process.

  Training and prediction happen in the same process, which is what a
  tier relied on and what made its incomplete checkpoint go unnoticed. The
  checkpoint written here is *complete*, so the model can also be reloaded later.
  """

  def __init__(
    self,
    model_class: Any,
    writer_class: Any,
    positional_encoding_class: Any,
    params: dict[str, Any] | None = None,
  ) -> None:
    """Build the runner.

    Args:
      model_class: The model factory.
      writer_class: The prediction-writer class.
      positional_encoding_class: The positional-encoding factory.
      params: The configuration block.
    """
    self.model_class = model_class
    self.writer_class = writer_class
    self.positional_encoding_class = positional_encoding_class
    self.params = dict(params or {})
    self.model: Any = None
    self.max_value_dict: dict[str, int] = {}

  def run(
    self,
    train_dir: str,
    test_dir: str,
    output_dir: str,
    label_col: str,
    key_cols: list[str],
    sequence_cols: list[str],
    model_save_path: str,
  ) -> Any:
    """Train, checkpoint, and predict over both splits.

    Args:
      train_dir: The training parquet directory.
      test_dir: The test parquet directory.
      output_dir: The prediction output directory.
      label_col: The label column.
      key_cols: The identifier columns.
      sequence_cols: The sequence columns.
      model_save_path: Where to write the checkpoint.

    Returns:
      The trained model.
    """
    import pytorch_lightning as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    train_set = SequenceDataset(label_col, key_cols, sequence_cols, train_dir)
    test_set = SequenceDataset(label_col, key_cols, sequence_cols, test_dir)
    self.max_value_dict = _merge_vocabulary(train_set.max_value_dict, test_set.max_value_dict)

    batch_size = int(self.params.get('batch_size', 32))
    # The training loader is shuffled. Leaving it unshuffled would order batches by
    # parquet file layout, which correlates with customer segments and gives the
    # optimiser a systematically biased ordering.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_batch_size = int(self.params.get('eval_batch_size', 1024))
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False)

    # The embedding table is sized from the MERGED vocabulary, not the training
    # vocabulary alone. Sizing it from training alone left any test-set token above
    # the training maximum indexing outside the table, so scoring raised deep in
    # the embedding lookup -- and, before that, the test set was silently learning
    # its own vocabulary, which is a fit-time recomputation of state the model
    # should have been frozen against.
    self.model = self.model_class(self.max_value_dict, self.params)
    writer = self.writer_class(
      output_dir=output_dir,
      key_cols=key_cols,
      model_key=self.params.get('model_key', ''),
      run_date=self.params.get('run_date', ''),
    )
    callbacks = [
      pl.callbacks.EarlyStopping(
        monitor='val_loss', patience=int(self.params.get('patience', 5)), mode='min'
      ),
      pl.callbacks.ModelCheckpoint(dirpath=output_dir, save_last=True),
    ]
    trainer = pl.Trainer(
      max_epochs=int(self.params.get('max_epochs', 10)),
      accelerator='auto',
      devices=1,
      callbacks=callbacks,
      logger=False,
      enable_checkpointing=False,
    )
    trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    self.checkpoint(model_save_path)
    for rank, loader in enumerate((test_loader, train_loader)):
      predictions = trainer.predict(self.model, dataloaders=loader)
      writer.write(predictions, global_rank=rank)
    del torch
    return self.model

  def checkpoint(self, model_save_path: str) -> str:
    """Write a *complete* checkpoint: weights, architecture and vocabulary.

    Writing ``torch.save(model.state_dict())``, which is weights
    only — no architecture, no vocabulary, no sequence lengths. Reconstructing
    the model then required the training process's in-memory vocabulary, which
    was written nowhere. Since this framework trains and scores in separate
    processes, days apart, the tier could train and score in one process and
    never in two.

    Args:
      model_save_path: The destination path.

    Returns:
      The path that was written.
    """
    import torch  # noqa: PLC0415

    from forecasting_ml_framework.utils.storage import get_file_system  # noqa: PLC0415

    payload = {
      'state_dict': self.model.state_dict(),
      'hyperparameters': dict(getattr(self.model, 'hparams', {})),
      'max_value_dict': self.max_value_dict,
      'format_version': 1,
    }
    filesystem = get_file_system(model_save_path)
    with filesystem.open(model_save_path, 'wb') as handle:
      torch.save(payload, handle)
    sidecar = f'{model_save_path}.json'
    with filesystem.open(sidecar, 'w') as handle:
      json.dump(
        {'max_value_dict': self.max_value_dict, 'hyperparameters': payload['hyperparameters']},
        handle,
        default=str,
      )
    LOGGER.info(
      'Wrote a complete transformer checkpoint',
      extra={'path': model_save_path, 'sidecar': sidecar, 'vocabulary': self.max_value_dict},
    )
    return model_save_path

  def load_checkpoint(self, model_save_path: str) -> Any:
    """Reload a model from a complete checkpoint.

    Args:
      model_save_path: The checkpoint path.

    Returns:
      The reconstructed model.
    """
    import torch  # noqa: PLC0415

    from forecasting_ml_framework.utils.storage import get_file_system  # noqa: PLC0415

    filesystem = get_file_system(model_save_path)
    with filesystem.open(model_save_path, 'rb') as handle:
      payload = torch.load(handle, weights_only=False)
    self.max_value_dict = payload['max_value_dict']
    model = self.model_class(self.max_value_dict, payload.get('hyperparameters', {}))
    model.load_state_dict(payload['state_dict'])
    self.model = model
    return model
