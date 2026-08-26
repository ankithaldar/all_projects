#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streaming reader decoding K sampled slices per MRI series.

The reader consumes precomputed index records (ordered SOP lists produced by
notebook 02 / ``main.py build-index``) and never touches pydicom ordering
logic at train time. Decoding failures fall back to zeros frames sized from
the index, matching the BLUEPRINT robustness requirement.
"""

from __future__ import annotations

import os

import numpy as np

from knee.helpers import intensity
from knee.helpers.dicom_io import DecoderRegistry
from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

DEFAULT_FALLBACK_SHAPE = (512, 512)


class SeriesReader:
    """Decode, normalize, and sample an evenly spaced 2.5D slice subset."""

    def __init__(
        self,
        dicom_root: str,
        registry: DecoderRegistry,
        n_slices: int,
        percentiles: tuple[float, float],
        autocrop_margin: float,
    ) -> None:
        """Configure the reader.

        Args:
            dicom_root: Root directory containing ``<study>/<series>/*.dcm``.
            registry: Decoder backend chain from configuration.
            n_slices: Number of evenly spaced slices returned per series.
            percentiles: Robust normalization bounds (see helpers.intensity).
            autocrop_margin: Background margin fraction kept around foreground.
        """
        self.dicom_root = dicom_root
        self.registry = registry
        self.n_slices = n_slices
        self.percentiles = percentiles
        self.autocrop_margin = autocrop_margin

    def _decode_stack(self, record: dict) -> tuple[np.ndarray, tuple[int, int]]:
        """Read the sampled slices into a raw float stack.

        Args:
            record: Index row mapping with keys ``study``, ``series``,
                ``sop_uids`` (ordered list), ``rows`` and ``cols``.

        Returns:
            Tuple of ``(stack (K, H, W) float32, fallback_shape)`` where
            unreadable frames contribute zero arrays.
        """
        sop_uids: list[str] = list(record['sop_uids'])
        indices = np.linspace(0, len(sop_uids) - 1, self.n_slices, dtype=int)
        fallback_shape = (
            int(record.get('rows') or 0),
            int(record.get('cols') or 0),
        )
        if fallback_shape[0] <= 0 or fallback_shape[1] <= 0:
            fallback_shape = DEFAULT_FALLBACK_SHAPE
        series_dir = os.path.join(self.dicom_root, str(record['study']), str(record['series']))
        frames = []
        for idx in indices:
            path = os.path.join(series_dir, f'{sop_uids[idx]}.dcm')
            pixels, info = self.registry.read_slice(path)
            if pixels is None:
                _LOGGER.debug('zeros frame substituted for %s (%s)', path, info['errors'])
                frames.append(np.zeros(fallback_shape, dtype=np.float32))
            else:
                frames.append(pixels.astype(np.float32))
        return np.stack(frames, axis=0), fallback_shape

    def read(self, record: dict) -> np.ndarray:
        """Produce the model-ready uint8 stack for one series.

        Pipeline per series (not per slice): rescale happens inside the
        registry, then percentile normalization over the sampled volume,
        cropping driven by the central frame's bounding box, uint8 cast.

        Args:
            record: Index row as consumed by :meth:`_decode_stack`.

        Returns:
            ``(n_slices, height, width)`` uint8 array resized upstream by
            the augmentation pipeline.
        """
        stack, fallback_shape = self._decode_stack(record)
        normalized = intensity.normalize_percentile(stack, self.percentiles)
        center = normalized[len(normalized) // 2]
        _, (y0, y1, x0, x1) = intensity.autocrop(center, self.autocrop_margin)
        cropped = normalized[:, y0:y1, x0:x1]
        del fallback_shape  # shape hint only needed for failed decodes above
        return intensity.to_uint8(cropped)
