#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DICOM pixel decoding with a pluggable backend registry.

The competition mixes four transfer syntaxes (Explicit/Implicit VR Little
Endian, JPEG Lossless, JPEG 2000). ``python-gdcm`` and the ``pylibjpeg-*``
plugins self-register with pydicom on import, so each backend strategy here
only needs to guarantee its library is imported before decoding.

Design: Strategy pattern via :class:`DecoderRegistry`; callers pass a
backend priority order that lives in ``configs/data.yaml``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pydicom

from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

DecodeFn = Callable[[pydicom.dataset.Dataset], np.ndarray]

# pydicom raises assorted exception types across versions and codecs;
# any failure during decode marks the slice unreadable for this backend.
_DECODE_ERRORS = (Exception,)


def _decode_native(dataset: pydicom.dataset.Dataset) -> np.ndarray:
    """Decode using whichever handlers are already registered.

    Args:
        dataset: Dataset opened with pixel data available.

    Returns:
        Decoded pixel array.

    Raises:
        Exception: Propagated when the transfer syntax has no handler.
    """
    return dataset.pixel_array


def _import_optional(module_name: str) -> bool:
    """Import an optional decoder plugin, reporting success.

    Args:
        module_name: Importable module name such as ``gdcm``.

    Returns:
        True when the import succeeded.
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        _LOGGER.warning('Optional DICOM backend %r unavailable', module_name)
        return False


def _make_plugin_backend(module_name: str) -> DecodeFn:
    """Create a decode function bound to an optional plugin import.

    Args:
        module_name: Plugin module whose import registers pydicom handlers.

    Returns:
        Decode function raising ``ImportError`` when the plugin is missing.
    """

    def decode(dataset: pydicom.dataset.Dataset) -> np.ndarray:
        """Decode after ensuring the plugin is imported.

        Args:
            dataset: Dataset opened with pixel data available.

        Returns:
            Decoded pixel array.
        """
        if not _import_optional(module_name):
            raise ImportError(f'Backend module {module_name!r} not installed')
        return dataset.pixel_array

    return decode


class DecoderRegistry:
    """Registry of named pixel-decoding strategies tried in priority order."""

    def __init__(self, backend_order: list[str]) -> None:
        """Initialize the registry with a caller-defined priority list.

        Args:
            backend_order: Backend names from configuration, e.g.
                ``['native', 'gdcm', 'pylibjpeg']``.
        """
        self._backends: dict[str, DecodeFn] = {
            'native': _decode_native,
            'gdcm': _make_plugin_backend('gdcm'),
            'pylibjpeg': _make_plugin_backend('pylibjpeg'),
        }
        self.backend_order = [name for name in backend_order if name in self._backends]
        unknown = set(backend_order) - set(self._backends)
        if unknown:
            _LOGGER.warning('Unknown decode backends ignored: %s', sorted(unknown))
        if not self.backend_order:
            self.backend_order = ['native']

    def register(self, name: str, decode_fn: DecodeFn) -> None:
        """Register or replace a named backend (open/closed extension point).

        Args:
            name: Backend identifier referenced by configuration.
            decode_fn: Callable mapping a Dataset to a pixel array.
        """
        self._backends[name] = decode_fn

    def decode_dataset(self, dataset: pydicom.dataset.Dataset) -> np.ndarray:
        """Decode an already-opened Dataset trying backends in order.

        Args:
            dataset: Dataset with pixel data element present.

        Returns:
            Float32 pixel array.

        Raises:
            ValueError: When every configured backend fails.
        """
        errors = []
        for name in self.backend_order:
            try:
                pixels = self._backends[name](dataset)
                return np.asarray(pixels, dtype=np.float32)
            except _DECODE_ERRORS as exc:  # noqa: BLE001 - deliberate catch-all
                errors.append(f'{name}: {type(exc).__name__}')
        raise ValueError(f'All decode backends failed -> {"; ".join(errors)}')

    def read_slice(self, path: str) -> tuple[np.ndarray | None, dict]:
        """Read one DICOM file into a normalized float32 array plus metadata.

        Args:
            path: Filesystem path to a single-slice ``.dcm`` file.

        Returns:
            Tuple of ``(pixels_or_None, info)``. ``pixels`` is None when every
            backend failed; ``info`` always carries rescale parameters and the
            error trail so callers can fall back to a zeros frame with logging.
        """
        try:
            dataset = pydicom.dcmread(path)
            pixels = self.decode_dataset(dataset)
            slope = float(dataset.get('RescaleSlope', 1) or 1)
            intercept = float(dataset.get('RescaleIntercept', 0) or 0)
            return pixels * slope + intercept, {'path': path, 'errors': []}
        except _DECODE_ERRORS as exc:  # noqa: BLE001 - deliberate catch-all
            _LOGGER.warning('Unreadable slice %s (%s: %s)', path, type(exc).__name__, exc)
            return None, {'path': path, 'errors': [f'{type(exc).__name__}: {exc}']}
