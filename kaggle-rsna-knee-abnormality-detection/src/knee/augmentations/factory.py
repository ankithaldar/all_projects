#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Albumentations pipeline construction from YAML specifications."""

from __future__ import annotations

from knee.config_params.loader import instantiate


def build_compose(
    specs: list[dict],
    img_size: int,
    normalize_output: dict,
) -> object:
    """Compose a complete per-slice transform pipeline.

    The returned chain always begins with a deterministic resize (contract
    between dataset and model) and always ends with float conversion,
    standardization, and tensor conversion, so experiment YAMLs only declare
    the *random* middle section.

    Args:
        specs: List of class_path/init_params specs (may be empty).
        img_size: Square resize target applied first.
        normalize_output: Mapping with ``mean`` and ``std`` channel lists
            applied after scaling pixels to [0, 1].

    Returns:
        ``albumentations.Compose`` instance ready for ``image=`` calls whose
        output['image'] is a ``(3, H, W)`` torch tensor.
    """
    import albumentations as album
    from albumentations.pytorch import ToTensorV2

    steps = [
        album.Resize(height=img_size, width=img_size, interpolation=1),
        *instantiate(specs or []),
        album.ToFloat(max_value=255.0),
        album.Normalize(
            mean=normalize_output['mean'],
            std=normalize_output['std'],
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
    ]
    return album.Compose(steps)
