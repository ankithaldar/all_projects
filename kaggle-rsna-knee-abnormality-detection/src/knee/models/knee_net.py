#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KneeNet: full study-level multi-target classifier.

Data flow (BLUEPRINT Section 4):

    slices -> SeriesEncoder -> series tokens -> StudyAggregator -> study token
    metadata --------------------------------> FiLM conditioning -------> trunk -> 12 logits
"""

from __future__ import annotations

from torch import Tensor, nn

from knee.layers.pooling import FiLM, StudyAggregator
from knee.models.backbones import TimmBackbone
from knee.models.series_encoder import SeriesEncoder


class KneeNet(nn.Module):
    """Hierarchical 2.5D multimodal classifier for the 12 knee targets."""

    def __init__(
        self,
        backbone_name: str,
        img_size: int,
        n_slices: int,
        n_series_tokens_max: int,
        n_targets: int,
        metadata_dim: int,
        slice_pool: nn.Module,
        study_aggregator: nn.Module,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        trunk_hidden: int = 512,
        pretrained: bool = True,
        pretrained_cfg: dict | None = None,
    ) -> None:
        """Assemble the network from configuration-driven components.

        Args:
            backbone_name: timm model name.
            img_size: Square input resolution per slice.
            n_slices: Slices sampled per series (dataset contract mirror).
            n_series_tokens_max: Padded series-token capacity.
            n_targets: Number of sigmoid outputs (12).
            metadata_dim: Width of the study metadata vector.
            slice_pool: AttentionPool2d instance from YAML.
            study_aggregator: StudyAggregator instance from YAML.
            dropout: Dropout of the classification trunk.
            drop_path_rate: Backbone stochastic-depth rate.
            trunk_hidden: Hidden width before the output head.
            pretrained: Use timm-pretrained weights when no file given.
            pretrained_cfg: Optional offline checkpoint descriptor.

        Raises:
            ValueError: When pooling dims disagree with the backbone output.
        """
        super().__init__()
        self.n_slices = n_slices
        self.n_series_tokens_max = n_series_tokens_max
        backbone = TimmBackbone(
            backbone_name=backbone_name,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
        )
        embed_dim = backbone.embed_dim
        if study_aggregator.cross_attention.embed_dim != embed_dim:
            raise ValueError(
                'study_aggregator.embed_dim must equal backbone embed dim '
                f'({study_aggregator.cross_attention.embed_dim} != {embed_dim})'
            )
        self.series_encoder = SeriesEncoder(backbone, slice_pool)
        self.aggregator = study_aggregator
        self.film = FiLM(metadata_dim=metadata_dim, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, trunk_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, n_targets),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute target logits for a collated study batch.

        Args:
            batch: Dictionary with keys ``slices``
                ``(batch * max_series * n_slices, 3, H, W)``, ``slice_counts``
                ``(batch, max_series)``, ``series_meta``
                ``(batch, max_series, series_meta_dim)``, and ``metadata``
                ``(batch, metadata_dim)``.

        Returns:
            ``(batch, n_targets)`` raw logits (apply sigmoid externally).
        """
        series_tokens, series_mask = self.series_encoder(
            batch['slices'], batch['slice_counts'], batch['series_meta']
        )
        study_embedding = self.aggregator(series_tokens, series_mask)
        conditioned = self.film(study_embedding, batch['metadata'])
        return self.head(conditioned)

    def parameter_groups(self, backbone_lr_scale: float) -> list[dict]:
        """Split parameters into head/backbone groups for differential LRs.

        Args:
            backbone_lr_scale: Multiplier applied to the optimizer's base lr
                for backbone weights.

        Returns:
            List of ``{'params', 'name'}`` dictionaries consumed by engines.
        """
        return [
            {'params': self.head.parameters(), 'lr_scale_tag': 'head'},
            {
                'params': self.series_encoder.backbone.parameters(),
                'lr_scale_tag': f'backbone_x{backbone_lr_scale}',
            },
        ]
