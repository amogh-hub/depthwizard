from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from depthwizard.height_model.model import (
    BidirectionalGatedFusion,
    ConvNormAct,
    CrossScaleContext,
    DecoderBlock,
    Encoder,
    HeightModelConfig,
    MetadataConditioner,
    ResidualBlock,
)


@dataclass(frozen=True)
class TerrainStructureConfig:
    """Research configuration for explicit terrain/above-ground decomposition.

    The model remains in the relative-height coordinate system used by the geometry prior. Metric
    elevation is still recovered only by the existing evidence-calibration subsystem. The optional
    coarse-terrain prior, when supplied, must already be expressed in the same relative coordinate
    system; this module never converts arbitrary metric DEM values implicitly.
    """

    rgb_channels: tuple[int, int, int, int] = (32, 64, 128, 192)
    terrain_channels: tuple[int, int, int, int] = (16, 32, 64, 96)
    semantic_classes: int = 6
    height_bin_edges_m: tuple[float, ...] = (
        1.0,
        2.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        24.0,
        32.0,
        48.0,
        64.0,
    )
    dropout: float = 0.05
    max_terrain_relative_residual: float = 1.0
    initial_structure_probability: float = 0.10
    architecture_version: str = "terrain-structure-v1"

    def __post_init__(self) -> None:
        if len(self.rgb_channels) != 4 or len(self.terrain_channels) != 4:
            raise ValueError("rgb_channels and terrain_channels must each contain four stages")
        if any(channel <= 0 for channel in self.rgb_channels + self.terrain_channels):
            raise ValueError("all encoder channel widths must be positive")
        if self.semantic_classes < 2:
            raise ValueError("semantic_classes must be at least 2")
        if not self.height_bin_edges_m:
            raise ValueError("height_bin_edges_m must not be empty")
        if any(not math.isfinite(edge) or edge <= 0 for edge in self.height_bin_edges_m):
            raise ValueError("height-bin edges must be finite and positive")
        if any(
            later <= earlier
            for earlier, later in zip(
                self.height_bin_edges_m,
                self.height_bin_edges_m[1:],
                strict=False,
            )
        ):
            raise ValueError("height-bin edges must be strictly increasing")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.max_terrain_relative_residual):
            raise ValueError("max_terrain_relative_residual must be finite")
        if self.max_terrain_relative_residual <= 0:
            raise ValueError("max_terrain_relative_residual must be positive")
        if not 0.0 < self.initial_structure_probability < 0.5:
            raise ValueError("initial_structure_probability must be in (0, 0.5)")
        if self.architecture_version != "terrain-structure-v1":
            raise ValueError("unsupported terrain-structure architecture_version")

    @property
    def height_bins(self) -> int:
        return len(self.height_bin_edges_m) + 1


@dataclass(frozen=True)
class TerrainStructureOutput:
    """Explicit relative terrain and above-ground outputs.

    ``relative_height`` is the scientific recomposition ``terrain_relative + above_ground_relative``.
    The terrain and structure components remain inspectable so a roof-height improvement cannot hide
    a compensating ground-surface regression.
    """

    terrain_relative: torch.Tensor
    terrain_residual: torch.Tensor
    above_ground_relative: torch.Tensor
    relative_height: torch.Tensor
    structure_logits: torch.Tensor
    structure_probability: torch.Tensor
    semantic_logits: torch.Tensor
    height_bin_logits: torch.Tensor
    boundary_probability: torch.Tensor
    normals: torch.Tensor
    terrain_log_variance: torch.Tensor
    structure_log_variance: torch.Tensor
    terrain_uncertainty: torch.Tensor
    structure_uncertainty: torch.Tensor


class TerrainStructureModel(nn.Module):
    """Terrain--Structure Decomposition candidate for remote-sensing height reconstruction.

    This is deliberately a research candidate and is not wired into the frozen production policy.
    It addresses the exposed V6 failure mode by predicting two inspectable quantities instead of a
    single residual field:

    ``relative DSM = relative terrain + relative above-ground height``.

    A shared RGB/context hierarchy is fused bidirectionally with terrain evidence. The terrain head
    is decoded only to a coarse feature scale before bilinear reconstruction, discouraging it from
    learning roof texture as bare-earth relief. The structure head returns to native input resolution
    and predicts explicit building support, continuous above-ground height, ordinal height bins,
    boundaries, normals, semantics, and uncertainty.

    At initialization both terrain residual and above-ground height are exactly zero, so the composed
    relative field equals the supplied geometry prior. That identity is a safe optimization starting
    point, not a production guarantee or a claim that the prior is a bare-earth terrain estimate.
    """

    def __init__(self, config: TerrainStructureConfig | None = None) -> None:
        super().__init__()
        self.config = config or TerrainStructureConfig()
        rgb_channels = self.config.rgb_channels
        terrain_channels = self.config.terrain_channels

        self.rgb_encoder = Encoder(3, rgb_channels, self.config.dropout)
        # geometry prior + optional canonical coarse terrain + availability channel
        self.terrain_encoder = Encoder(3, terrain_channels, self.config.dropout)
        self.fusions = nn.ModuleList(
            [
                BidirectionalGatedFusion(rgb_c, terrain_c, self.config.dropout)
                for rgb_c, terrain_c in zip(rgb_channels, terrain_channels, strict=True)
            ]
        )

        self.cross_scale_2 = CrossScaleContext(
            rgb_channels[2], rgb_channels[3], self.config.dropout
        )
        self.cross_scale_1 = CrossScaleContext(
            rgb_channels[1], rgb_channels[2], self.config.dropout
        )
        self.cross_scale_0 = CrossScaleContext(
            rgb_channels[0], rgb_channels[1], self.config.dropout
        )
        self.metadata = MetadataConditioner(rgb_channels[-1])

        # Terrain is intentionally decoded only to stage-2 (roughly quarter-resolution after the
        # four-stage encoder) and is then reconstructed smoothly at the input grid.
        self.terrain_decode = DecoderBlock(
            rgb_channels[3], rgb_channels[2], rgb_channels[2], self.config.dropout
        )
        self.terrain_refine = nn.Sequential(
            ConvNormAct(rgb_channels[2], rgb_channels[2]),
            ResidualBlock(rgb_channels[2], dropout=self.config.dropout),
        )
        self.terrain_residual_head = nn.Conv2d(rgb_channels[2], 1, kernel_size=1)
        nn.init.zeros_(self.terrain_residual_head.weight)
        if self.terrain_residual_head.bias is not None:
            nn.init.zeros_(self.terrain_residual_head.bias)

        # Structure returns to native resolution and keeps a dedicated fine branch rather than
        # forcing building geometry through the low-frequency terrain pathway.
        self.structure_decode3 = DecoderBlock(
            rgb_channels[3], rgb_channels[2], rgb_channels[2], self.config.dropout
        )
        self.structure_decode2 = DecoderBlock(
            rgb_channels[2], rgb_channels[1], rgb_channels[1], self.config.dropout
        )
        self.structure_decode1 = DecoderBlock(
            rgb_channels[1], rgb_channels[0], rgb_channels[0], self.config.dropout
        )
        self.structure_native = nn.Sequential(
            ConvNormAct(rgb_channels[0], rgb_channels[0]),
            ResidualBlock(rgb_channels[0], dropout=self.config.dropout),
            ResidualBlock(rgb_channels[0], dropout=self.config.dropout),
        )

        head_channels = rgb_channels[0]
        self.structure_support_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        nn.init.zeros_(self.structure_support_head.weight)
        if self.structure_support_head.bias is not None:
            probability = self.config.initial_structure_probability
            self.structure_support_head.bias.data.fill_(
                math.log(probability / (1.0 - probability))
            )

        self.above_ground_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        nn.init.zeros_(self.above_ground_head.weight)
        if self.above_ground_head.bias is not None:
            nn.init.zeros_(self.above_ground_head.bias)

        self.semantic_head = nn.Conv2d(
            head_channels,
            self.config.semantic_classes,
            kernel_size=1,
        )
        self.height_bin_head = nn.Conv2d(
            head_channels,
            self.config.height_bins,
            kernel_size=1,
        )
        self.boundary_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.normal_head = nn.Conv2d(head_channels, 3, kernel_size=1)

        self.terrain_log_variance_head = nn.Conv2d(rgb_channels[2], 1, kernel_size=1)
        self.structure_log_variance_head = nn.Conv2d(head_channels, 1, kernel_size=1)

    @staticmethod
    def _validate_inputs(
        rgb: torch.Tensor,
        geometry_prior: torch.Tensor,
        coarse_terrain_prior: torch.Tensor | None,
    ) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape N x 3 x H x W")
        if geometry_prior.ndim != 4 or geometry_prior.shape[1] != 1:
            raise ValueError("geometry_prior must have shape N x 1 x H x W")
        if rgb.shape[0] != geometry_prior.shape[0] or rgb.shape[-2:] != geometry_prior.shape[-2:]:
            raise ValueError("rgb and geometry_prior must share batch and spatial dimensions")
        if not torch.all(torch.isfinite(geometry_prior)):
            raise ValueError("geometry_prior must be finite")
        if (
            coarse_terrain_prior is not None
            and coarse_terrain_prior.shape != geometry_prior.shape
        ):
            raise ValueError("coarse_terrain_prior must match geometry_prior")

    @staticmethod
    def _terrain_evidence(
        geometry_prior: torch.Tensor,
        coarse_terrain_prior: torch.Tensor | None,
    ) -> torch.Tensor:
        if coarse_terrain_prior is None:
            coarse = torch.zeros_like(geometry_prior)
            availability = torch.zeros_like(geometry_prior)
        else:
            finite = torch.isfinite(coarse_terrain_prior)
            coarse = torch.where(finite, coarse_terrain_prior, torch.zeros_like(coarse_terrain_prior))
            availability = finite.to(dtype=geometry_prior.dtype)
        return torch.cat([geometry_prior, coarse, availability], dim=1)

    @staticmethod
    def _nonnegative_zero_centered(raw: torch.Tensor) -> torch.Tensor:
        # softplus(raw) - softplus(0) is exactly zero for a zero-initialized head while preserving
        # a useful positive derivative. Negative values are clipped because AGL cannot be negative.
        return torch.clamp_min(F.softplus(raw) - math.log(2.0), 0.0)

    def forward(
        self,
        rgb: torch.Tensor,
        geometry_prior: torch.Tensor,
        *,
        coarse_terrain_prior: torch.Tensor | None = None,
        gsd_m: torch.Tensor | None = None,
    ) -> TerrainStructureOutput:
        self._validate_inputs(rgb, geometry_prior, coarse_terrain_prior)
        input_size = rgb.shape[-2:]
        terrain_evidence = self._terrain_evidence(geometry_prior, coarse_terrain_prior)

        rgb_state = rgb
        terrain_state = terrain_evidence
        fused: list[torch.Tensor] = []
        for rgb_stage, terrain_stage, fusion in zip(
            self.rgb_encoder.stages,
            self.terrain_encoder.stages,
            self.fusions,
            strict=True,
        ):
            rgb_state = rgb_stage(rgb_state)
            terrain_state = terrain_stage(terrain_state)
            fusion_output = fusion(rgb_state, terrain_state)
            rgb_state = fusion_output.rgb
            terrain_state = fusion_output.geometry
            fused.append(fusion_output.joint)

        fused[3] = self.metadata(fused[3], gsd_m)
        fused[2] = self.cross_scale_2(fused[2], fused[3])
        fused[1] = self.cross_scale_1(fused[1], fused[2])
        fused[0] = self.cross_scale_0(fused[0], fused[1])

        terrain_feature = self.terrain_decode(fused[3], fused[2])
        terrain_feature = self.terrain_refine(terrain_feature)
        terrain_residual_coarse = torch.tanh(self.terrain_residual_head(terrain_feature))
        terrain_residual_coarse = (
            terrain_residual_coarse * self.config.max_terrain_relative_residual
        )
        terrain_residual = F.interpolate(
            terrain_residual_coarse,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        terrain_relative = geometry_prior + terrain_residual

        structure = self.structure_decode3(fused[3], fused[2])
        structure = self.structure_decode2(structure, fused[1])
        structure = self.structure_decode1(structure, fused[0])
        structure = F.interpolate(structure, size=input_size, mode="bilinear", align_corners=False)
        structure = self.structure_native(structure)

        structure_logits = self.structure_support_head(structure)
        structure_probability = torch.sigmoid(structure_logits)
        raw_above_ground = self.above_ground_head(structure)
        above_ground_amplitude = self._nonnegative_zero_centered(raw_above_ground)
        above_ground_relative = structure_probability * above_ground_amplitude
        relative_height = terrain_relative + above_ground_relative

        boundary_probability = torch.sigmoid(self.boundary_head(structure))
        normals = F.normalize(self.normal_head(structure), dim=1, eps=1e-6)

        terrain_log_variance = F.interpolate(
            self.terrain_log_variance_head(terrain_feature),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        ).clamp(-8.0, 8.0)
        structure_log_variance = self.structure_log_variance_head(structure).clamp(-8.0, 8.0)

        return TerrainStructureOutput(
            terrain_relative=terrain_relative,
            terrain_residual=terrain_residual,
            above_ground_relative=above_ground_relative,
            relative_height=relative_height,
            structure_logits=structure_logits,
            structure_probability=structure_probability,
            semantic_logits=self.semantic_head(structure),
            height_bin_logits=self.height_bin_head(structure),
            boundary_probability=boundary_probability,
            normals=normals,
            terrain_log_variance=terrain_log_variance,
            structure_log_variance=structure_log_variance,
            terrain_uncertainty=torch.exp(0.5 * terrain_log_variance),
            structure_uncertainty=torch.exp(0.5 * structure_log_variance),
        )


def legacy_config_as_terrain_structure(config: HeightModelConfig) -> TerrainStructureConfig:
    """Create a research TSD config with widths compatible with an existing height-model config.

    This helper transfers only architectural widths/dropout. It does not convert or claim checkpoint
    compatibility; the terrain/structure heads are fundamentally different and require new training.
    """

    return TerrainStructureConfig(
        rgb_channels=config.rgb_channels,
        terrain_channels=config.geometry_channels,
        semantic_classes=max(config.semantic_classes, 2),
        dropout=config.dropout,
    )
