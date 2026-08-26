from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, *, max_groups: int = 8) -> int:
    """Choose the largest valid GroupNorm divisor up to ``max_groups``."""
    if channels <= 0:
        raise ValueError("channels must be positive")
    if max_groups <= 0:
        raise ValueError("max_groups must be positive")
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise RuntimeError("unable to determine a valid GroupNorm divisor")


@dataclass(frozen=True)
class HeightModelConfig:
    """Configuration for the DepthWizard remote-sensing refinement network."""

    rgb_channels: tuple[int, int, int, int] = (32, 64, 128, 192)
    geometry_channels: tuple[int, int, int, int] = (16, 32, 64, 96)
    semantic_classes: int = 5
    height_bins: int = 16
    dropout: float = 0.05
    max_relative_correction: float = 0.35

    def __post_init__(self) -> None:
        if len(self.rgb_channels) != 4 or len(self.geometry_channels) != 4:
            raise ValueError("rgb_channels and geometry_channels must each contain four stages")
        if any(channel <= 0 for channel in self.rgb_channels + self.geometry_channels):
            raise ValueError("all encoder channel widths must be positive")
        if self.semantic_classes <= 0:
            raise ValueError("semantic_classes must be positive")
        if self.height_bins < 2:
            raise ValueError("height_bins must be at least 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.max_relative_correction <= 1.0:
            raise ValueError("max_relative_correction must be in (0, 1]")


@dataclass(frozen=True)
class HeightModelOutput:
    """Dense outputs produced by the final remote-sensing refinement model."""

    relative_height: torch.Tensor
    relative_correction: torch.Tensor
    log_variance: torch.Tensor
    uncertainty: torch.Tensor
    semantic_logits: torch.Tensor
    height_bin_logits: torch.Tensor
    normals: torch.Tensor
    boundary_probability: torch.Tensor


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int, int, int, int], dropout: float) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        previous = in_channels
        for index, current in enumerate(channels):
            stages.append(
                nn.Sequential(
                    ConvNormAct(previous, current, stride=1 if index == 0 else 2),
                    ResidualBlock(current, dropout=dropout),
                )
            )
            previous = current
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class GatedFusion(nn.Module):
    """Fuse overhead RGB appearance with monocular geometry evidence at one scale."""

    def __init__(self, rgb_channels: int, geometry_channels: int, dropout: float) -> None:
        super().__init__()
        self.geometry_projection = nn.Conv2d(geometry_channels, rgb_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = ResidualBlock(rgb_channels, dropout=dropout)

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        projected = self.geometry_projection(geometry)
        gate = self.gate(torch.cat([rgb, projected], dim=1))
        return self.refine(rgb + gate * projected)


class MetadataConditioner(nn.Module):
    """Condition bottleneck features on GSD while representing missing metadata explicitly."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 32)
        self.network = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        gsd_m: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        if gsd_m is None:
            metadata = torch.zeros((batch, 2), dtype=x.dtype, device=x.device)
            metadata[:, 1] = 1.0
        else:
            gsd = gsd_m.reshape(batch).to(dtype=x.dtype, device=x.device)
            finite_positive = torch.isfinite(gsd) & (gsd > 0)
            safe = torch.where(finite_positive, gsd, torch.ones_like(gsd))
            metadata = torch.stack(
                [torch.log1p(safe), (~finite_positive).to(dtype=x.dtype)],
                dim=1,
            )
        embedding = self.network(metadata).view(batch, -1, 1, 1)
        return x + embedding


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels + skip_channels, out_channels)
        self.refine = ResidualBlock(out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.project(torch.cat([x, skip], dim=1))
        return self.refine(x)


class DepthWizardHeightModel(nn.Module):
    """Dual-evidence remote-sensing height refinement model.

    DA3 geometry is an explicit prior. The trainable network predicts a bounded correction rather
    than replacing the prior outright. The correction head is initialized to zero, therefore an
    untrained model is an identity refinement of DA3. Metric elevation remains the responsibility
    of the DEM/GCP evidence-calibration subsystem.
    """

    def __init__(self, config: HeightModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeightModelConfig()
        rgb_channels = self.config.rgb_channels
        geometry_channels = self.config.geometry_channels

        self.rgb_encoder = Encoder(3, rgb_channels, self.config.dropout)
        self.geometry_encoder = Encoder(1, geometry_channels, self.config.dropout)
        self.fusions = nn.ModuleList(
            [
                GatedFusion(rgb_c, geo_c, self.config.dropout)
                for rgb_c, geo_c in zip(rgb_channels, geometry_channels, strict=True)
            ]
        )
        self.metadata = MetadataConditioner(rgb_channels[-1])

        self.decode3 = DecoderBlock(
            rgb_channels[3], rgb_channels[2], rgb_channels[2], self.config.dropout
        )
        self.decode2 = DecoderBlock(
            rgb_channels[2], rgb_channels[1], rgb_channels[1], self.config.dropout
        )
        self.decode1 = DecoderBlock(
            rgb_channels[1], rgb_channels[0], rgb_channels[0], self.config.dropout
        )
        self.full_resolution = nn.Sequential(
            ConvNormAct(rgb_channels[0], rgb_channels[0]),
            ResidualBlock(rgb_channels[0], dropout=self.config.dropout),
        )

        head_channels = rgb_channels[0]
        self.height_residual_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        nn.init.zeros_(self.height_residual_head.weight)
        nn.init.zeros_(self.height_residual_head.bias)
        self.log_variance_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.semantic_head = nn.Conv2d(
            head_channels,
            self.config.semantic_classes,
            kernel_size=1,
        )
        self.height_bin_head = nn.Conv2d(head_channels, self.config.height_bins, kernel_size=1)
        self.normal_head = nn.Conv2d(head_channels, 3, kernel_size=1)
        self.boundary_head = nn.Conv2d(head_channels, 1, kernel_size=1)

    @staticmethod
    def _validate_inputs(rgb: torch.Tensor, geometry_prior: torch.Tensor) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape N x 3 x H x W")
        if geometry_prior.ndim != 4 or geometry_prior.shape[1] != 1:
            raise ValueError("geometry_prior must have shape N x 1 x H x W")
        if rgb.shape[0] != geometry_prior.shape[0] or rgb.shape[-2:] != geometry_prior.shape[-2:]:
            raise ValueError("rgb and geometry_prior must share batch and spatial dimensions")

    def forward(
        self,
        rgb: torch.Tensor,
        geometry_prior: torch.Tensor,
        *,
        gsd_m: torch.Tensor | None = None,
    ) -> HeightModelOutput:
        self._validate_inputs(rgb, geometry_prior)
        input_size = rgb.shape[-2:]

        rgb_features = self.rgb_encoder(rgb)
        geometry_features = self.geometry_encoder(geometry_prior)
        fused = [
            fusion(rgb_feature, geometry_feature)
            for fusion, rgb_feature, geometry_feature in zip(
                self.fusions,
                rgb_features,
                geometry_features,
                strict=True,
            )
        ]

        x = self.metadata(fused[3], gsd_m)
        x = self.decode3(x, fused[2])
        x = self.decode2(x, fused[1])
        x = self.decode1(x, fused[0])
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.full_resolution(x)

        relative_correction = (
            torch.tanh(self.height_residual_head(x)) * self.config.max_relative_correction
        )
        relative_height = torch.clamp(geometry_prior + relative_correction, 0.0, 1.0)
        log_variance = torch.clamp(self.log_variance_head(x), min=-7.0, max=5.0)
        uncertainty = torch.exp(0.5 * log_variance)
        semantic_logits = self.semantic_head(x)
        height_bin_logits = self.height_bin_head(x)
        normals = F.normalize(self.normal_head(x), dim=1, eps=1e-6)
        boundary_probability = torch.sigmoid(self.boundary_head(x))

        return HeightModelOutput(
            relative_height=relative_height,
            relative_correction=relative_correction,
            log_variance=log_variance,
            uncertainty=uncertainty,
            semantic_logits=semantic_logits,
            height_bin_logits=height_bin_logits,
            normals=normals,
            boundary_probability=boundary_probability,
        )
