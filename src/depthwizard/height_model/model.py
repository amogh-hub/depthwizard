from __future__ import annotations

import math
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
    architecture_version: str = "bidirectional-cross-scale-v1"

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
        supported = {"bidirectional-cross-scale-v1", "confidence-gated-v2"}
        if self.architecture_version not in supported:
            raise ValueError("unsupported height-model architecture_version")


@dataclass(frozen=True)
class HeightModelOutput:
    """Dense outputs produced by the final remote-sensing refinement model.

    ``raw_relative_correction`` and ``correction_gate`` are populated by the production model. They
    remain optional so historical synthetic/unit-test outputs and V1 checkpoints stay source-level
    compatible. V1 emits a gate of ones; V2 learns an explicit confidence gate.
    """

    relative_height: torch.Tensor
    relative_correction: torch.Tensor
    log_variance: torch.Tensor
    uncertainty: torch.Tensor
    semantic_logits: torch.Tensor
    height_bin_logits: torch.Tensor
    normals: torch.Tensor
    boundary_probability: torch.Tensor
    raw_relative_correction: torch.Tensor | None = None
    correction_gate: torch.Tensor | None = None


@dataclass(frozen=True)
class FusionOutput:
    """Bidirectionally updated branch features plus their joint evidence representation."""

    rgb: torch.Tensor
    geometry: torch.Tensor
    joint: torch.Tensor


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


class BidirectionalGatedFusion(nn.Module):
    """Exchange RGB and geometry evidence in both directions at one encoder scale.

    The previous refiner injected geometry into RGB only after both encoders had already produced
    their independent feature pyramids. That made RGB a consumer of geometry but never allowed
    overhead appearance evidence to refine the geometry stream before the next scale. Here each
    branch receives a gated projection from the other branch and the updated branch states are fed
    into the next encoder stage. The joint feature remains RGB-width so the decoder contract stays
    compact and stable.
    """

    def __init__(self, rgb_channels: int, geometry_channels: int, dropout: float) -> None:
        super().__init__()
        self.geometry_to_rgb = nn.Conv2d(
            geometry_channels,
            rgb_channels,
            kernel_size=1,
            bias=False,
        )
        self.rgb_to_geometry = nn.Conv2d(
            rgb_channels,
            geometry_channels,
            kernel_size=1,
            bias=False,
        )
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(rgb_channels * 2, rgb_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.geometry_gate = nn.Sequential(
            nn.Conv2d(geometry_channels * 2, geometry_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.rgb_refine = ResidualBlock(rgb_channels, dropout=dropout)
        self.geometry_refine = ResidualBlock(geometry_channels, dropout=dropout)
        self.joint_projection = ConvNormAct(rgb_channels + geometry_channels, rgb_channels)
        self.joint_refine = ResidualBlock(rgb_channels, dropout=dropout)

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor) -> FusionOutput:
        geometry_for_rgb = self.geometry_to_rgb(geometry)
        rgb_for_geometry = self.rgb_to_geometry(rgb)

        rgb_gate = self.rgb_gate(torch.cat([rgb, geometry_for_rgb], dim=1))
        geometry_gate = self.geometry_gate(torch.cat([geometry, rgb_for_geometry], dim=1))

        rgb_updated = self.rgb_refine(rgb + rgb_gate * geometry_for_rgb)
        geometry_updated = self.geometry_refine(geometry + geometry_gate * rgb_for_geometry)
        joint = self.joint_projection(torch.cat([rgb_updated, geometry_updated], dim=1))
        joint = self.joint_refine(joint)
        return FusionOutput(rgb=rgb_updated, geometry=geometry_updated, joint=joint)


class CrossScaleContext(nn.Module):
    """Inject deeper scene context into a shallower fused feature using a learned gate."""

    def __init__(self, shallow_channels: int, deep_channels: int, dropout: float) -> None:
        super().__init__()
        self.deep_projection = nn.Conv2d(deep_channels, shallow_channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(shallow_channels * 2, shallow_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = ResidualBlock(shallow_channels, dropout=dropout)

    def forward(self, shallow: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        deep = F.interpolate(deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        projected = self.deep_projection(deep)
        gate = self.gate(torch.cat([shallow, projected], dim=1))
        return self.refine(shallow + gate * projected)


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
    """Bidirectional dual-evidence remote-sensing height refinement model.

    DA3 geometry is an explicit prior. RGB and geometry exchange information at every encoder scale,
    and deeper fused context is gated back into shallower structural features before decoding. The
    trainable network predicts only a bounded correction rather than replacing DA3 outright.

    ``bidirectional-cross-scale-v1`` applies that candidate correction directly. The production
    ``confidence-gated-v2`` variant additionally predicts where the candidate correction is safe to
    apply. Its gate starts at 0.10 while the correction head starts at exactly zero, preserving the
    exact DA3 identity invariant at initialization and making early optimization conservative.

    Metric elevation remains the responsibility of the DEM/GCP evidence-calibration subsystem.
    The corrected relative field is intentionally not hard-clipped to [0, 1], because local
    structural refinements may legitimately move beyond DA3's normalized nominal range.
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
                BidirectionalGatedFusion(rgb_c, geo_c, self.config.dropout)
                for rgb_c, geo_c in zip(rgb_channels, geometry_channels, strict=True)
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
        if self.height_residual_head.bias is not None:
            nn.init.zeros_(self.height_residual_head.bias)

        self.correction_gate_head: nn.Conv2d | None = None
        if self.config.architecture_version == "confidence-gated-v2":
            self.correction_gate_head = nn.Conv2d(head_channels, 1, kernel_size=1)
            nn.init.zeros_(self.correction_gate_head.weight)
            if self.correction_gate_head.bias is not None:
                initial_gate_probability = 0.10
                self.correction_gate_head.bias.data.fill_(
                    math.log(initial_gate_probability / (1.0 - initial_gate_probability))
                )

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

        rgb_state = rgb
        geometry_state = geometry_prior
        fused: list[torch.Tensor] = []
        for rgb_stage, geometry_stage, fusion in zip(
            self.rgb_encoder.stages,
            self.geometry_encoder.stages,
            self.fusions,
            strict=True,
        ):
            rgb_state = rgb_stage(rgb_state)
            geometry_state = geometry_stage(geometry_state)
            fusion_output = fusion(rgb_state, geometry_state)
            rgb_state = fusion_output.rgb
            geometry_state = fusion_output.geometry
            fused.append(fusion_output.joint)

        # Top-down context makes shallow edge/roof evidence aware of the broader scene geometry.
        fused[2] = self.cross_scale_2(fused[2], fused[3])
        fused[1] = self.cross_scale_1(fused[1], fused[2])
        fused[0] = self.cross_scale_0(fused[0], fused[1])

        x = self.metadata(fused[3], gsd_m)
        x = self.decode3(x, fused[2])
        x = self.decode2(x, fused[1])
        x = self.decode1(x, fused[0])
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.full_resolution(x)

        raw_relative_correction = (
            torch.tanh(self.height_residual_head(x)) * self.config.max_relative_correction
        )
        if self.correction_gate_head is None:
            correction_gate = torch.ones_like(raw_relative_correction)
        else:
            correction_gate = torch.sigmoid(self.correction_gate_head(x))
        relative_correction = raw_relative_correction * correction_gate
        relative_height = geometry_prior + relative_correction

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
            raw_relative_correction=raw_relative_correction,
            correction_gate=correction_gate,
        )