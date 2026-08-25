"""Minimal compatibility port of the released RDAH-Net inference architecture.

This module mirrors the public RDAH-Net ``test.py`` model graph at upstream commit
``373bca28299683ab0e5d892dfc87598ab967b564`` so the authors' checkpoints can be
loaded without importing their CUDA-hardcoded evaluation script or its legacy training
stack. RDAH-Net is MIT licensed; see ``docs/third-party-baselines.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BlockAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        block_size: int = 8,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.mlp_dim = mlp_dim or dim * 2
        self.scale = self.head_dim**-0.5

        self.local_proj = ConvLayer(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, self.mlp_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.mlp_dim, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if height % self.block_size or width % self.block_size:
            raise ValueError("RDAH attention feature maps must be divisible by block_size")
        local_feat = self.local_proj(x)
        blocks_h = height // self.block_size
        blocks_w = width // self.block_size
        num_blocks = blocks_h * blocks_w

        blocked = (
            x.reshape(
                batch,
                channels,
                blocks_h,
                self.block_size,
                blocks_w,
                self.block_size,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch * num_blocks, channels, self.block_size, self.block_size)
        )
        blocked_batch, blocked_channels, block_h, block_w = blocked.shape
        tokens = block_h * block_w
        qkv = self.qkv(blocked).reshape(blocked_batch, 3, blocked_channels, tokens)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.reshape(blocked_batch, self.num_heads, self.head_dim, tokens)
        k = k.reshape(blocked_batch, self.num_heads, self.head_dim, tokens)
        v = v.reshape(blocked_batch, self.num_heads, self.head_dim, tokens)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.contiguous().reshape(blocked_batch, blocked_channels, block_h, block_w)
        out = self.proj_drop(self.proj(out))
        out = (
            out.reshape(
                batch,
                blocks_h,
                blocks_w,
                channels,
                self.block_size,
                self.block_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, height, width)
        )
        x = x + local_feat + out
        return x + self.mlp(x)


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_heads: int = 4,
        block_size: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_channels % num_heads != 0:
            raise ValueError("out_channels must be divisible by num_heads")
        self.conv1 = ConvLayer(in_channels, out_channels, stride=stride)
        self.attention = BlockAttention(
            out_channels,
            num_heads,
            block_size,
            dropout=dropout,
        )
        self.conv2 = ConvLayer(out_channels, out_channels, groups=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.attention(self.conv1(x)))


class MobileViT_S_Light(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stem = ConvLayer(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            MobileViTBlock(32, 64, stride=2, num_heads=4, block_size=8),
            MobileViTBlock(64, 64, stride=1, num_heads=4, block_size=8),
        )
        self.stage2 = nn.Sequential(
            MobileViTBlock(64, 128, stride=2, num_heads=8, block_size=8),
            MobileViTBlock(128, 128, stride=1, num_heads=8, block_size=8),
        )
        self.stage3 = nn.Sequential(
            MobileViTBlock(128, 256, stride=2, num_heads=8, block_size=8),
            MobileViTBlock(256, 256, stride=1, num_heads=8, block_size=8),
        )
        self.proj1 = ConvLayer(64, 32, kernel_size=1, padding=0)
        self.proj2 = ConvLayer(128, 32, kernel_size=1, padding=0)
        self.proj3 = ConvLayer(256, 32, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        return [self.proj1(feat1), self.proj2(feat2), self.proj3(feat3)]


class CBAM(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_att = self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 32, height: int = 64, width: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        pos_x = torch.arange(width, dtype=torch.float32).repeat(height, 1)
        pos_y = torch.arange(height, dtype=torch.float32).repeat(width, 1).t()
        pos = torch.stack([pos_x, pos_y], dim=0)
        pe = torch.zeros(1, d_model, height, width)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[0, ::2, :, :] = torch.sin(pos[0:1, :, :] * div_term[None, :, None, None])
        pe[0, 1::2, :, :] = torch.cos(
            pos[1:2, :, :] * div_term[None, :, None, None]
        )
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), : x.size(2), : x.size(3)]


class LightCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        block_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.proj_q = nn.Conv2d(d_model, d_model, 1)
        self.proj_k = nn.Conv2d(d_model, d_model, 1)
        self.proj_v = nn.Conv2d(d_model, d_model, 1)
        self.proj_out = nn.Conv2d(d_model, d_model, 1)
        self.norm = nn.BatchNorm2d(d_model)

    def _blockify(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, channels, height, width = x.shape
        if height % self.block_size or width % self.block_size:
            raise ValueError("RDAH cross-attention maps must be divisible by block_size")
        blocks_h = height // self.block_size
        blocks_w = width // self.block_size
        blocked = (
            x.reshape(
                batch,
                channels,
                blocks_h,
                self.block_size,
                blocks_w,
                self.block_size,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(
                batch * blocks_h * blocks_w,
                channels,
                self.block_size,
                self.block_size,
            )
        )
        return blocked, blocks_h, blocks_w

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = q.shape
        q_original = q
        q_blocked, blocks_h, blocks_w = self._blockify(q)
        k_blocked, _, _ = self._blockify(k)
        v_blocked, _, _ = self._blockify(v)
        blocked_batch, blocked_channels, block_h, block_w = q_blocked.shape
        tokens = block_h * block_w
        q_proj = self.proj_q(q_blocked).reshape(
            blocked_batch,
            self.num_heads,
            self.head_dim,
            tokens,
        )
        k_proj = self.proj_k(k_blocked).reshape(
            blocked_batch,
            self.num_heads,
            self.head_dim,
            tokens,
        )
        v_proj = self.proj_v(v_blocked).reshape(
            blocked_batch,
            self.num_heads,
            self.head_dim,
            tokens,
        )
        attn = torch.einsum("bhdn,bhdm->bhnm", q_proj, k_proj) * self.scale
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v_proj)
        out = out.contiguous().reshape(blocked_batch, blocked_channels, block_h, block_w)
        out = self.proj_out(out)
        out = (
            out.reshape(
                batch,
                blocks_h,
                blocks_w,
                channels,
                self.block_size,
                self.block_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, height, width)
        )
        return self.norm(out + q_original)


class LightTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        hidden_dim: int = 64,
        block_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = LightCrossAttention(d_model, num_heads, block_size, dropout)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, d_model, 1),
        )
        self.norm = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x, x, x)
        return x + self.dropout(self.ffn(self.norm(x)))


class HeightPredTransformer(nn.Module):
    """Released RDAH-Net height-prediction graph, checkpoint-compatible."""

    def __init__(self, d_model: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth_encoder = MobileViT_S_Light(in_channels=1)
        self.img_encoder = MobileViT_S_Light(in_channels=3)
        self.cbam_blocks = nn.ModuleList([CBAM(d_model) for _ in range(3)])
        self.cross_attn_blocks = nn.ModuleList(
            [LightCrossAttention(d_model, num_heads, block_size=8) for _ in range(3)]
        )
        self.rev_cross_attn_blocks = nn.ModuleList(
            [LightCrossAttention(d_model, num_heads, block_size=8) for _ in range(3)]
        )
        self.pos_encoding = PositionalEncoding(d_model, height=64, width=64)
        self.global_transformer = LightTransformerBlock(
            d_model,
            num_heads,
            hidden_dim=64,
        )
        self.skip_projs = nn.ModuleList(
            [
                nn.Conv2d(d_model, 16, 1),
                nn.Conv2d(d_model, 32, 1),
                nn.Conv2d(d_model, 64, 1),
            ]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, depth: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        depth_feats = self.depth_encoder(depth)
        img_feats = self.img_encoder(img)
        depth_feats = [self.cbam_blocks[i](feat) for i, feat in enumerate(depth_feats)]
        img_feats = [self.cbam_blocks[i](feat) for i, feat in enumerate(img_feats)]

        fused_feats: list[torch.Tensor] = []
        for index in range(3):
            feat1 = self.cross_attn_blocks[index](
                q=depth_feats[index],
                k=img_feats[index],
                v=img_feats[index],
            )
            feat2 = self.rev_cross_attn_blocks[index](
                q=img_feats[index],
                k=depth_feats[index],
                v=depth_feats[index],
            )
            fused_feats.append((feat1 + feat2) / 2)

        x = self.pos_encoding(fused_feats[2])
        x = self.global_transformer(x)

        x = self.decoder[0:4](x)
        skip = self.skip_projs[2](fused_feats[2])
        x = x + F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)

        x = self.decoder[4:8](x)
        skip = self.skip_projs[1](fused_feats[1])
        x = x + F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)

        x = self.decoder[8:12](x)
        skip = self.skip_projs[0](fused_feats[0])
        x = x + F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)

        return self.decoder[12:](x)
