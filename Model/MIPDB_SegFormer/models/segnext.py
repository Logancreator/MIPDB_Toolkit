#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SegNeXt

Reference:
    Guo et al. "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation"
    https://arxiv.org/abs/2209.08575
"""

from __future__ import annotations

import logging
from typing import Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MSCA(nn.Module):
    """多尺度卷积注意力模块"""
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv0_1 = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels)
        self.conv0_2 = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels)
        self.conv1_1 = nn.Conv2d(channels, channels, (1, 11), padding=(0, 5), groups=channels)
        self.conv1_2 = nn.Conv2d(channels, channels, (11, 1), padding=(5, 0), groups=channels)
        self.conv2_1 = nn.Conv2d(channels, channels, (1, 21), padding=(0, 10), groups=channels)
        self.conv2_2 = nn.Conv2d(channels, channels, (21, 1), padding=(10, 0), groups=channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn = self.conv3(attn + attn_0 + attn_1 + attn_2)
        return attn * u


class MSCABlock(nn.Module):
    """MSCA Block with FFN"""
    
    def __init__(self, channels: int, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = MSCA(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(channels, mlp_hidden, 1), nn.GELU(), nn.Conv2d(mlp_hidden, channels, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 7, stride: int = 4, in_channels: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride, patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class MSCAN(nn.Module):
    def __init__(self, embed_dims: List[int] = [32, 64, 160, 256], mlp_ratios: List[float] = [8, 8, 4, 4],
                 depths: List[int] = [3, 3, 5, 2], drop_path_rate: float = 0.1) -> None:
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(7, 4, 3, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([MSCABlock(embed_dims[0], mlp_ratios[0], dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.BatchNorm2d(embed_dims[0])
        cur += depths[0]
        self.block2 = nn.ModuleList([MSCABlock(embed_dims[1], mlp_ratios[1], dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.BatchNorm2d(embed_dims[1])
        cur += depths[1]
        self.block3 = nn.ModuleList([MSCABlock(embed_dims[2], mlp_ratios[2], dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.BatchNorm2d(embed_dims[2])
        cur += depths[2]
        self.block4 = nn.ModuleList([MSCABlock(embed_dims[3], mlp_ratios[3], dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.BatchNorm2d(embed_dims[3])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        x = self.patch_embed1(x)
        for blk in self.block1: x = blk(x)
        outs.append(self.norm1(x))
        x = self.patch_embed2(x)
        for blk in self.block2: x = blk(x)
        outs.append(self.norm2(x))
        x = self.patch_embed3(x)
        for blk in self.block3: x = blk(x)
        outs.append(self.norm3(x))
        x = self.patch_embed4(x)
        for blk in self.block4: x = blk(x)
        outs.append(self.norm4(x))
        return outs


class LightHamHead(nn.Module):
    def __init__(self, in_channels: List[int], channels: int, num_classes: int) -> None:
        super().__init__()
        self.squeeze = nn.ModuleList([nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU()) for c in in_channels])
        self.align = nn.Sequential(nn.Conv2d(channels * 4, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.ham = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.cls = nn.Conv2d(channels, num_classes, 1)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[2:]
        squeezed = []
        for feat, sq in zip(features, self.squeeze):
            f = sq(feat)
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=True)
            squeezed.append(f)
        x = self.align(torch.cat(squeezed, dim=1))
        x = x * self.ham(x).expand_as(x) + x
        return self.cls(x)


@register_model('segnext')
class SegNeXt(nn.Module):
    """SegNeXt: 高效的多尺度卷积注意力分割网络"""
    
    SUPPORTED_BACKBONES = ('tiny', 'small', 'base')
    CONFIGS = {
        'tiny': {'embed_dims': [32, 64, 160, 256], 'depths': [3, 3, 5, 2]},
        'small': {'embed_dims': [64, 128, 320, 512], 'depths': [2, 2, 4, 2]},
        'base': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 3, 12, 3]},
    }
    
    def __init__(self, num_classes: int = 2, backbone: str = 'small', pretrained: bool = True,
                 pretrained_weights: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__()
        self.logger = get_logger()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"不支持: {backbone}，请选择: {self.SUPPORTED_BACKBONES}")
        
        config = self.CONFIGS[backbone]
        self.encoder = MSCAN(embed_dims=config['embed_dims'], depths=config['depths'])
        self.decoder = LightHamHead(config['embed_dims'], 256, num_classes)
        self._init_weights()
        
        # 加载预训练权重
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        elif pretrained:
            self.logger.warning(
                "SegNeXt 无内置在线预训练权重，将从头训练。"
                "如需预训练，请通过 --pretrained_weights 指定本地权重路径"
            )
        
        self.logger.info(
            f"SegNeXt 初始化完成 | 配置: {backbone} | 类别数: {num_classes} | "
            f"预训练: {'本地权重' if pretrained_weights else ('从头训练' if not pretrained else '从头训练(无可用预训练)')}"
        )
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained(self, weights_path: str) -> None:
        import os
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        self.logger.info(f"加载本地权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        out = self.decoder(self.encoder(x))
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return {'out': out}
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        return {'backbone': 'small', 'pretrained': True}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes}, backbone='{self.backbone_name}')"
