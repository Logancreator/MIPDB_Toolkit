#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CGNet

Reference:
    Wu et al. "CGNet: A Light-weight Context Guided Network for Semantic Segmentation"
    https://arxiv.org/abs/1811.08201
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ConvBNPReLU(nn.Module):
    """卷积 + BN + PReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class BNPReLU(nn.Module):
    """BN + PReLU"""
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(x))


class ChannelWiseConv(nn.Module):
    """通道分离卷积"""
    
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, 1, padding, 
                             dilation, groups=channels, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FGlo(nn.Module):
    """全局上下文聚合模块"""
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class ContextGuidedBlock(nn.Module):
    """上下文引导块"""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 2, 
                 reduction: int = 16, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        inter_channels = out_channels // 2
        
        self.conv1x1 = ConvBNPReLU(in_channels, inter_channels, 1)
        self.f_loc = ChannelWiseConv(inter_channels, 3, padding=1)
        self.f_sur = ChannelWiseConv(inter_channels, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.f_glo = FGlo(out_channels, reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        loc = self.f_loc(out)
        sur = self.f_sur(out)
        joi = torch.cat([loc, sur], dim=1)
        joi = self.prelu(self.bn(joi))
        out = self.f_glo(joi)
        
        if self.residual:
            out = out + x
        return out


class ContextGuidedBlockDown(nn.Module):
    """带下采样的上下文引导块"""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 2, reduction: int = 16) -> None:
        super().__init__()
        inter_channels = out_channels // 2
        
        self.conv1x1 = ConvBNPReLU(in_channels, inter_channels, 1)
        self.f_loc = nn.Conv2d(inter_channels, inter_channels, 3, stride=2, 
                              padding=1, groups=inter_channels, bias=False)
        self.f_sur = nn.Conv2d(inter_channels, inter_channels, 3, stride=2, 
                              padding=dilation, dilation=dilation, groups=inter_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.f_glo = FGlo(out_channels, reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        loc = self.f_loc(out)
        sur = self.f_sur(out)
        joi = torch.cat([loc, sur], dim=1)
        joi = self.prelu(self.bn(joi))
        return self.f_glo(joi)


class InputInjection(nn.Module):
    """输入注入模块"""
    
    def __init__(self, downsample_ratio: int) -> None:
        super().__init__()
        self.pool = nn.ModuleList()
        for _ in range(downsample_ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for pool in self.pool:
            x = pool(x)
        return x


@register_model('cgnet')
class CGNet(nn.Module):
    """
    CGNet: 上下文引导网络.
    
    特点：
    - 轻量级设计（~0.5M 参数）
    - 上下文引导块 (CG Block)
    - 局部特征 + 周围上下文 + 全局上下文
    - 输入注入保留空间信息
    
    Args:
        num_classes: 分割类别数
        backbone: 未使用
        M: Stage 2 中 CG Block 数量
        N: Stage 3 中 CG Block 数量
    """
    
    SUPPORTED_BACKBONES = ('none',)
    
    def __init__(
        self, 
        num_classes: int = 2,
        backbone: str = 'none',
        pretrained: bool = False,
        pretrained_weights: Optional[str] = None,
        M: int = 3,
        N: int = 21,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.num_classes = num_classes
        
        # CGNet 无预训练骨干网络
        if pretrained and pretrained_weights is None:
            self.logger.info("CGNet 为轻量级网络，无 ImageNet 预训练骨干，将从头训练")
        
        # Stage 1
        self.level1_0 = ConvBNPReLU(3, 32, 3, stride=2, padding=1)
        self.level1_1 = ConvBNPReLU(32, 32, 3, padding=1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, padding=1)
        
        self.sample1 = InputInjection(1)
        self.sample2 = InputInjection(2)
        
        self.bn_prelu_1 = BNPReLU(32 + 3)
        
        # Stage 2
        self.level2_0 = ContextGuidedBlockDown(32 + 3, 64, dilation=2, reduction=8)
        self.level2 = nn.ModuleList()
        for _ in range(M - 1):
            self.level2.append(ContextGuidedBlock(64, 64, dilation=2, reduction=8))
        
        self.bn_prelu_2 = BNPReLU(64 + 64 + 3)
        
        # Stage 3
        self.level3_0 = ContextGuidedBlockDown(64 + 64 + 3, 128, dilation=4, reduction=16)
        self.level3 = nn.ModuleList()
        for _ in range(N - 1):
            self.level3.append(ContextGuidedBlock(128, 128, dilation=4, reduction=16))
        
        self.bn_prelu_3 = BNPReLU(256)
        
        # 分类器
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
        # 初始化
        self._init_weights()
        
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        
        self.logger.info(f"CGNet 初始化完成 | 类别数: {num_classes} | M={M}, N={N}")
    
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
        
        # Stage 1
        out0 = self.level1_0(x)
        out0 = self.level1_1(out0)
        out0 = self.level1_2(out0)
        
        inp1 = self.sample1(x)
        inp2 = self.sample2(x)
        
        out0_cat = self.bn_prelu_1(torch.cat([out0, inp1], dim=1))
        
        # Stage 2
        out1_0 = self.level2_0(out0_cat)
        out1 = out1_0
        for layer in self.level2:
            out1 = layer(out1)
        
        out1_cat = self.bn_prelu_2(torch.cat([out1_0, out1, inp2], dim=1))
        
        # Stage 3
        out2_0 = self.level3_0(out1_cat)
        out2 = out2_0
        for layer in self.level3:
            out2 = layer(out2)
        
        out2_cat = self.bn_prelu_3(torch.cat([out2_0, out2], dim=1))
        
        # 分类
        out = self.classifier(out2_cat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        return {'out': out}
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        return {'backbone': 'none', 'M': 3, 'N': 21}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes})"
