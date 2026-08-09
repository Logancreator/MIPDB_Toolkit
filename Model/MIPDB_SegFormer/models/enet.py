#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation

Reference:
    Paszke et al. "ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation"
    https://arxiv.org/abs/1606.02147
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


class InitialBlock(nn.Module):
    """ENet 初始块：并行的卷积和最大池化"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv(x)
        pool = self.pool(x)
        x = torch.cat([conv, pool], dim=1)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class RegularBottleneck(nn.Module):
    """ENet 常规瓶颈块"""
    
    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 3, 
        padding: int = 1,
        dilation: int = 1,
        asymmetric: bool = False,
        dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        
        internal_channels = channels // 4
        
        # 1x1 压缩
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, 1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )
        
        # 主卷积（可能是不对称卷积或空洞卷积）
        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, (kernel_size, 1), 
                         padding=(padding, 0), bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
                nn.Conv2d(internal_channels, internal_channels, (1, kernel_size), 
                         padding=(0, padding), bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size, 
                         padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
            )
        
        # 1x1 扩展
        self.conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.dropout = nn.Dropout2d(dropout_prob)
        self.prelu = nn.PReLU(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.dropout(out)
        
        out = out + residual
        out = self.prelu(out)
        
        return out


class DownsamplingBottleneck(nn.Module):
    """ENet 下采样瓶颈块"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        
        internal_channels = in_channels // 4
        
        # 主分支
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 2, stride=2, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.dropout = nn.Dropout2d(dropout_prob)
        
        # 残差分支
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv_residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_residual = nn.BatchNorm2d(out_channels)
        
        self.prelu = nn.PReLU(out_channels)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 主分支
        main = self.conv1(x)
        main = self.conv2(main)
        main = self.conv3(main)
        main = self.dropout(main)
        
        # 残差分支
        residual, indices = self.pool(x)
        residual = self.conv_residual(residual)
        residual = self.bn_residual(residual)
        
        out = main + residual
        out = self.prelu(out)
        
        return out, indices


class UpsamplingBottleneck(nn.Module):
    """ENet 上采样瓶颈块"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        
        internal_channels = in_channels // 4
        
        # 主分支：使用转置卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(internal_channels, internal_channels, 3, stride=2, 
                              padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.dropout = nn.Dropout2d(dropout_prob)
        
        # 残差分支：使用最大反池化
        self.conv_residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_residual = nn.BatchNorm2d(out_channels)
        self.unpool = nn.MaxUnpool2d(2)
        
        self.prelu = nn.PReLU(out_channels)
    
    def forward(self, x: torch.Tensor, indices: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
        # 主分支
        main = self.conv1(x)
        main = self.conv2(main)
        main = self.conv3(main)
        main = self.dropout(main)
        
        # 残差分支
        residual = self.conv_residual(x)
        residual = self.bn_residual(residual)
        residual = self.unpool(residual, indices, output_size=output_size)
        
        out = main + residual
        out = self.prelu(out)
        
        return out


@register_model('enet')
class ENet(nn.Module):
    """
    ENet
    """
    
    SUPPORTED_BACKBONES = ('none',)  # ENet 不使用预训练骨干
    
    def __init__(
        self, 
        num_classes: int = 2,
        backbone: str = 'none',
        pretrained: bool = False,
        pretrained_weights: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.num_classes = num_classes
        
        # ENet 无预训练骨干网络
        if pretrained and pretrained_weights is None:
            self.logger.info("ENet 为轻量级网络，无 ImageNet 预训练骨干，将从头训练")
        
        # ===== 编码器 =====
        # 初始块: 3 -> 16, 下采样 2x
        self.initial = InitialBlock(3, 16)
        
        # Stage 1: 16 -> 64, 下采样 2x
        self.down1 = DownsamplingBottleneck(16, 64, dropout_prob=0.01)
        self.regular1 = nn.Sequential(
            RegularBottleneck(64, dropout_prob=0.01),
            RegularBottleneck(64, dropout_prob=0.01),
            RegularBottleneck(64, dropout_prob=0.01),
            RegularBottleneck(64, dropout_prob=0.01),
        )
        
        # Stage 2: 64 -> 128, 下采样 2x
        self.down2 = DownsamplingBottleneck(64, 128, dropout_prob=0.1)
        self.regular2 = nn.Sequential(
            RegularBottleneck(128, dropout_prob=0.1),
            RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1),
            RegularBottleneck(128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1),
            RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1),
            RegularBottleneck(128, dropout_prob=0.1),
            RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1),
            RegularBottleneck(128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1),
            RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1),
        )
        
        # Stage 3: 重复 stage 2 的结构（无下采样）
        self.regular3 = nn.Sequential(
            RegularBottleneck(128, dropout_prob=0.1),
            RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1),
            RegularBottleneck(128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1),
            RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1),
            RegularBottleneck(128, dropout_prob=0.1),
            RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1),
            RegularBottleneck(128, asymmetric=True, kernel_size=5, padding=2, dropout_prob=0.1),
            RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1),
        )
        
        # ===== 解码器 =====
        # Stage 4: 128 -> 64, 上采样 2x
        self.up4 = UpsamplingBottleneck(128, 64, dropout_prob=0.1)
        self.regular4 = nn.Sequential(
            RegularBottleneck(64, dropout_prob=0.1),
            RegularBottleneck(64, dropout_prob=0.1),
        )
        
        # Stage 5: 64 -> 16, 上采样 2x
        self.up5 = UpsamplingBottleneck(64, 16, dropout_prob=0.1)
        self.regular5 = RegularBottleneck(16, dropout_prob=0.1)
        
        # 最终上采样和分类
        self.fullconv = nn.ConvTranspose2d(16, num_classes, 2, stride=2)
        
        # 加载本地权重
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        
        self.logger.info(f"ENet 初始化完成 | 类别数: {num_classes}")
    
    def _load_pretrained(self, weights_path: str) -> None:
        import os
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        self.logger.info(f"加载本地权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[2:]
        
        # 编码器
        x = self.initial(x)
        
        x, indices1 = self.down1(x)
        size1 = x.shape[2:]
        x = self.regular1(x)
        
        x, indices2 = self.down2(x)
        size2 = x.shape[2:]
        x = self.regular2(x)
        x = self.regular3(x)
        
        # 解码器
        x = self.up4(x, indices2, size2)
        x = self.regular4(x)
        
        x = self.up5(x, indices1, size1)
        x = self.regular5(x)
        
        x = self.fullconv(x)
        
        # 确保输出尺寸与输入一致
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return {'out': x}
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        return {
            'backbone': 'none',
            'pretrained': False,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes})"
