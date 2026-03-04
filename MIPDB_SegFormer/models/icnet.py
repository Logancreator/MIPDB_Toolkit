#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICNet: Image Cascade Network for Real-time Semantic Segmentation

Reference:
    Zhao et al. "ICNet for Real-Time Semantic Segmentation on High-Resolution Images"
    https://arxiv.org/abs/1704.08545
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from . import register_model


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


class CascadeFeatureFusion(nn.Module):
    """级联特征融合模块 (CFF)"""
    
    def __init__(self, low_channels: int, high_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        
        # 低分辨率分支：1x1 卷积 + 上采样
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # 高分辨率分支：空洞卷积
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # 辅助分类器（用于深度监督）
        self.conv_cls = nn.Conv2d(out_channels, num_classes, 1)
        
    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 上采样低分辨率特征
        x_low = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        
        # 处理高分辨率特征
        x_high = self.conv_high(x_high)
        
        # 融合
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        
        # 辅助输出
        aux = self.conv_cls(x)
        
        return x, aux


class PyramidPoolingModule(nn.Module):
    """金字塔池化模块 (PSPNet style)"""
    
    def __init__(self, in_channels: int, out_channels: int, pool_sizes: tuple = (1, 2, 3, 6)) -> None:
        super().__init__()
        
        self.stages = nn.ModuleList()
        for pool_size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(out_channels // len(pool_sizes)),
                nn.ReLU(inplace=True),
            ))
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        pyramids = [x]
        
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            pyramids.append(out)
        
        return self.bottleneck(torch.cat(pyramids, dim=1))


@register_model('icnet')
class ICNet(nn.Module):
    """
    ICNet
    """
    
    SUPPORTED_BACKBONES = ('resnet18', 'resnet34', 'resnet50')
    
    def __init__(
        self, 
        num_classes: int = 2,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"不支持的骨干网络: {backbone}，请选择: {self.SUPPORTED_BACKBONES}")
        
        # 加载骨干网络
        backbone_fn = getattr(models, backbone)
        if pretrained and pretrained_weights is None:
            self.logger.info("使用 ImageNet 预训练权重")
            resnet = backbone_fn(weights='DEFAULT')
        else:
            resnet = backbone_fn(weights=None)
        
        # 构建多尺度分支
        # Branch 1: 1/4 分辨率 (conv1 -> layer1)
        self.conv_sub1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Branch 2: 1/8 分辨率 (使用 resnet 的前几层)
        self.conv_sub2 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        
        # Branch 4: 1/32 分辨率 (完整 resnet + PPM)
        self.conv_sub4 = nn.Sequential(
            resnet.layer3,
            resnet.layer4,
        )
        
        # 获取通道数
        if backbone in ('resnet18', 'resnet34'):
            high_channels = 128   # layer2 输出
            low_channels = 512    # layer4 输出
            mid_channels = 256
        else:  # resnet50
            high_channels = 512   # layer2 输出
            low_channels = 2048   # layer4 输出
            mid_channels = 512
        
        # 金字塔池化
        self.ppm = PyramidPoolingModule(low_channels, mid_channels)
        
        # 级联特征融合
        self.cff_42 = CascadeFeatureFusion(mid_channels, high_channels, mid_channels, num_classes)
        self.cff_21 = CascadeFeatureFusion(mid_channels, 64, 128, num_classes)
        
        # 最终分类器
        self.conv_cls = nn.Conv2d(128, num_classes, 1)
        
        # 加载本地权重
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        
        self.logger.info(f"ICNet 初始化完成 | 骨干: {backbone} | 类别数: {num_classes}")
    
    def _load_pretrained(self, weights_path: str) -> None:
        import os
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        self.logger.info(f"加载本地权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        
        # 多尺度输入
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        
        # Branch 1: 1/8 分辨率
        x1 = self.conv_sub1(x)
        
        # Branch 2: 1/16 分辨率
        x2 = self.conv_sub2(x_sub2)
        
        # Branch 4: 1/32 分辨率 + PPM
        x4 = self.conv_sub2(x_sub4)
        x4 = self.conv_sub4(x4)
        x4 = self.ppm(x4)
        
        # 级联融合
        x42, aux_42 = self.cff_42(x4, x2)
        x21, aux_21 = self.cff_21(x42, x1)
        
        # 最终输出
        out = self.conv_cls(x21)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        if self.training:
            aux_42 = F.interpolate(aux_42, size=(h, w), mode='bilinear', align_corners=True)
            aux_21 = F.interpolate(aux_21, size=(h, w), mode='bilinear', align_corners=True)
            return {'out': out, 'aux': aux_21, 'aux2': aux_42}
        
        return {'out': out}
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        return {
            'backbone': 'resnet50',
            'pretrained': True,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes}, backbone='{self.backbone_name}')"
