#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCRNet: Object-Contextual Representations for Semantic Segmentation

Reference:
    Yuan et al. "Object-Contextual Representations for Semantic Segmentation"
    https://arxiv.org/abs/1909.11065
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


class SpatialGatherModule(nn.Module):
    """空间聚集模块：根据软目标区域聚集像素特征"""
    
    def __init__(self, scale: int = 1) -> None:
        super().__init__()
        self.scale = scale
    
    def forward(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feats.shape
        K = probs.shape[1]
        
        # 归一化 probs
        probs = probs.view(B, K, -1)  # [B, K, HW]
        probs = F.softmax(self.scale * probs, dim=2)
        
        # 特征聚集
        feats = feats.view(B, C, -1)  # [B, C, HW]
        feats = feats.permute(0, 2, 1)  # [B, HW, C]
        
        # 加权聚合
        ocr_context = torch.bmm(probs, feats)  # [B, K, C]
        
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    """目标注意力块：计算像素与目标区域的关系"""
    
    def __init__(self, in_channels: int, key_channels: int, scale: int = 1) -> None:
        super().__init__()
        
        self.scale = scale
        self.key_channels = key_channels
        
        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, feats: torch.Tensor, ocr_context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feats.shape
        
        # Query from pixel features
        query = self.query_conv(feats)  # [B, key_C, H, W]
        query = query.view(B, self.key_channels, -1)  # [B, key_C, HW]
        query = query.permute(0, 2, 1)  # [B, HW, key_C]
        
        # Key from object context
        # 将 ocr_context 转换为适合卷积的形状
        K = ocr_context.shape[1]
        key = ocr_context.permute(0, 2, 1).unsqueeze(-1)  # [B, C, K, 1]
        key = key.view(B, C, K, 1).expand(B, C, K, 1)
        key = self.key_conv(key.view(B, C, K, 1).squeeze(-1).unsqueeze(-1).unsqueeze(-1))
        key = key.view(B, self.key_channels, -1)  # [B, key_C, K]
        
        # Value from object context
        value = ocr_context.permute(0, 2, 1).unsqueeze(-1)  # [B, C, K, 1]
        value = self.value_conv(value.view(B, C, K, 1).squeeze(-1).unsqueeze(-1).unsqueeze(-1))
        value = value.view(B, self.key_channels, -1)  # [B, key_C, K]
        value = value.permute(0, 2, 1)  # [B, K, key_C]
        
        # Attention
        sim = torch.bmm(query, key)  # [B, HW, K]
        sim = (self.key_channels ** -0.5) * sim
        sim = F.softmax(sim, dim=-1)
        
        # Context
        context = torch.bmm(sim, value)  # [B, HW, key_C]
        context = context.permute(0, 2, 1)  # [B, key_C, HW]
        context = context.view(B, self.key_channels, H, W)
        context = self.out_conv(context)
        
        return context


class OCRModule(nn.Module):
    """OCR 模块：完整的目标上下文表示"""
    
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int) -> None:
        super().__init__()
        
        self.soft_object_regions = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1),
        )
        
        self.pixel_representation = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        self.spatial_gather = SpatialGatherModule()
        
        self.object_context_block = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # 简化的注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        self.cls = nn.Conv2d(mid_channels, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 软目标区域
        soft_regions = self.soft_object_regions(x)
        
        # 像素表示
        pixel_repr = self.pixel_representation(x)
        
        # 聚集目标上下文
        B, C, H, W = pixel_repr.shape
        K = soft_regions.shape[1]
        
        # 简化的上下文聚合
        soft_regions_flat = soft_regions.view(B, K, -1)
        soft_regions_flat = F.softmax(soft_regions_flat, dim=2)
        
        pixel_repr_flat = pixel_repr.view(B, C, -1).permute(0, 2, 1)
        ocr_context = torch.bmm(soft_regions_flat, pixel_repr_flat)  # [B, K, C]
        
        # 目标上下文表示
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(-1)  # [B, C, K, 1]
        ocr_context = self.object_context_block(ocr_context.squeeze(-1).unsqueeze(-1).unsqueeze(-1))
        
        # 广播并与像素特征融合
        ocr_context = ocr_context.expand(-1, -1, H, W)
        
        # 融合
        aug_repr = torch.cat([pixel_repr, ocr_context], dim=1)
        aug_repr = self.attention(aug_repr)
        
        # 分类
        out = self.cls(aug_repr)
        
        return out, soft_regions


@register_model('ocrnet')
class OCRNet(nn.Module):
    """
    OCRNet
    """
    
    SUPPORTED_BACKBONES = ('resnet50', 'resnet101')
    
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
        
        # 编码器
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 通道数
        in_channels = 2048
        mid_channels = 512
        
        # OCR 模块
        self.ocr = OCRModule(in_channels, mid_channels, num_classes)
        
        # 加载本地权重
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        
        self.logger.info(f"OCRNet 初始化完成 | 骨干: {backbone} | 类别数: {num_classes}")
    
    def _load_pretrained(self, weights_path: str) -> None:
        import os
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        self.logger.info(f"加载本地权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        
        # 编码器
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # OCR 模块
        out, aux = self.ocr(x)
        
        # 上采样
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        if self.training:
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            return {'out': out, 'aux': aux}
        
        return {'out': out}
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        return {
            'backbone': 'resnet50',
            'pretrained': True,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes}, backbone='{self.backbone_name}')"
