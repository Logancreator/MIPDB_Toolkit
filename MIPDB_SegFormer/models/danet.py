#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DANet: Dual Attention Network for Scene Segmentation

Reference:
    Fu et al. "Dual Attention Network for Scene Segmentation"
    https://arxiv.org/abs/1809.02983
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


class PositionAttentionModule(nn.Module):
    """位置注意力模块 (PAM) - 捕获空间位置间的长程依赖"""
    
    def __init__(self, in_channels: int, reduction: int = 8) -> None:
        super().__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Query: B x C' x HW
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        # Key: B x C' x HW
        key = self.key_conv(x).view(B, -1, H * W)
        # Value: B x C x HW
        value = self.value_conv(x).view(B, -1, H * W)
        
        # Attention: B x HW x HW
        attention = self.softmax(torch.bmm(query, key))
        
        # Output: B x C x HW
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


class ChannelAttentionModule(nn.Module):
    """通道注意力模块 (CAM) - 捕获通道间的依赖关系"""
    
    def __init__(self) -> None:
        super().__init__()
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Query: B x C x HW
        query = x.view(B, C, -1)
        # Key: B x HW x C
        key = x.view(B, C, -1).permute(0, 2, 1)
        # Value: B x C x HW
        value = x.view(B, C, -1)
        
        # Attention: B x C x C
        attention = self.softmax(torch.bmm(query, key))
        
        # Output: B x C x HW
        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


class DualAttentionHead(nn.Module):
    """双注意力头：结合位置和通道注意力"""
    
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        
        inter_channels = in_channels // 4
        
        # 位置注意力分支
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.pam = PositionAttentionModule(inter_channels)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        
        # 通道注意力分支
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.cam = ChannelAttentionModule()
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        
        # 融合分类器
        self.conv_cls = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, num_classes, 1),
        )
        
        # 辅助分类器
        self.conv_aux_p = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, num_classes, 1),
        )
        self.conv_aux_c = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, num_classes, 1),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 位置注意力分支
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)
        
        # 通道注意力分支
        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)
        
        # 融合
        feat = feat_p + feat_c
        
        # 主输出
        out = self.conv_cls(feat)
        
        # 辅助输出
        out_p = self.conv_aux_p(feat_p)
        out_c = self.conv_aux_c(feat_c)
        
        return out, out_p, out_c


@register_model('danet')
class DANet(nn.Module):
    """
    DANet: 双注意力网络.
    
    特点：
    - 位置注意力模块 (PAM)：捕获空间长程依赖
    - 通道注意力模块 (CAM)：捕获通道间依赖
    - 自注意力机制增强特征表示
    - 深度监督训练
    
    Args:
        num_classes: 分割类别数
        backbone: 骨干网络（resnet50, resnet101）
        pretrained: 是否使用预训练权重
        pretrained_weights: 本地预训练权重路径
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
        
        # 修改 layer4 使用空洞卷积
        for name, module in self.layer4.named_modules():
            if 'conv2' in name:
                module.dilation = (2, 2)
                module.padding = (2, 2)
            elif 'downsample.0' in name:
                module.stride = (1, 1)
        
        # 双注意力头
        in_channels = 2048  # ResNet 最后一层输出通道数
        self.head = DualAttentionHead(in_channels, num_classes)
        
        # 加载本地权重
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        
        self.logger.info(f"DANet 初始化完成 | 骨干: {backbone} | 类别数: {num_classes}")
    
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
        
        # 双注意力头
        out, out_p, out_c = self.head(x)
        
        # 上采样
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        if self.training:
            out_p = F.interpolate(out_p, size=(h, w), mode='bilinear', align_corners=True)
            out_c = F.interpolate(out_c, size=(h, w), mode='bilinear', align_corners=True)
            # 辅助损失：取两个注意力分支输出的平均
            aux = (out_p + out_c) / 2
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
