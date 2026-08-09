#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BiSeNet

论文: BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation
https://arxiv.org/abs/1808.00897
"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from . import register_model


def get_logger() -> logging.Logger:
    """获取当前模块的日志记录器"""
    return logging.getLogger(__name__)


# ============================================================
# BiSeNet 构建块
# ============================================================

class ConvBNReLU(nn.Module):
    """Conv + BatchNorm + ReLU"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SpatialPath(nn.Module):
    """
    空间路径 (Spatial Path).
    
    保留空间信息，使用三个卷积层逐步降采样。
    输出分辨率为输入的 1/8。
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        
        inner_channels = 64
        
        # 三个下采样卷积块
        self.conv1 = ConvBNReLU(in_channels, inner_channels, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(inner_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # 1/2
        x = self.conv2(x)  # 1/4
        x = self.conv3(x)  # 1/8
        return x


class AttentionRefinementModule(nn.Module):
    """
    注意力精炼模块 (ARM - Attention Refinement Module).
    
    使用全局平均池化和注意力机制来精炼特征。
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 注意力分支
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        attention = self.attention(x)
        return x * attention


class ContextPath(nn.Module):
    """
    上下文路径 (Context Path).
    
    使用预训练的轻量级骨干网络提取上下文信息。
    支持 ResNet18 和 ResNet50 作为骨干。
    """
    
    SUPPORTED_BACKBONES = ('resnet18', 'resnet34', 'resnet50', 'resnet101')
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        out_channels: int = 128
    ) -> None:
        super().__init__()
        
        self.backbone_name = backbone
        
        # 加载骨干网络
        if backbone == 'resnet18':
            resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
            stage_channels = [256, 512]
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights='DEFAULT' if pretrained else None)
            stage_channels = [256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights='DEFAULT' if pretrained else None)
            stage_channels = [1024, 2048]
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights='DEFAULT' if pretrained else None)
            stage_channels = [1024, 2048]
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 提取骨干网络各阶段
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 注意力精炼模块
        self.arm16 = AttentionRefinementModule(stage_channels[0], out_channels)
        self.arm32 = AttentionRefinementModule(stage_channels[1], out_channels)
        
        # 全局平均池化分支
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(stage_channels[1], out_channels, kernel_size=1, padding=0)
        )
        
        # 输出通道
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feat16: 1/16 特征
            feat32: 1/32 特征（融合了全局上下文）
        """
        # 骨干网络前向传播
        x = self.layer0(x)   # 1/4
        x = self.layer1(x)   # 1/4
        x = self.layer2(x)   # 1/8
        feat16 = self.layer3(x)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        
        # 全局上下文
        global_context = self.global_context(feat32)
        global_context = F.interpolate(
            global_context, size=feat32.shape[2:],
            mode='bilinear', align_corners=True
        )
        
        # ARM 处理
        feat32 = self.arm32(feat32)
        feat32 = feat32 + global_context  # 融合全局上下文
        
        feat16 = self.arm16(feat16)
        
        return feat16, feat32


class FeatureFusionModule(nn.Module):
    """
    特征融合模块 (FFM - Feature Fusion Module).
    
    融合空间路径和上下文路径的特征。
    """
    
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 1) -> None:
        super().__init__()
        
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        
        # 注意力分支
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, sp_feat: torch.Tensor, cp_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sp_feat: 空间路径特征
            cp_feat: 上下文路径特征
        """
        # 拼接特征
        x = torch.cat([sp_feat, cp_feat], dim=1)
        x = self.conv1(x)
        
        # 注意力加权
        attention = self.attention(x)
        x_atten = x * attention
        
        return x + x_atten


class BiSeNetHead(nn.Module):
    """BiSeNet 分割头"""
    
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int) -> None:
        super().__init__()
        
        self.conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.conv_out(x)
        return x


# ============================================================
# BiSeNet 模型
# ============================================================

@register_model('bisenet')
class BiSeNet(nn.Module):
    """
    BiSeNet.
    
    论文: BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation
    https://arxiv.org/abs/1808.00897
    """
    
    SUPPORTED_BACKBONES: tuple[str, ...] = ('resnet18', 'resnet34', 'resnet50', 'resnet101')
    
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        use_aux: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_aux = use_aux
        
        # 验证骨干网络
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"不支持的骨干网络: {backbone}，"
                f"请选择: {self.SUPPORTED_BACKBONES}"
            )
        
        # 本地权重优先级最高
        use_pretrained = pretrained and pretrained_weights is None
        
        # 特征通道数
        sp_channels = 128
        cp_channels = 128
        
        # 空间路径
        self.spatial_path = SpatialPath(in_channels=3, out_channels=sp_channels)
        
        # 上下文路径
        self.context_path = ContextPath(
            backbone=backbone,
            pretrained=use_pretrained,
            out_channels=cp_channels
        )
        
        # 特征融合模块
        self.ffm = FeatureFusionModule(
            in_channels=sp_channels + cp_channels,
            out_channels=256
        )
        
        # 主分割头
        self.head = BiSeNetHead(
            in_channels=256,
            mid_channels=64,
            num_classes=num_classes
        )
        
        # 辅助分割头（深度监督）
        if use_aux:
            self.aux_head16 = BiSeNetHead(
                in_channels=cp_channels,
                mid_channels=64,
                num_classes=num_classes
            )
            self.aux_head32 = BiSeNetHead(
                in_channels=cp_channels,
                mid_channels=64,
                num_classes=num_classes
            )
        else:
            self.aux_head16 = None
            self.aux_head32 = None
        
        # 加载本地权重
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        
        # 初始化新增层的权重
        self._init_weights()
        
        self.logger.info(
            f"BiSeNet 初始化完成 | 骨干: {backbone} | "
            f"类别数: {num_classes} | 辅助头: {use_aux} | 预训练: {pretrained}"
        )
    
    def _init_weights(self) -> None:
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained(self, weights_path: str) -> None:
        """加载本地预训练权重"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        self.logger.info(f"加载本地权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        前向传播.
        
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            dict: 包含 'out' 的字典，训练时还包含 'aux1' 和 'aux2'
        """
        input_size = x.shape[2:]
        
        # 空间路径：保留空间细节
        sp_feat = self.spatial_path(x)  # 1/8
        
        # 上下文路径：提取语义信息
        feat16, feat32 = self.context_path(x)  # 1/16, 1/32
        
        # 上采样上下文特征到 1/8
        feat16_up = F.interpolate(
            feat16, size=sp_feat.shape[2:],
            mode='bilinear', align_corners=True
        )
        feat32_up = F.interpolate(
            feat32, size=sp_feat.shape[2:],
            mode='bilinear', align_corners=True
        )
        
        # 融合上下文特征
        cp_feat = feat16_up + feat32_up
        
        # 特征融合
        fused = self.ffm(sp_feat, cp_feat)
        
        # 主输出
        out = self.head(fused)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        result = {'out': out}
        
        # 辅助输出（仅训练时）
        if self.training and self.use_aux:
            aux1 = self.aux_head16(feat16)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
            
            aux2 = self.aux_head32(feat32)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
            
            # 合并辅助损失（加权平均）
            result['aux'] = (aux1 + aux2) / 2
        
        return result
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """获取默认配置"""
        return {
            'backbone': 'resnet18',
            'pretrained': True,
            'pretrained_weights': None,
            'use_aux': True,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"backbone='{self.backbone_name}', "
            f"use_aux={self.use_aux})"
        )