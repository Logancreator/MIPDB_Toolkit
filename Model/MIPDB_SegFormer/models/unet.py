#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UNet Model"""

from __future__ import annotations

import os
import logging
from token import OP
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
# UNet 构建块
# ============================================================

class DoubleConv(nn.Module):
    """
    双卷积块: (Conv -> BN -> ReLU) × 2
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mid_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样块: MaxPool -> DoubleConv
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样块: Upsample/ConvTranspose -> Concat -> DoubleConv
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bilinear: bool = True
    ) -> None:
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        x1 = self.up(x1)
        
        # 处理尺寸不匹配（填充）
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2
        ])
        
        # 拼接跳跃连接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出卷积层: 1×1 卷积
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================
# UNet 模型
# ============================================================

@register_model('unet')
class UNet(nn.Module):
    """
    UNet 语义分割模型.
    
    """
    
    def __init__(
        self, 
        num_classes: int = 2, 
        in_channels: int = 3, 
        base_channels: int = 64, 
        bilinear: bool = True,
        pretrained: bool = True,  # ← 添加（仅为接口统一）
        pretrained_weights: str | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear
        
        # UNet 无 ImageNet 预训练
        if pretrained and pretrained_weights is None:
            self.logger.warning("UNet 无 ImageNet 预训练，将从头开始训练")
        
        # 通道数计算因子
        factor = 2 if bilinear else 1
        
        # 编码器
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # 解码器
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # 输出层
        self.outc = OutConv(base_channels, num_classes)
        
        # 加载预训练权重
        if pretrained_weights is not None:
            self._load_pretrained(pretrained_weights)
        
        self.logger.info(
            f"UNet 初始化完成 | 输入通道: {in_channels} | 类别数: {num_classes} | "
            f"基础通道: {base_channels} | 双线性插值: {bilinear}"
        )
    
    def _load_pretrained(self, weights_path: str) -> None:
        """加载预训练权重"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        self.logger.info(f"加载预训练权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return {'out': logits}
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        return {
            'in_channels': 3,
            'base_channels': 64,
            'bilinear': True,
            'pretrained': True,
            'pretrained_weights': None,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"in_channels={self.in_channels}, "
            f"base_channels={self.base_channels}, "
            f"bilinear={self.bilinear})"
        )

# ============================================================
# UNet with ResNet Encoder
# ============================================================

@register_model('unet_resnet')
class UNetResNet(nn.Module):
    """
    使用 ResNet 编码器的 UNet.
    """
    
    SUPPORTED_BACKBONES: tuple[str, ...] = ('resnet34', 'resnet50', 'resnet101')
    
    # 不同 ResNet 的通道配置 [layer0, layer1, layer2, layer3, layer4]
    ENCODER_CHANNELS: dict[str, list[int]] = {
        'resnet34': [64, 64, 128, 256, 512],
        'resnet50': [64, 256, 512, 1024, 2048],
        'resnet101': [64, 256, 512, 1024, 2048],
    }
    
    def __init__(
        self, 
        num_classes: int = 2, 
        backbone: str = 'resnet34', 
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # 验证骨干网络
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"不支持的骨干网络: {backbone}，"
                f"请选择: {self.SUPPORTED_BACKBONES}"
            )
        
        # 获取通道配置
        encoder_channels = self.ENCODER_CHANNELS[backbone]
        
        # 构建编码器
        self._build_encoder(backbone, pretrained, pretrained_weights)
        
        # 构建解码器
        self.decoder4 = self._decoder_block(encoder_channels[4], encoder_channels[3])
        self.decoder3 = self._decoder_block(encoder_channels[3], encoder_channels[2])
        self.decoder2 = self._decoder_block(encoder_channels[2], encoder_channels[1])
        self.decoder1 = self._decoder_block(encoder_channels[1], encoder_channels[0])
        self.decoder0 = self._decoder_block(encoder_channels[0], 64)
        
        # 输出层
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.logger.info(
            f"UNetResNet 初始化完成 | 骨干: {backbone} | 类别数: {num_classes}"
        )
    
    def _build_encoder(
        self, 
        backbone: str, 
        pretrained: bool,
        pretrained_weights: Optional[str] = None
    ) -> None:
        """构建 ResNet 编码器"""
        # 确定权重加载方式
        resnet_fn = getattr(models, backbone)
    
        if pretrained_weights is not None:
            self.logger.info(f"从本地加载编码器权重: {pretrained_weights}")
            encoder = resnet_fn(weights=None)
            state_dict = torch.load(pretrained_weights, map_location='cpu', weights_only=True)
            encoder.load_state_dict(state_dict, strict=False)
        elif pretrained:
            self.logger.info("使用 ImageNet 预训练权重")
            encoder = resnet_fn(weights='DEFAULT')
        else:
            self.logger.warning("不使用预训练权重，从头开始训练!")
            encoder = resnet_fn(weights=None)
        
        # 提取编码器各层
        self.encoder0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.encoder1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4
    
    def _decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """构建解码器块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[2:]
        
        # 编码器（下采样）
        e0 = self.encoder0(x)   # 1/2
        e1 = self.encoder1(e0)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/32
        
        # 解码器（上采样 + 跳跃连接）
        d4 = self.decoder4(
            F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=True) + e3
        )
        d3 = self.decoder3(
            F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=True) + e2
        )
        d2 = self.decoder2(
            F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True) + e1
        )
        d1 = self.decoder1(
            F.interpolate(d2, size=e0.shape[2:], mode='bilinear', align_corners=True) + e0
        )
        d0 = self.decoder0(
            F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        )
        
        # 输出
        out = self.final(d0)
        
        return {'out': out}
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """获取默认配置"""
        return {
            'backbone': 'resnet34',
            'pretrained': True,
            'pretrained_weights': None,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"backbone='{self.backbone_name}')"
        )