#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PSPNet (Pyramid Scene Parsing Network)"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from . import register_model


def get_logger() -> logging.Logger:
    """获取当前模块的日志记录器"""
    return logging.getLogger(__name__)


class PyramidPoolingModule(nn.Module):
    """
    金字塔池化模块
    """
    
    def __init__(
        self, 
        in_channels: int, 
        pool_sizes: Optional[List[int]] = None, 
        out_channels: int = 512
    ) -> None:
        super().__init__()
        
        if pool_sizes is None:
            pool_sizes = [1, 2, 3, 6]
        
        self.pool_sizes = pool_sizes
        
        # 多尺度池化分支
        self.stages = nn.ModuleList()
        for pool_size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 瓶颈层：融合所有特征
        bottleneck_in_channels = in_channels + len(pool_sizes) * out_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        
        # 收集原始特征和各尺度池化特征
        pyramids = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)
            pyramids.append(upsampled)
        
        # 拼接并融合
        output = torch.cat(pyramids, dim=1)
        output = self.bottleneck(output)
        
        return output


@register_model('pspnet')
class PSPNet(nn.Module):
    """
    PSPNet
    
    论文: Pyramid Scene Parsing Network (CVPR 2017)
    https://arxiv.org/abs/1612.01105
    """
    
    SUPPORTED_BACKBONES: tuple[str, ...] = ('resnet50', 'resnet101')
    BACKBONE_CHANNELS: dict[str, int] = {
        'resnet50': 2048,
        'resnet101': 2048,
    }
    
    def __init__(
        self, 
        num_classes: int = 2, 
        backbone: str = 'resnet50', 
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        pool_sizes: Optional[list[int]] = None,
        use_aux: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_aux = use_aux
        
        if pool_sizes is None:
            pool_sizes = [1, 2, 3, 6]
        
        # 验证骨干网络
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"不支持的骨干网络: {backbone}，"
                f"请选择: {self.SUPPORTED_BACKBONES}"
            )
        
        # 构建骨干网络
        in_channels = self._build_backbone(backbone, pretrained, pretrained_weights)
        
        # 金字塔池化模块
        self.ppm = PyramidPoolingModule(in_channels, pool_sizes, out_channels=512)
        
        # 主分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
        # 辅助分类头（用于深度监督）
        if use_aux:
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )
        else:
            self.aux_classifier = None
        
        self.logger.info(
            f"PSPNet 初始化完成 | 骨干: {backbone} | 类别数: {num_classes} | "
            f"池化尺寸: {pool_sizes} | 辅助头: {use_aux}"
        )
    
    def _build_backbone(
        self, 
        backbone: str = 'resnet50', 
        pretrained: bool = False,
        pretrained_weights: Optional[str] = None
    ) -> int:
        """
        构建骨干网络.
        
        Returns:
            骨干网络输出通道数
        """
        resnet_fn = getattr(models, backbone)
    
        if pretrained_weights is not None:
            # 使用本地权重
            self.logger.info(f"从本地加载权重: {pretrained_weights}")
            resnet = resnet_fn(weights=None)
            state_dict = torch.load(pretrained_weights, map_location='cpu', weights_only=True)
            resnet.load_state_dict(state_dict, strict=False)
        elif pretrained:
            # 使用 ImageNet 预训练
            self.logger.info("使用 ImageNet 预训练权重")
            resnet = resnet_fn(weights='DEFAULT')
        else:
            # 从头训练
            self.logger.warning("不使用预训练权重，从头开始训练!")
            resnet = resnet_fn(weights=None)
    
        # 提取各层
        self.layer0 = nn.Sequential(
            resnet.conv1, 
            resnet.bn1, 
            resnet.relu, 
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 修改 layer3 和 layer4 为空洞卷积
        self._modify_for_dilated_conv()
        
        return self.BACKBONE_CHANNELS[backbone]
    
    def _modify_for_dilated_conv(self) -> None:
        """将 layer3 和 layer4 修改为空洞卷积，保持特征图分辨率"""
        # Layer3: dilation=2
        for name, module in self.layer3.named_modules():
            if 'conv2' in name:
                module.dilation = (2, 2)
                module.padding = (2, 2)
                module.stride = (1, 1)
            elif 'downsample.0' in name:
                module.stride = (1, 1)
        
        # Layer4: dilation=4
        for name, module in self.layer4.named_modules():
            if 'conv2' in name:
                module.dilation = (4, 4)
                module.padding = (4, 4)
                module.stride = (1, 1)
            elif 'downsample.0' in name:
                module.stride = (1, 1)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.size()[2:]
        
        # 骨干网络前向传播
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)  # 用于辅助损失
        x = self.layer4(x_aux)
        
        # 金字塔池化
        x = self.ppm(x)
        
        # 主输出
        out = self.classifier(x)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        # 辅助输出（仅训练时）
        if self.training and self.aux_classifier is not None:
            aux = self.aux_classifier(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=True)
            return {'out': out, 'aux': aux}
        
        return {'out': out}
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """获取默认配置"""
        return {
            'backbone': 'resnet50',
            'pretrained': True,
            'pretrained_weights': None,
            'pool_sizes': [1, 2, 3, 6],
            'use_aux': True,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"backbone='{self.backbone_name}', "
            f"use_aux={self.use_aux})"
        )