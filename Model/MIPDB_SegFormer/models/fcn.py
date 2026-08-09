#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FCN"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torchvision

from . import register_model


def get_logger() -> logging.Logger:
    """获取当前模块的日志记录器"""
    return logging.getLogger(__name__)


@register_model('fcn')
class FCN(nn.Module):
    """
    FCN
    """
    
    SUPPORTED_BACKBONES = ('resnet50', 'resnet101')
    
    def __init__(
        self, 
        num_classes: int = 2, 
        backbone: str = 'resnet50',
        pretrained: bool = True,  # ← 添加
        pretrained_weights: Optional[str] = None, 
        **kwargs: Any
    ) -> None:
        super().__init__()

        self.logger = get_logger()
        self.num_classes = num_classes
        self.backbone = backbone
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"不支持的骨干网络: {backbone}，"
                f"请选择: {self.SUPPORTED_BACKBONES}"
            )
        
        # 本地权重优先级最高
        use_torchvision_pretrained = pretrained and pretrained_weights is None

        
        # 构建模型
        self.model = self._build_model(backbone, use_torchvision_pretrained)
        
        # 加载本地权重
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        
        # 修改分类头
        self._modify_classifier(num_classes)
        
        self.logger.info(
            f"FCN 初始化完成 | 骨干: {backbone} | "
            f"类别数: {num_classes} | 预训练: {pretrained}"
        )
    
    def _build_model(self, backbone: str, pretrained: bool) -> nn.Module:
        """构建模型"""
        model_fn = getattr(torchvision.models.segmentation, f'fcn_{backbone}')
    
        if pretrained:
            self.logger.info("使用 COCO 预训练权重")
            return model_fn(weights='DEFAULT')
        else:
            self.logger.warning("不使用 COCO 预训练权重")
            return model_fn(weights=None, weights_backbone=None)
    
    def _load_pretrained(self, weights_path: str) -> None:
        """加载本地预训练权重"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        self.logger.info(f"加载本地权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
    
    def _modify_classifier(self, num_classes: int) -> None:
        """修改分类头"""
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, 1)
        
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier:
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """获取默认配置"""
        return {
            'backbone': 'resnet50',
            'pretrained': True,
            'pretrained_weights': None,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"backbone='{self.backbone}')"
        )