#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DeepLabV3 Model"""


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


@register_model('deeplabv3')
class DeepLabV3(nn.Module):
    """
    基于 ResNet 骨干网络的 DeepLabV3.
    """
    
    SUPPORTED_BACKBONES = ('resnet50', 'resnet101')
    
    def __init__(
        self, 
        num_classes: int = 2, 
        backbone: str = 'resnet50', 
        pretrained: bool = True,
        pretrained_weights: Optional[str]= None, 
        freeze_bn: bool = True, 
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.freeze_bn = freeze_bn
        self.num_classes = num_classes
        self.backbone = backbone
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 本地权重优先级最高
        use_torchvision_pretrained = pretrained and pretrained_weights is None

        
        # 构建模型
        self.model = self._build_model(backbone, use_torchvision_pretrained)

        # 加载本地权重
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        
        # 修改分类头
        self._modify_classifier(num_classes)

        # 冻结 BN
        if self.freeze_bn:
            self._freeze_batchnorm()
        
        self.logger.info(f"DeepLabV3 初始化完成 | 骨干: {backbone} | 类别数: {num_classes} | 冻结BN: {freeze_bn}")
    
    def _build_model(self, backbone: str, pretrained: bool) -> nn.Module:
        """构建模型"""
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"不支持的骨干网络: {backbone}，"
                f"请选择: {self.SUPPORTED_BACKBONES}"
            )
        
        model_fn = getattr(torchvision.models.segmentation, f'deeplabv3_{backbone}')
        
        if pretrained:
            # 使用默认预训练权重
            self.logger.info("使用 COCO 预训练权重")
            return model_fn(weights='DEFAULT')
        else:
            # 从头训练
            self.logger.warning("不使用 COCO 预训练权重")
            return model_fn(weights=None, weights_backbone=None)
    
    def _load_pretrained(self, weights_path: str) -> None:
        """加载预训练权重"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        
        self.logger.info(f"成功加载预训练权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
    
    def _modify_classifier(self, num_classes: int) -> None:
        """修改分类头以适应自定义类别数"""
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)
    
    def train(self, mode: bool = True) -> DeepLabV3:
        """覆盖 train() 方法，可选地冻结 BatchNorm 层"""
        super().train(mode)
        if mode and self.freeze_bn:
            self._freeze_batchnorm()
        return self
    
    def _freeze_batchnorm(self) -> None:
        """冻结所有 BatchNorm 层"""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """获取默认配置"""
        return {
            'backbone': 'resnet50',
            'freeze_bn': True,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"backbone='{self.backbone}', "
            f"freeze_bn={self.freeze_bn})"
        )