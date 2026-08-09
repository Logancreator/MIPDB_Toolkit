#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SegFormer Model
"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model

try:
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


# 骨干网络参数
SEGFORMER_BACKBONE_CONFIGS: dict[str, dict[str, Any]] = {
    'b0': dict(
        hidden_sizes=[32, 64, 160, 256],
        depths=[2, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=256,
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],

        num_channels=3,
        num_encoder_blocks=4,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        hidden_act='gelu',
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        reshape_last_stage=True,
    ),
    'b1': dict(
        hidden_sizes=[64, 128, 320, 512],
        depths=[2, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=256,
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        num_channels=3, num_encoder_blocks=4,
        patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
        hidden_act='gelu', hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1,
        initializer_range=0.02, drop_path_rate=0.1,
        layer_norm_eps=1e-6, reshape_last_stage=True,
    ),
    'b2': dict(
        hidden_sizes=[64, 128, 320, 512],
        depths=[3, 4, 6, 3],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=768,
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        num_channels=3, num_encoder_blocks=4,
        patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
        hidden_act='gelu', hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1,
        initializer_range=0.02, drop_path_rate=0.1,
        layer_norm_eps=1e-6, reshape_last_stage=True,
    ),
    'b3': dict(
        hidden_sizes=[64, 128, 320, 512],
        depths=[3, 4, 18, 3],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=768,
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        num_channels=3, num_encoder_blocks=4,
        patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
        hidden_act='gelu', hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1,
        initializer_range=0.02, drop_path_rate=0.1,
        layer_norm_eps=1e-6, reshape_last_stage=True,
    ),
    'b4': dict(
        hidden_sizes=[64, 128, 320, 512],
        depths=[3, 8, 27, 3],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=768,
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        num_channels=3, num_encoder_blocks=4,
        patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
        hidden_act='gelu', hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1,
        initializer_range=0.02, drop_path_rate=0.1,
        layer_norm_eps=1e-6, reshape_last_stage=True,
    ),
    'b5': dict(
        hidden_sizes=[64, 128, 320, 512],
        depths=[3, 6, 40, 3],
        num_attention_heads=[1, 2, 5, 8],
        decoder_hidden_size=768,
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        num_channels=3, num_encoder_blocks=4,
        patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
        hidden_act='gelu', hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1,
        initializer_range=0.02, drop_path_rate=0.1,
        layer_norm_eps=1e-6, reshape_last_stage=True,
    ),
}


if HAS_TRANSFORMERS:
    @register_model('segformer')
    class SegFormer(nn.Module):
        """
        SegFormer - 基于 Transformer 的语义分割模型.
        
        Args:
            num_classes: 分割类别数
            backbone: 骨干网络 'b0'~'b5'
            pretrained: 是否在线加载预训练权重
            pretrained_weights: 本地 HF 模型目录（离线训练用）
            hf_config_dict: HF config 字典（从 checkpoint 恢复用，最可靠）
        """
        
        BACKBONE_MODEL_MAP: dict[str, str] = {
            'b0': 'nvidia/segformer-b0-finetuned-ade-512-512',
            'b1': 'nvidia/segformer-b1-finetuned-ade-512-512',
            'b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
            'b3': 'nvidia/segformer-b3-finetuned-ade-512-512',
            'b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
            'b5': 'nvidia/segformer-b5-finetuned-ade-640-640',
        }
        
        SUPPORTED_BACKBONES: tuple[str, ...] = tuple(BACKBONE_MODEL_MAP.keys())
        
        def __init__(
            self, 
            num_classes: int = 2, 
            backbone: str = 'b0', 
            pretrained: bool = True,
            pretrained_weights: Optional[str] = None,
            hf_config_dict: Optional[dict] = None,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            
            self.logger = get_logger()
            self.num_classes = num_classes
            self.backbone = backbone
            
            if backbone not in self.SUPPORTED_BACKBONES:
                raise ValueError(
                    f"不支持的骨干网络: {backbone}，请选择: {self.SUPPORTED_BACKBONES}"
                )
            
            self.model = self._load_model(
                backbone, num_classes, pretrained, pretrained_weights, hf_config_dict
            )
            
            self.logger.info(
                f"SegFormer 初始化完成 | 骨干: {backbone} | 类别数: {num_classes}"
            )
        
        def _load_model(
            self,
            backbone: str,
            num_classes: int,
            pretrained: bool,
            pretrained_weights: Optional[str],
            hf_config_dict: Optional[dict],
        ) -> SegformerForSemanticSegmentation:
            """加载 SegFormer 模型"""
            
            if pretrained_weights is not None:
                # 从本地目录加载（离线训练用）
                self.logger.info(f"从本地目录加载: {pretrained_weights}")
                return SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_weights,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    local_files_only=True,
                )
            
            elif pretrained:
                # 在线下载
                model_path = self.BACKBONE_MODEL_MAP[backbone]
                self.logger.info(f"在线加载预训练: {model_path}")
                return SegformerForSemanticSegmentation.from_pretrained(
                    model_path,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
            
            else:
                # 创建空模型
                if hf_config_dict is not None:
                    self.logger.info("从 checkpoint config 恢复模型")
                    config = SegformerConfig(**hf_config_dict)
                    config.num_labels = num_classes
                else:
                    self.logger.info(f"创建 SegFormer-{backbone}")
                    config = self._get_local_config(backbone, num_classes)
                
                return SegformerForSemanticSegmentation(config)
        
        @staticmethod
        def _get_local_config(backbone: str, num_classes: int) -> SegformerConfig:
            """用硬编码参数创建 config"""
            if backbone not in SEGFORMER_BACKBONE_CONFIGS:
                raise ValueError(f"未知的 backbone: {backbone}")
            cfg = SEGFORMER_BACKBONE_CONFIGS[backbone]
            return SegformerConfig(num_labels=num_classes, **cfg)
        
        def get_hf_config_dict(self) -> dict:
            """
            导出内部模型的config.
            
            调用方式: checkpoint['hf_config'] = model.get_hf_config_dict()
            """
            return self.model.config.to_dict()

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            input_size = x.size()[2:]
            logits = self.model(x).logits
            logits = F.interpolate(
                logits, size=input_size, mode='bilinear', align_corners=False
            )
            return {'out': logits}
        
        @staticmethod
        def get_default_config() -> dict[str, Any]:
            return {
                'backbone': 'b0',
                'pretrained': True,
                'pretrained_weights': None,
            }
        
        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}("
                f"num_classes={self.num_classes}, backbone='{self.backbone}')"
            )

else:
    class SegFormer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "SegFormer 需要 'transformers' 库。\n"
                "pip install transformers"
            )
