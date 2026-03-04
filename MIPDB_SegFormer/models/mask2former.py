#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mask2Former Model
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
    from transformers import (
        Mask2FormerForUniversalSegmentation,
        Mask2FormerConfig,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


if HAS_TRANSFORMERS:
    @register_model('mask2former')
    class Mask2Former(nn.Module):
        """
        Mask2Former.
        """
        
        BACKBONE_MODEL_MAP: dict[str, str] = {
            'swin-tiny': 'facebook/mask2former-swin-tiny-ade-semantic',
            'swin-small': 'facebook/mask2former-swin-small-ade-semantic',
            'swin-base': 'facebook/mask2former-swin-base-ade-semantic',
            'swin-large': 'facebook/mask2former-swin-large-ade-semantic',
        }
        
        BACKBONE_MODEL_MAP_COCO: dict[str, str] = {
            'swin-tiny': 'facebook/mask2former-swin-tiny-coco-panoptic',
            'swin-small': 'facebook/mask2former-swin-small-coco-panoptic',
            'swin-base': 'facebook/mask2former-swin-base-IN21k-coco-panoptic',
            'swin-large': 'facebook/mask2former-swin-large-coco-panoptic',
        }
            
        SUPPORTED_BACKBONES: tuple[str, ...] = tuple(BACKBONE_MODEL_MAP.keys())
        
        def __init__(
            self,
            num_classes: int = 2,
            backbone: str = 'swin-tiny',
            pretrained: bool = True,
            pretrained_weights: Optional[str] = None,
            hf_config_dict: Optional[dict] = None,
            use_coco_pretrained: bool = False,
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
            
            model_map = self.BACKBONE_MODEL_MAP_COCO if use_coco_pretrained else self.BACKBONE_MODEL_MAP
            
            self.model = self._load_model(
                backbone, num_classes, pretrained, pretrained_weights,
                hf_config_dict, model_map,
            )
            
            self.logger.info(
                f"Mask2Former 初始化完成 | 骨干: {backbone} | 类别数: {num_classes}"
            )
        
        def _load_model(
            self,
            backbone: str,
            num_classes: int,
            pretrained: bool,
            pretrained_weights: Optional[str],
            hf_config_dict: Optional[dict],
            model_map: dict[str, str],
        ) -> Mask2FormerForUniversalSegmentation:
            
            if pretrained_weights is not None:
                if not os.path.exists(pretrained_weights):
                    raise FileNotFoundError(f"权重路径不存在: {pretrained_weights}")
                self.logger.info(f"从本地 HF 目录加载: {pretrained_weights}")
                config = Mask2FormerConfig.from_pretrained(pretrained_weights)
                config.num_labels = num_classes
                return Mask2FormerForUniversalSegmentation.from_pretrained(
                    pretrained_weights, config=config,
                    ignore_mismatched_sizes=True, local_files_only=True,
                )
                
            elif pretrained:
                model_name = model_map.get(backbone, model_map['swin-tiny'])
                self.logger.info(f"在线加载预训练: {model_name}")
                return Mask2FormerForUniversalSegmentation.from_pretrained(
                    model_name, num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
                
            else:
                # 创建空模型
                if hf_config_dict is not None:
                    self.logger.info("从 checkpoint config 恢复模型")
                    config = Mask2FormerConfig.from_dict(hf_config_dict)
                    config.num_labels = num_classes
                else:
                    model_name = model_map.get(backbone, model_map['swin-tiny'])
                    self.logger.warning(
                        f"checkpoint  hf_config，尝试在线获取 {backbone} 配置..."
                    )
                    try:
                        config = Mask2FormerConfig.from_pretrained(model_name)
                    except Exception as e:
                        raise RuntimeError(
                            f"无法获取 Mask2Former-{backbone} 配置: {e}\n"
                        ) from e
                    config.num_labels = num_classes
                
                return Mask2FormerForUniversalSegmentation(config)
        
        def get_hf_config_dict(self) -> dict:
            """导出 HF config 用于保存到 checkpoint"""
            return self.model.config.to_dict()

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            input_size = x.shape[2:]
            outputs = self.model(pixel_values=x)
            
            masks = outputs.masks_queries_logits
            class_logits = outputs.class_queries_logits
            
            masks = F.interpolate(
                masks, size=input_size, mode='bilinear', align_corners=False
            )
            
            class_probs = F.softmax(class_logits, dim=-1)[..., :-1]
            masks_softmax = F.softmax(masks, dim=1)
            semantic_logits = torch.einsum('bqhw,bqc->bchw', masks_softmax, class_probs)
            
            return {'out': semantic_logits}
        
        def forward_with_loss(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
            input_size = x.shape[2:]
            batch_size = x.shape[0]
            
            class_labels_list = []
            mask_labels_list = []
            
            for b in range(batch_size):
                label = labels[b]
                unique_classes = torch.unique(label)
                cls_labels = []
                msk_labels = []
                for cls in unique_classes:
                    if cls < self.num_classes:
                        cls_labels.append(cls)
                        msk_labels.append((label == cls).float())
                if len(cls_labels) > 0:
                    class_labels_list.append(torch.stack([torch.tensor(c) for c in cls_labels]))
                    mask_labels_list.append(torch.stack(msk_labels))
                else:
                    class_labels_list.append(torch.tensor([0]))
                    mask_labels_list.append(torch.zeros(1, *label.shape))
            
            outputs = self.model(
                pixel_values=x,
                class_labels=class_labels_list,
                mask_labels=mask_labels_list,
            )
            
            masks = outputs.masks_queries_logits
            class_logits = outputs.class_queries_logits
            masks = F.interpolate(masks, size=input_size, mode='bilinear', align_corners=False)
            class_probs = F.softmax(class_logits, dim=-1)[..., :-1]
            masks_softmax = F.softmax(masks, dim=1)
            semantic_logits = torch.einsum('bqhw,bqc->bchw', masks_softmax, class_probs)
            
            return {'out': semantic_logits, 'loss': outputs.loss}
        
        @staticmethod
        def get_default_config() -> dict[str, Any]:
            return {
                'backbone': 'swin-tiny',
                'pretrained': True,
                'pretrained_weights': None,
                'use_coco_pretrained': False,
            }
        
        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}("
                f"num_classes={self.num_classes}, backbone='{self.backbone}')"
            )


else:
    class Mask2Former:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "Mask2Former 需要 'transformers' 库。\n"
                "pip install transformers"
            )
