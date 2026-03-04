#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNet: Deep High-Resolution Representation Learning for Visual Recognition

Reference:
    Wang et al. "Deep High-Resolution Representation Learning for Visual Recognition"
    https://arxiv.org/abs/1908.07919
"""

from __future__ import annotations

import logging
from typing import Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """基础残差块"""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """瓶颈残差块"""
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class HighResolutionModule(nn.Module):
    """高分辨率模块：包含多分辨率并行分支和跨分辨率融合"""
    
    def __init__(
        self, 
        num_branches: int,
        num_blocks: List[int],
        num_channels: List[int],
        multi_scale_output: bool = True
    ) -> None:
        super().__init__()
        
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        
        # 各分支
        self.branches = self._make_branches(num_branches, num_blocks, num_channels)
        
        # 融合层
        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def _make_branches(
        self, 
        num_branches: int, 
        num_blocks: List[int], 
        num_channels: List[int]
    ) -> nn.ModuleList:
        branches = nn.ModuleList()
        
        for i in range(num_branches):
            layers = []
            for j in range(num_blocks[i]):
                layers.append(BasicBlock(num_channels[i], num_channels[i]))
            branches.append(nn.Sequential(*layers))
        
        return branches
    
    def _make_fuse_layers(self, num_branches: int, num_channels: List[int]) -> nn.ModuleList:
        if num_branches == 1:
            return nn.ModuleList()
        
        fuse_layers = nn.ModuleList()
        
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    # 上采样
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # 下采样
                    conv_list = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_list.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[i]),
                            ))
                        else:
                            conv_list.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True),
                            ))
                    fuse_layer.append(nn.Sequential(*conv_list))
            fuse_layers.append(fuse_layer)
        
        return fuse_layers
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # 各分支前向传播
        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])
        
        # 多尺度融合
        if len(self.fuse_layers) == 0:
            return x
        
        x_fuse = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = None
            for j, fuse in enumerate(fuse_layer):
                if fuse is None:
                    if y is None:
                        y = x[j]
                    else:
                        y = y + x[j]
                else:
                    out = fuse(x[j])
                    if j > i:
                        out = F.interpolate(out, size=x[i].shape[2:], mode='bilinear', align_corners=True)
                    if y is None:
                        y = out
                    else:
                        y = y + out
            x_fuse.append(self.relu(y))
        
        return x_fuse


@register_model('hrnet')
class HRNet(nn.Module):
    """
    HRNet: 高分辨率网络.
    """
    
    SUPPORTED_BACKBONES = ('w18', 'w32', 'w48')
    
    # 宽度配置
    WIDTH_CONFIG = {
        'w18': [18, 36, 72, 144],
        'w32': [32, 64, 128, 256],
        'w48': [48, 96, 192, 384],
    }
    
    def __init__(
        self, 
        num_classes: int = 2,
        backbone: str = 'w32',
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.logger = get_logger()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"不支持的配置: {backbone}，请选择: {self.SUPPORTED_BACKBONES}")
        
        channels = self.WIDTH_CONFIG[backbone]
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        
        # 过渡层 1
        self.transition1 = self._make_transition_layer([256], channels[:2])
        
        # Stage 2
        self.stage2 = HighResolutionModule(2, [4, 4], channels[:2])
        
        # 过渡层 2
        self.transition2 = self._make_transition_layer(channels[:2], channels[:3])
        
        # Stage 3
        self.stage3 = nn.Sequential(
            HighResolutionModule(3, [4, 4, 4], channels[:3]),
            HighResolutionModule(3, [4, 4, 4], channels[:3]),
            HighResolutionModule(3, [4, 4, 4], channels[:3]),
            HighResolutionModule(3, [4, 4, 4], channels[:3]),
        )
        
        # 过渡层 3
        self.transition3 = self._make_transition_layer(channels[:3], channels)
        
        # Stage 4
        self.stage4 = nn.Sequential(
            HighResolutionModule(4, [4, 4, 4, 4], channels),
            HighResolutionModule(4, [4, 4, 4, 4], channels),
            HighResolutionModule(4, [4, 4, 4, 4], channels, multi_scale_output=False),
        )
        
        # 分类头
        total_channels = channels[0]
        self.head = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, 1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels, num_classes, 1),
        )
        
        # 初始化权重
        self._init_weights()
        
        # 加载预训练权重
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)
        elif pretrained:
            self.logger.warning(
                "HRNet 无内置在线预训练权重，将从头训练。"
                "如需预训练，请通过 --pretrained_weights 指定本地权重路径"
            )
        
        self.logger.info(
            f"HRNet 初始化完成 | 配置: {backbone} | 类别数: {num_classes} | "
            f"预训练: {'本地权重' if pretrained_weights else '从头训练'}"
        )
    
    def _make_layer(self, block, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        layers = [block(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels * block.expansion, out_channels))
        return nn.Sequential(*layers)
    
    def _make_transition_layer(
        self, 
        in_channels_list: List[int], 
        out_channels_list: List[int]
    ) -> nn.ModuleList:
        num_branches_in = len(in_channels_list)
        num_branches_out = len(out_channels_list)
        
        transition_layers = nn.ModuleList()
        
        for i in range(num_branches_out):
            if i < num_branches_in:
                if in_channels_list[i] != out_channels_list[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(in_channels_list[i], out_channels_list[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(out_channels_list[i]),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    transition_layers.append(None)
            else:
                # 新分支：下采样
                conv_list = []
                for j in range(i - num_branches_in + 1):
                    in_ch = in_channels_list[-1] if j == 0 else out_channels_list[i]
                    conv_list.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_channels_list[i], 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_channels_list[i]),
                        nn.ReLU(inplace=True),
                    ))
                transition_layers.append(nn.Sequential(*conv_list))
        
        return transition_layers
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained(self, weights_path: str) -> None:
        import os
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        self.logger.info(f"加载本地权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # 过渡到 Stage 2
        x_list = []
        for i, trans in enumerate(self.transition1):
            if trans is not None:
                x_list.append(trans(x))
            else:
                x_list.append(x)
        
        # Stage 2
        x_list = self.stage2(x_list)
        
        # 过渡到 Stage 3
        y_list = []
        for i, trans in enumerate(self.transition2):
            if trans is not None:
                if i < len(x_list):
                    y_list.append(trans(x_list[i]))
                else:
                    y_list.append(trans(x_list[-1]))
            else:
                y_list.append(x_list[i])
        
        # Stage 3
        x_list = self.stage3(y_list)
        
        # 过渡到 Stage 4
        y_list = []
        for i, trans in enumerate(self.transition3):
            if trans is not None:
                if i < len(x_list):
                    y_list.append(trans(x_list[i]))
                else:
                    y_list.append(trans(x_list[-1]))
            else:
                y_list.append(x_list[i])
        
        # Stage 4
        x_list = self.stage4(y_list)
        
        # 分类头（只使用最高分辨率分支）
        out = self.head(x_list[0])
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        return {'out': out}
    
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        return {
            'backbone': 'w32',
            'pretrained': True,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_classes={self.num_classes}, backbone='{self.backbone_name}')"
