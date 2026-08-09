#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting utilities for training visualization"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def get_logger() -> logging.Logger:
    """获取当前模块的日志记录器"""
    return logging.getLogger(__name__)


# ============================================================
# 配色方案
# ============================================================

'''bash
| Hex     | 名称/说明 |
| ------- | ----- |
| #2E86AB | 蓝色    |
| #E94F37 | 红色    |
| #1B998B | 青色    |
| #FF6B6B | 粉红    |
| #4ECDC4 | 浅青    |
| #9B59B6 | 紫色    |
| #6C5B7B | 灰紫    |
| #F5B041 | 橙黄    |
| #16A085 | 深青    |
| #E67E22 | 深橙    |
| #34495E | 深灰蓝  |
| #F39C12 | 金橙    |
| #AF7AC5 | 淡紫    |
| #5D6D7E | 灰蓝    |
| #7D3C98 | 紫罗兰  |
| #45B39D | 青绿    |
| #F1948A | 淡粉红  |
| #48C9B0 | 浅青绿  |
| #D35400 | 深橙红  |
| #C0392B | 暗红    |
| #1ABC9C | 青绿色  |
| #27AE60 | 深绿色  |
| #E74C3C | 鲜红    |
| #2980B9 | 深蓝    |
| #8E44AD | 深紫    |
| #F1C40F | 明黄色  |
| #D5DBDB | 浅灰    |
| #AAB7B8 | 中灰    |
| #34495E | 深灰    |
| #2C3E50 | 蓝灰    |
| #EC7063 | 珊瑚红  |
| #45A29E | 蓝绿    |

'''

class ColorScheme:
    """绘图配色方案"""
    
    # 基础训练/验证
    # TRAIN = '#2E86AB'           # 蓝色 - 训练
    TRAIN = '#FFD700'
    # VAL = '#E94F37'             # 红色 - 验证
    VAL = '#228B22'
    LR = '#6C5B7B'              # 灰紫 - 学习率


    # 全局指标
    MIOU = '#1B998B'            # 青色 - mIoU
    DICE = '#FF6B6B'            # 粉色 - Dice
    PIXEL_ACCURACY = '#F5B041'  # 橙黄 - Pixel Accuracy
    MPIXEL_ACCURACY = '#7D3C98' # 紫罗兰 - mPixel Accuracy
    F1 = '#4ECDC4'              # 浅青 - F1
    PRECISION = '#2E86AB'       # 蓝色 - Precision
    RECALL = '#E94F37'          # 红色 - Recall

    # 边界指标
    BOUNDARY_IOU = '#9B59B6'        # 紫色 - Boundary IoU
    BOUNDARY_F1 = '#AF7AC5'         # 淡紫 - Boundary F1
    BOUNDARY_PRECISION = '#5D6D7E'  # 灰蓝 - Boundary Precision
    BOUNDARY_RECALL = '#F39C12'     # 金橙 - Boundary Recall

    # 每张图像指标
    IMAGE_IOU = '#16A085'           # 深青 - per image IoU
    IMAGE_DICE = '#E67E22'          # 深橙 - per image Dice

    # mean ± std
    MEAN_STD = '#34495E'            # 深灰蓝 - mean ± std

    @classmethod
    def as_dict(cls) -> dict[str, str]:
        """返回配色字典"""
        return {
            'train': cls.TRAIN,
            'val': cls.VAL,
            'lr': cls.LR,
            'mIoU': cls.MIOU,
            'Dice': cls.DICE,
            'Pixel Accuracy': cls.PIXEL_ACCURACY,
            'mPixel Accuracy': cls.MPIXEL_ACCURACY,
            'F1': cls.F1,
            'Precision': cls.PRECISION,
            'Recall': cls.RECALL,
            'Boundary IoU': cls.BOUNDARY_IOU,
            'Boundary F1': cls.BOUNDARY_F1,
            'Boundary Precision': cls.BOUNDARY_PRECISION,
            'Boundary Recall': cls.BOUNDARY_RECALL,
            'Image IoU': cls.IMAGE_IOU,
            'Image Dice': cls.IMAGE_DICE,
            'MeanStd': cls.MEAN_STD,
        }

class TrainingPlotter:
    """
    训练过程可视化绘图器.
    """
    
    # ImageNet 归一化参数
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    def __init__(
        self, 
        save_dir: Union[str, Path], 
        model_name: str = 'Model', 
        dpi: int = 150, 
        figsize_scale: float = 1.0
    ) -> None:
        self.logger = get_logger()
        
        self.save_dir = Path(save_dir)
        self.plots_dir = self.save_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.dpi = dpi
        self.figsize_scale = figsize_scale
        
        # 历史记录存储（扩展版）
        self.history: dict[str, list[float]] = {
            # 基础
            'epoch': [],
            'lr': [],
            'epoch_time': [],  # 新增：每个 epoch 的时间
            
            # 损失
            'train_loss': [],
            'val_loss': [],
            
            # mIoU
            'train_mIoU': [],
            'val_mIoU': [],
            'val_IoU_std': [],
            
            # Dice
            'train_Dice': [],
            'val_Dice': [],
            'val_Dice_std': [],
            
            # 像素精度
            'train_Pixel_Acc': [],
            'val_Pixel_Acc': [],
            'train_mPixel_Acc': [], 
            'val_mPixel_Acc': [], 
            
            # Precision / Recall / F1（训练和验证）
            'train_Precision': [],
            'train_Recall': [],
            'train_F1': [],
            'val_Precision': [],
            'val_Recall': [],
            'val_F1': [],
            
            # 边界指标
            'val_Boundary_IoU': [],
            'val_Boundary_F1': [],
            'val_Boundary_Precision': [],  # 新增
            'val_Boundary_Recall': [],     # 新增
            
            # 每类指标（二分类）
            'val_IoU_bg': [],       # 新增：背景 IoU
            'val_IoU_fg': [],       # 新增：前景 IoU
            'val_Dice_bg': [],      # 新增：背景 Dice
            'val_Dice_fg': [],      # 新增：前景 Dice
        }
        
        # 配色
        self.colors = ColorScheme.as_dict()
        
        # 设置绘图风格
        self._setup_style()
        
        self.logger.info(f"TrainingPlotter 初始化 | 保存目录: {self.plots_dir}")
    
    def _setup_style(self) -> None:
        """设置 matplotlib 绘图风格"""
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                self.logger.warning("无法设置 seaborn 风格，使用默认风格")
    
    def _scale_figsize(self, width: float, height: float) -> tuple[float, float]:
        """缩放图像尺寸"""
        return (width * self.figsize_scale, height * self.figsize_scale)
    
    def _save_figure(self, name: str, fig: plt.Figure | None = None) -> None:
        """保存图像（PNG + PDF）"""
        if fig is None:
            fig = plt.gcf()
        
        png_path = self.plots_dir / f'{name}.png'
        pdf_path = self.plots_dir / f'{name}.pdf'
        
        fig.savefig(png_path, dpi=self.dpi, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.debug(f"图像已保存: {name}")
    
    def update_history(
        self, 
        epoch: int, 
        lr: float, 
        train_loss: float, 
        train_metrics: dict[str, float], 
        val_loss: float, 
        val_metrics: dict[str, float],
        epoch_time: float | None = None
    ) -> None:
        """
        更新训练历史记录.
        
        Args:
            epoch: 当前 epoch
            lr: 当前学习率
            train_loss: 训练损失
            train_metrics: 训练指标字典
            val_loss: 验证损失
            val_metrics: 验证指标字典
            epoch_time: epoch 耗时（秒）
        """
        self.history['epoch'].append(epoch)
        self.history['lr'].append(lr)
        self.history['epoch_time'].append(epoch_time or 0)
        
        # 损失
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        # mIoU
        self.history['train_mIoU'].append(train_metrics.get('mIoU', 0))
        self.history['val_mIoU'].append(val_metrics.get('mIoU', 0))
        self.history['val_IoU_std'].append(val_metrics.get('IoU_std', 0))
        
        # Dice
        self.history['train_Dice'].append(train_metrics.get('Dice', 0))
        self.history['val_Dice'].append(val_metrics.get('Dice', 0))
        self.history['val_Dice_std'].append(val_metrics.get('Dice_std', 0))
        
        # 像素精度
        self.history['train_Pixel_Acc'].append(train_metrics.get('Pixel_Acc', 0))
        self.history['val_Pixel_Acc'].append(val_metrics.get('Pixel_Acc', 0))
        self.history['train_mPixel_Acc'].append(train_metrics.get('mPixel_Acc', 0))
        self.history['val_mPixel_Acc'].append(val_metrics.get('mPixel_Acc', 0))
        
        # Precision / Recall / F1
        self.history['train_Precision'].append(train_metrics.get('Precision', 0))
        self.history['train_Recall'].append(train_metrics.get('Recall', 0))
        self.history['train_F1'].append(train_metrics.get('F1', 0))
        self.history['val_Precision'].append(val_metrics.get('Precision', 0))
        self.history['val_Recall'].append(val_metrics.get('Recall', 0))
        self.history['val_F1'].append(val_metrics.get('F1', 0))
        
        # 边界指标
        self.history['val_Boundary_IoU'].append(val_metrics.get('Boundary_IoU', 0))
        self.history['val_Boundary_F1'].append(val_metrics.get('Boundary_F1', 0))
        self.history['val_Boundary_Precision'].append(val_metrics.get('Boundary_Precision', 0))
        self.history['val_Boundary_Recall'].append(val_metrics.get('Boundary_Recall', 0))
        
        # 每类指标（从 per_class_IoU 和 per_class_Dice 提取）
        per_class_iou = val_metrics.get('per_class_IoU', [0, 0])
        per_class_dice = val_metrics.get('per_class_Dice', [0, 0])
        
        if len(per_class_iou) >= 2:
            self.history['val_IoU_bg'].append(per_class_iou[0])
            self.history['val_IoU_fg'].append(per_class_iou[1])
        else:
            self.history['val_IoU_bg'].append(0)
            self.history['val_IoU_fg'].append(0)
        
        if len(per_class_dice) >= 2:
            self.history['val_Dice_bg'].append(per_class_dice[0])
            self.history['val_Dice_fg'].append(per_class_dice[1])
        else:
            self.history['val_Dice_bg'].append(0)
            self.history['val_Dice_fg'].append(0)
    
    def plot_loss_curves(self) -> None:
        """绘制训练和验证损失曲线"""
        fig, ax = plt.subplots(figsize=self._scale_figsize(10, 6))
        
        epochs = self.history['epoch']
        ax.plot(epochs, self.history['train_loss'], 
                label='Train Loss', color=self.colors['train'], linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 
                label='Val Loss', color=self.colors['val'], linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{self.model_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标注最小值
        min_idx = int(np.argmin(self.history['val_loss']))
        min_val = self.history['val_loss'][min_idx]
        ax.axvline(x=epochs[min_idx], color='gray', linestyle='--', alpha=0.5)
        ax.annotate(
            f'Min: {min_val:.4f}\nEpoch {epochs[min_idx]}',
            xy=(epochs[min_idx], min_val),
            xytext=(10, 10), textcoords='offset points',
            fontsize=9, color='gray'
        )
        
        plt.tight_layout()
        self._save_figure('loss_curves', fig)
    
    def plot_metric_curves(self) -> None:
        """绘制 mIoU 和 Dice 曲线"""
        fig, axes = plt.subplots(1, 2, figsize=self._scale_figsize(14, 5))
        epochs = self.history['epoch']
        
        # mIoU
        ax1 = axes[0]
        ax1.plot(epochs, self.history['train_mIoU'], 
                label='Train', color=self.colors['train'], linewidth=2)
        ax1.plot(epochs, self.history['val_mIoU'], 
                label='Val', color=self.colors['val'], linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('mIoU', fontsize=12)
        ax1.set_title('Mean IoU', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 标注最佳值
        best_idx = int(np.argmax(self.history['val_mIoU']))
        best_val = self.history['val_mIoU'][best_idx]
        ax1.axhline(y=best_val, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate(
            f'Best: {best_val:.4f}', 
            xy=(epochs[-1], best_val),
            xytext=(-60, 5), textcoords='offset points', 
            fontsize=9, color='gray'
        )
        
        # Dice
        ax1 = axes[1]
        ax1.plot(epochs, self.history['train_Dice'], 
                label='Train', color=self.colors['train'], linewidth=2)
        ax1.plot(epochs, self.history['val_Dice'], 
                label='Val', color=self.colors['val'], linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Dice', fontsize=12)
        ax1.set_title('Dice Coefficient', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        best_idx = int(np.argmax(self.history['val_Dice']))
        best_val = self.history['val_Dice'][best_idx]
        ax1.axhline(y=best_val, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate(
            f'Best: {best_val:.4f}', 
            xy=(epochs[-1], best_val),
            xytext=(-60, 5), textcoords='offset points', 
            fontsize=9, color='gray'
        )
        
        plt.tight_layout()
        self._save_figure('metric_curves', fig)
    
    def plot_precision_recall_f1(self) -> None:
        """绘制 Precision、Recall、F1 曲线（训练和验证）"""
        fig, axes = plt.subplots(1, 2, figsize=self._scale_figsize(14, 5))
        epochs = self.history['epoch']
        
        # 训练集
        ax1 = axes[0]
        ax1.plot(epochs, self.history['train_Precision'], 
                label='Precision', color=ColorScheme.PRECISION, linewidth=2)
        ax1.plot(epochs, self.history['train_Recall'], 
                label='Recall', color=ColorScheme.RECALL, linewidth=2)
        ax1.plot(epochs, self.history['train_F1'], 
                label='F1 Score', color=ColorScheme.F1, linewidth=2, linestyle='--')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Train - Precision / Recall / F1', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 验证集
        ax1 = axes[1]
        ax1.plot(epochs, self.history['val_Precision'], 
                label='Precision', color=ColorScheme.PRECISION, linewidth=2)
        ax1.plot(epochs, self.history['val_Recall'], 
                label='Recall', color=ColorScheme.RECALL, linewidth=2)
        ax1.plot(epochs, self.history['val_F1'], 
                label='F1 Score', color=ColorScheme.F1, linewidth=2, linestyle='--')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Val - Precision / Recall / F1', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        plt.tight_layout()
        self._save_figure('precision_recall_f1', fig)
    
    def plot_boundary_metrics(self) -> None:
        """绘制边界指标曲线（IoU, F1, Precision, Recall）"""
        fig, axes = plt.subplots(1, 2, figsize=self._scale_figsize(14, 5))
        epochs = self.history['epoch']
        
        # IoU & F1
        ax1 = axes[0]
        ax1.plot(epochs, self.history['val_Boundary_IoU'], 
                label='Boundary IoU', color=ColorScheme.BOUNDARY_IOU, linewidth=2)
        ax1.plot(epochs, self.history['val_Boundary_F1'], 
                label='Boundary F1', color=ColorScheme.BOUNDARY_F1, linewidth=2, linestyle='--')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Boundary IoU & F1', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Precision & Recall
        ax1 = axes[1]
        ax1.plot(epochs, self.history['val_Boundary_Precision'], 
                label='Boundary Precision', color=ColorScheme.BOUNDARY_PRECISION, linewidth=2)
        ax1.plot(epochs, self.history['val_Boundary_Recall'], 
                label='Boundary Recall', color=ColorScheme.BOUNDARY_RECALL, linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Boundary Precision & Recall', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        plt.tight_layout()
        self._save_figure('boundary_metrics', fig)
    
    def plot_per_class_metrics(self) -> None:
        """绘制每类 IoU 和 Dice 曲线"""
        fig, axes = plt.subplots(1, 2, figsize=self._scale_figsize(14, 5))
        epochs = self.history['epoch']
        
        # Per-class IoU
        ax1 = axes[0]
        ax1.plot(epochs, self.history['val_IoU_bg'], 
                label='Background IoU', color='#7FB3D5', linewidth=2)
        ax1.plot(epochs, self.history['val_IoU_fg'], 
                label='Foreground IoU', color='#27AE60', linewidth=2)
        ax1.plot(epochs, self.history['val_mIoU'], 
                label='Mean IoU', color=self.colors['val'], linewidth=2, linestyle='--')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('IoU', fontsize=12)
        ax1.set_title('Per-Class IoU', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Per-class Dice
        ax1 = axes[1]
        ax1.plot(epochs, self.history['val_Dice_bg'], 
                label='Background Dice', color='#7FB3D5', linewidth=2)
        ax1.plot(epochs, self.history['val_Dice_fg'], 
                label='Foreground Dice', color='#27AE60', linewidth=2)
        ax1.plot(epochs, self.history['val_Dice'], 
                label='Mean Dice', color=self.colors['val'], linewidth=2, linestyle='--')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Dice', fontsize=12)
        ax1.set_title('Per-Class Dice', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        plt.tight_layout()
        self._save_figure('per_class_metrics', fig)
    
    def plot_pixel_accuracy(self) -> None:
        """绘制像素精度和 mPA 曲线"""
        fig, axes = plt.subplots(1, 2, figsize=self._scale_figsize(14, 5))
        epochs = self.history['epoch']
        
        # Pixel Accuracy
        ax1 = axes[0]
        ax1.plot(epochs, self.history['train_Pixel_Acc'], 
                label='Train', color=self.colors['train'], linewidth=2)
        ax1.plot(epochs, self.history['val_Pixel_Acc'], 
                label='Val', color=self.colors['val'], linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Pixel Accuracy', fontsize=12)
        ax1.set_title('Pixel Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 标注最佳值
        best_idx = int(np.argmax(self.history['val_Pixel_Acc']))
        best_val = self.history['val_Pixel_Acc'][best_idx]
        ax1.axhline(y=best_val, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate(
            f'Best: {best_val:.4f}', 
            xy=(epochs[-1], best_val),
            xytext=(-60, 5), textcoords='offset points', 
            fontsize=9, color='gray'
        )
        
        # Mean Pixel Accuracy (mPA)
        ax1 = axes[1]
        ax1.plot(epochs, self.history['train_mPixel_Acc'], 
                label='Train', color=self.colors['train'], linewidth=2)
        ax1.plot(epochs, self.history['val_mPixel_Acc'], 
                label='Val', color=self.colors['val'], linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('mPixel Accuracy', fontsize=12)
        ax1.set_title('Mean Pixel Accuracy (mPA)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        best_idx = int(np.argmax(self.history['val_mPixel_Acc']))
        best_val = self.history['val_mPixel_Acc'][best_idx]
        ax1.axhline(y=best_val, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate(
            f'Best: {best_val:.4f}', 
            xy=(epochs[-1], best_val),
            xytext=(-60, 5), textcoords='offset points', 
            fontsize=9, color='gray'
        )
        
        plt.tight_layout()
        self._save_figure('pixel_accuracy', fig)
    
    def plot_lr_curve(self) -> None:
        """绘制学习率曲线"""
        fig, ax = plt.subplots(figsize=self._scale_figsize(10, 4))
        epochs = self.history['epoch']
        
        ax.plot(epochs, self.history['lr'], color=self.colors['lr'], linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure('lr_curve', fig)
    
    def plot_epoch_time(self) -> None:
        """绘制每个 epoch 的训练时间"""
        if not any(self.history['epoch_time']):
            return
        
        fig, ax = plt.subplots(figsize=self._scale_figsize(10, 4))
        epochs = self.history['epoch']
        times = self.history['epoch_time']
        
        ax.bar(epochs, times, color='#5DADE2', edgecolor='black', alpha=0.7)
        ax.axhline(y=np.mean(times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(times):.1f}s')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure('epoch_time', fig)
    
    def plot_metrics_with_std(self) -> None:
        """绘制带标准差带的 mIoU 和 Dice 曲线"""
        fig, axes = plt.subplots(1, 2, figsize=self._scale_figsize(14, 5))
        epochs = np.array(self.history['epoch'])
        
        # mIoU with std
        ax1 = axes[0]
        val_miou = np.array(self.history['val_mIoU'])
        val_iou_std = np.array(self.history['val_IoU_std'])
        
        ax1.plot(epochs, val_miou, label='Val mIoU', 
                color=self.colors['val'], linewidth=2)
        ax1.fill_between(
            epochs, 
            np.clip(val_miou - val_iou_std, 0, 1), 
            np.clip(val_miou + val_iou_std, 0, 1),
            color=self.colors['val'], alpha=0.2, label='±1 Std'
        )
        ax1.plot(epochs, self.history['train_mIoU'], label='Train mIoU', 
                color=self.colors['train'], linewidth=2, linestyle='--')
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('mIoU', fontsize=12)
        ax1.set_title('Mean IoU with Standard Deviation', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Dice with std
        ax1 = axes[1]
        val_dice = np.array(self.history['val_Dice'])
        val_dice_std = np.array(self.history['val_Dice_std'])
        
        ax1.plot(epochs, val_dice, label='Val Dice', 
                color=self.colors['val'], linewidth=2)
        ax1.fill_between(
            epochs, 
            np.clip(val_dice - val_dice_std, 0, 1), 
            np.clip(val_dice + val_dice_std, 0, 1),
            color=self.colors['val'], alpha=0.2, label='±1 Std'
        )
        ax1.plot(epochs, self.history['train_Dice'], label='Train Dice', 
                color=self.colors['train'], linewidth=2, linestyle='--')
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Dice', fontsize=12)
        ax1.set_title('Dice with Standard Deviation', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        plt.tight_layout()
        self._save_figure('metrics_with_std', fig)
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray, 
        epoch: int, 
        class_names: list[str] | None = None
    ) -> None:
        """绘制混淆矩阵热力图"""
        if class_names is None:
            class_names = ['Background', 'Foreground']
        
        fig, ax = plt.subplots(figsize=self._scale_figsize(8, 6))
        
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=class_names, 
            yticklabels=class_names,
            ax=ax, 
            annot_kws={'size': 14}
        )
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(
            f'{self.model_name} - Confusion Matrix (Epoch {epoch})', 
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        self._save_figure('confusion_matrix', fig)
    
    def plot_per_image_boxplot(
        self, 
        per_image_metrics: dict[str, np.ndarray], 
        epoch: int
    ) -> None:
        """绘制每张图像指标的箱线图"""
        fig, axes = plt.subplots(1, 2, figsize=self._scale_figsize(12, 5))
        
        for ax, (metric_name, data) in zip(axes, per_image_metrics.items()):
            color = self.colors.get(metric_name, self.colors['mIoU'])
            
            if len(data) > 0:
                bp = ax.boxplot([data], labels=[metric_name], patch_artist=True)
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_alpha(0.7)
                
                # 散点（添加抖动）
                jitter = np.random.normal(0, 0.04, len(data))
                ax.scatter(np.ones(len(data)) + jitter, data, 
                          alpha=0.5, s=20, color='gray')
                
                # 统计信息
                stats_text = (
                    f'Mean: {np.mean(data):.4f}\n'
                    f'Std: {np.std(data):.4f}\n'
                    f'Median: {np.median(data):.4f}\n'
                    f'Min: {np.min(data):.4f}\n'
                    f'Max: {np.max(data):.4f}'
                )
                ax.text(
                    0.02, 0.98, stats_text, 
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )
            
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'Per-Image {metric_name} Distribution', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'{self.model_name} - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure('per_image_boxplot', fig)
    
    def plot_predictions(
        self, 
        images: torch.Tensor, 
        masks: torch.Tensor, 
        preds: torch.Tensor, 
        epoch: int,
        max_samples: int = 4
    ) -> None:
        """绘制预测对比图"""
        n_samples = min(len(images), max_samples)
        fig, axes = plt.subplots(n_samples, 4, figsize=self._scale_figsize(16, 4 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # 反归一化图像
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = self.IMAGENET_STD * img + self.IMAGENET_MEAN
            img = np.clip(img, 0, 1)
            
            mask = masks[i].cpu().numpy()
            pred = preds[i].cpu().numpy()
            
            # 输入图像
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image', fontsize=11)
            axes[i, 0].axis('off')
            
            # 真实标签
            axes[i, 1].imshow(mask, cmap='Greens', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth', fontsize=11)
            axes[i, 1].axis('off')
            
            # 预测结果
            axes[i, 2].imshow(pred, cmap='Greens', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction', fontsize=11)
            axes[i, 2].axis('off')
            
            # 错误图（TP=绿, FP=红, FN=蓝）
            overlay = np.zeros((*mask.shape, 3))
            tp = (mask == 1) & (pred == 1)
            fp = (mask == 0) & (pred == 1)
            fn = (mask == 1) & (pred == 0)
            overlay[tp] = [0, 1, 0]  # 绿色
            overlay[fp] = [1, 0, 0]  # 红色
            overlay[fn] = [0, 0, 1]  # 蓝色
            
            blended = 0.6 * img + 0.4 * overlay
            blended = np.clip(blended, 0, 1)
            
            axes[i, 3].imshow(blended)
            axes[i, 3].set_title('Error Map (G:TP, R:FP, B:FN)', fontsize=11)
            axes[i, 3].axis('off')
        
        plt.suptitle(f'{self.model_name} - Predictions (Epoch {epoch})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(f'predictions_epoch_{epoch:03d}', fig)
    
    def plot_training_paper(self) -> None:
        """绘制训练综合概览（2x2 网格）"""
        fig = plt.figure(figsize=self._scale_figsize(10, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        epochs = self.history['epoch']

        # Row 1: mIoU, Dice
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.history['train_mIoU'], 
                label='Train', color=self.colors['train'], linewidth=2)
        ax1.plot(epochs, self.history['val_mIoU'], 
                label='Val', color=self.colors['val'], linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mIoU')
        ax1.set_title('Mean IoU', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.history['train_Dice'], 
                label='Train', color=self.colors['train'], linewidth=2)
        ax2.plot(epochs, self.history['val_Dice'], 
                label='Val', color=self.colors['val'], linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice')
        ax2.set_title('Dice Coefficient', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Row 2: P/R/F1, Boundary, Pixel Acc
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.history['val_Precision'], label='Precision', linewidth=2)
        ax3.plot(epochs, self.history['val_Recall'], label='Recall', linewidth=2)
        ax3.plot(epochs, self.history['val_F1'], label='F1', linewidth=2, linestyle='--')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_title('Precision / Recall / F1', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, self.history['val_Pixel_Acc'], 
                label='PA', color=ColorScheme.PIXEL_ACCURACY, linewidth=2)
        ax4.plot(epochs, self.history['val_mPixel_Acc'], 
                label='mPA', color=ColorScheme.MPIXEL_ACCURACY, linewidth=2, linestyle='--')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Pixel Accuracy & mPA', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
       
        plt.suptitle(f'{self.model_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        self._save_figure('training_paper', fig)

    def plot_training_overview(self) -> None:
        """绘制训练综合概览（4x3 网格）"""
        fig = plt.figure(figsize=self._scale_figsize(18, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        epochs = self.history['epoch']
        
        # Row 1: Loss, mIoU, Dice
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.history['train_loss'], 
                label='Train', color=self.colors['train'], linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 
                label='Val', color=self.colors['val'], linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(epochs, self.history['train_mIoU'], 
                label='Train', color=self.colors['train'], linewidth=2)
        ax1.plot(epochs, self.history['val_mIoU'], 
                label='Val', color=self.colors['val'], linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mIoU')
        ax1.set_title('Mean IoU', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(epochs, self.history['train_Dice'], 
                label='Train', color=self.colors['train'], linewidth=2)
        ax2.plot(epochs, self.history['val_Dice'], 
                label='Val', color=self.colors['val'], linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice')
        ax2.set_title('Dice Coefficient', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Row 2: P/R/F1, Boundary, Pixel Acc
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.history['val_Precision'], label='Precision', linewidth=2)
        ax3.plot(epochs, self.history['val_Recall'], label='Recall', linewidth=2)
        ax3.plot(epochs, self.history['val_F1'], label='F1', linewidth=2, linestyle='--')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_title('Precision / Recall / F1', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(epochs, self.history['val_Boundary_IoU'], 
                label='Boundary IoU', color=self.colors['Boundary IoU'], linewidth=2)
        ax5.plot(epochs, self.history['val_Boundary_F1'], 
                label='Boundary F1', color=self.colors['Boundary F1'], linewidth=2, linestyle='--')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Score')
        ax5.set_title('Boundary Metrics', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])
        
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(epochs, self.history['val_Pixel_Acc'], 
                label='PA', color=ColorScheme.PIXEL_ACCURACY, linewidth=2)
        ax4.plot(epochs, self.history['val_mPixel_Acc'], 
                label='mPA', color=ColorScheme.MPIXEL_ACCURACY, linewidth=2, linestyle='--')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Pixel Accuracy & mPA', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        # Row 3: Per-class IoU, Per-class Dice, LR
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(epochs, self.history['val_IoU_bg'], 
                label='Background', color='#7FB3D5', linewidth=2)
        ax7.plot(epochs, self.history['val_IoU_fg'], 
                label='Foreground', color='#27AE60', linewidth=2)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('IoU')
        ax7.set_title('Per-Class IoU', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, 1])
        
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(epochs, self.history['val_Dice_bg'], 
                label='Background', color='#7FB3D5', linewidth=2)
        ax8.plot(epochs, self.history['val_Dice_fg'], 
                label='Foreground', color='#27AE60', linewidth=2)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Dice')
        ax8.set_title('Per-Class Dice', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0, 1])
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(epochs, self.history['lr'], color='#355C7D', linewidth=2)
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Learning Rate')
        ax9.set_title('Learning Rate', fontweight='bold')
        ax9.set_yscale('log')
        ax9.grid(True, alpha=0.3)
        
        # Row 4: mIoU±Std, Epoch Time, Summary
        ax10 = fig.add_subplot(gs[3, 0])
        val_miou = np.array(self.history['val_mIoU'])
        val_iou_std = np.array(self.history['val_IoU_std'])
        epochs_arr = np.array(epochs)
        ax10.plot(epochs_arr, val_miou, color=self.colors['val'], linewidth=2)
        ax10.fill_between(
            epochs_arr, 
            np.clip(val_miou - val_iou_std, 0, 1), 
            np.clip(val_miou + val_iou_std, 0, 1),
            color=self.colors['val'], alpha=0.2
        )
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('mIoU')
        ax10.set_title('mIoU ± Std', fontweight='bold')
        ax10.grid(True, alpha=0.3)
        ax10.set_ylim([0, 1])
        
        ax11 = fig.add_subplot(gs[3, 1])
        if any(self.history['epoch_time']):
            times = self.history['epoch_time']
            ax11.bar(epochs, times, color='#5DADE2', alpha=0.7)
            ax11.axhline(y=np.mean(times), color='red', linestyle='--')
            ax11.set_xlabel('Epoch')
            ax11.set_ylabel('Time (s)')
            ax11.set_title(f'Epoch Time (Mean: {np.mean(times):.1f}s)', fontweight='bold')
        else:
            ax11.text(0.5, 0.5, 'No timing data', ha='center', va='center', fontsize=12)
            ax11.set_title('Epoch Time', fontweight='bold')
        ax11.grid(True, alpha=0.3)
        
        # Summary 文本框
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.axis('off')
        
        best_miou = max(self.history['val_mIoU'])
        best_dice = max(self.history['val_Dice'])
        best_biou = max(self.history['val_Boundary_IoU'])
        best_f1 = max(self.history['val_F1'])
        best_pa = max(self.history['val_Pixel_Acc'])
        best_mpa = max(self.history['val_mPixel_Acc'])
        best_epoch = self.history['epoch'][int(np.argmax(self.history['val_mIoU']))]
        total_time = sum(self.history['epoch_time']) if any(self.history['epoch_time']) else 0
        
        summary_text = (
            f"Best Results (Epoch {best_epoch})\n"
            f"{'=' * 30}\n\n"
            f"mIoU:           {best_miou:.4f}\n"
            f"Dice:           {best_dice:.4f}\n"
            f"F1 Score:       {best_f1:.4f}\n"
            f"Pixel Acc:      {best_pa:.4f}\n"
            f"mPixel Acc:     {best_mpa:.4f}\n" 
            f"Boundary IoU:   {best_biou:.4f}\n\n"
            f"{'=' * 30}\n"
            f"Total Epochs:   {len(epochs)}\n"
            f"Total Time:     {total_time/60:.1f} min\n"
        )
        ax12.text(
            0.1, 0.9, summary_text, 
            transform=ax12.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        )
        
        plt.suptitle(f'{self.model_name} - Training Overview', 
                    fontsize=16, fontweight='bold', y=0.98)
        self._save_figure('training_overview', fig)
    
    def plot_all(self) -> None:
        """生成所有图表"""
        if len(self.history['epoch']) < 1:
            self.logger.warning("历史记录为空，跳过绘图")
            return
        
        self.plot_loss_curves()
        self.plot_metric_curves()
        self.plot_precision_recall_f1()
        self.plot_pixel_accuracy()
        self.plot_boundary_metrics()
        self.plot_per_class_metrics()
        self.plot_lr_curve()
        self.plot_epoch_time()
        self.plot_metrics_with_std()
        self.plot_training_overview()
        self.plot_training_paper()
        
        self.logger.info(f"所有图表已生成: {self.plots_dir}")
    
    def get_best_metrics(self) -> dict[str, Any]:
        """获取最佳指标"""
        if len(self.history['epoch']) < 1:
            return {}
        
        best_miou_idx = int(np.argmax(self.history['val_mIoU']))
        
        return {
            'best_epoch': self.history['epoch'][best_miou_idx],
            'best_mIoU': max(self.history['val_mIoU']),
            'best_Dice': max(self.history['val_Dice']),
            'best_F1': max(self.history['val_F1']),
            'best_Pixel_Acc': max(self.history['val_Pixel_Acc']),
            'best_mPixel_Acc': max(self.history['val_mPixel_Acc']),
            'best_Boundary_IoU': max(self.history['val_Boundary_IoU']),
            'best_IoU_fg': max(self.history['val_IoU_fg']),
            'final_mIoU': self.history['val_mIoU'][-1],
            'final_Dice': self.history['val_Dice'][-1],
            'total_time': sum(self.history['epoch_time']),
        }
    
    def __repr__(self) -> str:
        return (
            f"TrainingPlotter("
            f"save_dir='{self.save_dir}', "
            f"model_name='{self.model_name}', "
            f"epochs={len(self.history['epoch'])})"
        )
    
    def __len__(self) -> int:
        return len(self.history['epoch'])



# ============================================================
# 模型对比绘图
# ============================================================

def plot_model_comparison(
    results_dict: dict[str, dict[str, float]], 
    save_path: str | Path, 
    metric: str = 'mIoU',
    figsize: tuple[float, float] = (12, 6)
) -> None:
    """
    绘制多模型对比柱状图.
    
    Args:
        results_dict: {model_name: metrics_dict}
        save_path: 保存路径
        metric: 对比的指标
        figsize: 图像尺寸
    """
    logger = get_logger()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(results_dict.keys())
    values = [results_dict[m].get(metric, 0) for m in models]
    stds = [results_dict[m].get(f'{metric}_std', 0) for m in models]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, values, yerr=stds, capsize=5, color=colors, edgecolor='black')
    
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() + 0.02,
            f'{val:.4f}', 
            ha='center', va='bottom', fontsize=10
        )
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"模型对比图已保存: {save_path}")