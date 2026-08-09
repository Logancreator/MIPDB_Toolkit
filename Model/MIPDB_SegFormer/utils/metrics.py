#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Segmentation metrics calculation"""

from __future__ import annotations

import logging
from typing import Any, Optional, List, Union

import numpy as np
import torch
from scipy import ndimage


def get_logger() -> logging.Logger:
    """获取当前模块的日志记录器"""
    return logging.getLogger(__name__)


class MetricsCalculator:
    """
    计算全面的语义分割指标，包括边界指标.
    
    支持的指标:
        - mIoU: 平均交并比
        - Dice: Dice 系数
        - Precision / Recall / F1
        - Pixel Accuracy: 像素准确率
        - Boundary IoU / F1: 边界区域指标
        - Per-image statistics: 每张图像的统计（均值±标准差）
    
    Args:
        num_classes: 分割类别数
        boundary_width: 边界区域宽度（像素）
        class_names: 类别名称列表
        ignore_index: 忽略的类别索引（如 255）
    
    Example:
        >>> calc = MetricsCalculator(num_classes=2, class_names=['background', 'corn'])
        >>> for pred, target in dataloader:
        ...     calc.update(pred, target)
        >>> metrics = calc.get_metrics()
        >>> calc.summary()
    """
    
    # 防止除零的小常数
    EPS: float = 1e-10
    
    def __init__(self, num_classes: int = 2, boundary_width: int = 3, class_names: Optional[List[str]] = None, ignore_index: Optional[int] = None) -> None:
        self.logger = get_logger()
        # self.logger = logger
        self.num_classes = num_classes
        self.boundary_width = boundary_width
        self.ignore_index = ignore_index
        
        # 设置类别名称
        if class_names is not None:
            if len(class_names) != num_classes:
                raise ValueError(
                    f"class_names 长度 ({len(class_names)}) 必须等于 num_classes ({num_classes})"
                )
            self.class_names = class_names
        else:
            self.class_names = [f'class_{i}' for i in range(num_classes)]
        
        # 初始化累加器
        self.reset()
        
        self.logger.debug(
            f"MetricsCalculator 初始化 | 类别数: {num_classes} | "
            f"边界宽度: {boundary_width} | 类别名: {self.class_names}"
        )
    
    def reset(self) -> None:
        """重置所有累加器"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.boundary_tp = 0
        self.boundary_fp = 0
        self.boundary_fn = 0
        self.per_image_iou: List[float] = []
        self.per_image_dice: List[float] = []
        self._num_samples = 0
    
    def _to_numpy(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """将输入转换为 numpy 数组"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    def _get_boundary_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        从二值 mask 中提取边界区域.
        
        边界定义为膨胀区域减去腐蚀区域。
        
        Args:
            mask: 二值 mask [H, W]
        
        Returns:
            边界 mask [H, W]
        """
        dilated = ndimage.binary_dilation(mask, iterations=self.boundary_width)
        eroded = ndimage.binary_erosion(mask, iterations=self.boundary_width)
        boundary = dilated.astype(np.float32) - eroded.astype(np.float32)
        return boundary > 0
    
    def _update_confusion_matrix(self, pred: np.ndarray, target: np.ndarray) -> None:
        """更新混淆矩阵"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # 创建有效像素 mask（同时过滤 pred 和 target 的越界值）
        valid_mask = (
            (target_flat >= 0) & (target_flat < self.num_classes) &
            (pred_flat >= 0) & (pred_flat < self.num_classes)
        )
        if self.ignore_index is not None:
            valid_mask &= (target_flat != self.ignore_index)
        
        # 计算混淆矩阵
        hist = np.bincount(
            self.num_classes * target_flat[valid_mask].astype(np.int64) + pred_flat[valid_mask].astype(np.int64),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist
    
    def _update_per_image_metrics(self, pred: np.ndarray, target: np.ndarray) -> None:
        """更新每张图像的指标"""
        # 前景类（类别 1）的 IoU 和 Dice
        pred_fg = (pred == 1)
        target_fg = (target == 1)
        
        intersection = np.logical_and(pred_fg, target_fg).sum()
        union = np.logical_or(pred_fg, target_fg).sum()
        pred_sum = pred_fg.sum()
        target_sum = target_fg.sum()
        
        if union > 0:
            self.per_image_iou.append(float(intersection / union))
        else:
            # GT 和 pred 都没有前景 → 完美匹配
            self.per_image_iou.append(1.0)
        
        if (pred_sum + target_sum) > 0:
            self.per_image_dice.append(float(2 * intersection / (pred_sum + target_sum)))
        else:
            self.per_image_dice.append(1.0)
    
    def _update_boundary_metrics(self, pred: np.ndarray, target: np.ndarray) -> None:
        """更新边界指标"""
        pred_boundary = self._get_boundary_mask(pred == 1)
        target_boundary = self._get_boundary_mask(target == 1)
        
        self.boundary_tp += np.logical_and(pred_boundary, target_boundary).sum()
        self.boundary_fp += np.logical_and(pred_boundary, ~target_boundary).sum()
        self.boundary_fn += np.logical_and(~pred_boundary, target_boundary).sum()
    
    def update(self, pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]) -> None:
        """
        使用批量预测更新指标.
        
        Args:
            pred: 预测标签 [N, H, W] 或 [H, W]
            target: 真实标签 [N, H, W] 或 [H, W]
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)
        
        # 确保是 3D 数组 [N, H, W]
        if pred_np.ndim == 2:
            pred_np = pred_np[np.newaxis, ...]
            target_np = target_np[np.newaxis, ...]
        
        batch_size = pred_np.shape[0]
        self._num_samples += batch_size
        
        # 更新混淆矩阵（整个 batch）
        self._update_confusion_matrix(pred_np, target_np)
        
        # 更新每张图像的指标
        for i in range(batch_size):
            self._update_per_image_metrics(pred_np[i], target_np[i])
            self._update_boundary_metrics(pred_np[i], target_np[i])
    
    def _compute_binary_metrics(self, cm: np.ndarray) -> dict[str, float]:
        """计算二分类指标"""
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]
        
        precision = tp / (tp + fp + self.EPS)
        recall = tp / (tp + fn + self.EPS)
        f1 = 2 * precision * recall / (precision + recall + self.EPS)
        dice = 2 * tp / (2 * tp + fp + fn + self.EPS)
        pixel_acc = (tp + tn) / (tp + tn + fp + fn + self.EPS)
        
        # mPA
        pa0 = tn / (tn + fp + self.EPS)
        pa1 = tp / (tp + fn + self.EPS)
        mpa = (pa0 + pa1) / 2

        return {
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'Dice': float(dice),
            'Pixel_Acc': float(pixel_acc),
            'mPixel_Acc': float(mpa)
        }
    
    def _compute_multiclass_metrics(self, cm: np.ndarray) -> dict[str, float]:
        """计算多分类指标"""
        precision = np.diag(cm) / (cm.sum(axis=0) + self.EPS)
        recall = np.diag(cm) / (cm.sum(axis=1) + self.EPS)
        f1 = 2 * precision * recall / (precision + recall + self.EPS)
        pixel_acc = np.diag(cm).sum() / (cm.sum() + self.EPS)

        # mPA
        pa_per_class = np.diag(cm) / (cm.sum(axis=1) + self.EPS)
        mpa = np.mean(pa_per_class)
        
        return {
            'Precision': float(np.mean(precision)),
            'Recall': float(np.mean(recall)),
            'F1': float(np.mean(f1)),
            'Dice': float(np.mean(f1)),  # 多分类时 Dice 等于 F1
            'Pixel_Acc': float(pixel_acc),
            'mPixel_Acc': float(mpa),
        }
    
    def get_metrics(self) -> dict[str, float]:
        """
        计算所有指标.
        
        Returns:
            包含所有指标的字典
        """
        cm = self.confusion_matrix
        
        # IoU per class
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
        iou = intersection / (union + self.EPS)
        miou = float(np.nanmean(iou))
        
        # Dice per class (从 IoU 转换: Dice = 2*IoU/(1+IoU), 分母 >= 1 无需 EPS)
        dice_per_class = 2 * iou / (1 + iou)
        
        # 根据类别数选择计算方式
        if self.num_classes == 2:
            basic_metrics = self._compute_binary_metrics(cm)
        else:
            basic_metrics = self._compute_multiclass_metrics(cm)
        
        # 边界指标
        boundary_precision = self.boundary_tp / (self.boundary_tp + self.boundary_fp + self.EPS)
        boundary_recall = self.boundary_tp / (self.boundary_tp + self.boundary_fn + self.EPS)
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + self.EPS)
        boundary_iou = self.boundary_tp / (self.boundary_tp + self.boundary_fp + self.boundary_fn + self.EPS)
        
        # 每张图像的统计
        iou_mean = float(np.mean(self.per_image_iou)) if self.per_image_iou else 0.0
        iou_std = float(np.std(self.per_image_iou)) if self.per_image_iou else 0.0
        dice_mean = float(np.mean(self.per_image_dice)) if self.per_image_dice else 0.0
        dice_std = float(np.std(self.per_image_dice)) if self.per_image_dice else 0.0
        
        # 汇总所有指标
        metrics: dict[str, Any] = {
            'mIoU': miou,
            **basic_metrics,
            
            # 边界指标
            'Boundary_IoU': float(boundary_iou),
            'Boundary_F1': float(boundary_f1),
            'Boundary_Precision': float(boundary_precision),
            'Boundary_Recall': float(boundary_recall),
            
            # 统计指标
            'IoU_mean': iou_mean,
            'IoU_std': iou_std,
            'Dice_mean': dice_mean,
            'Dice_std': dice_std,
            
            # 每类指标（列表形式，供 plotting 使用）
            'per_class_IoU': iou.tolist(),
            'per_class_Dice': dice_per_class.tolist(),
        }
        
        # 添加每类 IoU（命名形式，便于查看）
        for i, name in enumerate(self.class_names):
            metrics[f'IoU_{name}'] = float(iou[i])
            metrics[f'Dice_{name}'] = float(dice_per_class[i])
        
        return metrics
    
    def get_confusion_matrix(self, normalized: bool = True) -> np.ndarray:
        """
        获取混淆矩阵.
        
        Args:
            normalized: 是否行归一化
        
        Returns:
            混淆矩阵 [num_classes, num_classes]
        """
        cm = self.confusion_matrix.astype(np.float64)
        if normalized:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = cm / (row_sums + self.EPS)
        return cm
    
    def get_per_image_metrics(self) -> dict[str, np.ndarray]:
        """
        获取每张图像的指标（用于箱线图等可视化）.
        
        Returns:
            {'IoU': array, 'Dice': array}
        """
        return {
            'IoU': np.array(self.per_image_iou),
            'Dice': np.array(self.per_image_dice),
        }
    
    

    def summary(self, logger_output: bool = True) -> str:
        """
        打印/记录指标摘要.
        
        Args:
            logger_output: 是否同时输出到 logger
        
        Returns:
            格式化的摘要字符串
        """
        metrics = self.get_metrics()
        
        lines = [
            "",
            "=" * 50,
            "Metrics Summary",
            "=" * 50,
            f"Samples:         {self._num_samples}",
            "-" * 50,
            f"mIoU:            {metrics['mIoU']:.4f}",
            f"Dice:            {metrics['Dice']:.4f}",
            f"Pixel Accuracy:  {metrics['Pixel_Acc']:.4f}",
            f"mPixel Accuracy: {metrics['mPixel_Acc']:.4f}",
            f"Precision:       {metrics['Precision']:.4f}",
            f"Recall:          {metrics['Recall']:.4f}",
            f"F1 Score:        {metrics['F1']:.4f}",
            "-" * 50,
            f"Boundary IoU:    {metrics['Boundary_IoU']:.4f}",
            f"Boundary F1:     {metrics['Boundary_F1']:.4f}",
            "-" * 50,
            f"IoU (mean±std):  {metrics['IoU_mean']:.4f} ± {metrics['IoU_std']:.4f}",
            f"Dice (mean±std): {metrics['Dice_mean']:.4f} ± {metrics['Dice_std']:.4f}",
            "-" * 50,
            "Per-class IoU:",
        ]
        
        # 添加每类 IoU
        for name in self.class_names:
            lines.append(f"  {name}: {metrics[f'IoU_{name}']:.4f}")
        
        lines.append("=" * 50)
        lines.append("")
        
        summary_str = "\n".join(lines)
        
        # 输出
        if logger_output:
            self.logger.info(summary_str)
        else:
            print(summary_str)
        
        return summary_str
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"boundary_width={self.boundary_width}, "
            f"class_names={self.class_names})"
        )
    
    def __len__(self) -> int:
        """返回已处理的样本数"""
        return self._num_samples