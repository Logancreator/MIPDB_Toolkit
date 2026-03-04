#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset classes for segmentation"""

from __future__ import annotations

# 标准库
import logging
import random
from pathlib import Path
from typing import Any, Callable

# 第三方库
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_logger() -> logging.Logger:
    """获取当前模块的日志记录器"""
    return logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

SUPPORTED_EXTENSIONS: tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


# ============================================================
# 二分类分割数据集
# ============================================================

class SegmentationDataset(Dataset):
    """
    通用语义分割数据集.
    
    支持二分类和多分类分割任务。
    
    Args:
        image_dir: 图像目录路径
        mask_dir: 掩码目录路径
        image_list: 图像文件名列表（可选，默认自动扫描）
        transform: Albumentations 变换
        mask_suffix: 掩码文件后缀（如果与图像不同）
        mask_threshold: 二值化阈值（用于二分类）
    
    Example:
        >>> dataset = SegmentationDataset('./images', './masks', transform=train_transform)
        >>> image, mask, filename = dataset[0]
        >>> print(image.shape, mask.shape)  # [3, H, W], [H, W]
    """
    
    def __init__(
        self, 
        image_dir: str | Path, 
        mask_dir: str | Path, 
        image_list: list[str] | None = None, 
        transform: Callable | None = None,
        mask_suffix: str | None = None,
        mask_threshold: int = 127
    ) -> None:
        self.logger = get_logger()
        
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.mask_threshold = mask_threshold
        
        # 验证目录存在
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")
        
        # 获取图像列表
        if image_list is not None:
            self.image_list = image_list
        else:
            self.image_list = self._scan_images()
        
        if len(self.image_list) == 0:
            raise ValueError(f"在 {self.image_dir} 中未找到图像文件")
        
        self.logger.debug(f"数据集初始化: {len(self.image_list)} 张图像")
    
    def _scan_images(self) -> list[str]:
        """扫描图像目录获取文件列表"""
        images = []
        for ext in SUPPORTED_EXTENSIONS:
            images.extend([f.name for f in self.image_dir.glob(f'*{ext}')])
            images.extend([f.name for f in self.image_dir.glob(f'*{ext.upper()}')])
        return sorted(list(set(images)))
    
    def _find_mask_path(self, filename: str) -> Path:
        """查找对应的掩码文件"""
        stem = Path(filename).stem
        
        # 添加后缀（如果指定）
        if self.mask_suffix:
            stem = stem + self.mask_suffix
        
        # 尝试不同扩展名
        for ext in SUPPORTED_EXTENSIONS:
            candidate = self.mask_dir / f'{stem}{ext}'
            if candidate.exists():
                return candidate
        
        # 回退：使用原始文件名
        return self.mask_dir / filename
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.image_list[idx]
        image_path = self.image_dir / filename
        mask_path = self._find_mask_path(filename)
        
        # 读取图像 (BGR) 和掩码 (灰度)
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码: {mask_path}")
        
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 二值化掩码 (0/255 -> 0/1)
        mask = (mask > self.mask_threshold).astype(np.uint8)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask, filename
    
    def get_sample_info(self, idx: int) -> dict[str, Any]:
        """获取样本信息（不加载完整数据）"""
        filename = self.image_list[idx]
        image_path = self.image_dir / filename
        mask_path = self._find_mask_path(filename)
        
        # 只读取图像尺寸
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
        else:
            h, w = 0, 0
        
        return {
            'filename': filename,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'height': h,
            'width': w,
        }
    
    def get_statistics(self) -> dict[str, Any]:
        """获取数据集统计信息"""
        heights, widths = [], []
        
        for filename in self.image_list[:min(100, len(self.image_list))]:  # 采样
            image_path = self.image_dir / filename
            img = cv2.imread(str(image_path))
            if img is not None:
                heights.append(img.shape[0])
                widths.append(img.shape[1])
        
        return {
            'num_samples': len(self.image_list),
            'avg_height': np.mean(heights) if heights else 0,
            'avg_width': np.mean(widths) if widths else 0,
            'min_height': min(heights) if heights else 0,
            'max_height': max(heights) if heights else 0,
            'min_width': min(widths) if widths else 0,
            'max_width': max(widths) if widths else 0,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"image_dir='{self.image_dir}', "
            f"mask_dir='{self.mask_dir}', "
            f"num_samples={len(self)})"
        )


# ============================================================
# 多分类分割数据集
# ============================================================

class MultiClassSegmentationDataset(Dataset):
    """
    多分类语义分割数据集.
    
    掩码值应为类别索引 (0, 1, 2, ...)。
    
    Args:
        image_dir: 图像目录路径
        mask_dir: 掩码目录路径
        image_list: 图像文件名列表（可选）
        transform: Albumentations 变换
        num_classes: 类别数量（可选，用于验证）
        ignore_index: 忽略的类别索引（如 255）
    """
    
    def __init__(
        self, 
        image_dir: str | Path, 
        mask_dir: str | Path, 
        image_list: list[str] | None = None, 
        transform: Callable | None = None,
        num_classes: int | None = None,
        ignore_index: int | None = None
    ) -> None:
        self.logger = get_logger()
        
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # 验证目录
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")
        
        # 获取图像列表
        if image_list is not None:
            self.image_list = image_list
        else:
            self.image_list = self._scan_images()
        
        if len(self.image_list) == 0:
            raise ValueError(f"在 {self.image_dir} 中未找到图像文件")
        
        self.logger.debug(f"多分类数据集初始化: {len(self.image_list)} 张图像")
    
    def _scan_images(self) -> list[str]:
        """扫描图像目录"""
        images = []
        for ext in SUPPORTED_EXTENSIONS:
            images.extend([f.name for f in self.image_dir.glob(f'*{ext}')])
            images.extend([f.name for f in self.image_dir.glob(f'*{ext.upper()}')])
        return sorted(list(set(images)))
    
    def _find_mask_path(self, filename: str) -> Path:
        """查找对应的掩码文件"""
        stem = Path(filename).stem
        
        # 优先使用 PNG/TIFF（无损）
        for ext in ['.png', '.tif', '.tiff']:
            candidate = self.mask_dir / f'{stem}{ext}'
            if candidate.exists():
                return candidate
        
        return self.mask_dir / filename
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.image_list[idx]
        image_path = self.image_dir / filename
        mask_path = self._find_mask_path(filename)
        
        # 读取图像和掩码
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码: {mask_path}")
        
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 确保掩码是单通道
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask = mask.astype(np.uint8)
        
        # 验证类别范围
        if self.num_classes is not None:
            valid_mask = mask < self.num_classes
            if self.ignore_index is not None:
                valid_mask |= (mask == self.ignore_index)
            if not valid_mask.all():
                self.logger.warning(
                    f"掩码 {filename} 包含超出范围的类别值"
                )
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask, filename
    
    def get_class_distribution(self, num_samples: int = 100) -> dict[int, int]:
        """获取类别分布（采样）"""
        distribution: dict[int, int] = {}
        
        sample_indices = random.sample(
            range(len(self)), 
            min(num_samples, len(self))
        )
        
        for idx in sample_indices:
            filename = self.image_list[idx]
            mask_path = self._find_mask_path(filename)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            
            if mask is not None:
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                
                unique, counts = np.unique(mask, return_counts=True)
                for cls, cnt in zip(unique, counts):
                    distribution[int(cls)] = distribution.get(int(cls), 0) + int(cnt)
        
        return distribution
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"image_dir='{self.image_dir}', "
            f"num_samples={len(self)}, "
            f"num_classes={self.num_classes})"
        )


# ============================================================
# 数据加载器创建
# ============================================================

def create_dataloaders(
    image_dir: str | Path, 
    mask_dir: str | Path, 
    train_transform: Callable | None,
    val_transform: Callable | None,
    val_split: float = 0.2, 
    batch_size: int = 4, 
    num_workers: int = 0, 
    seed: int = 42,
    pin_memory: bool = True,
    drop_last: bool = True,
    dataset_class: type = SegmentationDataset,
    **dataset_kwargs: Any
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    """
    创建训练和验证数据加载器.
    
    Args:
        image_dir: 图像目录
        mask_dir: 掩码目录
        train_transform: 训练数据变换
        val_transform: 验证数据变换
        val_split: 验证集比例
        batch_size: 批次大小
        num_workers: 数据加载进程数
        seed: 随机种子
        pin_memory: 是否锁页内存
        drop_last: 是否丢弃最后不完整的批次
        dataset_class: 数据集类
        **dataset_kwargs: 传递给数据集的额外参数
    
    Returns:
        (train_loader, val_loader, train_images, val_images)
    
    Example:
        >>> train_loader, val_loader, train_imgs, val_imgs = create_dataloaders(
        ...     './images', './masks',
        ...     train_transform, val_transform,
        ...     val_split=0.2, batch_size=8
        ... )
    """
    logger = get_logger()
    
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    # 获取所有图像
    all_images = []
    for ext in SUPPORTED_EXTENSIONS:
        all_images.extend([f.name for f in image_dir.glob(f'*{ext}')])
        all_images.extend([f.name for f in image_dir.glob(f'*{ext.upper()}')])
    all_images = sorted(list(set(all_images)))
    
    if len(all_images) == 0:
        raise ValueError(f"在 {image_dir} 中未找到图像文件")
    
    # 随机打乱并划分
    random.seed(seed)
    random.shuffle(all_images)
    
    val_size = int(len(all_images) * val_split)
    train_images = all_images[val_size:]
    val_images = all_images[:val_size]
    
    logger.info(f"数据划分: 训练 {len(train_images)} | 验证 {len(val_images)}")
    
    # 创建数据集
    train_dataset = dataset_class(
        image_dir, mask_dir, train_images, train_transform, **dataset_kwargs
    )
    val_dataset = dataset_class(
        image_dir, mask_dir, val_images, val_transform, **dataset_kwargs
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=drop_last
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, train_images, val_images


def create_test_dataloader(
    image_dir: str | Path,
    mask_dir: str | Path | None,
    transform: Callable | None,
    batch_size: int = 1,
    num_workers: int = 0,
    dataset_class: type = SegmentationDataset,
    **dataset_kwargs: Any
) -> DataLoader:
    """
    创建测试数据加载器.
    
    Args:
        image_dir: 图像目录
        mask_dir: 掩码目录（可选，如果 None 则使用 image_dir 占位）
        transform: 数据变换
        batch_size: 批次大小
        num_workers: 数据加载进程数
        dataset_class: 数据集类
        **dataset_kwargs: 传递给数据集的额外参数
    
    Returns:
        test_loader
    """
    if mask_dir is None:
        mask_dir = image_dir  # 占位
    
    dataset = dataset_class(
        image_dir, mask_dir, transform=transform, **dataset_kwargs
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


# ============================================================
# 导出
# ============================================================

__all__ = [
    'SegmentationDataset',
    'MultiClassSegmentationDataset',
    'create_dataloaders',
    'create_test_dataloader',
    'SUPPORTED_EXTENSIONS',
]