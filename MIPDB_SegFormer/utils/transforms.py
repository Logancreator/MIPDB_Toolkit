#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data augmentation transforms using Albumentations"""

from __future__ import annotations

# 标准库
import logging
from typing import Literal

# 第三方库
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_logger() -> logging.Logger:
    """获取当前模块的日志记录器"""
    return logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

# ImageNet 归一化参数
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# 增强强度类型
AugStrength = Literal['light', 'medium', 'strong']


# ============================================================
# 数据增强配置
# ============================================================

def _get_light_augmentations() -> list[A.BasicTransform]:
    """轻度增强：仅基本翻转和轻微颜色变化"""
    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.1, 
            p=0.3
        ),
    ]


def _get_medium_augmentations() -> list[A.BasicTransform]:
    """中度增强：添加旋转、仿射变换和噪声"""
    return [
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.85, 1.15),
            rotate=(-30, 30),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            p=0.5
        ),
        
        # 颜色变换
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        
        # 噪声
        A.GaussNoise(std_range=(0.05, 0.2), p=0.3),
    ]


def _get_strong_augmentations() -> list[A.BasicTransform]:
    """强度增强：全面的几何、颜色、噪声和遮挡变换"""
    return [
        # ===== 几何变换 =====
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.7, 1.3),
            rotate=(-45, 45),
            shear=(-15, 15),
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            p=0.7
        ),
        A.Perspective(scale=(0.02, 0.08), p=0.3),
        
        # ===== 颜色变换 =====
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.RandomGamma(gamma_limit=(70, 130)),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
        ], p=0.7),
        
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20
            ),
            A.RGBShift(
                r_shift_limit=20, 
                g_shift_limit=20, 
                b_shift_limit=20
            ),
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1
            ),
        ], p=0.5),
        
        # ===== 噪声和模糊 =====
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.3)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),
        
        # ===== 随机遮挡 =====
        A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0,
            p=0.3
        ),
    ]


# ============================================================
# 主要接口
# ============================================================

def get_train_transform(
    strength: AugStrength = 'strong',
    input_size: int | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD
) -> A.Compose:
    """
    获取训练数据增强变换.
    
    Args:
        strength: 增强强度，可选 'light', 'medium', 'strong'
        input_size: 调整图像大小（可选）
        mean: 归一化均值
        std: 归一化标准差
    
    Returns:
        Albumentations Compose 变换
    
    Example:
        transform = get_train_transform(strength='strong', input_size=512)
        augmented = transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']
    """
    logger = get_logger()
    transforms_list: list[A.BasicTransform] = []
    
    # 调整大小
    if input_size is not None:
        transforms_list.append(A.Resize(input_size, input_size))
    
    # 根据强度获取增强
    augmentation_map = {
        'light': _get_light_augmentations,
        'medium': _get_medium_augmentations,
        'strong': _get_strong_augmentations,
    }
    
    if strength not in augmentation_map:
        raise ValueError(
            f"不支持的增强强度: {strength}，"
            f"请选择: {list(augmentation_map.keys())}"
        )
    
    transforms_list.extend(augmentation_map[strength]())
    
    # 归一化和转换为 Tensor
    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    logger.debug(f"训练变换已创建 | 强度: {strength} | 输入大小: {input_size}")
    
    return A.Compose(transforms_list)


def get_val_transform(
    input_size: int | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD
) -> A.Compose:
    """
    获取验证/测试数据变换（仅调整大小和归一化）.
    
    Args:
        input_size: 调整图像大小（可选）
        mean: 归一化均值
        std: 归一化标准差
    
    Returns:
        Albumentations Compose 变换
    """
    logger = get_logger()
    transforms_list: list[A.BasicTransform] = []
    
    if input_size is not None:
        transforms_list.append(A.Resize(input_size, input_size))
    
    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    logger.debug(f"验证变换已创建 | 输入大小: {input_size}")
    
    return A.Compose(transforms_list)


def get_test_transform(
    input_size: int | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD
) -> A.Compose:
    """
    获取测试数据变换（与验证变换相同）.
    
    Args:
        input_size: 调整图像大小（可选）
        mean: 归一化均值
        std: 归一化标准差
    
    Returns:
        Albumentations Compose 变换
    """
    return get_val_transform(input_size, mean, std)


# ============================================================
# 自定义变换构建器
# ============================================================

def get_custom_transform(
    input_size: int | None = None,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotate90: bool = True,
    affine: bool = False,
    color_jitter: bool = False,
    gaussian_noise: bool = False,
    coarse_dropout: bool = False,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD
) -> A.Compose:
    """
    获取自定义数据增强变换.
    
    允许精细控制每种增强的开关。
    
    Args:
        input_size: 调整图像大小（可选）
        horizontal_flip: 是否水平翻转
        vertical_flip: 是否垂直翻转
        rotate90: 是否随机旋转 90°
        affine: 是否仿射变换
        color_jitter: 是否颜色抖动
        gaussian_noise: 是否高斯噪声
        coarse_dropout: 是否随机遮挡
        mean: 归一化均值
        std: 归一化标准差
    
    Returns:
        Albumentations Compose 变换
    
    Example:
        # 只使用翻转和颜色抖动
        transform = get_custom_transform(
            input_size=512,
            horizontal_flip=True,
            vertical_flip=True,
            color_jitter=True,
            affine=False,
        )
    """
    transforms_list: list[A.BasicTransform] = []
    
    # 调整大小
    if input_size is not None:
        transforms_list.append(A.Resize(input_size, input_size))
    
    # 几何变换
    if horizontal_flip:
        transforms_list.append(A.HorizontalFlip(p=0.5))
    
    if vertical_flip:
        transforms_list.append(A.VerticalFlip(p=0.5))
    
    if rotate90:
        transforms_list.append(A.RandomRotate90(p=0.5))
    
    if affine:
        transforms_list.append(A.Affine(
            scale=(0.8, 1.2),
            rotate=(-30, 30),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            p=0.5
        ))
    
    # 颜色变换
    if color_jitter:
        transforms_list.append(A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ))
    
    # 噪声
    if gaussian_noise:
        transforms_list.append(A.GaussNoise(std_range=(0.05, 0.2), p=0.3))
    
    # 遮挡
    if coarse_dropout:
        transforms_list.append(A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0,
            p=0.3
        ))
    
    # 归一化和转换
    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


# ============================================================
# TTA (Test Time Augmentation)
# ============================================================

def get_tta_transforms(
    input_size: int | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD
) -> list[A.Compose]:
    """
    获取测试时增强 (TTA) 变换列表.
    
    返回多个变换，每个对应不同的翻转/旋转组合。
    
    Args:
        input_size: 调整图像大小（可选）
        mean: 归一化均值
        std: 归一化标准差
    
    Returns:
        变换列表（原图 + 各种翻转/旋转）
    
    Example:
        tta_transforms = get_tta_transforms(input_size=512)
        predictions = []
        for transform in tta_transforms:
            augmented = transform(image=image)
            pred = model(augmented['image'].unsqueeze(0))
            predictions.append(pred)
        # 对 predictions 取平均或投票
    """
    base_transforms: list[A.BasicTransform] = []
    if input_size is not None:
        base_transforms.append(A.Resize(input_size, input_size))
    
    final_transforms = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    
    tta_list: list[A.Compose] = []
    
    # 原图
    tta_list.append(A.Compose(base_transforms + final_transforms))
    
    # 水平翻转
    tta_list.append(A.Compose(
        base_transforms + [A.HorizontalFlip(p=1.0)] + final_transforms
    ))
    
    # 垂直翻转
    tta_list.append(A.Compose(
        base_transforms + [A.VerticalFlip(p=1.0)] + final_transforms
    ))
    
    # 水平 + 垂直翻转
    tta_list.append(A.Compose(
        base_transforms + [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0)
        ] + final_transforms
    ))
    
    # 旋转 90°, 180°, 270°
    for k in [1, 2, 3]:
        tta_list.append(A.Compose(
            base_transforms + [
                A.Rotate(limit=(90 * k, 90 * k), p=1.0, border_mode=0)
            ] + final_transforms
        ))
    
    return tta_list


def list_available_strengths() -> list[str]:
    """列出所有可用的增强强度"""
    return ['light', 'medium', 'strong']