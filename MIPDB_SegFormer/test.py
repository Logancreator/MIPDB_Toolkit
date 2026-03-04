#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理与评估脚本

有标签输出:
    output_dir/test_YYYYMMDD_HHMMSS/
    ├── test_metrics.json
    ├── pred_masks/          # 预测二值掩码
    ├── gt_masks/            # GT 二值掩码
    ├── pred_overlays/       # 预测叠加原图（天蓝色）
    ├── gt_overlays/         # GT 叠加原图（绿色）
    ├── errors/              # 错误图 TP=绿 FP=红 FN=蓝
    └── comparisons/         # 四格对比图

无标签输出:
    output_dir/test_YYYYMMDD_HHMMSS/
    ├── pred_masks/          # 预测二值掩码
    ├── pred_overlays/       # 预测叠加原图（天蓝色）
    └── comparisons/         # 三格对比（原图 | 预测掩码 | 预测叠加）

使用示例:
    # 有标签
    python test.py --checkpoint best.pth --image_dir ./images --mask_dir ./masks

    # 无标签
    python test.py --checkpoint best.pth --image_dir ./new_images

    # 指定模型和骨干（旧 checkpoint 必须指定）
    python test.py --checkpoint best.pth --image_dir ./images \\
        --model segformer --backbone b4
"""

# 推理时禁止 HuggingFace 联网（必须在 import transformers 之前）
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import get_model, list_models
from datasets import SegmentationDataset
from utils import (
    MetricsCalculator,
    get_val_transform,
    get_tta_transforms,
    get_logger,
    setup_logging,
    get_model_info,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ============================================================
# 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='分割模型推理/评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ===== 必需 =====
    req = parser.add_argument_group('Required')
    req.add_argument('--checkpoint', type=str, required=True, help='检查点路径')
    req.add_argument('--image_dir', type=str, required=True, help='图像目录')

    # ===== 可选 =====
    opt = parser.add_argument_group('Optional')
    opt.add_argument('--mask_dir', type=str, default=None, help='掩码目录（评估用）')
    opt.add_argument('--output_dir', type=str, default='./predictions', help='输出根目录')

    # ===== 模型 =====
    mdl = parser.add_argument_group('Model')
    mdl.add_argument('--model', type=str, default=None, choices=list_models(),
                     help='模型架构（不指定则从 checkpoint 读取）')
    mdl.add_argument('--backbone', type=str, default=None,
                     help='骨干网络（不指定则从 checkpoint 读取）')
    mdl.add_argument('--num_classes', type=int, default=2, help='类别数')
    mdl.add_argument('--input_size', type=int, default=None, help='输入尺寸')

    # ===== 推理 =====
    inf = parser.add_argument_group('Inference')
    inf.add_argument('--batch_size', type=int, default=1, help='批次大小')
    inf.add_argument('--num_workers', type=int, default=0, help='数据加载进程数')
    inf.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    inf.add_argument('--tta', action='store_true', help='测试时增强')

    # ===== 输出 =====
    out = parser.add_argument_group('Output')
    out.add_argument('--save_overlay', action='store_true', default=True,
                     help='保存叠加图（默认开启）')
    out.add_argument('--no_overlay', dest='save_overlay', action='store_false',
                     help='不保存叠加图')
    out.add_argument('--save_comparison', action='store_true', default=True,
                     help='保存对比图（默认开启）')
    out.add_argument('--no_comparison', dest='save_comparison', action='store_false',
                     help='不保存对比图')
    out.add_argument('--save_prob', action='store_true', help='保存概率图')
    out.add_argument('--overlay_alpha', type=float, default=0.5, help='叠加透明度')
    out.add_argument('--class_names', type=str, nargs='+', default=None)

    return parser.parse_args()


# ============================================================
# 推理数据集（无标签）
# ============================================================

class InferenceDataset(Dataset):
    """无掩码推理数据集，返回 (image, dummy_mask, filename)"""

    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def __init__(self, image_dir: str | Path, transform: Any = None) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_list: list[str] = sorted({
            f.name
            for ext in self.SUPPORTED_EXTENSIONS
            for f in list(self.image_dir.glob(f'*{ext}')) + list(self.image_dir.glob(f'*{ext.upper()}'))
        })
        if not self.image_list:
            raise ValueError(f"在 {image_dir} 中未找到图像")

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.image_list[idx]
        image = cv2.imread(str(self.image_dir / filename))
        if image is None:
            raise ValueError(f"无法读取: {self.image_dir / filename}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if self.transform:
            t = self.transform(image=image, mask=mask)
            image, mask = t['image'], t['mask'].long()
        return image, mask, filename


# ============================================================
# 可视化辅助函数
# ============================================================

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """图像张量 → BGR uint8 numpy"""
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip((std * img + mean) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _mask_color(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    vis = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    vis[mask == 1] = color
    return vis


def _overlay(image: np.ndarray, mask: np.ndarray,
             alpha: float, color: tuple[int, int, int]) -> np.ndarray:
    """掩码叠加到原图 + 轮廓线"""
    out = cv2.addWeighted(image, 1 - alpha, _mask_color(mask, color), alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def _error_map(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float) -> np.ndarray:
    err = np.zeros_like(image)
    err[(gt == 1) & (pred == 1)] = [0, 255, 0]    # TP 绿
    err[(gt == 0) & (pred == 1)] = [0, 0, 255]    # FP 红(BGR)
    err[(gt == 1) & (pred == 0)] = [255, 0, 0]    # FN 蓝(BGR)
    return cv2.addWeighted(image, 1 - alpha, err, alpha, 0)


def _add_label(panel: np.ndarray, label: str) -> None:
    h = panel.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = max(0.5, h / 800)
    th = max(1, int(h / 400))
    (tw, tht), _ = cv2.getTextSize(label, font, sc, th)
    cv2.rectangle(panel, (0, 0), (tw + 10, tht + 14), (0, 0, 0), -1)
    cv2.putText(panel, label, (5, tht + 8), font, sc, (255, 255, 255), th)


def _hconcat(panels: list[np.ndarray], gap: int = 2) -> np.ndarray:
    """横向拼接面板，白色间隔"""
    h = panels[0].shape[0]
    sep = np.full((h, gap, 3), 255, dtype=np.uint8)
    parts = []
    for i, p in enumerate(panels):
        if i > 0:
            parts.append(sep)
        parts.append(p)
    return np.concatenate(parts, axis=1)


def _save(img: np.ndarray, filename: str, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / filename), img)


# ============================================================
# 对比图
# ============================================================

def save_comparison_with_gt(
    image: np.ndarray, gt: np.ndarray, pred: np.ndarray,
    filename: str, output_dir: Path, alpha: float,
) -> None:
    """四格对比: 原图+GT轮廓 | GT叠加 | 预测叠加 | 错误图"""
    p1 = image.copy()
    contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, contours, -1, (255, 255, 0), 2)

    p2 = _overlay(image, gt, alpha, (0, 255, 0))
    p3 = _overlay(image, pred, alpha, (255, 180, 0))
    p4 = _error_map(image, pred, gt, alpha)

    labels = ['Image + GT', 'GT Overlay', 'Prediction', 'Error (G:TP R:FP B:FN)']
    panels = [p1, p2, p3, p4]
    for lb, pn in zip(labels, panels):
        _add_label(pn, lb)
    _save(_hconcat(panels), filename, output_dir)


def save_comparison_no_gt(
    image: np.ndarray, pred: np.ndarray,
    filename: str, output_dir: Path, alpha: float,
) -> None:
    """三格对比（无标签）: 原图 | 预测掩码(彩色) | 预测叠加"""
    p1 = image.copy()
    p2 = _mask_color(pred, (255, 180, 0))  # 天蓝色掩码
    p3 = _overlay(image, pred, alpha, (255, 180, 0))

    labels = ['Original', 'Pred Mask', 'Pred Overlay']
    panels = [p1, p2, p3]
    for lb, pn in zip(labels, panels):
        _add_label(pn, lb)
    _save(_hconcat(panels), filename, output_dir)


# 从 checkpoint state_dict 推断 SegFormer backbone 变体
# SegFormer 各变体的 depths（每个 stage 的层数）
_SEGFORMER_DEPTHS = {
    'b0': [2, 2, 2, 2],
    'b1': [2, 2, 2, 2],  # b0 vs b1 需要看 hidden_sizes
    'b2': [3, 4, 6, 3],
    'b3': [3, 4, 18, 3],
    'b4': [3, 8, 27, 3],
    'b5': [3, 6, 40, 3],
}


def _infer_segformer_backbone(state_dict: dict) -> str | None:
    """
    从 state_dict 的 key 推断 SegFormer backbone 变体.

    原理: encoder.block.{stage}.{layer} 的 layer 数量 = depths[stage]
    b0 vs b1 通过 hidden_sizes[0] 区分 (32 vs 64)
    """
    # 找出每个 stage 的最大 layer index
    stage_max: dict[int, int] = {}
    first_hidden = None
    for key in state_dict:
        # 典型 key: model.segformer.encoder.block.2.15.attention.self.query.weight
        parts = key.split('.')
        try:
            block_idx = parts.index('block')
            stage = int(parts[block_idx + 1])
            layer = int(parts[block_idx + 2])
            stage_max[stage] = max(stage_max.get(stage, 0), layer)
        except (ValueError, IndexError):
            pass
        # 获取 hidden_sizes[0] 用于区分 b0 vs b1
        if 'patch_embeddings.0.proj.weight' in key or key.endswith('patch_embeddings.0.proj.weight'):
            first_hidden = state_dict[key].shape[0]

    if len(stage_max) < 4:
        return None

    depths = [stage_max[s] + 1 for s in range(4)]

    # 匹配
    for variant, expected_depths in _SEGFORMER_DEPTHS.items():
        if depths == expected_depths:
            if variant in ('b0', 'b1'):
                # b0: hidden_sizes[0]=32, b1: hidden_sizes[0]=64
                if first_hidden == 32:
                    return 'b0'
                elif first_hidden == 64:
                    return 'b1'
                else:
                    return variant  # 猜不出来就返回第一个匹配
            return variant

    return None


# ============================================================
# 核心推理
# ============================================================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path | None = None,
    has_gt: bool = False,
    num_classes: int = 2,
    class_names: list[str] | None = None,
    save_overlay: bool = True,
    save_comparison: bool = True,
    overlay_alpha: float = 0.5,
) -> dict[str, Any]:
    """单次推理完成指标 + 可视化"""
    logger = get_logger()
    model.eval()

    metrics_calc = None
    if has_gt:
        metrics_calc = MetricsCalculator(num_classes=num_classes, class_names=class_names)

    # 输出子目录
    dirs: dict[str, Path] = {}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        dirs['pred_masks'] = output_dir / 'pred_masks'
        if save_overlay:
            dirs['pred_overlays'] = output_dir / 'pred_overlays'
        if save_comparison:
            dirs['comparisons'] = output_dir / 'comparisons'
        if has_gt:
            dirs['gt_masks'] = output_dir / 'gt_masks'
            dirs['errors'] = output_dir / 'errors'
            if save_overlay:
                dirs['gt_overlays'] = output_dir / 'gt_overlays'

    total_infer_time = 0.0
    count = 0

    for images, masks, filenames in dataloader:
        images = images.to(device)

        # 推理（计时）
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = model(images)['out'].argmax(dim=1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_infer_time += time.perf_counter() - t0
        count += images.size(0)

        # 指标
        if has_gt and metrics_calc is not None:
            metrics_calc.update(preds, masks.to(device))

        if output_dir is None:
            continue

        # 逐张保存可视化
        for i, fname in enumerate(filenames):
            pred_np = preds[i].cpu().numpy()
            img_bgr = _denormalize(images[i])
            alpha = overlay_alpha

            # 预测掩码（始终保存）
            _save((pred_np * 255).astype(np.uint8), fname, dirs['pred_masks'])

            # 预测叠加
            if save_overlay:
                _save(_overlay(img_bgr, pred_np, alpha, (255, 180, 0)),
                      fname, dirs['pred_overlays'])

            if has_gt:
                gt_np = masks[i].cpu().numpy()
                _save((gt_np * 255).astype(np.uint8), fname, dirs['gt_masks'])
                _save(_error_map(img_bgr, pred_np, gt_np, alpha), fname, dirs['errors'])
                if save_overlay:
                    _save(_overlay(img_bgr, gt_np, alpha, (0, 255, 0)),
                          fname, dirs['gt_overlays'])
                if save_comparison:
                    save_comparison_with_gt(img_bgr, gt_np, pred_np, fname,
                                           dirs['comparisons'], alpha)
            else:
                # 无标签: 三格对比（原图 | 掩码 | 叠加）
                if save_comparison:
                    save_comparison_no_gt(img_bgr, pred_np, fname,
                                         dirs['comparisons'], alpha)

    # 汇总
    result: dict[str, Any] = {}

    fps = count / total_infer_time if total_infer_time > 0 else 0
    latency = (total_infer_time / count * 1000) if count > 0 else 0
    result['speed'] = {
        'count': count,
        'total_time_s': round(total_infer_time, 2),
        'fps': round(fps, 2),
        'latency_ms': round(latency, 2),
    }
    logger.info(f"处理 {count} 张，推理 {total_infer_time:.2f}s | {fps:.1f} FPS ({latency:.1f} ms/img)")

    if has_gt and metrics_calc is not None:
        metrics = metrics_calc.get_metrics()
        result['metrics'] = metrics

        logger.info("=" * 50)
        logger.info("测试结果")
        logger.info("=" * 50)
        logger.info(f"mIoU:            {metrics['mIoU']:.4f}")
        logger.info(f"Dice:            {metrics['Dice']:.4f}")
        logger.info(f"Pixel Accuracy:  {metrics['Pixel_Acc']:.4f}")
        logger.info(f"mPixel Accuracy: {metrics['mPixel_Acc']:.4f}")
        logger.info(f"Precision:       {metrics['Precision']:.4f}")
        logger.info(f"Recall:          {metrics['Recall']:.4f}")
        logger.info(f"F1 Score:        {metrics['F1']:.4f}")
        logger.info("-" * 50)
        logger.info(f"Boundary IoU:    {metrics['Boundary_IoU']:.4f}")
        logger.info(f"Boundary F1:     {metrics['Boundary_F1']:.4f}")
        logger.info("-" * 50)
        logger.info(f"IoU (mean±std):  {metrics['IoU_mean']:.4f} ± {metrics['IoU_std']:.4f}")
        logger.info(f"Dice (mean±std): {metrics['Dice_mean']:.4f} ± {metrics['Dice_std']:.4f}")
        if class_names:
            logger.info("-" * 50)
            for name in class_names:
                logger.info(f"  IoU_{name}: {metrics[f'IoU_{name}']:.4f}")
        logger.info("=" * 50)

    # 输出汇总
    if output_dir:
        logger.info(f"结果保存到: {output_dir}")
        for name, d in dirs.items():
            if d.exists():
                n = len(list(d.iterdir()))
                logger.info(f"  {name + '/':20s} {n} 张")

    return result


# ============================================================
# 模型加载（完全离线）
# ============================================================

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    model_name: str | None = None,
    backbone: str | None = None,
    num_classes: int = 2,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    从检查点加载模型（完全离线，不联网）.

    参数优先级: 命令行 > checkpoint 存储 > 自动推断 > 默认值
    """
    logger = get_logger()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

    logger.info(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # checkpoint
    ckpt_keys = [k for k in checkpoint if k != 'model_state_dict' and k != 'optimizer_state_dict']
    logger.info(f"Checkpoint 元信息: {ckpt_keys}")

    # odel_name
    resolved_model = model_name or checkpoint.get('model_name', None)
    if resolved_model is None:
        raise ValueError(
            "无法确定模型架构！checkpoint 中没有 model_name。\n"
            "请用 --model 指定，例如: --model segformer / --model deeplabv3"
        )

    # backbone
    resolved_backbone = backbone or checkpoint.get('backbone', None)

    if resolved_backbone is None:
        if resolved_model == 'segformer':
            inferred = _infer_segformer_backbone(checkpoint.get('model_state_dict', {}))
            if inferred:
                resolved_backbone = inferred
                logger.info(f"从 state_dict 自动推断 SegFormer backbone: {resolved_backbone}")
            else:
                raise ValueError(
                    "无法推断 SegFormer backbone！\n"
                    "请用 --backbone 指定，例如: --backbone b4"
                )
        elif resolved_model == 'mask2former':
            raise ValueError(
                "checkpoint 中没有 backbone 信息。\n"
                "请用 --backbone 指定，例如: --backbone swin-tiny"
            )
        else:
            resolved_backbone = 'resnet50'
            logger.warning(f"checkpoint 无 backbone，使用默认: {resolved_backbone}")

    # num_classes
    ckpt_num_classes = checkpoint.get('num_classes', None)
    if ckpt_num_classes is not None and num_classes == 2 and ckpt_num_classes != 2:
        logger.info(f"从 checkpoint 恢复 num_classes={ckpt_num_classes}")
        num_classes = ckpt_num_classes

    logger.info(f"模型: {resolved_model} | 骨干: {resolved_backbone} | 类别数: {num_classes}")

    # 构建模型参数
    model_kwargs: dict[str, Any] = {
        'num_classes': num_classes,
        'backbone': resolved_backbone,
        'pretrained': False,  # 推理时不下载预训练权重
    }

    # 从 checkpoint恢复config
    hf_config = checkpoint.get('hf_config', None)
    if hf_config is not None:
        model_kwargs['hf_config_dict'] = hf_config
        logger.info("✓ 使用 checkpoint 中保存的 HF config 恢复架构（最可靠）")
    elif resolved_model in ('segformer', 'mask2former'):
        logger.warning(
            f" 请重新训练"
        )

    # 创建模型
    logger.info("创建模型架构...")
    model = get_model(resolved_model, **model_kwargs)

    # 加载权重
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()

    # 先检查 hape是否匹配
    shape_mismatches = []
    missing_in_ckpt = []
    unexpected_in_ckpt = []

    for key in model_state:
        if key in state_dict:
            if state_dict[key].shape != model_state[key].shape:
                shape_mismatches.append(
                    f"  {key}: checkpoint={list(state_dict[key].shape)} vs model={list(model_state[key].shape)}"
                )
        else:
            missing_in_ckpt.append(key)

    for key in state_dict:
        if key not in model_state:
            unexpected_in_ckpt.append(key)

    if shape_mismatches:
        msg = "权重 shape 不匹配！\n" + "\n".join(shape_mismatches[:15])
        if len(shape_mismatches) > 15:
            msg += f"\n  ...共 {len(shape_mismatches)} 个不匹配"
        msg += (
            "\n\n可能原因:\n"
            f"  1. backbone 不对 → 确认 --backbone {resolved_backbone} 是否与训练一致\n"
            f"  2. num_classes 不对 → 确认 --num_classes {num_classes} 是否与训练一致\n"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if missing_in_ckpt or unexpected_in_ckpt:
        if missing_in_ckpt:
            logger.info(f"模型有 {len(missing_in_ckpt)} 个 key 在 checkpoint 中缺失（通常是 buffer，无影响）")
            logger.debug(f"  缺失: {missing_in_ckpt[:5]}")
        if unexpected_in_ckpt:
            logger.info(f"checkpoint 有 {len(unexpected_in_ckpt)} 个多余 key（已忽略）")
            logger.debug(f"  多余: {unexpected_in_ckpt[:5]}")
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)

    logger.info("✓ 权重加载成功")
    model.to(device)
    model.eval()

    # 打印checkpoint
    if 'metrics' in checkpoint:
        m = checkpoint['metrics']
        logger.info("-" * 40)
        logger.info("检查点训练指标:")
        logger.info(f"  mIoU:  {m.get('val_mIoU', m.get('mIoU', 'N/A'))}")
        logger.info(f"  Dice:  {m.get('Dice', 'N/A')}")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, checkpoint


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    args = parse_args()

    # 时间戳子目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'test_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, _ = setup_logging(output_dir, name='test')
    logger.info("=" * 60)
    logger.info("分割模型推理")
    logger.info("=" * 60)

    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，使用 CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"设备: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 模型
    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        model_name=args.model,
        backbone=args.backbone,
        num_classes=args.num_classes,
    )

    if args.input_size:
        info = get_model_info(model, args.input_size, device)
        logger.info(f"参数量: {info['params_M']:.2f}M | FLOPs: {info['flops_str']} | {info['fps']:.1f} FPS")

    # 数据
    transform = get_val_transform(args.input_size)
    class_names = args.class_names
    if class_names is None and args.num_classes == 2:
        class_names = ['background', 'foreground']

    has_gt = args.mask_dir is not None

    if has_gt:
        logger.info(f"模式: 有标签评估 | 图像: {args.image_dir} | 掩码: {args.mask_dir}")
        dataset = SegmentationDataset(args.image_dir, args.mask_dir, transform=transform)
    else:
        logger.info(f"模式: 无标签推理 | 图像: {args.image_dir}")
        dataset = InferenceDataset(args.image_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    logger.info(f"图像数: {len(dataset)}")
    logger.info("-" * 40)

    # 推理
    result = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        has_gt=has_gt,
        num_classes=args.num_classes,
        class_names=class_names,
        save_overlay=args.save_overlay,
        save_comparison=args.save_comparison,
        overlay_alpha=args.overlay_alpha,
    )

    # 保存指标
    if 'metrics' in result:
        p = output_dir / 'test_metrics.json'
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({**result['metrics'], 'speed': result['speed']}, f, indent=2, ensure_ascii=False)
        logger.info(f"指标: {p}")
    else:
        p = output_dir / 'test_speed.json'
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(result['speed'], f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"完成! 结果保存到: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
