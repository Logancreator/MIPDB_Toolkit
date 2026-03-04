
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本

优先级：命令行参数 > 配置文件 > 默认值
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import warnings
warnings.filterwarnings("ignore", message="Some weights of")

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import get_model, list_models
from datasets import create_dataloaders
from utils import (
    MetricsCalculator,
    TrainingPlotter,
    get_train_transform,
    get_val_transform,
    get_logger,
    setup_logging,
    set_seed,
    CSVLogger,
    EarlyStopping,
    AverageMeter,
    Timer,
    get_model_info,
    save_model_info,
    save_checkpoint,
    load_checkpoint,
)


# ============================================================
# 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数.
    
    优先级: 命令行 > 配置文件 > 默认值
    """
    parser = argparse.ArgumentParser(
        description='Train segmentation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None, help='YAML 配置文件路径')
    
    # 数据参数
    parser.add_argument('--image_dir', type=str, default=None, help='训练图像目录')
    parser.add_argument('--mask_dir', type=str, default=None, help='掩码图像目录')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--input_size', type=int, default=512, help='输入图像大小')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数')
    parser.add_argument('--class_names', type=str, nargs='+', default=None, help='类别名称')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='deeplabv3', choices=list_models(), help='模型架构')
    parser.add_argument('--backbone', type=str, default='resnet50', help='骨干网络')
    parser.add_argument('--from_scratch', action='store_true', default=False,help='从头训练（不使用预训练权重）')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='预训练权重路径')
    # parser.add_argument('--local_model_path', type=str, default=None, help='本地模型路径')
    parser.add_argument('--freeze_bn', type=bool, default=True, help='冻结 BatchNorm')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载进程数')
    parser.add_argument('--aux_weight', type=float, default=0.4, help='辅助损失权重')
    
    # 调度器参数
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau', 'none'], help='调度器')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='最小学习率')
    parser.add_argument('--step_size', type=int, default=50, help='StepLR 步长')
    parser.add_argument('--step_gamma', type=float, default=0.5, help='StepLR 衰减系数')
    
    # 早停
    parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
    
    # 增强
    parser.add_argument('--aug_strength', type=str, default='strong', choices=['light', 'medium', 'strong'], help='增强强度')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./runs', help='保存目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')
    parser.add_argument('--save_freq', type=int, default=10, help='保存频率')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练路径')
    
    # 可视化
    parser.add_argument('--plot_freq', type=int, default=5, help='绘图频率')
    parser.add_argument('--num_vis_samples', type=int, default=4, help='可视化样本数')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--deterministic', action='store_true', default=False, help='确定性模式')
    
    # 第一次解析，获取所有参数（包括默认值）
    args = parser.parse_args()
    
    # 如果有配置文件，加载并覆盖默认值（但不覆盖命令行指定的参数）
    if args.config is not None:
        args = load_and_merge_config(args)
    
    # 验证必需参数
    if args.image_dir is None or args.mask_dir is None:
        parser.error("--image_dir 和 --mask_dir 是必需的")
    
    return args


def get_cli_args() -> set[str]:
    """获取命令行中显式指定的参数名"""
    cli_args = set()
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--'):
            # 提取参数名（去掉 -- 前缀，处理 --param=value 格式）
            param = arg[2:].split('=')[0].replace('-', '_')
            cli_args.add(param)
    
    return cli_args


def load_and_merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    加载配置文件并与参数合并.
    
    只覆盖命令行未显式指定的参数。
    支持模型特有参数（会动态添加到 args）。
    """
    config_path = Path(args.config)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 加载 YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 展平嵌套配置（支持两种格式）
    flat_config: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            # 嵌套格式: model: {name: unet, backbone: resnet50}
            flat_config.update(value)
        else:
            # 扁平格式: model: unet
            flat_config[key] = value
    
    # 获取命令行显式指定的参数
    cli_args = get_cli_args()
    
    # 用配置文件的值覆盖默认值（但不覆盖命令行参数）
    for key, value in flat_config.items():
        if key not in cli_args:
            # 动态添加参数（支持模型特有参数如 M, N 等）
            setattr(args, key, value)
    
    return args


# ============================================================
# 训练和验证
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 2,
    aux_weight: float = 0.4
) -> tuple[float, dict[str, float]]:
    """训练一个 epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator(num_classes=num_classes)
    
    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs['out'], masks)
        
        # 辅助损失
        if 'aux' in outputs:
            loss = loss + aux_weight * criterion(outputs['aux'], masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新统计
        loss_meter.update(loss.item(), images.size(0))
        preds = outputs['out'].argmax(dim=1)
        metrics_calc.update(preds, masks)

    
    return loss_meter.avg, metrics_calc.get_metrics()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
    class_names: list[str] | None = None
) -> tuple[float, dict[str, float], Any, dict[str, Any], tuple]:
    """验证模型"""
    model.eval()
    
    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator(num_classes=num_classes, class_names=class_names)
    
    vis_data = (None, None, None)
    
    for batch_idx, (images, masks, _) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        
        loss_meter.update(loss.item(), images.size(0))
        preds = outputs.argmax(dim=1)
        metrics_calc.update(preds, masks)
        
        # 保存第一个 batch 用于可视化
        if batch_idx == 0:
            vis_data = (images.cpu(), masks.cpu(), preds.cpu())
    
    return (
        loss_meter.avg,
        metrics_calc.get_metrics(),
        metrics_calc.get_confusion_matrix(),
        metrics_calc.get_per_image_metrics(),
        vis_data
    )


# ============================================================
# 构建器
# ============================================================

def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """构建模型
    
    支持通过 YAML 配置文件传递模型特有参数。
    
    """
    # 基础参数（所有模型通用）
    model_kwargs: dict[str, Any] = {
        'num_classes': args.num_classes,
        'backbone': args.backbone,
        'pretrained': not args.from_scratch,
    }
    
    # 预训练权重
    if args.pretrained_weights:
        model_kwargs['pretrained_weights'] = args.pretrained_weights
    
    # DeepLabV3 特有参数
    if args.model == 'deeplabv3':
        model_kwargs['freeze_bn'] = args.freeze_bn
    
    # 模型特有参数映射
    MODEL_SPECIFIC_PARAMS: dict[str, list[str]] = {
        'cgnet': ['M', 'N'],
        'icnet': [],  # ICNet 通过 backbone 配置
        'enet': [],   # ENet 无特有参数
        'hrnet': [],  # HRNet 通过 backbone (w18/w32/w48) 配置
        'segnext': [],  # SegNeXt 通过 backbone (tiny/small/base) 配置
        'danet': [],
        'ocrnet': [],
        'segformer': [],
        'mask2former': [],
    }
    
    # 传递模型特有参数
    if args.model in MODEL_SPECIFIC_PARAMS:
        for param_name in MODEL_SPECIFIC_PARAMS[args.model]:
            if hasattr(args, param_name):
                model_kwargs[param_name] = getattr(args, param_name)
    
    for key in vars(args):
        if key.startswith('model_'):
            param_name = key[6:]  # 去掉 'model_' 前缀
            model_kwargs[param_name] = getattr(args, key)
    
    model = get_model(args.model, **model_kwargs)
    model.to(device)
    return model


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """构建学习率调度器"""
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )
    elif args.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=args.min_lr
        )
    return None


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    args = parse_args()
    
    # 生成实验名称
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.model}_{timestamp}"
    
    save_dir = Path(args.save_dir) / args.exp_name
    
    # 设置日志和随机种子 
    logger, timestamp = setup_logging(save_dir)
    set_seed(args.seed, deterministic=args.deterministic)
    
    # 打印配置
    logger.info("=" * 60)
    logger.info(f"训练 {args.model.upper()}")
    logger.info("=" * 60)
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key}: {value}")
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
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 类别名称
    class_names = args.class_names
    if class_names is None and args.num_classes == 2:
        class_names = ['background', 'foreground']
    
    # 数据加载器
    logger.info("-" * 40)
    logger.info("准备数据...")
    
    train_transform = get_train_transform(args.aug_strength, args.input_size)
    val_transform = get_val_transform(args.input_size)
    
    train_loader, val_loader, train_images, val_images = create_dataloaders(
        args.image_dir, args.mask_dir,
        train_transform, val_transform,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    logger.info(f"数据集: 共 {len(train_images) + len(val_images)} 张 | 训练: {len(train_images)} | 验证: {len(val_images)}")
    logger.info(f"增强强度: {args.aug_strength}")
    
    # 构建模型
    logger.info("-" * 40)
    logger.info("构建模型...")
    
    model = build_model(args, device)
    logger.info(f"模型: {args.model} | 骨干: {args.backbone}")
    
    # 模型信息
    model_info = get_model_info(model, args.input_size, device)
    logger.info(f"参数量: {model_info['params_M']:.2f}M")
    logger.info(f"FLOPs: {model_info['flops_str']}")
    logger.info(f"速度: {model_info['fps']:.2f} FPS ({model_info['latency_ms']:.2f} ms)")
    
    save_model_info(model_info, args.model, save_dir / 'model_info.txt')
    
    # 损失函数、优化器、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args)
    
    logger.info(f"优化器: AdamW | LR: {args.lr} | 权重衰减: {args.weight_decay}")
    logger.info(f"调度器: {args.scheduler}")
    
    # 恢复训练
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"恢复训练: {resume_path}")
            start_epoch, ckpt_metrics = load_checkpoint(
                model, resume_path, optimizer, scheduler, device
            )
            best_miou = ckpt_metrics.get('val_mIoU', 0.0)
    
    # CSV 日志和绘图器
    csv_logger = CSVLogger(save_dir / f'metrics_{timestamp}.csv', [
        'epoch', 'lr',
        # 训练指标
        'train_loss', 'train_mIoU', 'train_Dice',
        'train_Precision', 'train_Recall', 'train_F1',
        'train_Pixel_Acc', 'train_mPixel_Acc',
        # 验证指标
        'val_loss', 'val_mIoU', 'val_Dice',
        'val_Precision', 'val_Recall', 'val_F1',
        'val_Pixel_Acc', 'val_mPixel_Acc',
        # 边界指标
        'val_Boundary_IoU', 'val_Boundary_F1',
        'val_Boundary_Precision', 'val_Boundary_Recall',
        # 每类指标（二分类: background/foreground）
        'val_IoU_bg', 'val_IoU_fg',
        'val_Dice_bg', 'val_Dice_fg',
        # 统计
        'val_IoU_mean', 'val_IoU_std',
        'val_Dice_mean', 'val_Dice_std',
        # 时间
        'epoch_time',
    ])
    
    # 模型规范名称（用于图表标题）
    MODEL_DISPLAY_NAMES = {
        'deeplabv3': 'DeepLabV3',
        'unet': 'UNet',
        'unet_resnet': 'UNet-ResNet',
        'pspnet': 'PSPNet',
        'fcn': 'FCN',
        'bisenet': 'BiSeNet',
        'icnet': 'ICNet',
        'enet': 'ENet',
        'cgnet': 'CGNet',
        'danet': 'DANet',
        'ocrnet': 'OCRNet',
        'hrnet': 'HRNet',
        'segnext': 'SegNeXt',
        'segformer': 'SegFormer',
        'mask2former': 'Mask2Former',
    }
    display_name = MODEL_DISPLAY_NAMES.get(args.model, args.model)
    
    plotter = TrainingPlotter(save_dir, model_name=display_name)
    
    # 早停
    early_stopping = EarlyStopping(patience=args.patience, mode='max') if args.patience > 0 else None
    if early_stopping:
        logger.info(f"早停: 开启 (patience={args.patience})")
    
    # ===== 训练循环 =====
    logger.info("-" * 40)
    logger.info("开始训练...")
    logger.info("-" * 40)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 训练
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            num_classes=args.num_classes,
            aux_weight=args.aux_weight
        )

        # 验证
        val_loss, val_metrics, confusion_matrix, per_image_metrics, vis_data = validate(
            model, val_loader, criterion, device,
            num_classes=args.num_classes,
            class_names=class_names
        )
        
        # 更新学习率
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['mIoU'])
            else:
                scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 日志
        logger.info(
            f"Epoch [{epoch + 1:3d}/{args.epochs}] "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} mIoU: {train_metrics['mIoU']:.4f} | "
            f"Val Loss: {val_loss:.4f} mIoU: {val_metrics['mIoU']:.4f} "
            f"Dice: {val_metrics['Dice']:.4f} "
            f"BIoU: {val_metrics['Boundary_IoU']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # CSV 记录
        csv_logger.log({
            'epoch': epoch + 1,
            'lr': current_lr,
            # 训练
            'train_loss': train_loss,
            'train_mIoU': train_metrics['mIoU'],
            'train_Dice': train_metrics['Dice'],
            'train_Precision': train_metrics['Precision'],
            'train_Recall': train_metrics['Recall'],
            'train_F1': train_metrics['F1'],
            'train_Pixel_Acc': train_metrics['Pixel_Acc'],
            'train_mPixel_Acc': train_metrics['mPixel_Acc'],
            # 验证
            'val_loss': val_loss,
            'val_mIoU': val_metrics['mIoU'],
            'val_Dice': val_metrics['Dice'],
            'val_Precision': val_metrics['Precision'],
            'val_Recall': val_metrics['Recall'],
            'val_F1': val_metrics['F1'],
            'val_Pixel_Acc': val_metrics['Pixel_Acc'],
            'val_mPixel_Acc': val_metrics['mPixel_Acc'],
            # 边界
            'val_Boundary_IoU': val_metrics['Boundary_IoU'],
            'val_Boundary_F1': val_metrics['Boundary_F1'],
            'val_Boundary_Precision': val_metrics['Boundary_Precision'],
            'val_Boundary_Recall': val_metrics['Boundary_Recall'],
            # 每类（二分类）
            'val_IoU_bg': val_metrics.get(f'IoU_{class_names[0]}', 0) if class_names else 0,
            'val_IoU_fg': val_metrics.get(f'IoU_{class_names[1]}', 0) if class_names and len(class_names) > 1 else 0,
            'val_Dice_bg': val_metrics.get(f'Dice_{class_names[0]}', 0) if class_names else 0,
            'val_Dice_fg': val_metrics.get(f'Dice_{class_names[1]}', 0) if class_names and len(class_names) > 1 else 0,
            # 统计
            'val_IoU_mean': val_metrics['IoU_mean'],
            'val_IoU_std': val_metrics['IoU_std'],
            'val_Dice_mean': val_metrics['Dice_mean'],
            'val_Dice_std': val_metrics['Dice_std'],
            # 时间
            'epoch_time': round(epoch_time, 1),
        })
        
        # 更新绘图器（传递 epoch_time）
        plotter.update_history(epoch + 1, current_lr, train_loss, train_metrics, val_loss, val_metrics, epoch_time)
        
        # 定期绘图
        if (epoch + 1) % args.plot_freq == 0 or epoch == 0:
            plotter.plot_all()
            plotter.plot_confusion_matrix(confusion_matrix, epoch + 1, class_names)
            plotter.plot_per_image_boxplot(per_image_metrics, epoch + 1)
            
            vis_images, vis_masks, vis_preds = vis_data
            if vis_images is not None:
                plotter.plot_predictions(
                    vis_images[:args.num_vis_samples],
                    vis_masks[:args.num_vis_samples],
                    vis_preds[:args.num_vis_samples],
                    epoch + 1
                )
        
        # 保存最佳模型
        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            
            # 保存完整元信息（含 HF config 用于离线推理）
            extra = dict(
                model_name=args.model,
                backbone=args.backbone,
                num_classes=args.num_classes,
            )
            if hasattr(model, 'get_hf_config_dict'):
                extra['hf_config'] = model.get_hf_config_dict()
            
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                {'val_mIoU': best_miou, **val_metrics},
                save_dir / 'best_model.pth',
                **extra,
            )
            logger.info(f"★ 新的最佳模型! mIoU: {best_miou:.4f}")
            plotter.plot_all()
            plotter.plot_confusion_matrix(confusion_matrix, epoch + 1, class_names)
        
        # 定期检查点
        if (epoch + 1) % args.save_freq == 0:
            extra = dict(
                model_name=args.model,
                backbone=args.backbone,
                num_classes=args.num_classes,
            )
            if hasattr(model, 'get_hf_config_dict'):
                extra['hf_config'] = model.get_hf_config_dict()
            
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                {'val_mIoU': val_metrics['mIoU'], **val_metrics},
                save_dir / f'epoch_{epoch + 1:03d}.pth',
                **extra,
            )
        
        # 早停检查
        if early_stopping and early_stopping(val_metrics['mIoU']):
            logger.info(f"早停触发于 epoch {epoch + 1}")
            break
    
    # ===== 训练完成 =====
    plotter.plot_all()
    
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model.upper()} + {args.backbone}")
    logger.info(f"最佳 mIoU: {best_miou:.4f}")
    logger.info(f"参数量: {model_info['params_M']:.2f}M | FLOPs: {model_info['flops_str']} | 速度: {model_info['fps']:.2f} FPS")
    logger.info(f"结果保存到: {save_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()