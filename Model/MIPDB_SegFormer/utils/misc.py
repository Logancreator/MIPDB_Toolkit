#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Miscellaneous utilities"""

from __future__ import annotations

import os
import csv
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 全局日志
# ============================================================

_LOGGER_NAME = 'semantic segmentation'
_logger: Optional[logging.Logger] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
   """获取 root logger"""
   return logging.getLogger()

def setup_logging(save_dir: Union[str, Path], name: str = 'train', level: int = logging.INFO, console_level: Optional[int] = None) -> Tuple[logging.Logger, str]:
    """
    配置日志系统（文件 + 控制台输出）.
    
    Args:
        save_dir: 日志文件保存目录
        name: 日志文件前缀
        level: 文件日志级别
        console_level: 控制台日志级别（默认与 level 相同）
    
    Returns:
        (logger, timestamp)
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir / f'{name}_{timestamp}.log'
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 配置 root logger（关键！这样所有模块的日志都能被捕获）
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # 清除已有 handlers

    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    root_logger.info(f"日志文件: {log_file}")
    
    return root_logger, timestamp


# ============================================================
# 随机种子
# ============================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    设置随机种子以保证可复现性.
    
    Args:
        seed: 随机种子
        deterministic: 是否启用确定性模式（可能降低性能）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    logger = get_logger()
    logger.debug(f"随机种子设置为 {seed}，确定性模式: {deterministic}")


# ============================================================
# CSV 日志
# ============================================================

class CSVLogger:
    """
    将指标记录到 CSV 文件.
    
    Args:
        filepath: CSV 文件路径
        fieldnames: 列名列表
    
    Example:
        logger = CSVLogger('metrics.csv', ['epoch', 'loss', 'mIoU'])
        logger.log({'epoch': 1, 'loss': 0.5, 'mIoU': 0.8})
    """
    
    def __init__(self, filepath: Union[str, Path], fieldnames: list[str]) -> None:
        self.filepath = Path(filepath)
        self.fieldnames = fieldnames
        
        # 创建文件并写入表头
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def log(self, row_Dict: Dict[str, Any]) -> None:
        """记录一行数据"""
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_Dict)
    
    def __repr__(self) -> str:
        return f"CSVLogger(filepath='{self.filepath}', columns={len(self.fieldnames)})"


# ============================================================
# 早停
# ============================================================

class EarlyStopping:
    """
    早停处理器.
    
    Args:
        patience: 容忍的无改善 epoch 数
        mode: 'max' 或 'min'（指标优化方向）
        min_delta: 最小改善量
    
    Example:
        early_stop = EarlyStopping(patience=10, mode='max')
        for epoch in range(100):
            val_miou = validate()
            if early_stop(val_miou):
                print(f"Early stopping at epoch {epoch}")
                break
    """
    
    def __init__(self, patience: int = 10, mode: str = 'max', min_delta: float = 0.0) -> None:
        if mode not in ('max', 'min'):
            raise ValueError(f"mode 必须是 'max' 或 'min'，而不是 '{mode}'")
        
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        
        self.logger = get_logger()
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停.
        
        Args:
            score: 当前指标值
        
        Returns:
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            self.logger.debug(f"EarlyStopping: 初始分数 {score:.4f}")
            return False
        
        # 检查是否有改善
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.logger.debug(f"EarlyStopping: 分数改善 {self.best_score:.4f} -> {score:.4f}")
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            self.logger.debug(f"EarlyStopping: 无改善 ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(f"EarlyStopping: 触发早停，最佳分数 {self.best_score:.4f}")
        
        return self.early_stop
    
    def reset(self) -> None:
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, mode='{self.mode}', "
            f"best_score={self.best_score}, counter={self.counter})"
        )


# ============================================================
# 模型分析
# ============================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数量.
    
    Args:
        model: PyTorch 模型
    
    Returns:
        (总参数量, 可训练参数量)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def compute_flops(model: nn.Module, input_size: int, device: torch.device,in_channels: int = 3) -> Tuple[float, str]:
    """
    计算模型 FLOPs.
    
    Args:
        model: PyTorch 模型
        input_size: 输入尺寸
        device: 计算设备
        in_channels: 输入通道数
    
    Returns:
        (FLOPs 数值, 格式化字符串)
    """
    logger = get_logger()
    
    try:
        from thop import profile, clever_format
        
        dummy_input = torch.randn(1, in_channels, input_size, input_size).to(device)
        model.eval()
        
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        flops_str, _ = clever_format([flops, params], "%.2f")
        logger.debug(f"FLOPs (thop): {flops_str}")
        
        return flops, flops_str
        
    except ImportError:
        logger.warning("thop 未安装，使用估算方法计算 FLOPs")
        return _estimate_flops(model, input_size, device, in_channels)


def _estimate_flops(model: nn.Module,input_size: int,device: torch.device,in_channels: int = 3) -> Tuple[float, str]:
    """估算 FLOPs（当 thop 不可用时的回退方法）"""
    dummy_input = torch.randn(1, in_channels, input_size, input_size).to(device)
    total_ops = 0
    
    def hook_fn(module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        nonlocal total_ops
        if isinstance(module, nn.Conv2d):
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            total_ops += kernel_ops * module.in_channels * module.out_channels * out_h * out_w
        elif isinstance(module, nn.Linear):
            total_ops += module.in_features * module.out_features
    
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    flops = total_ops * 2  # 乘加各算一次
    flops_str = _format_number(flops, suffix='FLOPs')
    
    return flops, flops_str


def _format_number(num: float, suffix: str = '') -> str:
    """格式化大数字"""
    if num >= 1e12:
        return f"{num / 1e12:.2f} T{suffix}"
    elif num >= 1e9:
        return f"{num / 1e9:.2f} G{suffix}"
    elif num >= 1e6:
        return f"{num / 1e6:.2f} M{suffix}"
    elif num >= 1e3:
        return f"{num / 1e3:.2f} K{suffix}"
    else:
        return f"{num:.2f} {suffix}"


def measure_inference_speed(model: nn.Module, input_size: int, device: torch.device, num_iterations: int = 100, warmup: int = 10, in_channels: int = 3) -> Tuple[float, float]:
    """
    测量推理速度.
    
    Args:
        model: PyTorch 模型
        input_size: 输入尺寸
        device: 计算设备
        num_iterations: 测试迭代次数
        warmup: 预热迭代次数
        in_channels: 输入通道数
    
    Returns:
        (FPS, 延迟毫秒)
    """
    model.eval()
    dummy_input = torch.randn(1, in_channels, input_size, input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # 同步 CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 正式测试
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time
    
    fps = num_iterations / elapsed_time
    latency_ms = (elapsed_time / num_iterations) * 1000
    
    return fps, latency_ms


def get_model_info(model: nn.Module, input_size: int, device: torch.device, in_channels: int = 3) -> Dict[str, Any]:
    """
    获取模型综合信息.
    
    Args:
        model: PyTorch 模型
        input_size: 输入尺寸
        device: 计算设备
        in_channels: 输入通道数
    
    Returns:
        包含参数量、FLOPs、速度等信息的字典
    """
    total_params, trainable_params = count_parameters(model)
    flops, flops_str = compute_flops(model, input_size, device, in_channels)
    fps, latency_ms = measure_inference_speed(model, input_size, device, in_channels=in_channels)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
        'flops': flops,
        'flops_str': flops_str,
        'fps': fps,
        'latency_ms': latency_ms,
    }
    
    logger = get_logger()
    logger.info(
        f"模型信息 | 参数: {info['params_M']:.2f}M | "
        f"FLOPs: {flops_str} | "
        f"速度: {fps:.2f} FPS ({latency_ms:.2f} ms)"
    )
    
    return info


def save_model_info(info: Dict[str, Any], model_name: str, save_path: Union[str, Path]) -> None:
    """
    保存模型信息到文本文件.
    
    Args:
        info: 模型信息字典
        model_name: 模型名称
        save_path: 保存路径
    """
    save_path = Path(save_path)
    
    content = f"""Model: {model_name}
{'=' * 40}
Parameters:
  Total:      {info['total_params']:,} ({info['params_M']:.2f}M)
  Trainable:  {info['trainable_params']:,} ({info['trainable_params_M']:.2f}M)
Computation:
  FLOPs:      {info['flops_str']}
Speed:
  FPS:        {info['fps']:.2f}
  Latency:    {info['latency_ms']:.2f} ms/image
{'=' * 40}
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger = get_logger()
    logger.debug(f"模型信息已保存到: {save_path}")


# ============================================================
# 检查点
# ============================================================

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Union[Any, None], epoch: int, metrics: Dict[str, Any], save_path: Union[str, Path],**extra_info: Any) -> None:
    """
    保存训练检查点.
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        epoch: 当前 epoch
        metrics: 指标字典
        save_path: 保存路径
        **extra_info: 额外信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        **extra_info,
    }
    
    save_path = Path(save_path)
    torch.save(checkpoint, save_path)
    
    logger = get_logger()
    logger.debug(f"检查点已保存: {save_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, optimizer: Optional[torch.optim.Optimizer]= None, scheduler: Union[Any, None] = None, device: str | torch.device = 'cuda', strict: bool = True) -> Tuple[int, Dict[str, Any]]:
    """
    加载训练检查点.
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        optimizer: 优化器（可选）
        scheduler: 调度器（可选）
        device: 加载设备
        strict: 是否严格匹配模型参数
    
    Returns:
        (epoch, metrics)
    """
    logger = get_logger()
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
    
    logger.info(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 加载模型
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # 加载优化器
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    logger.info(f"从 epoch {epoch} 恢复，指标: {metrics}")
    
    return epoch, metrics


# ============================================================
# 其他工具
# ============================================================

class AverageMeter:
    """
    计算并存储平均值和当前值.
    
    Example:
        losses = AverageMeter('Loss')
        for data in dataloader:
            loss = compute_loss(data)
            losses.update(loss.item(), batch_size)
        print(losses)
    """
    
    def __init__(self, name: str = 'Meter') -> None:
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """
    简单计时器.
    
    Example:
        timer = Timer()
        timer.start()
        # ... 一些操作
        elapsed = timer.stop()
        print(f"耗时: {elapsed:.2f}s")
        
        # 或者使用上下文管理器
        with Timer() as t:
            # ... 一些操作
        print(f"耗时: {t.elapsed:.2f}s")
    """
    
    def __init__(self) -> None:
        self.start_time: float | None = None
        self.elapsed: float = 0.0
    
    def start(self) -> Timer:
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer 未启动")
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def __enter__(self) -> Timer:
        self.start()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.stop()
    
    def __repr__(self) -> str:
        return f"Timer(elapsed={self.elapsed:.4f}s)"