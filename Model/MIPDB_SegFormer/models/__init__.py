#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Registry
"""

import logging
import torch.nn as nn
from typing import Type, Dict, Any, List, cast

# Model registry
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def get_logger() -> logging.Logger:
    """Get logger for models module."""
    return logging.getLogger(__name__)


def register_model(name: str):
    """
    装饰器：注册模型到全局注册表.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def register_model_class(name: str, cls: Type[nn.Module]) -> None:
    """
    手动注册模型类.
    """
    MODEL_REGISTRY[name.lower()] = cls


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """
    根据名称获取模型实例.
    """
    logger = get_logger()
    name = name.lower()
    
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"模型 '{name}' 未找到。可用模型: {available}")
    
    logger.info(f"构建模型: {name}")
    model_cls: Type[nn.Module] = MODEL_REGISTRY[name]
    model = model_cls(**kwargs)
    
    return model


def list_models() -> List[str]:
    """列出所有已注册的模型名称."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_config(name: str) -> Dict[str, Any]:
    """获取模型的默认配置."""
    name = name.lower()

    if name not in MODEL_REGISTRY:
        raise ValueError(f"模型 '{name}' 未找到")
    
    model_cls = cast(Any, MODEL_REGISTRY[name])
    
    if hasattr(model_cls, 'get_default_config'):
        config = model_cls.get_default_config()
        return cast(Dict[str, Any], config)
    return {}


def get_model_info(name: str) -> Dict[str, Any]:
    """获取模型的详细信息."""
    name = name.lower()
    
    if name not in MODEL_REGISTRY:
        raise ValueError(f"模型 '{name}' 未找到")
    
    model_cls = cast(Any, MODEL_REGISTRY[name])
    
    info = {
        'name': name,
        'class': model_cls.__name__,
        'module': model_cls.__module__,
        'default_config': get_model_config(name),
    }
    
    if hasattr(model_cls, 'SUPPORTED_BACKBONES'):
        info['supported_backbones'] = model_cls.SUPPORTED_BACKBONES
    
    if hasattr(model_cls, '__doc__') and model_cls.__doc__:
        info['description'] = model_cls.__doc__.strip().split('\n')[0]
    
    return info


from .deeplabv3 import DeepLabV3
from .unet import UNet, UNetResNet
from .pspnet import PSPNet
from .fcn import FCN
from .bisenet import BiSeNet

from .icnet import ICNet
from .enet import ENet
from .cgnet import CGNet

from .danet import DANet
from .ocrnet import OCRNet

from .hrnet import HRNet

from .segnext import SegNeXt

# Transformer 模型
try:
    from .segformer import SegFormer
except (ImportError, ModuleNotFoundError) as e:
    get_logger().debug(f"SegFormer 未加载: {e}")

try:
    from .mask2former import Mask2Former
except (ImportError, ModuleNotFoundError) as e:
    get_logger().debug(f"Mask2Former 未加载: {e}")


__all__ = [
    'MODEL_REGISTRY',
    'register_model',
    'register_model_class',
    'get_model',
    'list_models',
    'get_model_config',
    'get_model_info',
]