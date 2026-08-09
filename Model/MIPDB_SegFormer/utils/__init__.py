#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities module"""

# ============================================================
# Metrics
# ============================================================
from .metrics import MetricsCalculator

# ============================================================
# Plotting
# ============================================================
from .plotting import (
    TrainingPlotter, 
    plot_model_comparison,
    ColorScheme,
)

# ============================================================
# Transforms
# ============================================================
from .transforms import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
    get_custom_transform,
    get_tta_transforms,
    list_available_strengths,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# ============================================================
# Misc Utilities
# ============================================================
from .misc import (
    # Logging
    get_logger,
    setup_logging,
    # Random
    set_seed,
    # CSV
    CSVLogger,
    # Early stopping
    EarlyStopping,
    # Model analysis
    count_parameters,
    compute_flops,
    measure_inference_speed,
    get_model_info,
    save_model_info,
    # Checkpointing
    save_checkpoint,
    load_checkpoint,
    # Utilities
    AverageMeter,
    Timer,
)


# ============================================================
# Public API
# ============================================================
__all__ = [
    # Metrics
    'MetricsCalculator',
    
    # Plotting
    'TrainingPlotter',
    'plot_model_comparison',
    'ColorScheme',
    
    # Transforms
    'get_train_transform',
    'get_val_transform',
    'get_test_transform',
    'get_custom_transform',
    'get_tta_transforms',
    'list_available_strengths',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    
    # Logging
    'get_logger',
    'setup_logging',
    
    # Random
    'set_seed',
    
    # CSV
    'CSVLogger',
    
    # Early stopping
    'EarlyStopping',
    
    # Model analysis
    'count_parameters',
    'compute_flops',
    'measure_inference_speed',
    'get_model_info',
    'save_model_info',
    
    # Checkpointing
    'save_checkpoint',
    'load_checkpoint',
    
    # Utilities
    'AverageMeter',
    'Timer',
]