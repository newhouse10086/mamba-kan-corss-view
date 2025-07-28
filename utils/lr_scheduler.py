import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List

class WarmupMultiStepLR(_LRScheduler):
    """
    带预热的多步学习率调度器
    在前几个epoch使用线性预热，然后在指定的里程碑进行学习率衰减
    """
    
    def __init__(self, 
                 optimizer,
                 milestones: List[int],
                 gamma: float = 0.1,
                 warmup_epochs: int = 0,
                 warmup_factor: float = 0.1,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: 优化器
            milestones: 学习率衰减的里程碑epoch列表
            gamma: 学习率衰减因子
            warmup_epochs: 预热epoch数
            warmup_factor: 预热起始因子
            last_epoch: 上一个epoch索引
        """
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            alpha = self.last_epoch / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 正常训练阶段：多步衰减
            decay_factor = 1.0
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    decay_factor *= self.gamma
            return [base_lr * decay_factor for base_lr in self.base_lrs]

class WarmupCosineAnnealingLR(_LRScheduler):
    """
    带预热的余弦退火学习率调度器
    """
    
    def __init__(self,
                 optimizer,
                 T_max: int,
                 eta_min: float = 0,
                 warmup_epochs: int = 0,
                 warmup_factor: float = 0.1,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: 优化器
            T_max: 余弦退火的最大epoch数
            eta_min: 最小学习率
            warmup_epochs: 预热epoch数
            warmup_factor: 预热起始因子
            last_epoch: 上一个epoch索引
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段
            alpha = self.last_epoch / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            epoch = self.last_epoch - self.warmup_epochs
            T_max = self.T_max - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * epoch / T_max)) / 2
                   for base_lr in self.base_lrs]

class WarmupExponentialLR(_LRScheduler):
    """
    带预热的指数衰减学习率调度器
    """
    
    def __init__(self,
                 optimizer,
                 gamma: float,
                 warmup_epochs: int = 0,
                 warmup_factor: float = 0.1,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: 优化器
            gamma: 指数衰减因子
            warmup_epochs: 预热epoch数
            warmup_factor: 预热起始因子
            last_epoch: 上一个epoch索引
        """
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段
            alpha = self.last_epoch / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 指数衰减阶段
            epoch = self.last_epoch - self.warmup_epochs
            return [base_lr * (self.gamma ** epoch) for base_lr in self.base_lrs] 