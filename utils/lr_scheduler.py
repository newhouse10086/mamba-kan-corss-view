# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts


class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


def init_lr_scheduler(optimizer,
                      lr_scheduler='step',
                      stepsize=[20, 40],
                      gamma=0.1,
                      warmup_factor=100,
                      warmup_iters=20,
                      warmup_method='linear',
                      target_lr=0,
                      power=0.9,
                      T_max=50
                      ):
    if lr_scheduler == 'step':
        return StepLR(optimizer, step_size=stepsize[0], gamma=gamma)
    elif lr_scheduler == 'multistep':
        return MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)
    elif lr_scheduler == 'exponential':
        return ExponentialLR(optimizer, gamma=gamma)
    elif lr_scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=target_lr)
    elif lr_scheduler == 'cosine_warm':
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_max, T_mult=1, eta_min=target_lr)
    elif lr_scheduler == 'warmup_step':
        return WarmupMultiStepLR(optimizer, milestones=stepsize, gamma=gamma, warmup_factor=warmup_factor,
                                 warmup_iters=warmup_iters, warmup_method=warmup_method)
    elif lr_scheduler == 'warmup_cosine':
        return WarmupCosineAnnealingLR(optimizer, T_max=T_max, eta_min=target_lr, warmup_factor=warmup_factor,
                                       warmup_iters=warmup_iters, warmup_method=warmup_method)
    else:
        raise ValueError('Unsupported LR scheduler: {}'.format(lr_scheduler))