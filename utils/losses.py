import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Tuple, Optional

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    
    Equation: y = (1 - epsilon) * y + epsilon / K.
    
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class ContrastiveLoss(nn.Module):
    """对比损失：兼容 margin 或 temperature 参数"""
    def __init__(self, margin: float = 1.0, temperature: float = None, **kwargs):
        super(ContrastiveLoss, self).__init__()
        # 如果传入 temperature 则使用 temperature 作为缩放因子
        self.margin = temperature if temperature is not None else margin

    def forward(self, *inputs):
        """支持两种调用方式：
        1. (feat1, feat2, label_binary)
        2. (features, labels) 使用 InfoNCE 监督对比
        """
        if len(inputs) == 3:
            feat1, feat2, label = inputs
            euclidean_distance = F.pairwise_distance(feat1, feat2, p=2)
            loss_contrastive = torch.mean(
                label * torch.pow(euclidean_distance, 2) +
                (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
            )
            return loss_contrastive
        elif len(inputs) == 2:
            features, labels = inputs
            return self._supervised_contrastive_loss(features, labels)
        else:
            raise ValueError("ContrastiveLoss forward expects 2 or 3 inputs")

    def _supervised_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # InfoNCE 风格的监督对比学习 (简化版)
        device = features.device
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.t()) / self.margin  # 使用 margin 作为温度
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = -mean_log_prob_pos.mean()
        return loss

class CenterLoss(nn.Module):
    """
    中心损失函数，用于增强类内紧密性
    """
    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征 [N, D]
            labels: 标签 [N]
        """
        batch_size = features.size(0)
        
        # 计算特征到对应类别中心的距离
        centers_batch = self.centers[labels]  # [N, D]
        center_loss = F.mse_loss(features, centers_batch)
        
        return center_loss

class FocalLoss(nn.Module):
    """
    Focal Loss，用于处理类别不平衡问题
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测logits [N, C]
            targets: 真实标签 [N]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SupConLoss(nn.Module):
    """
    监督对比学习损失函数
    """
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征 [N, D]
            labels: 标签 [N]
        """
        device = features.device
        batch_size = features.shape[0]
        
        if labels is not None and labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 创建mask
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        
        # 排除自己与自己的比较
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算log概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算平均log似然
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss

class CircleLoss(nn.Module):
    """
    Circle Loss，统一的深度度量学习损失
    """
    def __init__(self, m: float = 0.25, gamma: float = 256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sp: 正样本相似度 [N, P]
            sn: 负样本相似度 [N, N]
        """
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)
        
        delta_p = 1 - self.m
        delta_n = self.m
        
        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
        
        return loss.mean()

class MultiSimilarityLoss(nn.Module):
    """
    多相似度损失函数
    """
    def __init__(self, thresh: float = 0.5, margin: float = 0.1, scale_pos: float = 2.0, 
                 scale_neg: float = 50.0):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = thresh
        self.margin = margin
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: 特征 [N, D]
            labels: 标签 [N]
        """
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        
        epsilon = 1e-5
        loss = list()
        
        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]
            
            neg_pair = neg_pair_[neg_pair_ > min(neg_pair_) + epsilon]
            pos_pair = pos_pair_[pos_pair_ < max(pos_pair_) - epsilon]
            
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
                
            # 计算损失
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)
        
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)
            
        loss = sum(loss) / batch_size
        return loss

class CombinedLoss(nn.Module):
    """
    组合损失函数，结合多种损失
    """
    def __init__(self, 
                 num_classes: int,
                 feat_dim: int,
                 loss_types: list = ['crossentropy', 'triplet'],
                 loss_weights: list = [1.0, 1.0],
                 **kwargs):
        super(CombinedLoss, self).__init__()
        
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.losses = nn.ModuleDict()
        
        # 初始化各种损失函数
        for loss_type in loss_types:
            if loss_type == 'crossentropy':
                self.losses[loss_type] = nn.CrossEntropyLoss()
            elif loss_type == 'crossentropy_smooth':
                self.losses[loss_type] = CrossEntropyLabelSmooth(num_classes, kwargs.get('epsilon', 0.1))
            elif loss_type == 'triplet':
                self.losses[loss_type] = TripletLoss(kwargs.get('margin', 0.3))
            elif loss_type == 'center':
                self.losses[loss_type] = CenterLoss(num_classes, feat_dim, kwargs.get('alpha', 0.5))
            elif loss_type == 'focal':
                self.losses[loss_type] = FocalLoss(kwargs.get('alpha', 1.0), kwargs.get('gamma', 2.0))
            elif loss_type == 'supcon':
                self.losses[loss_type] = SupConLoss(kwargs.get('temperature', 0.07))
            elif loss_type == 'circle':
                self.losses[loss_type] = CircleLoss(kwargs.get('m', 0.25), kwargs.get('gamma', 256))
            elif loss_type == 'ms':
                self.losses[loss_type] = MultiSimilarityLoss()
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, outputs: dict, targets: dict) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: 模型输出字典
            targets: 目标字典
        Returns:
            total_loss: 总损失
            loss_dict: 各损失的字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        for i, loss_type in enumerate(self.loss_types):
            weight = self.loss_weights[i]
            
            if loss_type in ['crossentropy', 'crossentropy_smooth', 'focal']:
                if 'logits' in outputs and 'labels' in targets:
                    loss = self.losses[loss_type](outputs['logits'], targets['labels'])
                    total_loss += weight * loss
                    loss_dict[loss_type] = loss.item()
            
            elif loss_type == 'triplet':
                if 'anchor' in outputs and 'positive' in outputs and 'negative' in outputs:
                    loss = self.losses[loss_type](outputs['anchor'], outputs['positive'], outputs['negative'])
                    total_loss += weight * loss
                    loss_dict[loss_type] = loss.item()
            
            elif loss_type == 'center':
                if 'features' in outputs and 'labels' in targets:
                    loss = self.losses[loss_type](outputs['features'], targets['labels'])
                    total_loss += weight * loss
                    loss_dict[loss_type] = loss.item()
            
            elif loss_type in ['supcon', 'ms']:
                if 'features' in outputs and 'labels' in targets:
                    loss = self.losses[loss_type](outputs['features'], targets['labels'])
                    total_loss += weight * loss
                    loss_dict[loss_type] = loss.item()
            
            elif loss_type == 'circle':
                if 'sp' in outputs and 'sn' in outputs:
                    loss = self.losses[loss_type](outputs['sp'], outputs['sn'])
                    total_loss += weight * loss
                    loss_dict[loss_type] = loss.item()
        
        return total_loss, loss_dict

def create_loss_fn(config: dict) -> nn.Module:
    """
    根据配置创建损失函数
    """
    loss_type = config.get('type', 'crossentropy')
    
    if loss_type == 'combined':
        return CombinedLoss(**config)
    elif loss_type == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'crossentropy_smooth':
        return CrossEntropyLabelSmooth(config['num_classes'], config.get('epsilon', 0.1))
    elif loss_type == 'triplet':
        return TripletLoss(config.get('margin', 0.3))
    elif loss_type == 'center':
        return CenterLoss(config['num_classes'], config['feat_dim'], config.get('alpha', 0.5))
    elif loss_type == 'focal':
        return FocalLoss(config.get('alpha', 1.0), config.get('gamma', 2.0))
    elif loss_type == 'supcon':
        return SupConLoss(config.get('temperature', 0.07))
    elif loss_type == 'circle':
        return CircleLoss(config.get('m', 0.25), config.get('gamma', 256))
    elif loss_type == 'ms':
        return MultiSimilarityLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

# 示例配置
LOSS_CONFIGS = {
    'crossentropy': {
        'type': 'crossentropy'
    },
    'combined_basic': {
        'type': 'combined',
        'num_classes': 701,
        'feat_dim': 768,
        'loss_types': ['crossentropy', 'triplet'],
        'loss_weights': [1.0, 0.5],
        'margin': 0.3
    },
    'combined_advanced': {
        'type': 'combined',
        'num_classes': 701,
        'feat_dim': 768,
        'loss_types': ['crossentropy_smooth', 'center', 'supcon'],
        'loss_weights': [1.0, 0.1, 0.3],
        'epsilon': 0.1,
        'alpha': 0.5,
        'temperature': 0.07
    }
}

if __name__ == "__main__":
    # 测试损失函数
    batch_size = 8
    num_classes = 701
    feat_dim = 768
    
    # 创建假数据
    logits = torch.randn(batch_size, num_classes)
    features = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 测试各种损失函数
    print("Testing loss functions...")
    
    # 交叉熵损失
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(logits, labels)
    print(f"CrossEntropy Loss: {loss_ce.item():.4f}")
    
    # 标签平滑交叉熵损失
    ces_loss = CrossEntropyLabelSmooth(num_classes)
    loss_ces = ces_loss(logits, labels)
    print(f"CrossEntropy Smooth Loss: {loss_ces.item():.4f}")
    
    # 三元组损失
    anchor = features[:4]
    positive = features[4:8] if batch_size >= 8 else features[:4]
    negative = torch.randn_like(anchor)
    triplet_loss = TripletLoss()
    loss_triplet = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss_triplet.item():.4f}")
    
    # 中心损失
    center_loss = CenterLoss(num_classes, feat_dim)
    loss_center = center_loss(features, labels)
    print(f"Center Loss: {loss_center.item():.4f}")
    
    # 组合损失
    combined_loss = CombinedLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        loss_types=['crossentropy', 'center'],
        loss_weights=[1.0, 0.1]
    )
    
    outputs = {'logits': logits, 'features': features}
    targets = {'labels': labels}
    total_loss, loss_dict = combined_loss(outputs, targets)
    print(f"Combined Loss: {total_loss.item():.4f}")
    print(f"Loss Dict: {loss_dict}")
    
    print("Loss function tests completed!") 