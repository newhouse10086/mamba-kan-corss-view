import torch
import numpy as np

def evaluate(gallery_features, query_features, gallery_labels=None, query_labels=None, k=1):
    """
    Evaluate retrieval performance using Recall@k and mAP
    """
    # If labels are not provided, generate dummy labels for testing
    if gallery_labels is None:
        gallery_labels = np.arange(gallery_features.size(0))
    if query_labels is None:
        query_labels = np.arange(query_features.size(0))
    
    # Convert to numpy
    gallery_features = gallery_features.numpy() if not isinstance(gallery_features, np.ndarray) else gallery_features
    query_features = query_features.numpy() if not isinstance(query_features, np.ndarray) else query_features
    
    # Compute similarity matrix
    similarity = np.dot(query_features, gallery_features.T)
    
    # Compute metrics
    top1_correct = 0
    top5_correct = 0
    total = query_features.shape[0]
    
    # For mAP calculation
    aps = []
    
    for i in range(total):
        # Get similarities for query i
        scores = similarity[i]
        # Sort in descending order
        indices = np.argsort(-scores)
        
        # Get the true label
        true_label = query_labels[i]
        
        # Check top-k matches
        top_k_labels = [gallery_labels[idx] for idx in indices[:k]]
        if true_label in top_k_labels:
            if k >= 1:
                top1_correct += 1
            if k >= 5:
                top5_correct += 1
        
        # Calculate AP for this query
        retrieved_labels = [gallery_labels[idx] for idx in indices]
        ap = compute_ap(retrieved_labels, true_label)
        aps.append(ap)
    
    recall_at_1 = top1_correct / total
    recall_at_5 = top5_correct / total if k >= 5 else 0
    mAP = np.mean(aps)
    
    return {
        'recall@1': recall_at_1,
        'recall@5': recall_at_5,
        'mAP': mAP
    }

def compute_ap(retrieved_labels, true_label, max_rank=None):
    """
    Compute Average Precision for a single query
    """
    if max_rank is None:
        max_rank = len(retrieved_labels)
    
    num_relevant = 0
    sum_precisions = 0.0
    
    for i, label in enumerate(retrieved_labels[:max_rank]):
        if label == true_label:
            num_relevant += 1
            sum_precisions += num_relevant / (i + 1)
    
    if num_relevant == 0:
        return 0.0
    
    return sum_precisions / num_relevant

def compute_distance_matrix(x, y):
    """
    Compute pairwise distance matrix between two sets of features
    """
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist.clamp(min=1e-12).sqrt()

def compute_recall_at_k(dist_matrix, query_labels, gallery_labels, k=1):
    """
    Compute Recall@k given distance matrix
    """
    # Sort indices by distance (ascending)
    sorted_indices = torch.argsort(dist_matrix, dim=1)
    
    correct = 0
    for i in range(dist_matrix.size(0)):
        # Get top-k gallery indices
        top_k_indices = sorted_indices[i, :k]
        # Get corresponding labels
        top_k_labels = gallery_labels[top_k_indices]
        # Check if true label is in top-k
        if query_labels[i] in top_k_labels:
            correct += 1
    
    return correct / dist_matrix.size(0)

def compute_map(dist_matrix, query_labels, gallery_labels):
    """
    Compute mAP given distance matrix
    """
    # Sort indices by distance (ascending)
    sorted_indices = torch.argsort(dist_matrix, dim=1)
    
    aps = []
    for i in range(dist_matrix.size(0)):
        # Get sorted gallery labels for this query
        sorted_labels = gallery_labels[sorted_indices[i]]
        # Compute AP for this query
        ap = compute_ap(sorted_labels.numpy(), query_labels[i].item())
        aps.append(ap)
    
    return np.mean(aps)

def plot_cmc_curve(cmc_scores, max_rank=20):
    """
    Plot CMC curve
    """
    import matplotlib.pyplot as plt
    
    if len(cmc_scores) < max_rank:
        max_rank = len(cmc_scores)
    
    ranks = range(1, max_rank + 1)
    plt.plot(ranks, cmc_scores[:max_rank], marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Recognition Rate')
    plt.title('CMC Curve')
    plt.grid(True)
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

def compute_distance_matrix(query_features: torch.Tensor, 
                          gallery_features: torch.Tensor,
                          metric: str = 'euclidean') -> torch.Tensor:
    """
    计算查询特征和画廊特征之间的距离矩阵
    
    Args:
        query_features: 查询特征 [N_q, D]
        gallery_features: 画廊特征 [N_g, D]
        metric: 距离度量方式 ['euclidean', 'cosine']
    
    Returns:
        distance_matrix: 距离矩阵 [N_q, N_g]
    """
    if metric == 'euclidean':
        # L2归一化
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
        
        # 计算欧氏距离
        dist_mat = torch.cdist(query_features, gallery_features, p=2)
        
    elif metric == 'cosine':
        # L2归一化
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
        
        # 计算余弦相似度，然后转换为距离
        similarity = torch.mm(query_features, gallery_features.t())
        dist_mat = 1 - similarity
        
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
    
    return dist_mat

# -------------- CMC 基于距离矩阵的实现 --------------

def _compute_cmc_from_dist(dist_mat: torch.Tensor,
                           query_labels: torch.Tensor,
                           gallery_labels: torch.Tensor,
                           max_rank: int = 50) -> np.ndarray:
    """基于距离矩阵的CMC实现（原始实现）"""
    num_q, num_g = dist_mat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(f"Warning: Gallery size ({num_g}) is smaller than max_rank ({max_rank})")

    # 排序获得索引
    indices = torch.argsort(dist_mat, dim=1)
    matches = (gallery_labels[indices] == query_labels.unsqueeze(1)).cpu().numpy()

    cmc = np.zeros(max_rank)
    for i in range(num_q):
        first_match = np.where(matches[i])[0]
        if len(first_match) > 0:
            first_match_idx = first_match[0]
            if first_match_idx < max_rank:
                cmc[first_match_idx:] += 1
    cmc = cmc / num_q
    return cmc

# -------------- 通用包装器 --------------

def compute_cmc(query_features_or_dist: torch.Tensor,
                gallery_features: Optional[torch.Tensor] = None,
                query_labels: Optional[torch.Tensor] = None,
                gallery_labels: Optional[torch.Tensor] = None,
                max_rank: int = 50,
                metric: str = 'euclidean') -> np.ndarray:
    """通用 CMC 计算函数，兼容特征或距离矩阵两种输入"""
    if gallery_features is None and query_labels is not None and gallery_labels is not None:
        # 旧接口：直接传入距离矩阵
        return _compute_cmc_from_dist(query_features_or_dist, query_labels, gallery_labels, max_rank)

    if gallery_features is not None and query_labels is not None and gallery_labels is not None:
        # 新接口：传入特征，需先计算距离矩阵
        dist_mat = compute_distance_matrix(query_features_or_dist, gallery_features, metric)
        return _compute_cmc_from_dist(dist_mat, query_labels, gallery_labels, max_rank)

    raise ValueError("Invalid arguments for compute_cmc: please provide either (dist_mat, query_labels, gallery_labels) or (query_features, gallery_features, query_labels, gallery_labels)")

def compute_ap(dist_mat: torch.Tensor, 
               query_labels: torch.Tensor, 
               gallery_labels: torch.Tensor) -> float:
    """
    计算平均精度 (Average Precision)
    
    Args:
        dist_mat: 距离矩阵 [N_q, N_g]
        query_labels: 查询标签 [N_q]
        gallery_labels: 画廊标签 [N_g]
    
    Returns:
        mAP: 平均精度
    """
    num_q = dist_mat.shape[0]
    ap_scores = []
    
    for i in range(num_q):
        # 获取当前查询的距离和标签
        distances = dist_mat[i].cpu().numpy()
        gt_labels = (gallery_labels == query_labels[i]).cpu().numpy().astype(int)
        
        # 按距离排序
        sorted_indices = np.argsort(distances)
        sorted_labels = gt_labels[sorted_indices]
        
        # 计算AP
        if np.sum(sorted_labels) > 0:
            ap = average_precision_score(sorted_labels, -distances[sorted_indices])
            ap_scores.append(ap)
    
    return np.mean(ap_scores) if ap_scores else 0.0

def compute_metrics(query_features: torch.Tensor,
                   gallery_features: torch.Tensor,
                   query_labels: torch.Tensor,
                   gallery_labels: torch.Tensor,
                   distance_metric: str = 'euclidean',
                   max_rank: int = 50) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        query_features: 查询特征 [N_q, D]
        gallery_features: 画廊特征 [N_g, D]
        query_labels: 查询标签 [N_q]
        gallery_labels: 画廊标签 [N_g]
        distance_metric: 距离度量方式
        max_rank: 最大rank
    
    Returns:
        metrics: 评估指标字典
    """
    # 计算距离矩阵
    dist_mat = compute_distance_matrix(query_features, gallery_features, distance_metric)
    
    # 计算CMC
    cmc = compute_cmc(dist_mat, query_labels, gallery_labels, max_rank)
    
    # 计算mAP
    mAP = compute_ap(dist_mat, query_labels, gallery_labels)
    
    # 组织结果
    metrics = {
        'mAP': mAP,
        'Rank-1': cmc[0],
        'Rank-5': cmc[4] if len(cmc) > 4 else cmc[-1],
        'Rank-10': cmc[9] if len(cmc) > 9 else cmc[-1],
        'Rank-20': cmc[19] if len(cmc) > 19 else cmc[-1],
    }
    
    # 添加所有rank的结果
    for i in range(min(max_rank, len(cmc))):
        metrics[f'Rank-{i+1}'] = cmc[i]
    
    return metrics

class AverageMeter:
    """
    用于计算和存储平均值和当前值的工具类
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricsTracker:
    """
    指标跟踪器，用于训练过程中跟踪各种指标
    """
    def __init__(self):
        self.metrics = {}
        
    def update(self, metrics_dict: Dict[str, float]):
        """更新指标"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter()
            self.metrics[key].update(value)
    
    def get_avg_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        return {key: meter.avg for key, meter in self.metrics.items()}
    
    def reset(self):
        """重置所有指标"""
        for meter in self.metrics.values():
            meter.reset()
    
    def __str__(self) -> str:
        """格式化输出"""
        metrics_str = []
        for key, meter in self.metrics.items():
            metrics_str.append(f'{key}: {meter.avg:.4f}')
        return ' | '.join(metrics_str)

def evaluate_cross_view_matching(model: torch.nn.Module,
                                query_loader: torch.utils.data.DataLoader,
                                gallery_loader: torch.utils.data.DataLoader, 
                                device: torch.device,
                                distance_metric: str = 'euclidean') -> Dict[str, float]:
    """
    评估跨视角匹配性能
    
    Args:
        model: 训练好的模型
        query_loader: 查询数据加载器
        gallery_loader: 画廊数据加载器
        device: 设备
        distance_metric: 距离度量方式
    
    Returns:
        metrics: 评估指标
    """
    model.eval()
    
    # 提取查询特征
    query_features = []
    query_labels = []
    
    with torch.no_grad():
        for batch in query_loader:
            images = batch['image'].to(device)
            labels = batch['class_id'].to(device)
            
            # 提取特征
            if hasattr(model, 'extract_features'):
                features = model.extract_features(images)
            else:
                # 假设模型直接返回特征
                features = model(images)
                if isinstance(features, tuple):
                    features = features[-1]  # 取最后一个输出作为特征
            
            query_features.append(features.cpu())
            query_labels.append(labels.cpu())
    
    query_features = torch.cat(query_features, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    
    # 提取画廊特征
    gallery_features = []
    gallery_labels = []
    
    with torch.no_grad():
        for batch in gallery_loader:
            images = batch['image'].to(device)
            labels = batch['class_id'].to(device)
            
            # 提取特征
            if hasattr(model, 'extract_features'):
                features = model.extract_features(images)
            else:
                features = model(images)
                if isinstance(features, tuple):
                    features = features[-1]
            
            gallery_features.append(features.cpu())
            gallery_labels.append(labels.cpu())
    
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_labels = torch.cat(gallery_labels, dim=0)
    
    # 计算指标
    metrics = compute_metrics(
        query_features, gallery_features, 
        query_labels, gallery_labels,
        distance_metric=distance_metric
    )
    
    return metrics

def visualize_retrieval_results(query_images: torch.Tensor,
                               gallery_images: torch.Tensor,
                               dist_mat: torch.Tensor,
                               query_labels: torch.Tensor,
                               gallery_labels: torch.Tensor,
                               query_idx: int = 0,
                               top_k: int = 5,
                               save_path: Optional[str] = None):
    """
    可视化检索结果
    
    Args:
        query_images: 查询图像
        gallery_images: 画廊图像
        dist_mat: 距离矩阵
        query_labels: 查询标签
        gallery_labels: 画廊标签
        query_idx: 要可视化的查询索引
        top_k: 显示top-k结果
        save_path: 保存路径
    """
    # 获取top-k结果
    _, indices = torch.topk(dist_mat[query_idx], top_k, largest=False)
    
    # 创建子图
    fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 3))
    
    # 显示查询图像
    query_img = query_images[query_idx].permute(1, 2, 0)
    query_img = (query_img * 0.229 + 0.485).clamp(0, 1)  # 反归一化
    axes[0].imshow(query_img)
    axes[0].set_title(f'Query\nLabel: {query_labels[query_idx].item()}')
    axes[0].axis('off')
    
    # 显示检索结果
    for i, idx in enumerate(indices):
        gallery_img = gallery_images[idx].permute(1, 2, 0)
        gallery_img = (gallery_img * 0.229 + 0.485).clamp(0, 1)  # 反归一化
        
        # 判断是否匹配
        is_match = gallery_labels[idx] == query_labels[query_idx]
        color = 'green' if is_match else 'red'
        
        axes[i + 1].imshow(gallery_img)
        axes[i + 1].set_title(f'Rank {i+1}\nLabel: {gallery_labels[idx].item()}\nDist: {dist_mat[query_idx, idx]:.3f}')
        axes[i + 1].axis('off')
        
        # 添加边框
        for spine in axes[i + 1].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

def plot_cmc_curve(cmc_scores: Dict[str, np.ndarray], 
                   save_path: Optional[str] = None):
    """
    绘制CMC曲线
    
    Args:
        cmc_scores: CMC分数字典，键为方法名，值为CMC数组
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    for method_name, cmc in cmc_scores.items():
        ranks = np.arange(1, len(cmc) + 1)
        plt.plot(ranks, cmc, marker='o', label=method_name)
    
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.title('Cumulative Matching Characteristic (CMC) Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, len(list(cmc_scores.values())[0]))
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

def analyze_failure_cases(query_features: torch.Tensor,
                         gallery_features: torch.Tensor,
                         query_labels: torch.Tensor,
                         gallery_labels: torch.Tensor,
                         distance_metric: str = 'euclidean',
                         num_cases: int = 5) -> List[Dict]:
    """
    分析失败案例
    
    Args:
        query_features: 查询特征
        gallery_features: 画廊特征
        query_labels: 查询标签
        gallery_labels: 画廊标签
        distance_metric: 距离度量方式
        num_cases: 分析的案例数量
    
    Returns:
        failure_cases: 失败案例列表
    """
    # 计算距离矩阵
    dist_mat = compute_distance_matrix(query_features, gallery_features, distance_metric)
    
    failure_cases = []
    
    for i in range(len(query_labels)):
        # 获取最近邻
        _, nearest_idx = torch.min(dist_mat[i], dim=0)
        nearest_label = gallery_labels[nearest_idx]
        
        # 如果最近邻不匹配，记录为失败案例
        if nearest_label != query_labels[i]:
            # 找到正确标签的最近距离
            correct_mask = gallery_labels == query_labels[i]
            if correct_mask.any():
                correct_distances = dist_mat[i][correct_mask]
                min_correct_distance = torch.min(correct_distances)
                
                failure_cases.append({
                    'query_idx': i,
                    'query_label': query_labels[i].item(),
                    'predicted_label': nearest_label.item(),
                    'predicted_distance': dist_mat[i, nearest_idx].item(),
                    'correct_min_distance': min_correct_distance.item(),
                    'rank_of_correct': (dist_mat[i] < min_correct_distance).sum().item() + 1
                })
    
    # 按错误程度排序（预测距离与正确距离的比值）
    failure_cases.sort(key=lambda x: x['predicted_distance'] / x['correct_min_distance'])
    
    return failure_cases[:num_cases]

def compute_recall_at_k(query_features: torch.Tensor,
                       gallery_features: torch.Tensor,
                       query_labels: torch.Tensor,
                       gallery_labels: torch.Tensor,
                       k: int = 1,
                       metric: str = 'euclidean') -> float:
    """计算 Recall@K
    Args:
        query_features: 查询特征 [N_q, D]
        gallery_features: 画廊特征 [N_g, D]
        query_labels: 查询标签 [N_q]
        gallery_labels: 画廊标签 [N_g]
        k: K 值
        metric: 距离度量方式
    Returns:
        recall_k: Recall@K 值 (0~1)
    """
    # 计算距离矩阵
    dist_mat = compute_distance_matrix(query_features, gallery_features, metric)
    
    # 取最小的 K 个索引
    topk_indices = torch.topk(dist_mat, k, largest=False).indices  # [N_q, K]
    # 检查是否命中正确标签
    matches = (gallery_labels[topk_indices] == query_labels.unsqueeze(1))
    recall_k = matches.any(dim=1).float().mean().item()
    return recall_k


def compute_map(query_features: torch.Tensor,
               gallery_features: torch.Tensor,
               query_labels: torch.Tensor,
               gallery_labels: torch.Tensor,
               metric: str = 'euclidean') -> float:
    """计算 mAP (mean Average Precision)
    Args 与 compute_recall_at_k 相同
    Returns:
        mAP 值 (0~1)
    """
    dist_mat = compute_distance_matrix(query_features, gallery_features, metric)
    return compute_ap(dist_mat, query_labels, gallery_labels)

if __name__ == "__main__":
    # 测试评估指标
    print("Testing evaluation metrics...")
    
    # 创建假数据
    num_query = 100
    num_gallery = 500
    feat_dim = 768
    num_classes = 701
    
    # 随机特征
    query_features = torch.randn(num_query, feat_dim)
    gallery_features = torch.randn(num_gallery, feat_dim)
    
    # 随机标签
    query_labels = torch.randint(0, num_classes, (num_query,))
    gallery_labels = torch.randint(0, num_classes, (num_gallery,))
    
    # 计算指标
    metrics = compute_metrics(
        query_features, gallery_features,
        query_labels, gallery_labels,
        distance_metric='euclidean',
        max_rank=20
    )
    
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 测试指标跟踪器
    tracker = MetricsTracker()
    tracker.update({'loss': 0.5, 'accuracy': 0.8})
    tracker.update({'loss': 0.4, 'accuracy': 0.85})
    
    print(f"\nMetrics tracker: {tracker}")
    print(f"Average metrics: {tracker.get_avg_metrics()}")
    
    print("Metrics test completed!") 