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

def compute_cmc(dist_mat: torch.Tensor, 
                query_labels: torch.Tensor, 
                gallery_labels: torch.Tensor,
                max_rank: int = 50) -> np.ndarray:
    """
    计算累积匹配特征曲线 (CMC)
    
    Args:
        dist_mat: 距离矩阵 [N_q, N_g]
        query_labels: 查询标签 [N_q]
        gallery_labels: 画廊标签 [N_g]
        max_rank: 最大rank
    
    Returns:
        cmc: CMC曲线 [max_rank]
    """
    num_q, num_g = dist_mat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print(f"Warning: Gallery size ({num_g}) is smaller than max_rank ({max_rank})")
    
    # 排序获得索引
    indices = torch.argsort(dist_mat, dim=1)  # [N_q, N_g]
    
    # 计算匹配情况
    matches = (gallery_labels[indices] == query_labels.unsqueeze(1)).cpu().numpy()
    
    # 初始化CMC
    cmc = np.zeros(max_rank)
    
    for i in range(num_q):
        # 找到第一个匹配的位置
        first_match = np.where(matches[i])[0]
        if len(first_match) > 0:
            first_match_idx = first_match[0]
            if first_match_idx < max_rank:
                cmc[first_match_idx:] += 1
    
    cmc = cmc / num_q
    return cmc

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