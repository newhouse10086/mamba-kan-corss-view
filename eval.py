#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSRA-VMK评估脚本
评估训练好的Vision Mamba Kolmogorov Network模型在University-1652数据集上的性能

支持的评估模式：
1. 标准评估：计算Recall@K、mAP、CMC等指标
2. 可视化评估：生成检索结果可视化
3. 性能分析：分析推理速度和内存使用
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入自定义模块
from models.vmamba_kan_fsra import FSRAVMambaKAN
from dataset.university1652_dataset import University1652Dataset
from utils.metrics import (
    compute_distance_matrix, compute_recall_at_k, 
    compute_map, compute_cmc, plot_cmc_curve
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FSRA-VMK Evaluation')
    
    # 基础参数
    parser.add_argument('--config', type=str, default='configs/fsra_vmk_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, default='./data/University-1652',
                        help='数据集根目录')
    
    # 评估参数
    parser.add_argument('--query_mode', type=str, default='drone_to_satellite',
                        choices=['drone_to_satellite', 'satellite_to_drone'],
                        help='查询模式')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--image_size', type=int, default=256,
                        help='图像尺寸')
    
    # 评估选项
    parser.add_argument('--flip_test', action='store_true',
                        help='使用翻转测试增强')
    parser.add_argument('--multi_scale_test', action='store_true',
                        help='使用多尺度测试')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0],
                        help='测试尺度')
    parser.add_argument('--reranking', action='store_true',
                        help='使用重排序')
    
    # 可视化和分析
    parser.add_argument('--visualize', action='store_true',
                        help='生成检索结果可视化')
    parser.add_argument('--num_vis_queries', type=int, default=10,
                        help='可视化的查询数量')
    parser.add_argument('--analyze_performance', action='store_true',
                        help='分析推理性能')
    parser.add_argument('--save_features', action='store_true',
                        help='保存提取的特征')
    
    # 输出设置
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='评估设备')
    
    return parser.parse_args()

class FSRAEvaluator:
    """FSRA-VMK评估器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 加载配置
        if os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        # 初始化数据加载器
        self._init_dataloader()
        
        # 初始化模型
        self._init_model()
        
        # 加载检查点
        self._load_checkpoint()
    
    def _init_dataloader(self):
        """初始化数据加载器"""
        print("🔄 初始化数据加载器...")
        
        # 查询集
        query_dataset = University1652Dataset(
            data_dir=self.args.data_dir,
            mode='query',
            query_mode=self.args.query_mode,
            image_size=self.args.image_size,
            augment=False
        )
        
        self.query_loader = DataLoader(
            query_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 画廊集
        gallery_dataset = University1652Dataset(
            data_dir=self.args.data_dir,
            mode='gallery',
            query_mode=self.args.query_mode,
            image_size=self.args.image_size,
            augment=False
        )
        
        self.gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"✅ 查询集: {len(query_dataset)} 样本")
        print(f"✅ 画廊集: {len(gallery_dataset)} 样本")
        
        # 保存数据集信息用于可视化
        self.query_dataset = query_dataset
        self.gallery_dataset = gallery_dataset
    
    def _init_model(self):
        """初始化模型"""
        print("🔄 初始化FSRA-VMK模型...")
        
        # 从配置或默认值获取模型参数
        model_config = self.config.get('model', {})
        
        self.model = FSRAVMambaKAN(
            num_classes=self.config.get('dataset', {}).get('num_classes', 1652),
            embed_dim=model_config.get('embed_dim', 512),
            depth=model_config.get('depth', 12),
            kan_grid_size=model_config.get('kan', {}).get('grid_size', 5),
            num_heads=model_config.get('num_heads', 8),
            image_size=self.args.image_size
        ).to(self.device)
        
        print("✅ 模型初始化完成")
    
    def _load_checkpoint(self):
        """加载模型检查点"""
        print(f"🔄 加载检查点: {self.args.checkpoint}")
        
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        
        # 兼容不同的保存格式
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'best_map' in checkpoint:
                print(f"📊 检查点最佳mAP: {checkpoint['best_map']:.4f}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("✅ 检查点加载完成")
    
    @torch.no_grad()
    def extract_features(self, dataloader, desc: str = "") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """提取特征"""
        print(f"🔄 提取{desc}特征...")
        
        all_features = []
        all_labels = []
        all_paths = []
        
        total_time = 0.0
        num_samples = 0
        
        for batch_idx, (images, labels, paths) in enumerate(dataloader):
            start_time = time.time()
            
            images = images.to(self.device)
            batch_size = images.size(0)
            
            # 多尺度测试
            if self.args.multi_scale_test:
                features_list = []
                for scale in self.args.scales:
                    if scale != 1.0:
                        h, w = images.shape[2], images.shape[3]
                        new_h, new_w = int(h * scale), int(w * scale)
                        scaled_images = F.interpolate(images, size=(new_h, new_w), 
                                                    mode='bilinear', align_corners=False)
                    else:
                        scaled_images = images
                    
                    outputs = self.model(scaled_images)
                    features_list.append(outputs['features'])
                
                # 特征融合
                features = torch.stack(features_list).mean(dim=0)
            else:
                outputs = self.model(images)
                features = outputs['features']
            
            # 翻转测试
            if self.args.flip_test:
                flipped_images = torch.flip(images, dims=[3])  # 水平翻转
                flipped_outputs = self.model(flipped_images)
                flipped_features = flipped_outputs['features']
                features = (features + flipped_features) / 2
            
            # L2归一化
            features = F.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
            all_paths.extend(paths)
            
            batch_time = time.time() - start_time
            total_time += batch_time
            num_samples += batch_size
            
            if batch_idx % 20 == 0:
                print(f"  处理进度: [{batch_idx}/{len(dataloader)}] "
                      f"批次时间: {batch_time:.3f}s "
                      f"平均FPS: {batch_size/batch_time:.1f}")
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        avg_time_per_sample = total_time / num_samples
        print(f"✅ {desc}特征提取完成: {all_features.shape}")
        print(f"⏱️  平均推理时间: {avg_time_per_sample*1000:.2f}ms/样本")
        
        return all_features, all_labels, all_paths
    
    def compute_metrics(self, query_features: torch.Tensor, 
                       gallery_features: torch.Tensor,
                       query_labels: torch.Tensor, 
                       gallery_labels: torch.Tensor) -> Dict[str, float]:
        """计算评估指标"""
        print("🔄 计算评估指标...")
        
        metrics = {}
        
        # Recall@K
        for k in [1, 5, 10, 20]:
            recall_k = compute_recall_at_k(query_features, gallery_features, 
                                         query_labels, gallery_labels, k=k)
            metrics[f'Recall@{k}'] = recall_k
        
        # mAP
        map_score = compute_map(query_features, gallery_features, 
                               query_labels, gallery_labels)
        metrics['mAP'] = map_score
        
        # CMC曲线
        cmc = compute_cmc(query_features, gallery_features, 
                         query_labels, gallery_labels, max_rank=50)
        
        for i, rank in enumerate([1, 5, 10, 20, 50]):
            if i < len(cmc):
                metrics[f'CMC@{rank}'] = cmc[i]
        
        # 平均排序
        distance_matrix = compute_distance_matrix(query_features, gallery_features)
        
        mean_rank = 0.0
        median_rank = []
        
        for i in range(len(query_labels)):
            # 找到正确匹配的gallery索引
            positive_mask = gallery_labels == query_labels[i]
            if positive_mask.sum() == 0:
                continue
            
            # 计算排序
            distances = distance_matrix[i]
            sorted_indices = torch.argsort(distances)
            
            # 找到第一个正确匹配的排序位置
            for rank, idx in enumerate(sorted_indices):
                if positive_mask[idx]:
                    mean_rank += (rank + 1)
                    median_rank.append(rank + 1)
                    break
        
        metrics['MeanRank'] = mean_rank / len(query_labels)
        metrics['MedianRank'] = float(np.median(median_rank))
        
        return metrics, cmc
    
    def rerank_features(self, query_features: torch.Tensor, 
                       gallery_features: torch.Tensor,
                       k1: int = 20, k2: int = 6, lambda_value: float = 0.3) -> torch.Tensor:
        """k-reciprocal重排序"""
        print("🔄 执行k-reciprocal重排序...")
        
        # 计算原始距离矩阵
        original_dist = compute_distance_matrix(query_features, gallery_features)
        
        # TODO: 实现完整的k-reciprocal重排序算法
        # 这里先返回原始距离矩阵
        return original_dist
    
    def visualize_results(self, query_features: torch.Tensor, 
                         gallery_features: torch.Tensor,
                         query_labels: torch.Tensor, 
                         gallery_labels: torch.Tensor,
                         query_paths: List[str],
                         gallery_paths: List[str]):
        """可视化检索结果"""
        print("🔄 生成检索结果可视化...")
        
        # 计算距离矩阵
        distance_matrix = compute_distance_matrix(query_features, gallery_features)
        
        # 随机选择查询样本进行可视化
        query_indices = np.random.choice(len(query_labels), 
                                       min(self.args.num_vis_queries, len(query_labels)), 
                                       replace=False)
        
        vis_dir = os.path.join(self.args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        for i, query_idx in enumerate(query_indices):
            # 获取查询信息
            query_label = query_labels[query_idx].item()
            query_path = query_paths[query_idx]
            
            # 获取检索结果（top-10）
            distances = distance_matrix[query_idx]
            sorted_indices = torch.argsort(distances)[:10]
            
            # 创建可视化图像
            fig, axes = plt.subplots(2, 6, figsize=(18, 6))
            fig.suptitle(f'Query {i+1}: Class {query_label}', fontsize=16)
            
            # 显示查询图像
            try:
                from PIL import Image
                query_img = Image.open(query_path).convert('RGB')
                axes[0, 0].imshow(query_img)
                axes[0, 0].set_title('Query', fontweight='bold')
                axes[0, 0].axis('off')
            except:
                axes[0, 0].text(0.5, 0.5, 'Query Image\nNot Found', 
                              ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].axis('off')
            
            # 隐藏第一行剩余的subplot
            for j in range(1, 6):
                axes[0, j].axis('off')
            
            # 显示检索结果
            for j, gallery_idx in enumerate(sorted_indices[:5]):
                gallery_label = gallery_labels[gallery_idx].item()
                gallery_path = gallery_paths[gallery_idx]
                distance = distances[gallery_idx].item()
                
                # 判断是否正确匹配
                is_correct = gallery_label == query_label
                color = 'green' if is_correct else 'red'
                
                try:
                    gallery_img = Image.open(gallery_path).convert('RGB')
                    axes[1, j].imshow(gallery_img)
                except:
                    axes[1, j].text(0.5, 0.5, 'Image\nNot Found', 
                                  ha='center', va='center', transform=axes[1, j].transAxes)
                
                axes[1, j].set_title(f'Rank {j+1}\nClass {gallery_label}\nDist: {distance:.3f}', 
                                   color=color, fontsize=10)
                axes[1, j].axis('off')
                
                # 添加边框
                for spine in axes[1, j].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
            
            # 隐藏最后一个subplot
            axes[1, 5].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'query_{i+1}_results.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ 可视化结果保存至: {vis_dir}")
    
    def analyze_performance(self):
        """分析推理性能"""
        print("🔄 分析推理性能...")
        
        # 创建测试数据
        test_input = torch.randn(1, 3, self.args.image_size, self.args.image_size).to(self.device)
        
        # 预热
        for _ in range(10):
            _ = self.model(test_input)
        
        # 测量推理时间
        torch.cuda.synchronize()
        times = []
        
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(test_input)
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
        
        # 计算统计信息
        mean_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        
        # 统计模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size = total_params * 4 / 1024 / 1024  # MB
        
        # 内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        performance_info = {
            'inference_time_ms': {
                'mean': mean_time,
                'std': std_time,
                'min': min_time,
                'max': max_time
            },
            'throughput_fps': 1000 / mean_time,
            'model_params': {
                'total': total_params,
                'trainable': trainable_params,
                'size_mb': model_size
            },
            'memory_usage_mb': {
                'allocated': memory_allocated,
                'reserved': memory_reserved
            }
        }
        
        print("📊 性能分析结果:")
        print(f"  推理时间: {mean_time:.2f}±{std_time:.2f}ms")
        print(f"  吞吐量: {1000/mean_time:.1f} FPS")
        print(f"  模型参数: {total_params:,} ({model_size:.2f} MB)")
        print(f"  GPU内存: {memory_allocated:.2f} MB")
        
        # 保存性能分析结果
        import json
        with open(os.path.join(self.args.output_dir, 'performance_analysis.json'), 'w') as f:
            json.dump(performance_info, f, indent=2)
        
        return performance_info
    
    def run_evaluation(self):
        """运行完整评估"""
        print("🚀 开始FSRA-VMK模型评估...")
        print(f"📊 评估配置:")
        print(f"  - 查询模式: {self.args.query_mode}")
        print(f"  - 翻转测试: {self.args.flip_test}")
        print(f"  - 多尺度测试: {self.args.multi_scale_test}")
        print(f"  - 重排序: {self.args.reranking}")
        print(f"  - 设备: {self.device}")
        print("=" * 80)
        
        start_time = time.time()
        
        # 提取特征
        query_features, query_labels, query_paths = self.extract_features(
            self.query_loader, "查询集")
        gallery_features, gallery_labels, gallery_paths = self.extract_features(
            self.gallery_loader, "画廊集")
        
        # 保存特征（可选）
        if self.args.save_features:
            features_path = os.path.join(self.args.output_dir, 'features.pth')
            torch.save({
                'query_features': query_features,
                'query_labels': query_labels,
                'query_paths': query_paths,
                'gallery_features': gallery_features,
                'gallery_labels': gallery_labels,
                'gallery_paths': gallery_paths
            }, features_path)
            print(f"💾 特征已保存至: {features_path}")
        
        # 重排序（可选）
        if self.args.reranking:
            # TODO: 实现重排序
            pass
        
        # 计算评估指标
        metrics, cmc = self.compute_metrics(query_features, gallery_features, 
                                          query_labels, gallery_labels)
        
        # 打印结果
        print("\n🏆 评估结果:")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"  {metric:12s}: {value:.4f}")
        print("=" * 50)
        
        # 绘制CMC曲线
        plt.figure(figsize=(10, 6))
        plot_cmc_curve(cmc, max_rank=20)
        plt.title(f'CMC Curve - {self.args.query_mode}')
        plt.savefig(os.path.join(self.args.output_dir, 'cmc_curve.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 可视化结果（可选）
        if self.args.visualize:
            self.visualize_results(query_features, gallery_features,
                                 query_labels, gallery_labels,
                                 query_paths, gallery_paths)
        
        # 性能分析（可选）
        if self.args.analyze_performance:
            performance_info = self.analyze_performance()
        
        # 保存评估结果
        results = {
            'metrics': metrics,
            'config': vars(self.args),
            'query_mode': self.args.query_mode,
            'num_queries': len(query_labels),
            'num_gallery': len(gallery_labels)
        }
        
        import json
        with open(os.path.join(self.args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  评估完成，总用时: {total_time:.2f}秒")
        print(f"📁 结果保存在: {self.args.output_dir}")

def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("🔍 FSRA-VMK模型评估")
    print("📋 评估配置:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("=" * 80)
    
    # 创建评估器并运行评估
    evaluator = FSRAEvaluator(args)
    evaluator.run_evaluation()

if __name__ == '__main__':
    main() 