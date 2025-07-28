#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSRA-VMK训练脚本
结合Vision Mamba和KAN技术的跨视角图像匹配模型训练

基于您的分析，实现FSRA-VMK: Vision Mamba Kolmogorov Network
- O(n)线性复杂度的Vision Mamba编码器
- Kolmogorov-Arnold Networks注意力机制
- 双向跨视角特征对齐
"""

import os
import sys
import time
import argparse
import datetime
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

# 导入自定义模块
from models.vmamba_kan_fsra import FSRAVMambaKAN
from dataset.university1652_dataset import University1652Dataset
from utils.losses import CrossEntropyLabelSmooth, TripletLoss, ContrastiveLoss
from utils.metrics import compute_recall_at_k, compute_map, compute_cmc
from utils.lr_scheduler import WarmupMultiStepLR

def set_seed(seed: int = 42):
    """设置随机种子保证实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FSRA-VMK Training')
    
    # 通用配置
    parser.add_argument('--config', type=str, default='', help='YAML 配置文件路径')

    # 数据集参数
    parser.add_argument('--data_dir', type=str, default='./data/University-1652',
                        help='数据集根目录')
    parser.add_argument('--num_classes', type=int, default=1652,
                        help='类别数量')
    parser.add_argument('--image_size', type=int, default=256,
                        help='图像尺寸')
    parser.add_argument('--query_mode', type=str, default='drone_to_satellite',
                        choices=['drone_to_satellite', 'satellite_to_drone'],
                        help='查询模式')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='特征维度')
    parser.add_argument('--depth', type=int, default=12,
                        help='VMamba深度')
    parser.add_argument('--kan_grid_size', type=int, default=5,
                        help='KAN网格大小')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='注意力头数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='学习率预热轮数')
    
    # 损失函数权重
    parser.add_argument('--id_loss_weight', type=float, default=1.0,
                        help='ID损失权重')
    parser.add_argument('--triplet_loss_weight', type=float, default=1.0,
                        help='三元组损失权重')
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.5,
                        help='对比损失权重')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_step', type=int, default=10,
                        help='评估间隔')
    parser.add_argument('--save_step', type=int, default=50,
                        help='保存间隔')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    
    return parser.parse_args()

class FSRATrainer:
    """FSRA-VMK训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # 创建TensorBoard记录器
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(args.log_dir, f'fsra_vmk_{timestamp}'))
        
        # 初始化数据加载器
        self._init_dataloader()
        
        # 初始化模型
        self._init_model()
        
        # 初始化损失函数
        self._init_loss_functions()
        
        # 初始化优化器和学习率调度器
        self._init_optimizer()
        
        # 训练状态
        self.start_epoch = 0
        self.best_map = 0.0
        self.best_recall1 = 0.0
        
        # 恢复训练
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _init_dataloader(self):
        """初始化数据加载器"""
        print("🔄 初始化数据加载器...")
        
        # 训练集
        train_dataset = University1652Dataset(
            data_dir=self.args.data_dir,
            mode='train',
            query_mode=self.args.query_mode,
            image_size=self.args.image_size,
            augment=True
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        # 测试集 - 查询集
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
        
        # 测试集 - 画廊集
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
        
        print(f"✅ 训练集: {len(train_dataset)} 样本")
        print(f"✅ 查询集: {len(query_dataset)} 样本")
        print(f"✅ 画廊集: {len(gallery_dataset)} 样本")
    
    def _init_model(self):
        """初始化模型"""
        print("🔄 初始化FSRA-VMK模型...")
        
        self.model = FSRAVMambaKAN(
            num_classes=self.args.num_classes,
            embed_dim=self.args.embed_dim,
            depth=self.args.depth,
            kan_grid_size=self.args.kan_grid_size,
            num_heads=self.args.num_heads,
            image_size=self.args.image_size
        ).to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ 模型参数总量: {total_params:,}")
        print(f"✅ 可训练参数量: {trainable_params:,}")
        print(f"✅ 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def _init_loss_functions(self):
        """初始化损失函数"""
        print("🔄 初始化损失函数...")
        
        # ID分类损失（带标签平滑）
        self.id_loss = CrossEntropyLabelSmooth(
            num_classes=self.args.num_classes,
            epsilon=0.1
        ).to(self.device)
        
        # 三元组损失
        self.triplet_loss = TripletLoss(
            margin=0.3,
            distance='euclidean'
        ).to(self.device)
        
        # 对比损失
        self.contrastive_loss = ContrastiveLoss(
            temperature=0.07
        ).to(self.device)
        
        print("✅ 损失函数初始化完成")
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        print("🔄 初始化优化器...")
        
        # 参数分组
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # AdamW优化器
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.args.lr},
            {'params': classifier_params, 'lr': self.args.lr * 10}  # 分类器使用更大学习率
        ], weight_decay=self.args.weight_decay)
        
        # 学习率调度器（带预热）
        self.scheduler = WarmupMultiStepLR(
            self.optimizer,
            milestones=[150, 250],
            gamma=0.1,
            warmup_epochs=self.args.warmup_epochs,
            warmup_factor=0.1
        )
        
        print(f"✅ 优化器: AdamW, 学习率: {self.args.lr}")
        print(f"✅ 学习率调度: 预热{self.args.warmup_epochs}轮，[150, 250]轮衰减")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        print(f"🔄 加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_map = checkpoint.get('best_map', 0.0)
        self.best_recall1 = checkpoint.get('best_recall1', 0.0)
        
        print(f"✅ 检查点加载完成，从第{self.start_epoch}轮开始训练")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'best_recall1': self.best_recall1,
            'args': self.args
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型: {best_path}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        losses = {
            'total': 0.0,
            'id': 0.0,
            'triplet': 0.0,
            'contrastive': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)  # [B, C, H, W]
            labels = labels.to(self.device)  # [B]
            
            # 前向传播
            outputs = self.model(images)
            logits = outputs['logits']      # [B, num_classes]
            features = outputs['features']  # [B, embed_dim]
            
            # 计算损失
            id_loss = self.id_loss(logits, labels)
            triplet_loss = self.triplet_loss(features, labels)
            contrastive_loss = self.contrastive_loss(features, labels)
            
            # 总损失
            total_loss = (self.args.id_loss_weight * id_loss + 
                         self.args.triplet_loss_weight * triplet_loss + 
                         self.args.contrastive_loss_weight * contrastive_loss)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 累计损失
            losses['total'] += total_loss.item()
            losses['id'] += id_loss.item()
            losses['triplet'] += triplet_loss.item()
            losses['contrastive'] += contrastive_loss.item()
            
            # 打印进度
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{self.args.epochs}] "
                      f"Batch [{batch_idx}/{num_batches}] "
                      f"Loss: {total_loss.item():.4f} "
                      f"ID: {id_loss.item():.4f} "
                      f"Triplet: {triplet_loss.item():.4f} "
                      f"Contrast: {contrastive_loss.item():.4f}")
        
        # 计算平均损失
        for key in losses:
            losses[key] /= num_batches
        
        return losses
    
    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """评估模型"""
        print("🔄 开始评估...")
        self.model.eval()
        
        # 提取查询特征
        query_features = []
        query_labels = []
        
        for images, labels, _ in self.query_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            features = outputs['features']
            
            query_features.append(features.cpu())
            query_labels.append(labels)
        
        query_features = torch.cat(query_features, dim=0)  # [N_q, D]
        query_labels = torch.cat(query_labels, dim=0)      # [N_q]
        
        # 提取画廊特征
        gallery_features = []
        gallery_labels = []
        
        for images, labels, _ in self.gallery_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            features = outputs['features']
            
            gallery_features.append(features.cpu())
            gallery_labels.append(labels)
        
        gallery_features = torch.cat(gallery_features, dim=0)  # [N_g, D]
        gallery_labels = torch.cat(gallery_labels, dim=0)      # [N_g]
        
        # 计算评估指标
        metrics = {}
        
        # Recall@K
        for k in [1, 5, 10]:
            recall_k = compute_recall_at_k(query_features, gallery_features, 
                                         query_labels, gallery_labels, k=k)
            metrics[f'Recall@{k}'] = recall_k
        
        # mAP
        map_score = compute_map(query_features, gallery_features, 
                               query_labels, gallery_labels)
        metrics['mAP'] = map_score
        
        # CMC曲线
        cmc = compute_cmc(query_features, gallery_features, 
                         query_labels, gallery_labels, max_rank=20)
        for i, rank in enumerate([1, 5, 10, 20]):
            if i < len(cmc):
                metrics[f'CMC@{rank}'] = cmc[i]
        
        print("📊 评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train(self):
        """主训练循环"""
        print("🚀 开始训练FSRA-VMK模型...")
        print(f"📊 训练配置:")
        print(f"  - 数据集: University-1652")
        print(f"  - 查询模式: {self.args.query_mode}")
        print(f"  - 批次大小: {self.args.batch_size}")
        print(f"  - 训练轮数: {self.args.epochs}")
        print(f"  - 学习率: {self.args.lr}")
        print(f"  - 设备: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_losses = self.train_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 记录训练损失
            for loss_name, loss_value in train_losses.items():
                self.writer.add_scalar(f'Train/Loss_{loss_name}', loss_value, epoch)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # 评估模型
            if (epoch + 1) % self.args.eval_step == 0 or epoch == self.args.epochs - 1:
                eval_metrics = self.evaluate(epoch)
                
                # 记录评估指标
                for metric_name, metric_value in eval_metrics.items():
                    self.writer.add_scalar(f'Eval/{metric_name}', metric_value, epoch)
                
                # 检查是否为最佳模型
                is_best = False
                if eval_metrics['mAP'] > self.best_map:
                    self.best_map = eval_metrics['mAP']
                    self.best_recall1 = eval_metrics['Recall@1']
                    is_best = True
                
                # 保存检查点
                if (epoch + 1) % self.args.save_step == 0 or is_best:
                    self._save_checkpoint(epoch, is_best)
            
            epoch_time = time.time() - epoch_start_time
            print(f"⏱️  Epoch {epoch} 完成，用时: {epoch_time:.2f}s, "
                  f"学习率: {current_lr:.6f}")
            print("-" * 80)
        
        total_time = time.time() - start_time
        print(f"🎉 训练完成！总用时: {total_time/3600:.2f}小时")
        print(f"🏆 最佳结果: mAP={self.best_map:.4f}, Recall@1={self.best_recall1:.4f}")
        
        self.writer.close()

def main():
    """主函数"""
    args = parse_args()

    # 如果提供了 YAML 配置文件，加载并覆盖默认参数
    if args.config:
        import yaml, argparse
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)

        # 递归扁平化字典
        def flatten_dict(d, parent_key='', sep='.'):  # simple flatten
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items

        flat_cfg = flatten_dict(cfg_dict)

        for key, value in flat_cfg.items():
            # 将嵌套键转换为属性名，例如 training.batch_size -> batch_size
            attr = key.split('.')[-1]
            if hasattr(args, attr):
                setattr(args, attr, value)
        print(f"🔧 从配置文件 {args.config} 加载参数并覆盖默认值")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印配置
    print("=" * 80)
    print("🚀 FSRA-VMK: Vision Mamba Kolmogorov Network")
    print("📋 训练配置:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("=" * 80)
    
    # 创建训练器并开始训练
    trainer = FSRATrainer(args)
    trainer.train()

if __name__ == '__main__':
    main() 