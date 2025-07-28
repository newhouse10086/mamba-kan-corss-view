import torch
import numpy as np
from typing import Iterator, List
from collections import defaultdict

class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    类均衡批次采样器
    确保每个batch内包含多个类别，每个类别有多个样本
    这对Triplet Loss的训练至关重要
    """
    
    def __init__(self, 
                 labels: List[int], 
                 batch_size: int, 
                 samples_per_class: int = 4):
        """
        Args:
            labels: 所有样本的标签列表
            batch_size: 批次大小
            samples_per_class: 每个类别在一个batch中的样本数
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        # 计算每个batch需要多少个类别
        self.classes_per_batch = batch_size // samples_per_class
        
        # 按类别分组样本索引
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)
        
        # 获取所有类别
        self.classes = list(self.class_to_indices.keys())
        
        # 过滤掉样本数不足的类别
        self.valid_classes = [c for c in self.classes if len(self.class_to_indices[c]) >= samples_per_class]
        
        if len(self.valid_classes) < self.classes_per_batch:
            raise ValueError(f"Not enough valid classes to form a batch. "
                             f"Need {self.classes_per_batch} classes, but only {len(self.valid_classes)} have >= {samples_per_class} samples.")
        
        # 估算每个epoch的迭代次数
        self.num_iterations = sum(len(self.class_to_indices[c]) for c in self.valid_classes) // batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """生成一个epoch的批次"""
        for _ in range(self.num_iterations):
            batch = []
            
            # 随机选择类别
            selected_classes = np.random.choice(self.valid_classes, self.classes_per_batch, replace=False)
            
            # 从每个类别中随机采样
            for cls in selected_classes:
                indices = self.class_to_indices[cls]
                selected_indices = np.random.choice(indices, self.samples_per_class, replace=False)
                batch.extend(selected_indices)
            
            np.random.shuffle(batch)
            yield batch
    
    def __len__(self) -> int:
        return self.num_iterations


class RandomIdentitySampler(torch.utils.data.Sampler):
    """
    随机身份采样器 - 更简单的版本
    确保每个batch包含P个身份，每个身份K个样本
    """
    
    def __init__(self, labels: List[int], batch_size: int, num_instances: int = 4):
        """
        Args:
            labels: 标签列表
            batch_size: 批次大小
            num_instances: 每个身份的实例数
        """
        self.labels = labels
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        
        # 按身份分组
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(labels):
            self.index_dic[pid].append(index)
        
        self.pids = list(self.index_dic.keys())
        
        # 估算长度
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        
        for pid in self.pids:
            idxs = self.index_dic[pid].copy()
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        
        avai_pids = list(batch_idxs_dict.keys())
        final_idxs = []
        
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        
        return iter(final_idxs)
    
    def __len__(self):
        return self.length 