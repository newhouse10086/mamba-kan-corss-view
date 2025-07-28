import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
import json
import random
from typing import Tuple, List, Dict, Optional

class University1652Dataset(data.Dataset):
    """
    University-1652数据集处理类
    支持跨视角图像匹配任务：Drone->Satellite 和 Satellite->Drone
    """
    
    def __init__(self, 
                 data_dir: str,
                 mode: str = 'train',
                 query_mode: str = 'drone_to_satellite',  # 'drone_to_satellite' or 'satellite_to_drone'
                 image_size: int = 256,
                 augment: bool = True):
        """
        初始化数据集
        Args:
            data_dir: 数据集根目录
            mode: 'train', 'test', 'query', 'gallery'
            query_mode: 查询模式
            image_size: 图像尺寸
            augment: 是否使用数据增强
        """
        self.data_dir = data_dir
        self.mode = mode
        self.query_mode = query_mode
        self.image_size = image_size
        self.augment = augment
        
        # 定义图像变换
        self.transform = self._get_transforms()
        
        # 加载数据
        self.data_list = self._load_data()
        
        print(f"Loaded {len(self.data_list)} samples for {mode} mode")
    
    def _get_transforms(self):
        """获取图像变换"""
        if self.mode == 'train' and self.augment:
            # 训练时的数据增强
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 测试时的变换
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    def _load_data(self) -> List[Dict]:
        """加载数据列表"""
        data_list = []
        
        if self.mode == 'train':
            # 训练模式：加载训练数据
            drone_dir = os.path.join(self.data_dir, 'train', 'drone')
            satellite_dir = os.path.join(self.data_dir, 'train', 'satellite')
            
            # 获取所有类别
            classes = sorted([d for d in os.listdir(drone_dir) if os.path.isdir(os.path.join(drone_dir, d))])
            
            for class_id, class_name in enumerate(classes):
                # 获取无人机图像
                drone_class_dir = os.path.join(drone_dir, class_name)
                drone_images = [f for f in os.listdir(drone_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # 获取卫星图像
                satellite_class_dir = os.path.join(satellite_dir, class_name)
                satellite_images = [f for f in os.listdir(satellite_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # 创建样本对
                for drone_img in drone_images:
                    for satellite_img in satellite_images:
                        data_list.append({
                            'drone_path': os.path.join(drone_class_dir, drone_img),
                            'satellite_path': os.path.join(satellite_class_dir, satellite_img),
                            'class_id': class_id,
                            'class_name': class_name
                        })
        
        elif self.mode == 'query':
            # 查询模式：根据查询模式加载查询图像
            if self.query_mode == 'drone_to_satellite':
                query_dir = os.path.join(self.data_dir, 'test', 'query_drone')
            else:  # satellite_to_drone
                query_dir = os.path.join(self.data_dir, 'test', 'query_satellite')
            
            if os.path.exists(query_dir):
                # 获取所有类别
                classes = sorted([d for d in os.listdir(query_dir) if os.path.isdir(os.path.join(query_dir, d))])
                
                for class_id, class_name in enumerate(classes):
                    query_class_dir = os.path.join(query_dir, class_name)
                    if os.path.exists(query_class_dir):
                        query_images = [f for f in os.listdir(query_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        for img in query_images:
                            data_list.append({
                                'image_path': os.path.join(query_class_dir, img),
                                'class_id': class_id,
                                'class_name': class_name
                            })
        
        elif self.mode == 'gallery':
            # 画廊模式：根据查询模式加载画廊图像
            if self.query_mode == 'drone_to_satellite':
                gallery_dir = os.path.join(self.data_dir, 'test', 'gallery_satellite')
            else:  # satellite_to_drone
                gallery_dir = os.path.join(self.data_dir, 'test', 'gallery_drone')
            
            if os.path.exists(gallery_dir):
                # 获取所有类别
                classes = sorted([d for d in os.listdir(gallery_dir) if os.path.isdir(os.path.join(gallery_dir, d))])
                
                for class_id, class_name in enumerate(classes):
                    gallery_class_dir = os.path.join(gallery_dir, class_name)
                    if os.path.exists(gallery_class_dir):
                        gallery_images = [f for f in os.listdir(gallery_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        for img in gallery_images:
                            data_list.append({
                                'image_path': os.path.join(gallery_class_dir, img),
                                'class_id': class_id,
                                'class_name': class_name
                            })

        elif self.mode == 'test':
            # 测试模式：根据查询模式加载数据
            if self.query_mode == 'drone_to_satellite':
                query_dir = os.path.join(self.data_dir, 'test', 'drone')
                gallery_dir = os.path.join(self.data_dir, 'test', 'satellite')
            else:  # satellite_to_drone
                query_dir = os.path.join(self.data_dir, 'test', 'satellite')
                gallery_dir = os.path.join(self.data_dir, 'test', 'drone')
            
            # 获取所有类别
            classes = sorted([d for d in os.listdir(query_dir) if os.path.isdir(os.path.join(query_dir, d))])
            
            for class_id, class_name in enumerate(classes):
                query_class_dir = os.path.join(query_dir, class_name)
                gallery_class_dir = os.path.join(gallery_dir, class_name)
                
                # 查询图像
                if os.path.exists(query_class_dir):
                    query_images = [f for f in os.listdir(query_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    for img in query_images:
                        data_list.append({
                            'image_path': os.path.join(query_class_dir, img),
                            'class_id': class_id,
                            'class_name': class_name,
                            'is_query': True
                        })
                
                # 画廊图像
                if os.path.exists(gallery_class_dir):
                    gallery_images = [f for f in os.listdir(gallery_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    for img in gallery_images:
                        data_list.append({
                            'image_path': os.path.join(gallery_class_dir, img),
                            'class_id': class_id,
                            'class_name': class_name,
                            'is_query': False
                        })
        
        return data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index: int) -> Tuple:
        """获取数据样本"""
        item = self.data_list[index]
        
        if self.mode == 'train':
            # 训练模式：返回配对的图像
            drone_img = Image.open(item['drone_path']).convert('RGB')
            satellite_img = Image.open(item['satellite_path']).convert('RGB')
            
            drone_img = self.transform(drone_img)
            satellite_img = self.transform(satellite_img)
            
            return (
                drone_img,  # 作为主图像
                item['class_id'],
                item['drone_path']  # 图像路径
            )
        
        elif self.mode in ['query', 'gallery']:
            # 查询/画廊模式：返回单个图像
            img = Image.open(item['image_path']).convert('RGB')
            img = self.transform(img)
            
            return (
                img,
                item['class_id'],
                item['image_path']
            )
        
        else:  # test mode
            # 测试模式：返回单个图像
            img = Image.open(item['image_path']).convert('RGB')
            img = self.transform(img)
            
            return (
                img,
                item['class_id'],
                item['image_path']
            )

class TripletDataset(data.Dataset):
    """
    三元组数据集，用于度量学习
    """
    
    def __init__(self, 
                 data_dir: str,
                 image_size: int = 256,
                 samples_per_class: int = 4):
        """
        Args:
            data_dir: 数据集根目录
            image_size: 图像尺寸
            samples_per_class: 每个类别的样本数
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.samples_per_class = samples_per_class
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 构建类别到图像的映射
        self.class_to_images = self._build_class_mapping()
        self.classes = list(self.class_to_images.keys())
    
    def _build_class_mapping(self) -> Dict:
        """构建类别到图像路径的映射"""
        class_to_images = {}
        
        # 处理无人机图像
        drone_dir = os.path.join(self.data_dir, 'train', 'drone')
        satellite_dir = os.path.join(self.data_dir, 'train', 'satellite')
        
        classes = sorted([d for d in os.listdir(drone_dir) if os.path.isdir(os.path.join(drone_dir, d))])
        
        for class_name in classes:
            class_to_images[class_name] = {'drone': [], 'satellite': []}
            
            # 无人机图像
            drone_class_dir = os.path.join(drone_dir, class_name)
            if os.path.exists(drone_class_dir):
                drone_images = [os.path.join(drone_class_dir, f) for f in os.listdir(drone_class_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                class_to_images[class_name]['drone'] = drone_images
            
            # 卫星图像
            satellite_class_dir = os.path.join(satellite_dir, class_name)
            if os.path.exists(satellite_class_dir):
                satellite_images = [os.path.join(satellite_class_dir, f) for f in os.listdir(satellite_class_dir) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                class_to_images[class_name]['satellite'] = satellite_images
        
        return class_to_images
    
    def __len__(self) -> int:
        return len(self.classes) * self.samples_per_class
    
    def __getitem__(self, index: int) -> Dict:
        """获取三元组样本"""
        # 选择anchor类别
        anchor_class = self.classes[index // self.samples_per_class]
        
        # 随机选择anchor图像（无人机或卫星）
        anchor_view = random.choice(['drone', 'satellite'])
        if len(self.class_to_images[anchor_class][anchor_view]) == 0:
            anchor_view = 'satellite' if anchor_view == 'drone' else 'drone'
        
        anchor_path = random.choice(self.class_to_images[anchor_class][anchor_view])
        
        # 选择positive图像（另一个视角的同类别图像）
        positive_view = 'satellite' if anchor_view == 'drone' else 'drone'
        if len(self.class_to_images[anchor_class][positive_view]) > 0:
            positive_path = random.choice(self.class_to_images[anchor_class][positive_view])
        else:
            # 如果没有另一个视角的图像，使用同视角的不同图像
            available_anchors = [p for p in self.class_to_images[anchor_class][anchor_view] if p != anchor_path]
            positive_path = random.choice(available_anchors) if available_anchors else anchor_path
        
        # 选择negative图像（不同类别）
        negative_classes = [c for c in self.classes if c != anchor_class]
        negative_class = random.choice(negative_classes)
        negative_view = random.choice(['drone', 'satellite'])
        
        if len(self.class_to_images[negative_class][negative_view]) == 0:
            negative_view = 'satellite' if negative_view == 'drone' else 'drone'
        
        negative_path = random.choice(self.class_to_images[negative_class][negative_view])
        
        # 加载和变换图像
        anchor_img = self.transform(Image.open(anchor_path).convert('RGB'))
        positive_img = self.transform(Image.open(positive_path).convert('RGB'))
        negative_img = self.transform(Image.open(negative_path).convert('RGB'))
        
        return {
            'anchor': anchor_img,
            'positive': positive_img,
            'negative': negative_img,
            'anchor_class': anchor_class,
            'negative_class': negative_class,
            'anchor_view': anchor_view,
            'positive_view': positive_view,
            'negative_view': negative_view
        }

def create_dataloaders(data_dir: str, 
                      batch_size: int = 32,
                      num_workers: int = 4,
                      image_size: int = 256) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    创建训练、验证和测试数据加载器
    """
    # 训练数据集
    train_dataset = University1652Dataset(
        data_dir=data_dir,
        mode='train',
        image_size=image_size,
        augment=True
    )
    
    # 测试数据集（无人机->卫星）
    test_dataset_d2s = University1652Dataset(
        data_dir=data_dir,
        mode='test',
        query_mode='drone_to_satellite',
        image_size=image_size,
        augment=False
    )
    
    # 测试数据集（卫星->无人机）
    test_dataset_s2d = University1652Dataset(
        data_dir=data_dir,
        mode='test',
        query_mode='satellite_to_drone',
        image_size=image_size,
        augment=False
    )
    
    # 数据加载器
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader_d2s = data.DataLoader(
        test_dataset_d2s,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader_s2d = data.DataLoader(
        test_dataset_s2d,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader_d2s, test_loader_s2d

def create_triplet_dataloader(data_dir: str,
                             batch_size: int = 32,
                             num_workers: int = 4,
                             image_size: int = 256) -> data.DataLoader:
    """创建三元组数据加载器"""
    triplet_dataset = TripletDataset(
        data_dir=data_dir,
        image_size=image_size
    )
    
    triplet_loader = data.DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return triplet_loader

if __name__ == "__main__":
    # 测试数据集
    data_dir = "./data"
    
    # 创建数据加载器
    train_loader, test_loader_d2s, test_loader_s2d = create_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,
        image_size=256
    )
    
    print("Testing train loader...")
    for batch in train_loader:
        print(f"Drone images shape: {batch['drone_img'].shape}")
        print(f"Satellite images shape: {batch['satellite_img'].shape}")
        print(f"Class IDs: {batch['class_id']}")
        break
    
    print("\nTesting test loader (drone to satellite)...")
    for batch in test_loader_d2s:
        print(f"Images shape: {batch['image'].shape}")
        print(f"Is query: {batch['is_query']}")
        print(f"Class IDs: {batch['class_id']}")
        break
    
    print("\nTesting triplet loader...")
    triplet_loader = create_triplet_dataloader(data_dir, batch_size=4, num_workers=0)
    for batch in triplet_loader:
        print(f"Anchor shape: {batch['anchor'].shape}")
        print(f"Positive shape: {batch['positive'].shape}")
        print(f"Negative shape: {batch['negative'].shape}")
        break
    
    print("Dataset test completed!") 