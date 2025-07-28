#!/usr/bin/env python3
"""
测试脚本：验证FSRA-VMK修复是否正确
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dataset():
    """测试数据集是否正确返回drone和satellite图像"""
    print("🧪 测试数据集...")
    
    try:
        from dataset.university1652_dataset import University1652Dataset
        
        # 创建测试数据集
        dataset = University1652Dataset(
            data_dir="./data",
            mode='train',
            query_mode='drone_to_satellite',
            image_size=256,
            augment=False
        )
        
        if len(dataset) == 0:
            print("⚠️ 数据集为空，请检查数据路径")
            return False
        
        # 测试数据加载
        sample = dataset[0]
        print(f"✅ 数据集样本格式: {len(sample)} 个元素")
        
        if len(sample) == 4:
            drone_img, satellite_img, class_id, path = sample
            print(f"✅ Drone图像形状: {drone_img.shape}")
            print(f"✅ Satellite图像形状: {satellite_img.shape}")
            print(f"✅ 类别ID: {class_id}")
            return True
        else:
            print(f"❌ 数据格式错误，期望4个元素，得到{len(sample)}个")
            return False
            
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        return False

def test_model():
    """测试模型前向传播"""
    print("\n🧪 测试模型...")
    
    try:
        from models.vmamba_kan_fsra import FSRAVMambaKAN
        
        # 创建模型
        model = FSRAVMambaKAN(
            num_classes=10,  # 测试用小数值
            dim=256,
            num_layers=6,
            num_heads=8,
            image_size=256
        )
        
        # 测试前向传播
        test_input = torch.randn(4, 3, 256, 256)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ 模型输出格式: {type(output)}")
        if isinstance(output, dict):
            print(f"✅ Logits形状: {output['logits'].shape}")
            print(f"✅ Features形状: {output['features'].shape}")
            return True
        else:
            print(f"❌ 模型输出格式错误")
            return False
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def test_losses():
    """测试损失函数"""
    print("\n🧪 测试损失函数...")
    
    try:
        from utils.losses import CrossEntropyLabelSmooth, TripletLoss, ContrastiveLoss
        
        # 创建测试数据
        features = torch.randn(8, 256)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        logits = torch.randn(8, 10)
        
        # 测试ID损失
        id_loss = CrossEntropyLabelSmooth(num_classes=10)
        id_loss_value = id_loss(logits, labels)
        print(f"✅ ID Loss: {id_loss_value.item():.4f}")
        
        # 测试Triplet损失
        triplet_loss = TripletLoss()
        triplet_loss_value = triplet_loss(features, labels)
        print(f"✅ Triplet Loss: {triplet_loss_value.item():.4f}")
        
        # 测试Contrastive损失
        contrastive_loss = ContrastiveLoss(temperature=0.07)
        contrastive_loss_value = contrastive_loss(features, labels)
        print(f"✅ Contrastive Loss: {contrastive_loss_value.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        return False

def test_sampler():
    """测试采样器"""
    print("\n🧪 测试采样器...")
    
    try:
        from utils.sampler import RandomIdentitySampler
        
        # 创建测试标签
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] * 10  # 重复确保足够样本
        
        sampler = RandomIdentitySampler(
            labels=labels,
            batch_size=12,
            num_instances=4
        )
        
        print(f"✅ 采样器长度: {len(sampler)}")
        
        # 测试采样
        sample_iter = iter(sampler)
        batch_indices = []
        for i, idx in enumerate(sample_iter):
            batch_indices.append(idx)
            if i >= 11:  # 获取一个batch
                break
        
        batch_labels = [labels[idx] for idx in batch_indices]
        unique_labels = set(batch_labels)
        print(f"✅ Batch标签: {batch_labels}")
        print(f"✅ 唯一类别数: {len(unique_labels)}")
        
        return len(unique_labels) >= 2  # 至少有2个不同类别
        
    except Exception as e:
        print(f"❌ 采样器测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 开始测试FSRA-VMK修复...")
    
    tests = [
        ("数据集", test_dataset),
        ("模型", test_model), 
        ("损失函数", test_losses),
        ("采样器", test_sampler)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{status} {name}测试")
        except Exception as e:
            print(f"❌ {name}测试异常: {e}")
            results.append(False)
    
    print(f"\n📊 测试总结: {sum(results)}/{len(results)} 通过")
    
    if all(results):
        print("\n🎉 所有测试通过！现在可以开始训练了。")
        print("\n🚀 推荐训练命令:")
        print("python train.py \\")
        print("    --config configs/fsra_vmk_config.yaml \\")
        print("    --data_dir ./data \\")
        print("    --query_mode drone_to_satellite \\")
        print("    --lr 0.0005 \\")
        print("    --warmup_epochs 5 \\")
        print("    --triplet_loss_weight 0.1 \\")
        print("    --contrastive_loss_weight 0.0")
    else:
        print("\n⚠️ 部分测试失败，请检查相关组件。")

if __name__ == "__main__":
    main() 