#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据目录结构检查脚本
用于诊断University-1652数据集的目录结构问题
"""

import os
import argparse

def check_data_structure(data_dir):
    """
    检查数据目录结构是否符合University-1652标准
    """
    print(f"=== 检查数据目录: {data_dir} ===")
    
    if not os.path.exists(data_dir):
        print(f"❌ 错误: 数据目录不存在: {data_dir}")
        return False
    
    print(f"✅ 数据目录存在: {data_dir}")
    
    # 检查子目录
    subdirs = ['drone', 'satellite', 'street']
    found_subdirs = []
    
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.exists(subdir_path):
            found_subdirs.append(subdir)
            print(f"✅ 找到子目录: {subdir}/")
            
            # 检查类别目录
            class_dirs = []
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                if os.path.isdir(item_path):
                    try:
                        class_id = int(item)
                        class_dirs.append(item)
                    except ValueError:
                        print(f"⚠️  警告: 发现非数字目录名: {subdir}/{item}")
            
            if class_dirs:
                print(f"   📁 找到 {len(class_dirs)} 个类别目录")
                print(f"   📁 类别范围: {min(class_dirs)} - {max(class_dirs)}")
                
                # 检查前几个类别的图像文件
                sample_classes = sorted(class_dirs)[:3]
                for class_dir in sample_classes:
                    class_path = os.path.join(subdir_path, class_dir)
                    images = []
                    for f in os.listdir(class_path):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            images.append(f)
                    print(f"   🖼️  类别 {class_dir}: {len(images)} 张图像")
                    if images:
                        print(f"      示例文件: {images[0]}")
            else:
                print(f"   ❌ {subdir}/ 目录下没有找到有效的类别目录")
        else:
            print(f"❌ 缺少子目录: {subdir}/")
    
    if not found_subdirs:
        print("\n❌ 错误: 没有找到任何标准子目录 (drone, satellite, street)")
        print("\n💡 建议的目录结构:")
        print(f"{data_dir}/")
        print("├── drone/")
        print("│   ├── 0001/")
        print("│   │   ├── image1.jpg")
        print("│   │   └── ...")
        print("│   ├── 0002/")
        print("│   └── ...")
        print("├── satellite/")
        print("│   ├── 0001/")
        print("│   │   └── 0001.jpg")
        print("│   ├── 0002/")
        print("│   └── ...")
        print("└── street/ (可选)")
        print("    ├── 0001/")
        print("    └── ...")
        return False
    
    print(f"\n✅ 数据结构检查完成，找到 {len(found_subdirs)} 个有效子目录")
    return True

def suggest_solutions(data_dir):
    """
    提供解决方案建议
    """
    print("\n=== 解决方案建议 ===")
    
    # 检查是否有其他可能的数据目录
    if os.path.exists(data_dir):
        print("1. 检查当前目录下的所有内容:")
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")
    
    print("\n2. 可能的解决方案:")
    print("   a) 确保已正确下载University-1652数据集")
    print("   b) 检查数据集解压路径是否正确")
    print("   c) 确认训练数据路径参数 --data_dir 是否正确")
    print("   d) 如果使用自定义数据，请按照标准格式组织目录结构")
    
    print("\n3. 标准的University-1652数据集应该包含:")
    print("   - train/ 目录 (用于训练)")
    print("   - test/ 目录 (用于测试)")
    print("   - 每个目录下都有 drone/, satellite/, street/ 子目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='检查University-1652数据集目录结构')
    parser.add_argument('--data_dir', default='./data/University-123/train', 
                       help='数据目录路径')
    
    args = parser.parse_args()
    
    print("University-1652 数据集结构检查工具")
    print("=" * 50)
    
    success = check_data_structure(args.data_dir)
    
    if not success:
        suggest_solutions(args.data_dir)
        
        # 检查一些常见的替代路径
        alternative_paths = [
            './data/University-1652/train',
            './data/train',
            './University-1652/train',
            './train'
        ]
        
        print(f"\n4. 检查常见的替代路径:")
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"   ✅ 找到可能的数据目录: {alt_path}")
                print(f"   💡 尝试使用: --data_dir {alt_path}")
            else:
                print(f"   ❌ 不存在: {alt_path}")
