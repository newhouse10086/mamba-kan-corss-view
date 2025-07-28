#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本用于将除了主干网络和训练脚本之外的其他文件同步为FSRA官方版本
"""

import os
import requests
import sys

def download_file(url, local_path):
    """下载文件"""
    try:
        print(f"正在下载 {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"已保存到 {local_path}")
        return True
    except Exception as e:
        print(f"下载失败 {url}: {e}")
        return False

def main():
    print("开始同步FSRA官方版本文件...")
    
    # FSRA官方仓库的原始文件基础URL
    base_url = "https://raw.githubusercontent.com/Dmmm1997/FSRA/main/"
    
    # 定义需要从官方仓库获取的文件列表
    files_to_sync = [
        "dataset/university1652_dataset.py",
        "utils/losses.py",
        "utils/lr_scheduler.py",
        "utils/metrics.py",
        "eval.py"
    ]
    
    # 下载所有文件
    success_count = 0
    for file_path in files_to_sync:
        url = base_url + file_path
        if download_file(url, file_path):
            success_count += 1
    
    print(f"\n同步完成! 成功下载 {success_count}/{len(files_to_sync)} 个文件")
    
    if success_count != len(files_to_sync):
        print("警告: 部分文件下载失败，请检查网络连接后重试")
        sys.exit(1)

if __name__ == "__main__":
    main()