#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSRA-Mamba训练脚本
仅使用Vision Mamba作为主干网络的跨视角图像匹配模型训练

基于您的分析，实现FSRA-Mamba: Vision Mamba Network
- O(n)线性复杂度的Vision Mamba编码器
- 保持FSRA的其他组件不变
"""

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
# 修改导入语句，使用新的模型
from models.fsra_mamba import FSMambaFSRA, VisionMambaEncoder, MambaBlock
# 修改导入语句，使用FSRA官方数据集
from dataset.university1652_dataset import make_dataset
from utils.losses import CrossEntropyLabelSmooth, TripletLoss
import yaml
import math
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- Argparse ---
def get_opts():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids')
    parser.add_argument('--name', default='fsra_mamba', type=str, help='output model name')
    parser.add_argument('--data_dir', default='./data/train', type=str, help='training dir path')
    parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
    parser.add_argument('--h', default=256, type=int, help='height')
    parser.add_argument('--w', default=256, type=int, help='width')
    parser.add_argument('--erasing_p', default=0.5, type=float, help='random erasing probability')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=120, type=int, help='num_epochs')
    parser.add_argument('--warm_epoch', default=10, type=int, help='warm up epoch')
    parser.add_argument('--block', default=1, type=int, help='number of blocks')
    parser.add_argument('--backbone', default='VIT-S', type=str, help='backbone: VIT-S or VIT-B')
    parser.add_argument('--triplet_loss_weight', default=1.0, type=float, help='triplet loss weight')
    parser.add_argument('--dataset', default='university', type=str, help='dataset name')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    
    opt = parser.parse_args()
    return opt

# --- Main Training Loop ---
def train_model(model, criterion_id, criterion_tri, optimizer, scheduler, dataloaders, num_epochs, opt):
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train']:
            model.train(True)
            running_loss = 0.0
            running_corrects = 0.0
            
            for data in tqdm(dataloaders['train']):
                inputs, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs, features = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    
                    id_loss = criterion_id(outputs, labels)
                    triplet_loss = criterion_tri(features, labels)
                    loss = id_loss + opt.triplet_loss_weight * triplet_loss
                    
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders['train'].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            scheduler.step()
        
        # Save model
        if (epoch + 1) % 10 == 0:
            save_network(model, epoch)
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model

# --- Helper Functions ---
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./checkpoints', opt.name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.to(device)

if __name__ == '__main__':
    opt = get_opts()
    
    # Setup device
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = [int(id) for id in str_ids]
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataloaders - 使用FSRA官方数据集加载方式
    train_loader, _, _, train_set_length, _, _ = make_dataset(
        dataset=opt.dataset,
        data_dir=opt.data_dir,
        height=opt.h,
        width=opt.w,
        batch_size=opt.batchsize,
        workers=opt.workers,
        erasing_p=opt.erasing_p,
        color_jitter=opt.color_jitter,
        train_all=False,
        sort=False
    )
    
    dataloaders = {
        'train': train_loader
    }
    
    # 获取类别数量 - 通过遍历数据集计算
    all_labels = []
    for data in train_loader:
        all_labels.extend(data[1].tolist())
    
    num_classes = len(set(all_labels))
    print(f"Number of classes: {num_classes}")

    # Model
    model = FSMambaFSRA(num_classes, opt.block, opt.backbone).to(device)

    # Loss Functions (保持与FSRA相同)
    criterion_id = CrossEntropyLabelSmooth(num_classes)
    criterion_tri = TripletLoss(margin=0.3)
    
    # Optimizer and Scheduler (保持与FSRA相同)
    ignored_params = list(map(id, model.backbone.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    
    optimizer = torch.optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.backbone.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    # Create checkpoints dir
    dir_name = os.path.join('./checkpoints', opt.name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    
    # Train
    model = train_model(model, criterion_id, criterion_tri, optimizer, exp_lr_scheduler, dataloaders, num_epochs=opt.num_epochs, opt=opt)