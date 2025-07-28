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
from models.vmamba_kan_fsra import VMambaKANFSRA, VisionMambaEncoder, MambaBlock
from dataset.university1652_dataset import UniversityDataset, RandomIdentitySampler, get_transforms
from utils.losses import CrossEntropyLabelSmooth, TripletLoss
import yaml
import math
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- Argparse ---
def get_opts():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids')
    parser.add_argument('--name', default='fsra_vmk', type=str, help='output model name')
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
    
    opt = parser.parse_args()
    return opt

# --- Main Training Loop ---
def train_model(model, criterion_id, criterion_tri, optimizer, scheduler, dataloaders, num_epochs):
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train']:
            model.train(True)
            running_loss = 0.0
            running_corrects = 0.0
            
            for data in tqdm(dataloaders['train']):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
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

    # Dataloaders
    image_datasets = {
        'train': UniversityDataset(opt.data_dir, opt.h, opt.w, opt.erasing_p, opt.color_jitter)
    }
    
    # --- 断言：检查数据集是否为空 ---
    assert len(image_datasets['train']) > 0, \
        f"Error: Training dataset is empty. Please check the data directory: {opt.data_dir}"
    
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            sampler=RandomIdentitySampler(image_datasets['train'], opt.batchsize, 4),
            batch_size=opt.batchsize,
            num_workers=8
        )
    }
    
    # Model
    num_classes = len(image_datasets['train'].pids)
    model = VMambaKANFSRA(num_classes, opt.block, opt.backbone).to(device)

    # Loss Functions
    criterion_id = CrossEntropyLabelSmooth(num_classes)
    criterion_tri = TripletLoss(margin=0.3)
    
    # Optimizer and Scheduler
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
    model = train_model(model, criterion_id, criterion_tri, optimizer, exp_lr_scheduler, dataloaders, num_epochs=opt.num_epochs) 