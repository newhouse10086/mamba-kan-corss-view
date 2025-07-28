#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSRA-Mamba评估脚本
评估训练好的Vision Mamba网络模型在University-1652数据集上的性能

支持的评估模式：
1. 标准评估：计算Recall@K、mAP、CMC等指标
2. 可视化评估：生成检索结果可视化
3. 性能分析：分析推理速度和内存使用
"""

import os
import sys
import time
import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
from utils.metrics import evaluate

######################################################################
# Options
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_dir',default='./data/University-1652/test',type=str, help='test data path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--name', default='fsra_mamba', type=str, help='save model path')
parser.add_argument('--checkpoint', default='net_59.pth', type=str, help='checkpoint')
parser.add_argument('--test_dir', default='./data/University-1652/test', type=str, help='test_dir')
parser.add_argument('--mode', default='1', type=str, help='drone to satellite(1), satellite to drone(2)')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.mode == '1':
    data_dir = opt.test_dir + '/gallery_satellite'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=False, num_workers=8) for x in ['gallery', 'query']}
else:
    data_dir = opt.test_dir + '/gallery_drone'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=False, num_workers=8) for x in ['gallery', 'query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./checkpoints', opt.name, opt.checkpoint)
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Extract feature
# ----------------------
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs, _ = model(input_img) 
            ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
    return labels

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.mode == '1':
    class_num = 701
else:
    class_num = 701

from models.fsra_mamba import FSMambaFSRA
model = FSMambaFSRA(class_num, block=1, backbone='VIT-S')

model = load_network(model)
model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
query_feature = extract_feature(model,dataloaders['query'])

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'query_f':query_feature.numpy()}
scipy.io.savemat('pytorch_result.mat',result)

print(opt.test_dir)
result = evaluate(gallery_feature, query_feature)
print(result)
