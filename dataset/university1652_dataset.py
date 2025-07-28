import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

class RandomErasing(object):
    """
    官方FSRA仓库中的随机擦除数据增强
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                    img[1, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                    img[2, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                return img
        return img

def get_transforms(h, w, erasing_p, color_jitter):
    """
    获取与官方FSRA一致的图像变换
    """
    transform_list = [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
    ]
    if color_jitter:
        transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if erasing_p > 0:
        transform_list.append(RandomErasing(probability=erasing_p))
    
    return transforms.Compose(transform_list)

class UniversityDataset(Dataset):
    """
    重构后与官方FSRA一致的数据集类
    """
    def __init__(self, data_dir, h, w, erasing_p, color_jitter, is_train=True):
        super(UniversityDataset, self).__init__()
        self.transform = get_transforms(h, w, erasing_p, color_jitter)
        self.is_train = is_train
        
        self.drone_dir = os.path.join(data_dir, 'drone')
        self.satellite_dir = os.path.join(data_dir, 'satellite')
        
        # --- 修复：遍历子目录加载图像 ---
        drone_images = []
        satellite_images = []
        
        drone_class_dirs = sorted(os.listdir(self.drone_dir))
        for class_dir in drone_class_dirs:
            class_path = os.path.join(self.drone_dir, class_dir)
            if os.path.isdir(class_path):
                images = [os.path.join(class_dir, f) for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                drone_images.extend(images)

        satellite_class_dirs = sorted(os.listdir(self.satellite_dir))
        for class_dir in satellite_class_dirs:
            class_path = os.path.join(self.satellite_dir, class_dir)
            if os.path.isdir(class_path):
                images = [os.path.join(class_dir, f) for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                satellite_images.extend(images)
        
        self.drone_images = sorted(drone_images)
        self.satellite_images = sorted(satellite_images)
        # --- 修复结束 ---
        
        # 标签和相机ID
        self.labels = np.array([int(os.path.basename(img).split('_')[0]) for img in self.drone_images])
        self.cameras = np.array([int(os.path.basename(img).split('_')[1]) for img in self.drone_images])
        
        # 合并图像路径 (现在需要加上drone_dir和satellite_dir)
        self.all_images = [os.path.join(self.drone_dir, img) for img in self.drone_images] + \
                          [os.path.join(self.satellite_dir, img) for img in self.satellite_images]
        self.all_labels = np.concatenate([self.labels, self.labels])
        
        # 用于RandomIdentitySampler
        self.pids = list(set(self.all_labels))
        self.pid_dic = {pid: i for i, pid in enumerate(self.pids)}
    
    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        path = self.all_images[index]
        label = self.all_labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
        return img, self.pid_dic[label]

class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    """
    与官方FSRA一致的RandomIdentitySampler
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {pid: [] for pid in data_source.pids}
        for i, pid in enumerate(data_source.all_labels):
            self.index_dic[pid].append(i)
        
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __iter__(self):
        batch_idxs_dict = {}
        for pid in self.pids:
            idxs = self.index_dic[pid]
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid] = batch_idxs
                    break
        
        avai_pids = self.pids[:]
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid]
                final_idxs.extend(batch_idxs)
                avai_pids.remove(pid)
        
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length if hasattr(self, 'length') else len(self.data_source) 