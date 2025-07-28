import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class University1652_Dataset(data.Dataset):
    def __init__(self, data_dir, transforms, dataset, mode, sort=False):
        self.data_dir = data_dir
        self.transforms = transforms
        self.dataset = dataset
        self.mode = mode
        self.sort = sort
        self.ids = []
        self.image_paths = []
        self.image_labels = []
        self.cam_ids = []
        self.load_dataset()
        
    def load_dataset(self):
        if self.dataset == 'university':
            if self.mode == 'train':
                self.load_university_train()
            elif self.mode == 'query' or self.mode == 'gallery':
                self.load_university_eval()
        else:
            print('unknow dataset mode!')

    def load_university_train(self):
        # 处理University-1652数据集结构
        train_dir = os.path.join(self.data_dir, 'train')
        if not os.path.exists(train_dir):
            # 如果没有train子目录，直接使用data_dir
            train_dir = self.data_dir
            
        # Load the directory names (class IDs)
        satellite_dir = os.path.join(train_dir, 'satellite')
        drone_dir = os.path.join(train_dir, 'drone')
        
        # Check if directories exist
        if not os.path.exists(satellite_dir):
            satellite_dir = os.path.join(train_dir, 'gallery_satellite')
        if not os.path.exists(drone_dir):
            drone_dir = os.path.join(train_dir, 'gallery_drone')
            
        sub_dirs = [d for d in os.listdir(satellite_dir) if os.path.isdir(os.path.join(satellite_dir, d))]
        if self.sort:
            sub_dirs.sort()
        else:
            random.shuffle(sub_dirs)
        
        for id_name in sub_dirs:
            id_dir_satellite = os.path.join(satellite_dir, id_name)
            id_dir_drone = os.path.join(drone_dir, id_name)
            
            # Check if both directories exist
            if not os.path.exists(id_dir_satellite) or not os.path.exists(id_dir_drone):
                continue
                
            # Load satellite images
            satellite_images = [f for f in os.listdir(id_dir_satellite) if f.endswith('.jpg') or f.endswith('.png')]
            if self.sort:
                satellite_images.sort()
            for img_name in satellite_images:
                self.image_paths.append(os.path.join(id_dir_satellite, img_name))
                self.image_labels.append(int(id_name))
                self.cam_ids.append(0)  # Satellite camera ID
                self.ids.append(int(id_name))
            
            # Load drone images
            drone_images = [f for f in os.listdir(id_dir_drone) if f.endswith('.jpg') or f.endswith('.png')]
            if self.sort:
                drone_images.sort()
            for img_name in drone_images:
                self.image_paths.append(os.path.join(id_dir_drone, img_name))
                self.image_labels.append(int(id_name))
                self.cam_ids.append(1)  # Drone camera ID
                self.ids.append(int(id_name))

    def load_university_eval(self):
        if self.mode == 'query':
            if 'query_drone' in os.listdir(self.data_dir):
                data_type = 'query_drone'
            else:
                data_type = 'drone'  # fallback
            cam_id = 1
        elif self.mode == 'gallery':
            if 'gallery_satellite' in os.listdir(self.data_dir):
                data_type = 'gallery_satellite'
            else:
                data_type = 'satellite'  # fallback
            cam_id = 0
        
        data_path = os.path.join(self.data_dir, data_type)
        # Fallback to test directory if needed
        if not os.path.exists(data_path):
            test_dir = os.path.join(self.data_dir, 'test')
            if os.path.exists(test_dir):
                data_path = os.path.join(test_dir, data_type)
        
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} does not exist")
            return
            
        sub_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        if self.sort:
            sub_dirs.sort()
            
        for id_name in sub_dirs:
            id_dir = os.path.join(data_path, id_name)
            if not os.path.exists(id_dir):
                continue
            images = [f for f in os.listdir(id_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if self.sort:
                images.sort()
            for img_name in images:
                self.image_paths.append(os.path.join(id_dir, img_name))
                self.image_labels.append(int(id_name))
                self.cam_ids.append(cam_id)
                self.ids.append(int(id_name))

    def __getitem__(self, index):
        path = self.image_paths[index]
        label = self.image_labels[index]
        cam_id = self.cam_ids[index]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label, cam_id, index

    def __len__(self):
        return len(self.image_paths)

    def get_ids(self):
        return self.ids

    def get_cam_ids(self):
        return self.cam_ids

def make_dataset(dataset, data_dir, height, width, batch_size, workers, erasing_p, color_jitter, train_all=False, sort=False):
    # Transforms for training
    train_transform_list = [
        transforms.Resize((height, width), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((height, width)),
        transforms.RandomHorizontalFlip(),
    ]
    if color_jitter:
        train_transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if erasing_p > 0:
        train_transform_list.append(RandomErasing(probability=erasing_p, mean=[0.0, 0.0, 0.0]))
    
    train_transform = transforms.Compose(train_transform_list)
    
    # Transforms for testing
    test_transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_set = University1652_Dataset(data_dir, train_transform, dataset, 'train', sort=sort)
    query_set = University1652_Dataset(data_dir, test_transform, dataset, 'query', sort=sort)
    gallery_set = University1652_Dataset(data_dir, test_transform, dataset, 'gallery', sort=sort)
    
    # Create dataloaders
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    query_loader = data.DataLoader(
        query_set,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True
    )
    
    gallery_loader = data.DataLoader(
        gallery_set,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, query_loader, gallery_loader, len(train_set), len(query_set), len(gallery_set)