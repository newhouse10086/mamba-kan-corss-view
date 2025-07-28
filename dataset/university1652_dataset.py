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
    支持University-1652标准数据集结构
    """
    def __init__(self, data_dir, h, w, erasing_p, color_jitter, is_train=True):
        super(UniversityDataset, self).__init__()
        self.transform = get_transforms(h, w, erasing_p, color_jitter)
        self.is_train = is_train
        self.data_dir = data_dir

        print(f"[UniversityDataset] Loading data from: {data_dir}")

        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # 定义子目录路径
        self.drone_dir = os.path.join(data_dir, 'drone')
        self.satellite_dir = os.path.join(data_dir, 'satellite')
        self.street_dir = os.path.join(data_dir, 'street')  # 添加street目录支持

        print(f"[UniversityDataset] Checking directories:")
        print(f"  - Drone dir: {self.drone_dir} (exists: {os.path.exists(self.drone_dir)})")
        print(f"  - Satellite dir: {self.satellite_dir} (exists: {os.path.exists(self.satellite_dir)})")
        print(f"  - Street dir: {self.street_dir} (exists: {os.path.exists(self.street_dir)})")

        # 初始化数据列表
        all_images = []
        all_labels = []

        # 加载Drone图像 (University-1652标准格式)
        drone_count = self._load_images_from_dir(self.drone_dir, all_images, all_labels, 'drone')

        # 加载Satellite图像 (University-1652标准格式)
        satellite_count = self._load_images_from_dir(self.satellite_dir, all_images, all_labels, 'satellite')

        # 可选：加载Street图像
        street_count = self._load_images_from_dir(self.street_dir, all_images, all_labels, 'street')

        print(f"[UniversityDataset] Loaded images:")
        print(f"  - Drone: {drone_count}")
        print(f"  - Satellite: {satellite_count}")
        print(f"  - Street: {street_count}")
        print(f"  - Total: {len(all_images)}")

        if len(all_images) == 0:
            print(f"[UniversityDataset] ERROR: No images found!")
            print(f"[UniversityDataset] Please check your data directory structure:")
            print(f"[UniversityDataset] Expected structure:")
            print(f"  {data_dir}/")
            print(f"    ├── drone/")
            print(f"    │   ├── 0001/")
            print(f"    │   │   ├── image1.jpg")
            print(f"    │   │   └── ...")
            print(f"    │   └── ...")
            print(f"    ├── satellite/")
            print(f"    │   ├── 0001/")
            print(f"    │   │   ├── 0001.jpg")
            print(f"    │   │   └── ...")
            print(f"    │   └── ...")
            print(f"    └── street/ (optional)")
            raise ValueError("No training images found. Please check data directory structure.")

        self.all_images = all_images
        self.all_labels = np.array(all_labels)

        # 用于RandomIdentitySampler
        self.pids = sorted(list(set(self.all_labels)))
        self.pid_dic = {pid: i for i, pid in enumerate(self.pids)}

        print(f"[UniversityDataset] Found {len(self.pids)} unique classes/buildings")

        # 过滤掉标签不存在于pid_dic中的样本
        valid_indices = [i for i, label in enumerate(self.all_labels) if label in self.pid_dic]
        self.all_images = [self.all_images[i] for i in valid_indices]
        self.all_labels = [self.all_labels[i] for i in valid_indices]

        print(f"[UniversityDataset] Final dataset size: {len(self.all_images)} images")

    def _load_images_from_dir(self, base_dir, all_images, all_labels, view_type):
        """
        从指定目录加载图像，支持University-1652标准格式
        """
        count = 0
        if not os.path.exists(base_dir):
            print(f"[UniversityDataset] Warning: {view_type} directory not found: {base_dir}")
            return count

        # 遍历类别目录 (如 0001, 0002, ...)
        for class_dir in sorted(os.listdir(base_dir)):
            class_path = os.path.join(base_dir, class_dir)
            if not os.path.isdir(class_path):
                continue

            try:
                # 类别ID必须是数字
                class_id = int(class_dir)
            except ValueError:
                print(f"[UniversityDataset] Skipping non-numeric directory: {class_dir}")
                continue

            # 加载该类别下的所有图像
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)

                    # 检查是否为图像文件
                    if self._is_image_file(img_file) and os.path.isfile(img_path):
                        all_images.append(img_path)
                        all_labels.append(class_id)
                        count += 1

        return count

    def _is_image_file(self, filename):
        """
        检查文件是否为图像文件
        """
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        return any(filename.lower().endswith(ext) for ext in img_extensions)
    
    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        path = self.all_images[index]
        label = self.all_labels[index]
        
        # 获取映射后的PID
        pid = self.pid_dic[label]
        
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
        return img, pid

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