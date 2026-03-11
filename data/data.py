import PIL
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from config import *
import torch

class FERGMultiLabelDataset(Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot
        self.transform = transform
        self.samples = []

        # 定义身份映射
        self.id_to_idx = {'aia': 0, 'bonnie': 1, 'jules': 2, 'malcolm': 3, 'mery': 4, 'ray': 5}
        # 定义表情映射 (来源: FERG 有 7 种 cardinal expressions)
        self.exp_to_idx = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
            'neutral': 4, 'sadness': 5, 'surprise': 6
        }

        # 遍历第一层：身份文件夹 (aia, bonnie...)
        for id_name in os.listdir(dataroot):
            id_path = os.path.join(dataroot, id_name)
            if not os.path.isdir(id_path): continue

            # 遍历第二层：表情文件夹 (aia_anger, aia_joy...)
            for exp_folder_name in os.listdir(id_path):
                exp_path = os.path.join(id_path, exp_folder_name)
                if not os.path.isdir(exp_path): continue

                # 提取表情关键字，例如从 "aia_anger" 提取 "anger"
                # 假设格式总是 "角色名_表情名"
                current_exp_type = exp_folder_name.split('_')[-1]

                # 获取图像文件
                for img_name in os.listdir(exp_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_full_path = os.path.join(exp_path, img_name)
                        # 保存路径、身份索引、表情索引
                        self.samples.append((
                            img_full_path,
                            self.id_to_idx[id_name.lower()],
                            self.exp_to_idx[current_exp_type.lower()]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, id_label, exp_label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (PIL.UnidentifiedImageError, OSError):
            # 如果当前图片损坏，随机选一张别的图片返回，或者递归调用
            print(f"Warning: Skipping corrupted image {img_path}")
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
        if self.transform:
            image = self.transform(image)

        return image, id_label, exp_label


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#  实例化 Dataset

ferg_dataset = FERGMultiLabelDataset(dataroot=dataroot, transform=transform)

#  创建 DataLoader

dataloader = DataLoader(ferg_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # 4. 验证读取结果
    real_batch = next(iter(dataloader))
    images, id_labels, exp_labels = real_batch
    print(f"图像 Batch 形状: {images.size()}")  # 应为 [9, 10]
    print(f"身份标签样例: {id_labels[:5]}")  # 5 个角色的索引
    print(f"表情标签样例: {exp_labels[:5]}")  # 5 个表情的索引
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title('test images')
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
                            (1, 2, 0)))
    plt.show()

