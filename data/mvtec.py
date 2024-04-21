import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class MVTecDataset(Dataset):

    textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
    nt = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    objects = [*textures]

    def __init__(self, object_name='carpet', resize=512, split="test", exclude_combined=False,
                 data_root="datasets/mvtec_anomaly_detection"):
        self.data_root = data_root
        self.object_name = object_name
        self.split = split
        self.resize = resize
        self.exclude_combined = exclude_combined
        self.data_root = Path(data_root)

        self.img_paths, self.labels, self.mask_paths = self.load_dataset_folder()
        self.label_names = []
        self.label_ids = self.build_label_ids()

        self.transform_x = T.Compose([T.Resize((resize, resize), T.InterpolationMode.BILINEAR),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize((resize, resize), T.InterpolationMode.BILINEAR),
                                         T.ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, mask = str(self.img_paths[idx]), self.labels[idx], self.mask_paths[idx]

        img = Image.open(img_path).convert('RGB')
        hw = img.size
        img = self.transform_x(img)

        if mask is None:
            mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return img, label, mask, hw, img_path, idx

    def build_label_ids(self):
        mapping = {}
        self.label_names = []
        ids = []
        last = 0
        for label in self.labels:
            if label in mapping:
                ids.append(mapping[label])
            else:
                self.label_names.append(label)
                mapping[label] = last
                ids.append(last)
                last += 1
        return np.array(ids)

    def load_dataset_folder(self):
        img_paths, labels, mask_paths = [], [], []

        test_dir = self.data_root / self.object_name / self.split
        gt_dir = self.data_root / self.object_name / 'ground_truth'

        for defect_dir in test_dir.iterdir():
            label = defect_dir.name
            if label == "combined" and self.exclude_combined:
                continue
            paths = sorted(defect_dir.iterdir(), key=lambda x: int(re.findall(r'\d+', x.stem)[-1]))

            length = len(paths)
            img_paths.extend(paths)
            labels.extend([label] * length)

            if label == 'good':
                mask_paths.extend([None] * length)
            else:
                gt_paths = sorted((gt_dir / label).iterdir(), key=lambda x: int(re.findall(r'\d+', x.stem)[-1]))
                mask_paths.extend(gt_paths)

        return img_paths, labels, mask_paths

    @staticmethod
    def path_tokens(img_path):
        obj_name, _, class_name, img_name = str(img_path).split('/')[-4:]
        return obj_name, class_name, img_name
