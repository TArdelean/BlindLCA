from pathlib import Path

import torch
from PIL import Image
import tifffile as tiff

from data import MVTecDataset
from torchvision import transforms as T


class MaskMemory:
    def __init__(self, masks_root, dataset, device):
        self.masks_root = Path(masks_root)
        self.is_tiff = 'tiff' in self.masks_root.name
        self.ss_transform = T.Compose([T.ToTensor()])
        self.memory = self.load(dataset).to(device)

    def load(self, dataset: MVTecDataset):
        memory = []
        for img_path in dataset.img_paths:
            obj_name, class_name, img_name = dataset.path_tokens(img_path)
            if self.is_tiff:
                self_path = self.masks_root / obj_name / "test" / class_name / (img_name.split('.')[0] + '.tiff')
                ss_target = tiff.imread(self_path)
            else:
                self_path = self.masks_root / obj_name / class_name / (img_name.split('.')[0] + '.jpg')
                ss_target = Image.open(self_path)
            ss_target = (self.ss_transform(ss_target)).float()
            memory.append(ss_target)
        return torch.stack(memory)

    def get(self, indices: torch.tensor):
        return self.memory[indices]
