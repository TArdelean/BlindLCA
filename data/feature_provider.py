from pathlib import Path

import torch
from PIL import Image
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class FeatureProvider:
    def __init__(self, cache_dir='cache', save_on_disk=True, save_in_memory=True, device='cpu'):
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.save_on_disk = save_on_disk
        self.save_in_memory = save_in_memory
        self.fe_tag = None
        self.fe = None
        self.img_paths = []
        self.cache_path = None
        self.load_image = None
        self.path_tokens = None
        self.memory = None

    def init(self, dataset):
        assert self.fe_tag is not None
        self.img_paths = dataset.img_paths
        self.cache_path = self.cache_dir / dataset.data_root.name / self.fe_tag / dataset.object_name

        def load_image(img_path):
            img = Image.open(img_path).convert('RGB')
            img = dataset.transform_x(img)
            return img[None].to(self.device)
        self.load_image = load_image
        self.path_tokens = dataset.path_tokens

        if self.save_on_disk:
            self._save_on_disk()
        if self.save_in_memory:
            self._save_in_memory()
        return self

    def get(self, indices: torch.tensor):
        if self.save_in_memory:
            return self.memory[indices]
        else:
            if hasattr(indices, "__iter__"):
                return torch.stack([self._get(ind) for ind in indices]).to(self.device)
            else:
                return self._get(indices).to(self.device)

    def _save_on_disk(self):
        if self.cache_path.exists():
            return self.cache_path
        self.cache_path.mkdir(parents=True)
        for idx in range(len(self.img_paths)):
            img_path = str(self.img_paths[idx])
            img = self.load_image(img_path)
            with torch.no_grad():
                one = self.fe(img)[0]

            obj_name, class_name, img_name = self.path_tokens(img_path)
            file_path = self.cache_path / f"{class_name}_{img_name.split('.')[0]}.pt"
            torch.save(one, file_path)

    def _save_in_memory(self):
        del self.memory  # Clean previous cache
        torch.cuda.empty_cache()
        memory = []
        for ind in range(len(self.img_paths)):
            memory.append(self._get(ind))
        self.memory = torch.stack(memory).to(self.device)

    def _get(self, ind):
        img_path = str(self.img_paths[ind])
        if self.save_on_disk:
            obj_name, class_name, img_name = self.path_tokens(img_path)
            file_path = self.cache_path / f"{class_name}_{img_name.split('.')[0]}.pt"
            return torch.load(file_path, map_location=self.device)
        else:
            img = self.load_image(img_path)
            with torch.no_grad():
                return self.fe(img)[0]


class WideFeatures(FeatureProvider):
    def __init__(self, version=1, keep=6, resize=512, cache_dir='cache',
                 save_on_disk=True, save_in_memory=True, device='cpu'):
        super(WideFeatures, self).__init__(cache_dir,
                                           save_on_disk=save_on_disk, save_in_memory=save_in_memory, device=device)
        self.fe_tag = f"Wide_ResNet50_2_Weights_V{version}_{keep}_{resize}"
        if version == 1:
            cnn = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif version == 2:
            cnn = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        else:
            raise Exception("Invalid version")
        cnn = cnn.eval().to(device)
        self.fe = torch.nn.Sequential(*list(cnn.children())[:6])


class FeaturesFromFolder(FeatureProvider):
    def __init__(self, fe_tag, cache_dir='cache', save_in_memory=True, device='cpu', **kwargs):
        super().__init__(cache_dir, save_on_disk=True, save_in_memory=save_in_memory, device=device)
        self.fe_tag = fe_tag
