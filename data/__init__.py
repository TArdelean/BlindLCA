import torch.utils.data

from .mvtec import MVTecDataset
from .mtd import MTDDataset
from .leaves import LeavesDataset


class DatasetProvider:
    def __init__(self, dataset_class, feature_provider, *args, **kwargs):
        self.dataset_class = dataset_class
        self.feature_provider = feature_provider
        self.args = args
        self.kwargs = kwargs

    def get_instances(self):
        for object_name in self.dataset_class.objects:
            yield self.dataset_class(object_name, *self.args, **self.kwargs)


class IndicesDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        super(IndicesDataset, self).__init__()
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, item):
        return self.tensor[item]


class CustomDataLoader:
    def __init__(self, feature_provider, dataset, *args, device='cuda:0', batch_size=4, **kwargs):
        self.dataset = dataset
        self.feature_provider = feature_provider.init(self.dataset)
        self.data_loader = torch.utils.data.DataLoader(dataset, *args, batch_size=batch_size, **kwargs)
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        self.iter = iter(self.data_loader)
        return self

    def __next__(self):
        data = next(self.iter)
        idx = data[-1]
        features = self.feature_provider.get(idx)
        return features, *data

    def feature_loader(self, shuffle=True, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        indices = IndicesDataset(torch.arange(0, len(self.dataset), device=self.device))
        data_loader = torch.utils.data.DataLoader(indices, batch_size=batch_size, shuffle=shuffle)
        for chunk in data_loader:
            features = self.feature_provider.get(chunk)
            yield features
