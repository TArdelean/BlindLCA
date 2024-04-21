import re

from .mvtec import MVTecDataset


class MTDDataset(MVTecDataset):
    objects = ['magnetic']

    def __init__(self, object_name, resize=512, data_root="datasets/MTD", **kwargs):
        super().__init__(object_name, resize, "test", False, data_root)
