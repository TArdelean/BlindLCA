from .mvtec import MVTecDataset


class LeavesDataset(MVTecDataset):
    objects = ['group_all_uniform']

    def __init__(self, object_name, resize=512, data_root="datasets/leaves", **kwargs):
        super().__init__(object_name, resize, "test", False, data_root)
