from torch.utils.data import Dataset


class DatasetBase(Dataset):

    def __init__(self):
        super(DatasetBase, self).__init__()

    def __getitem__(self, idx):
        pass
