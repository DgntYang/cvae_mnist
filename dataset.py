import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils import data
from utils import *


def to_ont_hot(y):
    s = y.shape[0] if isinstance(y, torch.Tensor) else len(y)
    y_ht = torch.zeros(s, 10)
    for i in range(s):
        y_ht[i, y[i]] = 1
    return y_ht

class MNIST(data.Dataset):
    def __init__(self, cfg, mode):
        self.data, self.target = load_data(mode, cfg.data.data_folder + '/MNIST/raw')
        self.data = self.data.float()/255.
        self.hot_target = to_ont_hot(self.target)

    def __getitem__(self, index):
        return self.data[index], self.hot_target[index]

    def __len__(self):
        return len(self.data)


def set_dataset(cfg, mode):
    if cfg.data.data_name == 'Mnist':

        train_data = torchvision.datasets.MNIST(root=cfg.data.data_folder, train=True,
                                                   transform=torchvision.transforms.ToTensor(), download=True,)
        test_data = torchvision.datasets.MNIST(root=cfg.data.data_folder, train=False,
                                             transform=torchvision.transforms.ToTensor(), download=True,)

        return MNIST(cfg, mode)

    else:
        raise RuntimeError("Invalid data type: {}!".format(cfg.data.data_name))





