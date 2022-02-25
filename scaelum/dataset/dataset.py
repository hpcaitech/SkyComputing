#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import random

import torch
import torchvision
import torchvision.transforms as transforms
from scaelum.registry import DATASET
from torch.utils.data import Dataset


@DATASET.register_module
class RandomMlpDataset(Dataset):
    def __init__(self, num=1000, dim=1024):
        self.dim = dim
        self.data = torch.rand(num, dim)  # , 224, 224)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], random.randint(0, self.dim - 1)


@DATASET.register_module
class CIFAR10Dataset(Dataset):
    def __init__(self, mean, std, *args, **kwargs):
        transform_train = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.cifar10dataset = torchvision.datasets.CIFAR10(
            transform=transform_train, *args, **kwargs
        )

    def __len__(self):
        return self.cifar10dataset.__len__()

    def __getitem__(self, idx):
        return self.cifar10dataset.__getitem__(idx)
