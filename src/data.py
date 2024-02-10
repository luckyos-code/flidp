from typing import List

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10


def load_datasets(dataset_chache: str, client_sizes: List[float], val_size: float = 0.1):
    transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    trainset = CIFAR10(dataset_chache, train=True, download=True, transform=transform)
    testset = CIFAR10(dataset_chache, train=False, download=True, transform=transform)

    client_sizes = [int(len(trainset) * cs) for cs in client_sizes]
    datasets = random_split(trainset, client_sizes, torch.Generator().manual_seed(42))

    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = int(len(ds) * val_size)
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader