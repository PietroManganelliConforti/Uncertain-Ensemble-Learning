import torch
import torchvision
import torchvision.transforms as transforms

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_train_test_loader(dataset_name: str, data_folder: str = './work/project/data', batch_size: int = 64, num_workers: int = 8):


    data_folder = os.path.join(data_folder, dataset_name)

    if dataset_name == "cifar10":

        n_cls = 10

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        # 2. Caricamento del dataset CIFAR-10
        train_set = torchvision.datasets.CIFAR10(root=data_folder,
                                                 train=True,
                                                 download=True,
                                                 transform=train_transform)
        
        test_set = torchvision.datasets.CIFAR10(root=data_folder,
                                                train=False,
                                                download=True,
                                                transform=test_transform)

        #classes are 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        
        print("CIFAR-10 sets created")
    

    elif dataset_name == "cifar100":

        n_cls = 100

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        train_set = datasets.CIFAR100(root=data_folder,
                                        download=True,
                                        train=True,
                                        transform=train_transform)

        test_set = datasets.CIFAR100(root=data_folder,
                                    download=True,
                                    train=False,
                                    transform=test_transform)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported from get_train_test_loader function")


    train_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    
    test_loader = DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    

    return train_loader, test_loader, n_cls
