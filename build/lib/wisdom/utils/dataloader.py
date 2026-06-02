# wisdom/utils/dataloader.py
import os
import urllib
import urllib.request
import json

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

# This is for forming a custom dataset
class SelectorDataset(torch.utils.data.Dataset):
    def __init__(self, image_dataset, layer_info, attribution_labels, attribution_methods):
        super(SelectorDataset, self).__init__()
        self.image_dataset = image_dataset
        self.layer_info = layer_info
        self.attribution_labels = attribution_labels
        # Map method to index
        self.method_to_idx = {method: idx for idx, method in enumerate(attribution_methods)}  

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Get image and label from the original dataset
        image, _ = self.image_dataset[idx]
        
        # Get layer information for the current sample
        layer_info = self.layer_info[idx]
        
        # Convert attribution method to index
        attribution_label = self.method_to_idx[self.attribution_labels[idx]]
        
        return image, layer_info, torch.tensor(attribution_label, dtype=torch.long)

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets, datasets['train'], datasets['val']

def data_loader(root, batch_size=256, workers=1, pin_memory=True, shuffle=False):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_dataset, val_dataset

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets

# Load the ImageNet dataset
def load_ImageNet(batch_size=32, root='./datasets/ImageNet', num_workers=2, use_val=False, label_path='./datasets/imagenet_labels.json'):
    
    val_path = os.path.join(root, 'val/')
    
    if not os.path.exists(label_path):
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        urllib.request.urlretrieve(url, "imagenet_labels.json")
        label_path = "imagenet_labels.json"

    # Load the labels from the JSON file
    with open(label_path) as f:
        classes = json.load(f)
    
    if use_val:
        # Optional: use val_dataset as the training dataset for shorter training time
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
        datasets, train_dataset, val_dataset  = train_val_dataset(val_dataset, val_split=0.25)
        dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                       for x in ['train', 'val']}
        trainloader = dataloaders['train']
        testloader = dataloaders['val']
    else:
        trainloader, testloader, train_dataset, val_dataset = data_loader(root, batch_size, num_workers, True)

    return trainloader, testloader, train_dataset, val_dataset, classes

#  Load the CIFAR-10 dataset
def load_CIFAR(batch_size=32, root='./datasets', shuffle=True):

    transform = transforms.Compose([
         transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, train_dataset, test_dataset, classes


#  Load the MNIST dataset
def load_MNIST(batch_size=32, root='./datasets', channel_first=False, train_all=False):
    # transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    
    transform_list = [
        transforms.Resize(32),  # Upscale from 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    if channel_first:
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))  # If you want 3 channels
    transform = transforms.Compose(transform_list)

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)
    
    if train_all:
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    return train_loader, test_loader, train_dataset, test_dataset, classes


def get_data(dataset_name, batch_size, data_path):
    ### Dataset settings
    if dataset_name == 'cifar10':
        trainloader, testloader, train_dataset, test_dataset, classes = load_CIFAR(batch_size=batch_size, root=data_path, shuffle=True)
    elif dataset_name == 'mnist':
        trainloader, testloader, train_dataset, test_dataset, classes = load_MNIST(batch_size=batch_size, root=data_path)
    elif dataset_name == 'imagenet':
        trainloader, testloader, train_dataset, test_dataset, classes = load_ImageNet(batch_size=batch_size, 
                                                         root=data_path + '/ImageNet', 
                                                         num_workers=2, 
                                                         use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    
    return trainloader, testloader, train_dataset, test_dataset, classes