#!/usr/bin/python3

import argparse
import os
import shutil
import time
import sys

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

from dir_lookup import imbalance_cfg
sys.path.insert(0, './train')

transform_train=[
            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            transforms.RandomRotation(10),     #Rotates the image to a specified angel
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            ]
####################################################
# classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
####################################################
def get_cifar10(batch_size, num_workers, 
        train=True, shuffle=True, is_gray=False, has_augmentation=False, clsid=0, keep_ratio=1.0):
    print("Running on CIFAR10")
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize  = transforms.Normalize(mean=[0], std=[1])
    # this is for cifar10-gray
    if is_gray:
        transform_list = [transforms.Grayscale(num_output_channels=1)]
    else:
        transform_list = []
    transform_list.extend([transforms.ToTensor(), normalize])
    if train:
        transform_train.extend(transform_list)
        trainsform_list = transform_train

    if has_augmentation:
        print("Using augmentation. Std deviation of the noise while testing/evaluation = " + str(stddev))
        transform_list.append(transforms.Lambda(lambda img: img + distbn.sample(img.shape)))

    data_loader = torch.utils.data.DataLoader(
                ImbalancedDataset(
                    datasets.CIFAR10(root='./data', 
                                 train=train, 
                                 transform=transforms.Compose(transform_list), 
                                 download=True),
                    clsid, keep_ratio),
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=num_workers, pin_memory=True)
    return data_loader

def get_cifar100(batch_size, num_workers, 
        train=True, shuffle=True,  has_augmentation=False):
    print("Running on CIFAR10")
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize  = transforms.Normalize(mean=[0], std=[1])
    transform_list = [transforms.ToTensor(), normalize]
    
    if train:
        transform_list = trainsform_train.extend(transform_list)
    
    if has_augmentation:
        print("Using augmentation. Std deviation of the noise while testing/evaluation = " + str(stddev))
        transform_list.append(transforms.Lambda(lambda img: img + distbn.sample(img.shape)))

    data_loader = torch.utils.data.DataLoader(
                ImbalancedDataset(
                    datasets.CIFAR100(root='./data', 
                                 train=train, 
                                 transform=transforms.Compose(transform_list), 
                                 download=True),
                    clsid, keep_ratio),
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=num_workers, pin_memory=True)
    return data_loader

def get_mnist(batch_size, num_workers, train=True, shuffle=True, has_augmentation=False):
    normalize = transforms.Normalize(mean=[0], std=[1]) #Images are already loaded in [0,1]
    transform_list = [transforms.ToTensor(), normalize]
    if has_augmentation:
        print("Using augmentation. Std deviation of the noise while testing/evaluation = " + str(stddev))
        transform_list.append(transforms.Lambda(lambda img: img + distbn.sample(img.shape)))
    else:
        print("No augmentation used in testing")
            
    data_loader = torch.utils.data.DataLoader(
                ImbalancedDataset(
                    datasets.MNIST(root='../data', 
                                train=train, 
                                transform=transforms.Compose(transform_list), 
                                download=True),
                    clsid, keep_ratio),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers, 
            pin_memory=True)
    
    return data_loader


def get_data(dataset, batch_size, num_workers, train=True, shuffle=False, has_augmentation=False):
    if imbalance_cfg is not None and train:
        clsid = imbalance_cfg.clsid
        keep_ratio = imbalance_cfg.keep_ratio
        print("===============================")
        print(f'imbalance_cfg: clsid={clsid}, keep_ratio={keep_ratio:.4f}')
        print("===============================")
    else:
        clsid = None
        keep_ratio = 1.0
    if(dataset == "CIFAR10-gray"):
        return get_cifar10(batch_size, num_workers, is_gray=True, 
                        train=train, shuffle=shuffle, has_augmentation=has_augmentation)
    elif(dataset == "CIFAR10-rgb"):
        return get_cifar10(batch_size, num_workers, is_gray=False,
                        train=train, shuffle=shuffle, has_augmentation=has_augmentation,
                        clsid = clsid, keep_ratio=keep_ratio)
    elif(dataset == "CIFAR100-rgb"):
        return get_cifar100(batch_size, num_workers, 
                        train=train, shuffle=shuffle, has_augmentation=has_augmentation)
    elif (dataset == "MNIST"):
        return get_mnist(batch_size, num_workers, 
                        train=train, shuffle=shuffle, has_augmentation=has_augmentation)
    else:
        print("Unknown datasets")
        sys.exit()

def stat_minist():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', default=2)
    ap.add_argument('--batch_size', default=30)
    ap.add_argument('--test_batch_size', default=30)
    args = ap.parse_args()

    train_data = get_mnist(args.batch_size, args.workers, train=True, shuffle=False)
    val_data = get_mnist(args.test_batch_size, args.workers, train=False, shuffle=False)
    max_s = 0
    min_s = 1000
    for i, (data, target) in enumerate(train_data):
        s = data.reshape([data.shape[0], -1]).sum(dim=1)
        if s.max() > max_s:
            max_s = s.max()
        if s.min() < min_s:
            min_s = s.min()
    print(f'On training: max_s={max_s}, min_s={min_s}')

    max_s = 0
    min_s = 1000
    for i, (data, target) in enumerate(val_data):
        s = data.reshape([data.shape[0], -1]).sum(dim=1)
        if s.max() > max_s:
            max_s = s.max()
        if s.min() < min_s:
            min_s = s.min()
    print(f'On eval: max_s={max_s}, min_s={min_s}')


class ImbalancedDataset(torch.utils.data.Dataset):
    def __init__(self, ds, clsid=0, keep_ratio=1.0):
        self.imb_cls = clsid
        self.ds = ds
        
        ids = np.array([ds[i][1] for i in range(len(self.ds))])
        if keep_ratio < 1.0:
            cls_ids = np.argwhere(ids == self.imb_cls).squeeze()
            np.random.seed(1)
            del_ids = np.random.choice(cls_ids, max(1, int(len(cls_ids) * (1 - keep_ratio))))
            self.new_ids = [i for i in ids if i not in del_ids]
        else:
            self.new_ids = ids

    def __len__(self):
        return len(self.new_ids)

    def __getitem__(self, idx):
        return self.ds[self.new_ids[idx]]



if __name__ == '__main__':
    stat_minist()
    #ds = get_data("CIFAR10-rgb", )
