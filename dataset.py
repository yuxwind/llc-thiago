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

sys.path.insert(0, './train')

def get_cifar10(args, is_gray=False, has_augmentation=False):
    print("Running on CIFAR10")
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize  = transforms.Normalize(mean=[0], std=[1])
    # this is for cifar10-gray
    if is_gray:
        transform_list = [transforms.Grayscale(num_output_channels=1)]
    else:
        transform_list = []
    transform_list.extend([transforms.ToTensor(), normalize])
    
    if has_augmentation:
        print("Using augmentation. Std deviation of the noise while testing/evaluation = " + str(stddev))
        transform_list.append(transforms.Lambda(lambda img: img + distbn.sample(img.shape)))

    val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root='./data', 
                                 train=False, 
                                 transform=transforms.Compose(transform_list), 
                                 download=True),
                batch_size=args.test_batch_size, 
                shuffle=False,
                num_workers=args.workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(root='./data', 
                                 train=True, 
                                 transform=transforms.Compose(transform_list), 
                                 download=True),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
    return val_loader, train_loader

def get_mnist(args, has_augmentation=False):
    normalize = transforms.Normalize(mean=[0], std=[1]) #Images are already loaded in [0,1]
    transform_list = [transforms.ToTensor(), normalize]
    if has_augmentation:
        print("Using augmentation. Std deviation of the noise while testing/evaluation = " + str(stddev))
        transform_list.append(transforms.Lambda(lambda img: img + distbn.sample(img.shape)))
    else:
        print("No augmentation used in testing")
            
    val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='../data', 
                           train=False, 
                           transform=transforms.Compose(transform_list), 
                           download=True),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers, 
            pin_memory=True)
    
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.Compose(transform_list), 
                            download=True),
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.workers, 
            pin_memory=True)
    return val_loader, train_loader


def stat_minist():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', default=2)
    ap.add_argument('--batch_size', default=30)
    ap.add_argument('--test_batch_size', default=30)
    args = ap.parse_args()

    train_data, val_data = get_mnist(args)
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

if __name__ == '__main__':
    stat_minist()
