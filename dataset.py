#!/usr/bin/python3

"""
"""

import argparse
import os
import shutil
import time
import sys

import torch
import numpy as np

sys.path.insert(0, './train')

def get_cifar10(is_gray=False, has_augmentation=False):
    print("Running on CIFAR10")
    input_dim  = 1024
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

def get_mnist(has_augmentation=False)
    input_dim = 784
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
