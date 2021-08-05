#!/bin/bash

gdrivex=/uusoc/exports/scratch/xiny/software/gdrive

version=$1
cp results/MNIST/collected_results_ap.txt results/mnist.collected_results_ap.v${version}.txt
cp results/MNIST/collected_results_np.txt results/mnist.collected_results_np.v${version}.txt
cp results/CIFAR100-rgb/collected_results_ap.txt results/CIFAR100-rgb.collected_results_ap.v${version}.txt
cp results/CIFAR10-rgb/collected_results_ap.txt results/CIFAR10-rgb.collected_results_ap.v${version}.txt
cp results/CIFAR10-gray/collected_results_ap.txt results/CIFAR10-gray.collected_results_ap.v${version}.txt
cp ./results/CIFAR10-rgb/collected_results_old.txt ./results/CIFAR10-rgb.collected_results_old.v${version}.txt
cp ./results/CIFAR100-rgb/collected_results_old.txt ./results/CIFAR100-rgb.collected_results_old.v${version}.txt
$gdrivex upload results/CIFAR10-gray.collected_results_ap.v${version}.txt
$gdrivex upload results/CIFAR10-rgb.collected_results_ap.v${version}.txt
$gdrivex upload results/CIFAR100-rgb.collected_results_ap.v${version}.txt
$gdrivex upload results/mnist.collected_results_ap.v${version}.txt
$gdrivex upload results/mnist.collected_results_np.v${version}.txt
$gdrivex upload ./results/CIFAR10-rgb.collected_results_old.v${version}.txt
$gdrivex upload ./results/CIFAR100-rgb.collected_results_old.v${version}.txt
