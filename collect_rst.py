import os
import sys
from glob import glob
import numpy as np

from common.io import mkdir,mkpath
from common.time import today_, now_
from cfg import type_arch,large_types 
####################################################
# Track progress of the models: whether the training is done, whether the MILP is done
# Ouput a todo list for the unfinished tasks:
#       python track_progress.py cifar10-rgb tr 0  # check training
#       python track_progress.py cifar10-gray ap 0 # check MILP with preprocessing using all training samples 
####################################################

model_dir   = './model_dir'
rst_dir     = './results/'
cnt_rst     = 'counting_results/'
stb_neuron  = 'stable_neurons/'

# results of counting stable neurons with/w.o preprecessing
NOPRE       = 'results-no_preprocess'
ALLPRE      = 'results-preprocess_all'
PARTPRE     = 'results-preprocess_partial'

# Get experiment list for four datasets
p_mnist = './track_progress/track_mnist.txt'
p_cifar10_gray  = 'track_progress/track_cifar10-gray.txt'
p_cifar10_rgb = 'track_progress/track_cifar10-rgb.txt'
p_cifar100_rgb = 'track_progress/track_cifar100-rgb.txt'

mnist_track_list = open(p_mnist, 'r').readlines()
cifar10_gray_track_list = open(p_cifar10_gray + '.orig', 'r').readlines()
cifar10_rgb_track_list = open(p_cifar10_rgb + '.orig', 'r').readlines()
cifar100_rgb_track_list = open(p_cifar100_rgb + '.orig', 'r').readlines()

# the stable neurons from the training samples
def collect_stable_from_sample(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    rst_path = os.path.join(model_name, 'stable_neurons.npy')
    if not os.path.exists(rst_path):
        return '-','-'
    else:
        rst = np.load(rst_path, allow_pickle=True).item()
        return len(rst['stably_active']), len(rst['stably_inactive']) 


# check whether MILP with preprocessing all training samples is done
def collect_AP(model_name, tag=ALLPRE):
    dataset = os.path.basename(model_name).split('_')[1]
    rst_path = os.path.join(rst_dir, dataset, tag, cnt_rst, os.path.basename(model_name) + '.txt')
    if not os.path.exists(rst_path):
        return os.path.basename(model_name)
    rst = [l.strip().split('/')[-1] for l in open(rst_path, 'r').readlines()]
    for l in rst:
        if 'relaxation' in l:
            return l
    return os.path.basename(model_name)

dataset = sys.argv[1] # this can be 'mnist', 'cifar10-gray', 'cifar10-rgb', 'cifar100-rgb' 
action  = sys.argv[2] # this can be 'ap', 'np', 'pp' 

if dataset == 'mnist':
    track_list = mnist_track_list
    p_track = p_mnist
    dataset = 'MNIST'
elif dataset == 'cifar10-gray':
    track_list = cifar10_gray_track_list
    p_track = p_cifar10_gray
    dataset = 'CIFAR10-gray'
elif dataset == 'cifar10-rgb':
    track_list = cifar10_rgb_track_list
    p_track = p_cifar10_rgb
    dataset = 'CIFAR10-rgb'
elif dataset == 'cifar100-rgb':
    track_list = cifar100_rgb_track_list
    p_track = p_cifar100_rgb
    dataset = 'CIFAR100-rgb'
else:
    print('Unknown dataset')

path_rst = os.path.join(rst_dir, dataset, f'collected_results_{action}.txt')
f_rst   = open(path_rst, 'w')

f_rst.write('\n')
# example of tag format: TR-D, AP-D, NP-X 
for i,l in enumerate(track_list):
    arrs = l.strip().split('#')
    exp = arrs[-1]
    arch = os.path.basename(exp).split('_')[2]
    if action == 'ap':
        tag = ALLPRE
    elif action == 'np':
        tag = NOPRE
    elif action == 'pp':
        tag = PARTPRE
    else:
        print('Unkown action')

    pre_act, pre_inact=collect_stable_from_sample(exp)

    rst = collect_AP(exp, tag)
    f_rst.write(rst.strip() + ',' + str(pre_act) + ',' + str(pre_inact) + '\n')
      
f_rst.close() 
