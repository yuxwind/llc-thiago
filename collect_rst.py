import os
import sys
from glob import glob
import numpy as np
import torch

from common.io import mkdir,mkpath
from common.time import today_, now_
from cfg import type_arch,large_types 
from dir_lookup import model_root as model_dir
from dir_lookup import stb_root as rst_dir
from dir_lookup import cnt_rst, stb_neuron

####################################################
# Track progress of the models: whether the training is done, whether the MILP is done
# Ouput a todo list for the unfinished tasks:
#       python track_progress.py cifar10-rgb tr 0  # check training
#       python track_progress.py cifar10-gray ap 0 # check MILP with preprocessing using all training samples 
####################################################

#model_dir   = './model_dir'
#rst_dir     = './results/'
##rst_dir     = 'results-restrict_input/'
#cnt_rst     = 'counting_results/'
#stb_neuron  = 'stable_neurons/'

# results of counting stable neurons with/w.o preprecessing
NOPRE       = 'results-no_preprocess'
ALLPRE      = 'results-preprocess_all'
PARTPRE     = 'results-preprocess_partial'
OLD         = 'results-old-approach'

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
    print('----', rst)
    for l in rst:
        #if 'relaxation' in l:
        if tag == OLD:
            if 'neuron' in l:
                return l
        else:
            if 'network' in l:
                return l
    return os.path.basename(model_name)
    #import pdb;pdb.set_trace()
    #return ''

def collect_EVAL(model_name):
    root = model_name.strip('./').split('/')[0]
    model_name = model_name.replace(root, model_dir.strip('./'))
    rst_path = os.path.join(model_name, 'eval.txt')
    if os.path.exists(rst_path):
        rst = [l.strip() for l in open(rst_path, 'r').readlines()]
        rst = rst[-1] 
    else:
        rst = model_name + ',-,-,-'
    m_pruned_ckp_path = os.path.join(model_name, 'magnitude_pruned_checkpoint_120.tar')
    if os.path.exists(m_pruned_ckp_path):
        ckp = torch.load(m_pruned_ckp_path)
        prune_ratio_weight = ckp['prune_ratio_weight']
        prune_ratio_neuron = ckp['prune_ratio_neuron']
        arch = os.path.basename(model_name).split('_')[2]
        rst += f',{arch},{prune_ratio_neuron},{prune_ratio_weight}'
        
    return rst

dataset = sys.argv[1] # this can be 'mnist', 'cifar10-gray', 'cifar10-rgb', 'cifar100-rgb' 
action  = sys.argv[2] # this can be 'ap', 'np', 'pp', 'eval' 

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
    if l == '\n' or l == '':
        continue
    arrs = l.strip().split('#')
    exp = arrs[-1]
    arch = os.path.basename(exp).split('_')[2]
    if action == 'ap':
        tag = ALLPRE
    elif action == 'np':
        tag = NOPRE
    elif action == 'pp':
        tag = PARTPRE
    elif action == 'old':
        tag = OLD
    elif action == 'eval':
        tag = 'eval'
    else:
        print('Unkown action')

    if action == 'eval':
        rst = collect_EVAL(exp)
        print(rst)
        f_rst.write(rst.strip() + '\n')
    else:
        pre_act, pre_inact=collect_stable_from_sample(exp)

        rst = collect_AP(exp, tag)
        print(rst)
        #import pdb;pdb.set_trace()
        if len(rst.split(',')) > 2:
            f_rst.write(rst.strip() + ',' + str(pre_act) + ',' + str(pre_inact) + '\n')
        else:
            f_rst.write(rst.strip() + '\n')
        

print(path_rst)
f_rst.close() 
