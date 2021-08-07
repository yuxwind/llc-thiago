#!/usr/bin/python

import os

from common.io import mkpath, mkdir

NOPRE       = 'results-no_preprocess'
ALLPRE      = 'results-preprocess_all'
PARTPRE     = 'results-preprocess_partial'
OLD         = 'results-old-approach'
#stb_root    = './results/'
stb_root    = './results-restrict_input/'
cnt_rst     = 'counting_results/'
stb_neuron  = 'stable_neurons/'
model_root  = './model_dir/'
def get_stb_dir(dataset, tag):
    stb_dir = mkdir(os.path.join(stb_root, dataset, tag, cnt_rst))
    return stb_dir

def parse_exp_name(exp):
    _, dataset, arch, regr, run = exp.strip().split('_')
    return dataset, arch, regr, run

def get_MILP_stb_path(tag, exp):
    exp_name = os.path.basename(exp)
    dataset, _, _, _ = parse_exp_name(exp_name)
    stable_neurons_path = mkpath(os.path.join(
            stb_root, dataset, tag, stb_neuron, exp_name + '.npy'))
    return stable_neurons_path

def get_train_stb_path(exp):
    exp_name = os.path.basename(exp)
    dataset, _,_,_ = parse_exp_name(exp_name)
    train_stb_path = os.path.join(model_dir, dataset, exp_name, 'stable_neurons.npy')
    return train_stb_path

def get_MILP_rst_path(tag, exp):
    exp_name = os.path.basename(exp)
    dataset, _, _, _ = parse_exp_name(exp_name)
    stable_neurons_path = mkpath(os.path.join(
            stb_root, dataset, tag, cnt_rst, exp_name + '.txt'))
    return stable_neurons_path

def collect_rst(model_name, tag):
    dataset = os.path.basename(model_name).split('_')[1]
    rst_path = os.path.join(stb_root, dataset, tag, cnt_rst, os.path.basename(model_name) + '.txt')
    if not os.path.exists(rst_path):
        return [model_name + ': NO RESULT']
    rst = [l.strip() for l in open(rst_path, 'r').readlines()]
    return rst
