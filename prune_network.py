"""
    Update the weights for the network after pruning
    A. For stably inactive neurons, remove the corresponding weights before and after them.
    B. For stably active neurons, remove the weights before and update the weights and bias:
    We have the following notations:
        0. h1_a: [c1_a]
                 the stably active neurons
        1. w1_a: [c1_a, c0] 
                 the weights before the active neurons. 
           w1_a = {w11_a, w12_a}, whose rank is r1 and r1 < c1_a 
           w11_a: [r1, c0] 
                  the weights before h11_a;
                  the r1 row vectors which can represent the base in R^r1
           w12_a: [c1_a - r1, c0]
                  the weights before h12_a;
                  the rest row vectors can be presented as a linear combination of the w11_a
        2. w12_a = K1*w11_a, 
           K1: [c1_a - r1, r1]
        3. w2_a:  [c2, c1_a] 
                  the weights after the active neurons.
           w2_a = {w21_a, w22_a}
           w21_a: [c2, r1]
                  the weights after h11_a;
           w22_a: [c2, c1_a - r1]
                  the weights after h12_a;
        4. b1_a = {b11_a, b12_a}
           b11_a: [r1, ]
           b12_a: [c1_a - r1, ]
    We update the weights before and after h1_a as below:
        1. remove w12_a from w1_a
        2. merge w22_a into w21_a:
           w21_a' = w21_a + w22_a * K1
        3. update b2:
           b2'    = b2 + w22_a * (b12_a - K1 * b11_a)
        4. remove w22_a
        
"""
import os
import sys

import numpy as np
from numpy.linalg import matrix_rank, norm
import torch
#import sympy

from common.timer import Timer
from dir_lookup import *

DEBUG = False
timer = Timer()

#######################################################################################
# remove the corresponding weights before and after the stably inactive neurons
#######################################################################################
def prune_inactive_per_layer(w1, w2, ind_inact):
    """
    w1: [c1, c0] the weights before the current layer
    w2: [c2, c1] the weithgs after the current layer
    ind_inact: a list of the index of the stably inactive neurons
    """
    w1 = np.delete(w1, ind_inact, axis=0)
    w2 = np.delete(w2, ind_inact, axis=1)
    return w1, w2


#######################################################################################
# 1. remove the corresponding weights before the stably active neurons
# 2. update the weights after them
#######################################################################################
def prune_active_per_layer(w1, w2, b1, b2, ind_act):
    """
    w1: [c1, c0] the weights before the current layer
    w2: [c2, c1] the weithgs after the current layer
    ind_inact: a list of the index of the stably inactive neurons
    b1: [c1] 
    b2: [c2]
    """
    ind_act = np.array(ind_act)
    w1_a = w1[ind_act, :]
    b1_a = b1[ind_act]
    r1  = np.linalg.matrix_rank(w1_a.astype(np.float64))
    if r1 == w1_a.shape[0]:
        return w2,b2,[]
    
    print('Start to solve linearly dependent row vectors')
    K1, is_indep = solve_linearly_depend(w1_a)
    #timer.stop('solve_linearly_depend is done')
    w2_a = w2[:, ind_act]
    b2_old = b2[:]
    b11 = b1_a[is_indep]
    b12 = b1_a[is_indep == 0]
    w21 = w2_a[:, is_indep]
    w22 = w2_a[:, is_indep == 0]
    #   1. remove w12_a from w1_a
    #w1 = np.delete(w1, ind_act[is_indep==0], axis = 0)
    #    2. merge w22_a into w21_a:
    #       w21_a' = w21_a + w22_a * K1
    #    3. update b2:
    #       b2'    = b2 + w22_a * (b12_a - K1 * b11_a)
    w2[:, ind_act[is_indep]] = w21 + w22 @ K1 
    b2                       = b2  + w22 @ (b12 - K1 @ b11)
    #    4. remove w22
    #w2 = np.delete(w2, ind_act[is_indep==0])
    if DEBUG:
        for run in range(10):
            h = np.random.randn(w1.shape[1])
            x = w1 @ h + b1
            x = x[ind_act]
            y_old = w2_a @ x + b2_old
            y_new = w2[:, ind_act[is_indep]] @ x[is_indep] + b2
            if not np.allclose(y_old, y_new):
                import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    return w2, b2, ind_act[is_indep==0]


#######################################################################################
#ref: https://stackoverflow.com/a/44556151/4874916
#######################################################################################
def find_independ_rows(A):
    # find the independent row vectors
    # only work for square matrix
    #lambdas, V = np.linalg.eig(A.T)
    #return lambdas!=0
    
    #import pdb;pdb.set_trace()
    inds = find_independ_rows2(A)
    
    # too slow
    #_, inds = sympy.Matrix(A).T.rref()
    #import pdb;pdb.set_trace()
    return inds

def find_independ_rows2(A):
    r = np.linalg.matrix_rank(A.astype(np.float64))
    base = [A[0,:]]
    base_ind = [0]
    row = 1
    cur_r = 1 
    while cur_r < r:
        tmp = base + [A[row,:]]
        #import pdb;pdb.set_trace()
        if np.linalg.matrix_rank(np.stack(tmp, axis=0).astype(np.float64)) > cur_r:
            cur_r += 1
            base.append(A[row,:])
            base_ind.append(row)
        row += 1
    return base_ind


#######################################################################################
# 1. find the base row vectors for w11
# 2. represent the non-base row vectors as a linear combination of the base
#######################################################################################
def solve_linearly_depend(w11):
    """
        Args:
            w11: [c1_a, c0]
        Returns:
            K  : [c1_a -rank(w11), rank(w11)]
            is_indep: [rank(w11),]
    """
    ##timer.stop('start find_independ_rows')
    # find the independent row vectors
    is_indep = find_independ_rows(w11)
    ##timer.stop('finsh find_independ_rows')
    is_indep = np.array([i in is_indep for i in range(w11.shape[0])])
    #ind_indep    = find_li_vectors(w11)
    dep = w11[is_indep == 0, :] # the linearly dependent row vectors
    base = w11[is_indep, :] # the independent row vectors
    # solve a linear equation: dep = K * base
    K = []
    for i in range(dep.shape[0]):
        y  = dep[i, :]
        A  = np.concatenate([base, y[None,:]], axis=0).T
        is_indep_A = np.array(find_independ_rows(A))
        #print('base[:,is_indep_A]: ', base[:,is_indep_A].shape)
        ##timer.stop(f'start np.linalg.solve: {i} ')
        #import pdb;pdb.set_trace()
        k  = np.linalg.solve(base[:,is_indep_A].T, y[is_indep_A])
        ##timer.stop(f'start np.linalg.solve: {i}')
        assert(np.allclose(np.dot(base.T, k), y))
        K.append(k)
    K = np.stack(K, axis=0)
    return K, is_indep 
 

def sanity_ckp():
    w1 = np.array(
            [[1,0,2,1],
             [1,1,0,0],
             [2,1,2,1],
             [1,2,3,0],
             [11,12,13,0]])
    b1 = np.arange(5)
    w2 = np.arange(10).reshape(2,5)
    b2 = np.array([11,12])
    act_neurons = np.array([[1, 0], [1,1], [1,2]]).T
    inact_neurons = np.array([[1,4]]).T
    w_names = ['w1', 'w2']
    b_names = ['b1', 'b2']
    return [w1, w2], [b1, b2], act_neurons, inact_neurons, w_names, b_names
   
def read_MILP_stb(model_path, tag=ALLPRE):
    stb_path = get_MILP_stb_path(tag, model_path)
    stb_neurons = np.load(stb_path, allow_pickle=True).item()
    act_dict = stb_neurons['stably_active']
    inact_dict = stb_neurons['stably_inactive']
    act_arr, inact_arr = [], []
    for i,j in act_dict.items():
        act_arr.extend([[i,jj] for jj in j])
    act_neurons = np.array(act_arr).T
    
    for i,j in inact_dict.items():
        inact_arr.extend([[i,jj] for jj in j])
    inact_neurons = np.array(inact_arr).T
    return act_neurons, inact_neurons

def read_train_stb(model_path):  
    #stb_path = os.path.join(model_path, 'stable_neurons.npy')
    stb_path = get_train_stb_path(model_path)
    stb_neurons = np.load(stb_path, allow_pickle=True).item()
    import pdb;pdb.set_trace()
    act_neurons = stb_neurons['stably_active'].squeeze(axis=2).T
    inact_neurons = stb_neurons['stably_inactive'].squeeze(axis=2).T
    return act_neurons, inact_neurons 
    ##timer.stop('loaded the checkpoint')

#######################################################################################
# Sort all weights in descending order and prune the smallest ones 
#######################################################################################
def magnituded_based_pruning(weights, keep_ratio):
    all_scores = torch.cat([torch.flatten(torch.tensor(w)).abs() for w in weights]) 
    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold,out = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1].numpy()
    for i, w in enumerate(weights):
        w[np.absolute(w) <= acceptable_score] = 0
        #weights[i] = w # updating w happens inplace
    return weights


#######################################################################################
# Apply magnituded based pruning 
#   tag: the method to get sbable neurons, whoses results is under the folder
#        'results/${dataset}/${tag}' 
#        e.g:  results-no_preprocess, results-preprocess_all, 
#              results-old-approach, results-preprocess_partial
#######################################################################################
def magnituded_based_prune_ckp(model_path, tag):
    #1. load model after lossless pruning
    pruned_ckp_path = os.path.join(model_path, 'pruned_checkpoint_120.tar')
    if not os.path.exists(pruned_ckp_path):
        prune_ckp(model_path,tag)
    pruned_ckp = torch.load(pruned_ckp_path)
    _, _, pruned_weights, _, _ = weights_bias_from_fcnn_ckp(pruned_ckp)

    #2. load original model
    ckp_path = os.path.join(model_path, 'checkpoint_120.tar')
    if not os.path.exists(ckp_path):
        print(ckp_path, 'not exists')
        return
    ckp = torch.load(ckp_path)
    w_names, _, weights, _, device = weights_bias_from_fcnn_ckp(ckp)
    
    #3. count the weights
    weights_cnt = torch.tensor([w.size for w in weights]).sum()
    keep_cnt = torch.tensor([w.size for w in pruned_weights]).sum()
    keep_ratio = keep_cnt/weights_cnt.float()
    #keep_ratio = 0.05
    #4. apply magnituded pruning
    m_pruned_weights = magnituded_based_pruning(weights, keep_ratio)
    for i, name in enumerate(w_names):
        ckp['state_dict'][name] = torch.from_numpy(m_pruned_weights[i]).cuda(device=device)
    ckp['prune_ratio_weight'] = 1 - keep_ratio
    neurons_cnt = torch.tensor([w.shape[0]  for w in weights[:-1]]).sum().numpy()
    keep_neuron_cnt = torch.tensor([w.shape[0]  for w in pruned_weights[:-1]]).sum().numpy()
    ckp['prune_ratio_neuron'] = 1 - float(keep_neuron_cnt) / float(neurons_cnt)
    ckp['arch'] = os.path.basename(model_path).split('_')[2]

    # save the checkpoint 
    m_pruned_ckp_path = os.path.join(model_path, 'magnitude_pruned_checkpoint_120.tar')
    torch.save(ckp, m_pruned_ckp_path) 

    print(model_path)
    print('pruning_ratio_weights:', 1 - keep_ratio)
    print('pruning_ratio_neurons:', ckp['prune_ratio_neuron'])


def weights_bias_from_fcnn_ckp(ckp):
    w_names = sorted([name for name in ckp['state_dict'].keys() 
                            if 'weight' in name and 'features' in name])
    b_names = sorted([name for name in ckp['state_dict'].keys() 
                            if 'bias' in name and 'features' in name])
    w_names.append('classifier.0.weight')
    b_names.append('classifier.0.bias')

    device = ckp['state_dict'][w_names[0]].device

    weights = []
    bias    = []
    for name in w_names:
        weights.append(ckp['state_dict'][name].cpu().numpy())
    for name in b_names:
        bias.append(ckp['state_dict'][name].cpu().numpy())
    return w_names, b_names, weights, bias, device

#######################################################################################
# Load the weights and bias from the checkpoints and the neuron stability from the state files,
# Prune the model according to neuron stability
#   tag: the method to get sbable neurons, whoses results is under the folder
#        'results/${dataset}/${tag}' 
#        e.g:  results-no_preprocess, results-preprocess_all, 
#              results-old-approach, results-preprocess_partial
#######################################################################################
def prune_ckp(model_path,tag):
     
    if DEBUG:
        weights, bias, act_neurons, inact_neurons, w_names, b_names = sanity_ckp()
    else:
        timer.start()
        ckp_path = os.path.join(model_path, 'checkpoint_120.tar')
        pruned_ckp_path = os.path.join(model_path, 'pruned_checkpoint_120.tar')
        MILP_rst      = collect_rst(model_path, tag)
        #stb_path = os.path.join(model_path, 'stable_neurons.npy')
        #stb_path = get_stb_path(ALLPRE, model_path)
        stb_path = get_MILP_stb_path(tag, model_path)
        if not os.path.exists(ckp_path):
            print(ckp_path, 'not exists')
            return
        if not os.path.exists(stb_path):
            print(stb_path, 'not exists')
            return
        ckp = torch.load(ckp_path)
        w_names, b_names, weights, bias, device = weights_bias_from_fcnn_ckp(ckp)
        act_neurons, inact_neurons = read_MILP_stb(model_path, tag) 
        ##timer.stop('loaded the checkpoint')
   
    # Get a new model by applying lossless pruning. 
    #   Note this reduces the model size instead of masking pruned weights and biases as zeros
    pruned_numbers = ''
    for l in range(1, len(weights)):
        if len(act_neurons) > 0:
            ind_act  = act_neurons[1,   act_neurons[0,:] == l]
        else:
            ind_act  = []
        if len(inact_neurons) > 0:
            ind_inact = inact_neurons[1, inact_neurons[0,:] == l]
        else:
            ind_inact = []
        w1 = weights[l-1]
        w2 = weights[l]
        b1 = bias[l-1]
        b2 = bias[l]
        prune_ind = []
        # get the index of stably active neurons to prune
        if len(ind_act) > 0:
            #import pdb;pdb.set_trace()
            w2, b2, prune_ind_act = prune_active_per_layer(w1, w2, b1, b2, ind_act)
            prune_ind.extend(prune_ind_act)
        else:
            prune_ind_act = []
        # all stably inactive neurons to prune
        if len(ind_inact) > 0:
            prune_ind.extend(ind_inact)
        # delete all the neurons to be pruned
        if len(prune_ind) > 0:
            w1 = np.delete(w1, prune_ind, axis=0)
            b1 = np.delete(b1, prune_ind, axis=0)
            w2 = np.delete(w2, prune_ind, axis=1)
        print(f'Layer-{l}: prune {len(prune_ind_act)} stably active neurons')
        print(f'layer-{l}: prune {len(ind_inact)} stably inactive neurons')
        pruned_numbers += f'{len(prune_ind_act)}, {len(ind_inact)},,'
        # update the weights and bias
        weights[l-1] = w1
        bias[l-1]    = b1
        weights[l]   = w2
        bias[l]      = b2
        ##timer.stop(f'{l} layer is pruned')
        #import pdb;pdb.set_trace()
    
    # update the ckeckpoints
    for i, name in enumerate(w_names):
        ckp['state_dict'][name] = torch.from_numpy(weights[i]).cuda(device=device)
    for i, name in enumerate(b_names):
        ckp['state_dict'][name] = torch.from_numpy(bias[i]).cuda(device=device)
    # save the checkpoint 
    torch.save(ckp, pruned_ckp_path)
    
    # append the lossless pruning results into the stat file
    if 'NO RESULT' in MILP_rst[0]:
        _,arch,_,_ = parse_exp_name(model_path)
        MILP_rst = [model_path + ', , ,' + ' , , ,,' * len(arch.split('-')) + ' ,, , ,, ,,'] 
    MILP_rst_path = get_MILP_rst_path(tag, model_path)
    with open(MILP_rst_path, 'w') as f:
        for l in MILP_rst:
            f.write(l + pruned_numbers + '\n')  
    print(MILP_rst_path)

if __name__ == '__main__':
    #model_path = 'model_dir/CIFAR100-rgb/dnn_CIFAR100-rgb_400-400_7.500000000000001e-05_0001'
    #model_path = 'model_dir/CIFAR10-rgb/dnn_CIFAR10-rgb_400-400_0.000175_0002'
    model_name = os.path.basename(sys.argv[1])
    dataset = model_name.split('_')[1]
    model_path = os.path.join(model_root, dataset, sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == 'magnitude':
        magnituded_based_prune_ckp(model_path, ALLPRE)
    else:
        prune_ckp(model_path, ALLPRE)
