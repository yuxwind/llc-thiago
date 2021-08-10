#!/usr/bin/python

"""
    Sample Run -
    ./get_activation_patterns.py -i weight_files/XOR.dat -c
    ./get_activation_patterns.py -i ../forward_pass/activation_pattern/models/fcnn_run_54/weights.dat    | tee ../forward_pass/activation_pattern/models/fcnn_run_54/bounds.log
    ./get_activation_patterns.py -i ../forward_pass/activation_pattern/models/fcnn_run_54/weights.dat -a | tee ../forward_pass/activation_pattern/models/fcnn_run_54/bounds.log
    ./get_activation_patterns.py -i ../forward_pass/activation_pattern/models/fcnn_run_132/weights.dat -m 0.001 | tee ../forward_pass/activation_pattern/models/fcnn_run_132/bounds_width_0.001.log
    ./get_activation_patterns.py -i ../forward_pass/activation_pattern/models/fcnn_run_132/weights.dat -x local_stability/mnist_sample_class_0.dat -L 0.001 | tee ../forward_pass/activation_pattern/models/fcnn_run_132/bounds_xbar_0_width_0.001.log

    This function calls gurobipy on a deep neural network weights and biases to
    frame an optimisation problem and then obtain the solution to decide the
    minimum and maximum of each node at each layer. The input is assumed to be
    bounded. Then these bounds are used to calculate the activation pattern
    at each of the nodes barring the input nodes. It also write two files in the
    same directory as the weights file -
    1) inactive_nodes_file - File which specifies which layer and index, the
        units are off
    2) activation_pattern_file - File which contains all possible activation
        patterns of the network

    The code is implementation of ideas in the paper and the exact optimisation
    problem used are the equations (12)-(14) of the paper
    Empirical Bounds on Linear Regions of Deep Rectifier Networks
    - T Serra and S Ramalingam
    https://arxiv.org/pdf/1810.03370.pdf

    Modes in which the functions work:
    1) Normal mode - Solve exactly for till total_layers-2 and get a feasible
        solution at total_layers-1 to see the maxima and minima. If no maxima
        or minima is found at the top_layer-1, that means the solution does
        not exist

    2) Approx mode - Get a feasible solution at all layers and use
        model.objBound which is the solution of LP - the relaxed version of
        the MILP to get the H and Hbars.

    Note:
    We never solve for the final layers for classification models since that
    is the softmax layer and is required for classification

    Version 11 Abhinav Kumar 2019-04-30 (functionality of xbar and width around xbar for local stability analysis added)
    Version 10 Abhinav Kumar 2019-04-22 (maxima option added in arguments)
    Version 9  Abhinav Kumar 2019-04-02 (args.bounds_only_flag_added to get only bounds)
    Version 8  Abhinav Kumar 2019-03-27 (Bounds default value > 0 and used checks for activation pattern as > 0)
    Version 7  Abhinav Kumar 2019-03-26 (Classification and Regression Models fixed)
    Version 6  Abhinav Kumar 2019-03-22 (Flush added in callback)
    Version 5  Abhinav Kumar 2019-03-22 (Activations logged to a file)
    Version 4  Abhinav Kumar 2019-03-20 (Weird terminations handled as per Thiago)
    Version 3  Abhinav Kumar 2019-03-20 (ObjBound used when nothing Optimal available)
    Version 2  Abhinav Kumar 2019-03-19 (approx_method_no_solution_val fixed to -0.0)
    Version 1  Abhinav Kumar 2019-03-16
"""

import argparse
import os
import numpy as np
import time
from gurobipy import *
import re

import random
import copy

import torch
from torch import nn

from common.io import mkpath, mkdir
from dir_lookup import *
from nn_milp_utils import parse_file, parse_npy


accuracy = None

time_before = time.time()

ap = argparse.ArgumentParser()
ap.add_argument    ('-i', '--input'                                  , help = 'path of the input dat file'           , default='./weight_files/XOR.dat')
ap.add_argument    ('-a', '--approx'           , action='store_true'  , help = 'use approx algorithm for calculation (default: False)')
ap.add_argument    ('-b', '--bounds_only_flag' , action='store_true'  , help = 'bounds only flag (default: False)')
ap.add_argument    ('--preprocess_all_samples' , action='store_true'  , help = 'preprocess the neuron stability using all training samples(default: False)')
ap.add_argument    ('--preprocess_partial_samples' , action='store_true'  , help = 'preprocess the neuron stability using partial training samples(default: False)')
ap.add_argument    ('-c', '--classify_flag'    , action='store_false' , help = 'classification flag (default: True)')
ap.add_argument    ('-m', '--maximum'          , type=float           , help = 'maxima of the nodes (default: 1)'     , default='1')
ap.add_argument    ('-x', '--xbar_file'                               , help = 'Center of the individual input nodes in csv format. Could be an image of the validation set.')
ap.add_argument    ('-L', '--width_around_xbar', type=float           , help = 'Width around xbar of each individual node. (default: 0.0001)' ,default='0.0001')
ap.add_argument    ('-f', '--formulation'                             , help = 'Formulation to be used: neuron, layer, network (default: network)', default='network')
ap.add_argument    ('-F', '--feasible'                                , help = 'Injection of feasible solution based on network input: relaxation, random, off (default: relaxation; not available for neuron formulation)', default='relaxation')
ap.add_argument    ('-t', '--time_limit', type=float                  , help = 'Time limit in seconds to conclude MILP solve (default: None)', default=None)
ap.add_argument    ('--dataset'        , dest='dataset', type=str         , default='MNIST'     , help='Dataset to be used (default: MNIST)')
ap.add_argument    ('--limit_input', action = 'store_true', help='Limit the input for MNIST (default: False)')
args = ap.parse_args()


determine_stability_per_network = (args.formulation=='network')
determine_stability_per_layer = (args.formulation=='layer')
determine_stability_per_unit = (args.formulation=='neuron')

if args.formulation=='neuron':
    args.feasible = 'off'

inject_relaxed_solution = (args.feasible=='relaxation')
inject_random_solution = (args.feasible=='random')

################################################################################
# Parameters
################################################################################
# Bound on the input nodes
# Input_min should not be negated as the hidden units since these bounds are
# directly used by the constraints of MILP formulation
input_min = 0
input_max = args.maximum

# If approx method fails, solution value
approx_method_no_solution_val = 1

# Optimisation Display
disp_opt = False

# Saving options
save_model  = False
save_folder = "lp_models"

# Activation Pattern options
show_activations = False
print_freq       = 1000


################################################################################
# Initialisations
################################################################################
#layers, nodes_per_layer, weights, bias = parse_file(args.input)
layers, nodes_per_layer, weights, bias = parse_npy(args.input)
import pdb;pdb.set_trace()

# Total number of layers including input is layers+1
tot_layers = layers+1
max_nodes  = np.max(nodes_per_layer)

# bounds contain two values for every node
# 0 index in bounds is for maxima and 1 index in bounds is for minima
bounds =  12.34*np.ones((tot_layers, max_nodes, 2))

# Initialize bounds for layer 0 (input layer)
if (args.xbar_file is None):
    bounds[0, 0:nodes_per_layer[0], 0] = input_max
    bounds[0, 0:nodes_per_layer[0], 1] = input_min

    # Inactive_nodes and activation patterns file
    inactive_nodes_file     = "inactive_input_" + str(input_min) + "_" + str(input_max) + ".dat"
    active_nodes_file       = "active_input_"   + str(input_min) + "_" + str(input_max) + ".dat"
    activation_pattern_file = "activation_pattern_abhinav_input_" + str(input_min) + "_" + str(input_max) + ".dat"

else:
    input_nodes_center = np.genfromtxt(args.xbar_file,  delimiter=',')
    width_around_xbar  = args.width_around_xbar

    assert input_nodes_center.shape[0] == nodes_per_layer[0]

    print("Ignoring values supplied by command argument maximum !!!")
    print("Putting center around the values given by the file {}".format(args.xbar_file))
    print("Width around xbar = {:.5f}".format(width_around_xbar))

    for i in range(nodes_per_layer[0]):
        bounds[0, i, 0] = input_nodes_center[i] + width_around_xbar
        bounds[0, i, 1] = input_nodes_center[i] - width_around_xbar

    # Inactive_nodes and activation patterns file
    inactive_nodes_file     = "inactive_center_" + os.path.basename(args.xbar_file[:-4]) + "_width_around_xbar_" + str(width_around_xbar) + ".dat"
    active_nodes_file       = "active_center_"   + os.path.basename(args.xbar_file[:-4]) + "_width_around_xbar_" + str(width_around_xbar) + ".dat"
    activation_pattern_file = "activation_pattern_abhinav_center_" + os.path.basename(args.xbar_file[:-4]) + "_width_around_xbar_" + str(width_around_xbar) + ".dat"


# Intialisation list for variables
lst = []
# ***************************** Changed here *******************************
for i in range(tot_layers):
    for j in range(nodes_per_layer[i]):
        lst += [(i, j)]


print("\n\n===============================================================================");
print("Input File    = %s" %(args.input))
print("Num of layers = %d" %(layers))
print("Nodes per layer")
print(nodes_per_layer)
print("===============================================================================");
#print("\nWeights")
#print(weights)
#print("\nBias")
#print(bias)

print("")
if (args.approx):
    print("\nSolving using the approx method ...\n\n")


print("Lower bound of input node = {}".format(input_min))
print("Upper bound of input node = {}".format(input_max))

# ***************************** Changed here *******************************
if (args.classify_flag):
    print("Classification Model. Ignore the last layer since that is used for classes")
    run_till_layer_index =  tot_layers-1
else:
    print("Not a Classification Model. Running for all layers including the last layer")
    run_till_layer_index =  tot_layers
print("------------------------------------------------------------------------")


network = args.input[:args.input.rfind("/")] #args.input.split("/")[0]
print("Network",network)
print("Accuracy",accuracy)
#f = open("RESULTS.txt","a+")

if args.formulation == 'neuron':
    tag = OLD
else:
    if args.preprocess_all_samples:
        tag = ALLPRE
    elif args.preprocess_partial_samples:
        tag = PARTPRE
    else:
        tag = NOPRE

stb_dir = mkdir(os.path.join(stb_root, args.dataset, tag, cnt_rst))
exp_name = os.path.basename(os.path.dirname(args.input))
stable_neurons_path = mkpath(os.path.join(stb_dir, args.dataset, tag, stb_neuron, exp_name + '.npy'))
f = open(mkpath(os.path.join(stb_dir, exp_name + '.txt')), "a+")
f.write(network+", "+str(accuracy)+", , ")

timeouts = 0
stably_active = {}
stably_inactive = {}

timed_out = False

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

# Initialize p_lst and q_list by adding all nodes
p_lst = [(i,j) for i in range(1,tot_layers) for j in range(nodes_per_layer[i]) ]
q_lst = [(i,j) for i in range(1,tot_layers) for j in range(nodes_per_layer[i]) ]
print(-1,len(p_lst),len(q_lst))
last_size = len(p_lst)+len(q_lst)
last_update = -1
max_nonupdates = 1

remove_p = True
remove_q = True

# Load the training dataset to preprocess p_lst and q_lst
normalize = transforms.Normalize(mean=[0], std=[1]) #Images are already loaded in [0,1]
transform_list = [transforms.ToTensor(), normalize]
if args.dataset == "MNIST": 
    data = datasets.MNIST(root='./data', train=True, transform=transforms.Compose(transform_list), download=True)
elif args.dataset == "CIFAR10-gray":
    transform_list.insert(0, transforms.Grayscale(num_output_channels=1))
    data = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose(transform_list), download=True)
elif args.dataset == "CIFAR10-rgb":
    data = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose(transform_list), download=True)
elif args.dataset == "CIFAR100-rgb":
    data = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose(transform_list), download=True)
n = data.__len__()
max_nonupdates = 10

# Initialize p_lst and q_lst by loading the stability from running model on training data
to_preprocess_partial = args.preprocess_partial_samples
to_preprocess_all = args.preprocess_all_samples
if to_preprocess_partial:
    if not determine_stability_per_unit:
      for i in range(n):
        #(img, target) = data.__getitem__(random.randint(0,n))
        (img, target) = data.__getitem__(i)
        imgf = torch.flatten(img)
        input = [imgf[j].item() for j in range(nodes_per_layer[0])]
        for l in range(1,tot_layers):
            output = []
            for j in range(nodes_per_layer[l]):
                g = bias[l-1][j][0] + sum([ weights[l-1][j,k]*input[k] for k in range(nodes_per_layer[l-1]) ])
                if g>0 and (l,j) in p_lst and remove_p:
                    p_lst.remove((l,j))
                elif g<0 and (l,j) in q_lst and remove_q:
                    q_lst.remove((l,j))
                output.append(max(0,g))
            input = output
        print(i, len(p_lst), len(q_lst))
        size = len(p_lst)+len(q_lst)
        if size < last_size:
            last_size = size
            last_update = i
        if len(p_lst)+len(q_lst) < 1 or i > last_update + max_nonupdates:
            #print(p_lst, q_lst)
            print(i, last_update, max_nonupdates)
            break
if to_preprocess_all:
    stable_from_sample_path = os.path.join(os.path.dirname(args.input), 'stable_neurons.npy')
    stable_from_sample = np.load(stable_from_sample_path, allow_pickle=True).item()
    q_lst_ = stable_from_sample['stably_active'].squeeze()
    p_lst_ = stable_from_sample['stably_inactive'].squeeze()
    if len(q_lst_.shape) == 1:
        q_lst_ = q_lst_[None,:]
    if len(p_lst_.shape) == 1:
        p_lst_ = p_lst_[None,:]
    q_lst = [(q_lst_[i,0], q_lst_[i,1]) for i in range(q_lst_.shape[0]) ]
    p_lst = [(p_lst_[i,0], p_lst_[i,1]) for i in range(p_lst_.shape[0]) ]
remaining = len(p_lst)+len(q_lst)

# Create a new model
model = Model("mip1")

if (not(disp_opt)):
    # Donot display output solving
    # https://stackoverflow.com/a/37137612
    model.params.outputflag = 0
# global variables 
p = model.addVars(p_lst, name="p")
q = model.addVars(q_lst, name="q")

def networkcallback(model, where):
    global p, q, i, nodes_per_layer, positive_units, negative_units
    global h
    global lst

    if where == GRB.Callback.MIPSOL:
        print("FOUND A SOLUTION")
        p_value = model.cbGetSolution(p)
        q_value = model.cbGetSolution(q)
        for (m,n) in p_lst:
            if p_value[m,n] == 1:
                positive_units.add((m,n))
                model.cbLazy(p[m,n] == 0)
                #print("+",m,n)
        for (m,n) in q_lst:
            if q_value[m,n] == 1:
                negative_units.add((m,n))
                model.cbLazy(q[m,n] == 0)
                #print("-",m,n)
    elif where == GRB.Callback.MIP:
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        print("BOUND:", objbnd)
        if objbnd<0.5:
            model.terminate()
    elif where == GRB.Callback.MIPNODE:
        print("MIPNODE")
        vars = []
        values = []

        if inject_relaxed_solution:
            for input in range(nodes_per_layer[0]):
                vars.append(h[0,input])
            values = model.cbGetNodeRel(vars)
            model.cbSetSolution(vars,values)
        elif inject_random_solution:
            for input in range(nodes_per_layer[0]):
                vars.append(h[0,input])
                values.append(bounds[0, input, 0] + random.random()*(bounds[0, input, 1]-bounds[0, input, 0]))
            model.cbSetSolution(vars,values)

        #obj = model.cbUseSolution()
        #print("GOT",obj)
    #elif where == GRB.Callback.MESSAGE:
    #    msg = model.cbGet(GRB.Callback.MSG_STRING)
    #    print(msg)
    #    import pdb;pdb.set_trace()

class inputSolver():
    def __init__(self, input_shape, input_min, input_max, name='input'):
        self.input_shape = [s.item() if type(s) == np.int64 else s for s in input_shape]
        self.input_min   = input_min
        self.input_max   = input_max
        self.name        = name

    def get_output_bounds(self):
        # Specify bounds of the input variables.
        # 0 index in bounds is for maxima and 1 index in bounds is for minima
        bound_max = input_max*np.ones(self.input_shape) 
        bound_min = input_min*np.ones(self.input_shape) 
        self.bounds = np.stack([bound_max, bound_min], axis=-1)

    def add_constrs(self):
        self.h = model.addVars(*self.input_shape, lb=self.input_min, ub=self.input_max, 
                name=self.name + '_h')
        if args.limit_input and args.dataset == 'MNIST':
            sum_max = 320.0
            sum_min = 15.0
            model.addConstr( quicksum(self.h) <= sum_max )
            model.addConstr( quicksum(self.h) >= sum_min )

class lenetSolver():
    def __init__(self, name='lenet'):
        self.name = name
        self.solvers = [inputSolver([3, 32, 32], input_min, input_max)]
        # layers, nodes_per_layer, weights, bias = parse_file(args.input)
        for l in range(1, run_till_layer_index):
            if l < 3:
                if l == 1:
                    hw = 28
                else:
                    hw = 10
                self.solvers.append(conv2dSolver(weights[l-1], bias[l-1], hw, hw, 
                    name = f'conv_{l}'))
                self.solvers.append(reluSolver(l, name=f'relu_{l}'))
                self.solvers.append(maxpool2dSolver(2, name=f'max_pool2d_{l}'))
            else:
                self.solvers.append(linearSolver(weights[l-1], bias[l-1], name=f'fc_{l}'))
                self.solvers.append(reluSolver(l, name=f'relu_{l}'))
      
            stably_active[l] = []
            stably_inactive[l] = []
    
        for i in range(len(self.solvers)):
            if i == 0:
                self.solvers[i].get_output_bounds()
            else:
                self.solvers[i].get_output_bounds(self.solvers[i-1].bounds)
            import pdb;pdb.set_trace()

    def add_constrs(self):
        for i in range(len(self.solvers)):
            print('layer: ', i, self.solvers[i].name)
            if i == 0:
                self.solvers[i].add_constrs()
            else:
                self.solvers[i].add_constrs(self.solvers[i-1].h)

class mlpSolver():
    def __init__(self, name='mlp'):
        self.name = name
        self.solvers = [inputSolver([nodes_per_layer[0]], input_min, input_max)]
        # layers, nodes_per_layer, weights, bias = parse_file(args.input)
        for l in range(1, run_till_layer_index):
            self.solvers.append(linearSolver(weights[l-1], bias[l-1], name=f'fc_{l}'))
            self.solvers.append(reluSolver(l, name=f'relu_{l}'))
      
            stably_active[l] = []
            stably_inactive[l] = []
    
        for i in range(len(self.solvers)):
            if i == 0:
                self.solvers[i].get_output_bounds()
            else:
                self.solvers[i].get_output_bounds(self.solvers[i-1].bounds)
            import pdb;pdb.set_trace()

    def add_constrs(self):
        for i in range(len(self.solvers)):
            print('layer: ', i, self.solvers[i].name)
            if i == 0:
                self.solvers[i].add_constrs()
            else:
                self.solvers[i].add_constrs(self.solvers[i-1].h)

class conv2dSolver():
    def __init__(self, w,b, h_out, w_out, stride=1, name='conv2d'):
        self.w = w # cout x cin x kernel_size x kernel_size
        self.b = b # cout
        self.cout,self.cin, self.ck, _ = w.shape
        self.Hout, self.Wout, self.stride = h_out, w_out, stride
        self.name = name
        self.bounds = 12.34*np.ones((self.cout, self.Hout, self.Wout, 2)) # cout
        self.conv2d = nn.Conv2d(self.cin, self.cout, self.ck)

    def get_output_bounds(self, input_bounds):
        # input_bounds: cin x h_in x w_in x 2
        in_ = torch.tensor(input_bounds[:,:,:,0])[None,:,:,:]
        w1  = torch.tensor(self.w)
        w1[w1<0] = 0
        self.conv2d.weight.data = w1 
        self.conv2d.bias.data   = torch.tensor(self.b)
        out1 = self.conv2d(in_) # 1 x cin x h_out x w_out
        self.bounds[:,:,:,0] = out1.squeeze(dim=0).detach().numpy()

        w2  = torch.tensor(self.w)
        w2[w2>0] = 0
        self.conv2d.weight = w2 
        out2 = self.conv2d(in_) # 1 x cin x h_out x w_out
        self.bounds[:,:,:,1] = -out2.squeeze(dim=0).detach().numpy()

        self.bounds[self.bounds < 1.0] = 1.0

    ###################################### Verified!
    # self.w:  cout x cin x ck x ck
    # self.b:  cout
    # self.h:  cout x h_out x w_out
    # vars_in: cin  x h_in  x h_in
    # h[i,j,k] = self.b[i]+ \
    #           sum_k1(sum_k2(sum_k3(w[i,k1,k2,k3] * vars_in[k1, h_in_+k2, w_in_+k3])))
    #           k1: 0 to cin
    #           k2: 0 to ck
    #           k3: 0 to ck
    #           h_in_ = j when stride=1, as the starting index of h_in on the input patch
    #           w_in_ = k when stride=1, as the starting index of w_in on the input patch
    ######################################
    def add_constrs(self, vars_in):
        self.h = model.addVars(self.cout, self.Hout, self.Wout)
        for i in range(self.cout):
            for j in range(self.Hout):
                h_in_ = j # stride = 1
                for k in range(self.Wout):
                    w_in_ = k # stride=1
                    model.addConstr(
                            quicksum(
                                vars_in[k1, h_in_ + k2, w_in_ + k3] * self.w[i,k1,k2,k3] 
                                for k1 in range(self.cin)
                                    for k2 in range(self.ck)
                                        for k3 in range(self.ck)
                            ) + self.b[i] == self.h[i, j, k])
        

class linearSolver():
    def __init__(self, w,b, name='fc'):
        self.w = w # cout x cin
        self.b = b.reshape(-1) # cout
        self.cout, self.cin = w.shape
        self.name = name 
        self.bounds = 12.34*np.ones((self.cout, 2)) # cout x 2
        self.fc     = nn.Linear(self.cin, self.cout)

    def get_output_bounds(self, input_bounds):
        # input_bounds: cin x 2
        in_ = torch.tensor(input_bounds[:,0])[None,:]
        w1 = torch.tensor(self.w)
        w1[w1<0.0] = 0.0
        self.fc.weight.data = w1
        self.fc.bias.data   = torch.tensor(self.b)
        out1 = self.fc(in_)
        self.bounds[:,0] = out1.squeeze(dim=0).detach().numpy()

        w2 = torch.tensor(self.w)
        w2[w2>0.0] = 0.0
        self.fc.weight.data = w2
        out2 = self.fc(in_)
        self.bounds[:,1] = -out2.squeeze(dim=0).detach().numpy()
        self.bounds[self.bounds < 1.0] = 1.0
    
    # the new and old methods have been verified to be equal to cal bounds.
    # Notes: we use a trick to cal bounds for a linear layer. 
    #        We assume its input are always zeros or postive.
    # TODO:  when the input of the network has negative values? 
    def get_output_bounds_old(self, input_bounds):
        # input_bounds: cin x 2
        for j in range(self.cout):
            max_unit = self.b[j]
            min_unit = self.b[j]
            for jj in range(self.cin):
                #TODO: 
                impact = self.w[j,jj]*input_bounds[jj,0]
                if impact > 0:
                    max_unit = max_unit + impact
                else:
                    min_unit = min_unit + impact
            self.bounds[j,0] = max(max_unit,1)
            self.bounds[j,1] = max(-min_unit,1)
        
    def add_constrs(self, vars_in):
        self.h = model.addVars(self.cout, lb = -GRB.INFINITY, name = self.name)
        for i in range(self.cout):
            model.addConstr(
                    quicksum(
                        self.w[i, k] * vars_in[k] for k in range(self.cin)
                    ) + self.b[i] - self.h[i] == 0, 
                self.name + "_1")

class maxpool2dSolver():
    def __init__(self, kernel_size, stride=None, name='maxpool2d'):
        self.ck = kernel_size
        self.cs = self.ck if stride is None else stride
        self.name = name

    def get_output_bounds(self, input_bounds):
        # inputs_bounds:  cin x h_in x w_in x 2
        self.inputs_bounds = input_bounds
        self.cin, self.h_in, self.w_in, _ = input_bound.shape
        self.cout = self.cin
        self.h_out = math.floor((self.h_in - self.ck)/self.stride) + 1
        self.w_out = math.floor((self.h_out - self.ck)/self.stride) + 1
        self.bounds = np.zeros([self.cout, self.h_out, self.w_out, 2])
        in_ = torch.tensor(self.nput_bounds[:,:,:0])[None,:,:,:]
        out_ = F.max_pool2d(in_, self.ck, stride=self.cs)
        
        self.bounds[:,:,:,0] = out_.sqeeze().numpy()
      
    ##########################################verified!
    # vars_in: cin  x hin  x win
    # h:       cout x hout x wout
    # ck:      2
    # stride:  ck in defalt
    # h[i,j,k] = max_k1(max_k2(vars_in[i, hin_ + k1, win_ + k2] ))
    #            k1: in the range (0, ck)
    #            k2: in the range (0, ck)
    #            hin_ = j * stride
    #            hout_= k * stride
    ##########################################
    def add_constrs(self, vars_in):
        s_vars_in = [self.cin, self.h_in, self.w_in]
        M         = self.input_bounds[:,:,:,0] + abs(self.input_bounds[:,:,:,1])
        self.h    = model.addVars(*s_vars_in, lb=0.0, name = self.name + '_h')
        self.z    = model.addVars(*s_vars_in, vtype=GRB.BINARY, name =self.name + '_z')
         
        for i in range(self.cout):
            for j in range(self.h_out):
                h_in_ = j * self.cs
                for k in range(self.w_out):
                    w_in_ = k * self.cs
                    for k1 in range(self.ck):
                        for k2 in range(self.ck):
                            hh = h_in_ + k1
                            ww = w_in_ + k2
                            model.addConstr(self.h[i,j,k] >= vars_in[i, hh, ww], 
                                    name=self.name + '_1')
                            model.addConstr(self.h[i,j,k] <= vars_in[i, hh, ww] + \
                                        M[i, hh, ww] * (1 - self.z[i, hh , ww]),
                                    name=self.name + '_2')
                    model.addConstr(
                            quicksum(self.z[i, h_in_ + k1, w_in_ + k2] 
                                for k1 in range(self.ck)
                                for k2 in range(self.ck)
                            ) == 1, name = self.name + '_3') 

class reluSolver():
    def __init__(self, layer_idx, name='relu'):
        self.bounds = None
        self.input_bounds = None
        self.name = name
        self.layer_idx = layer_idx

    def get_output_bounds(self, input_bounds):
        self.input_bounds = input_bounds
        self.bounds = copy.deepcopy(input_bounds)
        self.bounds[:,1] = 0

    def add_constrs(self, vars_in):
        s_vars_in = [s.item() if type(s) == np.int64 else s for s in self.input_bounds.shape[:-1]]
        self.h    = model.addVars(*s_vars_in, lb=0.0, name = self.name + '_h')
        self.hbar = model.addVars(*s_vars_in, lb=0.0, name = self.name + '_hbar')
        self.z    = model.addVars(*s_vars_in, vtype=GRB.BINARY, name =self.name + '_z')
         
        for ii in range(s_vars_in[0]):
            # when the inputs are 1D
            if len(s_vars_in) == 1: 
                model.addConstr(vars_in[ii]  ==  self.h[ii] - self.hbar[ii], self.name + "_1")
                model.addConstr(self.h[ii]   <= 2*self.input_bounds[ii, 0] * self.z[ii], 
                        self.name + "_2")
                model.addConstr(self.hbar[ii]<= 2*self.input_bounds[ii, 1] * (1 - self.z[ii]), 
                        self.name + "_3")

                #import pdb;pdb.set_trace()
                if (self.layer_idx, ii) in p_lst:
                    print('p_lst', self.layer_idx, ii)
                    model.addConstr(p[self.layer_idx, ii] <= self.z[ii])
                if (self.layer_idx, ii) in q_lst:
                    print('q_lst', self.layer_idx, ii)
                    model.addConstr(q[self.layer_idx, ii] <= 1-self.z[ii])
        
            # when the inputs are 3D
            if len(s_vars_in) == 3:
                cin, hin, win = s_vars_in
                for jj in range(hin):
                    for kk in range(win):
                        model.addConstr(vars_in[ii,jj,kk] == self.h[ii,jj,kk] - self.hbar[ii,jj,kk], 
                                self.name + "_1")
                        model.addConstr(
                                self.h[ii,jj,kk] <= 2*self.input_bounds[ii,jj,kk,0]*self.z[ii,jj,kk],
                                self.name + "_2")
                        model.addConstr(
                                self.hbar[ii,jj,kk] <=  2 * self.input_bounds[ii,jj,kk,1] * \
                                        (1-self.z[ii,jj,kk]), 
                                self.name + "_3")
                        # convert 3D to 1D in the same way as flatten array in numpy
                        idx = ii * hin * win + jj * win + kk 
                        if (self.layer_idx, idx) in p_lst:
                            model.addConstr(p[self.layer_idx, idx] <= self.z[ii, jj, kk])
                        if (self.layer_idx, idx) in q_lst:
                            model.addConstr(q[self.layer_idx, idx] <= 1-z[ii, jj, kk])
                        
#
def mlp_solver_old():
    global p, q, h 
    
    for i in range(1,run_till_layer_index):

      stably_active[i] = []
      stably_inactive[i] = []

      for j in range(nodes_per_layer[i]):
        max_unit = bias[i-1][j]
        min_unit = bias[i-1][j]
        for jj in range(nodes_per_layer[i-1]):
          impact = weights[i-1][j,jj]*bounds[i-1,jj,0]
          if impact > 0:
              max_unit = max_unit + impact
          else:
              min_unit = min_unit + impact
        bounds[i,j,0] = max(max_unit,1)
        bounds[i,j,1] = max(-min_unit,1)

    # Create variables
    g    = model.addVars(lst, lb=-GRB.INFINITY, name="g")
    h    = model.addVars(lst, lb=0.0          , name="h")
    hbar = model.addVars(lst, lb=0.0          , name="hbar")
    z    = model.addVars(lst, vtype=GRB.BINARY, name="z")


    # Specify bounds of the input variables.
    # 0 index in bounds is for maxima and 1 index in bounds is for minima
    model.addConstrs( h[0, k] <= bounds[0, k, 0] for k in range(nodes_per_layer[0]))
    model.addConstrs( h[0, k] >= bounds[0, k, 1] for k in range(nodes_per_layer[0]))
    if args.limit_input and args.dataset == 'MNIST':
        model.addConstr( quicksum(h[0, k] for k in range(nodes_per_layer[0])) <= 320.0 )
        model.addConstr( quicksum(h[0, k] for k in range(nodes_per_layer[0])) >= 15.0  )

    # Add constraints for all the nodes starting from the input layer till nodes in this layer
    for m in range(1, run_till_layer_index):
        for n in range(nodes_per_layer[m]):

            name = "c_" + str(m) + str(n)
            model.addConstr(
                    quicksum(
                        weights[m-1][n,k] * h[m-1,k] for k in range(nodes_per_layer[m-1])
                    ) + bias[m-1][n] - g[m,n] == 0, 
                    name + "_1")
            model.addConstr(g[m,n]    == h[m,n] - hbar[m,n], name + "_2")
            model.addConstr(h[m,n]    <= 2*bounds[m, n, 0] * z[m,n], name + "_3")

            model.addConstr(hbar[m,n] <= 2*bounds[m, n, 1] * ( 1 - z[m,n] ), name + "_4")

    for (m,n) in p_lst:
            model.addConstr(p[m,n] <= z[m,n])
    for (m,n) in q_lst:
            model.addConstr(q[m,n] <= 1-z[m,n])


# the original methods to add constraints for MLP 
#mlp_solver_old()   
# the new methods to add constraints for MLP

# test mlp_solver
#mlp_solver = mlpSolver() 
#mlp_solver.add_constrs()
# test lenet_solver
lenet_solver = lenetSolver()
lenet_solver.add_constrs()
import pdb;pdb.set_trace()
# model.setObjective(quicksum(p[m,n]+q[m,n] for m in range(1, run_till_layer_index) for n in range(nodes_per_layer[m])) , GRB.MAXIMIZE)
model.setObjective(
        quicksum(p[m,n] for (m,n) in p_lst) +  quicksum(q[m,n] for (m,n) in q_lst) , 
        GRB.MAXIMIZE)
try:
        print("SOLVING FOR",network)
        positive_units = set()
        negative_units = set()
        model.params.LazyConstraints = 1
        model.params.StartNodeLimit = 1
        if args.time_limit != None:
            model.setParam(GRB.Param.TimeLimit, args.time_limit-time.time()+time_before)
        model.optimize(networkcallback) 
        if args.time_limit != None and time.time()-time_before > args.time_limit:
            timed_out = True
except GurobiError as e:
        print("3 Error reported")

print('Obj: %g' % (model.objVal))
print(quicksum(p[m, n] for (m, n) in p_lst))
print(quicksum(q[m, n] for (m, n) in q_lst))

for m in range(1, run_till_layer_index):
  for n in range(nodes_per_layer[m]):
    #if (m,n) in positive_units and not (m,n) in negative_units:
    if (m,n) in q_lst and not (m,n) in negative_units:
        stably_active[m].append(n)
    #elif (m,n) in negative_units and not (m,n) in positive_units:
    elif (m,n) in p_lst and not (m,n) in positive_units:
        stably_inactive[m].append(n)    

  if not timed_out:
    print("Layer %d Completed..." %(m))

    matrix_list = []
    #for j in stably_active[m]:
    #  matrix_list.append([weights[m-1][j,k] for k in range(nodes_per_layer[m-1])])
    #  import pdb;pdb.set_trace()
    matrix_list = [weights[m-1][j] for j in stably_active[m]]
    print("Active: ", stably_active[m])
    import numpy
    rank = numpy.linalg.matrix_rank(numpy.array(matrix_list))
    #rank = torch.linalg.matrix_rank(torch.tensor(matrix_list))
    #rank = len(stably_active[m]) # torch.matrix_rank(torch.tensor(matrix_list))

    print("Active rank: ", rank, "out of", len(stably_active[m]))
    print("Inactive: ", stably_inactive[m])
    f.write(str(len(stably_active[m]))+", "+str(rank)+", "+str(len(stably_inactive[m]))+",, ")
  else:
    f.write("-, -, -,, ")

# Print the maxima and the minima from first hidden layer to output layer

time_after = time.time()
f.write(str(time_after-time_before)+",, ")
f.write(args.formulation+", "+args.feasible+",, "+str(remaining)+",, \n")
f.close()
np.save(stable_neurons_path, {'stably_active': stably_active, 'stably_inactive': stably_inactive})
#print_bounds(tot_layers, nodes_per_layer, bounds)


# If we have to do only bounds, do not go for activations
################################################################################
if (args.bounds_only_flag):
    sys.exit()


# Reseting the parameters of the model
model.reset()

print("")

# Writing the activation patterns to a file
my_file = open(os.path.join(os.path.dirname(args.input), activation_pattern_file), "w")
my_file.write("n = [" + ', '.join(map(str, nodes_per_layer)) + "]\n")



