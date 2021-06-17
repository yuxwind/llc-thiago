import os
import sys
from glob import glob

from common.io import mkdir,mkpath
from common.time import today_, now_
from cfg import type_arch,large_types 
####################################################
# Track progress of the models: whether the training is done, whether the MILP is done
# Ouput a todo list for the unfinished tasks:
#       python track_progress.py cifar10-rgb tr 0  # check training
#       python track_progress.py cifar10-gray ap 0 # check MILP with preprocessing using all training samples 
####################################################

TRAIN       = 'TR'      # training
AP          = 'AP'      # couting stable with preprocessing all training samples
NP          = 'NP'      # couting stable without preprocessing
#PP         = '#pp:'    # couting stable with preprocessing partial training sampls
PRE         = 'PRE'     # counting stable from training samples
RUN         = 'R'       # it is running
DONE        = 'D'       # it is done
UNDONE      = 'U'       # it is not finished
UNKNOWN     = 'X'       # it is unknown
ACTIONS     = [TRAIN, AP, NP, PRE]

model_dir   = './model_dir'
rst_dir     = './results/'
cnt_rst     = 'counting_results/'
stb_neuron  = 'stable_neurons/'

# results of counting stable neurons with/w.o preprecessing
NOPRE       = 'results-no_preprocess'
ALLPRE      = 'results-preprocess_all'
#PARTPRE     = 'results-preprocess_partial'

# Get experiment list for four datasets
p_mnist = './track_progress/track_mnist.txt'
p_cifar10_gray  = 'track_progress/track_cifar10-gray.txt'
p_cifar10_rgb = 'track_progress/track_cifar10-rgb.txt'
p_cifar100_rgb = 'track_progress/track_cifar100-rgb.txt'

mnist_track_list = open(p_mnist, 'r').readlines()
cifar10_gray_track_list = open(p_cifar10_gray + '.orig', 'r').readlines()
cifar10_rgb_track_list = open(p_cifar10_rgb + '.orig', 'r').readlines()
cifar100_rgb_track_list = open(p_cifar100_rgb + '.orig', 'r').readlines()


# check whether training is done
def check_training(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    model_path = os.path.join(model_dir, dataset, os.path.basename(model_name), 'checkpoint_120.tar')
    if os.path.exists(model_path):
        return True
    else:
        return False

def check_PRE(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    model_path = os.path.join(model_dir, dataset, os.path.basename(model_name), 'stable_neurons.npy')
    if os.path.exists(model_path):
        return True
    else:
        return False
def start_PRE(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    arch = os.path.basename(model_name).split('_')[2]
    model_root = os.path.join(model_dir, dataset, os.path.basename(model_name))
    cmd = "python train_fcnn.py --arch " + type_arch[arch] + " --resume " + \
            os.path.join(model_root, 'checkpoint_120.tar') + \
            "  -e --eval-stable --eval-train-data" + " --dataset " + dataset
    return cmd

# check whether MILP with preprocessing all training samples is done
def check_AP(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    rst_path = os.path.join(rst_dir, dataset, ALLPRE, cnt_rst, os.path.basename(model_name) + '.txt')
    if not os.path.exists(rst_path):
        return False
    rst = [l.strip() for l in open(rst_path, 'r').readlines()]
    for l in rst:
        if 'relaxation' in l:
            return True
    return None

# check whether MILP without preprocessing is done
def check_NP(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    rst_path = os.path.join(rst_dir, dataset, NOPRE, cnt_rst, os.path.basename(model_name) + '.txt')
    if not os.path.exists(rst_path):
        return False
    rst = [l.stript() for l in open(rst_path, 'r').readlines()]
    for l in rst:
        if 'relaxation' in l:
            return True
    return None

def start_training(model_name):
    terms = os.path.basename(model_name).split('_')
    dataset = terms[1]
    arch    = terms[2]
    l1      = terms[3]
    folder = mkdir(os.path.join(model_dir, dataset, os.path.basename(model_name)))
    
    env = ''
    cmd = "python train_fcnn.py --arch " + type_arch[arch] + " --save-dir " + folder  + " --l1 " + l1 + " --dataset " + dataset + " --eval-stable "
    log = mkpath(f"logs/training/{os.path.basename(model_name)}.log")
    #os.system(f"{env} {cmd} > {log} 2>&1 & ")
    print(log)
    return cmd
    #return f"{env} {cmd} > {log} 2>&1 & "

def start_AP(model_name):
    time_limit = 10800
    env = "GRB_LICENSE_FILE=~/gurobi-license/`sh ~/get_uname.sh`/gurobi.lic "
    
    terms = os.path.basename(model_name).split('_')
    dataset = terms[1]
    folder = mkdir(os.path.join(model_dir, dataset, os.path.basename(model_name)))

    cmd = "python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation network --time_limit " + str(time_limit) + " --dataset " + dataset + " --preprocess_all_samples"
    return cmd

def start_NP(model_name):
    pass


dataset = sys.argv[1] # this can be 'mnist', 'cifar10-gray', 'cifar10-rgb', 'cifar100-rgb' 
action = sys.argv[2] # this can be 'tr', 'ap', 'np'
start_job   = int(sys.argv[3])   # this can be 1 or 0 to indicate whether to start running   

if dataset == 'mnist':
    track_list = mnist_track_list
    p_track = p_mnist
elif dataset == 'cifar10-gray':
    track_list = cifar10_gray_track_list
    p_track = p_cifar10_gray
elif dataset == 'cifar10-rgb':
    track_list = cifar10_rgb_track_list
    p_track = p_cifar10_rgb
elif dataset == 'cifar100-rgb':
    track_list = cifar100_rgb_track_list
    p_track = p_cifar100_rgb
else:
    print('Unknown dataset')

if action == 'tr': 
    aid = 0
elif action == 'ap':
    aid = 1
elif action == 'np':
    aid = 2
elif action == 'pre':
    aid = 3
else:
    print('Unknown action')

todo = []
unknown = []
track_dir = './track_progress'
name_todo = f'todo_{ACTIONS[aid]}_{dataset}.sh'
name_unknown = f'unknow_{ACTIONS[aid]}_{dataset}.sh'
path_todo = os.path.join(track_dir, name_todo)
path_unknown = os.path.join(track_dir, name_unknown)
now = now_()
if os.path.exists(path_todo):
    print('Path exists: ', path_todo)
    print('Rename to ', os.path.join(track_dir, now, name_todo))
    os.rename(path_todo, mkpath(os.path.join(track_dir, now, name_todo)))
if os.path.exists(path_unknown):
    print('Path exists: ', path_unknown)
    print('Rename to ', os.path.join(track_dir, now, name_unknown))
    os.rename(path_unknown, mkpath(os.path.join(track_dir, now, name_unknown)))

f_todo   = open(path_todo, 'w')
f_unknown = open(path_unknown, 'w')

# example of tag format: TR-D, AP-D, NP-X 
for i,l in enumerate(track_list):
    l = l.strip()
    if l == '\n' or l == '':
        continue
    arrs = l.strip().split('#')
    exp = arrs[-1]
    arch = os.path.basename(exp).split('_')[2]
    if len(arrs) == 2:
        prev_tag = arrs[0].split(',')
        if len(prev_tag) < 4:
            prev_tag = [f'{TRAIN}-{UNKNOWN}', f'{AP}-{UNKNOWN}', f'{NP}-{UNKNOWN}', f'{PRE}-{UNKNOWN}']
    else:
        prev_tag = [f'{TRAIN}-{UNKNOWN}', f'{AP}-{UNKNOWN}', f'{NP}-{UNKNOWN}', f'{PRE}-{UNKNOWN}']
    print(aid, prev_tag)
    prev_state = prev_tag[aid].split('-')[1]
    
    if prev_state != f'{DONE}':
        tr_done = check_training(exp)
        if aid == 0:
            done = tr_done
        elif aid == 1:
            done = check_AP(exp)
        elif aid == 2:
            done = check_NP(exp)
        elif aid == 3:
            done = check_PRE(exp)
            #start_PRE(exp)

        if done is None:
            cur_state = 'X'
            #unknown.append(exp)
            todo.append(exp)

        elif done:
            cur_state = 'D'
        else:
            if start_job:
                cur_state = 'R'
            else:
                cur_state = 'U'
            if aid == 0:
                todo.append(exp)
            else:
                #if tr_done and arch in large_types:
                if tr_done:
                    todo.append(exp)
        prev_tag[aid] = f"{ACTIONS[aid]}-{cur_state}"

        cur_exp = ','.join(prev_tag) + '#' + exp
        track_list[i] = cur_exp
        print(cur_exp)
        
with open(p_track, 'w') as f:
    for l in track_list:
        f.write(l+'\n')

for l in todo:
    if aid == 0:
        f_todo.write(start_training(l) + '\n')
    elif aid == 1:
        f_todo.write(start_AP(l) + '\n')
    elif aid == 2:
        f_todo.write(start_NP(l) + '\n')
    elif aid == 3:
        f_todo.write(start_PRE(l) + '\n')

for l in unknown:
    f_unknown.write(l + '\n')
 
