import os
import sys
from glob import glob

from common.io import mkdir,mkpath
from cfg import type_arch 
####################################################
# progress: if the model path starts with 
#   #: training is done
#   
####################################################

TRAIN       = 'TR'      # training
AP          = 'AP'      # couting stable with preprocessing all training samples
NP          = 'NP'      # couting stable without preprocessing
#PP         = '#pp:'    # couting stable with preprocessing partial training sampls
RUN         = 'R'       # it is running
DONE        = 'D'       # it is done
UNDONE      = 'U'       # it is not finished
UNKNOWN     = 'X'       # it is unknown
ACTIONS     = [TRAIN, AP, NP]

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
cifar10_gray_track_list = open(p_cifar10_gray, 'r').readlines()
cifar10_rgb_track_list = open(p_cifar10_rgb, 'r').readlines()
cifar100_rgb_track_list = open(p_cifar100_rgb, 'r').readlines()


# check whether training is done
def check_training(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    model_path = os.path.join(model_dir, dataset, os.path.basename(model_name), 'checkpoint_120.tar')
    if os.path.exists(model_path):
        return True
    else:
        return False

# check whether MILP with preprocessing all training samples is done
def check_AP(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    rst_path = os.path.join(rst_dir, dataset, ALLPRE, cnt_rst, os.path.basename(model_name) + '.txt')
    if not os.path.exists(rst_path):
        return False
    rst = [l.stript() for l in open(rst_path, 'r').readlines()]
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
    return f"{env} {cmd} > {log} 2>&1 & "

def start_AP(model_name):
    time_limit = 10800
    env = "GRB_LICENSE_FILE=~/gurobi-license/`sh ~/get_uname.sh`/gurobi.lic "
    
    terms = os.path.basename(model_name).split('_')
    dataset = terms[1]
    folder = mkdir(os.path.join(model_dir, dataset, os.path.basename(model_name)))

    cmd = "python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation network --time_limit " + str(time_limit) + " --dataset " + dataset
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
else:
    print('Unknown action')

todo = []
unknown = []
f_todo = open(f'./track_progress/todo_{ACTIONS[aid]}_{dataset}.txt', 'w')
f_unknow = open(f'./track_progress/unknow_{ACTIONS[aid]}_{dataset}.txt', 'w')

# example of tag format: TR-D, AP-D, NP-X 
for i,l in enumerate(track_list):
    arrs = l.strip().split('#')
    exp = arrs[-1]
    if len(arrs) == 2:
        prev_tag = arrs[1].split(',')
    else:
        prev_tag = [f'{TRAIN}-X', f'{AP}-X', f'{NP}-X']
    prev_state = prev_tag[aid].split('-')[1]
    
    if prev_state != f'{DONE}':
        if aid == 0:
            done = check_training(exp)
        elif aid == 1:
            done = check_AP(exp)
        else:
            done = check_NP(exp)
        if done is None:
            cur_state = 'X'
            unknow.append(exp)
        elif done:
            cur_state = 'D'
        else:
            if start_job:
                cur_state = 'R'
            else:
                cur_state = 'U'
            todo.append(exp)
        prev_tag[aid] = f"{ACTIONS[aid]}-{cur_state}"

        cur_exp = ','.join(prev_tag) + '#' + exp
        track_list[i] = cur_exp
        print(cur_exp)
        
with open(p_track, 'w') as f:
    import pdb;pdb.set_trace()
    for l in track_list:
        f.write(l+'\n')

for l in todo:
    f_todo.write(start_training(l) + '\n')
for l in unknown:
    f_unknown.wreit(l + '\n')
 
