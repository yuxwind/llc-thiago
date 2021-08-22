import os
import sys
from glob import glob

from common.io import mkdir,mkpath
from common.time import today_, now_
from cfg import type_arch,small_types,large_types 
from dir_lookup import collect_rst, cnt_rst, stb_neuron
from dir_lookup import model_root as model_dir
from dir_lookup import stb_root as rst_dir

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
OD          = 'OD'      # old approach to count stable 
PRUNE       = 'PRUNE'   # prune the network with the neuron stability 
PRUNEM      = 'PRUNEM'  # prune the network with the magnitude based pruning 
EVALM       = 'EVALM'   # evaluate the pruned model by magnitude based algorithm
EVAL       = 'EVAL'   # evaluate the model 
EVALL       = 'EVALL'   # evaluate the llc model  

RUN         = 'R'       # it is running
DONE        = 'D'       # it is done
UNDONE      = 'U'       # it is not finished
UNKNOWN     = 'X'       # it is unknown
ACTIONS     = [TRAIN, AP, NP, PRE, OD, PRUNE, PRUNEM, EVALM, EVAL, EVALL]
act_dict    = dict(zip(ACTIONS, range(len(ACTIONS))))

#model_dir   = './model_dir'
#rst_dir     = './results/'
#rst_dir     = './results-restrict_input/'
#cnt_rst     = 'counting_results/'
#stb_neuron  = 'stable_neurons/'
import pdb;pdb.set_trace()

#SCRIPT      = 'get_activation_patterns.py'
SCRIPT      = 'nn_milp_per_network.py'

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

mnist_track_list = open(p_mnist + '.orig', 'r').readlines()
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

def check_EVAL(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    eval_path = os.path.join(model_dir, dataset, os.path.basename(model_name), 'eval.txt')
    if os.path.exists(eval_path):
        f = open(eval_path, 'r')
        # format: model_name,acc_orig,acc_llc,acc_mc
        acc_orig = f.readlines()[-1].split(',')[1].strip()
        if acc_orig == '-':
            return False
        else:
            return True
    else:
        return False

def start_EVAL(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    arch = os.path.basename(model_name).split('_')[2]
    model_root = os.path.join(model_dir, dataset, os.path.basename(model_name))
    cmd = "python train_fcnn.py --arch " + type_arch[arch] + " --resume " + \
            os.path.join(model_root, 'checkpoint_120.tar') + \
            " -e " + " --dataset " + dataset
    return cmd

def check_EVALM(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    eval_path = os.path.join(model_dir, dataset, os.path.basename(model_name), 'eval.txt')
    if os.path.exists(eval_path):
        f = open(eval_path, 'r')
        # format: model_name,acc_orig,acc_llc,acc_mc
        rst = f.readlines()[-1].split(',')
        acc_mc = rst[3].strip()
        if acc_mc == '-' or len(rst) < 7:
            return False
        else:
            return True
    else:
        return False

def start_EVALM(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    arch = os.path.basename(model_name).split('_')[2]
    model_root = os.path.join(model_dir, dataset, os.path.basename(model_name))
    cmd = "python train_fcnn.py --arch " + type_arch[arch] + " --resume " + \
            os.path.join(model_root, 'magnitude_pruned_checkpoint_120.tar') + \
            " -e " + " --dataset " + dataset
    return cmd
# check whether MILP with preprocessing all training samples is done
def check_OD(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    rst_path = os.path.join(rst_dir, dataset, OLD, cnt_rst, os.path.basename(model_name) + '.txt')
    if not os.path.exists(rst_path):
        return False
    rst = [l.strip() for l in open(rst_path, 'r').readlines()]
    for l in rst:
        if 'neuron' in l and '-,' not in l:
            return True
    return None

# check whether MILP with preprocessing all training samples is done
def check_AP(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    rst_path = os.path.join(rst_dir, dataset, ALLPRE, cnt_rst, os.path.basename(model_name) + '.txt')
    if not os.path.exists(rst_path):
        return False
    rst = [l.strip() for l in open(rst_path, 'r').readlines()]
    for l in rst:
        if 'relaxation' in l and '-,' not in l:
            return True
    return None

# check whether MILP without preprocessing is done
def check_NP(model_name):
    dataset = os.path.basename(model_name).split('_')[1]
    rst_path = os.path.join(rst_dir, dataset, NOPRE, cnt_rst, os.path.basename(model_name) + '.txt')
    if not os.path.exists(rst_path):
        return False
    rst = [l.strip() for l in open(rst_path, 'r').readlines()]
    for l in rst:
        if 'relaxation' in l and '-,' not in l:
            return True
    return None

def start_training(model_name):
    terms = os.path.basename(model_name).split('_')
    dataset = terms[1]
    arch    = terms[2]
    l1      = terms[3]
    folder = mkdir(os.path.join(model_dir, dataset, os.path.basename(model_name)))
    
    cmd = "python train_fcnn.py --arch " + type_arch[arch] + " --save-dir " + folder  + " --l1 " + l1 + " --dataset " + dataset + " --eval-stable "
    log = mkpath(f"logs/training/{os.path.basename(model_name)}.log")
    #print(log)
    return cmd

def check_PRUNE(model_name):
    if os.path.exists(os.path.join(model_name, 'pruned_checkpoint_120.tar')):
        return True
    else:
        return False

def start_PRUNE(model_name):
    cmd = "python prune_network.py " + model_name
    return cmd

def check_PRUNEM(model_name):
    if os.path.exists(os.path.join(model_name, 'magnitude_pruned_checkpoint_120.tar')):
        return True
    else:
        return False

def start_PRUNEM(model_name):
    cmd = "python prune_network.py " + model_name + "  magnitude "
    return cmd

def start_OD(model_name):
    time_limit = 10800
    
    terms = os.path.basename(model_name).split('_')
    dataset = terms[1]
    folder = mkdir(os.path.join(model_dir, dataset, os.path.basename(model_name)))

    cmd = "python " + SCRIPT + " -b --input " + folder + "/weights.dat" + " --formulation neuron --time_limit " + str(time_limit) + " --dataset " + dataset + " --preprocess_all_samples"
    return cmd

def start_AP(model_name):
    time_limit = 10800
    
    terms = os.path.basename(model_name).split('_')
    dataset = terms[1]
    folder = mkdir(os.path.join(model_dir, dataset, os.path.basename(model_name)))

    cmd = "python " + SCRIPT + " -b --input " + folder + "/weights.dat" + " --formulation network --time_limit " + str(time_limit) + " --dataset " + dataset + " --preprocess_all_samples"
    return cmd

def start_NP(model_name):
    time_limit = 10800
    
    terms = os.path.basename(model_name).split('_')
    dataset = terms[1]
    folder = mkdir(os.path.join(model_dir, dataset, os.path.basename(model_name)))

    cmd = "python " + SCRIPT + " -b --input " + folder + "/weights.dat" + " --formulation network --time_limit " + str(time_limit) + " --dataset " + dataset
    return cmd


dataset = sys.argv[1] # this can be 'mnist', 'cifar10-gray', 'cifar10-rgb', 'cifar100-rgb' 
action = sys.argv[2] # this can be 'tr', 'ap', 'np', 'prune', 'prune_magnitude'
#start_job   = int(sys.argv[3])   # this can be 1 or 0 to indicate whether to start running   

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
elif action == 'old':
    aid = 4
elif action == 'prune':
    aid = 5
elif action == 'prune_magnitude':
    aid = 6
elif action == 'eval_magnitude':
    aid = 7
elif action == 'eval':
    aid = 8
elif action == 'eval_llc':
    aid = 9
else:
    print('Unknown action')

todo = []
unknown = []
track_dir = './track_progress'
name_todo = f'todo_{ACTIONS[aid]}_{dataset}.sh'
name_unknown = f'unknow_{ACTIONS[aid]}_{dataset}.sh'
path_todo = os.path.join(track_dir, name_todo)
path_unknown = os.path.join(track_dir, name_unknown)


name_todo_rst = f'todo_{ACTIONS[aid]}_{dataset}_rst.sh'
name_unknown_rst = f'unknow_{ACTIONS[aid]}_{dataset}_rst.sh'
path_todo_rst = os.path.join(track_dir, name_todo_rst)
path_unknown_rst = os.path.join(track_dir, name_unknown_rst)

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
f_todo_rst   = open(path_todo_rst, 'w')
f_unknown_rst = open(path_unknown_rst, 'w')

# example of tag format: TR-D, AP-D, NP-X 
for i,l in enumerate(track_list):
    l = l.strip()
    if l == '\n' or l == '':
        continue
    arrs = l.strip().split('#')
    exp = arrs[-1]
    arch = os.path.basename(exp).split('_')[2]
    #if len(arrs) == 2:
    #    prev_tag = arrs[0].split(',')
    #    # get states for each actions
    #    states = {}
    #    for tag in prev_tag:
    #        t,s = tag.split('-')
    #        states[t] = s
    #    # set the state of missed action as UNKNOWN 
    #    for t in ACTIONS:
    #        if t not in states:
    #            states[t] = UNKNOWN
    #    # update prev_tag 
    #    prev_tag = ['-'.join([t, states[t]]) for t in ACTIONS]
    #else:
    #    prev_tag = [f'{TRAIN}-{UNKNOWN}', f'{AP}-{UNKNOWN}', f'{NP}-{UNKNOWN}', f'{PRE}-{UNKNOWN}', 
    #                f'{OD}-{UNKNOWN}', f'{PRUNE}-{UNKNOWN}']
    #prev_state = prev_tag[aid].split('-')[1]

    #if prev_state != f'{DONE}':
    if True:  #always check the states 
        tr_done = check_training(exp)
        ap_done = check_AP(exp) 
        if aid == 0:
            done = tr_done
        elif aid == 1:
            done = check_AP(exp)
        elif aid == 2:
            done = check_NP(exp)
        elif aid == 3:
            done = check_PRE(exp)
            #start_PRE(exp)
        elif aid == 4:
            done = check_OD(exp)
        elif aid == 5:
            done = check_PRUNE(exp)
        elif aid == 6:
            done = check_PRUNEM(exp)
        elif aid == 7:
            done = check_EVALM(exp)
        elif aid == 8:
            done = check_EVAL(exp)
        elif aid == 9:
            done = check_EVALML(exp)

        #done=False
        if done is None:
            #cur_state = 'X'
            unknown.append(exp)
            #todo.append(exp)
        elif done:
            cur_state = 'D'
        else:
            #if start_job:
            #    cur_state = 'R'
            #else:
            #    cur_state = 'U'
            # TRAIN should be done before starting other actions
            if aid == 0:
                todo.append(exp)
            elif aid == 5:
                if ap_done:
                    todo.append(exp)
            else:
                if not tr_done:
                    import pdb;pdb.set_trace()
                    print(exp)
                #if tr_done and arch in large_types:
                if tr_done:
                    todo.append(exp)
        #prev_tag[aid] = f"{ACTIONS[aid]}-{cur_state}"

        #cur_exp = ','.join(prev_tag) + '#' + exp.strip()
        #track_list[i] = cur_exp
        ##print(cur_exp)
        
#with open(p_track, 'w') as f:
#    for l in track_list:
#        f.write(l+'\n')
import pdb;pdb.set_trace()
print('model_dir: ', model_dir)
print(path_todo)
print('todo: ', len(todo))
print(path_unknown)
print('unknown: ', len(unknown))
for l in todo:
    if aid == 0:
        f_todo.write(start_training(l) + '\n')
    elif aid == 1:
        f_todo.write(start_AP(l) + '\n')
    elif aid == 2:
        f_todo.write(start_NP(l) + '\n')
    elif aid == 3:
        f_todo.write(start_PRE(l)+ '\n')
    elif aid == 4:
        f_todo.write(start_OD(l) + '\n')
    elif aid == 5:
        f_todo.write(start_PRUNE(l) + '\n')
    elif aid == 6:
        f_todo.write(start_PRUNEM(l) + '\n')
    elif aid == 7:
        f_todo.write(start_EVALM(l) + '\n')
    elif aid == 8:
        f_todo.write(start_EVAL(l) + '\n')
    elif aid == 9:
        f_todo.write(start_EVALL(l) + '\n')
    for ll in collect_rst(l, ACTIONS[aid]):
        f_todo_rst.write(ll + '\n')

for l in unknown:
    f_unknown.write(l + '\n')
    for ll in collect_rst(l, ACTIONS[aid]):
        f_unknown_rst.write(ll + '\n')
