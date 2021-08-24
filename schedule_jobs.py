import os
import sys

from common.io import mkpath,mkdir

################################################################
# 
################################################################
try:
    from dir_lookup import imbalance_cfg
    cfg_info = f'.{imbalance_cfg.keep_ratio:.04f}_1'
except:
cfg_info = ''


todo    = sys.argv[1]
#todo = 'track_progress/todo_OD_cifar10_cifar100_large_net.sh'
#todo = './track_progress/todo_AP_mnist.sh'
#todo = './track_progress/todo_AP_cifar10-rgb.sh'
todo = './track_progress/todo_AP_cifar10-rgb.sh{cfg_info}'
cmds    = open(todo, 'r').readlines()
#todo2 = 'track_progress/todo_OD_cifar10_cifar100_large_net.sh'
#cmds2    = open(todo2, 'r').readlines()


ind     = int(sys.argv[2])
cnt     = int(sys.argv[3])

start   = ind * cnt
end     = min(start + cnt, len(cmds))
sub  = cmds[start:end]

name = os.path.basename(todo).split('.')[0] + f'.{start}_{end}'
script  = mkpath(os.path.join('./scripts/todo', name + '.sh'))
log  = os.path.join(f'./logs/{name}.log')

# on cade
env = "GRB_LICENSE_FILE=~/gurobi-license/`sh ~/get_uname.sh`/gurobi.lic "
# on multiple GPU envs
#env = 'CUDA_VISIBLE_DEVICES=0'
#env = sys.argv[4]

with open(script, 'w') as f:
    for l in sub:
        f.write(l)
    #for l in cmds2[start:end]:
    #    f.write(l)
print(f"{env} sh {script} > {log} 2>&1 &")
print(log)
os.system(f"{env} sh {script} > {log} 2>&1 &")
#print(f"{env} sh {script} > {log} 2>&1 &")
