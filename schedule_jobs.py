import os
import sys

from common.io import mkpath,mkdir

################################################################
# 
################################################################


todo    = sys.argv[1]
cmds    = open(todo, 'r').readlines()

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
print(f"{env} sh {script} > {log} 2>&1 &")
print(log)
os.system(f"{env} sh {script} > {log} 2>&1 &")
#print(f"{env} sh {script} > {log} 2>&1 &")
