import os
import sys


from common.io import mkpath

todo    = sys.argv[1]
cmds    = open(todo, 'r').readlines()

ind     = int(sys.argv[2])
cnt     = int(sys.argv[3])

start   = ind * cnt
end     = min(start + cnt, len(cmds))
sub  = cmds[start:end]

name = os.path.basename(todo).split('.')[0] + f'.{start}_{end}'
script  = os.path.join('./scripts/todo', name + '.sh')
log  = os.path.join(f'./logs/{log}.log')

env = "GRB_LICENSE_FILE=~/gurobi-license/`sh ~/get_uname.sh`/gurobi.lic "
env = ''

open(out, 'w') as f:
    for l in sub:
        f.write(l+'\n')
print(log)
#os.system(f"{env} sh {script} > {log} 2>&1 &")
print(f"{env} sh {script} > {log} 2>&1 &")
