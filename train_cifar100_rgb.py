import os
import sys

import glob

from common.io import mkpath

info = 'train/CIFAR100-rgb'

script_dir = f'scripts/{info}/net*.sh'
scripts = sorted(glob.glob(script_dir))
for i, s in enumerate(scripts):
    print(i+1, s)
ind = int(sys.argv[1]) -  1
script = scripts[ind]

env = "GRB_LICENSE_FILE=~/gurobi-license/`sh ~/get_uname.sh`/gurobi.lic "
log = mkpath(os.path.join(f"logs/{info}/{os.path.basename(script).replace('.sh', '.log')}"))

print(f"{i+1}: {script}")
os.system(f"{env} sh {script}")

