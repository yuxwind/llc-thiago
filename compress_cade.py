import os
import sys

import glob

from common.io import mkpath

script_dir = 'scripts/compress/compress-net*.sh'
scripts = sorted(glob.glob(script_dir))
print(scripts)
ind = int(sys.argv[1])
script = scripts[ind]

env = "GRB_LICENSE_FILE=~/gurobi-license/`sh ~/get_uname.sh`/gurobi.lic "
log = mkpath(os.path.join(f"logs/compress/{os.path.basename(script).replace('.sh', '.log')}"))

print(f"{env} sh {script}")
os.system(f"{env} sh {script}")

