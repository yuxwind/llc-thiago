import os
import sys
import numpy as np

types = ["100-100", "200-200", "400-400", "800-800", "100-100-100", "100-100-100-100", "100-100-100-100-100"] 
#types = ["100-100-100-100-100"] 
type_arch = {"25-25": "fcnn2", 
                "50-50": "fcnn2a", 
                "100-100": "fcnn2b", 
                "200-200": "fcnn2c", 
                "400-400": "fcnn2d", 
                "800-800": "fcnn2e", 
                "25-25-25": "fcnn3", 
                "50-50-50": "fcnn3a", 
                "100-100-100": "fcnn3b", 
                "25-25-25-25": "fcnn4", 
                "50-50-50-50": "fcnn4a", 
                "100-100-100-100": "fcnn4b", 
                "100-100-100-100-100": "fcnn5b"}
c0 = np.arange(0,0.00021, 0.000025)
c1 = np.arange(0,0.00041, 0.000025)
l1_reg = { "25-25": [ 0.001 ], 
            "50-50": [ 0.0, 0.00015, 0.0003 ], 
            "100-100": c1,           
            "200-200": c0, 
            "400-400": c0,
            "800-800": c0, 
            "25-25-25": [0.0003], 
            "50-50-50": [0.0003], 
            "100-100-100": c0, 
            "25-25-25-25": [0.0007], 
            "50-50-50-50": [0.0, 0.0002, 0.0003],
            "100-100-100-100": c0, 
            "100-100-100-100-100": c0}
if len(sys.argv) >= 2:
    type_id = int(sys.argv[1])
else:
    type_id = None
if len(sys.argv) >= 4:
    first_network = int(sys.argv[2])
    last_network = int(sys.argv[3])#5 
else:
    first_network = 1
    last_network = 5 
if len(sys.argv) >= 5:
    lr_idx = [int(s) for s in sys.argv[4].strip().split(',')]
    script_path = f'./compress-net{type_id}-run_{first_network}_{last_network}-{sys.argv[4].strip()}.sh'
else:
    lr_idx = None
    script_path = f'compress-net{type_id}-run_{first_network}_{last_network}.sh'

dataset = "MNIST" # Can also be "CIFAR10" for gray CIFAR10

train_networks = False
test_new_compression = False
test_old_compression = False
collect_results = True

time_limit = 10800 #600

model_dir = './model_dir'
#env = "GRB_LICENSE_FILE=~/gurobi-license/`sh ~/get_uname.sh`/gurobi.lic "
f = open(os.path.join('scripts/compress', script_path), 'w')

if collect_results:
    f_result = open(f'./collected_results.txt', 'w')

def parse_result(line):
    eg_line = './model_dir/dnn_MNIST_100-100-100-100-100_0.0002_0002, 98.08999633789062, , 20, 20, 19,, 26, 26, 16,, 42, 42, 18,,     49, 49, 13,, 28, 28, 16,, 4338.198356389999,, network, relaxation,, 1020,,'
    eg_cnt = len(eg_line.strip().split(','))
    terms = line.strip().replace(' ', '').split(',,')
    name,acc = terms[0].split(',')
    runtime = terms[-3]
    info = terms[-2]
    

for idx,type in enumerate(types):
    if type_id is not None and idx != type_id:
        continue
    #f.write(f'#{type_arch[type]}:\n')
    all_l1 = l1_reg[type] if lr_idx is None else l1_reg[type][lr_idx]
    for l1 in all_l1:
        for network in range(first_network, last_network+1):
            folder = os.path.join(model_dir, "dnn_"+dataset+"_"+type+"_"+str(l1)+"_"+str(network).zfill(4))
            if train_networks:
                f.write("python train_fcnn.py --arch " + type_arch[type] + " --save-dir " + folder + " --l1 " + str(l1) + " --dataset " + dataset + '\n')

            if test_old_compression:
                f.write("python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation neuron --time_limit " + str(time_limit) + " --dataset " + dataset + '\n')
            if test_new_compression:
                f.write("python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation network --time_limit " + str(time_limit) + " --dataset " + dataset + '\n')
            if collect_results:
                exp_path = os.path.join('./results', os.path.basename(folder) + '.txt')
                if os.path.exists(exp_path):
                    with open(exp_path, 'r') as f_exp:
                        s_exp  = f_exp.readlines()
                        if len(s_exp) >= 2:
                            print('run multiple times: ', exp_path)
                        if len(s_exp) == 0:
                            print('result not ready: ', exp_path)
                        for line in s_exp:
                            line = './model_dir/' + line.strip().split('./model_dir/')[-1]
                            f_result.write(line + '\n')
                else:
                    print(f'File not exists: {exp_path}')
f.close()
if collect_results:
    f_result.close()
