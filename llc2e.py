import os
import sys

#types = ["100-100-100-100-100"] # "25-25-25" #["800-800"] # [ "25-25", "50-50", "100-100", "200-200", "400-400", "25-25-25", "50-50-50", "100-100-100" ] 
#types = ["100-100-100-100-100"] 
types = ["100-100", "200-200", "400-400", "800-800", "100-100-100", "100-100-100-100", "100-100-100-100-100"]
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
first_network = int(sys.argv[1])
nb_networks = 1 #5
type_id = int(sys.argv[2])

dataset = "MNIST" # Can also be "CIFAR10" for gray CIFAR10

train_networks = False
test_new_compression = True
test_old_compression = False

time_limit = 7200 #600

model_dir = 'model_dir'

for idx, type in enumerate(types):
    if idx != type_id:
        continue
    for l1 in l1_reg[type]:
        for network in range(first_network,first_network+nb_networks):
            folder = os.path.join(model_dir, "dnn_"+dataset+"_"+type+"_"+str(l1)+"_"+str(network).zfill(4))

            if train_networks:
                os.system("python train_fcnn.py --arch " + type_arch[type] + " --save-dir " + folder + " --l1 " + str(l1) + " --dataset " + dataset)

            if test_old_compression:
                os.system("python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation neuron --time_limit " + str(time_limit) + " --dataset " + dataset)
            if  test_compression:
                os.system("python get_activation_patterns.py -b --input " + folder + "/weights.dat" + " --formulation network --time_limit " + str(time_limit) + " --dataset " + dataset)
