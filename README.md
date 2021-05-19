# Equivalent and Approximate Transformations of Deep Neural Networks
Code of the paper [Equivalent and Approximate Transformations of Deep Neural Networks](https://arxiv.org/pdf/1905.11428.pdf ) by Abhinav Kumar, Thiago Serra and Srikumar Ramalingam

### Requirements
1. Python 3
2. [Pytorch](http://pytorch.org)
3. Torchvision
4. Cuda 8.0 or higher
5. [Gurobi 7.51](https://www.gurobi.com/downloads/gurobi-optimizer-eula/)

### Cloning the repo
Navigate to the desired directory and clone the repo
```bash
git clone https://github.com/abhi1kumar/bound_nodes_NN.git
```

### Directory structure
We need to make some extra directories to store the dataset models
```bash
cd $PROJECT_DIR

# For storing train datasets
mkdir data

# This directory stores the models of the training in its sub-directories
mkdir -p forward_pass/activation_pattern/models/
```

The directory structure should look like this
```bash
./bound_nodes_NN/
|--- data/
|
|--- forward_pass/
|      |---activation_pattern/
|             |---models/
|
|
|--- local_stability/
|--- plots/
|--- src/
|  ...

```

### Training the models on all combinations of regularisation
```bash
python train_bulk_script.py -s 100 -a fcnn4b
```
This will make directories ```fcnn_run_100``` to ```fcnn_run_111``` inside ```forward_pass/activation_pattern/models/``` folder.

### Transforming to 2- hidden layer network using all possible activation patterns
```bash
python transform_networks_using_all_nodes.py -w ./forward_pass/activation_pattern/models/fcnn_run_100/weights.dat 
```

### Getting the feasible activation patterns
This will invoke the Gurobi optimizer to list out all feasible activation patterns in a file ```activation_pattern_abhinav_input_0_1.dat``` inside the respective directories.

```bash
./count_linear_regions_script.sh -s 100 -e 111
```

### Transforming to 2- hidden layer network using only feasible activation patterns
```bash
python transform_networks_using_all_nodes.py -w ./forward_pass/activation_pattern/models/fcnn_run_100/weights.dat -i ./forward_pass/activation_pattern/models/fcnn_run_100/activation_pattern_abhinav_input_0_1.dat
```

### References
Please cite the following paper if you find the code useful in your research

```
@article{kumar2019equivalent,
  title={Equivalent and Approximate Transformations of Deep Neural Networks},
  author={Kumar, Abhinav and Serra, Thiago and Ramalingam, Srikumar},
  journal={arXiv preprint arXiv:1905.11428},
  year={2019}
}
```

<br/><br/>
***
## Contact
Feel free to drop an email to this address -
```abhinav.kumar@utah.edu```

