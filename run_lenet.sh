set -x
gpu=$1
reg=$2
run=$3
CUDA_VISIBLE_DEVICES=$gpu python train_fcnn.py --arch lenet --save-dir ./model_dir/cifar10/dnn_cifar10_lenet_${reg}_000${run} --l1 ${reg} --dataset CIFAR10-rgb --eval-stable
set +x
