exp=$1
gpu=$2
echo ==================
echo $exp
CUDA_VISIBLE_DEVICES=$gpu python train_fcnn.py --arch lenet --resume $exp/checkpoint_120.tar  -e --eval-stable --eval-train-data --dataset CIFAR10-rgb
