set -x 
for i in `seq 27 40`; do
    CUDA_VISIBLE_DEVICES=0 python train_cifar100_rgb.py $i &
done

for i in `seq 1 6`; do
    CUDA_VISIBLE_DEVICES=0 python train_cifar10_gray.py $i &
done

for i in `seq 7 26`; do
    CUDA_VISIBLE_DEVICES=1 python train_cifar10_gray.py $i &
done

set +x
