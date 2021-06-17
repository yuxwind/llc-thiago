set -x 
#for i in `seq 27 40`; do
#    CUDA_VISIBLE_DEVICES=0 python train_cifar100_rgb.py $i &
#done
#
#for i in `seq 1 6`; do
#    CUDA_VISIBLE_DEVICES=0 python train_cifar10_gray.py $i &
#done
#
#for i in `seq 7 26`; do
#    CUDA_VISIBLE_DEVICES=1 python train_cifar10_gray.py $i &
#done

for i in `seq 0 8`; do
    #python schedule_jobs.py track_progress/todo_PRE_mnist.sh $i 9 CUDA_VISIBLE_DEVICES=0 
    python schedule_jobs.py track_progress/todo_TR_cifar100-rgb.sh $i 1 CUDA_VISIBLE_DEVICES=0 
done
#for i in `seq 20 39`; do
#    python schedule_jobs.py track_progress/todo_PRE_mnist.sh $i 9 CUDA_VISIBLE_DEVICES=1 
#done
#python schedule_jobs.py track_progress/todo_TR_cifar100-rgb.sh 8 10 CUDA_VISIBLE_DEVICES=1 
set +x
