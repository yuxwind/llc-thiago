set -x 
#for i in `seq 0 19`; do
#    #python schedule_jobs.py track_progress/todo_PRE_mnist.sh $i 20 CUDA_VISIBLE_DEVICES=1 
#    #python schedule_jobs.py track_progress/todo_PRE_cifar100-rgb.sh $i 10 CUDA_VISIBLE_DEVICES=1
#    #python schedule_jobs.py track_progress/todo_PRE_cifar10-rgb.sh $i 20 CUDA_VISIBLE_DEVICES=0
#    python schedule_jobs.py track_progress/todo_PRE_mnist.sh $i 18 CUDA_VISIBLE_DEVICES=1
#done
#for i in `seq 20 39`; do
#    python schedule_jobs.py track_progress/todo_PRE_mnist.sh $i 9 CUDA_VISIBLE_DEVICES=1 
#done
for i in `seq 0 11`; do
    python schedule_jobs.py track_progress/todo_TR_cifar10-rgb.sh $i 29 CUDA_VISIBLE_DEVICES=1 
done
for i in `seq 12 23`; do
    python schedule_jobs.py track_progress/todo_TR_cifar10-rgb.sh $i 29 CUDA_VISIBLE_DEVICES=0
done

## on longclaw
#for i in `seq 24 29`; do
#    python schedule_jobs.py track_progress/todo_TR_cifar10-rgb.sh $i 29 CUDA_VISIBLE_DEVICES=0
#done
#for i in `seq 30 35`; do
#    python schedule_jobs.py track_progress/todo_TR_cifar10-rgb.sh $i 29 CUDA_VISIBLE_DEVICES=1
#done
#for i in `seq 36 41`; do
#    python schedule_jobs.py track_progress/todo_TR_cifar10-rgb.sh $i 29 CUDA_VISIBLE_DEVICES=2
#done
set +x
