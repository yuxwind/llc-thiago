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

for i in `seq 0 24`; do
    python schedule_jobs.py track_progress/todo_AP_cifar10-rgb.sh $i 10
done
for i in `seq 14 32`; do
    python schedule_jobs.py track_progress/todo_TR_cifar10-rgb.sh $i 6 CUDA_VISIBLE_DEVICES=1 
done
python schedule_jobs.py track_progress/todo_TR_cifar10-rgb.sh 33 7 CUDA_VISIBLE_DEVICES=1 
set +x
