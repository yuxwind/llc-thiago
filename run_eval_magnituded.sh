for i in `seq 0 4`; do 
    python schedule_jobs.py ./track_progress/todo_EVALM_cifar10-rgb.sh $i 27 CUDA_VISIBLE_DEVICES=0
done

for i in `seq 5 9`; do 
    python schedule_jobs.py ./track_progress/todo_EVALM_cifar10-rgb.sh $i 27 CUDA_VISIBLE_DEVICES=1
done

for i in `seq 10 14`; do 
    python schedule_jobs.py ./track_progress/todo_EVALM_cifar10-rgb.sh $i 27 CUDA_VISIBLE_DEVICES=2
done
