net=$1
sh train1_net${net}.sh > logs/train1_net${net}.log 2>&1 &
sh train2_net${net}.sh > logs/train2_net${net}.log 2>&1 &
sh train3_net${net}.sh > logs/train3_net${net}.log 2>&1 &
sh train4_net${net}.sh > logs/train4_net${net}.log 2>&1 &
sh train5_net${net}.sh > logs/train5_net${net}.log 2>&1 &
