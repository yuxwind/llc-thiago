cat collected_results.txt |awk -F ',' '{print $1}'|awk -F 'MNIST_' '{print $2}'|awk -F '_000' '{print $1}'|awk -F '_' '{print $2}' > exp_lr.txt
