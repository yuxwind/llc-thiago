import sys
import os

path = sys.argv[1]

f = open(path, 'r')
flist = [l.strip().split(' ')[5] for l in f.readlines()]
paths = [os.path.dirname(l) for l in flist]
paths = [os.path.join(l, 'preprocesss-train_test.txt') for l in paths]
f.close()

p_rst = os.path.basename(path).strip('todo_PRE_').strip('.sh') + '-pre_results-train_val.txt'

print(p_rst)

f_rst = open(p_rst, 'w')

for p in paths:
    if os.path.exists(p):
        ff = open(p, 'r') 
        rst = ff.readlines()[-1] 
        ff.close()
    else:
        rst = os.path.join(os.path.dirname(p), 'checkpoint_120.tar') + '\n' 
    arr = rst.split(',')
    arr[0] = os.path.basename(os.path.dirname(arr[0]))
    rst = ','.join(arr)
    if not rst.endswith('\n'):
        rst = rst + '\n'
    print(p)
    print(rst)
    f_rst.write(rst)
f_rst.close()

