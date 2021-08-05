import os
import numpy as np 

lines = open

f = open('data.txt', 'r')
arr = [l.split('\t') for l in f.readlines()] 

d = dict()
i = 0
arch_,rg_ = '',0
for l in arr:
    c1,c2,arch,rg = l
    c1 = float(c1)
    c2 = float(c2)
    rg = rg.strip()
    if arch == '':
        arch = arch_
    else:
        arch_= arch
    if rg  == '':
        rg = rg_
    else:
        rg = float(rg.strip())
        rg_= rg

    d.setdefault(arch, {})
    d[arch].setdefault(rg, [[],[]])
    d[arch][rg][0].append(c1)
    d[arch][rg][1].append(c2)

r = []

for arch,v in d.items():
    for rg,c in v.items():
        c1 = sorted(c[0])
        c2 = sorted(c[1])
        c1_m = c1[len(c1)//2]
        c2_m = c2[len(c2)//2]
        r.append(c1_m/c2_m)
        print(f'{arch}\t{rg}\t{c1_m:.2f}\t{c2_m:.2f}\t{c1_m/c2_m:.2f}')
print('total: ',np.array(r).mean())
