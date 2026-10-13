import numpy as np, os, sys
cases = sys.argv[1]
npass=nfail=0
for line in open(cases):
    n = line.split()[0]
    g = f'output/{n}/golden_y.bin'
    if not os.path.exists(g): continue
    yv = np.fromfile(f'output/{n}/y.bin', dtype=np.float32); gv = np.fromfile(g, dtype=np.float32)
    bad = int((np.abs(yv-gv) > (1e-4+1e-4*np.abs(gv))).sum())
    if bad: print('FAIL', n, bad); nfail+=1
    else: npass+=1
print(f'{cases}: {npass} PASS / {nfail} FAIL')
