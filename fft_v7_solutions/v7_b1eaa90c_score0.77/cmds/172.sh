cd /workdir && python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0,'/workdir')
import solution
# all-zero B
r = solution.transform(9, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1)
print("all-zero B:", r.shape, r.dtype)
# single huge-B small-L sanity (memory/time)
import time
t0=time.perf_counter()
r = solution.transform(9, 3000,0,0,0,0,0,0,0, 10,1,1,1,1,1,1,1)
t1=time.perf_counter()
print(f"B6=3000 m=10: {t1-t0:.2f}s shape {r.shape}")
EOF