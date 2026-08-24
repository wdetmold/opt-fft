cd /workdir && rm -f implementation.so && python3 -c "
import time
t0=time.time()
import sys; sys.path.insert(0,'/workdir')
import solution
print(f'fresh import+compile: {time.time()-t0:.1f}s')
import numpy as np
r1 = solution.transform(3, 4,3,2,2,2,1,1,1, 7,7,7,7,7,7,7,7)
r2 = solution.transform(3, 4,3,2,2,2,1,1,1, 7,7,7,7,7,7,7,7)
print('deterministic:', np.array_equal(r1, r2))
print('dtype/shape:', r1.dtype, r1.shape, 2*(4*216+3*512+2*2197+2*4913+2*12167+46656+91125+262144))
"