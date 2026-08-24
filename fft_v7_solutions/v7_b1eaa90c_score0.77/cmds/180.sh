cd /workdir && python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0,'/workdir')
import solution
a1 = solution.transform(42, 3,3,3,3,3,2,2,2, 9,8,7,6,5,4,3,2)
b  = solution.transform(17, 1,5,2,9,1,3,1,1, 4,4,4,4,4,4,4,4)
a2 = solution.transform(42, 3,3,3,3,3,2,2,2, 9,8,7,6,5,4,3,2)
print("cross-call determinism:", np.array_equal(a1, a2))
EOF