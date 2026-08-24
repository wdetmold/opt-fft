cd /workdir/dev && python3 - <<'EOF'
# Simulate the alternating-PFA-45 pipeline on INDICES to derive & verify all maps.
# State between sweeps: "stage1 outputs": value A[coset_id][k1] laid out in slots.
# Even sweep consumes 9pt-stage1 outputs (A9[b][k1], b in [0,5) coset-id, k1 in [0,9)),
#   slot layout: slot(5*k1 + b) = A9[b][k1]  (groups by k1: 5 slabs)
# Even sweep: for k1: 5pt over b of A9[b][k1] -> X[k] with k%9==k1? verify:
import numpy as np
def dft(N): 
    j=np.arange(N); return np.exp(-2j*np.pi*np.outer(j,j)/N)
rng=np.random.default_rng(0)
x=rng.standard_normal(45)+1j*rng.standard_normal(45)
X=np.fft.fft(x)
# 9pt stage1 (over a), input map n=(5a+9b)%45: A9[b][k1] = sum_a W9^{a k1} x[(5a+9b)%45]
A9=np.zeros((5,9),complex)
for b in range(5):
    xs=np.array([x[(5*a+9*b)%45] for a in range(9)])
    A9[b]=dft(9)@xs
# even-stage2: 5pt over b: for each k1: Y[k2p] = sum_b W5^{b k2p} A9[b][k1]
# claim Y[k2p] = X[k] with k%9 == ??? and k%5 == ???
for k1 in range(9):
    Y=dft(5)@A9[:,k1]
    for k2p in range(5):
        # find k
        diffs=[abs(Y[k2p]-X[k]) for k in range(45)]
        k=int(np.argmin(diffs))
        assert diffs[k]<1e-9, (k1,k2p,diffs[k])
        # record mapping
        if k1==0: pass
        # verify k%9, relation of k%5 to (k1,k2p)
        assert k%9 == (5*k1)%9 or True
        print(f"k1={k1} k2p={k2p} -> k={k} (k%9={k%9}, k%5={k%5})", end="; ")
    print()
EOF