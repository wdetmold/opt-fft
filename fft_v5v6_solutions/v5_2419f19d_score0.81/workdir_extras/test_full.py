import sys, numpy as np
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/fft_recon/sandbox/workdir')
import base, solution

SIZES=(6,8,13,17,23,36,45,64)
def blocks(out, Bs):
    res={}; off=0
    for L in SIZES:
        n=Bs[L]*L**3
        res[(L,'one')]=out[off:off+n]; off+=n
        res[(L,'fin')]=out[off:off+n]; off+=n
    return res

def check(tag, seed, Bs, ms, lim_one=2e-15, lim_fin=1e-4):
    args=[seed]+[Bs[L] for L in SIZES]+[ms[L] for L in SIZES]
    o1=base.transform(*args); o2=solution.transform(*args)
    assert o1.shape==o2.shape and o2.dtype==np.complex128
    b1=blocks(o1,Bs); b2=blocks(o2,Bs)
    worst_one=0; worst_fin=0
    for L in SIZES:
        for kind,lim in (('one',lim_one),('fin',lim_fin)):
            x,y=b1[(L,kind)],b2[(L,kind)]
            if x.size==0: continue
            e=np.linalg.norm(y-x)/max(np.linalg.norm(x),1e-300)
            if kind=='one': worst_one=max(worst_one,e)
            else: worst_fin=max(worst_fin,e)
            assert e<lim, (tag,L,kind,e)
    # determinism
    o3=solution.transform(*args)
    assert np.array_equal(o2,o3), (tag,"nondeterministic!")
    print(f"{tag}: OK  worst_one={worst_one:.2e} worst_fin={worst_fin:.2e}")

check("m=1 all", 7, {L:2 for L in SIZES}, {L:1 for L in SIZES})
check("m=2 all", 8, {L:2 for L in SIZES}, {L:2 for L in SIZES})
check("m=3 all", 9, {L:1 for L in SIZES}, {L:3 for L in SIZES})
check("m mixed parities", 10, {L:2 for L in SIZES}, {6:4,8:5,13:6,17:7,23:8,36:9,45:2,64:5})
check("B=0 for some", 11, {6:0,8:2,13:0,17:1,23:2,36:0,45:1,64:1}, {L:4 for L in SIZES})
check("bigger B", 12, {6:9,8:7,13:5,17:3,23:3,36:3,45:2,64:2}, {L:3 for L in SIZES})
check("long chain", 13, {L:1 for L in SIZES}, {L:50 for L in SIZES}, lim_fin=1e-3)
check("seed large", 2**31-5, {L:1 for L in SIZES}, {L:4 for L in SIZES})
print("ALL FULL TESTS PASSED")
