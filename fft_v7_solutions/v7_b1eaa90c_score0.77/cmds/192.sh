cd /workdir && python3 - <<'EOF'
import sys, numpy as np, random
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
rnd = random.Random(12345)
trials = []
for trial in range(40):
    Bs = tuple(rnd.choice([0,1,2,3,4,5,6,7,8,9,11,13,16,17]) if L<20 else rnd.choice([0,1,2,3,4,5]) for L in SIZES)
    ms = tuple(rnd.choice([1,2,3,4,5,7,10,15,25]) for _ in SIZES)
    seed = rnd.randrange(0, 10**6)
    trials.append((seed,Bs,ms))
t37 = trials[37]
print("trial37:", t37)
# run trial37 alone
def check45(args):
    g=solution.transform(*args); r=base.transform(*args)
    Bs=args[1:9]
    off=0
    for L,B in zip(SIZES,Bs):
        n=B*L**3
        if L==45 and n:
            e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
            return e1
        off+=2*n
    return None
args37=(t37[0],)+t37[1]+t37[2]
print("trial37 alone: one-step-45 err =", check45(args37))
# now run trial36 then 37
args36=(trials[36][0],)+trials[36][1]+trials[36][2]
solution.transform(*args36)
print("after trial36: err =", check45(args37))
# try full prefix
import importlib
importlib.reload(solution)
for k in range(37):
    a=(trials[k][0],)+trials[k][1]+trials[k][2]
    solution.transform(*a)
print("after full prefix: err =", check45(args37))
print("trial36 params:", trials[36])
EOF