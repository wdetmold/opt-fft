cd /workdir && cp dev/tune.py /tmp/ && python3 - <<'EOF'
import sys, time, numpy as np
sys.path.insert(0, '/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
# rebuild solution .so from current implementation.c
import subprocess, os
os.chdir('/workdir/dev')
import json
env = dict(os.environ); env['GENCFG'] = json.dumps({'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'}})
subprocess.run([sys.executable,'gen.py'], check=True, env=env)
os.chdir('/workdir')
subprocess.run(['gcc','-O3','-march=native','-shared','-fPIC','implementation.c','-o','implementation.so','-lm'], check=True)
import solution
Bs=(2315,977,171,60,24,11,5,2)   # ~0.5M elems each
ms=(290,280,180,130,120,110,100,90)
args=(7,)+Bs+ms
t0=time.perf_counter(); r1=solution.transform(*args); t1=time.perf_counter()
t2=time.perf_counter(); r2=solution.transform(*args); t3=time.perf_counter()
print(f"solution: {t1-t0:.2f}s / {t3-t2:.2f}s")
t4=time.perf_counter(); rb=base.transform(*args); t5=time.perf_counter()
print(f"base:     {t5-t4:.2f}s  ratio={min(t1-t0,t3-t2)/(t5-t4):.4f}")
err=np.linalg.norm(r2-rb)/np.linalg.norm(rb)
print("overall rel err:", err)
EOF