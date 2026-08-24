cd /workdir/dev && python3 - <<'EOF'
# bake best config as defaults in gen.py
import json
best = json.load(open('best_cfg.json'))
src = open('gen.py').read()
old = """CONFIG = {'pw': {}, 'pf': {}}"""
new = f"""CONFIG = {{'pw': {json.dumps(best['pw'])}, 'pf': {json.dumps(best['pf'])}, 'hu': {json.dumps(best['hu'])}, 'scheme': {json.dumps(best['scheme'])}}}"""
assert old in src
src = src.replace(old, new)
# deep-merge env config over defaults
old2 = """if os.environ.get('GENCFG'):
    CONFIG.update(json.loads(os.environ['GENCFG']))"""
new2 = """if os.environ.get('GENCFG'):
    _e = json.loads(os.environ['GENCFG'])
    for _k, _v in _e.items():
        if isinstance(_v, dict):
            CONFIG.setdefault(_k, {}).update(_v)
        else:
            CONFIG[_k] = _v"""
assert old2 in src
src = src.replace(old2, new2)
open('gen.py','w').write(src)
print("baked")
EOF
python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 dev/check.py && python3 - <<'EOF'
import sys, time, numpy as np
sys.path.insert(0,'/workdir')
import importlib
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
Bs=(2315,977,171,60,24,11,5,2)
ms=(290,280,180,130,120,110,100,90)
args=(7,)+Bs+ms
best=1e9
for _ in range(2):
    t0=time.perf_counter(); r=solution.transform(*args); t1=time.perf_counter()
    best=min(best,t1-t0)
t4=time.perf_counter(); rb=base.transform(*args); t5=time.perf_counter()
print(f"solution {best:.2f}s base {t5-t4:.2f}s ratio {best/(t5-t4):.4f}")
EOF