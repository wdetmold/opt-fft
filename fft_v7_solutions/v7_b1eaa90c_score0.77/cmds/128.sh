cd /workdir/dev && grep -n "PFON\|sqpf" gen.py | head -5; python3 - <<'EOF'
# Set final baked defaults
import json
src = open('gen.py').read()
import re
m = re.search(r"CONFIG = \{[^\n]*\}", src)
print("current:", m.group(0))
new = """CONFIG = {'pw': {'13': 'alt', '17': 'alt', '23': 'sqrtnr', '45': 'sqrtnr', '6': 'alt', '8': 'sqrtnr'}, 'pf': {}, 'hu': {'13': '16', '17': '16', '23': '1'}, 'scheme': {}, 'hm2': {}, 'nt': {}, 'k3pf': {}, 'sqpw': {}}"""
src = src[:m.start()] + new + src[m.end():]
open('gen.py','w').write(src)
print("baked final defaults")
EOF
python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 dev/check.py | tail -1 && cd dev && python3 -c "
import tune
so='/workdir/implementation.so'
import numpy as np
allr = {}
for i in range(4):
    r=tune.bench(so, reps=4)
    for L,v in r.items(): allr.setdefault(L,[]).append(v)
print('FINAL per-size ns/elem-iter:', {L: round(min(v),2) for L,v in allr.items()})
"