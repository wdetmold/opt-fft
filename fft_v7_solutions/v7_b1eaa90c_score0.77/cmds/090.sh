cd /workdir/dev && python3 - <<'EOF'
# A/B comparison: build two .so variants and alternate benchmarks to cancel drift
import tune
base_cfg = {'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'},'hu':{'13':'16'}}
# variant: L=8 without alternation (plain sqrtnr)
import copy, json
A = tune.build(base_cfg, 'A')
cfgB = copy.deepcopy(base_cfg); cfgB['pw']['45']='sqrtnr'; cfgB['pf']={'45':'256'}
B = tune.build(cfgB, 'B')
import ctypes
for i in range(3):
    ra = tune.bench(A, Ls=(8,45), reps=4)
    rb = tune.bench(B, Ls=(8,45), reps=4)
    print("A", {L: round(v,2) for L,v in ra.items()}, "B", {L: round(v,2) for L,v in rb.items()})
EOF