#!/usr/bin/env python3
"""Strict replay of hot_16d44d13 implementation.c construction, per attempt.log.

Steps (log line refs in attempt.log):
  1. merge a90-regen + impl_3f30.c + run_64 dispatcher      (L4734-4757)
  2. sed -i '1i #include <math.h>'                          (L4767)
  3. patch64.py: run64_alt alternation driver               (L5027-5164)
  4. [append prof64alt tail + later strip = net no-op]      (L5172, L5669)
  5. edit A: MAP_STYLE 3/4 blocks in mapv                   (L5230-5275)
  6. edit B: r2g guard + MAP_STYLE 2 -> 3                   (L5284-5301)
  7. deliverable = merged_alt (prof tail stripped)          (L5669-5676)
  8. impl_m3.c: + impl_mine.c L=13 group engine + routing   (L5946-5979)
  9. final /workdir/implementation.c = dev/impl_m3.c        (L6747)
"""
import re, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, 'w')
WD = os.path.join(HERE, 'work', 'workdir')
DEV = os.path.join(WD, 'dev')

# ---- step 1: initial merge (verbatim from log L4735-4756) ----
a90 = open(f'{W}/a90/dev_generators_final/implementation.c').read()
f30 = open(f'{W}/d43/impl_3f30.c').read()
assert 'void run_64(' in a90
a90 = a90.replace('void run_64(', 'void run_64_a90(')
f30l = []
for ln in f30.split('\n'):
    if ln.startswith('#include'):
        continue
    f30l.append(ln)
f30s = '\n'.join(f30l)
merged = a90 + '\n/* ==== merged engine: v6_3f30 (L=64) ==== */\n' + f30s + r'''

/* dispatcher: route L=64 to the 3f30 engine */
static int f30_inited_;
void run_64(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (!f30_inited_) { init_tables(); f30_inited_ = 1; }
    run64(B, m, x0, c, out1, outm);
}
'''
open(f'{DEV}/impl_merged.c', 'w').write(merged)
print('merged size', len(merged))

# ---- step 2: sed -i '1i #include <math.h>' impl_merged.c (L4767) ----
src = open(f'{DEV}/impl_merged.c').read()
open(f'{DEV}/impl_merged.c', 'w').write('#include <math.h>\n' + src)

# ---- step 3: cp impl_merged.c impl_merged_alt.c && python3 patch64.py (L5164) ----
alt = f'{DEV}/impl_merged_alt.c'
open(alt, 'w').write(open(f'{DEV}/impl_merged.c').read())
r = subprocess.run([sys.executable, os.path.join(HERE, 'patch64.py'), alt],
                   capture_output=True, text=True)
print('patch64:', r.stdout.strip(), r.stderr.strip())
assert r.returncode == 0

# ---- step 4: prof64alt append + strip = net no-op (verified: strip truncates
#      exactly at the appended '#include <stdio.h>\n#include <time.h>\nstatic double nwp(void)') ----

# ---- steps 5-6: editA.py / editB.py operate on impl_merged_alt.c in cwd ----
for s in ('editA.py', 'editB.py'):
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=DEV,
                       capture_output=True, text=True)
    print(s, ':', r.stdout.strip(), r.stderr.strip())
    assert r.returncode == 0

# ---- step 7: deliverable = merged_alt (prof tail already absent here) (L5669) ----
src = open(alt).read()
i = src.find('#include <stdio.h>\n#include <time.h>\nstatic double nwp(void)')
if i > 0:
    src = src[:i]
open(f'{WD}/implementation.c', 'w').write(src)
print('deliverable implementation.c bytes:', len(src))

# ---- step 8: impl_m3.c merge (verbatim from log L5950-5978, paths retargeted) ----
mine = open(f'{W}/d43/impl_mine.c').read()
merged = open(f'{WD}/implementation.c').read()

mine = '\n'.join(ln for ln in mine.split('\n') if not ln.startswith('#include'))
renames = ['alloc_huge_st', 'alloc_huge', 'map2', 'stagger_ctr', 'VONE_', 'VHALF_',
           'run_13', 'run_17', 'run_23', 'run_36', 'run_45', 'run_6', 'run_8']
for name in renames:
    mine = re.sub(r'\b' + name + r'\b', 'f2_' + name, mine)
undefs = '\n'.join(f'#undef {m}' for m in ['ALIGN64', 'TR8', 'VHALF', 'VONE'])

merged = merged.replace('void run_13(const double* x0', 'void run_13_a90(const double* x0')

out = merged + f'''
/* ==== merged engine: d43 impl_mine (L=13 full groups) ==== */
{undefs}
''' + mine + '''
/* dispatcher: L=13: full groups + big remainders -> f2 engine; small remainders -> a90 pv */
void run_13(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    long G = (B / 8) * 8, rem = B - G;
    if (rem >= 6) { G += rem; rem = 0; }
    if (G) f2_run_13(x0, c, out1, outm, G, m);
    if (rem) run_13_a90(x0 + G*2*2197, c + G*2*2197, out1 + G*2*2197, outm + G*2*2197, rem, m);
}
'''
open(f'{DEV}/impl_m3.c', 'w').write(out)
print('impl_m3 ok', len(out))

# ---- step 9: final ----
open(f'{WD}/implementation.c', 'w').write(out)
print('FINAL implementation.c bytes:', len(out))
