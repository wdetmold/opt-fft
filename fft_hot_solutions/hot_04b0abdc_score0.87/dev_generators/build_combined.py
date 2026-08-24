#!/usr/bin/env python3
# Build the combined implementation.c:
#  - prior 00291a90 engine (verbatim) for L=6,8,13,17,23 (+ its 36/45/64 renamed away)
#  - my engines for 36,45,64 (e64_nopf.c + pfa.c)
import re, sys, os

prior = open('/tmp/w/00291a90/implementation_final.c').read()
# rename prior run_36/45/64 so they don't collide; keep everything else
for L in (36,45,64):
    prior = prior.replace(f'void run_{L}(', f'void prior_run_{L}(')
# prior's run_36 dispatcher calls run_36A/run_36B internally; it became prior_run_36 - fine.

mine64 = open('/tmp/w/eng/e64v8.c').read()
# strip test helpers and profiled runner from e64
for marker in ('// ---- test helpers ----',):
    if marker in mine64:
        mine64 = mine64[:mine64.index(marker)]
pfa = open('/tmp/w/eng/pfa.c').read()
pfa_impl = open('/tmp/w/eng/pfa_impl.h').read()
# strip test helpers in pfa_impl
i = pfa_impl.index('void SUF(t_round)')
j = pfa_impl.index('#undef NYB')
pfa_impl = pfa_impl[:i] + pfa_impl[j:]
# remove profiling counters from pfa_impl (TCP refs fine, keep)
pfa = pfa.replace('#include "pfa_impl.h"', '@@PFA_IMPL@@')
pfa = pfa.replace('@@PFA_IMPL@@', pfa_impl, 1)
pfa = pfa.replace('@@PFA_IMPL@@', pfa_impl.replace('DFT5','DFT5_'), 0) if False else pfa
# second include: replace remaining marker
pfa = pfa.replace('@@PFA_IMPL@@', pfa_impl)

# namespace my code to avoid collisions with prior (TR8, MAP2, rdt0, alloc_big2...)
def ns(src, names):
    for n in names:
        src = re.sub(r'\b'+n+r'\b', 'MY_'+n, src)
    return src
common = ['TR8','MAP2','ALIGN64','rdt0','alloc_big2','alloc_big','TCP','get_tcp','TC','get_tc',
          'DFT8','DECL16','LOAD16','STORE16','CTW','TW7','TW64','S8C','SCRA','SCRB','ds',
          'DFT3','DFT9','DFT5','DFT4','CTW2','TW9','init_tw9','C3','S3','C51','C52','S51','S52',
          'X64','CS64','CP64','RS64','PS64','MAPV','pf_t1']
mine64 = ns(mine64, common)
pfa = ns(pfa, common)
# avoid collisions with prior engine's internal names (trin_45, SB_45, ...)
pfa = pfa.replace('#define SUF(x) x##_45', '#define SUF(x) x##_n45')
pfa = pfa.replace('#define SUF(x) x##_36', '#define SUF(x) x##_n36')
# remove duplicate rdt0/get_tcp block in pfa (mine64 already has MY_rdt0? no: e64 has rdt() via MY_rdt0... pfa defines MY_rdt0 again)
pfa = pfa.replace('''static inline unsigned long long MY_rdt0(void){ unsigned a,d,c; __asm__ volatile("rdtscp":"=a"(a),"=d"(d),"=c"(c)); return ((unsigned long long)d<<32)|a; }
static unsigned long long MY_TCP[16];
void MY_get_tcp(unsigned long long* o){ for(int i=0;i<16;i++){o[i]=MY_TCP[i]; MY_TCP[i]=0;} }''','''static unsigned long long MY_TCP[16];''')
pfa += '''
void run_45(const double* x0, const double* c, double* out1, double* outm, long B, long m){ run_n45(x0,c,out1,outm,B,m); }
void prior_run_36(const double* x0, const double* c, double* out1, double* outm, long B, long m);
void run_36(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    long rem = B % 8, G8 = B - rem;
    if(G8) prior_run_36(x0, c, out1, outm, G8, m);
    if(rem) run_n36(x0 + G8*2*46656, c + G8*2*46656, out1 + G8*2*46656, outm + G8*2*46656, rem, m);
}
'''
# MAPV define for e64 (it uses #if MY_MAPV... fix: set directly)
mine64 = '#define MY_MAPV 1\n' + mine64

out = []
out.append('/* Combined engine: prior-work kernels (L=6,8,13,17,23) + new two-stage')
out.append(' * register-flow engines (L=36,45,64). Single-threaded, no FFT libraries. */')
out.append(prior)
out.append('/* ===================== new 64 engine ===================== */')
out.append(mine64)
out.append('/* ===================== new 36/45 engines ===================== */')
# pfa includes immintrin etc again: fine (idempotent), but duplicate static names like TR8 macro remain inside pfa (MY_TR8) - already renamed.
out.append(pfa)
src = '\n'.join(out)
open('/tmp/w/eng/combined.c','w').write(src)
print('wrote combined.c', len(src))
