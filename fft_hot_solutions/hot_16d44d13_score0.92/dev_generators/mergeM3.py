import re
mine = open('/tmp/w/d43/impl_mine.c').read()
merged = open('/workdir/implementation.c').read()

# strip includes from mine
mine = '\n'.join(ln for ln in mine.split('\n') if not ln.startswith('#include'))
# prefix colliding globals in mine
renames = ['alloc_huge_st', 'alloc_huge', 'map2', 'stagger_ctr', 'VONE_', 'VHALF_',
           'run_13', 'run_17', 'run_23', 'run_36', 'run_45', 'run_6', 'run_8']
for name in renames:
    mine = re.sub(r'\b' + name + r'\b', 'f2_' + name, mine)
# macros: undef before section
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
open('/workdir/dev/impl_m3.c', 'w').write(out)
print('ok', len(out))
