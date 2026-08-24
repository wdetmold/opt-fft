#!/usr/bin/env python3
import re, os
HERE = os.path.dirname(os.path.abspath(__file__))
W00 = '/work/prior_work2/solutions/warm_00291a90_score0.97/implementation_final.c'
F30 = '/work/prior_work/solutions/v6_3f30d81f_score0.88/implementation.c'

def main():
    mine_path = os.path.join(HERE, 'my_kernels.c')
    mine = open(mine_path).read() if os.path.exists(mine_path) else ''
    repl = set(int(m) for m in re.findall(r'void run_(\d+)\(', mine))
    src = open(W00).read()
    # always rename w00's run_L -> w00_run_L for the sizes we override (mine or 3f30)
    override = set(repl) | {64}
    for L in sorted(override):
        src = src.replace(f'void run_{L}(const double*', f'void w00_run_{L}(const double*')
    f30 = open(F30).read()
    # 3f30: strip its #error guard? keep. rename its public syms
    f30 = f30.replace('void init_tables(void)', 'static void f30_init_tables(void)')
    f30 = f30.replace('void run64(long long', 'static void f30_run64(long long')
    # its GEN_ wrappers define other runNN via macro token pasting 'void run##LL' -> make static to avoid exporting
    f30 = f30.replace('void run##LL(long long', 'static void f30b_run##LL(long long')
    shim64 = '''
void run_64(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    static int f30_inited = 0;
    if (!f30_inited) { f30_init_tables(); f30_inited = 1; }
    f30_run64((long long)B, (long long)m, x0, c, out1, outm);
}
'''
    out = ('#pragma GCC push_options\n#pragma GCC optimize("schedule-insns","sched-pressure")\n'
           + src
           + '\n#pragma GCC pop_options\n'
           + '\n/* ======== 3f30 engine (L=64) ======== */\n'
           + f30
           + shim64
           + '\n/* ================= MY KERNELS ================= */\n' + mine)
    open(os.path.join(HERE, 'implementation.c'), 'w').write(out)
    print('override sizes:', sorted(override), 'bytes:', len(out))
if __name__ == '__main__':
    main()
