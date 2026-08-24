#!/usr/bin/env python3
import re, os
HERE = os.path.dirname(os.path.abspath(__file__))
W00 = '/Users/wdetmold/MIT Dropbox/William Detmold/2026_problems/opt-fft/fft_warm_solutions/warm_00291a90_score0.97/implementation_final.c'
F30 = '/Users/wdetmold/MIT Dropbox/William Detmold/2026_problems/opt-fft/fft_v5v6_solutions/v6_3f30d81f_score0.88/implementation.c'
S39 = '/Users/wdetmold/MIT Dropbox/William Detmold/2026_problems/opt-fft/fft_warm_solutions/warm_d43251c2_score0.99/impl_3907.c'
DM = '/Users/wdetmold/MIT Dropbox/William Detmold/2026_problems/opt-fft/fft_warm_solutions/warm_d43251c2_score0.99/impl_mine.c'

def main():
    mine_path = os.path.join(HERE, 'my_kernels.c')
    mine = open(mine_path).read() if os.path.exists(mine_path) else ''
    repl = set(int(m) for m in re.findall(r'void run_(\d+)\(', mine))
    src = open(W00).read()
    # always rename w00's run_L -> w00_run_L for the sizes we override (mine or 3f30)
    override = set(repl) | {64, 45}
    for L in sorted(override):
        src = src.replace(f'void run_{L}(const double*', f'void w00_run_{L}(const double*')
    f30 = open(F30).read()
    # 3f30: strip its #error guard? keep. rename its public syms
    f30 = f30.replace('void init_tables(void)', 'static void f30_init_tables(void)')
    f30 = f30.replace('void run64(long long', 'static void f30_run64(long long')
    # its GEN_ wrappers define other runNN via macro token pasting 'void run##LL' -> make static to avoid exporting
    f30 = f30.replace('void run##LL(long long', 'static void f30b_run##LL(long long')
    s39 = open(S39).read()
    for nm in ('TR8', 'conv_in_64'):
        s39 = re.sub(r'\b' + nm + r'\b', 's39_' + nm, s39)
    s39 = re.sub(r'^void run_(\d+)\(', r'static void s39_run_\1(', s39, flags=re.M)
    s39 = re.sub(r'^static void run_(\d+)\(', r'static void s39_run_\1(', s39, flags=re.M)
    s39 = re.sub(r'\bvoid run\(int lid', 'static void s39_run(int lid', s39)
    s39 = re.sub(r'\brun_(\d+)\(B,m,x0,c,o1,om\)', r's39_run_\1(B,m,x0,c,o1,om)', s39)
    s39 = s39.replace('void init_all(void)', 'static void s39_init_all(void)')
    shim45 = '''
void run_45(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (B < 8) { w00_run_45(x0, c, out1, outm, B, m); return; }
    static int s39_inited = 0;
    if (!s39_inited) { s39_init_all(); s39_inited = 1; }
    s39_run_45(B, m, x0, c, out1, outm);
}
'''
    shim64 = '''
void run_64(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    static int f30_inited = 0;
    if (!f30_inited) { f30_init_tables(); f30_inited = 1; }
    f30_run64((long long)B, (long long)m, x0, c, out1, outm);
}
'''
    out = ('#pragma GCC push_options\n#pragma GCC optimize("O3","unroll-loops","schedule-insns","sched-pressure")\n'
           + src
           + '\n#pragma GCC pop_options\n'
           + '\n/* ======== 3f30 engine (L=64) ======== */\n#pragma GCC push_options\n#pragma GCC optimize("O3","unroll-loops","no-math-errno","no-trapping-math")\n'
           + f30
           + shim64
           + '\n#pragma GCC pop_options\n'
           + '\n/* ======== 3907 engine (L=45) ======== */\n#pragma GCC push_options\n#pragma GCC optimize("O3","unroll-loops","no-math-errno","no-trapping-math")\n'
           + s39
           + '\n#pragma GCC pop_options\n'
           + shim45
           + '\n/* ================= MY KERNELS ================= */\n' + mine)
    open(os.path.join(HERE, 'implementation.c'), 'w').write(out)
    print('override sizes:', sorted(override), 'bytes:', len(out))
if __name__ == '__main__':
    main()
