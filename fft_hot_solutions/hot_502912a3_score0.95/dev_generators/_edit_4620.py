src = open('assemble.py').read()
old = """    f30 = f30.replace('void run##LL(long long', 'static void f30b_run##LL(long long')"""
new = """    f30 = f30.replace('void run##LL(long long', 'static void f30b_run##LL(long long')
    s81 = open(S81).read()
    # isolate colliding macros/idents
    for mac in ('TR8', 'DEINT'):
        s81 = re.sub(r'\\b' + mac + r'\\b', 's81_' + mac, s81)
    s81 = re.sub(r'\\bbig_alloc\\b', 's81_big_alloc', s81)
    s81 = s81.replace('void run_size(long L,', 'static void s81_run_size_impl(long L,')
    # make every other exported run_* static to avoid clashes (it may define run_NN too)
    s81 = re.sub(r'^void run_(\\d+)\\(', r'static void s81_run_\\1(', s81, flags=re.M)"""
assert old in src
src = src.replace(old, new)
old2 = """    shim64 = '''"""
new2 = """    shim45 = '''
void run_45(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    s81_run_size_impl(45L, B, m, x0, c, out1, outm, 0L);
}
'''
    shim64 = '''"""
assert old2 in src
src = src.replace(old2, new2)
old3 = """           + '\\n/* ======== 3f30 engine (L=64) ======== */\\n'
           + f30
           + shim64"""
new3 = """           + '\\n/* ======== 3f30 engine (L=64) ======== */\\n'
           + f30
           + shim64
           + '\\n/* ======== s81 engine (L=45) ======== */\\n#pragma GCC push_options\\n#pragma GCC optimize("schedule-insns","sched-pressure")\\n'
           + s81
           + '#pragma GCC pop_options\\n'
           + shim45"""
assert old3 in src
src = src.replace(old3, new3)
old4 = """    override = set(repl) | {64}"""
new4 = """    override = set(repl) | {64, 45}"""
assert old4 in src
src = src.replace(old4, new4)
old5 = """F30 = '/work/prior_work/solutions/v6_3f30d81f_score0.88/implementation.c'"""
new5 = """F30 = '/work/prior_work/solutions/v6_3f30d81f_score0.88/implementation.c'
S81 = '/tmp/cand/d43/impl_s81.c'"""
assert old5 in src
src = src.replace(old5, new5)
open('assemble.py','w').write(src)
print('ok')
