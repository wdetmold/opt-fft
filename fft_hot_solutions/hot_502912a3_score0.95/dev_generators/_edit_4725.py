# fix assemble.py: drop s81, keep w00 pragma explicit
src = open('assemble.py').read()
src = src.replace("""    override = set(repl) | {64, 45}""", """    override = set(repl) | {64}""")
i0 = src.index("    s81 = open(S81).read()")
i1 = src.index("    shim45 = '''")
src = src[:i0] + src[i1:]
src = src.replace("""    shim45 = '''
void run_45(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    s81_run_size_impl(45L, B, m, x0, c, out1, outm, 0L);
}
'''
""", "")
src = src.replace("""           + '\\n/* ======== s81 engine (L=45) ======== */\\n#pragma GCC push_options\\n#pragma GCC optimize(S81OPT)\\n'
           + s81
           + '\\n#pragma GCC pop_options\\n'
           + shim45""", "")
src = src.replace("""'#pragma GCC push_options\\n#pragma GCC optimize("schedule-insns","sched-pressure")\\n'""",
                  """'#pragma GCC push_options\\n#pragma GCC optimize("O3","unroll-loops","schedule-insns","sched-pressure")\\n'""")
open('assemble.py','w').write(src)
print('assemble cleaned')
