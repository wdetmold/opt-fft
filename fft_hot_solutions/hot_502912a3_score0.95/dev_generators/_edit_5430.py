src = open('assemble.py').read()
old = """    shim64 = '''"""
new = """    s39 = open(S39).read()
    for nm in ('TR8', 'conv_in_64'):
        s39 = re.sub(r'\\b' + nm + r'\\b', 's39_' + nm, s39)
    s39 = re.sub(r'^void run_(\\d+)\\(', r'static void s39_run_\\1(', s39, flags=re.M)
    s39 = re.sub(r'\\bvoid run\\(int lid', 'static void s39_run(int lid', s39)
    s39 = re.sub(r'\\brun_(\\d+)\\(B,m,x0,c,o1,om\\)', r's39_run_\\1(B,m,x0,c,o1,om)', s39)
    s39 = s39.replace('void init_all(void)', 'static void s39_init_all(void)')
    shim45 = '''
void run_45(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    static int s39_inited = 0;
    if (!s39_inited) { s39_init_all(); s39_inited = 1; }
    s39_run_45(B, m, x0, c, out1, outm);
}
'''
    shim64 = '''"""
assert old in src
src = src.replace(old, new, 1)
src = src.replace("""S81 = '/tmp/cand/d43/impl_s81.c'""", """S39 = '/tmp/cand/d43/impl_3907.c'""")
src = src.replace("""    override = set(repl) | {64}""", """    override = set(repl) | {64, 45}""")
src = src.replace("""           + shim64
           + '\\n/* ================= MY KERNELS ================= */\\n' + mine)""",
"""           + shim64
           + '\\n/* ======== 3907 engine (L=45) ======== */\\n#pragma GCC push_options\\n#pragma GCC optimize("O3","unroll-loops","no-math-errno","no-trapping-math")\\n'
           + s39
           + '\\n#pragma GCC pop_options\\n'
           + shim45
           + '\\n/* ================= MY KERNELS ================= */\\n' + mine)""")
open('assemble.py','w').write(src)
print('ok')
