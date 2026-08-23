src = open('assemble.py').read()
old = """    shim64 = '''"""
new = """    dmine = open(DM).read()
    for nm in ('ALIGN64', 'TR8', 'VHALF', 'VONE', 'alloc_huge', 'alloc_huge_st', 'map2'):
        dmine = re.sub(r'\\b' + nm + r'\\b', 'dm_' + nm, dmine)
    dmine = re.sub(r'^void run_(\\d+)\\(', r'static void dm_run_\\1(', dmine, flags=re.M)
    dmine = re.sub(r'^static void run_(\\d+)\\(', r'static void dm_run_\\1(', dmine, flags=re.M)
    # fix any internal calls run_13t style kept; calls to run_XX within file:
    dmine = re.sub(r'\\brun_(13|17|23|36|45|6|8)\\(', r'dm_run_\\1(', dmine)
    shim13 = '''
void run_13(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    dm_run_13(x0, c, out1, outm, B, m);
}
'''
    shim64 = '''"""
assert old in src
src = src.replace(old, new, 1)
src = src.replace("""S39 = '/tmp/cand/d43/impl_3907.c'""", """S39 = '/tmp/cand/d43/impl_3907.c'
DM = '/tmp/cand/d43/impl_mine.c'""")
src = src.replace("""    override = set(repl) | {64, 45}""", """    override = set(repl) | {64, 45, 13}""")
src = src.replace("""           + '\\n#pragma GCC pop_options\\n'
           + shim45
           + '\\n/* ================= MY KERNELS ================= */\\n' + mine)""",
"""           + '\\n#pragma GCC pop_options\\n'
           + shim45
           + '\\n/* ======== d43-mine engine (L=13) ======== */\\n#pragma GCC push_options\\n#pragma GCC optimize("O3","unroll-loops","no-math-errno","no-trapping-math")\\n'
           + dmine
           + '\\n#pragma GCC pop_options\\n'
           + shim13
           + '\\n/* ================= MY KERNELS ================= */\\n' + mine)""")
open('assemble.py','w').write(src)
print('ok')
