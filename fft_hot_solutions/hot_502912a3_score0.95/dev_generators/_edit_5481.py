src = open('assemble.py').read()
old = """           + shim64

           + '\\n/* ================= MY KERNELS ================= */\\n' + mine)"""
new = """           + shim64
           + '\\n/* ======== 3907 engine (L=45) ======== */\\n#pragma GCC push_options\\n#pragma GCC optimize("O3","unroll-loops","no-math-errno","no-trapping-math")\\n'
           + s39
           + '\\n#pragma GCC pop_options\\n'
           + shim45
           + '\\n/* ================= MY KERNELS ================= */\\n' + mine)"""
assert old in src
open('assemble.py','w').write(src.replace(old,new))
print('fixed')
