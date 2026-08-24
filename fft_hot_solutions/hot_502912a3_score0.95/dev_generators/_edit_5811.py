src = open('assemble.py').read()
old = """           + '\\n/* ======== 3f30 engine (L=64) ======== */\\n'
           + f30
           + shim64"""
new = """           + '\\n/* ======== 3f30 engine (L=64) ======== */\\n#pragma GCC push_options\\n#pragma GCC optimize("O3","unroll-loops","no-math-errno","no-trapping-math")\\n'
           + f30
           + shim64
           + '\\n#pragma GCC pop_options\\n'"""
assert old in src
open('assemble.py','w').write(src.replace(old,new))
print('ok')
