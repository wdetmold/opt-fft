src = open('assemble.py').read()
# remove dm13 graft
i0 = src.index("    dmine = open(DM).read()")
i1 = src.index("    shim64 = '''")
src = src[:i0] + src[i1:]
src = src.replace("""    override = set(repl) | {64, 45, 13}""", """    override = set(repl) | {64, 45}""")
src = src.replace("""           + '\\n/* ======== d43-mine engine (L=13) ======== */\\n#pragma GCC push_options\\n#pragma GCC optimize("O3","unroll-loops","no-math-errno","no-trapping-math")\\n'
           + dmine
           + '\\n#pragma GCC pop_options\\n'
           + shim13
""", "")
open('assemble.py','w').write(src)
print('reverted')
