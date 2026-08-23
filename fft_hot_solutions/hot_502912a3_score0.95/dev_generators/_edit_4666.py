src = open('assemble.py').read()
src = src.replace("""+ s81
           + '#pragma GCC pop_options\\n'""", """+ s81
           + '\\n#pragma GCC pop_options\\n'""")
src = src.replace('''\\n/* ======== s81 engine (L=45) ======== */\\n#pragma GCC push_options\\n#pragma GCC optimize("schedule-insns","sched-pressure")\\n''',
                  '''\\n/* ======== s81 engine (L=45) ======== */\\n#pragma GCC push_options\\n#pragma GCC optimize(S81OPT)\\n''')
open('assemble.py','w').write(src)
