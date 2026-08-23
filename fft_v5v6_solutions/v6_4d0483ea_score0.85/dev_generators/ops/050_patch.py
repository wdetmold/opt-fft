src = open('solution.py').read()
src = src.replace("""    if B > 0 and m > 0:
        _lib.mp_run(L, B, m,""",
"""    if B > 0 and m <= 0:   # out of spec; return inputs deterministically
        one[:] = x0.ravel(); fin[:] = x0.ravel()
        return one, fin
    if B > 0 and m > 0:
        _lib.mp_run(L, B, m,""")
open('solution.py','w').write(src)
