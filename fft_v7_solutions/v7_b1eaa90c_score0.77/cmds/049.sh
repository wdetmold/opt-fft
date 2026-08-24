cd /workdir/dev && python3 - <<'EOF'
# improve fft8 in codelets.py: 0.707-factored twiddles
src = open('codelets.py').read()
old = '''def fft8(e, x):
    Ev = fft4(e, [x[0], x[2], x[4], x[6]])
    Od = fft4(e, [x[1], x[3], x[5], x[7]])
    out = [None]*8
    for k in range(4):
        c, s = tw(k, 8)
        w = cmul_w(e, Od[k], c, s)
        out[k]   = cadd(e, Ev[k], w)
        out[k+4] = csub(e, Ev[k], w)
    return out'''
new = '''def fft8(e, x):
    Ev = fft4(e, [x[0], x[2], x[4], x[6]])
    Od = fft4(e, [x[1], x[3], x[5], x[7]])
    C = "0x1.6a09e667f3bcdp-1"  # sqrt(2)/2
    out = [None]*8
    for k in range(4):
        o = Od[k]
        if k == 0:
            w = o
        elif k == 1:   # W8^1 = c*(1 - i)
            w = (e.t(f"{C} * ({o[0]} + {o[1]})"), e.t(f"{C} * ({o[1]} - {o[0]})"))
        elif k == 2:   # -i
            w = cmulmi(e, o)
        else:          # W8^3 = -c*(1 + i)
            w = (e.t(f"{C} * ({o[1]} - {o[0]})"), e.t(f"-({C} * ({o[0]} + {o[1]}))"))
        out[k]   = cadd(e, Ev[k], w)
        out[k+4] = csub(e, Ev[k], w)
    return out'''
assert old in src
src = src.replace(old, new)
open('codelets.py','w').write(src)
print("fft8 improved")
EOF
python3 test_cores.py 2>/dev/null | tail -3