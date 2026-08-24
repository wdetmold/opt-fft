cd /workdir/dev && python3 - <<'EOF'
import re
src = open('codelets.py').read()
# fix fft5 comb: computed S uses tw's negative sin => swap outputs
src = src.replace("""    def comb(Ek, Sk):
        a = (e.t(f"{Ek[0]} + {Sk[1]}"), e.t(f"{Ek[1]} - {Sk[0]}"))
        b = (e.t(f"{Ek[0]} - {Sk[1]}"), e.t(f"{Ek[1]} + {Sk[0]}"))
        return a, b""",
"""    def comb(Ek, Sk):
        a = (e.t(f"{Ek[0]} - {Sk[1]}"), e.t(f"{Ek[1]} + {Sk[0]}"))
        b = (e.t(f"{Ek[0]} + {Sk[1]}"), e.t(f"{Ek[1]} - {Sk[0]}"))
        return a, b""")
# fix halfmatrix: same swap
src = src.replace("""        out[k]   = (e.t(f"{er} + {si}"), e.t(f"{ei} - {sr}"))
        out[N-k] = (e.t(f"{er} - {si}"), e.t(f"{ei} + {sr}"))""",
"""        out[k]   = (e.t(f"{er} - {si}"), e.t(f"{ei} + {sr}"))
        out[N-k] = (e.t(f"{er} + {si}"), e.t(f"{ei} - {sr}"))""")
open('codelets.py','w').write(src)
EOF
python3 test_cores.py