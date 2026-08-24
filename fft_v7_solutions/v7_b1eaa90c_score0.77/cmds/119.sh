cd /workdir/dev && for H in NOPF T1 T2; do
  if [ "$H" = "NOPF" ]; then
    sed -i 's/long o0 = k2\*{SLICE};/long o0 = k2*{SLICE}; if (0) {{/; ' /dev/null 2>/dev/null
    python3 - <<'EOF'
src=open('gen.py').read()
import re
old_start = "          if (g+1 < {G}) {{\n            long o0 = k2*{SLICE};"
src = src.replace(old_start, "          if (PFON && g+1 < {G}) {{\n            long o0 = k2*{SLICE};", 1)
src = src.replace("def gen_sq(L, G):\n    SLICE", "def gen_sq(L, G):\n    PFON = int(CONFIG.get('sqpf',{}).get(str(L), 1))\n    SLICE", 1) if "PFON = int" not in src else src
open('gen.py','w').write(src)
EOF
    GENCFG='{"sqpf":{"36":"0","64":"0"}}' python3 gen.py >/dev/null
  elif [ "$H" = "T1" ]; then
    sed -i 's/_MM_HINT_T0/_MM_HINT_T1/g' gen.py && GENCFG='{"sqpf":{"36":"1","64":"1"}}' python3 gen.py >/dev/null
  else
    sed -i 's/_MM_HINT_T1/_MM_HINT_T2/g' gen.py && GENCFG='{"sqpf":{"36":"1","64":"1"}}' python3 gen.py >/dev/null
  fi
  gcc -O3 -march=native -shared -fPIC ../implementation.c -o /tmp/impl_$H.so -lm
  python3 -c "
import tune
r1=tune.bench('/tmp/impl_$H.so', Ls=(36,64), reps=4)
r2=tune.bench('/tmp/impl_$H.so', Ls=(36,64), reps=4)
print('$H', {L: round(min(r1[L],r2[L]),2) for L in (36,64)})
"
done