cd /workdir/dev && sed -i 's/if (PFON && g+1 < {G}) {{/if ({PFON} \&\& g+1 < {G}) {{/' gen.py && for H in NOPF T2; do
  if [ "$H" = "NOPF" ]; then GENCFG='{"sqpf":{"36":"0","64":"0"}}' python3 gen.py >/dev/null; else GENCFG='{"sqpf":{"36":"1","64":"1"}}' python3 gen.py >/dev/null; fi
  gcc -O3 -march=native -shared -fPIC ../implementation.c -o /tmp/impl_$H.so -lm
  python3 -c "
import tune
r1=tune.bench('/tmp/impl_$H.so', Ls=(36,64), reps=4)
r2=tune.bench('/tmp/impl_$H.so', Ls=(36,64), reps=4)
print('$H', {L: round(min(r1[L],r2[L]),2) for L in (36,64)})
"
done