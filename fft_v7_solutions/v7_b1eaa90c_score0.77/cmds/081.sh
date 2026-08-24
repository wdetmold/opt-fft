cd /workdir/dev && for s in soa slab; do
python3 -c "
import tune
cfg={'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'},'scheme':{'23':'$s'}}
so=tune.build(cfg,'s23$s')
r=tune.bench(so, Ls=(23,), reps=5)
print('23-$s', {L: round(v,2) for L,v in r.items()})
"
done
# also verify correctness of current default config end-to-end
GENCFG='{"pw":{"13":"alt","17":"alt","23":"alt","45":"alt"}}' python3 gen.py && cd /workdir && rm -f implementation.so && gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm && python3 dev/check.py