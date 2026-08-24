cd /workdir/dev && python3 -c "
import tune
cfg={'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'}}
so=tune.build(cfg,'db')
for i in range(3):
    r=tune.bench(so, reps=5)
    print({L: round(v,2) for L,v in r.items()})
"