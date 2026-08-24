cd /workdir/dev && python3 - <<'EOF'
import tune, json
grids = [
  ("slab17-sqrtnr", {'pw':{}, 'pf':{}}),
  ("slab17-alt", {'pw':{str(L):'alt' for L in (13,17,23,36,45,64)}, 'pf':{}}),
  ("soa17-alt", {'scheme':{'17':'soa'}, 'pw':{str(L):'alt' for L in (13,17,23,36,45,64)}, 'pf':{}}),
  ("soa1723-alt", {'scheme':{'17':'soa','23':'soa'}, 'pw':{str(L):'alt' for L in (13,17,23,36,45,64)}, 'pf':{}}),
  ("soa17-alt-pf0", {'scheme':{'17':'soa'}, 'pw':{str(L):'alt' for L in (13,17,23,36,45,64)}, 'pf':{str(L):0 for L in (6,8,13,17,23,36,45,64)}}),
]
for tag, cfg in grids:
    so = tune.build(cfg, tag.replace('-','_'))
    r = tune.bench(so)
    print(f"{tag:18s}", {L: round(v,2) for L,v in r.items()}, flush=True)
EOF