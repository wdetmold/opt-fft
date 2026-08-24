import tune, copy, json
SIZES=(6,8,13,17,23,36,45,64)
best = {'pw':{'13':'alt','17':'alt','23':'alt','45':'sqrtnr','6':'alt','8':'sqrtnr'},
        'pf':{'45':'256'}, 'hu':{'13':'16'}, 'scheme':{}}
def run(cfg, Ls, tag):
    so = tune.build(cfg, tag)
    r = {}
    for _ in range(2):
        out = tune.bench(so, Ls=Ls, reps=4)
        for L,v in out.items(): r[L] = min(r.get(L,1e9), v)
    return r
opts = {
  6:  [('pw','sqrtnr'),('pw','newton'),('pw','alt')],
  8:  [('pw','sqrtnr'),('pw','newton'),('pw','alt')],
  13: [('pw','alt'),('pw','sqrtnr'),('hu','1'),('hu','16')],
  17: [('pw','alt'),('pw','sqrtnr'),('hu','16'),('hu','1'),('pf','256'),('pf','0')],
  23: [('pw','alt'),('pw','sqrtnr'),('hu','1'),('pf','256'),('pf','0'),('scheme','slab'),('scheme','soa')],
  36: [('pf','0'),('pf','256')],
  45: [('pw','sqrtnr'),('pw','alt'),('pf','256'),('pf','0')],
  64: [('pf','0'),('pf','256')],
}
for L in (8, 13, 17, 23, 36, 45, 64, 6):
    results = {}
    groups = {}
    for key, val in opts[L]:
        groups.setdefault(key, []).append(val)
    for key, vals in groups.items():
        scores = {}
        for val in vals:
            cfg = copy.deepcopy(best)
            cfg[key][str(L)] = val
            r = run(cfg, (L,), f"s{L}{key}{val}")
            scores[val] = r[L]
        pick = min(scores, key=scores.get)
        best[key][str(L)] = pick
        print(f"L={L} {key}: {scores} -> {pick}", flush=True)
print("BEST CONFIG:", json.dumps(best))
json.dump(best, open('best_cfg.json','w'))
