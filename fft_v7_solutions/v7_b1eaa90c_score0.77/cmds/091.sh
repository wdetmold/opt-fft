cd /workdir/dev && python3 - <<'EOF'
src = open('gen.py').read()
old = """            V f = (i & 1) ? pw_newton(zr, zi) : pw_sqrtnr(zr, zi);"""
new = """            V f = PW68SEL;"""
assert old in src
src = src.replace(old, new)
old2 = "    pwfn_ = cfg_pw(L)\n    pwdef = f'#define PW68 pw_{pwfn_ if pwfn_ != \"alt\" else \"sqrtnr\"}'"
new2 = """    pwfn_ = cfg_pw(L)
    if pwfn_ == 'alt':
        pwdef = '#define PW68SEL ((i & 1) ? pw_newton(zr, zi) : pw_sqrtnr(zr, zi))'
    else:
        pwdef = f'#define PW68SEL pw_{pwfn_}(zr, zi)'"""
assert old2 in src
src = src.replace(old2, new2)
src = src.replace("#undef PW68\n", "#undef PW68SEL\n")
open('gen.py','w').write(src)
print("ok")
EOF
cat > search.py <<'PYEOF'
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
PYEOF
python3 search.py