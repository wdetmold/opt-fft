"""NOTE: the v4-era problem may have timed a 1x (not 3x) workload -- its C_opt/3\nconversions carry that caveat. GCP n2 turbo (~3.4 GHz) can legitimately beat our Gold\n6326 (~3.0-3.3 under zmm) by ~10-15%, so 0.80-1.0x is "fast tier", not impossible.\n\nGrader-vs-our-node comparison: for every collected rival attempt, put the grader's
REPORTED wall time next to OUR pinned bare-metal measurement of the same code.

Reported C_opt is on the 3x workload; /3 gives 1x. A VM tier can only be SLOWER than
bare metal, so ratio = (C_opt/3) / ours_bare is bounded below by ~1.0; the measured
honest tier factor is ~1.3-1.5. Verdicts:
    < 1.00        IMPOSSIBLE  (grader faster than bare metal: lucky window / flaky clock)
    1.00-1.25     SUSPICIOUS  (VM matching bare metal)
    1.25-1.70     CONSISTENT  (normal tier slowdown)
    > 1.70        SLOW-SHOT   (grader unlucky window -- flaky in the other direction)
Shot spread = max/min of the grader's own opt wall array, when the notes include it.
"""
import json, os, re, sys, glob, statistics

FFT='/home/lqcd/wdetmold/fft'
def parse_readme(path):
    try: s=open(path, encoding='utf-8', errors='replace').read()
    except Exception: return {}
    out={}
    m=re.search(r'C_opt[^0-9\n]*?([0-9]+\.[0-9]+)', s)
    if m: out['c_opt']=float(m.group(1))
    m=re.search(r'C_sota[^0-9\n]*?([0-9]+\.[0-9]+)', s)
    if m: out['c_sota']=float(m.group(1))
    m=re.search(r'score[=\s|*]+([01]\.[0-9]+)', s)
    if m: out['score']=float(m.group(1))
    m=re.search(r'opt[^\[\n]*\[([^\]]+)\]', s)
    if m:
        try:
            shots=[float(x) for x in re.findall(r'[0-9]+\.[0-9]+', m.group(1))]
            if len(shots)>=2: out['shots']=shots
        except ValueError: pass
    return out

ours={}
persize={}
p=os.path.join(FFT,'bench/ice/results/rivals_icelake/rivals.json')
if os.path.exists(p):
    for row in json.load(open(p)):
        d={}
        for L,rec in row['sizes'].items():
            s=rec.get('s')
            if s and s>1e-4 and rec.get('chain_rel',1)<0.5: d[L]=s
        persize[row['attempt']]=d
for log in glob.glob(os.path.join(FFT,'bench/ice/results/warm_icelake/bench_*.log')):
    for line in open(log):
        m=re.match(r'\s+(warm_\S+) L=\s*(\d+): best ([0-9.]+)s', line)
        if m and float(m.group(3))>1e-4:
            persize.setdefault(m.group(1),{})[m.group(2)]=float(m.group(3))
# final x86-regenerated 00291a90 supersedes the pre-final rebuild numbers
if 'warm_00291a90_score0.97' in persize:
    fin={}
    for line in open(os.path.join(FFT,'bench/ice/results/warm_icelake/bench_00291a90_final.log')):
        m=re.match(r'\s+warm_00291a90\S* L=\s*(\d+): best ([0-9.]+)s', line)
        if m: fin[m.group(1)]=float(m.group(2))
    if fin: persize['warm_00291a90_score0.97']=fin
rt=os.path.join(FFT,'bench/ice/results/rivaltime_last.json')
if os.path.exists(rt):
    for row in json.load(open(rt)):
        d={}
        for L,rec in row.get('sizes',{}).items():
            if rec.get('best_s',0)>1e-4 and rec.get('chain_rel',1)<0.5: d[L]=rec['best_s']
        if d: persize[row['attempt']]=d
ALL=('6','8','13','17','23','36','45','64')
med={L:statistics.median([d[L] for d in persize.values() if L in d]) for L in ALL}
for att,d in persize.items():
    filled=sum(d.get(L,med[L]) for L in ALL)
    ours[att]=(filled, len(d))
MULT={'fft_v4_solutions':1.0}
rows=[]
corpus_of={}
for corpus in ('fft_v4_solutions','fft_v5v6_solutions','fft_warm_solutions','fft_v7_solutions','fft_hot_solutions'):
    for d in sorted(glob.glob(os.path.join(FFT,corpus,'*','README.md'))):
        att=os.path.basename(os.path.dirname(d)); corpus_of[att]=corpus
        rep=parse_readme(d)
        meas=ours.get(att)
        if not rep.get('c_opt'): continue
        r1x=rep['c_opt']/MULT.get(corpus,3.0)
        if meas and meas[0]>0:
            full = meas[1]>=8
            ratio=r1x/meas[0]
            spread=max(rep['shots'])/min(rep['shots']) if rep.get('shots') else None
            if ratio<0.80: v='ANOMALOUS (<0.8x bare)'
            elif ratio<1.00: v='fast-tier (clock-explainable)'
            elif ratio<1.25: v='tier~bare'
            elif ratio<1.70: v='consistent'
            else: v='SLOW-SHOT'
            if not full: v+=' ~est'
            rows.append((att,rep.get('score'),rep['c_opt'],r1x,meas[0],meas[1],ratio,spread,v))
        else:
            rows.append((att,rep.get('score'),rep['c_opt'],r1x,None,0,None,
                         max(rep['shots'])/min(rep['shots']) if rep.get('shots') else None,'not measured'))
rows.sort(key=lambda r:(r[6] is None, r[6] if r[6] is not None else 9))
print(f"{'attempt':<30} {'score':>6} {'C_opt(3x)':>9} {'->1x':>7} {'ours(bare)':>10} {'grader/bare':>11} {'shot-spread':>11}  verdict")
for att,sc,co,r1,ms,n,ra,sp,v in rows:
    print(f"{att:<30} {sc if sc else '':>6} {co:>9.3f} {r1:>7.3f} "
          f"{(f'{ms:.3f}'+('~' if n<8 else ' ') if ms else '--  '):>10} {(f'{ra:.2f}x' if ra else '--'):>11} "
          f"{(f'{sp:.2f}x' if sp else '--'):>11}  {v}")
good=[r[6] for r in rows if r[6] and r[8].startswith(('consistent','tier~bare'))]
if good:
    print(f"\ncalibration: honest tier factor median {statistics.median(good):.2f}x over {len(good)} consistent attempts")
imp=[r for r in rows if r[8].startswith('ANOMALOUS')]
if imp: print(f"flagged as grader-flaky: {', '.join(r[0] for r in imp)}")

if '--md' in sys.argv:
    C={'fft_v4_solutions':'v4 (1x)','fft_v5v6_solutions':'v5/v6','fft_warm_solutions':'warm','fft_v7_solutions':'v7','fft_hot_solutions':'hot'}
    print()
    print("| attempt | cohort | score | reported C_opt | implied 1x | ours bare 1x | grader/bare | shots max/min | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("| **our panel (ice_r8)** | ice | — | — | — | **0.7791** | — | — | fastest measured |")
    order={'hot':0,'warm':1,'v7':2,'v5/v6':3,'v4 (1x)':4}
    for att,sc,co,r1,ms,n,ra,sp,v in sorted(rows,key=lambda r:(order.get(C.get(corpus_of.get(r[0],''),''),9), r[4] if r[4] else 9)):
        c=C.get(corpus_of.get(att,''),'?')
        print(f"| {att} | {c} | {sc if sc else '—'} | {co:.3f} | {r1:.3f} | "
              f"{(f'{ms:.4f}'+('~' if 0<n<8 else '')) if ms else '—'} | {(f'{ra:.2f}x' if ra else '—')} | "
              f"{(f'{sp:.2f}x' if sp else '—')} | {v} |")
