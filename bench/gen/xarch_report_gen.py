"""Portability report: the acceptance suite on the scoring Ice Lake node vs an advisory
machine. Absolute times never cross machines -- only the ours-vs-best-library ratio and
the identity of the winning entry travel. Flags: a cell whose winning entry CHANGES, or
whose ratio degrades by more than 25%, is a portability finding for the next round.
usage: xarch_report_gen.py <icx_round> <xarch_round> [--md]
"""
import glob, json, math, os, sys

LIBS=('mkl_dfti','mkl2026_dfti','fftw3_estimate','fftw3_measure','fftw3_patient','fftw3_guru','ducc0_c2c','baseline_matrix')
HERE=os.path.dirname(os.path.abspath(__file__))

def load(rd):
    best,chk,one={},{},{}
    for p in glob.glob(f'{HERE}/results/{rd}/t_*.json'):
        d=json.load(open(p))
        if not d.get('supported'): continue
        k=(d['L'],d['name']); t=d['per_execute_seconds']['min']
        if k not in best or t<best[k]: best[k]=t
    for pre,st in (('c_',chk),('o_',one)):
        for p in glob.glob(f'{HERE}/results/{rd}/{pre}*.json'):
            b=os.path.basename(p)[len(pre):-5]; n,_,r=b.rpartition('_L'); L,_,B=r.partition('_B')
            try: st[(int(L),n)]=json.load(open(p))
            except Exception: pass
    return best,chk,one

def gate(chk,L):
    rels=[c['chain_rel_l2'] for (l,n),c in chk.items() if l==L and n in LIBS and 'chain_rel_l2' in c]
    anc=[c.get('anchor_rel_l2',0) or 0 for (l,n),c in chk.items() if l==L]
    raw=max([300*r for r in rels]+[300*a for a in anc]+[1e-10])
    e=math.floor(math.log10(raw)); m=raw/10**e
    return (1 if m<=1 else (3 if m<=3 else 10))*10**e

def cells(rd):
    best,chk,one=load(rd); out={}
    for L in sorted({l for (l,_) in best}):
        g=gate(chk,L); ours=[]
        for (l,n),t in best.items():
            if l!=L or n in LIBS: continue
            c=chk.get((L,n),{}); o=one.get((L,n),{})
            if c.get('rel_l2',1)<1e-12 and c.get('chain_rel_l2',1)<g and o.get('one_ok'): ours.append((t,n))
        lib=min((t for (l,n),t in best.items() if l==L and n in LIBS and n!='baseline_matrix'),default=None)
        if ours and lib: t,n=min(ours); out[L]=(n,t,lib,lib/t)
    return out

icx,xar=sys.argv[1],sys.argv[2]; md='--md' in sys.argv
A,B=cells(icx),cells(xar)
arch=xar.split('_')[1].upper()
rows=[]; flags=[]
for L in sorted(A):
    an,at,al,ar=A[L]
    if L not in B: rows.append((L,an,ar,'--','--','NO RESULT on '+arch)); flags.append(f"L={L}: no {arch} result"); continue
    bn,bt,bl,br=B[L]
    f=[]
    if bn!=an: f.append(f"winner changed {an}->{bn}")
    if br<0.75*ar: f.append(f"ratio degraded {ar:.2f}x->{br:.2f}x")
    if f: flags.append(f"L={L}: "+'; '.join(f))
    rows.append((L,an,ar,bn,br,'; '.join(f) or 'ok'))
if md:
    print(f"# Cross-arch advisory: {icx} (Ice Lake, scored) vs {xar} ({arch}, advisory)\n")
    print("| L | ICX winner | ICX vs-lib | %s winner | %s vs-lib | finding |"%(arch,arch))
    print("|---|---|---|---|---|---|")
    for L,an,ar,bn,br,f in rows:
        print(f"| {L} | {an} | {ar:.2f}x | {bn} | {br if isinstance(br,str) else f'{br:.2f}x'} | {f} |")
    print("\nFlags for the next round:" if flags else "\nNo portability findings: every cell wins on both machines with the same entry class.")
    for f in flags: print(f"- {f}")
else:
    print(f"{'L':>4} {'ICX winner':>16} {'vs-lib':>7} | {arch+' winner':>16} {'vs-lib':>7}  finding")
    for L,an,ar,bn,br,f in rows:
        print(f"{L:>4} {an:>16} {ar:6.2f}x | {bn:>16} {br if isinstance(br,str) else f'{br:6.2f}x'}  {f}")
