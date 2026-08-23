"""Forensic retimer for rival solutions (persisted from the job-tmp rivalbench/warmbench).

Beyond the paired timing + two-part gates, this version records per-rep telemetry chosen
to discriminate the known ways a grader clock lies:
  * cpu/wall ratio (getrusage, all threads): > ~1.1 means the "single-core" code is
    multithreaded -- on a 4-vCPU grader slice that alone explains up to ~3-4x.
  * peak thread count (/proc/self/status Threads, sampled during the run).
  * anti-memoization: one input element is perturbed between reps; the final state must
    change (a solver that caches the chain end times an empty call otherwise).
  * spin-calibrated clock check: a fixed dependent-FMA spin is timed before/after each
    rep; if the spin's apparent rate drifts, the wall clock itself is suspect.

usage: python3 rivaltime.py <solutions_root> [attempt-substring]
Writes JSON next to results/rivals_icelake/. Pin externally with taskset.
"""
import importlib.util, json, os, resource, sys, threading, time
import numpy as np

POINTS = [(6,64,4856),(8,64,2572),(13,32,1278),(17,32,98),(23,16,165),(36,8,64),(45,4,177),(64,2,134)]
REPS = 5

def spin(n=2_000_000):
    t0=time.perf_counter(); x=1.0000001
    for _ in range(n): x=x*1.0000001+1e-9
    return n/(time.perf_counter()-t0), x

class ThreadPeak:
    def __init__(self): self.peak=0; self.stop=False
    def run(self):
        while not self.stop:
            try:
                with open('/proc/self/status') as f:
                    for l in f:
                        if l.startswith('Threads:'): self.peak=max(self.peak,int(l.split()[1]))
            except Exception: pass
            time.sleep(0.05)

root=sys.argv[1]; only=sys.argv[2] if len(sys.argv)>2 else ''
data, ref, ref1 = {}, {}, {}
for L,B,m in POINTS:
    rx=np.random.default_rng(7+L)
    x0=np.ascontiguousarray(rx.standard_normal((B,L,L,L))+1j*rx.standard_normal((B,L,L,L)))
    rc=np.random.default_rng(1000+L)
    c=np.ascontiguousarray(0.1*(rc.standard_normal((B,L,L,L))+1j*rc.standard_normal((B,L,L,L))))
    data[L]=(x0,c); st=x0.copy()
    for _ in range(m):
        z=np.fft.fftn(st,axes=(-3,-2,-1))+c; st=z/(1+np.abs(z))
    ref[L]=st
    z=np.fft.fftn(x0,axes=(-3,-2,-1))+c; ref1[L]=z/(1+np.abs(z))
    print(f"ref L={L} ready", flush=True)

out=[]
for name in sorted(os.listdir(root)):
    d=os.path.join(root,name)
    if only and only not in name: continue
    sol=os.path.join(d,'solution.py')
    if not os.path.isfile(sol): continue
    row={'attempt':name,'sizes':{},'error':None}
    try:
        spec=importlib.util.spec_from_file_location(f"rv_{name.replace('-','_')}",sol)
        mod=importlib.util.module_from_spec(spec)
        old=os.getcwd(); os.chdir(d)
        try: spec.loader.exec_module(mod)
        finally: os.chdir(old)
        run=mod._run
        for L,B,m in POINTS:
            x0,c=data[L]
            one,fin=run(L,B,m,x0,c)
            one=np.asarray(one).reshape(ref1[L].shape); fin=np.asarray(fin).reshape(ref[L].shape)
            r1=float(np.linalg.norm(one-ref1[L])/np.linalg.norm(ref1[L]))
            rch=float(np.linalg.norm(fin-ref[L])/np.linalg.norm(ref[L]))
            reps=[]
            for k in range(REPS):
                xp=x0.copy(); xp.flat[k]*=(1+1e-13)          # anti-memoization perturb
                rate0,_=spin()
                tp=ThreadPeak(); th=threading.Thread(target=tp.run,daemon=True); th.start()
                ru0=resource.getrusage(resource.RUSAGE_SELF)
                t0=time.perf_counter(); _,fp=run(L,B,m,xp,c); wall=time.perf_counter()-t0
                ru1=resource.getrusage(resource.RUSAGE_SELF)
                tp.stop=True; th.join(timeout=0.3)
                rate1,_=spin()
                cpu=(ru1.ru_utime-ru0.ru_utime)+(ru1.ru_stime-ru0.ru_stime)
                fp=np.asarray(fp).reshape(ref[L].shape)
                changed=bool(np.linalg.norm(fp-fin)>0)
                reps.append(dict(wall=wall,cpu_over_wall=cpu/wall if wall>0 else None,
                                 thread_peak=tp.peak,out_changed=changed,
                                 spin_drift=rate1/rate0))
            best=min(r['wall'] for r in reps)
            row['sizes'][str(L)]={'best_s':best,'one_rel':r1,'chain_rel':rch,'reps':reps}
            flags=[]
            if any(r['cpu_over_wall'] and r['cpu_over_wall']>1.15 for r in reps): flags.append('MULTITHREADED')
            if any(not r['out_changed'] for r in reps): flags.append('MEMOIZED')
            if any(abs(r['spin_drift']-1)>0.05 for r in reps): flags.append('CLOCK-DRIFT')
            print(f"  {name} L={L:3d}: best {best:.4f}s cpu/wall {max(r['cpu_over_wall'] for r in reps):.2f} "
                  f"thr {max(r['thread_peak'] for r in reps)} one {r1:.1e} chain {rch:.1e} {' '.join(flags)}", flush=True)
    except Exception:
        import traceback; row['error']=traceback.format_exc(limit=3); print(row['error'])
    out.append(row)
json.dump(out,open('/home/lqcd/wdetmold/fft/bench/ice/results/rivaltime_last.json','w'),indent=1)
print("DONE")
