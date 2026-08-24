cd /workdir/dev && python3 - <<'EOF'
import tune, subprocess, sys, os, json
env = dict(os.environ); env['GENCFG'] = '{}'
subprocess.run([sys.executable,'gen.py'], check=True, env=env, capture_output=True)
flagsets = {
 'O3': ['-O3','-march=native'],
 'O3unroll': ['-O3','-march=native','-funroll-loops'],
 'O3fast-align': ['-O3','-march=native','-falign-functions=64'],
}
sos={}
for tag, fl in flagsets.items():
    so=f"/tmp/impl_fl_{tag}.so"
    subprocess.run(['gcc']+fl+['-shared','-fPIC','../implementation.c','-o',so,'-lm'], check=True)
    sos[tag]=so
for i in range(2):
    for tag, so in sos.items():
        r = tune.bench(so, reps=3)
        print(f"{tag:12s}", {L: round(v,2) for L,v in r.items()}, flush=True)
EOF