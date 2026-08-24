cd /workdir/dev && python3 - <<'EOF'
import tune
A = tune.build({'sqpar':{}}, 'parA')
B = tune.build({'sqpar':{'36':'1'}}, 'parB')
# correctness of B first
import subprocess, sys
import shutil
shutil.copy('/tmp/impl_parB.so', '/workdir/implementation.so')
r = subprocess.run([sys.executable, '/workdir/dev/check.py'], capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1])
for i in range(3):
    ra = tune.bench(A, Ls=(36,), reps=4)
    rb = tune.bench(B, Ls=(36,), reps=4)
    print("nopar", round(ra[36],3), " par", round(rb[36],3))
EOF