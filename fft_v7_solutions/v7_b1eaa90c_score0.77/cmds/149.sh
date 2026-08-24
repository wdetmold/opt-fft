cd /workdir/dev && python3 - <<'EOF'
import tune
A = tune.build({'sqpar':{'36':'1'}}, 'p45A')
B = tune.build({'sqpar':{'36':'1','45':'1'}}, 'p45B')
for i in range(3):
    ra = tune.bench(A, Ls=(45,), reps=4)
    rb = tune.bench(B, Ls=(45,), reps=4)
    print("nopar45", round(ra[45],3), " par45", round(rb[45],3))
EOF