cd /workdir/dev && python3 - <<'EOF'
import tune
A = tune.build({'scheme':{'45':'slab'}}, 'ap45A')
B = tune.build({'scheme':{'45':'apfa'}}, 'ap45B')
for i in range(3):
    ra = tune.bench(A, Ls=(45,), reps=4)
    rb = tune.bench(B, Ls=(45,), reps=4)
    print("slab", round(ra[45],3), " apfa", round(rb[45],3))
EOF