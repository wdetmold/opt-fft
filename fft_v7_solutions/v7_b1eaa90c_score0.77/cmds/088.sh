cd /workdir/dev && for PF in 0 256 512 1024; do
python3 -c "
import tune
cfg={'pw':{'13':'alt','17':'alt','23':'alt','45':'alt'},'hu':{'13':'16'},'pf':{'45':'$PF','36':'$PF','64':'$PF'}}
so=tune.build(cfg,'pf$PF')
r=tune.bench(so, Ls=(36,45,64), reps=4)
print('PF=$PF', {L: round(v,2) for L,v in r.items()})
"
done