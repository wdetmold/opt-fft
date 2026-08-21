import re,sys,os
def variant(path, fma=False):
    txt=open(path).read()
    i=txt.find('#else')
    if i<0: return txt
    return txt[:i] if fma else txt[i:]

tokre=re.compile(r'\b(T[0-9a-zA-Z]+)\b')

def liveness(path, fma=False):
    body=variant(path,fma)
    lines=[l.strip() for l in body.split('\n')]
    stmts=[]
    for l in lines:
        if re.match(r'^(E|V|R|INT|const|static|DK|DVK)\b', l): continue
        if not l.endswith(';'): continue
        if l.startswith('for') or l.startswith('return'): continue
        if not tokre.search(l): continue
        stmts.append(l)
    defs={}; last={}
    for idx,s in enumerate(stmts):
        if '=' in s and not s.startswith('ST('):
            lhs,rhs=s.split('=',1)
        else:
            lhs,rhs='',s
        for v in tokre.findall(rhs): last[v]=idx
        lv=tokre.findall(lhs)
        if lv and '[' not in lhs and lv[0] not in defs:
            defs[lv[0]]=idx; last.setdefault(lv[0],idx)
    peak=0; live=set()
    for idx,s in enumerate(stmts):
        for v,d in defs.items():
            if d==idx: live.add(v)
        peak=max(peak,len(live))
        for v in list(live):
            if last.get(v,-1)<=idx: live.discard(v)
    return len(stmts), len(defs), peak

for p in sys.argv[1:]:
    n,d,pk=liveness(p); nf,df,pkf=liveness(p,True)
    print(f"{os.path.basename(p):12s} nonFMA stmts={n:5d} vals={d:4d} peak_live={pk:4d} | FMA stmts={nf:5d} vals={df:4d} peak_live={pkf:4d}")
