# B33 (log 02:41:26): apply the load-first structure to k36/k45
src=open('impl_head.c').read()
# k36: copy input in sequential order first
old36 = '''AI void k36(v8 *restrict p, const long st){
    v8 Tr[4][9], Ti[4][9];
    #pragma GCC unroll 9
    for(int n2=0;n2<9;n2++){
        v8 x0r=p[2*st*IN36[0][n2]], x0i=p[2*st*IN36[0][n2]+1];
        v8 x1r=p[2*st*IN36[1][n2]], x1i=p[2*st*IN36[1][n2]+1];
        v8 x2r=p[2*st*IN36[2][n2]], x2i=p[2*st*IN36[2][n2]+1];
        v8 x3r=p[2*st*IN36[3][n2]], x3i=p[2*st*IN36[3][n2]+1];
        DFT4(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2],
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i);
    }'''
new36 = '''AI void k36(v8 *restrict p, const long st){
    v8 X[72];
    #pragma GCC unroll 36
    for(int j=0;j<36;j++){ X[2*j]=p[2*st*j]; X[2*j+1]=p[2*st*j+1]; }
    v8 Tr[4][9], Ti[4][9];
    #pragma GCC unroll 9
    for(int n2=0;n2<9;n2++){
        v8 x0r=X[2*IN36[0][n2]], x0i=X[2*IN36[0][n2]+1];
        v8 x1r=X[2*IN36[1][n2]], x1i=X[2*IN36[1][n2]+1];
        v8 x2r=X[2*IN36[2][n2]], x2i=X[2*IN36[2][n2]+1];
        v8 x3r=X[2*IN36[3][n2]], x3i=X[2*IN36[3][n2]+1];
        DFT4(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2],
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i);
    }'''
assert old36 in src; src=src.replace(old36,new36)
old45 = '''AI void k45(v8 *restrict p, const long st){
    v8 Tr[5][9], Ti[5][9];
    #pragma GCC unroll 9
    for(int n2=0;n2<9;n2++){
        v8 x0r=p[2*st*IN45[0][n2]], x0i=p[2*st*IN45[0][n2]+1];
        v8 x1r=p[2*st*IN45[1][n2]], x1i=p[2*st*IN45[1][n2]+1];
        v8 x2r=p[2*st*IN45[2][n2]], x2i=p[2*st*IN45[2][n2]+1];
        v8 x3r=p[2*st*IN45[3][n2]], x3i=p[2*st*IN45[3][n2]+1];
        v8 x4r=p[2*st*IN45[4][n2]], x4i=p[2*st*IN45[4][n2]+1];
        DFT5(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2],Tr[4][n2],Ti[4][n2],
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i);
    }'''
new45 = '''AI void k45(v8 *restrict p, const long st){
    v8 X[90];
    #pragma GCC unroll 45
    for(int j=0;j<45;j++){ X[2*j]=p[2*st*j]; X[2*j+1]=p[2*st*j+1]; }
    v8 Tr[5][9], Ti[5][9];
    #pragma GCC unroll 9
    for(int n2=0;n2<9;n2++){
        v8 x0r=X[2*IN45[0][n2]], x0i=X[2*IN45[0][n2]+1];
        v8 x1r=X[2*IN45[1][n2]], x1i=X[2*IN45[1][n2]+1];
        v8 x2r=X[2*IN45[2][n2]], x2i=X[2*IN45[2][n2]+1];
        v8 x3r=X[2*IN45[3][n2]], x3i=X[2*IN45[3][n2]+1];
        v8 x4r=X[2*IN45[4][n2]], x4i=X[2*IN45[4][n2]+1];
        DFT5(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2],Tr[4][n2],Ti[4][n2],
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i);
    }'''
assert old45 in src; src=src.replace(old45,new45)
open('impl_head.c','w').write(src)
print("ok")
