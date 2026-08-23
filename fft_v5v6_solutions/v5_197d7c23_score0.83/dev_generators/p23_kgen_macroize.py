# B49 (log 02:58:02): macro-ize k64/k36/k45 with an XCOPY flag; direct (d) variants
# skip the up-front dense copy for L1-hot z-strip buffers.
src=open('impl_head.c').read()

# ---- k64 -> macro with XCOPY flag
old = src[src.index('// 64 = 8 x 8 Cooley-Tukey; loads all inputs up-front'):src.index('// ---------------------------------------------------------------- 8x8 transpose')]
new = '''// 64 = 8 x 8 Cooley-Tukey. K64GEN(name, 1) copies inputs up-front (better for
// strided/missing loads); K64GEN(name, 0) reads the (dense, L1-hot) input directly.
#define K64GEN(NAME, XCOPY) \\
AI void NAME(v8 *restrict p, const long st){ \\
    v8 Xb[XCOPY?128:1]; \\
    const v8 *restrict X; \\
    if(XCOPY){ \\
        _Pragma("GCC unroll 64") \\
        for(int j=0;j<64;j++){ Xb[2*j]=p[2*st*j]; Xb[2*j+1]=p[2*st*j+1]; } \\
        X=Xb; \\
    } else X=p; \\
    v8 T[128]; \\
    _Pragma("GCC unroll 8") \\
    for(int j1=0;j1<8;j1++){ \\
        v8 r[8], i[8]; \\
        _Pragma("GCC unroll 8") \\
        for(int t=0;t<8;t++){ r[t]=X[(8*t+j1)*2]; i[t]=X[(8*t+j1)*2+1]; } \\
        dft8(r,i); \\
        if(j1==0){ \\
            _Pragma("GCC unroll 8") \\
            for(int k1=0;k1<8;k1++){ T[(k1*8)*2]=r[k1]; T[(k1*8)*2+1]=i[k1]; } \\
        }else{ \\
            T[(0*8+j1)*2]=r[0]; T[(0*8+j1)*2+1]=i[0]; \\
            _Pragma("GCC unroll 8") \\
            for(int k1=1;k1<8;k1++){ \\
                v8 wr=splat(tw64[(j1*8+k1)*2]), wi=splat(tw64[(j1*8+k1)*2+1]); \\
                v8 tr,ti; CMUL(tr,ti, r[k1],i[k1], wr,wi); \\
                T[(k1*8+j1)*2]=tr; T[(k1*8+j1)*2+1]=ti; \\
            } \\
        } \\
    } \\
    _Pragma("GCC unroll 8") \\
    for(int k1=0;k1<8;k1++){ \\
        v8 r[8], i[8]; \\
        _Pragma("GCC unroll 8") \\
        for(int t=0;t<8;t++){ r[t]=T[(k1*8+t)*2]; i[t]=T[(k1*8+t)*2+1]; } \\
        dft8(r,i); \\
        _Pragma("GCC unroll 8") \\
        for(int k2=0;k2<8;k2++){ p[2*st*(k1+8*k2)]=r[k2]; p[2*st*(k1+8*k2)+1]=i[k2]; } \\
    } \\
}
K64GEN(k64, 1)
K64GEN(k64d, 0)
'''
src = src.replace(old, new)

# ---- k36 XCOPY variants
old36 = src[src.index('AI void k36(v8 *restrict p, const long st){'):src.index('// 45 = PFA(5,9)')]
new36 = '''#define K36GEN(NAME, XCOPY) \\
AI void NAME(v8 *restrict p, const long st){ \\
    v8 Xb[XCOPY?72:1]; \\
    const v8 *restrict X; \\
    if(XCOPY){ \\
        _Pragma("GCC unroll 36") \\
        for(int j=0;j<36;j++){ Xb[2*j]=p[2*st*j]; Xb[2*j+1]=p[2*st*j+1]; } \\
        X=Xb; \\
    } else X=p; \\
    v8 Tr[4][9], Ti[4][9]; \\
    _Pragma("GCC unroll 9") \\
    for(int n2=0;n2<9;n2++){ \\
        v8 x0r=X[2*IN36[0][n2]], x0i=X[2*IN36[0][n2]+1]; \\
        v8 x1r=X[2*IN36[1][n2]], x1i=X[2*IN36[1][n2]+1]; \\
        v8 x2r=X[2*IN36[2][n2]], x2i=X[2*IN36[2][n2]+1]; \\
        v8 x3r=X[2*IN36[3][n2]], x3i=X[2*IN36[3][n2]+1]; \\
        DFT4(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2], \\
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i); \\
    } \\
    _Pragma("GCC unroll 4") \\
    for(int k1=0;k1<4;k1++){ \\
        dft9(Tr[k1], Ti[k1]); \\
        _Pragma("GCC unroll 9") \\
        for(int k2=0;k2<9;k2++){ \\
            p[2*st*K36T[k1][k2]]   = Tr[k1][k2]; \\
            p[2*st*K36T[k1][k2]+1] = Ti[k1][k2]; \\
        } \\
    } \\
}
K36GEN(k36, 1)
K36GEN(k36d, 0)
'''
src = src.replace(old36, new36)

# ---- k45 XCOPY variants
old45 = src[src.index('AI void k45(v8 *restrict p, const long st){'):src.index('// 64 = 8 x 8 Cooley-Tukey')]
new45 = '''#define K45GEN(NAME, XCOPY) \\
AI void NAME(v8 *restrict p, const long st){ \\
    v8 Xb[XCOPY?90:1]; \\
    const v8 *restrict X; \\
    if(XCOPY){ \\
        _Pragma("GCC unroll 45") \\
        for(int j=0;j<45;j++){ Xb[2*j]=p[2*st*j]; Xb[2*j+1]=p[2*st*j+1]; } \\
        X=Xb; \\
    } else X=p; \\
    v8 Tr[5][9], Ti[5][9]; \\
    _Pragma("GCC unroll 9") \\
    for(int n2=0;n2<9;n2++){ \\
        v8 x0r=X[2*IN45[0][n2]], x0i=X[2*IN45[0][n2]+1]; \\
        v8 x1r=X[2*IN45[1][n2]], x1i=X[2*IN45[1][n2]+1]; \\
        v8 x2r=X[2*IN45[2][n2]], x2i=X[2*IN45[2][n2]+1]; \\
        v8 x3r=X[2*IN45[3][n2]], x3i=X[2*IN45[3][n2]+1]; \\
        v8 x4r=X[2*IN45[4][n2]], x4i=X[2*IN45[4][n2]+1]; \\
        DFT5(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2],Tr[4][n2],Ti[4][n2], \\
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i); \\
    } \\
    _Pragma("GCC unroll 5") \\
    for(int k1=0;k1<5;k1++){ \\
        dft9(Tr[k1], Ti[k1]); \\
        _Pragma("GCC unroll 9") \\
        for(int k2=0;k2<9;k2++){ \\
            p[2*st*K45T[k1][k2]]   = Tr[k1][k2]; \\
            p[2*st*K45T[k1][k2]+1] = Ti[k1][k2]; \\
        } \\
    } \\
}
K45GEN(k45, 1)
K45GEN(k45d, 0)
'''
src = src.replace(old45, new45)
open('impl_head.c','w').write(src)
print("kernels macro-ized")
