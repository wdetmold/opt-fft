// ---------------------------------------------------------------- buffers
static v8 *SBUF, *CBUF;               // big shared scratch (one unit: group or volume) + c
#define MAXBLK (64*512)               // largest unit in blocks (L=64 volume)
static void *xalloc(size_t bytes){
    size_t sz=(bytes+2097151)&~(size_t)2097151;
    void *p=mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(p==MAP_FAILED){ p=aligned_alloc(64,sz); }
    else madvise(p, sz, MADV_HUGEPAGE);
    memset(p,0,sz);
    return p;
}
void init_buffers(void){
    SBUF=(v8*)xalloc((size_t)MAXBLK*128);
    CBUF=(v8*)xalloc((size_t)MAXBLK*128);
}

// ---------------------------------------------------------------- interleave helpers
AI void deint(const double *s, v8 *re, v8 *im){ // 8 complex -> split
    v8 v0=*(const v8u*)s, v1=*(const v8u*)(s+8);
    *re=__builtin_shufflevector(v0,v1,0,2,4,6,8,10,12,14);
    *im=__builtin_shufflevector(v0,v1,1,3,5,7,9,11,13,15);
}
AI void ileav(double *d, v8 re, v8 im){ // split -> 8 complex
    *(v8u*)d     = __builtin_shufflevector(re,im,0,8,1,9,2,10,3,11);
    *(v8u*)(d+8) = __builtin_shufflevector(re,im,4,12,5,13,6,14,7,15);
}

// ---------------------------------------------------------------- BL (batch-lane): lanes = 8 volumes
#define GEN_BL(SFX, LL, KFN) \
static void bl_step_##SFX(v8 *restrict S, const v8 *restrict C){ \
    const long L2=(long)LL*LL; \
    for(long yz=0; yz<L2; yz++) KFN(S + yz*2, L2);      /* x lines */ \
    for(long x=0;x<LL;x++){ \
        v8 *pl = S + x*L2*2; \
        const v8 *cpl = C + x*L2*2; \
        for(long y=0;y<LL;y++) KFN(pl + y*LL*2, 1);     /* z lines */ \
        for(long z=0;z<LL;z++) KFN(pl + z*2, LL);       /* y lines */ \
        for(long i=0;i<L2;i++){ \
            v8 zr=pl[2*i]+cpl[2*i], zi=pl[2*i+1]+cpl[2*i+1]; \
            vmap(&zr,&zi); \
            pl[2*i]=zr; pl[2*i+1]=zi; \
        } \
    } \
}
GEN_BL(6, 6, k6)
GEN_BL(8, 8, k8)
GEN_BL(13, 13, k13)
GEN_BL(17, 17, k17)
GEN_BL(23, 23, k23)

static void bl_in(const double *restrict x0, long B, long g, v8 *restrict S, long L3){
    long e8=L3&~7L;
    for(long e=0;e<e8;e+=8){
        v8 R[8], I[8];
        for(int v=0;v<8;v++){
            long vol=g*8+v;
            if(vol<B) deint(x0+vol*L3*2+e*2, &R[v], &I[v]);
            else { R[v]=VZERO; I[v]=VZERO; }
        }
        tr8(R); tr8(I);
        for(int t=0;t<8;t++){ S[(e+t)*2]=R[t]; S[(e+t)*2+1]=I[t]; }
    }
    for(long e=e8;e<L3;e++){
        double *d=(double*)(S+e*2);
        for(int v=0;v<8;v++){
            long vol=g*8+v;
            if(vol<B){ d[v]=x0[vol*L3*2+2*e]; d[8+v]=x0[vol*L3*2+2*e+1]; }
            else { d[v]=0.0; d[8+v]=0.0; }
        }
    }
}
static void bl_out(double *restrict o, long B, long g, const v8 *restrict S, long L3){
    long e8=L3&~7L, nv = B-g*8; if(nv>8) nv=8;
    for(long e=0;e<e8;e+=8){
        v8 R[8], I[8];
        for(int t=0;t<8;t++){ R[t]=S[(e+t)*2]; I[t]=S[(e+t)*2+1]; }
        tr8(R); tr8(I);
        for(int v=0;v<nv;v++) ileav(o+(g*8+v)*L3*2+e*2, R[v], I[v]);
    }
    for(long e=e8;e<L3;e++){
        const double *d=(const double*)(S+e*2);
        for(int v=0;v<nv;v++){ o[(g*8+v)*L3*2+2*e]=d[v]; o[(g*8+v)*L3*2+2*e+1]=d[8+v]; }
    }
}
static void bl_run(long LL, long B, long m, const double *x0, const double *c,
                   double *o1, double *of, void(*step)(v8*,const v8*)){
    long L3=LL*LL*LL, G=(B+7)/8;
    for(long g=0; g<G; g++){
        bl_in(x0,B,g,SBUF,L3);
        bl_in(c,B,g,CBUF,L3);
        for(long s=0;s<m;s++){
            step(SBUF,CBUF);
            if(s==0 && m>1) bl_out(o1,B,g,SBUF,L3);
        }
        bl_out(of,B,g,SBUF,L3);
    }
    if(m==1) memcpy(o1, of, (size_t)B*L3*16);
}

// ---------------------------------------------------------------- PV (per-volume): lanes = z chunks
#define GEN_PV(SFX, LL, RSY, RB, KFN) \
static void pv_step_##SFX(v8 *restrict V, const v8 *restrict C){ \
    const long PB=(long)RSY*RB; \
    for(long y=0;y<LL;y++) \
        for(long zc=0; zc<RB; zc++) KFN(V + (y*RB+zc)*2, PB);   /* x lines */ \
    for(long x=0;x<LL;x++){ \
        v8 *pl = V + x*PB*2; \
        const v8 *cpl = C + x*PB*2; \
        for(long zc=0; zc<RB; zc++) KFN(pl + zc*2, RB);         /* y lines */ \
        for(long yc=0; yc<RSY/8; yc++){                          /* z lines */ \
            v8 buf[RB*8*2]; \
            for(long t=0;t<RB;t++){ \
                v8 ar[8], ai[8]; \
                for(int e=0;e<8;e++){ ar[e]=pl[((yc*8+e)*RB+t)*2]; ai[e]=pl[((yc*8+e)*RB+t)*2+1]; } \
                tr8(ar); tr8(ai); \
                for(int e=0;e<8;e++){ buf[(t*8+e)*2]=ar[e]; buf[(t*8+e)*2+1]=ai[e]; } \
            } \
            KFN(buf, 1); \
            for(long t=0;t<RB;t++){ \
                v8 ar[8], ai[8]; \
                for(int e=0;e<8;e++){ ar[e]=buf[(t*8+e)*2]; ai[e]=buf[(t*8+e)*2+1]; } \
                tr8(ar); tr8(ai); \
                for(int e=0;e<8;e++){ pl[((yc*8+e)*RB+t)*2]=ar[e]; pl[((yc*8+e)*RB+t)*2+1]=ai[e]; } \
            } \
        } \
        for(long i=0;i<PB;i++){ \
            v8 zr=pl[2*i]+cpl[2*i], zi=pl[2*i+1]+cpl[2*i+1]; \
            vmap(&zr,&zi); \
            pl[2*i]=zr; pl[2*i+1]=zi; \
        } \
    } \
}
GEN_PV(6,  6,  8, 1, k6)
GEN_PV(8,  8,  8, 1, k8)
GEN_PV(13, 13, 16, 2, k13)
GEN_PV(17, 17, 24, 3, k17)
GEN_PV(23, 23, 24, 3, k23)
GEN_PV(36, 36, 40, 5, k36)
GEN_PV(45, 45, 48, 6, k45)
GEN_PV(64, 64, 64, 8, k64)

static void pv_in(const double *restrict src, v8 *restrict V, long LL, long RB, long PB){
    memset(V, 0, (size_t)LL*PB*128);
    long full=LL/8, tail=LL%8;
    for(long x=0;x<LL;x++) for(long y=0;y<LL;y++){
        const double *s = src + ((x*LL+y)*LL)*2;
        v8 *row = V + (x*PB + y*RB)*2;
        for(long zc=0; zc<full; zc++) deint(s+zc*16, &row[zc*2], &row[zc*2+1]);
        if(tail){
            double *rr=(double*)(row+full*2);
            for(long t=0;t<tail;t++){ rr[t]=s[(full*8+t)*2]; rr[8+t]=s[(full*8+t)*2+1]; }
        }
    }
}
static void pv_out(double *restrict dst, const v8 *restrict V, long LL, long RB, long PB){
    long full=LL/8, tail=LL%8;
    for(long x=0;x<LL;x++) for(long y=0;y<LL;y++){
        double *t_ = dst + ((x*LL+y)*LL)*2;
        const v8 *row = V + (x*PB + y*RB)*2;
        for(long zc=0; zc<full; zc++) ileav(t_+zc*16, row[zc*2], row[zc*2+1]);
        if(tail){
            const double *rr=(const double*)(row+full*2);
            for(long tt=0;tt<tail;tt++){ t_[(full*8+tt)*2]=rr[tt]; t_[(full*8+tt)*2+1]=rr[8+tt]; }
        }
    }
}
static void pv_run(long LL, long RB, long PB, long B, long m, const double *x0, const double *c,
                   double *o1, double *of, void(*step)(v8*,const v8*)){
    long L3=LL*LL*LL;
    for(long v=0; v<B; v++){
        pv_in(x0+v*L3*2, SBUF, LL, RB, PB);
        pv_in(c+v*L3*2, CBUF, LL, RB, PB);
        for(long s=0;s<m;s++){
            step(SBUF,CBUF);
            if(s==0 && m>1) pv_out(o1+v*L3*2, SBUF, LL, RB, PB);
        }
        pv_out(of+v*L3*2, SBUF, LL, RB, PB);
    }
    if(m==1) memcpy(o1, of, (size_t)B*L3*16);
}

// ---------------------------------------------------------------- API
static int FORCE_SCHEME = 0;  // 0=auto, 1=BL, 2=PV  (bench hook)
void set_scheme(int s){ FORCE_SCHEME=s; }

typedef void (*stepfn)(v8*,const v8*);
void run_size(long L, long B, long m, const double *x0, const double *c,
              double *out_one, double *out_final){
    if(B<=0 || m<=0) return;
    stepfn bl=0, pv=0; long RB=0, PB=0;
    switch(L){
        case 6:  bl=bl_step_6;  pv=pv_step_6;  RB=1; PB=8;   break;
        case 8:  bl=bl_step_8;  pv=pv_step_8;  RB=1; PB=8;   break;
        case 13: bl=bl_step_13; pv=pv_step_13; RB=2; PB=32;  break;
        case 17: bl=bl_step_17; pv=pv_step_17; RB=3; PB=72;  break;
        case 23: bl=bl_step_23; pv=pv_step_23; RB=3; PB=72;  break;
        case 36: pv=pv_step_36; RB=5; PB=200; break;
        case 45: pv=pv_step_45; RB=6; PB=288; break;
        case 64: pv=pv_step_64; RB=8; PB=512; break;
    }
    int use_bl;
    if(FORCE_SCHEME) use_bl = (FORCE_SCHEME==1) && bl;
    else use_bl = bl && (B>=6);   // heuristic, tuned later
    if(use_bl) bl_run(L,B,m,x0,c,out_one,out_final,bl);
    else       pv_run(L,RB,PB,B,m,x0,c,out_one,out_final,pv);
}
void setup(void){ init_tables(); init_buffers(); }
