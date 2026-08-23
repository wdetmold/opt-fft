# B3/T2 (log 02:15:42): hybrid BL/PV batch-splitting dispatch (final API section)
src = open('impl_tail.c').read()
old = src[src.index('// ---------------------------------------------------------------- API'):]
new = '''// ---------------------------------------------------------------- API
static int FORCE_SCHEME = 0;  // 0=auto, 1=BL, 2=PV  (bench hook)
void set_scheme(int s){ FORCE_SCHEME=s; }

typedef void (*stepfn)(v8*,const v8*);
void run_size(long L, long B, long m, const double *x0, const double *c,
              double *out_one, double *out_final){
    if(B<=0 || m<=0) return;
    stepfn bl=0, pv=0; long RB=0, PB=0, remt=0;
    switch(L){
        case 6:  bl=bl_step_6;  pv=pv_step_6;  RB=1; PB=8;   remt=4; break;
        case 8:  bl=bl_step_8;  pv=pv_step_8;  RB=1; PB=8;   remt=6; break;
        case 13: bl=bl_step_13; pv=pv_step_13; RB=2; PB=26;  remt=5; break;
        case 17: bl=bl_step_17; pv=pv_step_17; RB=3; PB=51;  remt=4; break;
        case 23: bl=bl_step_23; pv=pv_step_23; RB=3; PB=69;  remt=7; break;
        case 36: pv=pv_step_36; RB=5; PB=180; break;
        case 45: pv=pv_step_45; RB=6; PB=270; break;
        case 64: pv=pv_step_64; RB=8; PB=512; break;
    }
    if(!bl){ pv_run(L,RB,PB,B,m,x0,c,out_one,out_final,pv); return; }
    if(FORCE_SCHEME==1){ bl_run(L,B,m,x0,c,out_one,out_final,bl); return; }
    if(FORCE_SCHEME==2){ pv_run(L,RB,PB,B,m,x0,c,out_one,out_final,pv); return; }
    long L3=L*L*L, Gfull=B/8, r=B%8;
    if(r==0){ bl_run(L,B,m,x0,c,out_one,out_final,bl); return; }
    if(r<=remt){
        if(Gfull>0) bl_run(L,Gfull*8,m,x0,c,out_one,out_final,bl);
        long off=Gfull*8*L3*2;
        pv_run(L,RB,PB,r,m,x0+off,c+off,out_one+off,out_final+off,pv);
    }else{
        bl_run(L,B,m,x0,c,out_one,out_final,bl);
    }
}
void setup(void){ init_tables(); init_buffers(); }
'''
src = src[:src.index('// ---------------------------------------------------------------- API')] + new
open('impl_tail.c','w').write(src)
print("ok")
