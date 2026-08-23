#include <stdio.h>
#include <time.h>
#include "implementation.c"
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
int main(){
    long LL=64*64, SL=LL+8;
    double *XR = skew_alloc(64*SL,0), *XI = skew_alloc(64*SL,1);
    double *CR = skew_alloc(64*SL,4), *CI = skew_alloc(64*SL,5);
    for (long i=0;i<64*SL;i++){ XR[i]=0.5; XI[i]=0.3; CR[i]=0.1; CI[i]=0.05; }
    double best=1e9;
    for (int r=0;r<60;r++){
        double t0=now();
        iter_v64(XR, XI, CR, CI);
        double t1=now();
        if (t1-t0<best) best=t1-t0;
    }
    printf("full iter best: %7.1f us/vol\n", best*1e6);
    printf("chk %g\n", XR[5]);
    return 0;
}
