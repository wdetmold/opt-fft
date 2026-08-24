#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
typedef __uint128_t u128;
#define PCG_MUL ((((u128)2549297995355413924ULL) << 64) | 4865540595714422341ULL)
static inline uint64_t rotr64(uint64_t v, unsigned rot){ return (v >> rot) | (v << ((64u - rot) & 63u)); }
static inline uint64_t pcg_out_s(u128 s){ return rotr64((uint64_t)(s >> 64) ^ (uint64_t)s, (unsigned)(s >> 122)); }
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
static uint64_t BUF[16384];
int main(){
  u128 s0=1,s1=2,s2=3,s3=4, a=PCG_MUL*PCG_MUL*PCG_MUL*PCG_MUL, c=12345;
  double t0=now(); long R=2000;
  for (long r=0;r<R;r++){
    for (long t=0;t<16384;t+=4){
      BUF[t]=pcg_out_s(s0); BUF[t+1]=pcg_out_s(s1); BUF[t+2]=pcg_out_s(s2); BUF[t+3]=pcg_out_s(s3);
      s0=s0*a+c; s1=s1*a+c; s2=s2*a+c; s3=s3*a+c;
    }
  }
  double t1=now();
  printf("refill: %.2f ns/u64 (chk %llu)\n", (t1-t0)/(R*16384.0)*1e9, (unsigned long long)BUF[7]);
  // consumer-only: ziggurat fast path on prefilled buffer
  static double OUT[16384];
  double w[256], f[256]; uint64_t k[256];
  for (int i=0;i<256;i++){ w[i]=1e-16*(i+1); f[i]=1.0/(i+1); k[i]=(uint64_t)1<<51; }
  t0=now();
  double acc=0;
  for (long r=0;r<R;r++){
    const uint64_t* p = BUF;
    for (long i=0;i<16384;i++){
      uint64_t rr = *p++;
      int idx = (int)(rr & 0xff);
      uint64_t rabs = (rr >> 9) & 0x000fffffffffffffULL;
      double x = (double)rabs * w[idx];
      if (rr & 0x100) x = -x;
      if (__builtin_expect(rabs < k[idx],1)) { OUT[i] = x; continue; }
      OUT[i] = x + f[idx];
    }
    acc += OUT[55];
  }
  t1=now();
  printf("consume: %.2f ns/sample (acc %f)\n", (t1-t0)/(R*16384.0)*1e9, acc);
  return 0;
}
