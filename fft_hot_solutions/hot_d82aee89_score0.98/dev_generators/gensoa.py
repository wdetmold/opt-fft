"""Generic SoA-8-volume engine: one-sweep-per-step, all passes vertical."""
import numpy as np
from glib import Emit, trig, hexd, PRELUDE

def emit_conv(e):
    # natural interleaved (8 volumes) <-> SoA split. Per 8-site block:
    # in: vol v row chunk: 2 zmm (16 doubles interleaved) -> split re/im (PERM2) -> TR8 across vols
    e("""
// convert 8 sites x 8 vols from natural (per-vol interleaved) to SoA split
static inline __attribute__((always_inline)) void soa_in8(const double* xv, long vstride, long site, double* RE, double* IM){
    V r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7;
#define LD1(v) { V a=VLU(xv + v*vstride + 2*site), b=VLU(xv + v*vstride + 2*site + 8); \
    r##v = PERM2(a, IDXR_, b); i##v = PERM2(a, IDXI_, b); }
    LD1(0) LD1(1) LD1(2) LD1(3) LD1(4) LD1(5) LD1(6) LD1(7)
#undef LD1
    TR8(r0,r1,r2,r3,r4,r5,r6,r7);
    TR8(i0,i1,i2,i3,i4,i5,i6,i7);
    VS(RE + site*8, r0); VS(RE + site*8+8, r1); VS(RE + site*8+16, r2); VS(RE + site*8+24, r3);
    VS(RE + site*8+32, r4); VS(RE + site*8+40, r5); VS(RE + site*8+48, r6); VS(RE + site*8+56, r7);
    VS(IM + site*8, i0); VS(IM + site*8+8, i1); VS(IM + site*8+16, i2); VS(IM + site*8+24, i3);
    VS(IM + site*8+32, i4); VS(IM + site*8+40, i5); VS(IM + site*8+48, i6); VS(IM + site*8+56, i7);
}
// same but with zero-fill for vols >= nv (tail groups)
static void soa_in8_nv(const double* xv, long vstride, long site, double* RE, double* IM, int nv){
    V r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7;
#define LD1(v) if (nv > v) { V a=VLU(xv + v*vstride + 2*site), b=VLU(xv + v*vstride + 2*site + 8); \
    r##v = PERM2(a, IDXR_, b); i##v = PERM2(a, IDXI_, b); } else { r##v = _mm512_setzero_pd(); i##v = _mm512_setzero_pd(); }
    LD1(0) LD1(1) LD1(2) LD1(3) LD1(4) LD1(5) LD1(6) LD1(7)
#undef LD1
    TR8(r0,r1,r2,r3,r4,r5,r6,r7);
    TR8(i0,i1,i2,i3,i4,i5,i6,i7);
    VS(RE + site*8, r0); VS(RE + site*8+8, r1); VS(RE + site*8+16, r2); VS(RE + site*8+24, r3);
    VS(RE + site*8+32, r4); VS(RE + site*8+40, r5); VS(RE + site*8+48, r6); VS(RE + site*8+56, r7);
    VS(IM + site*8, i0); VS(IM + site*8+8, i1); VS(IM + site*8+16, i2); VS(IM + site*8+24, i3);
    VS(IM + site*8+32, i4); VS(IM + site*8+40, i5); VS(IM + site*8+48, i6); VS(IM + site*8+56, i7);
}
// SoA -> natural (snapshot), 8 sites x nv vols
static void soa_out8(const double* RE, const double* IM, long site, double* ov, long vstride, int nv){
    V r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7;
    r0=VL(RE+site*8); r1=VL(RE+site*8+8); r2=VL(RE+site*8+16); r3=VL(RE+site*8+24);
    r4=VL(RE+site*8+32); r5=VL(RE+site*8+40); r6=VL(RE+site*8+48); r7=VL(RE+site*8+56);
    i0=VL(IM+site*8); i1=VL(IM+site*8+8); i2=VL(IM+site*8+16); i3=VL(IM+site*8+24);
    i4=VL(IM+site*8+32); i5=VL(IM+site*8+40); i6=VL(IM+site*8+48); i7=VL(IM+site*8+56);
    TR8(r0,r1,r2,r3,r4,r5,r6,r7);
    TR8(i0,i1,i2,i3,i4,i5,i6,i7);
#define ST1(v) if (nv > v) { VSU(ov + v*vstride + 2*site, PERM2(r##v, IDXLO_, i##v)); \
    VSU(ov + v*vstride + 2*site + 8, PERM2(r##v, IDXHI_, i##v)); }
    ST1(0) ST1(1) ST1(2) ST1(3) ST1(4) ST1(5) ST1(6) ST1(7)
#undef ST1
}
""")

def emit_map_pass(e, name):
    # map a run of nsite sites (SoA): z += c ; z/(1+|z|). csoa aligned.
    e(f"""
static void {name}(double* RE, double* IM, const double* CRE, const double* CIM, long n8){{
    for (long i = 0; i < n8; i++) {{
        V zr = VL(RE + i*8), zi = VL(IM + i*8);
        zr = VADD(zr, VL(CRE + i*8)); zi = VADD(zi, VL(CIM + i*8));
        MAP2(zr, zi);
        VS(RE + i*8, zr); VS(IM + i*8, zi);
    }}
}}
""")
