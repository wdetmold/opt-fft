
#define TR8(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    V u0=_mm512_unpacklo_pd(r0,r1), u1=_mm512_unpackhi_pd(r0,r1); \
    V u2=_mm512_unpacklo_pd(r2,r3), u3=_mm512_unpackhi_pd(r2,r3); \
    V u4=_mm512_unpacklo_pd(r4,r5), u5=_mm512_unpackhi_pd(r4,r5); \
    V u6=_mm512_unpacklo_pd(r6,r7), u7=_mm512_unpackhi_pd(r6,r7); \
    V s0=_mm512_shuffle_f64x2(u0,u2,0x88), s1=_mm512_shuffle_f64x2(u1,u3,0x88); \
    V s2=_mm512_shuffle_f64x2(u0,u2,0xdd), s3=_mm512_shuffle_f64x2(u1,u3,0xdd); \
    V s4=_mm512_shuffle_f64x2(u4,u6,0x88), s5=_mm512_shuffle_f64x2(u5,u7,0x88); \
    V s6=_mm512_shuffle_f64x2(u4,u6,0xdd), s7=_mm512_shuffle_f64x2(u5,u7,0xdd); \
    r0=_mm512_shuffle_f64x2(s0,s4,0x88); r4=_mm512_shuffle_f64x2(s0,s4,0xdd); \
    r1=_mm512_shuffle_f64x2(s1,s5,0x88); r5=_mm512_shuffle_f64x2(s1,s5,0xdd); \
    r2=_mm512_shuffle_f64x2(s2,s6,0x88); r6=_mm512_shuffle_f64x2(s2,s6,0xdd); \
    r3=_mm512_shuffle_f64x2(s3,s7,0x88); r7=_mm512_shuffle_f64x2(s3,s7,0xdd); \
} while(0)
