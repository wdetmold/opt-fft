import re
src = open('implementation.c').read()
a = src.index("void k6(vd *restrict xr")
b = src.index("}", src.index("KSTORE(xr[5*s]", a)) + 1
new6 = open('k6.txt').read()
src = src[:a] + new6 + src[b:]
a = src.index("void k8(vd *restrict xr")
b = src.index("}", src.index("KSTORE(xr[7*s]", a)) + 1
new8 = open('k8.txt').read()
src = src[:a] + new8 + src[b:]
old = 'static void conv_in_aosoaC(const double *restrict src, long nl, long N, long CNT, vd *restrict d) {'
new = open('convmap.txt').read() + old
assert old in src
src = src.replace(old, new, 1)
src = src.replace('''#define KSTOREP(dstR, dstI, vr_, vi_, cR, cI, DOMAP, PAR) do { \\''',
'''#define KSTORE3(dstR, dstI, vr_, vi_, cR, cI, MODE) do { \\
    vd _r = (vr_), _i = (vi_);                            \\
    if ((MODE) == 1) { _r += (cR); _i += (cI); }          \\
    (dstR) = _r; (dstI) = _i;                             \\
} while (0)
#define KSTOREP(dstR, dstI, vr_, vi_, cR, cI, DOMAP, PAR) do { \\''')
old = src[src.index("#define GEN_RUN_AOSOA(LL, XS, KP, KM)"):src.index("GEN_RUN_AOSOA(6,  36,  k6_p,  k6_m)")]
new = open('aosoa3.txt').read()
src = src.replace(old, new + "\n")
src = src.replace("GEN_RUN_AOSOA(6,  36,  k6_p,  k6_m)", "GEN_RUN_AOSOA3(6,  36,  k6)")
src = src.replace("GEN_RUN_AOSOA(8,  68,  k8_p,  k8_m)", "GEN_RUN_AOSOA3(8,  68,  k8)")
src = src.replace("GEN_RUN_AOSOA(13, 169, k13_p, k13_m)", "GEN_RUN_AOSOA3(13, 169, kg13)")
src = src.replace("GEN_RUN_AOSOA(17, 289, k17_p, k17_m)", "GEN_RUN_AOSOA3(17, 289, kg17)")
# GEN_WRAPPERS for k6/k8 now broken (signature same; k6_p used by...? nothing in AOSOA3; GEN_VOL for 23/36/45 still in template but replaced at assemble; k64 wrappers still used by 64)
open('implementation.c','w').write(src)
print("impl redone")
