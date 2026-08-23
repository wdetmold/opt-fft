import re
src = open('implementation.c').read()
for N in (13,17,23):
    NN=N*N; PS=N*N*16
    # 1. declare CP arena next to XG/CG decls
    anchor = f"static double* XG_{N} = 0;"
    assert anchor in src
    src = src.replace(anchor, anchor + f"\nstatic double* CP_{N} = 0;", 1)
    # 2. alloc + build in run_N
    anchor = f"if(!XG_{N}){{ XG_{N} = alloc_huge_st((long){N}*{PS}*8); CG_{N} = alloc_huge_st((long){N}*{PS}*8); }}"
    assert anchor in src, (N,'alloc')
    src = src.replace(anchor, f"if(!XG_{N}){{ XG_{N} = alloc_huge_st((long){N}*{PS}*8); CG_{N} = alloc_huge_st((long){N}*{PS}*8); CP_{N} = alloc_huge_st((long){N}*{PS}*8); }}", 1)
    anchor = f"""        convin_{N}(srcs, CG_{N});
        S_{N}(XG_{N}, CG_{N}, 0, 0);"""
    assert anchor in src, (N,'convin c')
    src = src.replace(anchor, f"""        convin_{N}(srcs, CG_{N});
        {{   // column-order copy of c for the P pass (sequential map loads)
            for(int i=0;i<{N};i++){{
                const double* s = CG_{N} + (long)i*{PS};
                double* d = CP_{N} + (long)i*16;
                for(long e=0;e<{NN};e++){{
                    _mm512_store_pd(d + e*{N*16}, _mm512_load_pd(s + e*16));
                    _mm512_store_pd(d + e*{N*16} + 8, _mm512_load_pd(s + e*16 + 8));
                }}
            }}
        }}
        S_{N}(XG_{N}, CG_{N}, 0, 0);""", 1)
    # 3. P call sites pass CP
    src = src.replace(f"P_{N}(XG_{N}, CG_{N}, 0);", f"P_{N}(XG_{N}, CP_{N}, 0);")
    src = src.replace(f"if((t & 1) == 0) P_{N}(XG_{N}, CG_{N}, t<m);", f"if((t & 1) == 0) P_{N}(XG_{N}, CP_{N}, t<m);")
    # 4. P map loads from contiguous layout: replace cp usage
    sP = src.index(f"static void P_{N}(double* X")
    eP = src.index("\nstatic", sP+10)
    body = src[sP:eP]
    body = body.replace(f"const double* cp = C + e*16;", f"const double* cp = C + e*{N*16};")
    n0 = body.count("cp2 + t*")
    body = body.replace(f"_mm512_load_pd(cp2 + t*{PS} + {PS} + 8)", f"_mm512_load_pd(cp2 + t*16 + 24)")
    body = body.replace(f"_mm512_load_pd(cp2 + t*{PS} + {PS})", f"_mm512_load_pd(cp2 + t*16 + 16)")
    body = body.replace(f"_mm512_load_pd(cp2 + t*{PS} + 8)", f"_mm512_load_pd(cp2 + t*16 + 8)")
    body = body.replace(f"_mm512_load_pd(cp2 + t*{PS})", f"_mm512_load_pd(cp2 + t*16)")
    assert f"t*{PS}" not in body.replace(f"p2 + t*{PS}", "XX"), (N, "leftover cp strides")
    src = src[:sP] + body + src[eP:]
open('implementation.c','w').write(src)
print("patch13 applied")