"""Generic SoA-8 driver for one L: arenas, conversions, sweeps, run entry. XS = padded x-block stride (doubles)."""
from glib import Emit

def emit_soa_engine(e, L, emit_pass_funcs, XS=None):
    N3 = L*L*L
    if XS is None: XS = L*L*8
    e(f"// ============ L={L} SoA-8 (XS={XS}) ============")
    e(f"static double S{L}RE[{L*XS}] ALIGN64;")
    e(f"static double S{L}IM[{L*XS}] ALIGN64;")
    e(f"static double C{L}RE[{L*XS}] ALIGN64;")
    e(f"static double C{L}IM[{L*XS}] ALIGN64;")
    emit_pass_funcs(e, L)
    e(f"""
static void sw{L}_SyX(int y0, int pre){{
    long b0 = (long)y0*{L}*8;
    p{L}_xxm(S{L}RE + b0, S{L}IM + b0, C{L}RE + b0, C{L}IM + b0, {L}, 8);
    if (pre) {{
        for (int x = 0; x < {L}; x++)
            p{L}_zz(S{L}RE + (long)x*{XS} + (long)y0*{L}*8, S{L}IM + (long)x*{XS} + (long)y0*{L}*8, 1, 8);
        p{L}_xx(S{L}RE + b0, S{L}IM + b0, {L}, 8);
    }}
}}
static void sw{L}_PxY(int x0, int pre){{
    long b0 = (long)x0*{XS};
    p{L}_yym(S{L}RE + b0, S{L}IM + b0, C{L}RE + b0, C{L}IM + b0, {L}, 8);
    if (pre) {{
        p{L}_zz(S{L}RE + b0, S{L}IM + b0, {L}, (long){L}*8);
        p{L}_yy(S{L}RE + b0, S{L}IM + b0, {L}, 8);
    }}
}}
void run_{L}(const double* x0, const double* c, double* out1, double* outm, long B, long m){{
    if (m < 1) m = 1;
    long vs = 2*(long){N3};
    for (long g0 = 0; g0 < B; g0 += 8) {{
        int nv = (B - g0) < 8 ? (int)(B - g0) : 8;
        const double* xg = x0 + g0*vs;
        const double* cg = c + g0*vs;
        for (int x = 0; x < {L}; x++) {{
            long nb = ({L}*{L}/8)*8;
            long sb = (long)x*{L}*{L};
            if (nv == 8) {{
                for (long s = 0; s < nb; s += 8) {{
                    soa_in8(xg + 2*sb, vs, s, S{L}RE + (long)x*{XS}, S{L}IM + (long)x*{XS});
                    soa_in8(cg + 2*sb, vs, s, C{L}RE + (long)x*{XS}, C{L}IM + (long)x*{XS});
                }}
            }} else {{
                for (long s = 0; s < nb; s += 8) {{
                    soa_in8_nv(xg + 2*sb, vs, s, S{L}RE + (long)x*{XS}, S{L}IM + (long)x*{XS}, nv);
                    soa_in8_nv(cg + 2*sb, vs, s, C{L}RE + (long)x*{XS}, C{L}IM + (long)x*{XS}, nv);
                }}
            }}
            for (long s = nb; s < {L}*{L}; s++) {{
                for (int v = 0; v < 8; v++) {{
                    S{L}RE[(long)x*{XS}+s*8+v] = v < nv ? xg[v*vs + 2*(sb+s)] : 0.0;
                    S{L}IM[(long)x*{XS}+s*8+v] = v < nv ? xg[v*vs + 2*(sb+s) + 1] : 0.0;
                    C{L}RE[(long)x*{XS}+s*8+v] = v < nv ? cg[v*vs + 2*(sb+s)] : 0.0;
                    C{L}IM[(long)x*{XS}+s*8+v] = v < nv ? cg[v*vs + 2*(sb+s) + 1] : 0.0;
                }}
            }}
        }}
        for (int x = 0; x < {L}; x++) {{
            long b0 = (long)x*{XS};
            p{L}_zz(S{L}RE + b0, S{L}IM + b0, {L}*{L}, 8);   // wrong: z-pencils per (x,y): base x*XS + y*L*8 stride 8... zz with n={L} per y
        }}
        for (int x = 0; x < {L}; x++) {{
            long b0 = (long)x*{XS};
            p{L}_yy(S{L}RE + b0, S{L}IM + b0, {L}, 8);
        }}
        for (long t = 1; t <= m; t++) {{
            int snap = (t == 1) || (t == m);
            int pre = (t < m);
            int dopre = pre && !snap;
            if (t & 1) {{ for (int y0 = 0; y0 < {L}; y0++) sw{L}_SyX(y0, dopre); }}
            else       {{ for (int xp = 0; xp < {L}; xp++) sw{L}_PxY(xp, dopre); }}
            if (snap) {{
                double* tgt1 = (t == 1) ? out1 : 0;
                double* tgtm = (t == m) ? outm : 0;
                for (int x = 0; x < {L}; x++) {{
                    long nb = ({L}*{L}/8)*8;
                    long sb = (long)x*{L}*{L};
                    if (tgt1) for (long s = 0; s < nb; s += 8)
                        soa_out8(S{L}RE + (long)x*{XS}, S{L}IM + (long)x*{XS}, s, tgt1 + g0*vs + 2*sb, vs, nv);
                    if (tgtm && tgtm != tgt1) for (long s = 0; s < nb; s += 8)
                        soa_out8(S{L}RE + (long)x*{XS}, S{L}IM + (long)x*{XS}, s, tgtm + g0*vs + 2*sb, vs, nv);
                    if (tgt1 == tgtm && tgt1) {{}}
                    for (long s = nb; s < {L}*{L}; s++) for (int v = 0; v < nv; v++) {{
                        if (tgt1) {{ tgt1[g0*vs + v*vs + 2*(sb+s)] = S{L}RE[(long)x*{XS}+s*8+v]; tgt1[g0*vs + v*vs + 2*(sb+s)+1] = S{L}IM[(long)x*{XS}+s*8+v]; }}
                        if (tgtm) {{ tgtm[g0*vs + v*vs + 2*(sb+s)] = S{L}RE[(long)x*{XS}+s*8+v]; tgtm[g0*vs + v*vs + 2*(sb+s)+1] = S{L}IM[(long)x*{XS}+s*8+v]; }}
                    }}
                }}
                if (t == 1 && t == m) {{
                    // outm copy handled above via tgtm
                }}
            }}
            if (pre && !dopre) {{
                for (int x = 0; x < {L}; x++)
                    p{L}_zz(S{L}RE + (long)x*{XS}, S{L}IM + (long)x*{XS}, {L}*{L}, 8);
                if (t & 1) {{
                    for (int y0 = 0; y0 < {L}; y0++)
                        p{L}_xx(S{L}RE + (long)y0*{L}*8, S{L}IM + (long)y0*{L}*8, {L}, 8);
                }} else {{
                    for (int x0 = 0; x0 < {L}; x0++)
                        p{L}_yy(S{L}RE + (long)x0*{XS}, S{L}IM + (long)x0*{XS}, {L}, 8);
                }}
            }}
        }}
    }}
}}
""")
