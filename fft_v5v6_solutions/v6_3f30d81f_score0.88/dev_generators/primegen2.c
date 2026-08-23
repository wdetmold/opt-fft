// generated: 13-point DFT, k-block 2, twiddles: both
static __attribute__((always_inline)) inline
void k13(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
          const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd ur[7], ui[7], vr[7], vi[7];
    vd x0r = xr[0], x0i = xi[0];
    vd s0r = x0r, s0i = x0i, s1r = (vd){0}, s1i = (vd){0};
    _Pragma("GCC unroll 16")
    for (int j = 1; j <= 6; j++) {
        vd ar = xr[(ptrdiff_t)j*s],      ai = xi[(ptrdiff_t)j*s];
        vd br = xr[(ptrdiff_t)(13-j)*s], bi = xi[(ptrdiff_t)(13-j)*s];
        ur[j] = ar + br; ui[j] = ai + bi;
        vr[j] = ar - br; vi[j] = ai - bi;
        if (j & 1) { s0r += ur[j]; s0i += ui[j]; }
        else       { s1r += ur[j]; s1i += ui[j]; }
    }
    KSTORE(xr[0], xi[0], s0r + s1r, s0i + s1i, cr[0], ci[0], domap);
    const vd cw1 = SPLAT(CW13[1]), sw1 = SPLAT(SW13[1]);
    const vd cw2 = SPLAT(CW13[2]), sw2 = SPLAT(SW13[2]);
    const vd cw3 = SPLAT(CW13[3]), sw3 = SPLAT(SW13[3]);
    const vd cw4 = SPLAT(CW13[4]), sw4 = SPLAT(SW13[4]);
    const vd cw5 = SPLAT(CW13[5]), sw5 = SPLAT(SW13[5]);
    const vd cw6 = SPLAT(CW13[6]), sw6 = SPLAT(SW13[6]);
    { // k = 1..2
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          A0 += u*cw1; C0 += iu*cw1; B0 += iv*sw1; D0 += v*sw1;
          A1 += u*cw2; C1 += iu*cw2; B1 += iv*sw2; D1 += v*sw2;
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          A0 += u*cw2; C0 += iu*cw2; B0 += iv*sw2; D0 += v*sw2;
          A1 += u*cw4; C1 += iu*cw4; B1 += iv*sw4; D1 += v*sw4;
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          A0 += u*cw3; C0 += iu*cw3; B0 += iv*sw3; D0 += v*sw3;
          A1 += u*cw6; C1 += iu*cw6; B1 += iv*sw6; D1 += v*sw6;
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          A0 += u*cw4; C0 += iu*cw4; B0 += iv*sw4; D0 += v*sw4;
          A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sw5; D1 -= v*sw5;
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          A0 += u*cw5; C0 += iu*cw5; B0 += iv*sw5; D0 += v*sw5;
          A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sw3; D1 -= v*sw3;
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          A0 += u*cw6; C0 += iu*cw6; B0 += iv*sw6; D0 += v*sw6;
          A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sw1; D1 -= v*sw1;
        }
        { ptrdiff_t ka = (ptrdiff_t)1*s, kb2 = (ptrdiff_t)12*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)2*s, kb2 = (ptrdiff_t)11*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 3..4
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          A0 += u*cw3; C0 += iu*cw3; B0 += iv*sw3; D0 += v*sw3;
          A1 += u*cw4; C1 += iu*cw4; B1 += iv*sw4; D1 += v*sw4;
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          A0 += u*cw6; C0 += iu*cw6; B0 += iv*sw6; D0 += v*sw6;
          A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sw5; D1 -= v*sw5;
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          A0 += u*cw4; C0 += iu*cw4; B0 -= iv*sw4; D0 -= v*sw4;
          A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sw1; D1 -= v*sw1;
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sw1; D0 -= v*sw1;
          A1 += u*cw3; C1 += iu*cw3; B1 += iv*sw3; D1 += v*sw3;
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          A0 += u*cw2; C0 += iu*cw2; B0 += iv*sw2; D0 += v*sw2;
          A1 += u*cw6; C1 += iu*cw6; B1 -= iv*sw6; D1 -= v*sw6;
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          A0 += u*cw5; C0 += iu*cw5; B0 += iv*sw5; D0 += v*sw5;
          A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sw2; D1 -= v*sw2;
        }
        { ptrdiff_t ka = (ptrdiff_t)3*s, kb2 = (ptrdiff_t)10*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)4*s, kb2 = (ptrdiff_t)9*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 5..6
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          A0 += u*cw5; C0 += iu*cw5; B0 += iv*sw5; D0 += v*sw5;
          A1 += u*cw6; C1 += iu*cw6; B1 += iv*sw6; D1 += v*sw6;
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          A0 += u*cw3; C0 += iu*cw3; B0 -= iv*sw3; D0 -= v*sw3;
          A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sw1; D1 -= v*sw1;
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          A0 += u*cw2; C0 += iu*cw2; B0 += iv*sw2; D0 += v*sw2;
          A1 += u*cw5; C1 += iu*cw5; B1 += iv*sw5; D1 += v*sw5;
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sw6; D0 -= v*sw6;
          A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sw2; D1 -= v*sw2;
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sw1; D0 -= v*sw1;
          A1 += u*cw4; C1 += iu*cw4; B1 += iv*sw4; D1 += v*sw4;
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          A0 += u*cw4; C0 += iu*cw4; B0 += iv*sw4; D0 += v*sw4;
          A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sw3; D1 -= v*sw3;
        }
        { ptrdiff_t ka = (ptrdiff_t)5*s, kb2 = (ptrdiff_t)8*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)6*s, kb2 = (ptrdiff_t)7*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
}

// generated: 17-point DFT, k-block 3, twiddles: cos
static __attribute__((always_inline)) inline
void k17(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
          const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd ur[9], ui[9], vr[9], vi[9];
    vd x0r = xr[0], x0i = xi[0];
    vd s0r = x0r, s0i = x0i, s1r = (vd){0}, s1i = (vd){0};
    _Pragma("GCC unroll 16")
    for (int j = 1; j <= 8; j++) {
        vd ar = xr[(ptrdiff_t)j*s],      ai = xi[(ptrdiff_t)j*s];
        vd br = xr[(ptrdiff_t)(17-j)*s], bi = xi[(ptrdiff_t)(17-j)*s];
        ur[j] = ar + br; ui[j] = ai + bi;
        vr[j] = ar - br; vi[j] = ai - bi;
        if (j & 1) { s0r += ur[j]; s0i += ui[j]; }
        else       { s1r += ur[j]; s1i += ui[j]; }
    }
    KSTORE(xr[0], xi[0], s0r + s1r, s0i + s1i, cr[0], ci[0], domap);
    const vd cw1 = SPLAT(CW17[1]);
    const vd cw2 = SPLAT(CW17[2]);
    const vd cw3 = SPLAT(CW17[3]);
    const vd cw4 = SPLAT(CW17[4]);
    const vd cw5 = SPLAT(CW17[5]);
    const vd cw6 = SPLAT(CW17[6]);
    const vd cw7 = SPLAT(CW17[7]);
    const vd cw8 = SPLAT(CW17[8]);
    { // k = 1..3
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        vd A2 = x0r, C2 = x0i, B2 = (vd){0}, D2 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW17[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A2 += u*cw3; C2 += iu*cw3; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW17[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A2 += u*cw6; C2 += iu*cw6; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW17[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A2 += u*cw8; C2 += iu*cw8; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW17[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A2 += u*cw5; C2 += iu*cw5; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW17[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A2 += u*cw2; C2 += iu*cw2; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW17[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A2 += u*cw1; C2 += iu*cw1; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW17[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A2 += u*cw4; C2 += iu*cw4; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW17[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A2 += u*cw7; C2 += iu*cw7; B2 += iv*sb; D2 += v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)1*s, kb2 = (ptrdiff_t)16*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)2*s, kb2 = (ptrdiff_t)15*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)3*s, kb2 = (ptrdiff_t)14*s;
          KSTORE(xr[ka], xi[ka], A2 + B2, C2 - D2, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A2 - B2, C2 + D2, cr[kb2], ci[kb2], domap); }
    }
    { // k = 4..6
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        vd A2 = x0r, C2 = x0i, B2 = (vd){0}, D2 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW17[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A2 += u*cw6; C2 += iu*cw6; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW17[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A2 += u*cw5; C2 += iu*cw5; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW17[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A2 += u*cw1; C2 += iu*cw1; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW17[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A2 += u*cw7; C2 += iu*cw7; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW17[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A2 += u*cw4; C2 += iu*cw4; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW17[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A2 += u*cw2; C2 += iu*cw2; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW17[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A2 += u*cw8; C2 += iu*cw8; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW17[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A2 += u*cw3; C2 += iu*cw3; B2 -= iv*sb; D2 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)4*s, kb2 = (ptrdiff_t)13*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)5*s, kb2 = (ptrdiff_t)12*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)6*s, kb2 = (ptrdiff_t)11*s;
          KSTORE(xr[ka], xi[ka], A2 + B2, C2 - D2, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A2 - B2, C2 + D2, cr[kb2], ci[kb2], domap); }
    }
    { // k = 7..8
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW17[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW17[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW17[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW17[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW17[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW17[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW17[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW17[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)7*s, kb2 = (ptrdiff_t)10*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)8*s, kb2 = (ptrdiff_t)9*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
}

// generated: 23-point DFT, k-block 2, twiddles: cos
static __attribute__((always_inline)) inline
void k23(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
          const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd ur[12], ui[12], vr[12], vi[12];
    vd x0r = xr[0], x0i = xi[0];
    vd s0r = x0r, s0i = x0i, s1r = (vd){0}, s1i = (vd){0};
    _Pragma("GCC unroll 16")
    for (int j = 1; j <= 11; j++) {
        vd ar = xr[(ptrdiff_t)j*s],      ai = xi[(ptrdiff_t)j*s];
        vd br = xr[(ptrdiff_t)(23-j)*s], bi = xi[(ptrdiff_t)(23-j)*s];
        ur[j] = ar + br; ui[j] = ai + bi;
        vr[j] = ar - br; vi[j] = ai - bi;
        if (j & 1) { s0r += ur[j]; s0i += ui[j]; }
        else       { s1r += ur[j]; s1i += ui[j]; }
    }
    KSTORE(xr[0], xi[0], s0r + s1r, s0i + s1i, cr[0], ci[0], domap);
    const vd cw1 = SPLAT(CW23[1]);
    const vd cw2 = SPLAT(CW23[2]);
    const vd cw3 = SPLAT(CW23[3]);
    const vd cw4 = SPLAT(CW23[4]);
    const vd cw5 = SPLAT(CW23[5]);
    const vd cw6 = SPLAT(CW23[6]);
    const vd cw7 = SPLAT(CW23[7]);
    const vd cw8 = SPLAT(CW23[8]);
    const vd cw9 = SPLAT(CW23[9]);
    const vd cw10 = SPLAT(CW23[10]);
    const vd cw11 = SPLAT(CW23[11]);
    { // k = 1..2
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)1*s, kb2 = (ptrdiff_t)22*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)2*s, kb2 = (ptrdiff_t)21*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 3..4
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)3*s, kb2 = (ptrdiff_t)20*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)4*s, kb2 = (ptrdiff_t)19*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 5..6
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)5*s, kb2 = (ptrdiff_t)18*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)6*s, kb2 = (ptrdiff_t)17*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 7..8
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)7*s, kb2 = (ptrdiff_t)16*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)8*s, kb2 = (ptrdiff_t)15*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 9..10
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)9*s, kb2 = (ptrdiff_t)14*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)10*s, kb2 = (ptrdiff_t)13*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 11..11
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 += iv*sb; D0 += v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)11*s, kb2 = (ptrdiff_t)12*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
    }
}
