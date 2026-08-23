static const double CC_13[12] ALIGN64 = {0x1.c55a7e00740e9p-1,0x1.22d961ea71119p-1,0x1.edb7debaa3ed8p-4,-0x1.6b1d8b2365da1p-2,-0x1.7f3ccd0032e0cp-1,-0x1.f11f493053d00p-1,0x1.dbe064267c47cp-2,0x1.a55e242a4c3d2p-1,0x1.fc44566966769p-1,0x1.deba72ef20147p-1,0x1.5384d024c2f84p-1,0x1.ea1e54bc48dbfp-3};
static void dftp13_v(double* re, double* im, long es){
  double scr[152] __attribute__((aligned(64)));
  {
  const __m512d C0 = _mm512_set1_pd(CC_13[0]);
  const __m512d C1 = _mm512_set1_pd(CC_13[1]);
  const __m512d C2 = _mm512_set1_pd(CC_13[2]);
  const __m512d C3 = _mm512_set1_pd(CC_13[3]);
  const __m512d C4 = _mm512_set1_pd(CC_13[4]);
  const __m512d C5 = _mm512_set1_pd(CC_13[5]);
  __m512d x0r = _mm512_load_pd(re);
  __m512d s0r = x0r;
  __m512d ar1 = x0r;
  __m512d ar2 = x0r;
  __m512d ar3 = x0r;
  __m512d ar4 = x0r;
  __m512d ar5 = x0r;
  __m512d ar6 = x0r;
  { __m512d p = _mm512_load_pd(re + 1*es), q = _mm512_load_pd(re + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 0, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C0, e, ar1);
    ar2 = _mm512_fmadd_pd(C1, e, ar2);
    ar3 = _mm512_fmadd_pd(C2, e, ar3);
    ar4 = _mm512_fmadd_pd(C3, e, ar4);
    ar5 = _mm512_fmadd_pd(C4, e, ar5);
    ar6 = _mm512_fmadd_pd(C5, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 2*es), q = _mm512_load_pd(re + 11*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 8, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C1, e, ar1);
    ar2 = _mm512_fmadd_pd(C3, e, ar2);
    ar3 = _mm512_fmadd_pd(C5, e, ar3);
    ar4 = _mm512_fmadd_pd(C4, e, ar4);
    ar5 = _mm512_fmadd_pd(C2, e, ar5);
    ar6 = _mm512_fmadd_pd(C0, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 3*es), q = _mm512_load_pd(re + 10*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 16, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C2, e, ar1);
    ar2 = _mm512_fmadd_pd(C5, e, ar2);
    ar3 = _mm512_fmadd_pd(C3, e, ar3);
    ar4 = _mm512_fmadd_pd(C0, e, ar4);
    ar5 = _mm512_fmadd_pd(C1, e, ar5);
    ar6 = _mm512_fmadd_pd(C4, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 4*es), q = _mm512_load_pd(re + 9*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 24, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C3, e, ar1);
    ar2 = _mm512_fmadd_pd(C4, e, ar2);
    ar3 = _mm512_fmadd_pd(C0, e, ar3);
    ar4 = _mm512_fmadd_pd(C2, e, ar4);
    ar5 = _mm512_fmadd_pd(C5, e, ar5);
    ar6 = _mm512_fmadd_pd(C1, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 5*es), q = _mm512_load_pd(re + 8*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 32, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C4, e, ar1);
    ar2 = _mm512_fmadd_pd(C2, e, ar2);
    ar3 = _mm512_fmadd_pd(C1, e, ar3);
    ar4 = _mm512_fmadd_pd(C5, e, ar4);
    ar5 = _mm512_fmadd_pd(C0, e, ar5);
    ar6 = _mm512_fmadd_pd(C3, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 6*es), q = _mm512_load_pd(re + 7*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 40, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C5, e, ar1);
    ar2 = _mm512_fmadd_pd(C0, e, ar2);
    ar3 = _mm512_fmadd_pd(C4, e, ar3);
    ar4 = _mm512_fmadd_pd(C1, e, ar4);
    ar5 = _mm512_fmadd_pd(C3, e, ar5);
    ar6 = _mm512_fmadd_pd(C2, e, ar6);
  }
  _mm512_store_pd(re, s0r);
  _mm512_store_pd(scr + 104, ar1);
  _mm512_store_pd(scr + 112, ar2);
  _mm512_store_pd(scr + 120, ar3);
  _mm512_store_pd(scr + 128, ar4);
  _mm512_store_pd(scr + 136, ar5);
  _mm512_store_pd(scr + 144, ar6);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_13[6]);
  const __m512d S1 = _mm512_set1_pd(CC_13[7]);
  const __m512d S2 = _mm512_set1_pd(CC_13[8]);
  const __m512d S3 = _mm512_set1_pd(CC_13[9]);
  const __m512d S4 = _mm512_set1_pd(CC_13[10]);
  const __m512d S5 = _mm512_set1_pd(CC_13[11]);
  __m512d x0i = _mm512_load_pd(im);
  _mm512_store_pd(scr + 96, x0i);
  __m512d s0i = x0i;
  __m512d bi1 = _mm512_setzero_pd();
  __m512d bi2 = _mm512_setzero_pd();
  __m512d bi3 = _mm512_setzero_pd();
  __m512d bi4 = _mm512_setzero_pd();
  __m512d bi5 = _mm512_setzero_pd();
  __m512d bi6 = _mm512_setzero_pd();
  { __m512d p = _mm512_load_pd(im + 1*es), q = _mm512_load_pd(im + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 48, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S0, o, bi1);
    bi2 = _mm512_fmadd_pd(S1, o, bi2);
    bi3 = _mm512_fmadd_pd(S2, o, bi3);
    bi4 = _mm512_fmadd_pd(S3, o, bi4);
    bi5 = _mm512_fmadd_pd(S4, o, bi5);
    bi6 = _mm512_fmadd_pd(S5, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 2*es), q = _mm512_load_pd(im + 11*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 56, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S1, o, bi1);
    bi2 = _mm512_fmadd_pd(S3, o, bi2);
    bi3 = _mm512_fmadd_pd(S5, o, bi3);
    bi4 = _mm512_fnmadd_pd(S4, o, bi4);
    bi5 = _mm512_fnmadd_pd(S2, o, bi5);
    bi6 = _mm512_fnmadd_pd(S0, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 3*es), q = _mm512_load_pd(im + 10*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 64, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S2, o, bi1);
    bi2 = _mm512_fmadd_pd(S5, o, bi2);
    bi3 = _mm512_fnmadd_pd(S3, o, bi3);
    bi4 = _mm512_fnmadd_pd(S0, o, bi4);
    bi5 = _mm512_fmadd_pd(S1, o, bi5);
    bi6 = _mm512_fmadd_pd(S4, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 4*es), q = _mm512_load_pd(im + 9*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 72, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S3, o, bi1);
    bi2 = _mm512_fnmadd_pd(S4, o, bi2);
    bi3 = _mm512_fnmadd_pd(S0, o, bi3);
    bi4 = _mm512_fmadd_pd(S2, o, bi4);
    bi5 = _mm512_fnmadd_pd(S5, o, bi5);
    bi6 = _mm512_fnmadd_pd(S1, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 5*es), q = _mm512_load_pd(im + 8*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 80, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S4, o, bi1);
    bi2 = _mm512_fnmadd_pd(S2, o, bi2);
    bi3 = _mm512_fmadd_pd(S1, o, bi3);
    bi4 = _mm512_fnmadd_pd(S5, o, bi4);
    bi5 = _mm512_fnmadd_pd(S0, o, bi5);
    bi6 = _mm512_fmadd_pd(S3, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 6*es), q = _mm512_load_pd(im + 7*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 88, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S5, o, bi1);
    bi2 = _mm512_fnmadd_pd(S0, o, bi2);
    bi3 = _mm512_fmadd_pd(S4, o, bi3);
    bi4 = _mm512_fnmadd_pd(S1, o, bi4);
    bi5 = _mm512_fmadd_pd(S3, o, bi5);
    bi6 = _mm512_fnmadd_pd(S2, o, bi6);
  }
  _mm512_store_pd(im, s0i);
  { __m512d a = _mm512_load_pd(scr + 104);
    _mm512_store_pd(re + 1*es, _mm512_add_pd(a, bi1));
    _mm512_store_pd(re + 12*es, _mm512_sub_pd(a, bi1)); }
  { __m512d a = _mm512_load_pd(scr + 112);
    _mm512_store_pd(re + 2*es, _mm512_add_pd(a, bi2));
    _mm512_store_pd(re + 11*es, _mm512_sub_pd(a, bi2)); }
  { __m512d a = _mm512_load_pd(scr + 120);
    _mm512_store_pd(re + 3*es, _mm512_add_pd(a, bi3));
    _mm512_store_pd(re + 10*es, _mm512_sub_pd(a, bi3)); }
  { __m512d a = _mm512_load_pd(scr + 128);
    _mm512_store_pd(re + 4*es, _mm512_add_pd(a, bi4));
    _mm512_store_pd(re + 9*es, _mm512_sub_pd(a, bi4)); }
  { __m512d a = _mm512_load_pd(scr + 136);
    _mm512_store_pd(re + 5*es, _mm512_add_pd(a, bi5));
    _mm512_store_pd(re + 8*es, _mm512_sub_pd(a, bi5)); }
  { __m512d a = _mm512_load_pd(scr + 144);
    _mm512_store_pd(re + 6*es, _mm512_add_pd(a, bi6));
    _mm512_store_pd(re + 7*es, _mm512_sub_pd(a, bi6)); }
  }
  {
  const __m512d C0 = _mm512_set1_pd(CC_13[0]);
  const __m512d C1 = _mm512_set1_pd(CC_13[1]);
  const __m512d C2 = _mm512_set1_pd(CC_13[2]);
  const __m512d C3 = _mm512_set1_pd(CC_13[3]);
  const __m512d C4 = _mm512_set1_pd(CC_13[4]);
  const __m512d C5 = _mm512_set1_pd(CC_13[5]);
  __m512d x0i = _mm512_load_pd(scr + 96);
  __m512d ai1 = x0i;
  __m512d ai2 = x0i;
  __m512d ai3 = x0i;
  __m512d ai4 = x0i;
  __m512d ai5 = x0i;
  __m512d ai6 = x0i;
  { __m512d e = _mm512_load_pd(scr + 48);
    ai1 = _mm512_fmadd_pd(C0, e, ai1);
    ai2 = _mm512_fmadd_pd(C1, e, ai2);
    ai3 = _mm512_fmadd_pd(C2, e, ai3);
    ai4 = _mm512_fmadd_pd(C3, e, ai4);
    ai5 = _mm512_fmadd_pd(C4, e, ai5);
    ai6 = _mm512_fmadd_pd(C5, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 56);
    ai1 = _mm512_fmadd_pd(C1, e, ai1);
    ai2 = _mm512_fmadd_pd(C3, e, ai2);
    ai3 = _mm512_fmadd_pd(C5, e, ai3);
    ai4 = _mm512_fmadd_pd(C4, e, ai4);
    ai5 = _mm512_fmadd_pd(C2, e, ai5);
    ai6 = _mm512_fmadd_pd(C0, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 64);
    ai1 = _mm512_fmadd_pd(C2, e, ai1);
    ai2 = _mm512_fmadd_pd(C5, e, ai2);
    ai3 = _mm512_fmadd_pd(C3, e, ai3);
    ai4 = _mm512_fmadd_pd(C0, e, ai4);
    ai5 = _mm512_fmadd_pd(C1, e, ai5);
    ai6 = _mm512_fmadd_pd(C4, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 72);
    ai1 = _mm512_fmadd_pd(C3, e, ai1);
    ai2 = _mm512_fmadd_pd(C4, e, ai2);
    ai3 = _mm512_fmadd_pd(C0, e, ai3);
    ai4 = _mm512_fmadd_pd(C2, e, ai4);
    ai5 = _mm512_fmadd_pd(C5, e, ai5);
    ai6 = _mm512_fmadd_pd(C1, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 80);
    ai1 = _mm512_fmadd_pd(C4, e, ai1);
    ai2 = _mm512_fmadd_pd(C2, e, ai2);
    ai3 = _mm512_fmadd_pd(C1, e, ai3);
    ai4 = _mm512_fmadd_pd(C5, e, ai4);
    ai5 = _mm512_fmadd_pd(C0, e, ai5);
    ai6 = _mm512_fmadd_pd(C3, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 88);
    ai1 = _mm512_fmadd_pd(C5, e, ai1);
    ai2 = _mm512_fmadd_pd(C0, e, ai2);
    ai3 = _mm512_fmadd_pd(C4, e, ai3);
    ai4 = _mm512_fmadd_pd(C1, e, ai4);
    ai5 = _mm512_fmadd_pd(C3, e, ai5);
    ai6 = _mm512_fmadd_pd(C2, e, ai6);
  }
  _mm512_store_pd(scr + 104, ai1);
  _mm512_store_pd(scr + 112, ai2);
  _mm512_store_pd(scr + 120, ai3);
  _mm512_store_pd(scr + 128, ai4);
  _mm512_store_pd(scr + 136, ai5);
  _mm512_store_pd(scr + 144, ai6);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_13[6]);
  const __m512d S1 = _mm512_set1_pd(CC_13[7]);
  const __m512d S2 = _mm512_set1_pd(CC_13[8]);
  const __m512d S3 = _mm512_set1_pd(CC_13[9]);
  const __m512d S4 = _mm512_set1_pd(CC_13[10]);
  const __m512d S5 = _mm512_set1_pd(CC_13[11]);
  __m512d br1 = _mm512_setzero_pd();
  __m512d br2 = _mm512_setzero_pd();
  __m512d br3 = _mm512_setzero_pd();
  __m512d br4 = _mm512_setzero_pd();
  __m512d br5 = _mm512_setzero_pd();
  __m512d br6 = _mm512_setzero_pd();
  { __m512d o = _mm512_load_pd(scr + 0);
    br1 = _mm512_fmadd_pd(S0, o, br1);
    br2 = _mm512_fmadd_pd(S1, o, br2);
    br3 = _mm512_fmadd_pd(S2, o, br3);
    br4 = _mm512_fmadd_pd(S3, o, br4);
    br5 = _mm512_fmadd_pd(S4, o, br5);
    br6 = _mm512_fmadd_pd(S5, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 8);
    br1 = _mm512_fmadd_pd(S1, o, br1);
    br2 = _mm512_fmadd_pd(S3, o, br2);
    br3 = _mm512_fmadd_pd(S5, o, br3);
    br4 = _mm512_fnmadd_pd(S4, o, br4);
    br5 = _mm512_fnmadd_pd(S2, o, br5);
    br6 = _mm512_fnmadd_pd(S0, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 16);
    br1 = _mm512_fmadd_pd(S2, o, br1);
    br2 = _mm512_fmadd_pd(S5, o, br2);
    br3 = _mm512_fnmadd_pd(S3, o, br3);
    br4 = _mm512_fnmadd_pd(S0, o, br4);
    br5 = _mm512_fmadd_pd(S1, o, br5);
    br6 = _mm512_fmadd_pd(S4, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 24);
    br1 = _mm512_fmadd_pd(S3, o, br1);
    br2 = _mm512_fnmadd_pd(S4, o, br2);
    br3 = _mm512_fnmadd_pd(S0, o, br3);
    br4 = _mm512_fmadd_pd(S2, o, br4);
    br5 = _mm512_fnmadd_pd(S5, o, br5);
    br6 = _mm512_fnmadd_pd(S1, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 32);
    br1 = _mm512_fmadd_pd(S4, o, br1);
    br2 = _mm512_fnmadd_pd(S2, o, br2);
    br3 = _mm512_fmadd_pd(S1, o, br3);
    br4 = _mm512_fnmadd_pd(S5, o, br4);
    br5 = _mm512_fnmadd_pd(S0, o, br5);
    br6 = _mm512_fmadd_pd(S3, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 40);
    br1 = _mm512_fmadd_pd(S5, o, br1);
    br2 = _mm512_fnmadd_pd(S0, o, br2);
    br3 = _mm512_fmadd_pd(S4, o, br3);
    br4 = _mm512_fnmadd_pd(S1, o, br4);
    br5 = _mm512_fmadd_pd(S3, o, br5);
    br6 = _mm512_fnmadd_pd(S2, o, br6);
  }
  { __m512d a = _mm512_load_pd(scr + 104);
    _mm512_store_pd(im + 1*es, _mm512_sub_pd(a, br1));
    _mm512_store_pd(im + 12*es, _mm512_add_pd(a, br1)); }
  { __m512d a = _mm512_load_pd(scr + 112);
    _mm512_store_pd(im + 2*es, _mm512_sub_pd(a, br2));
    _mm512_store_pd(im + 11*es, _mm512_add_pd(a, br2)); }
  { __m512d a = _mm512_load_pd(scr + 120);
    _mm512_store_pd(im + 3*es, _mm512_sub_pd(a, br3));
    _mm512_store_pd(im + 10*es, _mm512_add_pd(a, br3)); }
  { __m512d a = _mm512_load_pd(scr + 128);
    _mm512_store_pd(im + 4*es, _mm512_sub_pd(a, br4));
    _mm512_store_pd(im + 9*es, _mm512_add_pd(a, br4)); }
  { __m512d a = _mm512_load_pd(scr + 136);
    _mm512_store_pd(im + 5*es, _mm512_sub_pd(a, br5));
    _mm512_store_pd(im + 8*es, _mm512_add_pd(a, br5)); }
  { __m512d a = _mm512_load_pd(scr + 144);
    _mm512_store_pd(im + 6*es, _mm512_sub_pd(a, br6));
    _mm512_store_pd(im + 7*es, _mm512_add_pd(a, br6)); }
  }
}
static const double CC_17[16] ALIGN64 = {0x1.dd6d000370991p-1,0x1.7a5f6075d4884p-1,0x1.c86fa2b2883cdp-2,0x1.79ee63259b75ep-4,-0x1.183b1c61f0d01p-2,-0x1.348c86ed5f1bbp-1,-0x1.b34fa910ea3b9p-1,-0x1.f7484007faef3p-1,0x1.71e955d8e7cdcp-2,0x1.58eea2a9d6da3p-1,0x1.ca52d7c9e640bp-1,0x1.fdd0deb564b22p-1,0x1.ec746923c349fp-1,0x1.9895b6c9a05f6p-1,0x1.0d8884363dd80p-1,0x1.7851aacd6c6b4p-3};
static void dftp17_v(double* re, double* im, long es){
  double scr[200] __attribute__((aligned(64)));
  {
  const __m512d C0 = _mm512_set1_pd(CC_17[0]);
  const __m512d C1 = _mm512_set1_pd(CC_17[1]);
  const __m512d C2 = _mm512_set1_pd(CC_17[2]);
  const __m512d C3 = _mm512_set1_pd(CC_17[3]);
  const __m512d C4 = _mm512_set1_pd(CC_17[4]);
  const __m512d C5 = _mm512_set1_pd(CC_17[5]);
  const __m512d C6 = _mm512_set1_pd(CC_17[6]);
  const __m512d C7 = _mm512_set1_pd(CC_17[7]);
  __m512d x0r = _mm512_load_pd(re);
  __m512d s0r = x0r;
  __m512d ar1 = x0r;
  __m512d ar2 = x0r;
  __m512d ar3 = x0r;
  __m512d ar4 = x0r;
  __m512d ar5 = x0r;
  __m512d ar6 = x0r;
  __m512d ar7 = x0r;
  __m512d ar8 = x0r;
  { __m512d p = _mm512_load_pd(re + 1*es), q = _mm512_load_pd(re + 16*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 0, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C0, e, ar1);
    ar2 = _mm512_fmadd_pd(C1, e, ar2);
    ar3 = _mm512_fmadd_pd(C2, e, ar3);
    ar4 = _mm512_fmadd_pd(C3, e, ar4);
    ar5 = _mm512_fmadd_pd(C4, e, ar5);
    ar6 = _mm512_fmadd_pd(C5, e, ar6);
    ar7 = _mm512_fmadd_pd(C6, e, ar7);
    ar8 = _mm512_fmadd_pd(C7, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 2*es), q = _mm512_load_pd(re + 15*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 8, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C1, e, ar1);
    ar2 = _mm512_fmadd_pd(C3, e, ar2);
    ar3 = _mm512_fmadd_pd(C5, e, ar3);
    ar4 = _mm512_fmadd_pd(C7, e, ar4);
    ar5 = _mm512_fmadd_pd(C6, e, ar5);
    ar6 = _mm512_fmadd_pd(C4, e, ar6);
    ar7 = _mm512_fmadd_pd(C2, e, ar7);
    ar8 = _mm512_fmadd_pd(C0, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 3*es), q = _mm512_load_pd(re + 14*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 16, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C2, e, ar1);
    ar2 = _mm512_fmadd_pd(C5, e, ar2);
    ar3 = _mm512_fmadd_pd(C7, e, ar3);
    ar4 = _mm512_fmadd_pd(C4, e, ar4);
    ar5 = _mm512_fmadd_pd(C1, e, ar5);
    ar6 = _mm512_fmadd_pd(C0, e, ar6);
    ar7 = _mm512_fmadd_pd(C3, e, ar7);
    ar8 = _mm512_fmadd_pd(C6, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 4*es), q = _mm512_load_pd(re + 13*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 24, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C3, e, ar1);
    ar2 = _mm512_fmadd_pd(C7, e, ar2);
    ar3 = _mm512_fmadd_pd(C4, e, ar3);
    ar4 = _mm512_fmadd_pd(C0, e, ar4);
    ar5 = _mm512_fmadd_pd(C2, e, ar5);
    ar6 = _mm512_fmadd_pd(C6, e, ar6);
    ar7 = _mm512_fmadd_pd(C5, e, ar7);
    ar8 = _mm512_fmadd_pd(C1, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 5*es), q = _mm512_load_pd(re + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 32, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C4, e, ar1);
    ar2 = _mm512_fmadd_pd(C6, e, ar2);
    ar3 = _mm512_fmadd_pd(C1, e, ar3);
    ar4 = _mm512_fmadd_pd(C2, e, ar4);
    ar5 = _mm512_fmadd_pd(C7, e, ar5);
    ar6 = _mm512_fmadd_pd(C3, e, ar6);
    ar7 = _mm512_fmadd_pd(C0, e, ar7);
    ar8 = _mm512_fmadd_pd(C5, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 6*es), q = _mm512_load_pd(re + 11*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 40, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C5, e, ar1);
    ar2 = _mm512_fmadd_pd(C4, e, ar2);
    ar3 = _mm512_fmadd_pd(C0, e, ar3);
    ar4 = _mm512_fmadd_pd(C6, e, ar4);
    ar5 = _mm512_fmadd_pd(C3, e, ar5);
    ar6 = _mm512_fmadd_pd(C1, e, ar6);
    ar7 = _mm512_fmadd_pd(C7, e, ar7);
    ar8 = _mm512_fmadd_pd(C2, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 7*es), q = _mm512_load_pd(re + 10*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 48, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C6, e, ar1);
    ar2 = _mm512_fmadd_pd(C2, e, ar2);
    ar3 = _mm512_fmadd_pd(C3, e, ar3);
    ar4 = _mm512_fmadd_pd(C5, e, ar4);
    ar5 = _mm512_fmadd_pd(C0, e, ar5);
    ar6 = _mm512_fmadd_pd(C7, e, ar6);
    ar7 = _mm512_fmadd_pd(C1, e, ar7);
    ar8 = _mm512_fmadd_pd(C4, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 8*es), q = _mm512_load_pd(re + 9*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 56, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C7, e, ar1);
    ar2 = _mm512_fmadd_pd(C0, e, ar2);
    ar3 = _mm512_fmadd_pd(C6, e, ar3);
    ar4 = _mm512_fmadd_pd(C1, e, ar4);
    ar5 = _mm512_fmadd_pd(C5, e, ar5);
    ar6 = _mm512_fmadd_pd(C2, e, ar6);
    ar7 = _mm512_fmadd_pd(C4, e, ar7);
    ar8 = _mm512_fmadd_pd(C3, e, ar8);
  }
  _mm512_store_pd(re, s0r);
  _mm512_store_pd(scr + 136, ar1);
  _mm512_store_pd(scr + 144, ar2);
  _mm512_store_pd(scr + 152, ar3);
  _mm512_store_pd(scr + 160, ar4);
  _mm512_store_pd(scr + 168, ar5);
  _mm512_store_pd(scr + 176, ar6);
  _mm512_store_pd(scr + 184, ar7);
  _mm512_store_pd(scr + 192, ar8);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_17[8]);
  const __m512d S1 = _mm512_set1_pd(CC_17[9]);
  const __m512d S2 = _mm512_set1_pd(CC_17[10]);
  const __m512d S3 = _mm512_set1_pd(CC_17[11]);
  const __m512d S4 = _mm512_set1_pd(CC_17[12]);
  const __m512d S5 = _mm512_set1_pd(CC_17[13]);
  const __m512d S6 = _mm512_set1_pd(CC_17[14]);
  const __m512d S7 = _mm512_set1_pd(CC_17[15]);
  __m512d x0i = _mm512_load_pd(im);
  _mm512_store_pd(scr + 128, x0i);
  __m512d s0i = x0i;
  __m512d bi1 = _mm512_setzero_pd();
  __m512d bi2 = _mm512_setzero_pd();
  __m512d bi3 = _mm512_setzero_pd();
  __m512d bi4 = _mm512_setzero_pd();
  __m512d bi5 = _mm512_setzero_pd();
  __m512d bi6 = _mm512_setzero_pd();
  __m512d bi7 = _mm512_setzero_pd();
  __m512d bi8 = _mm512_setzero_pd();
  { __m512d p = _mm512_load_pd(im + 1*es), q = _mm512_load_pd(im + 16*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 64, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S0, o, bi1);
    bi2 = _mm512_fmadd_pd(S1, o, bi2);
    bi3 = _mm512_fmadd_pd(S2, o, bi3);
    bi4 = _mm512_fmadd_pd(S3, o, bi4);
    bi5 = _mm512_fmadd_pd(S4, o, bi5);
    bi6 = _mm512_fmadd_pd(S5, o, bi6);
    bi7 = _mm512_fmadd_pd(S6, o, bi7);
    bi8 = _mm512_fmadd_pd(S7, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 2*es), q = _mm512_load_pd(im + 15*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 72, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S1, o, bi1);
    bi2 = _mm512_fmadd_pd(S3, o, bi2);
    bi3 = _mm512_fmadd_pd(S5, o, bi3);
    bi4 = _mm512_fmadd_pd(S7, o, bi4);
    bi5 = _mm512_fnmadd_pd(S6, o, bi5);
    bi6 = _mm512_fnmadd_pd(S4, o, bi6);
    bi7 = _mm512_fnmadd_pd(S2, o, bi7);
    bi8 = _mm512_fnmadd_pd(S0, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 3*es), q = _mm512_load_pd(im + 14*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 80, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S2, o, bi1);
    bi2 = _mm512_fmadd_pd(S5, o, bi2);
    bi3 = _mm512_fnmadd_pd(S7, o, bi3);
    bi4 = _mm512_fnmadd_pd(S4, o, bi4);
    bi5 = _mm512_fnmadd_pd(S1, o, bi5);
    bi6 = _mm512_fmadd_pd(S0, o, bi6);
    bi7 = _mm512_fmadd_pd(S3, o, bi7);
    bi8 = _mm512_fmadd_pd(S6, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 4*es), q = _mm512_load_pd(im + 13*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 88, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S3, o, bi1);
    bi2 = _mm512_fmadd_pd(S7, o, bi2);
    bi3 = _mm512_fnmadd_pd(S4, o, bi3);
    bi4 = _mm512_fnmadd_pd(S0, o, bi4);
    bi5 = _mm512_fmadd_pd(S2, o, bi5);
    bi6 = _mm512_fmadd_pd(S6, o, bi6);
    bi7 = _mm512_fnmadd_pd(S5, o, bi7);
    bi8 = _mm512_fnmadd_pd(S1, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 5*es), q = _mm512_load_pd(im + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 96, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S4, o, bi1);
    bi2 = _mm512_fnmadd_pd(S6, o, bi2);
    bi3 = _mm512_fnmadd_pd(S1, o, bi3);
    bi4 = _mm512_fmadd_pd(S2, o, bi4);
    bi5 = _mm512_fmadd_pd(S7, o, bi5);
    bi6 = _mm512_fnmadd_pd(S3, o, bi6);
    bi7 = _mm512_fmadd_pd(S0, o, bi7);
    bi8 = _mm512_fmadd_pd(S5, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 6*es), q = _mm512_load_pd(im + 11*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 104, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S5, o, bi1);
    bi2 = _mm512_fnmadd_pd(S4, o, bi2);
    bi3 = _mm512_fmadd_pd(S0, o, bi3);
    bi4 = _mm512_fmadd_pd(S6, o, bi4);
    bi5 = _mm512_fnmadd_pd(S3, o, bi5);
    bi6 = _mm512_fmadd_pd(S1, o, bi6);
    bi7 = _mm512_fmadd_pd(S7, o, bi7);
    bi8 = _mm512_fnmadd_pd(S2, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 7*es), q = _mm512_load_pd(im + 10*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 112, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S6, o, bi1);
    bi2 = _mm512_fnmadd_pd(S2, o, bi2);
    bi3 = _mm512_fmadd_pd(S3, o, bi3);
    bi4 = _mm512_fnmadd_pd(S5, o, bi4);
    bi5 = _mm512_fmadd_pd(S0, o, bi5);
    bi6 = _mm512_fmadd_pd(S7, o, bi6);
    bi7 = _mm512_fnmadd_pd(S1, o, bi7);
    bi8 = _mm512_fmadd_pd(S4, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 8*es), q = _mm512_load_pd(im + 9*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 120, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S7, o, bi1);
    bi2 = _mm512_fnmadd_pd(S0, o, bi2);
    bi3 = _mm512_fmadd_pd(S6, o, bi3);
    bi4 = _mm512_fnmadd_pd(S1, o, bi4);
    bi5 = _mm512_fmadd_pd(S5, o, bi5);
    bi6 = _mm512_fnmadd_pd(S2, o, bi6);
    bi7 = _mm512_fmadd_pd(S4, o, bi7);
    bi8 = _mm512_fnmadd_pd(S3, o, bi8);
  }
  _mm512_store_pd(im, s0i);
  { __m512d a = _mm512_load_pd(scr + 136);
    _mm512_store_pd(re + 1*es, _mm512_add_pd(a, bi1));
    _mm512_store_pd(re + 16*es, _mm512_sub_pd(a, bi1)); }
  { __m512d a = _mm512_load_pd(scr + 144);
    _mm512_store_pd(re + 2*es, _mm512_add_pd(a, bi2));
    _mm512_store_pd(re + 15*es, _mm512_sub_pd(a, bi2)); }
  { __m512d a = _mm512_load_pd(scr + 152);
    _mm512_store_pd(re + 3*es, _mm512_add_pd(a, bi3));
    _mm512_store_pd(re + 14*es, _mm512_sub_pd(a, bi3)); }
  { __m512d a = _mm512_load_pd(scr + 160);
    _mm512_store_pd(re + 4*es, _mm512_add_pd(a, bi4));
    _mm512_store_pd(re + 13*es, _mm512_sub_pd(a, bi4)); }
  { __m512d a = _mm512_load_pd(scr + 168);
    _mm512_store_pd(re + 5*es, _mm512_add_pd(a, bi5));
    _mm512_store_pd(re + 12*es, _mm512_sub_pd(a, bi5)); }
  { __m512d a = _mm512_load_pd(scr + 176);
    _mm512_store_pd(re + 6*es, _mm512_add_pd(a, bi6));
    _mm512_store_pd(re + 11*es, _mm512_sub_pd(a, bi6)); }
  { __m512d a = _mm512_load_pd(scr + 184);
    _mm512_store_pd(re + 7*es, _mm512_add_pd(a, bi7));
    _mm512_store_pd(re + 10*es, _mm512_sub_pd(a, bi7)); }
  { __m512d a = _mm512_load_pd(scr + 192);
    _mm512_store_pd(re + 8*es, _mm512_add_pd(a, bi8));
    _mm512_store_pd(re + 9*es, _mm512_sub_pd(a, bi8)); }
  }
  {
  const __m512d C0 = _mm512_set1_pd(CC_17[0]);
  const __m512d C1 = _mm512_set1_pd(CC_17[1]);
  const __m512d C2 = _mm512_set1_pd(CC_17[2]);
  const __m512d C3 = _mm512_set1_pd(CC_17[3]);
  const __m512d C4 = _mm512_set1_pd(CC_17[4]);
  const __m512d C5 = _mm512_set1_pd(CC_17[5]);
  const __m512d C6 = _mm512_set1_pd(CC_17[6]);
  const __m512d C7 = _mm512_set1_pd(CC_17[7]);
  __m512d x0i = _mm512_load_pd(scr + 128);
  __m512d ai1 = x0i;
  __m512d ai2 = x0i;
  __m512d ai3 = x0i;
  __m512d ai4 = x0i;
  __m512d ai5 = x0i;
  __m512d ai6 = x0i;
  __m512d ai7 = x0i;
  __m512d ai8 = x0i;
  { __m512d e = _mm512_load_pd(scr + 64);
    ai1 = _mm512_fmadd_pd(C0, e, ai1);
    ai2 = _mm512_fmadd_pd(C1, e, ai2);
    ai3 = _mm512_fmadd_pd(C2, e, ai3);
    ai4 = _mm512_fmadd_pd(C3, e, ai4);
    ai5 = _mm512_fmadd_pd(C4, e, ai5);
    ai6 = _mm512_fmadd_pd(C5, e, ai6);
    ai7 = _mm512_fmadd_pd(C6, e, ai7);
    ai8 = _mm512_fmadd_pd(C7, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 72);
    ai1 = _mm512_fmadd_pd(C1, e, ai1);
    ai2 = _mm512_fmadd_pd(C3, e, ai2);
    ai3 = _mm512_fmadd_pd(C5, e, ai3);
    ai4 = _mm512_fmadd_pd(C7, e, ai4);
    ai5 = _mm512_fmadd_pd(C6, e, ai5);
    ai6 = _mm512_fmadd_pd(C4, e, ai6);
    ai7 = _mm512_fmadd_pd(C2, e, ai7);
    ai8 = _mm512_fmadd_pd(C0, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 80);
    ai1 = _mm512_fmadd_pd(C2, e, ai1);
    ai2 = _mm512_fmadd_pd(C5, e, ai2);
    ai3 = _mm512_fmadd_pd(C7, e, ai3);
    ai4 = _mm512_fmadd_pd(C4, e, ai4);
    ai5 = _mm512_fmadd_pd(C1, e, ai5);
    ai6 = _mm512_fmadd_pd(C0, e, ai6);
    ai7 = _mm512_fmadd_pd(C3, e, ai7);
    ai8 = _mm512_fmadd_pd(C6, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 88);
    ai1 = _mm512_fmadd_pd(C3, e, ai1);
    ai2 = _mm512_fmadd_pd(C7, e, ai2);
    ai3 = _mm512_fmadd_pd(C4, e, ai3);
    ai4 = _mm512_fmadd_pd(C0, e, ai4);
    ai5 = _mm512_fmadd_pd(C2, e, ai5);
    ai6 = _mm512_fmadd_pd(C6, e, ai6);
    ai7 = _mm512_fmadd_pd(C5, e, ai7);
    ai8 = _mm512_fmadd_pd(C1, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 96);
    ai1 = _mm512_fmadd_pd(C4, e, ai1);
    ai2 = _mm512_fmadd_pd(C6, e, ai2);
    ai3 = _mm512_fmadd_pd(C1, e, ai3);
    ai4 = _mm512_fmadd_pd(C2, e, ai4);
    ai5 = _mm512_fmadd_pd(C7, e, ai5);
    ai6 = _mm512_fmadd_pd(C3, e, ai6);
    ai7 = _mm512_fmadd_pd(C0, e, ai7);
    ai8 = _mm512_fmadd_pd(C5, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 104);
    ai1 = _mm512_fmadd_pd(C5, e, ai1);
    ai2 = _mm512_fmadd_pd(C4, e, ai2);
    ai3 = _mm512_fmadd_pd(C0, e, ai3);
    ai4 = _mm512_fmadd_pd(C6, e, ai4);
    ai5 = _mm512_fmadd_pd(C3, e, ai5);
    ai6 = _mm512_fmadd_pd(C1, e, ai6);
    ai7 = _mm512_fmadd_pd(C7, e, ai7);
    ai8 = _mm512_fmadd_pd(C2, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 112);
    ai1 = _mm512_fmadd_pd(C6, e, ai1);
    ai2 = _mm512_fmadd_pd(C2, e, ai2);
    ai3 = _mm512_fmadd_pd(C3, e, ai3);
    ai4 = _mm512_fmadd_pd(C5, e, ai4);
    ai5 = _mm512_fmadd_pd(C0, e, ai5);
    ai6 = _mm512_fmadd_pd(C7, e, ai6);
    ai7 = _mm512_fmadd_pd(C1, e, ai7);
    ai8 = _mm512_fmadd_pd(C4, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 120);
    ai1 = _mm512_fmadd_pd(C7, e, ai1);
    ai2 = _mm512_fmadd_pd(C0, e, ai2);
    ai3 = _mm512_fmadd_pd(C6, e, ai3);
    ai4 = _mm512_fmadd_pd(C1, e, ai4);
    ai5 = _mm512_fmadd_pd(C5, e, ai5);
    ai6 = _mm512_fmadd_pd(C2, e, ai6);
    ai7 = _mm512_fmadd_pd(C4, e, ai7);
    ai8 = _mm512_fmadd_pd(C3, e, ai8);
  }
  _mm512_store_pd(scr + 136, ai1);
  _mm512_store_pd(scr + 144, ai2);
  _mm512_store_pd(scr + 152, ai3);
  _mm512_store_pd(scr + 160, ai4);
  _mm512_store_pd(scr + 168, ai5);
  _mm512_store_pd(scr + 176, ai6);
  _mm512_store_pd(scr + 184, ai7);
  _mm512_store_pd(scr + 192, ai8);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_17[8]);
  const __m512d S1 = _mm512_set1_pd(CC_17[9]);
  const __m512d S2 = _mm512_set1_pd(CC_17[10]);
  const __m512d S3 = _mm512_set1_pd(CC_17[11]);
  const __m512d S4 = _mm512_set1_pd(CC_17[12]);
  const __m512d S5 = _mm512_set1_pd(CC_17[13]);
  const __m512d S6 = _mm512_set1_pd(CC_17[14]);
  const __m512d S7 = _mm512_set1_pd(CC_17[15]);
  __m512d br1 = _mm512_setzero_pd();
  __m512d br2 = _mm512_setzero_pd();
  __m512d br3 = _mm512_setzero_pd();
  __m512d br4 = _mm512_setzero_pd();
  __m512d br5 = _mm512_setzero_pd();
  __m512d br6 = _mm512_setzero_pd();
  __m512d br7 = _mm512_setzero_pd();
  __m512d br8 = _mm512_setzero_pd();
  { __m512d o = _mm512_load_pd(scr + 0);
    br1 = _mm512_fmadd_pd(S0, o, br1);
    br2 = _mm512_fmadd_pd(S1, o, br2);
    br3 = _mm512_fmadd_pd(S2, o, br3);
    br4 = _mm512_fmadd_pd(S3, o, br4);
    br5 = _mm512_fmadd_pd(S4, o, br5);
    br6 = _mm512_fmadd_pd(S5, o, br6);
    br7 = _mm512_fmadd_pd(S6, o, br7);
    br8 = _mm512_fmadd_pd(S7, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 8);
    br1 = _mm512_fmadd_pd(S1, o, br1);
    br2 = _mm512_fmadd_pd(S3, o, br2);
    br3 = _mm512_fmadd_pd(S5, o, br3);
    br4 = _mm512_fmadd_pd(S7, o, br4);
    br5 = _mm512_fnmadd_pd(S6, o, br5);
    br6 = _mm512_fnmadd_pd(S4, o, br6);
    br7 = _mm512_fnmadd_pd(S2, o, br7);
    br8 = _mm512_fnmadd_pd(S0, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 16);
    br1 = _mm512_fmadd_pd(S2, o, br1);
    br2 = _mm512_fmadd_pd(S5, o, br2);
    br3 = _mm512_fnmadd_pd(S7, o, br3);
    br4 = _mm512_fnmadd_pd(S4, o, br4);
    br5 = _mm512_fnmadd_pd(S1, o, br5);
    br6 = _mm512_fmadd_pd(S0, o, br6);
    br7 = _mm512_fmadd_pd(S3, o, br7);
    br8 = _mm512_fmadd_pd(S6, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 24);
    br1 = _mm512_fmadd_pd(S3, o, br1);
    br2 = _mm512_fmadd_pd(S7, o, br2);
    br3 = _mm512_fnmadd_pd(S4, o, br3);
    br4 = _mm512_fnmadd_pd(S0, o, br4);
    br5 = _mm512_fmadd_pd(S2, o, br5);
    br6 = _mm512_fmadd_pd(S6, o, br6);
    br7 = _mm512_fnmadd_pd(S5, o, br7);
    br8 = _mm512_fnmadd_pd(S1, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 32);
    br1 = _mm512_fmadd_pd(S4, o, br1);
    br2 = _mm512_fnmadd_pd(S6, o, br2);
    br3 = _mm512_fnmadd_pd(S1, o, br3);
    br4 = _mm512_fmadd_pd(S2, o, br4);
    br5 = _mm512_fmadd_pd(S7, o, br5);
    br6 = _mm512_fnmadd_pd(S3, o, br6);
    br7 = _mm512_fmadd_pd(S0, o, br7);
    br8 = _mm512_fmadd_pd(S5, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 40);
    br1 = _mm512_fmadd_pd(S5, o, br1);
    br2 = _mm512_fnmadd_pd(S4, o, br2);
    br3 = _mm512_fmadd_pd(S0, o, br3);
    br4 = _mm512_fmadd_pd(S6, o, br4);
    br5 = _mm512_fnmadd_pd(S3, o, br5);
    br6 = _mm512_fmadd_pd(S1, o, br6);
    br7 = _mm512_fmadd_pd(S7, o, br7);
    br8 = _mm512_fnmadd_pd(S2, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 48);
    br1 = _mm512_fmadd_pd(S6, o, br1);
    br2 = _mm512_fnmadd_pd(S2, o, br2);
    br3 = _mm512_fmadd_pd(S3, o, br3);
    br4 = _mm512_fnmadd_pd(S5, o, br4);
    br5 = _mm512_fmadd_pd(S0, o, br5);
    br6 = _mm512_fmadd_pd(S7, o, br6);
    br7 = _mm512_fnmadd_pd(S1, o, br7);
    br8 = _mm512_fmadd_pd(S4, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 56);
    br1 = _mm512_fmadd_pd(S7, o, br1);
    br2 = _mm512_fnmadd_pd(S0, o, br2);
    br3 = _mm512_fmadd_pd(S6, o, br3);
    br4 = _mm512_fnmadd_pd(S1, o, br4);
    br5 = _mm512_fmadd_pd(S5, o, br5);
    br6 = _mm512_fnmadd_pd(S2, o, br6);
    br7 = _mm512_fmadd_pd(S4, o, br7);
    br8 = _mm512_fnmadd_pd(S3, o, br8);
  }
  { __m512d a = _mm512_load_pd(scr + 136);
    _mm512_store_pd(im + 1*es, _mm512_sub_pd(a, br1));
    _mm512_store_pd(im + 16*es, _mm512_add_pd(a, br1)); }
  { __m512d a = _mm512_load_pd(scr + 144);
    _mm512_store_pd(im + 2*es, _mm512_sub_pd(a, br2));
    _mm512_store_pd(im + 15*es, _mm512_add_pd(a, br2)); }
  { __m512d a = _mm512_load_pd(scr + 152);
    _mm512_store_pd(im + 3*es, _mm512_sub_pd(a, br3));
    _mm512_store_pd(im + 14*es, _mm512_add_pd(a, br3)); }
  { __m512d a = _mm512_load_pd(scr + 160);
    _mm512_store_pd(im + 4*es, _mm512_sub_pd(a, br4));
    _mm512_store_pd(im + 13*es, _mm512_add_pd(a, br4)); }
  { __m512d a = _mm512_load_pd(scr + 168);
    _mm512_store_pd(im + 5*es, _mm512_sub_pd(a, br5));
    _mm512_store_pd(im + 12*es, _mm512_add_pd(a, br5)); }
  { __m512d a = _mm512_load_pd(scr + 176);
    _mm512_store_pd(im + 6*es, _mm512_sub_pd(a, br6));
    _mm512_store_pd(im + 11*es, _mm512_add_pd(a, br6)); }
  { __m512d a = _mm512_load_pd(scr + 184);
    _mm512_store_pd(im + 7*es, _mm512_sub_pd(a, br7));
    _mm512_store_pd(im + 10*es, _mm512_add_pd(a, br7)); }
  { __m512d a = _mm512_load_pd(scr + 192);
    _mm512_store_pd(im + 8*es, _mm512_sub_pd(a, br8));
    _mm512_store_pd(im + 9*es, _mm512_add_pd(a, br8)); }
  }
}
static const double CC_23[22] ALIGN64 = {0x1.ed037ea3d2dbbp-1,0x1.b57675cf309eep-1,0x1.5d779b07cfef7p-1,0x1.d71b4a0c5a6c8p-2,0x1.a0ad8bd1e2882p-3,-0x1.17855b599f3b9p-4,-0x1.56eaae597c776p-2,-0x1.2742a4a775cfbp-1,-0x1.8d2a07c16d46fp-1,-0x1.d59cb83ef99bcp-1,-0x1.fb3b3035aa6cdp-1,0x1.14459ad2be466p-2,0x1.0a06e851db7cap-1,0x1.763021aaa15dap-1,0x1.c698e42f47b09p-1,0x1.f54a827142577p-1,0x1.fece70dfd3efbp-1,0x1.e270060999288p-1,0x1.a249e0b897ca9p-1,0x1.431df5838f7efp-1,0x1.97f6748e524b2p-2,0x1.16de8a4564f0ap-3};
static void dftp23_v(double* re, double* im, long es){
  double scr[272] __attribute__((aligned(64)));
  {
  const __m512d C0 = _mm512_set1_pd(CC_23[0]);
  const __m512d C1 = _mm512_set1_pd(CC_23[1]);
  const __m512d C2 = _mm512_set1_pd(CC_23[2]);
  const __m512d C3 = _mm512_set1_pd(CC_23[3]);
  const __m512d C4 = _mm512_set1_pd(CC_23[4]);
  const __m512d C5 = _mm512_set1_pd(CC_23[5]);
  const __m512d C6 = _mm512_set1_pd(CC_23[6]);
  const __m512d C7 = _mm512_set1_pd(CC_23[7]);
  const __m512d C8 = _mm512_set1_pd(CC_23[8]);
  const __m512d C9 = _mm512_set1_pd(CC_23[9]);
  const __m512d C10 = _mm512_set1_pd(CC_23[10]);
  __m512d x0r = _mm512_load_pd(re);
  __m512d s0r = x0r;
  __m512d ar1 = x0r;
  __m512d ar2 = x0r;
  __m512d ar3 = x0r;
  __m512d ar4 = x0r;
  __m512d ar5 = x0r;
  __m512d ar6 = x0r;
  __m512d ar7 = x0r;
  __m512d ar8 = x0r;
  __m512d ar9 = x0r;
  __m512d ar10 = x0r;
  __m512d ar11 = x0r;
  { __m512d p = _mm512_load_pd(re + 1*es), q = _mm512_load_pd(re + 22*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 0, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C0, e, ar1);
    ar2 = _mm512_fmadd_pd(C1, e, ar2);
    ar3 = _mm512_fmadd_pd(C2, e, ar3);
    ar4 = _mm512_fmadd_pd(C3, e, ar4);
    ar5 = _mm512_fmadd_pd(C4, e, ar5);
    ar6 = _mm512_fmadd_pd(C5, e, ar6);
    ar7 = _mm512_fmadd_pd(C6, e, ar7);
    ar8 = _mm512_fmadd_pd(C7, e, ar8);
    ar9 = _mm512_fmadd_pd(C8, e, ar9);
    ar10 = _mm512_fmadd_pd(C9, e, ar10);
    ar11 = _mm512_fmadd_pd(C10, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 2*es), q = _mm512_load_pd(re + 21*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 8, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C1, e, ar1);
    ar2 = _mm512_fmadd_pd(C3, e, ar2);
    ar3 = _mm512_fmadd_pd(C5, e, ar3);
    ar4 = _mm512_fmadd_pd(C7, e, ar4);
    ar5 = _mm512_fmadd_pd(C9, e, ar5);
    ar6 = _mm512_fmadd_pd(C10, e, ar6);
    ar7 = _mm512_fmadd_pd(C8, e, ar7);
    ar8 = _mm512_fmadd_pd(C6, e, ar8);
    ar9 = _mm512_fmadd_pd(C4, e, ar9);
    ar10 = _mm512_fmadd_pd(C2, e, ar10);
    ar11 = _mm512_fmadd_pd(C0, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 3*es), q = _mm512_load_pd(re + 20*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 16, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C2, e, ar1);
    ar2 = _mm512_fmadd_pd(C5, e, ar2);
    ar3 = _mm512_fmadd_pd(C8, e, ar3);
    ar4 = _mm512_fmadd_pd(C10, e, ar4);
    ar5 = _mm512_fmadd_pd(C7, e, ar5);
    ar6 = _mm512_fmadd_pd(C4, e, ar6);
    ar7 = _mm512_fmadd_pd(C1, e, ar7);
    ar8 = _mm512_fmadd_pd(C0, e, ar8);
    ar9 = _mm512_fmadd_pd(C3, e, ar9);
    ar10 = _mm512_fmadd_pd(C6, e, ar10);
    ar11 = _mm512_fmadd_pd(C9, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 4*es), q = _mm512_load_pd(re + 19*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 24, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C3, e, ar1);
    ar2 = _mm512_fmadd_pd(C7, e, ar2);
    ar3 = _mm512_fmadd_pd(C10, e, ar3);
    ar4 = _mm512_fmadd_pd(C6, e, ar4);
    ar5 = _mm512_fmadd_pd(C2, e, ar5);
    ar6 = _mm512_fmadd_pd(C0, e, ar6);
    ar7 = _mm512_fmadd_pd(C4, e, ar7);
    ar8 = _mm512_fmadd_pd(C8, e, ar8);
    ar9 = _mm512_fmadd_pd(C9, e, ar9);
    ar10 = _mm512_fmadd_pd(C5, e, ar10);
    ar11 = _mm512_fmadd_pd(C1, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 5*es), q = _mm512_load_pd(re + 18*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 32, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C4, e, ar1);
    ar2 = _mm512_fmadd_pd(C9, e, ar2);
    ar3 = _mm512_fmadd_pd(C7, e, ar3);
    ar4 = _mm512_fmadd_pd(C2, e, ar4);
    ar5 = _mm512_fmadd_pd(C1, e, ar5);
    ar6 = _mm512_fmadd_pd(C6, e, ar6);
    ar7 = _mm512_fmadd_pd(C10, e, ar7);
    ar8 = _mm512_fmadd_pd(C5, e, ar8);
    ar9 = _mm512_fmadd_pd(C0, e, ar9);
    ar10 = _mm512_fmadd_pd(C3, e, ar10);
    ar11 = _mm512_fmadd_pd(C8, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 6*es), q = _mm512_load_pd(re + 17*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 40, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C5, e, ar1);
    ar2 = _mm512_fmadd_pd(C10, e, ar2);
    ar3 = _mm512_fmadd_pd(C4, e, ar3);
    ar4 = _mm512_fmadd_pd(C0, e, ar4);
    ar5 = _mm512_fmadd_pd(C6, e, ar5);
    ar6 = _mm512_fmadd_pd(C9, e, ar6);
    ar7 = _mm512_fmadd_pd(C3, e, ar7);
    ar8 = _mm512_fmadd_pd(C1, e, ar8);
    ar9 = _mm512_fmadd_pd(C7, e, ar9);
    ar10 = _mm512_fmadd_pd(C8, e, ar10);
    ar11 = _mm512_fmadd_pd(C2, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 7*es), q = _mm512_load_pd(re + 16*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 48, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C6, e, ar1);
    ar2 = _mm512_fmadd_pd(C8, e, ar2);
    ar3 = _mm512_fmadd_pd(C1, e, ar3);
    ar4 = _mm512_fmadd_pd(C4, e, ar4);
    ar5 = _mm512_fmadd_pd(C10, e, ar5);
    ar6 = _mm512_fmadd_pd(C3, e, ar6);
    ar7 = _mm512_fmadd_pd(C2, e, ar7);
    ar8 = _mm512_fmadd_pd(C9, e, ar8);
    ar9 = _mm512_fmadd_pd(C5, e, ar9);
    ar10 = _mm512_fmadd_pd(C0, e, ar10);
    ar11 = _mm512_fmadd_pd(C7, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 8*es), q = _mm512_load_pd(re + 15*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 56, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C7, e, ar1);
    ar2 = _mm512_fmadd_pd(C6, e, ar2);
    ar3 = _mm512_fmadd_pd(C0, e, ar3);
    ar4 = _mm512_fmadd_pd(C8, e, ar4);
    ar5 = _mm512_fmadd_pd(C5, e, ar5);
    ar6 = _mm512_fmadd_pd(C1, e, ar6);
    ar7 = _mm512_fmadd_pd(C9, e, ar7);
    ar8 = _mm512_fmadd_pd(C4, e, ar8);
    ar9 = _mm512_fmadd_pd(C2, e, ar9);
    ar10 = _mm512_fmadd_pd(C10, e, ar10);
    ar11 = _mm512_fmadd_pd(C3, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 9*es), q = _mm512_load_pd(re + 14*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 64, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C8, e, ar1);
    ar2 = _mm512_fmadd_pd(C4, e, ar2);
    ar3 = _mm512_fmadd_pd(C3, e, ar3);
    ar4 = _mm512_fmadd_pd(C9, e, ar4);
    ar5 = _mm512_fmadd_pd(C0, e, ar5);
    ar6 = _mm512_fmadd_pd(C7, e, ar6);
    ar7 = _mm512_fmadd_pd(C5, e, ar7);
    ar8 = _mm512_fmadd_pd(C2, e, ar8);
    ar9 = _mm512_fmadd_pd(C10, e, ar9);
    ar10 = _mm512_fmadd_pd(C1, e, ar10);
    ar11 = _mm512_fmadd_pd(C6, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 10*es), q = _mm512_load_pd(re + 13*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 72, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C9, e, ar1);
    ar2 = _mm512_fmadd_pd(C2, e, ar2);
    ar3 = _mm512_fmadd_pd(C6, e, ar3);
    ar4 = _mm512_fmadd_pd(C5, e, ar4);
    ar5 = _mm512_fmadd_pd(C3, e, ar5);
    ar6 = _mm512_fmadd_pd(C8, e, ar6);
    ar7 = _mm512_fmadd_pd(C0, e, ar7);
    ar8 = _mm512_fmadd_pd(C10, e, ar8);
    ar9 = _mm512_fmadd_pd(C1, e, ar9);
    ar10 = _mm512_fmadd_pd(C7, e, ar10);
    ar11 = _mm512_fmadd_pd(C4, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 11*es), q = _mm512_load_pd(re + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 80, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C10, e, ar1);
    ar2 = _mm512_fmadd_pd(C0, e, ar2);
    ar3 = _mm512_fmadd_pd(C9, e, ar3);
    ar4 = _mm512_fmadd_pd(C1, e, ar4);
    ar5 = _mm512_fmadd_pd(C8, e, ar5);
    ar6 = _mm512_fmadd_pd(C2, e, ar6);
    ar7 = _mm512_fmadd_pd(C7, e, ar7);
    ar8 = _mm512_fmadd_pd(C3, e, ar8);
    ar9 = _mm512_fmadd_pd(C6, e, ar9);
    ar10 = _mm512_fmadd_pd(C4, e, ar10);
    ar11 = _mm512_fmadd_pd(C5, e, ar11);
  }
  _mm512_store_pd(re, s0r);
  _mm512_store_pd(scr + 184, ar1);
  _mm512_store_pd(scr + 192, ar2);
  _mm512_store_pd(scr + 200, ar3);
  _mm512_store_pd(scr + 208, ar4);
  _mm512_store_pd(scr + 216, ar5);
  _mm512_store_pd(scr + 224, ar6);
  _mm512_store_pd(scr + 232, ar7);
  _mm512_store_pd(scr + 240, ar8);
  _mm512_store_pd(scr + 248, ar9);
  _mm512_store_pd(scr + 256, ar10);
  _mm512_store_pd(scr + 264, ar11);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_23[11]);
  const __m512d S1 = _mm512_set1_pd(CC_23[12]);
  const __m512d S2 = _mm512_set1_pd(CC_23[13]);
  const __m512d S3 = _mm512_set1_pd(CC_23[14]);
  const __m512d S4 = _mm512_set1_pd(CC_23[15]);
  const __m512d S5 = _mm512_set1_pd(CC_23[16]);
  const __m512d S6 = _mm512_set1_pd(CC_23[17]);
  const __m512d S7 = _mm512_set1_pd(CC_23[18]);
  const __m512d S8 = _mm512_set1_pd(CC_23[19]);
  const __m512d S9 = _mm512_set1_pd(CC_23[20]);
  const __m512d S10 = _mm512_set1_pd(CC_23[21]);
  __m512d x0i = _mm512_load_pd(im);
  _mm512_store_pd(scr + 176, x0i);
  __m512d s0i = x0i;
  __m512d bi1 = _mm512_setzero_pd();
  __m512d bi2 = _mm512_setzero_pd();
  __m512d bi3 = _mm512_setzero_pd();
  __m512d bi4 = _mm512_setzero_pd();
  __m512d bi5 = _mm512_setzero_pd();
  __m512d bi6 = _mm512_setzero_pd();
  __m512d bi7 = _mm512_setzero_pd();
  __m512d bi8 = _mm512_setzero_pd();
  __m512d bi9 = _mm512_setzero_pd();
  __m512d bi10 = _mm512_setzero_pd();
  __m512d bi11 = _mm512_setzero_pd();
  { __m512d p = _mm512_load_pd(im + 1*es), q = _mm512_load_pd(im + 22*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 88, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S0, o, bi1);
    bi2 = _mm512_fmadd_pd(S1, o, bi2);
    bi3 = _mm512_fmadd_pd(S2, o, bi3);
    bi4 = _mm512_fmadd_pd(S3, o, bi4);
    bi5 = _mm512_fmadd_pd(S4, o, bi5);
    bi6 = _mm512_fmadd_pd(S5, o, bi6);
    bi7 = _mm512_fmadd_pd(S6, o, bi7);
    bi8 = _mm512_fmadd_pd(S7, o, bi8);
    bi9 = _mm512_fmadd_pd(S8, o, bi9);
    bi10 = _mm512_fmadd_pd(S9, o, bi10);
    bi11 = _mm512_fmadd_pd(S10, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 2*es), q = _mm512_load_pd(im + 21*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 96, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S1, o, bi1);
    bi2 = _mm512_fmadd_pd(S3, o, bi2);
    bi3 = _mm512_fmadd_pd(S5, o, bi3);
    bi4 = _mm512_fmadd_pd(S7, o, bi4);
    bi5 = _mm512_fmadd_pd(S9, o, bi5);
    bi6 = _mm512_fnmadd_pd(S10, o, bi6);
    bi7 = _mm512_fnmadd_pd(S8, o, bi7);
    bi8 = _mm512_fnmadd_pd(S6, o, bi8);
    bi9 = _mm512_fnmadd_pd(S4, o, bi9);
    bi10 = _mm512_fnmadd_pd(S2, o, bi10);
    bi11 = _mm512_fnmadd_pd(S0, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 3*es), q = _mm512_load_pd(im + 20*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 104, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S2, o, bi1);
    bi2 = _mm512_fmadd_pd(S5, o, bi2);
    bi3 = _mm512_fmadd_pd(S8, o, bi3);
    bi4 = _mm512_fnmadd_pd(S10, o, bi4);
    bi5 = _mm512_fnmadd_pd(S7, o, bi5);
    bi6 = _mm512_fnmadd_pd(S4, o, bi6);
    bi7 = _mm512_fnmadd_pd(S1, o, bi7);
    bi8 = _mm512_fmadd_pd(S0, o, bi8);
    bi9 = _mm512_fmadd_pd(S3, o, bi9);
    bi10 = _mm512_fmadd_pd(S6, o, bi10);
    bi11 = _mm512_fmadd_pd(S9, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 4*es), q = _mm512_load_pd(im + 19*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 112, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S3, o, bi1);
    bi2 = _mm512_fmadd_pd(S7, o, bi2);
    bi3 = _mm512_fnmadd_pd(S10, o, bi3);
    bi4 = _mm512_fnmadd_pd(S6, o, bi4);
    bi5 = _mm512_fnmadd_pd(S2, o, bi5);
    bi6 = _mm512_fmadd_pd(S0, o, bi6);
    bi7 = _mm512_fmadd_pd(S4, o, bi7);
    bi8 = _mm512_fmadd_pd(S8, o, bi8);
    bi9 = _mm512_fnmadd_pd(S9, o, bi9);
    bi10 = _mm512_fnmadd_pd(S5, o, bi10);
    bi11 = _mm512_fnmadd_pd(S1, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 5*es), q = _mm512_load_pd(im + 18*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 120, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S4, o, bi1);
    bi2 = _mm512_fmadd_pd(S9, o, bi2);
    bi3 = _mm512_fnmadd_pd(S7, o, bi3);
    bi4 = _mm512_fnmadd_pd(S2, o, bi4);
    bi5 = _mm512_fmadd_pd(S1, o, bi5);
    bi6 = _mm512_fmadd_pd(S6, o, bi6);
    bi7 = _mm512_fnmadd_pd(S10, o, bi7);
    bi8 = _mm512_fnmadd_pd(S5, o, bi8);
    bi9 = _mm512_fnmadd_pd(S0, o, bi9);
    bi10 = _mm512_fmadd_pd(S3, o, bi10);
    bi11 = _mm512_fmadd_pd(S8, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 6*es), q = _mm512_load_pd(im + 17*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 128, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S5, o, bi1);
    bi2 = _mm512_fnmadd_pd(S10, o, bi2);
    bi3 = _mm512_fnmadd_pd(S4, o, bi3);
    bi4 = _mm512_fmadd_pd(S0, o, bi4);
    bi5 = _mm512_fmadd_pd(S6, o, bi5);
    bi6 = _mm512_fnmadd_pd(S9, o, bi6);
    bi7 = _mm512_fnmadd_pd(S3, o, bi7);
    bi8 = _mm512_fmadd_pd(S1, o, bi8);
    bi9 = _mm512_fmadd_pd(S7, o, bi9);
    bi10 = _mm512_fnmadd_pd(S8, o, bi10);
    bi11 = _mm512_fnmadd_pd(S2, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 7*es), q = _mm512_load_pd(im + 16*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 136, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S6, o, bi1);
    bi2 = _mm512_fnmadd_pd(S8, o, bi2);
    bi3 = _mm512_fnmadd_pd(S1, o, bi3);
    bi4 = _mm512_fmadd_pd(S4, o, bi4);
    bi5 = _mm512_fnmadd_pd(S10, o, bi5);
    bi6 = _mm512_fnmadd_pd(S3, o, bi6);
    bi7 = _mm512_fmadd_pd(S2, o, bi7);
    bi8 = _mm512_fmadd_pd(S9, o, bi8);
    bi9 = _mm512_fnmadd_pd(S5, o, bi9);
    bi10 = _mm512_fmadd_pd(S0, o, bi10);
    bi11 = _mm512_fmadd_pd(S7, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 8*es), q = _mm512_load_pd(im + 15*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 144, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S7, o, bi1);
    bi2 = _mm512_fnmadd_pd(S6, o, bi2);
    bi3 = _mm512_fmadd_pd(S0, o, bi3);
    bi4 = _mm512_fmadd_pd(S8, o, bi4);
    bi5 = _mm512_fnmadd_pd(S5, o, bi5);
    bi6 = _mm512_fmadd_pd(S1, o, bi6);
    bi7 = _mm512_fmadd_pd(S9, o, bi7);
    bi8 = _mm512_fnmadd_pd(S4, o, bi8);
    bi9 = _mm512_fmadd_pd(S2, o, bi9);
    bi10 = _mm512_fmadd_pd(S10, o, bi10);
    bi11 = _mm512_fnmadd_pd(S3, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 9*es), q = _mm512_load_pd(im + 14*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 152, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S8, o, bi1);
    bi2 = _mm512_fnmadd_pd(S4, o, bi2);
    bi3 = _mm512_fmadd_pd(S3, o, bi3);
    bi4 = _mm512_fnmadd_pd(S9, o, bi4);
    bi5 = _mm512_fnmadd_pd(S0, o, bi5);
    bi6 = _mm512_fmadd_pd(S7, o, bi6);
    bi7 = _mm512_fnmadd_pd(S5, o, bi7);
    bi8 = _mm512_fmadd_pd(S2, o, bi8);
    bi9 = _mm512_fnmadd_pd(S10, o, bi9);
    bi10 = _mm512_fnmadd_pd(S1, o, bi10);
    bi11 = _mm512_fmadd_pd(S6, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 10*es), q = _mm512_load_pd(im + 13*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 160, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S9, o, bi1);
    bi2 = _mm512_fnmadd_pd(S2, o, bi2);
    bi3 = _mm512_fmadd_pd(S6, o, bi3);
    bi4 = _mm512_fnmadd_pd(S5, o, bi4);
    bi5 = _mm512_fmadd_pd(S3, o, bi5);
    bi6 = _mm512_fnmadd_pd(S8, o, bi6);
    bi7 = _mm512_fmadd_pd(S0, o, bi7);
    bi8 = _mm512_fmadd_pd(S10, o, bi8);
    bi9 = _mm512_fnmadd_pd(S1, o, bi9);
    bi10 = _mm512_fmadd_pd(S7, o, bi10);
    bi11 = _mm512_fnmadd_pd(S4, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 11*es), q = _mm512_load_pd(im + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 168, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S10, o, bi1);
    bi2 = _mm512_fnmadd_pd(S0, o, bi2);
    bi3 = _mm512_fmadd_pd(S9, o, bi3);
    bi4 = _mm512_fnmadd_pd(S1, o, bi4);
    bi5 = _mm512_fmadd_pd(S8, o, bi5);
    bi6 = _mm512_fnmadd_pd(S2, o, bi6);
    bi7 = _mm512_fmadd_pd(S7, o, bi7);
    bi8 = _mm512_fnmadd_pd(S3, o, bi8);
    bi9 = _mm512_fmadd_pd(S6, o, bi9);
    bi10 = _mm512_fnmadd_pd(S4, o, bi10);
    bi11 = _mm512_fmadd_pd(S5, o, bi11);
  }
  _mm512_store_pd(im, s0i);
  { __m512d a = _mm512_load_pd(scr + 184);
    _mm512_store_pd(re + 1*es, _mm512_add_pd(a, bi1));
    _mm512_store_pd(re + 22*es, _mm512_sub_pd(a, bi1)); }
  { __m512d a = _mm512_load_pd(scr + 192);
    _mm512_store_pd(re + 2*es, _mm512_add_pd(a, bi2));
    _mm512_store_pd(re + 21*es, _mm512_sub_pd(a, bi2)); }
  { __m512d a = _mm512_load_pd(scr + 200);
    _mm512_store_pd(re + 3*es, _mm512_add_pd(a, bi3));
    _mm512_store_pd(re + 20*es, _mm512_sub_pd(a, bi3)); }
  { __m512d a = _mm512_load_pd(scr + 208);
    _mm512_store_pd(re + 4*es, _mm512_add_pd(a, bi4));
    _mm512_store_pd(re + 19*es, _mm512_sub_pd(a, bi4)); }
  { __m512d a = _mm512_load_pd(scr + 216);
    _mm512_store_pd(re + 5*es, _mm512_add_pd(a, bi5));
    _mm512_store_pd(re + 18*es, _mm512_sub_pd(a, bi5)); }
  { __m512d a = _mm512_load_pd(scr + 224);
    _mm512_store_pd(re + 6*es, _mm512_add_pd(a, bi6));
    _mm512_store_pd(re + 17*es, _mm512_sub_pd(a, bi6)); }
  { __m512d a = _mm512_load_pd(scr + 232);
    _mm512_store_pd(re + 7*es, _mm512_add_pd(a, bi7));
    _mm512_store_pd(re + 16*es, _mm512_sub_pd(a, bi7)); }
  { __m512d a = _mm512_load_pd(scr + 240);
    _mm512_store_pd(re + 8*es, _mm512_add_pd(a, bi8));
    _mm512_store_pd(re + 15*es, _mm512_sub_pd(a, bi8)); }
  { __m512d a = _mm512_load_pd(scr + 248);
    _mm512_store_pd(re + 9*es, _mm512_add_pd(a, bi9));
    _mm512_store_pd(re + 14*es, _mm512_sub_pd(a, bi9)); }
  { __m512d a = _mm512_load_pd(scr + 256);
    _mm512_store_pd(re + 10*es, _mm512_add_pd(a, bi10));
    _mm512_store_pd(re + 13*es, _mm512_sub_pd(a, bi10)); }
  { __m512d a = _mm512_load_pd(scr + 264);
    _mm512_store_pd(re + 11*es, _mm512_add_pd(a, bi11));
    _mm512_store_pd(re + 12*es, _mm512_sub_pd(a, bi11)); }
  }
  {
  const __m512d C0 = _mm512_set1_pd(CC_23[0]);
  const __m512d C1 = _mm512_set1_pd(CC_23[1]);
  const __m512d C2 = _mm512_set1_pd(CC_23[2]);
  const __m512d C3 = _mm512_set1_pd(CC_23[3]);
  const __m512d C4 = _mm512_set1_pd(CC_23[4]);
  const __m512d C5 = _mm512_set1_pd(CC_23[5]);
  const __m512d C6 = _mm512_set1_pd(CC_23[6]);
  const __m512d C7 = _mm512_set1_pd(CC_23[7]);
  const __m512d C8 = _mm512_set1_pd(CC_23[8]);
  const __m512d C9 = _mm512_set1_pd(CC_23[9]);
  const __m512d C10 = _mm512_set1_pd(CC_23[10]);
  __m512d x0i = _mm512_load_pd(scr + 176);
  __m512d ai1 = x0i;
  __m512d ai2 = x0i;
  __m512d ai3 = x0i;
  __m512d ai4 = x0i;
  __m512d ai5 = x0i;
  __m512d ai6 = x0i;
  __m512d ai7 = x0i;
  __m512d ai8 = x0i;
  __m512d ai9 = x0i;
  __m512d ai10 = x0i;
  __m512d ai11 = x0i;
  { __m512d e = _mm512_load_pd(scr + 88);
    ai1 = _mm512_fmadd_pd(C0, e, ai1);
    ai2 = _mm512_fmadd_pd(C1, e, ai2);
    ai3 = _mm512_fmadd_pd(C2, e, ai3);
    ai4 = _mm512_fmadd_pd(C3, e, ai4);
    ai5 = _mm512_fmadd_pd(C4, e, ai5);
    ai6 = _mm512_fmadd_pd(C5, e, ai6);
    ai7 = _mm512_fmadd_pd(C6, e, ai7);
    ai8 = _mm512_fmadd_pd(C7, e, ai8);
    ai9 = _mm512_fmadd_pd(C8, e, ai9);
    ai10 = _mm512_fmadd_pd(C9, e, ai10);
    ai11 = _mm512_fmadd_pd(C10, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 96);
    ai1 = _mm512_fmadd_pd(C1, e, ai1);
    ai2 = _mm512_fmadd_pd(C3, e, ai2);
    ai3 = _mm512_fmadd_pd(C5, e, ai3);
    ai4 = _mm512_fmadd_pd(C7, e, ai4);
    ai5 = _mm512_fmadd_pd(C9, e, ai5);
    ai6 = _mm512_fmadd_pd(C10, e, ai6);
    ai7 = _mm512_fmadd_pd(C8, e, ai7);
    ai8 = _mm512_fmadd_pd(C6, e, ai8);
    ai9 = _mm512_fmadd_pd(C4, e, ai9);
    ai10 = _mm512_fmadd_pd(C2, e, ai10);
    ai11 = _mm512_fmadd_pd(C0, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 104);
    ai1 = _mm512_fmadd_pd(C2, e, ai1);
    ai2 = _mm512_fmadd_pd(C5, e, ai2);
    ai3 = _mm512_fmadd_pd(C8, e, ai3);
    ai4 = _mm512_fmadd_pd(C10, e, ai4);
    ai5 = _mm512_fmadd_pd(C7, e, ai5);
    ai6 = _mm512_fmadd_pd(C4, e, ai6);
    ai7 = _mm512_fmadd_pd(C1, e, ai7);
    ai8 = _mm512_fmadd_pd(C0, e, ai8);
    ai9 = _mm512_fmadd_pd(C3, e, ai9);
    ai10 = _mm512_fmadd_pd(C6, e, ai10);
    ai11 = _mm512_fmadd_pd(C9, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 112);
    ai1 = _mm512_fmadd_pd(C3, e, ai1);
    ai2 = _mm512_fmadd_pd(C7, e, ai2);
    ai3 = _mm512_fmadd_pd(C10, e, ai3);
    ai4 = _mm512_fmadd_pd(C6, e, ai4);
    ai5 = _mm512_fmadd_pd(C2, e, ai5);
    ai6 = _mm512_fmadd_pd(C0, e, ai6);
    ai7 = _mm512_fmadd_pd(C4, e, ai7);
    ai8 = _mm512_fmadd_pd(C8, e, ai8);
    ai9 = _mm512_fmadd_pd(C9, e, ai9);
    ai10 = _mm512_fmadd_pd(C5, e, ai10);
    ai11 = _mm512_fmadd_pd(C1, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 120);
    ai1 = _mm512_fmadd_pd(C4, e, ai1);
    ai2 = _mm512_fmadd_pd(C9, e, ai2);
    ai3 = _mm512_fmadd_pd(C7, e, ai3);
    ai4 = _mm512_fmadd_pd(C2, e, ai4);
    ai5 = _mm512_fmadd_pd(C1, e, ai5);
    ai6 = _mm512_fmadd_pd(C6, e, ai6);
    ai7 = _mm512_fmadd_pd(C10, e, ai7);
    ai8 = _mm512_fmadd_pd(C5, e, ai8);
    ai9 = _mm512_fmadd_pd(C0, e, ai9);
    ai10 = _mm512_fmadd_pd(C3, e, ai10);
    ai11 = _mm512_fmadd_pd(C8, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 128);
    ai1 = _mm512_fmadd_pd(C5, e, ai1);
    ai2 = _mm512_fmadd_pd(C10, e, ai2);
    ai3 = _mm512_fmadd_pd(C4, e, ai3);
    ai4 = _mm512_fmadd_pd(C0, e, ai4);
    ai5 = _mm512_fmadd_pd(C6, e, ai5);
    ai6 = _mm512_fmadd_pd(C9, e, ai6);
    ai7 = _mm512_fmadd_pd(C3, e, ai7);
    ai8 = _mm512_fmadd_pd(C1, e, ai8);
    ai9 = _mm512_fmadd_pd(C7, e, ai9);
    ai10 = _mm512_fmadd_pd(C8, e, ai10);
    ai11 = _mm512_fmadd_pd(C2, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 136);
    ai1 = _mm512_fmadd_pd(C6, e, ai1);
    ai2 = _mm512_fmadd_pd(C8, e, ai2);
    ai3 = _mm512_fmadd_pd(C1, e, ai3);
    ai4 = _mm512_fmadd_pd(C4, e, ai4);
    ai5 = _mm512_fmadd_pd(C10, e, ai5);
    ai6 = _mm512_fmadd_pd(C3, e, ai6);
    ai7 = _mm512_fmadd_pd(C2, e, ai7);
    ai8 = _mm512_fmadd_pd(C9, e, ai8);
    ai9 = _mm512_fmadd_pd(C5, e, ai9);
    ai10 = _mm512_fmadd_pd(C0, e, ai10);
    ai11 = _mm512_fmadd_pd(C7, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 144);
    ai1 = _mm512_fmadd_pd(C7, e, ai1);
    ai2 = _mm512_fmadd_pd(C6, e, ai2);
    ai3 = _mm512_fmadd_pd(C0, e, ai3);
    ai4 = _mm512_fmadd_pd(C8, e, ai4);
    ai5 = _mm512_fmadd_pd(C5, e, ai5);
    ai6 = _mm512_fmadd_pd(C1, e, ai6);
    ai7 = _mm512_fmadd_pd(C9, e, ai7);
    ai8 = _mm512_fmadd_pd(C4, e, ai8);
    ai9 = _mm512_fmadd_pd(C2, e, ai9);
    ai10 = _mm512_fmadd_pd(C10, e, ai10);
    ai11 = _mm512_fmadd_pd(C3, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 152);
    ai1 = _mm512_fmadd_pd(C8, e, ai1);
    ai2 = _mm512_fmadd_pd(C4, e, ai2);
    ai3 = _mm512_fmadd_pd(C3, e, ai3);
    ai4 = _mm512_fmadd_pd(C9, e, ai4);
    ai5 = _mm512_fmadd_pd(C0, e, ai5);
    ai6 = _mm512_fmadd_pd(C7, e, ai6);
    ai7 = _mm512_fmadd_pd(C5, e, ai7);
    ai8 = _mm512_fmadd_pd(C2, e, ai8);
    ai9 = _mm512_fmadd_pd(C10, e, ai9);
    ai10 = _mm512_fmadd_pd(C1, e, ai10);
    ai11 = _mm512_fmadd_pd(C6, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 160);
    ai1 = _mm512_fmadd_pd(C9, e, ai1);
    ai2 = _mm512_fmadd_pd(C2, e, ai2);
    ai3 = _mm512_fmadd_pd(C6, e, ai3);
    ai4 = _mm512_fmadd_pd(C5, e, ai4);
    ai5 = _mm512_fmadd_pd(C3, e, ai5);
    ai6 = _mm512_fmadd_pd(C8, e, ai6);
    ai7 = _mm512_fmadd_pd(C0, e, ai7);
    ai8 = _mm512_fmadd_pd(C10, e, ai8);
    ai9 = _mm512_fmadd_pd(C1, e, ai9);
    ai10 = _mm512_fmadd_pd(C7, e, ai10);
    ai11 = _mm512_fmadd_pd(C4, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 168);
    ai1 = _mm512_fmadd_pd(C10, e, ai1);
    ai2 = _mm512_fmadd_pd(C0, e, ai2);
    ai3 = _mm512_fmadd_pd(C9, e, ai3);
    ai4 = _mm512_fmadd_pd(C1, e, ai4);
    ai5 = _mm512_fmadd_pd(C8, e, ai5);
    ai6 = _mm512_fmadd_pd(C2, e, ai6);
    ai7 = _mm512_fmadd_pd(C7, e, ai7);
    ai8 = _mm512_fmadd_pd(C3, e, ai8);
    ai9 = _mm512_fmadd_pd(C6, e, ai9);
    ai10 = _mm512_fmadd_pd(C4, e, ai10);
    ai11 = _mm512_fmadd_pd(C5, e, ai11);
  }
  _mm512_store_pd(scr + 184, ai1);
  _mm512_store_pd(scr + 192, ai2);
  _mm512_store_pd(scr + 200, ai3);
  _mm512_store_pd(scr + 208, ai4);
  _mm512_store_pd(scr + 216, ai5);
  _mm512_store_pd(scr + 224, ai6);
  _mm512_store_pd(scr + 232, ai7);
  _mm512_store_pd(scr + 240, ai8);
  _mm512_store_pd(scr + 248, ai9);
  _mm512_store_pd(scr + 256, ai10);
  _mm512_store_pd(scr + 264, ai11);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_23[11]);
  const __m512d S1 = _mm512_set1_pd(CC_23[12]);
  const __m512d S2 = _mm512_set1_pd(CC_23[13]);
  const __m512d S3 = _mm512_set1_pd(CC_23[14]);
  const __m512d S4 = _mm512_set1_pd(CC_23[15]);
  const __m512d S5 = _mm512_set1_pd(CC_23[16]);
  const __m512d S6 = _mm512_set1_pd(CC_23[17]);
  const __m512d S7 = _mm512_set1_pd(CC_23[18]);
  const __m512d S8 = _mm512_set1_pd(CC_23[19]);
  const __m512d S9 = _mm512_set1_pd(CC_23[20]);
  const __m512d S10 = _mm512_set1_pd(CC_23[21]);
  __m512d br1 = _mm512_setzero_pd();
  __m512d br2 = _mm512_setzero_pd();
  __m512d br3 = _mm512_setzero_pd();
  __m512d br4 = _mm512_setzero_pd();
  __m512d br5 = _mm512_setzero_pd();
  __m512d br6 = _mm512_setzero_pd();
  __m512d br7 = _mm512_setzero_pd();
  __m512d br8 = _mm512_setzero_pd();
  __m512d br9 = _mm512_setzero_pd();
  __m512d br10 = _mm512_setzero_pd();
  __m512d br11 = _mm512_setzero_pd();
  { __m512d o = _mm512_load_pd(scr + 0);
    br1 = _mm512_fmadd_pd(S0, o, br1);
    br2 = _mm512_fmadd_pd(S1, o, br2);
    br3 = _mm512_fmadd_pd(S2, o, br3);
    br4 = _mm512_fmadd_pd(S3, o, br4);
    br5 = _mm512_fmadd_pd(S4, o, br5);
    br6 = _mm512_fmadd_pd(S5, o, br6);
    br7 = _mm512_fmadd_pd(S6, o, br7);
    br8 = _mm512_fmadd_pd(S7, o, br8);
    br9 = _mm512_fmadd_pd(S8, o, br9);
    br10 = _mm512_fmadd_pd(S9, o, br10);
    br11 = _mm512_fmadd_pd(S10, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 8);
    br1 = _mm512_fmadd_pd(S1, o, br1);
    br2 = _mm512_fmadd_pd(S3, o, br2);
    br3 = _mm512_fmadd_pd(S5, o, br3);
    br4 = _mm512_fmadd_pd(S7, o, br4);
    br5 = _mm512_fmadd_pd(S9, o, br5);
    br6 = _mm512_fnmadd_pd(S10, o, br6);
    br7 = _mm512_fnmadd_pd(S8, o, br7);
    br8 = _mm512_fnmadd_pd(S6, o, br8);
    br9 = _mm512_fnmadd_pd(S4, o, br9);
    br10 = _mm512_fnmadd_pd(S2, o, br10);
    br11 = _mm512_fnmadd_pd(S0, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 16);
    br1 = _mm512_fmadd_pd(S2, o, br1);
    br2 = _mm512_fmadd_pd(S5, o, br2);
    br3 = _mm512_fmadd_pd(S8, o, br3);
    br4 = _mm512_fnmadd_pd(S10, o, br4);
    br5 = _mm512_fnmadd_pd(S7, o, br5);
    br6 = _mm512_fnmadd_pd(S4, o, br6);
    br7 = _mm512_fnmadd_pd(S1, o, br7);
    br8 = _mm512_fmadd_pd(S0, o, br8);
    br9 = _mm512_fmadd_pd(S3, o, br9);
    br10 = _mm512_fmadd_pd(S6, o, br10);
    br11 = _mm512_fmadd_pd(S9, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 24);
    br1 = _mm512_fmadd_pd(S3, o, br1);
    br2 = _mm512_fmadd_pd(S7, o, br2);
    br3 = _mm512_fnmadd_pd(S10, o, br3);
    br4 = _mm512_fnmadd_pd(S6, o, br4);
    br5 = _mm512_fnmadd_pd(S2, o, br5);
    br6 = _mm512_fmadd_pd(S0, o, br6);
    br7 = _mm512_fmadd_pd(S4, o, br7);
    br8 = _mm512_fmadd_pd(S8, o, br8);
    br9 = _mm512_fnmadd_pd(S9, o, br9);
    br10 = _mm512_fnmadd_pd(S5, o, br10);
    br11 = _mm512_fnmadd_pd(S1, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 32);
    br1 = _mm512_fmadd_pd(S4, o, br1);
    br2 = _mm512_fmadd_pd(S9, o, br2);
    br3 = _mm512_fnmadd_pd(S7, o, br3);
    br4 = _mm512_fnmadd_pd(S2, o, br4);
    br5 = _mm512_fmadd_pd(S1, o, br5);
    br6 = _mm512_fmadd_pd(S6, o, br6);
    br7 = _mm512_fnmadd_pd(S10, o, br7);
    br8 = _mm512_fnmadd_pd(S5, o, br8);
    br9 = _mm512_fnmadd_pd(S0, o, br9);
    br10 = _mm512_fmadd_pd(S3, o, br10);
    br11 = _mm512_fmadd_pd(S8, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 40);
    br1 = _mm512_fmadd_pd(S5, o, br1);
    br2 = _mm512_fnmadd_pd(S10, o, br2);
    br3 = _mm512_fnmadd_pd(S4, o, br3);
    br4 = _mm512_fmadd_pd(S0, o, br4);
    br5 = _mm512_fmadd_pd(S6, o, br5);
    br6 = _mm512_fnmadd_pd(S9, o, br6);
    br7 = _mm512_fnmadd_pd(S3, o, br7);
    br8 = _mm512_fmadd_pd(S1, o, br8);
    br9 = _mm512_fmadd_pd(S7, o, br9);
    br10 = _mm512_fnmadd_pd(S8, o, br10);
    br11 = _mm512_fnmadd_pd(S2, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 48);
    br1 = _mm512_fmadd_pd(S6, o, br1);
    br2 = _mm512_fnmadd_pd(S8, o, br2);
    br3 = _mm512_fnmadd_pd(S1, o, br3);
    br4 = _mm512_fmadd_pd(S4, o, br4);
    br5 = _mm512_fnmadd_pd(S10, o, br5);
    br6 = _mm512_fnmadd_pd(S3, o, br6);
    br7 = _mm512_fmadd_pd(S2, o, br7);
    br8 = _mm512_fmadd_pd(S9, o, br8);
    br9 = _mm512_fnmadd_pd(S5, o, br9);
    br10 = _mm512_fmadd_pd(S0, o, br10);
    br11 = _mm512_fmadd_pd(S7, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 56);
    br1 = _mm512_fmadd_pd(S7, o, br1);
    br2 = _mm512_fnmadd_pd(S6, o, br2);
    br3 = _mm512_fmadd_pd(S0, o, br3);
    br4 = _mm512_fmadd_pd(S8, o, br4);
    br5 = _mm512_fnmadd_pd(S5, o, br5);
    br6 = _mm512_fmadd_pd(S1, o, br6);
    br7 = _mm512_fmadd_pd(S9, o, br7);
    br8 = _mm512_fnmadd_pd(S4, o, br8);
    br9 = _mm512_fmadd_pd(S2, o, br9);
    br10 = _mm512_fmadd_pd(S10, o, br10);
    br11 = _mm512_fnmadd_pd(S3, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 64);
    br1 = _mm512_fmadd_pd(S8, o, br1);
    br2 = _mm512_fnmadd_pd(S4, o, br2);
    br3 = _mm512_fmadd_pd(S3, o, br3);
    br4 = _mm512_fnmadd_pd(S9, o, br4);
    br5 = _mm512_fnmadd_pd(S0, o, br5);
    br6 = _mm512_fmadd_pd(S7, o, br6);
    br7 = _mm512_fnmadd_pd(S5, o, br7);
    br8 = _mm512_fmadd_pd(S2, o, br8);
    br9 = _mm512_fnmadd_pd(S10, o, br9);
    br10 = _mm512_fnmadd_pd(S1, o, br10);
    br11 = _mm512_fmadd_pd(S6, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 72);
    br1 = _mm512_fmadd_pd(S9, o, br1);
    br2 = _mm512_fnmadd_pd(S2, o, br2);
    br3 = _mm512_fmadd_pd(S6, o, br3);
    br4 = _mm512_fnmadd_pd(S5, o, br4);
    br5 = _mm512_fmadd_pd(S3, o, br5);
    br6 = _mm512_fnmadd_pd(S8, o, br6);
    br7 = _mm512_fmadd_pd(S0, o, br7);
    br8 = _mm512_fmadd_pd(S10, o, br8);
    br9 = _mm512_fnmadd_pd(S1, o, br9);
    br10 = _mm512_fmadd_pd(S7, o, br10);
    br11 = _mm512_fnmadd_pd(S4, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 80);
    br1 = _mm512_fmadd_pd(S10, o, br1);
    br2 = _mm512_fnmadd_pd(S0, o, br2);
    br3 = _mm512_fmadd_pd(S9, o, br3);
    br4 = _mm512_fnmadd_pd(S1, o, br4);
    br5 = _mm512_fmadd_pd(S8, o, br5);
    br6 = _mm512_fnmadd_pd(S2, o, br6);
    br7 = _mm512_fmadd_pd(S7, o, br7);
    br8 = _mm512_fnmadd_pd(S3, o, br8);
    br9 = _mm512_fmadd_pd(S6, o, br9);
    br10 = _mm512_fnmadd_pd(S4, o, br10);
    br11 = _mm512_fmadd_pd(S5, o, br11);
  }
  { __m512d a = _mm512_load_pd(scr + 184);
    _mm512_store_pd(im + 1*es, _mm512_sub_pd(a, br1));
    _mm512_store_pd(im + 22*es, _mm512_add_pd(a, br1)); }
  { __m512d a = _mm512_load_pd(scr + 192);
    _mm512_store_pd(im + 2*es, _mm512_sub_pd(a, br2));
    _mm512_store_pd(im + 21*es, _mm512_add_pd(a, br2)); }
  { __m512d a = _mm512_load_pd(scr + 200);
    _mm512_store_pd(im + 3*es, _mm512_sub_pd(a, br3));
    _mm512_store_pd(im + 20*es, _mm512_add_pd(a, br3)); }
  { __m512d a = _mm512_load_pd(scr + 208);
    _mm512_store_pd(im + 4*es, _mm512_sub_pd(a, br4));
    _mm512_store_pd(im + 19*es, _mm512_add_pd(a, br4)); }
  { __m512d a = _mm512_load_pd(scr + 216);
    _mm512_store_pd(im + 5*es, _mm512_sub_pd(a, br5));
    _mm512_store_pd(im + 18*es, _mm512_add_pd(a, br5)); }
  { __m512d a = _mm512_load_pd(scr + 224);
    _mm512_store_pd(im + 6*es, _mm512_sub_pd(a, br6));
    _mm512_store_pd(im + 17*es, _mm512_add_pd(a, br6)); }
  { __m512d a = _mm512_load_pd(scr + 232);
    _mm512_store_pd(im + 7*es, _mm512_sub_pd(a, br7));
    _mm512_store_pd(im + 16*es, _mm512_add_pd(a, br7)); }
  { __m512d a = _mm512_load_pd(scr + 240);
    _mm512_store_pd(im + 8*es, _mm512_sub_pd(a, br8));
    _mm512_store_pd(im + 15*es, _mm512_add_pd(a, br8)); }
  { __m512d a = _mm512_load_pd(scr + 248);
    _mm512_store_pd(im + 9*es, _mm512_sub_pd(a, br9));
    _mm512_store_pd(im + 14*es, _mm512_add_pd(a, br9)); }
  { __m512d a = _mm512_load_pd(scr + 256);
    _mm512_store_pd(im + 10*es, _mm512_sub_pd(a, br10));
    _mm512_store_pd(im + 13*es, _mm512_add_pd(a, br10)); }
  { __m512d a = _mm512_load_pd(scr + 264);
    _mm512_store_pd(im + 11*es, _mm512_sub_pd(a, br11));
    _mm512_store_pd(im + 12*es, _mm512_add_pd(a, br11)); }
  }
}