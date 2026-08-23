src = open('implementation.c').read()
src = src.replace("// ---------------------------------------------------------------- conversions",
"""#ifdef EXPOSE_SWEEPS
void micro_line(int which, long n) {   // run codelet on L1-resident scratch
  for (long i = 0; i < n; i++) {
    switch (which) {
      case 6:  fft6_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 8:  fft8_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 13: fft13_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 17: fft17_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 23: fft23_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 36: fft36_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 45: fft45_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 64: fft64_line(SZA, SZB, 8, 0, 0, 0, 0); break;
    }
  }
}
#endif
// ---------------------------------------------------------------- conversions""")
open('implementation.c','w').write(src)
