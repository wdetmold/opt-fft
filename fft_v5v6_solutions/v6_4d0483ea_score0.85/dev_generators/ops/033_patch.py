src = open('implementation.c').read()
# split DEF_ENGINE into A and B parts
old = src[src.index("#define DEF_ENGINE(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)"):src.index("DEF_ENGINE(6, 6, 8, 8, 8, 72, fft6_line)")]
new = old.replace("#define DEF_ENGINE(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)", "#define DEF_SWEEPA(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)")
# cut at sweepB definition
ia = new.index("  SWEEPVIS void sweepB_##LN(void)")
parta = new[:ia].rstrip().rstrip('\\\\').rstrip() + "\n\n"
partb = "#define DEF_SWEEPB(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)                               \\\n" + new[ia:]
# fix trailing: parta currently ends with the closing brace of sweepA -- check
src = src.replace(old, parta + partb)
src = src.replace("DEF_ENGINE(6, 6, 8, 8, 8, 72, fft6_line)", "DEF_SWEEPB(6, 6, 8, 8, 8, 72, fft6_line)")
src = src.replace("DEF_ENGINE(8, 8, 8, 8, 8, 72, fft8_line)", "DEF_SWEEPB(8, 8, 8, 8, 8, 72, fft8_line)")
for args in ("13, 13, 16, 16, 16, 264, fft13_line", "17, 17, 24, 24, 24, 584, fft17_line",
             "23, 23, 24, 24, 24, 584, fft23_line", "36, 36, 40, 40, 40, 1608, fft36_line",
             "45, 45, 48, 48, 48, 2312, fft45_line", "64, 64, 64, 72, 64, 4616, fft64_line"):
    src = src.replace(f"DEF_ENGINE({args})", f"DEF_SWEEPA({args})\nDEF_SWEEPB({args})")

# custom register-resident sweepA for 6 and 8
custom = '''
// register-resident fused Z+Y sweep for tiny sizes (slice fits in registers)
SWEEPVIS void sweepA_6(void) {
  DEFCONSTS
  for (long x = 0; x < 6; x++) {
    double *br = Xre + x * 72, *bi = Xim + x * 72;
    v8 y0r = VL(br + 0 * 8), y0i = VL(bi + 0 * 8);
    v8 y1r = VL(br + 1 * 8), y1i = VL(bi + 1 * 8);
    v8 y2r = VL(br + 2 * 8), y2i = VL(bi + 2 * 8);
    v8 y3r = VL(br + 3 * 8), y3i = VL(bi + 3 * 8);
    v8 y4r = VL(br + 4 * 8), y4i = VL(bi + 4 * 8);
    v8 y5r = VL(br + 5 * 8), y5i = VL(bi + 5 * 8);
    FFT6_CORE(y0, y1, y2, y3, y4, y5);       // pass Y (across rows)
    v8 y6r = vzero, y7r = vzero, y6i = vzero, y7i = vzero;
    TR8(y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r);
    TR8(y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i);
    FFT6_CORE(y0, y1, y2, y3, y4, y5);       // pass Z (across columns)
    TR8(y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r);
    TR8(y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i);
    VS(br + 0 * 8, y0r); VS(bi + 0 * 8, y0i);
    VS(br + 1 * 8, y1r); VS(bi + 1 * 8, y1i);
    VS(br + 2 * 8, y2r); VS(bi + 2 * 8, y2i);
    VS(br + 3 * 8, y3r); VS(bi + 3 * 8, y3i);
    VS(br + 4 * 8, y4r); VS(bi + 4 * 8, y4i);
    VS(br + 5 * 8, y5r); VS(bi + 5 * 8, y5i);
  }
}
SWEEPVIS void sweepA_8(void) {
  DEFCONSTS
  for (long x = 0; x < 8; x++) {
    double *br = Xre + x * 72, *bi = Xim + x * 72;
    v8 y0r = VL(br + 0 * 8), y0i = VL(bi + 0 * 8);
    v8 y1r = VL(br + 1 * 8), y1i = VL(bi + 1 * 8);
    v8 y2r = VL(br + 2 * 8), y2i = VL(bi + 2 * 8);
    v8 y3r = VL(br + 3 * 8), y3i = VL(bi + 3 * 8);
    v8 y4r = VL(br + 4 * 8), y4i = VL(bi + 4 * 8);
    v8 y5r = VL(br + 5 * 8), y5i = VL(bi + 5 * 8);
    v8 y6r = VL(br + 6 * 8), y6i = VL(bi + 6 * 8);
    v8 y7r = VL(br + 7 * 8), y7i = VL(bi + 7 * 8);
    FFT8_CORE(y0, y1, y2, y3, y4, y5, y6, y7);
    TR8(y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r);
    TR8(y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i);
    FFT8_CORE(y0, y1, y2, y3, y4, y5, y6, y7);
    TR8(y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r);
    TR8(y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i);
    VS(br + 0 * 8, y0r); VS(bi + 0 * 8, y0i);
    VS(br + 1 * 8, y1r); VS(bi + 1 * 8, y1i);
    VS(br + 2 * 8, y2r); VS(bi + 2 * 8, y2i);
    VS(br + 3 * 8, y3r); VS(bi + 3 * 8, y3i);
    VS(br + 4 * 8, y4r); VS(bi + 4 * 8, y4i);
    VS(br + 5 * 8, y5r); VS(bi + 5 * 8, y5i);
    VS(br + 6 * 8, y6r); VS(bi + 6 * 8, y6i);
    VS(br + 7 * 8, y7r); VS(bi + 7 * 8, y7i);
  }
}
'''
src = src.replace("DEF_SWEEPB(6, 6, 8, 8, 8, 72, fft6_line)",
                  custom + "\nDEF_SWEEPB(6, 6, 8, 8, 8, 72, fft6_line)")
open('implementation.c','w').write(src)
print("ok")
