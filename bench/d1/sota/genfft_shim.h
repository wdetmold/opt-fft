/* Minimal standalone shim for genfft-emitted _notw codelets, replacing FFTW's kernel
 * headers. The codelet is a pure straight-line DAG over type R with strided split-array
 * access -- compiling it with R=double gives the scalar reference form; compiling with
 * R = an 8-wide gcc vector type turns the SAME generated code into an SoA batch-lane
 * kernel (gcc broadcasts the scalar twiddle constants over vector operands). */
#ifndef GENFFT_SHIM_H
#define GENFFT_SHIM_H
typedef long INT;
typedef INT stride;
#define WS(s, i) ((s) * (i))
#define MAKE_VOLATILE_STRIDE(n, s) 0
#ifndef R
#define R double
#endif
typedef R E;
#define DK(name, val) static const double name = (val)
#define K(x) ((double)x)
#define FMA(a, b, c)  (((a) * (b)) + (c))
#define FMS(a, b, c)  (((a) * (b)) - (c))
#define FNMA(a, b, c) (-(((a) * (b)) + (c)))
#define FNMS(a, b, c) ((c) - ((a) * (b)))
#endif
