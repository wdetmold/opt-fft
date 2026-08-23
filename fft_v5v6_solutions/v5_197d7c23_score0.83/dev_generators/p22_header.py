# B39 (log 02:46:25): final documentation header
head = open('impl_head.c').read()
# Replace the top comment block with a thorough header
old_top = head[:head.index('#include <stdint.h>')]
new_top = '''// ============================================================================
// Iterated batched 3D complex-to-complex FFTs for L in {6,8,13,17,23,36,45,64}
// with the elementwise map  x <- z/(1+|z|),  z = FFT3(x) + c,  iterated m_L
// times per volume. Forward unnormalized DFT, C-order volumes.
//
// ALL transform arithmetic in this file is hand-written (no FFT libraries):
//   - 6  = Good-Thomas PFA(2,3) of hand-coded DFT2/DFT3 kernels
//   - 8  = hand-coded split DFT8 (DIT radix-2 with trivial twiddles)
//   - 13,17,23 = direct symmetric (cos/sin folded) prime DFT, FMA-dominated,
//                4-way blocked over output pairs (k, L-k)
//   - 36 = Good-Thomas PFA(4,9), DFT9 = 3x3 Cooley-Tukey of DFT3s
//   - 45 = Good-Thomas PFA(5,9)
//   - 64 = 8x8 Cooley-Tukey of two hand-coded DFT8 layers + 49 twiddles
//
// Strategy: vertical SIMD (8-wide AVX-512 doubles), split re/im storage in
// 128-byte blocks of 8 complex values. Two batching schemes:
//   BL ("batch-lane"): SIMD lanes = 8 independent volumes. Zero shuffle
//       overhead, zero padding waste; used for small L when the batch B
//       is large enough. Working set = one 8-volume group, L3-resident.
//   PV ("per-volume"): SIMD lanes = 8 consecutive z samples (z chunks padded
//       to 8); the z-axis pass uses in-register 8x8 transposes (strips of 8
//       rows, with an overlapped final strip when L%8 != 0 so no y padding
//       is needed). Used for large L (36,45,64) and for small batches.
// Each step does:  sweep1 = x-axis line FFTs;  sweep2 = per-x-plane y-axis
// and z-axis FFTs followed by the fused "+c and map" pass (plane-sequential,
// cache-friendly). Order of axis transforms is irrelevant mathematically.
//
// The map z/(1+|z|) uses rsqrt14 + 2 Newton steps for sqrt and rcp14 + 2
// Newton steps for the reciprocal (both good to ~1e-16 relative, verified
// against the long-double reference; one-step block error ~9e-16 rel L2).
// Twiddle/cos/sin tables are computed once at import in 80-bit long double
// with exact mod-L argument reduction.
//
// Single-threaded. No OpenMP, no pthreads, no external compute libraries.
// ============================================================================
'''
head = new_top + head[head.index('#include <stdint.h>'):]
open('impl_head.c','w').write(head)
print("header written")
