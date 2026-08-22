/* L8_warpradix8 -- NOT YET IMPLEMENTED. You are the first implementer for this entry.
 *
 * L = 8 on one A100. Read ../PANEL_BRIEF.md for the contract, the machine and the
 * rules, ../fft3d_gpu_api.h for the ABI, and
 * ../../../docs/literature/09-gpu-small-batched-a100.md for the GPU corpus and its
 * per-geometry opening strategies.
 *
 * Strategy: ONE VOLUME PER WARP, radix-8 in registers with shuffles.
 * 512 points across 32 lanes is 16 complex doubles per lane. A radix-8 butterfly maps onto
 * a warp naturally: 8 lanes hold one line, shuffles do the transpose between axes, and no
 * shared memory is touched at all. Compare against L8_blockfused honestly -- shuffle
 * latency against shared-memory bandwidth is exactly the tradeoff this pair exists to
 * settle.
 *
 * Replace this file entirely. Make fft3d_gpu_supports() accept L = 8, keep a
 * strategy record in ../strategies/L8_warpradix8.md, and verify with
 * ./tryout.sh L8_warpradix8 8 <batch>.
 */
#include <cuda_runtime.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

struct fft3d_gpu_plan { int L; int batch; };

extern "C" const char *fft3d_gpu_name(void) { return "L8_warpradix8"; }
extern "C" const char *fft3d_gpu_description(void) { return "L8_warpradix8: stub, not yet implemented"; }

/* Returns 0 so the harness skips this entry until it is written. */
extern "C" int fft3d_gpu_supports(int L) { (void)L; return 0; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{ (void)L; (void)batch; return NULL; }
extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{ (void)p; (void)in; (void)out; }
extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p) { (void)p; }
