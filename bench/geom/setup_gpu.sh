#!/bin/bash
# Phase 3: hand the baton to the single-A100 competition.
#
# Installed as the multicore harness's after_series.sh, so it runs when that series ends.
# The GPU harness itself (driver.cu, fft3d_gpu_api.h, the cuFFT baseline, the library-free
# floor, sweep/submit/tryout, the brief) was built and smoke-tested on the A100 in advance --
# this script only seeds the panel, arms the series, and flips the phase pointer. Nothing
# here has to work at 3am for the first time.
set -u
cd "$(dirname "$(readlink -f "$0")")"
HERE=$(pwd)
BENCH=$(readlink -f "$HERE/..")
[ -d "$BENCH/gpu" ] || BENCH=$(readlink -f "$HERE/../..")   # when run from bench/mt
GPU=$BENCH/gpu
LOG() { printf '[%s] setup-gpu: %s\n' "$(date '+%F %T')" "$*"; }

if [ ! -d "$GPU" ]; then LOG "no GPU harness at $GPU -- cannot start phase 3"; exit 1; fi

# ---- sweep cases --------------------------------------------------------------------
# Batches are much larger than the CPU phases': 108 SMs need work to fill. B=1 is kept
# because on a GPU it is the launch-overhead case, which is a real and separate question.
cat > "$GPU/cases.txt" <<'CASES'
# <L>:<batch>.  Batch points are defined by WORKING SET rather than by round numbers, which
# is how the GPU literature benchmarks these (VkFFT reports 500 MB - 1 GB) and makes the
# geometries comparable to each other:
#   B=1     the launch-overhead / latency case, scored separately and read as such
#   B_L2    in+out ~= 32 MiB, so the whole problem lives in the A100's 40 MiB L2
#   B_HBM   one buffer ~= 1 GiB, so it cannot: this is the primary score
6:1     6:4854     6:310608
8:1     8:2048     8:131072
13:1    13:477     13:30549
17:1    17:213     17:13660
23:1    23:86      23:5515
36:1    36:22      36:1438
45:1    45:11      45:736
64:1    64:4       64:256
CASES
LOG "wrote cases.txt ($(grep -vE '^\s*(#|$)' "$GPU/cases.txt" | wc -w) cases)"

# ---- seed the panel -----------------------------------------------------------------
mkdir -p "$GPU/impl_1"

new_stub() {  # new_stub <name> <L> <brief...>
  local name=$1 L=$2; shift 2
  local path="$GPU/impl_1/$name.cu"
  if [ -f "$path" ]; then LOG "  $name already exists, leaving it alone"; return 0; fi
  {
    echo "/* $name -- NOT YET IMPLEMENTED. You are the first implementer for this entry."
    echo " *"
    echo " * L = $L on one A100. Read ../PANEL_BRIEF.md for the contract, the machine and the"
    echo " * rules, ../fft3d_gpu_api.h for the ABI, and"
    echo " * ../../../docs/literature/09-gpu-small-batched-a100.md for the GPU corpus and its"
    echo " * per-geometry opening strategies."
    echo " *"
    printf ' * %s\n' "$@"
    echo " *"
    echo " * Replace this file entirely. Make fft3d_gpu_supports() accept L = $L, keep a"
    echo " * strategy record in ../strategies/$name.md, and verify with"
    echo " * ./tryout.sh $name $L <batch>."
    echo " */"
    echo '#include <cuda_runtime.h>'
    echo '#include <stdlib.h>'
    echo ''
    echo '#include "../fft3d_gpu_api.h"'
    echo ''
    echo 'struct fft3d_gpu_plan { int L; int batch; };'
    echo ''
    echo "extern \"C\" const char *fft3d_gpu_name(void) { return \"$name\"; }"
    echo "extern \"C\" const char *fft3d_gpu_description(void) { return \"$name: stub, not yet implemented\"; }"
    echo ''
    echo '/* Returns 0 so the harness skips this entry until it is written. */'
    echo 'extern "C" int fft3d_gpu_supports(int L) { (void)L; return 0; }'
    echo ''
    echo 'extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)'
    echo '{ (void)L; (void)batch; return NULL; }'
    echo 'extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)'
    echo '{ (void)p; (void)in; (void)out; }'
    echo 'extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p) { (void)p; }'
  } > "$path"
  LOG "  created $name.cu (L=$L)"
}

new_stub L6_warpvolume 6 \
  "Strategy: ONE VOLUME PER WARP, entirely in registers." \
  "216 complex doubles is 3.4 KB: across 32 lanes that is 6.75 elements per lane, so a whole" \
  "volume fits in registers with room to spare. Do all three axes with warp shuffles and" \
  "never touch shared memory or global memory in between -- one coalesced read in, one" \
  "coalesced write out. 6 = 2*3, so the line transform is a radix-2 stage over a 3-point" \
  "Winograd module (see ../../geom/strategies/L6_unrolled.md for the arithmetic, already" \
  "worked out and measured on the CPU)."

new_stub L6_batchcoalesced 6 \
  "Strategy: BATCH-MAJOR, PERFECTLY COALESCED." \
  "The contrarian entry: ignore the volume structure and make memory access ideal. Thread t" \
  "handles volume t, so consecutive threads touch consecutive volumes and every load is a" \
  "fully coalesced 128-byte transaction across the batch. Each thread then runs a scalar" \
  "6^3 transform on its own volume. Register pressure is the risk (216 complex doubles per" \
  "thread is impossible, so you will need shared memory or a strip-mined loop over the" \
  "volume) -- work out the tradeoff and report it. This is the GPU analogue of the CPU" \
  "phase's batch-major SIMD entry, which won at L=8."

new_stub L8_blockfused 8 \
  "Strategy: SEVERAL VOLUMES PER BLOCK IN SHARED MEMORY, all three axes fused." \
  "8 KB per volume means ~20 volumes fit in one SM's 164 KB shared memory. Stage a tile of" \
  "volumes, do x, y and z without returning to global memory, write once. All 8-point" \
  "twiddles are 8th roots of unity, so the only irrational constant is 1/sqrt(2) -- see" \
  "../../geom/strategies/L8_radix8.md, which reached the published minimum operation count." \
  "Mind the shared-memory bank conflicts on the strided axes: 8 complex doubles is exactly" \
  "128 bytes, which is a degenerate stride against 32 banks of 4 bytes. Padding is the fix" \
  "and measuring it is the point."

new_stub L8_warpradix8 8 \
  "Strategy: ONE VOLUME PER WARP, radix-8 in registers with shuffles." \
  "512 points across 32 lanes is 16 complex doubles per lane. A radix-8 butterfly maps onto" \
  "a warp naturally: 8 lanes hold one line, shuffles do the transpose between axes, and no" \
  "shared memory is touched at all. Compare against L8_blockfused honestly -- shuffle" \
  "latency against shared-memory bandwidth is exactly the tradeoff this pair exists to" \
  "settle."

new_stub L17_dmma 17 \
  "Strategy: DENSE MATRIX VIA FP64 TENSOR CORES (mma.sync / DMMA)." \
  "This is the most interesting entry in the whole GPU phase. On the CPU, a dense" \
  "conjugate-symmetric 17x17 matvec BEAT Rader and every library, by 4.97x, because a matvec" \
  "has perfect data flow. The A100 has FP64 tensor cores at 19.5 TFLOP/s -- twice the" \
  "vanilla FP64 rate -- reachable only by a matrix formulation, never by a butterfly. A" \
  "17-point DFT along an axis is a 17x17 complex matrix times a 17xN panel, which is four" \
  "real matrix products (or three with Karatsuba). Read section 09 of the corpus on the" \
  "fragment shapes and what fraction of peak is achievable, and read" \
  "../../geom/strategies/L17_matrixsimd.md for the conjugate-pair folding that halves the" \
  "work. If DMMA turns out not to pay, say so with the measurement -- a well-evidenced" \
  "negative result on this question is genuinely valuable."

new_stub L17_raderfused 17 \
  "Strategy: RADER IN SHARED MEMORY, whole volume resident." \
  "A 17^3 volume is 78.6 KB, so two fit in one SM's shared memory: stage a volume, do all" \
  "three axes, write once. 17 is prime and 17-1 = 16 is a pure power of two, which makes" \
  "Rader's 16-point cyclic convolution ideal -- two radix-2 FFT16s plus 16 pointwise" \
  "multiplies, with the b-transform precomputed in create(). See" \
  "../../geom/strategies/L17_rader.md. On the CPU this lost narrowly to the dense matvec at" \
  "B=1 but WON at large batch; find out whether that inversion survives on a GPU."

new_stub L36_sharedtiled 36 \
  "Strategy: PENCIL TILES IN SHARED MEMORY." \
  "746 KB per volume does not fit shared memory, so the volume must be processed in tiles." \
  "Stage a (36 x t x t) pencil, transform all three axes' worth of work that the tile" \
  "supports, and choose t so the tile fits comfortably in shared memory with enough blocks" \
  "resident to hide latency. 36 = 4*9, so the line transform is radix-4 then radix-9 (or" \
  "6x6) -- ../../geom/strategies/L36_mixedradix.md is the CPU leader and has the" \
  "decomposition worked out."

new_stub L36_globalpass 36 \
  "Strategy: THREE BANDWIDTH-OPTIMAL GLOBAL PASSES." \
  "The contrarian entry, and on current evidence maybe the right one: accept three passes" \
  "over global memory and make each one perfectly coalesced with vectorized double2 loads," \
  "rather than fighting for locality. At these sizes the transform is bandwidth-bound, so" \
  "the question is what fraction of the A100's ~1.55 TB/s you can reach -- the driver" \
  "reports effective GB/s for exactly this reason. Measure it, compare with the minimum" \
  "traffic (read in, write out), and report how far off the roofline you are."

new_stub L13_dmma 13 \
  "Strategy: DENSE MATRIX, tensor cores if they pay." \
  "The smaller prime: 169 complex MACs per line against 17's 289. A 13^3 volume is 35 KB so" \
  "four fit in shared memory. Same question as L17_dmma at a size where the arithmetic" \
  "disadvantage of the dense form is smaller -- coordinate with that entry's record rather" \
  "than rediscovering the fragment shapes."

new_stub L23_rader 23 \
  "Strategy: RADER where p-1 is not a power of two." \
  "23-1 = 22 = 2*11, so the cyclic convolution needs an 11-point sub-transform. A 23^3" \
  "volume is 190 KB and does NOT fit shared memory, unlike 13 and 17 -- so this entry has to" \
  "solve the prime problem and the tiling problem at once. See" \
  "../../geom/strategies/L23_rader.md if the CPU wave-2 panel got there first."

new_stub L45_pfa 45 \
  "Strategy: GOOD-THOMAS 9x5, no radix-2 anywhere." \
  "45 = 9*5, coprime, so the prime-factor map removes inter-stage twiddles entirely. On a" \
  "GPU the awkwardness is that neither 9 nor 5 divides 32 lanes, so warp-level mappings are" \
  "unnatural and the index permutation costs shuffles or shared-memory round trips. 1.42 MB" \
  "per volume means tiling is mandatory."

new_stub L64_radix8 64 \
  "Strategy: TWO RADIX-8 PASSES, and the power-of-two stride problem." \
  "64 = 8^2, so each axis is two radix-8 passes, and the L=8 kernels already exist to build" \
  "from. The new difficulty is that a 64^3 volume is 4.19 MB and every stride is a power of" \
  "two: in shared memory that is the worst case for 32 banks, and in global memory it is the" \
  "worst case for partition camping. Padding in shared memory and swizzling in global are" \
  "the standard answers; measure both."

ln -sfn impl_1 "$GPU/impl"
LOG "seeded impl_1 with $(ls "$GPU/impl_1"/*.cu | wc -l) entries"

# ---- config, state, phase pointer ---------------------------------------------------
mkdir -p "$GPU/results"
cat > "$GPU/results/.rounds_config" <<'CFG'
FFT_ROUND_PREFIX=gpu_r
FFT_FIRST_ROUND=1
FFT_PARTITION=a100l
FFT_TIME=120
FFT_SRC_EXT=cu
FFT_DEV_CMD=./tryout.sh
FFT_API_HEADER=fft3d_gpu_api.h
CFG
printf '1 6\n' > "$GPU/results/.rounds_state"
LOG "armed gpu_r1 .. gpu_r6 on partition a100l"

# Claim the 8-GPU node NOW, at handover, rather than holding it idle through the CPU
# phases. Best effort: if the partition is busy, the phase still works -- submit.sh queues a
# whole-node job per round instead, and implementers fall back to the login node's A100.
if [ -x "$GPU/reserve.sh" ]; then
  LOG "claiming a GPU node for the phase"
  ( cd "$GPU" && ./reserve.sh --hours "${FFT_GPU_HOURS:-12}" 2>&1 | sed 's/^/    /' ) || \
    LOG "could not claim a node now; rounds will queue whole-node jobs instead"
fi

printf 'gpu\n' > "$BENCH/PHASE"
LOG "bench/PHASE -> gpu; cron will start gpu_r1 on its next tick"

# Last phase in the chain: nothing further to arm.
rm -f "$HERE/after_series.sh"
LOG "chain complete -- no further phase armed after the GPU competition"
