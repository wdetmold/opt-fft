#!/bin/bash
# Wave 2: widen the geometry pool once the first series of rounds is finished.
#
# Installed as after_series.sh, so run_rounds.sh executes it the moment the current series
# completes and then cron picks up the re-armed state file. Runs exactly once: it removes
# itself as the hook when it succeeds.
#
# The four new sizes are not a variety pack -- each settles a specific question the existing
# four cannot, and each is named in docs/LITERATURE.md's open questions:
#
#   L=13  prime, and FFTW's own generator switches to Rader at exactly rader_min = 13.
#         The dense conjugate-symmetric matrix beat Rader at L=17 (B=1). Does that hold at
#         the smaller prime, where 289 -> 169 multiply-adds changes the balance?
#   L=23  prime with p-1 = 22 = 2*11. L=17 won by 4.98x partly because p-1 = 16 is a pure
#         power of two, making Rader's convolution ideal. At 23 the convolution needs an
#         11-point sub-transform, so this separates "prime is easy for us" from "17 was a
#         lucky prime".
#   L=45  45 = 9*5: coprime factors and NO factor of two anywhere. Every kernel the panel
#         has built leans on radix-2 somewhere; this is the case where it cannot.
#   L=64  2^6, a 4.19 MB volume that escapes L2 entirely, and power-of-two strides
#         throughout -- the cache-set conflict and padding pathology that section 04/05 of
#         the corpus argue about and nobody has measured here.
set -u
cd "$(dirname "$(readlink -f "$0")")"
GEOM=$(pwd)
LOG() { printf '[%s] expand: %s\n' "$(date '+%F %T')" "$*"; }

ROUNDS_NEXT=6      # rounds to run for the new wave

LOG "widening the geometry pool"

# ---------------------------------------------------------------- 1. stub entries
#
# run_rounds.sh builds its roster from impl/*.c, and the implementer prompt tells each agent
# to read its own file first -- so the strategy brief for a new geometry lives in the stub's
# header comment. fft3d_supports() returns 0 until an agent writes the real thing, which
# makes the harness skip it cleanly (exit 3) instead of failing the round.

IMPL=$(readlink -f "$GEOM/impl")     # the current round's directory
[ -d "$IMPL" ] || { LOG "no impl directory at $IMPL -- aborting"; exit 1; }

new_stub() {  # new_stub <name> <L> <brief...>
  local name=$1 L=$2; shift 2
  local path="$IMPL/$name.c"
  if [ -f "$path" ]; then LOG "  $name already exists, leaving it alone"; return 0; fi
  {
    echo "/* $name -- NOT YET IMPLEMENTED. You are the first implementer for this geometry."
    echo " *"
    echo " * L = $L. Read ../PANEL_BRIEF.md for the contract, the machine and the rules, and"
    echo " * ../../docs/LITERATURE.md for the strategy table and the open questions."
    echo " *"
    printf ' * %s\n' "$@"
    echo " *"
    echo " * Replace this file entirely: implement the ABI in ../fft3d_api.h, make"
    echo " * fft3d_supports() accept L = $L, and keep a strategy record in"
    echo " * ../strategies/$name.md. Verify with ./tryout.sh --on wallaby $name $L <batch>."
    echo " */"
    echo '#include <stdlib.h>'
    echo ''
    echo '#include "../fft3d_api.h"'
    echo ''
    echo 'struct fft3d_plan { int L; int batch; };'
    echo ''
    echo "const char *fft3d_name(void) { return \"$name\"; }"
    echo "const char *fft3d_description(void) { return \"$name: stub, not yet implemented\"; }"
    echo ''
    echo '/* Returns 0 so the harness skips this entry until it is written. */'
    echo 'int fft3d_supports(int L) { (void)L; return 0; }'
    echo ''
    echo 'fft3d_plan *fft3d_create(int L, int batch) { (void)L; (void)batch; return NULL; }'
    echo 'void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)'
    echo '{ (void)p; (void)in; (void)out; }'
    echo 'void fft3d_destroy(fft3d_plan *p) { (void)p; }'
  } > "$path"
  LOG "  created $name.c (L=$L)"
}

new_stub L13_direct 13 \
  "Strategy: DENSE CONJUGATE-SYMMETRIC MATRIX per axis." \
  "At L=17 this approach (L17_matrixsimd) beat Rader at B=1 -- 16.4 us against 18.5 -- because" \
  "a matrix-vector product has perfect data flow: no permutations, no index tables, no" \
  "cross-lane shuffles, pure FMA on contiguous data. 13 is smaller, so the arithmetic" \
  "disadvantage shrinks (169 versus 289 complex MACs per line) while the data-flow advantage" \
  "stays. Fold the j <-> 13-j conjugate pairs first, as the L=17 entry does; read" \
  "../strategies/L17_matrixsimd.md before you start, and say what you took from it."

new_stub L13_rader 13 \
  "Strategy: RADER with a 12-point cyclic convolution." \
  "12 = 4*3 is smooth, so the convolution decomposes cleanly, and 13 is exactly where FFTW's" \
  "own generator switches to Rader (rader_min = 13 in genfft/magic.ml). That makes this the" \
  "one size where a library ships a hand-tuned prime codelet, so the comparison is unusually" \
  "meaningful. Read ../strategies/L17_rader.md first: it already has a working Rader for a" \
  "prime, and its convolution length was a power of two -- yours is not."

new_stub L23_rader 23 \
  "Strategy: RADER where p-1 is NOT a power of two." \
  "23-1 = 22 = 2*11, so the cyclic convolution needs an 11-point sub-transform rather than" \
  "the clean radix-2 chain that made L=17 so good. This entry exists to answer a specific" \
  "question: was the panel's 4.98x win at L=17 about primes being easy for us, or about 17" \
  "being a lucky prime? Report the operation count and compare it honestly with L=17's." \
  "Nested Rader (11 -> 10 = 2*5) or a direct 11-point Winograd module are both options."

new_stub L23_matrixsimd 23 \
  "Strategy: DENSE CONJUGATE-SYMMETRIC MATRIX per axis, at a bigger prime." \
  "529 complex MACs per line before folding conjugate pairs -- nearly twice L=17's 289 -- so" \
  "this is where the dense approach should finally lose to Rader if it is ever going to." \
  "Finding the crossover is the point: build it, measure it against L23_rader on the same" \
  "data, and record where the two cross as a function of batch size."

new_stub L45_pfa 45 \
  "Strategy: GOOD-THOMAS PFA (9 x 5) with NO radix-2 anywhere." \
  "45 = 9*5, coprime, so the prime-factor map eliminates inter-stage twiddles entirely. Note" \
  "what makes this different from every kernel the panel has written so far: there is no" \
  "factor of two, so radix-2 butterflies and the pairing tricks that come with them are" \
  "unavailable, and a 512-bit vector holds 4 complex doubles which divides neither 9 nor 5." \
  "How you fill the SIMD lanes is the whole problem -- batch-major is the obvious answer." \
  "Read ../strategies/L36_pfa.md: PFA lost badly at L=36 (225 us against 118) before that" \
  "entry adopted the winner's structure, and its record says why."

new_stub L45_mixedradix 45 \
  "Strategy: COOLEY-TUKEY radix 3 x 3 x 5 (or 5 x 9), fully unrolled." \
  "The counterpart to L45_pfa: same size, twiddles kept, no permutation. At L=36 the plain" \
  "mixed-radix row-column kernel beat PFA decisively, so the expectation is that it wins here" \
  "too -- but at L=36 both factors were even and here neither is. Read" \
  "../strategies/L36_mixedradix.md, which is the current L=36 leader, and say what carries" \
  "over and what does not."

new_stub L64_radix8 64 \
  "Strategy: TWO RADIX-8 PASSES per axis (8 x 8), fully unrolled." \
  "64 = 8^2, and the panel's L=8 kernels already reach 40 GF/s with a radix-8 codelet whose" \
  "only irrational constant is 1/sqrt(2). Compose two of those passes per axis. The new" \
  "difficulty is not arithmetic: a 64^3 volume is 4.19 MB, so it does not fit L2 (1 MB on the" \
  "scoring node) and every pass re-reads it from L3 or memory. Read" \
  "../strategies/L8_radix8.md and ../strategies/L8_fusedaxes.md first."

new_stub L64_blocked 64 \
  "Strategy: CACHE-BLOCKED, WITH EXPLICIT PADDING against set conflicts." \
  "This is the geometry where the corpus's padding argument must finally bite. At L=64 every" \
  "stride is a power of two, so the z-stride, the y-stride and the volume-to-volume stride" \
  "are all exact multiples of the cache-set stride, and section 04 of the corpus predicts a" \
  "single-set catastrophe (Bailey's worst case) while section 05 argues it is harmless when" \
  "the volume fits in cache -- at 4.19 MB it does not. Measure it: build padded and unpadded" \
  "variants, pad so strides are an ODD number of cache lines, and report the difference with" \
  "perf counters if you can get them. Whatever you find is new information."

# ---------------------------------------------------------------- 2. sweep cases

cat > "$GEOM/cases.txt" <<'CASES'
# Sweep cases: <L>:<batch>.  B=1 is the non-batched problem, scored separately; the batched
# points are chosen per geometry to sit (a) inside L2 and (b) well past L3.
# bytes per volume = 16 * L^3
6:1     6:64      6:4096    6:32768
8:1     8:64      8:2048    8:16384
17:1    17:8      17:256    17:2048
36:1    36:4      36:32     36:256
# ---- wave 2 ----
13:1    13:16     13:512
23:1    23:4      23:128
45:1    45:2      45:16
64:1    64:2      64:8
CASES
LOG "wrote cases.txt ($(grep -vcE '^\s*(#|$)' "$GEOM/cases.txt") case lines, $(grep -vE '^\s*(#|$)' "$GEOM/cases.txt" | wc -w) cases)"

# ---------------------------------------------------------------- 3. round config
#
# The sweep is now 28 cases instead of 16. The expensive dense-matrix floor is skipped
# automatically on large cases, which claws back most of the added time, but leave the job
# a little more headroom. devel's hard cap is one hour; if a round ever times out, switch
# FFT_PARTITION to prod (a one-day limit) here -- no code change needed.
cat > "$GEOM/results/.rounds_config" <<'CFG'
FFT_PARTITION=devel
FFT_TIME=59
CFG
LOG "wrote results/.rounds_config (partition devel, 59 min)"

# ---------------------------------------------------------------- 4. re-arm the series

STATE=$GEOM/results/.rounds_state
if [ -f "$STATE" ]; then
  read -r _ LAST < "$STATE"
else
  LAST=7
fi
NEW_NEXT=$((LAST + 1))
NEW_LAST=$((NEW_NEXT + ROUNDS_NEXT - 1))
printf '%s %s\n' "$NEW_NEXT" "$NEW_LAST" > "$STATE"
LOG "re-armed: panel_r$NEW_NEXT .. panel_r$NEW_LAST ($ROUNDS_NEXT rounds over 8 geometries)"

# ---------------------------------------------------------------- 5. announce, and disarm

cat > "$GEOM/results/WAVE2.md" <<NOTE
# Wave 2: the geometry pool widened after panel_r$LAST

Added: **L = 13, 23, 45, 64**, two competing entries each, alongside the original
L = 6, 8, 17, 36 (which continue to be revised).

| L | why it is here | what it settles |
|---|---|---|
| 13 | prime, and FFTW's generator switches to Rader at exactly rader_min = 13 | whether the dense conjugate-symmetric kernel that beat Rader at 17 still wins where the arithmetic gap is smaller |
| 23 | prime with p-1 = 2*11, not a power of two | whether the 4.98x win at L=17 was about primes or about 17 being a lucky prime |
| 45 | 9*5, coprime, no factor of two anywhere | how to fill SIMD lanes when no radix-2 is available and 4 complex doubles divides neither factor |
| 64 | 2^6, 4.19 MB volume, power-of-two strides throughout | the cache-set conflict / padding question sections 04 and 05 of the corpus disagree about, at the first size where the volume cannot hide in cache |

Rounds panel_r$NEW_NEXT through panel_r$NEW_LAST run this wider pool. The new entries start as
stubs whose header comments carry their strategy brief; \`fft3d_supports()\` returns 0 until
an implementer writes the real kernel, so the harness skips them cleanly in the meantime.

Not included, and why: **anisotropic geometries** (32x32x64, 48x48x96 -- the shapes the real
workload has) need an ABI change, since \`fft3d_api.h\` takes a single cube side L. That is a
deliberate separate step, not a stub away.
NOTE
LOG "wrote results/WAVE2.md"

rm -f "$GEOM/after_series.sh"
LOG "disarmed the hook; cron will start panel_r$NEW_NEXT on its next tick"
