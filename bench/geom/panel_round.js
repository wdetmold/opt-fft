export const meta = {
  name: 'fft-panel-round',
  description: 'Panel of implementers writes optimized library-free 3D FFTs for L=6,8,17,36; monitor benchmarks them on an isolated node',
  whenToUse: 'One round of the optimization competition. Pass args {round, seed, previous_leaderboard}.',
  phases: [
    { title: 'Implement', detail: '12 implementers: 3 competing strategies x 4 geometries' },
    { title: 'Monitor', detail: 'isolated-node benchmark: fresh random data, verify, time, rank' },
  ],
}

const ROUND = (args && args.round) || 'panel_r1'
const SEED = (args && args.seed) || 424242
const PREVIOUS = (args && args.previous_leaderboard) || null
const GEOM = '/home/lqcd/wdetmold/fft/bench/geom'

const COMMON = `
# Your task

You are one implementer on a panel building an optimized complex 3D FFT for a FIXED
cube size. Read these first, in this order:

  ${GEOM}/PANEL_BRIEF.md      the rules, the contract, the machine, the targets
  ${GEOM}/fft3d_api.h         the exact ABI you must implement
  ${GEOM}/driver.c            how you are timed (read it: it decides your score)
  /home/lqcd/wdetmold/fft/docs/LITERATURE.md    the cited corpus's per-size strategy table

and then the corpus sections most relevant to your strategy, in
/home/lqcd/wdetmold/fft/docs/literature/ (01 small-n codelets and codegen, 02 prime and
awkward lengths, 03 vector radix, 04 SIMD/layout/registers, 05 memory hierarchy,
06 autotuning, 07 register-level fusion). These are long; read the sections that bear on
your assignment rather than all of them.

# Hard rules (violating any of these makes your entry worthless)

* NO FFT library calls inside fft3d_execute(). Not FFTW/MKL/cuFFT/ducc0/anything. The
  arithmetic must be yours. libm during setup is fine.
* Single-threaded. No OpenMP, no pthreads.
* Write EXACTLY TWO files: ${GEOM}/impl/<YOUR_FILENAME>.c (the implementation) and
  ${GEOM}/strategies/<YOUR_FILENAME>.md (your strategy record) — nothing else, anywhere.
  Do not modify driver.c, fft3d_api.h, Makefile, sweep.sh, check.py, PANEL_BRIEF.md,
  another implementer's impl/*.c, or anything under python/.
* Do NOT run \`make\` in ${GEOM} — it shares driver.o and bin/ with 11 other agents
  running right now and will race. Compile your own file into your own scratch
  directory with an explicit gcc command (the brief shows it).
* All precomputation goes in fft3d_create(); it is excluded from your timing. Be as
  extravagant there as you like.
* Correctness is a gate, not a tradeoff: relative L2 vs numpy must be < 1e-12, and a
  correct implementation lands near 1e-16. Verify with check.py at B=1 AND at a large
  batch, and verify that calling fft3d_execute twice on one plan still gives the right
  answer. An entry that is fast and wrong scores nothing.

# The machine

Scored on an isolated Xeon Gold 5218 (Cascade Lake, 2.30 GHz) WITH AVX-512
(avx512f/dq/bw/vl/cd/vnni), compiled there with -O3 -march=native -std=gnu11. Your dev
machine is Haswell (AVX2 only) so AVX-512 code compiles locally with an explicit
-mavx512f but CANNOT run locally. Structure it so the portable path is what you verify
locally and the AVX-512 path is guarded by #ifdef __AVX512F__ — and make sure BOTH
paths are correct, because the graded build takes the AVX-512 one.

You may request a different compiler or an autotuning framework (gcc 5-16, LLVM 5-22,
Intel oneAPI are installed as modules; pip is available) — say so in your return value
and use the default gcc for now.

# What to return

A structured summary: your filename, the technique in two sentences, the local timings
you measured (B=1 and batched, per transform), the correctness numbers you got, what you
tried that did NOT work, and what you would do with another iteration. Be honest about
measured numbers — the monitor re-measures everything on the real node and any
discrepancy will be visible.
`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    L: { type: 'integer' },
    technique: { type: 'string' },
    local_us_per_transform_b1: { type: 'number' },
    local_us_per_transform_batched: { type: 'number' },
    batched_b_used: { type: 'integer' },
    rel_l2_error: { type: 'number' },
    verified_repeatable: { type: 'boolean' },
    uses_avx512_path: { type: 'boolean' },
    what_failed: { type: 'string' },
    next_iteration: { type: 'string' },
    requests: { type: 'string', description: 'compiler/framework requests, or empty' },
  },
  required: ['file', 'L', 'technique', 'rel_l2_error', 'verified_repeatable'],
}

const STRATEGIES = [
  // ---------------- L = 6 : 216 points, 6 = 2*3, tiny; codelet territory
  { file: 'L6_unrolled', L: 6, brief: `Strategy: FULLY UNROLLED STRAIGHT-LINE CODELET.
Hand-write (or generate at build time, or generate into a table in fft3d_create) a
complete straight-line 6-point DFT with no loops and no runtime twiddle lookups: 6 = 2*3
so it is a radix-2 stage over a Winograd/Rader 3-point module, and every twiddle for
n=6 is a constant you can write literally (the 3-point module needs only cos(2pi/3) =
-1/2 and sin(2pi/3) = sqrt(3)/2). Then apply that codelet along all three axes. The
whole 216-point volume is 3.4 KB and fits in L1 many times over, so aim to load a volume
once, do all three axes, store once. Section 01 of the corpus is your section: read what
genfft does about scheduling and register pressure before you write the DAG by hand.` },
  { file: 'L6_pfa', L: 6, brief: `Strategy: GOOD-THOMAS / PRIME-FACTOR ALGORITHM.
6 = 2*3 with 2 and 3 coprime, so the Good-Thomas index map turns the 6-point DFT into a
2x3 array of independent small DFTs with NO twiddle factors between stages at all (that
is the PFA's whole selling point versus Cooley-Tukey). Build the index permutation in
fft3d_create as a table, then the execute path is: permute, 3 two-point DFTs, 2
three-point DFTs, permute back — and for 3D you can compose the permutation across all
three axes into one index table so the data is permuted once, not six times. Section 02
covers PFA; section 03 covers whether fusing the axes pays. Quantify the multiply count
you achieve against Cooley-Tukey and say it in your return.` },
  { file: 'L6_batchsimd', L: 6, brief: `Strategy: BATCH-MAJOR SIMD (vectorize ACROSS volumes).
A 6-point transform is far too short to vectorize within, so do not try. Instead put 8
different volumes in the 8 lanes of an AVX-512 zmm register (4 for AVX2/ymm) and run one
scalar-shaped algorithm in lockstep across them: every lane does the same arithmetic on
its own volume, so there are NO cross-lane shuffles and no permutation networks at all.
That needs a repack from the driver's layout (volume-major) into lane-major and back;
count that cost honestly and consider whether repacking on the fly per 8-volume block
beats a full pre-pass. This is the strategy section 04 argues is the winning move for
small transforms — read it, then beat MKL's 0.370 us at B=1 and 22.6 GF/s.` },

  // ---------------- L = 8 : 512 points, 2^3, register-resident
  { file: 'L8_radix8', L: 8, brief: `Strategy: SINGLE-PASS RADIX-8 CODELET.
An 8-point DFT is one radix-8 butterfly: 8 = 2^3, all twiddles are 8th roots of unity,
i.e. +-1, +-i and +-(1+-i)/sqrt(2), so the whole codelet needs ONE real multiplicative
constant (1/sqrt(2)) and otherwise only adds, subtracts and swaps of real/imaginary
parts. Write it fully unrolled with no twiddle table at all, then apply along all three
axes of the 8 KB volume, which is L1-resident. Get the operation count down to the
published optimum for n=8 (section 01 has the tables) and say what you achieved. The bar
is MKL at 0.653 us / 35.3 GF/s at B=1 — the strongest library number in the whole table,
so this is a hard target.` },
  { file: 'L8_fusedaxes', L: 8, brief: `Strategy: FUSE ALL THREE AXES, NO INTERMEDIATE STORE.
A 512-point complex volume is 8 KB: 128 AVX-512 registers' worth, so not register
resident, but comfortably L1 resident. Load one volume, do x, y and z axes without ever
writing an intermediate to memory beyond L1, and store the result once — the
accelerator-world pattern in section 07 (cuFFTDx holds the whole small transform in
registers/shared memory), transplanted to a CPU. Consider processing a 8x8 pencil at a
time so 8 of the 8-point transforms are in flight simultaneously and the SIMD width is
filled by the OTHER two axes' indices rather than by batching. Compare against a
straightforward three-pass row-column version you also write, and report both numbers so
we learn whether fusion actually pays here.` },
  { file: 'L8_batchsimd', L: 8, brief: `Strategy: BATCH-MAJOR SIMD (vectorize ACROSS volumes).
Put 8 volumes in the 8 lanes of an AVX-512 zmm (4 in AVX2) and run one scalar-shaped
radix-8-based algorithm in lockstep, so no cross-lane shuffles are ever needed. Repack
from the driver's volume-major layout into lane-major and back; measure that repack cost
separately and report it, because it is the whole question. This is the same idea as the
L6 batch-SIMD entry but at 8^3, where the batched case is MEMORY BOUND at large B (MKL
drops from 35.3 GF/s at B=1 to 17.1 GF/s at B=2048): so also think about whether you can
make the batched regime bandwidth-optimal, e.g. streaming stores, and report B=1 and
large-B separately. Section 04 is your section; section 05 for the bandwidth argument.` },

  // ---------------- L = 17 : 4913 points, prime -- the standing opportunity
  { file: 'L17_rader', L: 17, brief: `Strategy: RADER WITH A 16-POINT CONVOLUTION.
This is the geometry where every library does badly (best is FFTW at 81.7 us / 3.7
GF/s, roughly 6x below what the same libraries reach at L=8), so it is the biggest
opening on the board. 17 is prime and 17-1 = 16 is a PURE POWER OF TWO, which is the
best possible case for Rader: the 16-point cyclic convolution is two radix-2 FFTs of
length 16 plus 16 complex multiplies, and the transform of the b-sequence is a CONSTANT
you precompute once in fft3d_create. So one 17-point DFT costs about 2 FFT16 + 16 mults
+ O(17) adds instead of 289 multiply-adds. Write the FFT16 as a fully unrolled codelet
(all twiddles are 16th roots of unity). Read section 02 first — it covers exactly this
case and gives published operation counts for the 17-point module. Report your
multiply-add count per 17-point transform.` },
  { file: 'L17_winograd', L: 17, brief: `Strategy: A HAND-DERIVED 17-POINT MODULE (Winograd / nested small-n).
Rader is the obvious route for a prime; this entry exists to test whether a directly
derived 17-point module beats it. Look at Winograd's small-n modules and Nussbaumer's
tables (section 02 has the citations and counts), and at the special structure of 17 as
a Fermat prime (17 = 2^4+1): the multiplicative group mod 17 is cyclic of order 16, so
the Rader permutation decomposes into four radix-2 stages, which can be nested and
simplified symbolically rather than executed as a generic convolution. Derive the module,
verify it against python/slow_dft.py's definition symbolically or numerically BEFORE
optimizing it, then apply it along all three axes. If you conclude the derivation cannot
beat Rader, say so with the operation counts that show it — a well-evidenced negative
result is a real contribution here.` },
  { file: 'L17_matrixsimd', L: 17, brief: `Strategy: DENSE 17x17 MATRIX, MAXIMALLY VECTORIZED.
The contrarian entry: 289 multiply-adds per line is more arithmetic than Rader's ~200,
but a dense matrix-vector product has perfect data flow — no permutations, no index
tables, no cross-lane shuffles, pure FMA on contiguous data, and the matrix (17x17
complex = 4.6 KB) stays in L1. Modern cores can be limited by shuffles and dependency
chains rather than FLOPs, which is section 04's and section 06's central point. Do it
properly: precompute the matrix, process many lines at once so the FMA units stay fed
(the 17^3 volume has 289 lines per axis), exploit the symmetry W[k][j] = W[j][k] and
conjugate pairs if it helps, and use AVX-512 FMA with the batch or line index in the
lanes. Then report the honest comparison: arithmetic count versus achieved time.` },

  // ---------------- L = 36 : 46656 points, 36 = 4*9, cache starts to matter
  { file: 'L36_mixedradix', L: 36, brief: `Strategy: ROW-COLUMN MIXED RADIX WITH CACHE BLOCKING.
36 = 4*9 = 2^2*3^2. Build a 36-point line transform from a radix-4 stage and a radix-9
stage (or 6x6), both as fully unrolled codelets with precomputed twiddles. The volume is
746 KB, so it does NOT fit L1 (32 KB) or L2 (1 MB, barely) — this is the geometry where
the transposes and the memory hierarchy decide the result, which is section 05's
subject. Block the passes so each axis's working set is L1-sized (e.g. process a
tile of lines at a time), and handle the strided access along the slow axes explicitly
rather than leaving it to the hardware prefetcher. The bar is MKL at 163.6 us / 22.1
GF/s, which is 1.8x ahead of FFTW — so MKL is doing something right here that you need
to match.` },
  { file: 'L36_pfa', L: 36, brief: `Strategy: GOOD-THOMAS PFA (4 x 9) PLUS TILING.
4 and 9 are coprime, so Good-Thomas applies to L=36 and eliminates the inter-stage
twiddle multiplications entirely: the 36-point DFT becomes 9 four-point DFTs and 4
nine-point DFTs under an index permutation, with no twiddle factors between them. Against
that saving you pay a permutation, which at this size means non-contiguous access on a
746 KB volume — so the whole question is whether the arithmetic saving survives the
memory behaviour. Read section 02 for the PFA mechanics and section 05 for the access
cost, build the permutation into fft3d_create as a table, and fuse the permutation with
the axis transposes so the data moves as few times as possible. Report the multiply count
saved and whether it translated into time; either answer is informative.` },
  { file: 'L36_pencilfused', L: 36, brief: `Strategy: PENCIL/TILE-FUSED PASSES (avoid the transposes).
At 746 KB per volume the row-column algorithm's three passes each re-read the whole
volume, so you are bandwidth-limited before you are FLOP-limited. Restructure so that a
TILE of the volume (say 36 x t x t, sized to fit L1/L2) has all three of its axis
transforms done while resident — a four-step/six-step-style blocked formulation, or a
vector-radix pass that works on 2x2x2 blocks of the output. Sections 03 (vector radix,
and whether its arithmetic saving is real) and 05 (four-step/six-step with crossover
numbers, and cache-oblivious results) are your sections. Also test streaming/non-temporal
stores for the final write, and report the achieved bandwidth against the machine's peak
so we know whether you are at the roofline or still on arithmetic.` },
]

phase('Implement')
const built = await parallel(STRATEGIES.map(s => () =>
  agent(`${COMMON}

# Your assignment

  Geometry:  L = ${s.L}  (cube ${s.L}^3 = ${s.L ** 3} complex doubles per volume)
  Filename:  ${GEOM}/impl/${s.file}.c
  fft3d_name() must return exactly: "${s.file}"

${s.brief}

${PREVIOUS ? `# Results of the previous round (you are REVISING; your file may already exist -- read it first and improve it)\n\n${PREVIOUS}\n` : ''}
Put a comment header at the top of your .c file stating the technique, the operation
count per transform, and any assumption you rely on.

You MUST also maintain your strategy record at
${GEOM}/strategies/<YOUR_FILENAME>.md — the format is specified in PANEL_BRIEF.md
("Keep a strategy record"). Downstream rounds optimize from these records, so write it
for the next implementer, not for a grader: the derivation and operation count, the
layout and SIMD decisions, what you measured, and above all what you tried that did NOT
work together with the number that killed it. If the file already exists from an earlier
round, APPEND a new round section; never overwrite the history.

Then build it, verify it, and iterate on your own until you cannot improve it further
within this session.`,
    { label: s.file, phase: 'Implement', schema: IMPL_SCHEMA, effort: 'high' })
))

const entries = built.filter(Boolean)
for (const e of entries) {
  log(`${e.file}: L=${e.L} rel_l2=${e.rel_l2_error} b1=${e.local_us_per_transform_b1 ?? '?'}us avx512=${e.uses_avx512_path}`)
}

phase('Monitor')
const monitor = await agent(`You are the MONITOR for the FFT optimization panel. Twelve implementers have just
written library-free optimized 3D FFTs into ${GEOM}/impl/ (round "${ROUND}").

Your job: benchmark them all on an ISOLATED node, on freshly generated random data, and
report a leaderboard. Be adversarial about correctness — a fast wrong answer must be
caught and named.

Steps:

1. \`cd ${GEOM} && source /home/lqcd/wdetmold/fft/env.sh\`
2. List what the panel delivered: \`ls impl/\`. Note anything missing or oddly named
   (fft3d_name() must match the filename stem).
3. Check that every entry also left a strategy record in ${GEOM}/strategies/ and name any
   that did not (the records guide the next round, so a missing one is a real omission).
4. Sanity-build locally FIRST (Haswell login node, AVX2 only) so a broken file is caught
   before the node is occupied: \`make -k 2>&1 | tail -40\`. For each file that fails to
   compile, record the exact error. Do NOT fix other people's code; report it.
5. Submit the round to an isolated node:
      ./submit.sh --round ${ROUND} --seed ${SEED} --partition devel --time 55 --samples 12 --runs 3
   This is sbatch with --exclusive, so the timings are clean. It rebuilds ON that node
   (Xeon Gold 5218, AVX-512), generates fresh random data per case with the given seed,
   times every backend on identical data with the same driver, verifies every output
   against numpy, and writes results/${ROUND}/leaderboard.txt.
6. Wait for it, without hammering the scheduler:
      until [ -f results/${ROUND}/leaderboard.txt ]; do sleep 60; squeue -u $USER -h -o "%i %t %M" | grep fft || true; done
   If the job dies or the partition is busy, check \`sacct\`/the slurm-*.out file and say
   what happened. Try the \`prod\` partition only if \`devel\` is unavailable for a long time.
7. Read results/${ROUND}/leaderboard.txt, results/${ROUND}/environment.txt,
   results/${ROUND}/failures.txt (if present) and results/${ROUND}/build_errors.txt.
8. Cross-check the implementers' self-reported local timings against what you measured.
   Flag any entry that is more than ~2x off its own claim, and any entry whose
   correctness check failed or was never produced.

Return the leaderboard in structured form, per geometry, with the library baselines
included so the comparison is visible, plus a short verdict per implementation.`,
  { label: `monitor:${ROUND}`, phase: 'Monitor', effort: 'high', schema: {
      type: 'object',
      properties: {
        node: { type: 'string' },
        build_failures: { type: 'array', items: { type: 'object', properties: {
          file: { type: 'string' }, error: { type: 'string' } }, required: ['file', 'error'] } },
        correctness_failures: { type: 'array', items: { type: 'object', properties: {
          name: { type: 'string' }, detail: { type: 'string' } }, required: ['name', 'detail'] } },
        per_geometry: { type: 'array', items: { type: 'object', properties: {
          L: { type: 'integer' },
          regime: { type: 'string', description: 'non-batched or batched B=N' },
          ranking: { type: 'array', items: { type: 'object', properties: {
            name: { type: 'string' }, us_per_transform: { type: 'number' },
            gflops: { type: 'number' }, correct: { type: 'boolean' },
            is_library: { type: 'boolean' } }, required: ['name', 'us_per_transform', 'correct'] } },
          best_panel_vs_best_library: { type: 'string' },
        }, required: ['L', 'regime', 'ranking'] } },
        claim_discrepancies: { type: 'array', items: { type: 'string' } },
        missing_strategy_records: { type: 'array', items: { type: 'string' } },
        leaderboard_text: { type: 'string', description: 'the raw leaderboard.txt contents' },
        verdict: { type: 'string' },
      },
      required: ['node', 'build_failures', 'correctness_failures', 'per_geometry', 'verdict'],
    } })

return { round: ROUND, entries, monitor }
