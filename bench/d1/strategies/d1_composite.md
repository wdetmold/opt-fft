# d1_composite — strategy record

Class: composites (12/24/36/60, PFA/mixed-radix). Only L=60 is in the measured
case list (four regimes: B=1/512 x m=1/chained); 12/24/36 keep a correct dense
floor and got no tuning effort.

## Round d1_r1 (2026-09-02, fresh restart round)

### What the implementation is now

Good–Thomas PFA, 60 = 4·3·5 (pairwise coprime, so ZERO twiddle factors — the
whole transform is 15 DFT-4s + 20 DFT-3s + 12 DFT-5s plus two CRT
permutations, straight from the survey's d1_composite line). Input map
n = (15n1+20n2+12n3) mod 60, output map k = (45k1+40k2+36k3) mod 60, both
verified against numpy before any C was written (PFA err 7e-15). DFT-4 is
add-only; DFT-3/DFT-5 are the real-constant forms (cos/sin of 2π/5, 4π/5, 2π/3),
so a complex×constant is one vmulpd/vfmadd and ±i is one in-lane vpermilpd with
the sign folded into a (s,−s) constant — no complex-multiply shuffles anywhere.

Four code paths, chosen per call shape:

1. **B=1 execute — `fft60_ymm1`**: one complex per 128-bit lane, PAIRED OVER n1
   (2 complexes/ymm). Key observation: n1 only matters in stage A (DFT-4 is the
   n1 direction), so after stage A builds rows (n1=0|n1=1) and (n1=2|n1=3),
   stages B (DFT-3 over n2) and C (DFT-5 over n3) run two-wide with NO repacking
   at all. Stage A's DFT-4 itself is done in ymm: with A=(x0|x1), B=(x2|x3),
   S=A+B=(t0|t2), D=A−B=(t1|t3), then one vperm2f128 pair + one vpermilpd + two
   FMAs with E4=(1,1,1,−1) deliver (y0|y1) and (y2|y3) directly.
2. **Batched execute — `fft60_ymm2`**: two TRANSFORMS per ymm (transform b in
   the low 128 lane, b+1 high). All permutation indices identical across lanes,
   so the only overhead is one vinsertf128 per operand load and one vextractf128
   per store. A zmm 4-transform clone (`fft60_zmm4`, kept under -DUSE_ZMM4)
   measured a statistical TIE with ymm2 — the 512-bit shuffle tax (3
   inserts/extracts per operand, port-5-only) eats exactly what the width gains.
3. **Chained, B≥8 — SoA-resident chain** (own `fft1d_chain`): 8 transforms per
   zmm in split-complex SoA rows (lane = transform; zero shuffles, ±i is an
   array swap). The state lives in SoA across ALL m steps: gathered in once,
   scattered out once, so the AoS↔SoA cost that killed the SoA execute path
   amortizes to nothing. The inter-step permutation is pure row renaming
   (SROW = PININV∘KOUT), and SROW maps the 12 DFT-5 output blocks onto each
   other as an INVOLUTION (pairs 1↔2, 3↔9, 4↔11, 5↔10, 7↔8; 0,6 fixed), so the
   step runs fully IN PLACE: compute+map block P into registers, compute+map its
   partner Q, store both. Stages A+B are fused per n3-slice (the 12 DFT-4
   outputs of cols {n3,n3+5,n3+10} are exactly the 12 DFT-3 inputs at that n3 —
   no round trip). The map (state←z/(1+|z|)) is fused into stage C, 8-wide.
4. **Chained, B<8 — `chain60_ymm_step`**: the ymm1 kernel with stage C landing
   in split zr/zi buffers so the fused map runs 8-wide shuffle-free, then a
   two-vpermt2var interleave writes the state back.

The map's sqrt+div pair is replaced by **rsqrt14/rcp14 + 2 Newton rounds each**
(q clamped at 1e-300 so q=0 stays finite). A/B against exact sqrt+div: chain
rel_l2 6.599e-13 vs 6.302e-13 at m=600 — statistically indistinguishable, and
the strict m=2 one-step gate passes at 4.7e-16 (tol 3e-14). Speed: B=512 chain
0.089 → 0.054 us, B=1 chain 0.155 → 0.118 us. This is the single biggest
chain win after the SoA residency itself.

### Operation count

PFA-60 ≈ 490 complex-pair vector ops (in the xmm/ymm forms): stage A 15×9,
stage B 20×7, stage C 12×18, plus 60 loads/60 stores and ~60 port-5 shuffles.
Nominal-5NlogN would be 1771 flops; the PFA does ~980 real flops/transform.

### Measured on wallaby (SPR, quiet core, warm; medians of driver medians)

| cell | ours | MKL same core | FFTW-measure same core |
|---|---|---|---|
| B=1, m=1 | **0.045–0.046 us** | 0.036–0.041 | 0.048 |
| B=512, m=1 | **0.042–0.050 us** | 0.036–0.052 (turbo-dependent) | 0.072 |
| B=1, m=60000 | **0.105–0.112 us** | 0.204–0.214 | 0.208–0.214 |
| B=512, m=600 | **0.048–0.056 us** | 0.191–0.226 | 0.232–0.237 |

Correctness: rel_l2 = 2.2e-16 (single call), chain gates 3.2e-15 (B=1,
m=60000) and 6.6e-13 (B=512, m=600, tol 1e-10). Anti-memoization and
repeatability (bit-identical across runs) pass. Edge batches 2/3/5/8/12/13 all
pass both gates; L=12/24/36 dense floor passes.

WARNING for whoever reads these numbers: wallaby cores run schedutil and swing
~2.7–4.1 GHz between runs — a 3x apparent "regression" mid-session was pure
frequency. Only interleaved same-core A/B runs mean anything here. The library
baseline table (results/library_baseline/BASELINE.md) per-transform numbers for
the chained rows INCLUDE the driver-side map cost — do not read them as FFT
cost; MKL's actual m=1 kernels are ~5x faster than the chained rows suggest.

### What did NOT work, with the numbers that killed it

- **8-lane SoA (gather in / scatter out) for the m=1 batched execute**: 0.129
  us vs 0.061 for a plain per-transform kernel loop at B=512. 120 gathers + 120
  scatters per 8 transforms cost more than the 8-wide arithmetic saves in a
  single pass. The same machinery WINS for chains, where the transposes happen
  once per chain instead of once per call.
- **Plain scalar (re/im doubles) kernel**: 0.178 us at B=1 — gcc partially
  auto-vectorized it but left stages B/C scalar. The explicit 128-bit
  complex-pair form is ~2.8x faster; don't trust the autovectorizer here.
- **zmm 4-transform batched kernel**: tie with ymm2 (0.041–0.050 both, three
  interleaved A/B rounds). Kept in the file under -DUSE_ZMM4; not default
  because 512-bit clock licensing can only hurt that cell.
- **512-bit ymm→zmm stage-A rewrite fear check**: the ymm DFT-4 (perm2f128
  form) vs xmm-then-insert stage A: 0.045 vs 0.047 — small but real, adopted.

### Borrowed / verified against others

Everything here traces to docs/literature_1d/00-SURVEY.md (this round's other
entries were all still stubs when I looked): the across-batch split-complex
zero-shuffle lever (survey's top batched recommendation) became the SoA chain;
the "twiddle-free Good–Thomas for composites" line is the whole entry. The
SROW-involution in-place trick and the rsqrt/rcp map are, as far as I know, new
here — d1_prime/d1_rader could reuse the map trick verbatim in their chains,
and the n1-pairing argument (pair over the one dimension a stage doesn't
touch) generalizes to any PFA/CT split.

### Where the four cells stand vs best library (same-host estimates)

m=1 B=1: ~0.80x of MKL (we lose ~1.25x; beat FFTW). m=1 B=512: parity with MKL
within noise. Chained B=1: ~1.9x win. Chained B=512: ~4x win. The chained wins
come from owning fft1d_chain (no per-step AoS round trip + fused approx map);
no library can see across the FFT/map barrier.

### Next round

1. The B=1 m=1 gap to MKL (~35 cycles) is stage A's load+insert front end and
   the 60 scattered stores. Ideas: fold the output permutation into a
   store-order that pairs KOUT-adjacent outputs (KOUT has residue structure);
   or a 4-lane stage B/C using both n1-pairs in one zmm (needs one cross-lane
   shuffle per stage — port analysis says ~wash, but measure on ICL).
2. Measure zmm4 vs ymm2 on the actual ICL scoring node and flip USE_ZMM4 if it
   wins there (frequency behavior differs from SPR).
3. Batched chain: stages A+B store/load round trip per step is still ~240 mem
   ops/group; a fully register-resident A+B+C over one n2-slice needs 24 zmm —
   feasible, ~10-15% more.
4. If a d1_planner/d1_race layer appears: expose ymm1/ymm2 kernel choice and
   the chain ownership as capabilities; the crossover B for SoA-chain vs
   per-transform chain is between 2 and 8 (untested inside that range).

## Round d1_r2 (2026-09-02)

### Where r1 left me on the scoring node (a80n0, Ice Lake Gold 6326)
Chained cells won (B=1 1.75x, B=512 3.35x over MKL). Both m=1 cells LOST:
B=1 0.0690 vs MKL 0.0612 (1.13x behind), B=512 0.0647 vs 0.0498 (1.30x). No
Ice Lake reservation existed this round either (job 440371 dead, queue empty,
and the brief forbids submitting our own), so all r2 numbers are wallaby
(SPR 6448Y, pinned core 101, interleaved same-core A/B vs the freshly rebuilt
build/wallaby/bin/mkl1d_dfti).

### The one change that mattered: force FULL UNROLL of every hot loop
objdump on the r1 binary showed gcc 11.4 kept all kernel loops ROLLED despite
-funroll-loops: every stage-A/B/C iteration re-loaded PIN/KOUT indices
(movslq), did scalar address arithmetic (~50 mov + 42 movslq + 36 add per
ymm1 call), and spilled the wp[] working set (101 vmovapd). MKL's codelets
are straight-line with constant offsets -- that was most of the m=1 gap.
Fix: `_Pragma("GCC unroll N")` on all 30 fixed-trip loops (works inside macro
bodies too, where #pragma cannot go). The CRT tables then constant-fold into
addressing displacements; the kernels become pure straight-line SIMD.

Measured (wallaby, min over 8 samples, stable across interleaved passes):
| cell | r1 | r2 | MKL same core |
|---|---|---|---|
| B=1, m=1 | 0.045 us | **0.033** | 0.036 |
| B=512, m=1 | 0.040 us | **0.032** | 0.034 |
| B=1, m=60000 | 0.105 | 0.105 (latency-bound, unchanged) | 0.191 |
| B=512, m=600 | 0.046 | **0.044** | 0.185 |

Both previously-lost m=1 cells now beat MKL on wallaby. Correctness: single
call rel_l2 2.2e-16; strict m=2 gate 3.8-5.3e-16 (tol 3e-14); chain gates
1.2e-14 (B=1 m=60000... 3.2e-15 on the graded seed) and 8.1e-13 (B=512
m=600, tol 1e-10); edge batches 2/3/5/8/13 all pass; L=12/24/36 dense floor
passes; bit-identical across runs.

A second reason to expect this to carry to the node: the removed overhead was
scalar/front-end work, which hurt MORE on ICL (one FP-shuffle port, narrower
front end) than on SPR -- consistent with r1's node ratios being worse than
the wallaby ratios for the same binary pair.

### What did NOT work this round, with the numbers
- **fft60_zmm2x2** (NEW kernel, kept in-file under -DUSE_ZMM2X2): 2 transforms
  x 2 n1-pairs per zmm -- ymm1's pairing widened, stages B/C at HALF ymm2's
  per-transform op count, stage A via two vpermt2pd per column (the t3 element
  swap folded into the permute indices). Codegen verified clean: 415 instr/
  transform vs ymm2's 565, few spills. Still LOST: 0.045 -> 0.037 (after
  rewriting loads as two 256-bit halves + one vinsertf64x4, and stores as one
  vextractf64x4 + memory-form vextractf128, which are pure stores) vs ymm2's
  0.032. Lesson recorded: at 512-bit ALL shuffle/blend/FMA uops share p0+p5,
  so instruction count is not the bound -- ymm2's 256-bit mix spreads across
  p0/p1/p5 and wins despite ~35% more instructions. Same conclusion as r1's
  zmm4 tie, now with the load/store side de-p5'd and it still loses.
- **fft60_ymm1 as the batched kernel** (-DUSE_YMM1_BATCH): 0.033-0.034 vs
  ymm2's 0.031-0.032. Fewer ops/transform (204 vs 228) and no w[60] stack
  round-trip, but the per-call n1-pair shuffle overhead (30 vshuff64x2/call)
  costs more than the round-trip saves. ymm2 stays default.
- **zmm4 rechecked post-unroll**: 0.038 vs ymm2 0.032 -- now a clear loss,
  not the r1 tie. Unrolling helped ymm2 more than zmm4.
- Memory-operand folding facts verified in disassembly (useful to others):
  vextractf128/vextractf64x2-to-MEMORY is a pure store (no shuffle-port uop),
  and vinsertf128-from-memory is load + p015 blend -- but their REGISTER-source
  forms are p5 shuffles. The STAGEA_XMM variant (still in-file) loses partly
  because its inserts are register-source: 0.036 vs 0.033 at B=1.

### Borrowed
Nothing new adopted this round; checked d1_prime's and d1_batchlane's r1
records first. d1_prime's "check the disassembly, gcc won't do it for you"
lesson pointed straight at the rolled-loop discovery -- credit where due;
their split-accumulator trick is not applicable here (PFA stages are already
shallow). My movslq/constant-fold finding is the same disease in a different
organ: gcc keeps table-indexed loops rolled even at -O3 -funroll-loops;
_Pragma("GCC unroll N") is the cure and every entry with static index tables
should check for it.

### Next round
1. If an ICL reservation is live: PMU the ymm2 batched loop (port 5 uops,
   front-end stalls) -- the choice ymm2-vs-zmm2x2 deserves one on-node A/B
   since the p0/p1/p5 balance differs from SPR (both variants ship in-file).
2. B=1 chain (0.105) is latency-bound: the step's stage-A loads depend on the
   previous step's interleaved zmm stores (store-forward of 16B from 64B
   stores). Keeping the B=1 chain state SPLIT across steps (like the B>=8 SoA
   path, lanes = n1-pair slots) would remove the interleave round-trip; the
   fused map already works split. Est. ~10-15%.
3. The batched chain's stages A+B still round-trip w[] through the stack;
   a register-resident A+B+C per n2-slice needs ~24 zmm and might give
   ~10% -- but that cell already wins 4.2x, so only touch it if the race
   entry needs the margin.
