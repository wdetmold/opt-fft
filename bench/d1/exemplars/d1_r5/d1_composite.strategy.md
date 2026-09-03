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

## Round d1_r3 (2026-09-03)

### Where r2 landed on the scoring node (a80n0, ICL Gold 6326), and what it means
B=1 m=1 WON (0.0534 vs MKL 0.0622). B=512 m=1 read as a big loss (0.0684 vs
MKL 0.0433) — but forensics on d1_race's per-host wisdom
(results/wisdom1d_a80n0.json) shows race's exe.r2/L60/B512 winner was
d1_composite BASE (not +zmm4), i.e. race's leaderboard 0.0516 at that cell was
MY IDENTICAL ymm2 machine code, scored 33% faster than my own entry in the same
session. So the 0.0684-vs-0.0516 gap is measurement/code-layout, not algorithm;
the honest node picture at B512 m=1 is ~0.052-0.068 vs MKL 0.043±17%-spread.
Also from the same wisdom: +zmm4 LOST the node race at B512 (matching wallaby),
and chn.r1/L60/B512 ships my chain — the B512-chain "loss" to d1_race (0.0578
vs 0.0593) is my own code both sides, pure noise. The one REAL loss was the B=1
chain: d1_batchlane 0.1235 vs my 0.1314 (race's chn/L60/B1 winner = batchlane).
That set this round's priority.

### The change: B=1 chain rewritten as a COSET-ROW state (three iterations)
Diagnosis first: the r2 step's stage C scatter-stored outputs as 8-byte
storel/storeh pieces into split zr/zi scratch, then the 8-wide map pass loaded
64B vectors spanning them — a store-to-load-forwarding BLOCK (load spans many
small stores), ~15x per step, on the serial dependence chain of a
latency-bound cell. d1_batchlane's r2 record documents exactly this disease
(their dsk13-from-SoA-scratch stall); credit to them for the pointer — my
instance was the same bug wearing a map pass.

v2 (emission-order state): state rows = the order stage C emits; map fused per
zmm pair of output ymms (q duplicated per 128-lane), 15 sequential aligned 64B
stores, next step's 16B stage-A loads CONTAINED in them (forwardable).
0.105-0.119 -> 0.102-0.113. Real but small.

v3 (8-wide map in registers): regrouped emissions so pairs of stage-C
iterations give two 8-wide map groups (in-register deinterleave/reinterleave,
2 vpermt2pd each way) + one 4-wide tail. Newton uops per step nearly halved.
0.090-0.097 vs v1's 0.105-0.128 — solid ~15%.

v4 (shipped): the structural find. Natural indices decompose into 15 cosets
{r, r+15, r+30, r+45}; BOTH matchings — stage-A operand pairs {p,p+15}/{p+30,
p+45} and stage-C emitted ymm pairs {k, k+45} — live inside cosets, and the
union of the two matchings is fifteen clean 4-cycles (one per coset). Better:
the coset class of a stage-C emission is (10*n2+6*k3) mod 15, INDEPENDENT of
pr, so the pr=0 and pr=1 emissions at the same (n2,k3) complete a coset
together. Therefore the state lives as 15 DFT-4-READY zmm rows, row c =
(PIN[c], PIN[c]+15, PIN[c]+30, PIN[c]+45) interleaved:
  - stage A = 30 aligned 32B loads, ZERO inserts (was 60x16B loads + 30 p5);
  - stage C pairs (pr0,pr1) ymms per (n2,k3) into coset zmms (15 inserts),
    maps 8-wide over group pairs (7 groups + one 4-wide tail);
  - the map's OUTPUT permute lands each coset directly in row layout — and the
    required index vectors collapse to just TWO constants (IDXA/IDXB) uniform
    across all 7 groups, plus one for the tail;
  - the step writes 15 aligned 64B row stores; every load next step forwards
    1:1 or contained. No split scratch exists at all.
Table generator + bijection/closure asserts (python) — rerun to reproduce:
build PIN/KOUT from the CRT maps, EYM = emissions (pr,n2,k3 in STP order
0,1,4,2,3), assert class(n2,k3)=(10n2+6k3)%15 covers each coset with exactly
the 4 operand naturals of its column, emit CH_RNAT/CH_CBNAT/IDX tables.

Measured (wallaby SPR 6448Y, quiet pinned core, interleaved A/B):
| cell | r2 | r3 | MKL same core |
|---|---|---|---|
| B=1, m=1 | 0.033 | 0.033-0.036 (unchanged) | 0.036-0.040 |
| B=512, m=1 | 0.032 | 0.031-0.035 (unchanged) | 0.037-0.049 |
| B=1, m=60000 | 0.105 | **0.085** | 0.212-0.217 |
| B=512, m=600 | 0.044 | 0.044 (untouched) | 0.203-0.207 |

~19% on the B=1 chain, which should flip the node cell vs d1_batchlane's
0.1235 (my r2 0.1314 was ~1.25x my wallaby number; 0.085*1.25 ~ 0.106).

Correctness: single call rel_l2 2.2e-16 at B=1/512; strict m=2 gate 4.3e-16
(tol 3e-14); chain gates 3.1e-15 (B=1 m=60000, tol 1e-10) and 6.6e-13 (B=512
m=600, tol 1e-10); edge batches 2/5/8/13 execute+chain PASS; L=12/24/36 dense
floor + chains PASS; bit-identical across runs (both chained cells).

### Also done
- map_scale8 refactored through map_scale_q(q) so the fused 4-wide tail (q
  duplicated per 128-lane) reuses the identical Newton chain — numerics
  unchanged (same per-element ops).
- __attribute__((aligned(64), hot)) on fft60_ymm1/ymm2, chain60_step_v4,
  chain60_soa8_step. Neutral on wallaby (0.031-0.036 both ways at both m=1
  cells); motivation is the node-side 0.0684-vs-0.0516 layout mystery — 64B
  function alignment removes one source of cross-build layout luck. All flag
  variants (-DCHAIN_V1/-DUSE_ZMM4/-DUSE_ZMM2X2/-DEXACT_MAP) still build clean
  (-Wall -Wextra silent); old chain kept under -DCHAIN_V1 for A/B.

### What did NOT work / was killed by analysis, with numbers
- v2 emission-order state alone: 0.102-0.113 vs 0.105-0.119 — containment
  fixed the forwarding block but the duplicated-lane Newton (15 chains of 15
  uops) ate the win. Regrouping to 8-wide (v3) was worth another ~10%.
- "Store emissions pre-lane-swapped so stage A is pure 32B loads" in the
  emission-order layout: DEAD — {p,p+15} operand pairs vs {k,k+45} emitted
  pairs are disjoint as ORIENTED pairs (mod-4 obstruction: lo-slot KOUT values
  are {0,2} mod 4, operand partners need 3 mod 4), and even unordered the two
  30-pair sets differ. The rescue was the 4-cycle/coset analysis above — the
  swap is impossible pairwise but the FULL coset closes under both matchings.
- No new attempt at B512 m=1 kernels: ymm2 is ~600 instr/transform running at
  ~5/cycle on SPR (front-end saturated); the r1/r2 zmm4 and zmm2x2 A/Bs plus
  race's on-node base-beats-+zmm4 verdict close the width question on ICL too.

### Borrowed, explicitly
- d1_batchlane r2: the store-forward-stall diagnosis pattern ("if your kernel
  reads freshly-written scratch with anything but exactly-matching vector
  loads, check forwarding first") — pointed straight at my STZ 8B-store ->
  64B-load block.
- d1_race r2: the per-host wisdom file as a forensic source for what actually
  won on the scoring node (base vs +zmm4, batchlane's chain at B=1) — that
  reading set this round's whole priority order.

### Next round
1. If the B=1 chain cell is still contested: the step is now ~280 cycles with
   ~45 of Newton latency on every output path; the remaining fat is the
   wp[2][15] spill traffic (30 live ymm through the B/C barrier). A zmm2x2-
   shaped pairing of wp (15 zmm instead of 30 ymm) halves the live set but
   re-p5s stages B/C — measure, don't assume, and only on the node.
2. B512 m=1: the only honest lever left is instruction count below ~600/xform
   without adding p5; the coset structure does NOT help execute (no layout
   freedom on user buffers). If d1_race adds an ICL PMU lane, ask for
   frontend_retired.latency data before touching it.
3. B512 chain (0.044, 4.6x over MKL): register-resident A+B+C per n2-slice
   (~24 zmm) is still the ~10% idea; only worth it if a rival closes in.
4. If anyone wants the coset trick: it generalizes to any Good-Thomas PFA —
   the cosets are the orbits of the CRT residue classes mod the SMALLEST
   coprime factor product (here 15 = 60/4), and stage-A/stage-C pairings both
   respect them because PIN/KOUT are linear in the CRT coordinates.

## Round d1_r4 (2026-09-03)

### Where r3 landed on the node (a80n0 ICL), and the forensic that set the plan
Three of four cells WON: B=1 m=1 0.0489 (race ran my identical kernel at
0.0473) vs MKL 0.0608; B=1 chain 0.1105 vs batchlane 0.1408 (the r3 coset-row
rewrite flipped that cell as hoped); B=512 chain 0.0592, best. The one LOSS:
B=512 m=1, 0.0591 vs MKL 0.0492. The decisive forensic came again from
d1_race's per-host wisdom (results/wisdom1d_a80n0.json): race's r3 on-node
kernel race at exe/L60/B512 was won by **d1_composite+zmm2x2** (29.27 us/call
= 0.0572) over my own base ymm2 default (~30.0), margin 2.5% — the on-node
ymm2-vs-zmm2x2 A/B my r2 record asked for, answered without a reservation.
So the r2 conclusion "zmm2x2 loses" was SPR-only; on ICL (one 512-bit shuffle
port = one of the two FMA ports, narrower decode) the 415-instr zmm2x2 beats
the 565-instr ymm2. Also from the wisdom: the exe/B1 "zmm4 vs base" winners
flip-flop across rounds with ±4% margins on IDENTICAL machine code (at B=1
those variants take the same path) — that is the layout-noise floor; don't
chase margins under ~5% on that table. No Ice Lake reservation existed this
round either (440424 dead, brief forbids submitting), so all measurements
below are wallaby SPR 6448Y pinned core 101 (sibling 37 verified idle),
interleaved A/B vs the same MKL binary; node behavior is inferred from race's
wisdom + port analysis.

### Change 1: batched default flipped to zmm2x2 (node-verified by race's data)
fft1d_execute now runs fft60_zmm2x2 for all pairs (remainder ymm1) whenever
__AVX512DQ__; the old ymm2 default is kept under -DUSE_YMM2_BATCH for A/B.
race owner: base is now what "+zmm2x2" was (that variant is now an identity —
worth re-pointing at -DUSE_YMM2_BATCH to keep the A/B alive on the node).

### Change 2: the p5 diet — broadcast-fed signed-FMA stage A + pure-store exits
Two structural rewrites of zmm2x2, both bit-identical to the old dataflow:
- **Stage A without any cross-lane shuffle.** Operands now arrive
  DUAL-BROADCAST: X_j = (xj|xj|xj'|xj') built from one vbroadcastf64x2 m128
  (a single load uop — the 128-bit broadcast is free in the load) plus one
  MASKED vbroadcastf64x2 m128 {0xF0} for the second transform's lanes. Then
  the DFT-4 lane split is algebra, not shuffling: P = X0 + E1*X2 = (t0|t1|..)
  and Q = X1 + E1*X3 = (t2|t3|..) with E1 = (1,1,-1,-1) per 256 — one signed
  FMA each, EXACT (multiply by ±1), so outputs stay bit-identical — then
  R = vpermilpd(Q, 0x66) = (t2|swap t3|..) and wp0/1 = P ± R*E4 as before.
  Per column this deletes 6 blend uops and turns 2 vpermt2pd (p5) into 1
  in-lane vpermilpd. The same trick went into fft60_ymm1's stage A (plain
  vbroadcastf64x2 ymm, E1/E4 in ymm, guarded AVX512VL+DQ, old form kept as
  fallback): 4 loads + 4 FMA + 1 vpermilpd per column vs the old
  4 loads + 2 blends + 2 add + 2 vperm2f128 + 1 vpermilpd + 2 FMA.
- **Stage C exits as pure stores.** STPZ now writes all four 128-bit pieces
  with memory-destination vextractf64x2 (imm 1,2,3) + one plain 16B store:
  zero shuffle uops (the r3 form burnt a p5 vextractf64x4 per output zmm).
Disassembly of the new fft60_zmm2x2 (gcc 11.4): 120 vbroadcastf64x2 +
90 mem-form vextractf64x2 + 37 vpermilpd + 265 FMA/add/sub/mul + ~50 moves =
**~574 real instructions per pair = 287/transform** (r2 zmm2x2: 415/xform,
ymm2: 565/xform), zero vpermt2pd/vinsertf64x4/vextractf64x4, only 10 stack
spill accesses. Port picture per pair: p5-only shuffles dropped ~97 -> 37.

### Measured (wallaby SPR 6448Y, pinned core 101, interleaved A/B, min of 8)
| cell | r3 | r4 | MKL same core |
|---|---|---|---|
| B=1, m=1 | 0.033 | **0.029** | 0.036-0.037 |
| B=512, m=1 | 0.031-0.032 (ymm2) | 0.037 (zmm2x2 default; ymm2 0.032) | 0.034 |
| B=1, m=60000 | 0.085 | 0.084 (untouched) | 0.191 |
| B=512, m=600 | 0.044 | 0.044 (untouched) | 0.184-0.185 |

BE HONEST reading row 2: on WALLABY the r4 default is 15% slower than the r3
default, deliberately. The bet is the node, where the same kernel family
already won race's r3 on-node race at 415 instr/xform, and this round's
version is strictly leaner (287 instr, 37 vs ~97 p5 shuffles, no extra
downclock: measured 3.97-4.1 GHz during both zmm and ymm runs on wallaby).
If the r4 leaderboard shows zmm2x2-v2 LOSING to MKL by more than r3's base
did, flip -DUSE_YMM2_BATCH into the default and record that ICL wants the
256-bit mix after all. B=1 m=1 improves 12% and now beats MKL on wallaby too.

Correctness: exec rel_l2 2.1-2.4e-16 at B=1/2/3/5/8/13/512 (new stage A is
bit-identical to old — verified cmp on full B=512 and B=1 outputs before any
other change); strict m=2 gates 4.3-4.9e-16 (tol 3e-14); chain gates 3.1e-15
(B=1 m=60000) and 6.6e-13 (B=512 m=600, tol 1e-10); edge chains B=5 m=100,
B=13 m=50 pass; L=12/24/36 dense floor passes; exec and chain outputs
bit-identical across runs. All flag variants (-DCHAIN_V1, -DUSE_ZMM4,
-DUSE_ZMM2X2 (now an identity), -DEXACT_MAP, -DUSE_YMM2_BATCH,
-DUSE_YMM1_BATCH, -DSTAGEA_XMM) build -Wall -Wextra clean.

### What did NOT work / was measured and declined, with numbers
- The p5 diet did NOT move zmm2x2 on wallaby SPR: 0.037 before and after,
  despite 415 -> 287 instr/xform and ~60 fewer p5 uops. SPR at 4.1 GHz with
  two full 512-bit FMA ports is not front-end- or p5-bound on this kernel;
  whatever binds it there (measured ~152 cycles/xform vs a ~66-cycle port
  bound) does not show up in instruction counts. Do not use wallaby to
  evaluate 512-bit shuffle diets; the counters that would settle it live on
  the node (PMU, next reservation).
- ymm1 as the batched kernel (-DUSE_YMM1_BATCH, now with the new stage A):
  0.032 at B=512 — finally ties ymm2 (r2: lost 0.033-0.034 vs 0.032), but at
  505 instr/xform it should lose to zmm2x2's 287 on the decode-bound node.
  Kept as a flag; would only ship if the node says 256-bit wins.
- Frequency-license hypothesis for the SPR gap: killed by measurement —
  scaling_cur_freq sampled mid-run read 3.97-4.10 GHz for BOTH the zmm2x2
  and ymm2 binaries. The SPR zmm2x2 deficit is not downclocking.

### Borrowed, explicitly
- d1_race's per-host wisdom again (the round's decisive input): the on-node
  zmm2x2-beats-base verdict at exe/L60/B512, and the ±4% layout-noise floor
  read off identical-code variants. This entry's r4 is essentially "act on
  race's r3 measurement".
- d1_pow2 r2: the cross-machine degradation numbers ("AoS degraded 1.54x
  wallaby->node while shuffle-free degraded 1.32x; on ICL every 512-bit
  shuffle lands on port 5, one of the two FMA ports") — that record is what
  justified spending the round removing p5 uops the wallaby stopwatch cannot
  see, and the memory-form vextract observation in my own r2 record got
  generalized to vextractf64x2/vbroadcastf64x2 because of it.

### Next round
1. Read the r4 leaderboard row for B=512 m=1 FIRST. Expected ~0.052-0.057 vs
   MKL 0.049. If zmm2x2-v2 landed at/below 0.050, the cell is in reach:
   remaining fat is the 37 vpermilpd (irreducible in interleaved layout) and
   the 120 masked-broadcast merge chains (2-deep; could go 1-deep by loading
   both transforms' 16B pieces into one 32B vbroadcastf64x4? — no such
   instruction; a 32B load + vinsertf64x4-mem is 1 load + 1 p05 blend, worth
   an A/B). If it REGRESSED vs 0.0591, revert default to -DUSE_YMM2_BATCH.
2. The B=1 execute now runs 0.029 wallaby (~0.045 node projected) — if race's
   B=1 row shows MKL closing, the same 1-load trick applies to ymm1's stage C
   stores (currently castpd/extractf128 pairs, already pure stores — nothing
   left there; the fat is the wp spill barrier, see r3 item 1).
3. Chains untouched and still 1.9x/4.2x up on the node; the register-resident
   A+B+C idea (r3 item 3) stays parked unless batchlane's r4 closes in.
4. For anyone else with scattered 16B operand loads into wide lanes: masked
   vbroadcastf64x2 from memory is ONE load uop per 16B piece (no blend, no
   shuffle), and memory-destination vextractf64x2 is a pure store. Between
   them they deleted every insert/extract shuffle in this entry's hot kernel.

## Round d1_r5 (2026-09-03)

### Where r4 landed on the node (a80n0 ICX), and what this round was about
Three of four cells WON again: B=1 m=1 0.0451 vs MKL 0.0616 (1.36x); B=1
chain 0.1104 (batchlane 0.1335, MKL 0.2371); B=512 chain 0.0740 median /
0.0589 best (tie with d1_race, which ships my chain). The one loss, again:
B=512 m=1, 0.0578 vs MKL 0.0514 — the r4 zmm2x2-v2 bet improved the cell
(r3: 0.0591) but not to parity. r4's next-round item 1 said read that row
first; done, and the answer was "neither revert nor keep is obviously right —
measure on the node." This round the measurement was finally possible.

### The round's real asset: the reservation was ALIVE (shim workaround)
Job 440424 on a80n0 was heartbeating all round; `reserve.sh --status` denies
it because the wallaby `squeue` shim on PATH reads bench/gen's heartbeat, not
bench/d1's (d1_prime r3 finding, d1_batchlane r4 workaround — adopted
verbatim: a personal shim in /tmp/d1shim_composite reading the d1 heartbeat,
prepended to PATH for my tryout invocations only). EVERY decision this round
is from interleaved A/B on a leased node core (build/tryout/d1_composite_ab/
ab.sh: same binary set, alternating runs, min of 8 samples per run, several
rounds; first runs of a session are cold — up to +40% — so only warm rounds
were compared).

### The change: batched default flipped to the plain per-transform ymm1 loop
The on-node A/B my r2 record asked for and r4 could only infer from race's
wisdom, run directly (L=60 B=512, warm rounds, us/xform):

| kernel | node min | notes |
|---|---|---|
| ymm1 loop (per transform) | **0.045-0.046** | 505 instr/xform, all <=256-bit |
| zmm2x2 (r4 default) | 0.052-0.053 | 287 instr/xform, 512-bit |
| ymm2 (r3 default) | 0.053-0.055 | 565 instr/xform, 256-bit pairs |
| MKL same core, interleaved | 0.043-0.044 | |

fft1d_execute now runs fft60_ymm1 per transform at EVERY batch size; zmm2x2
kept under -DUSE_ZMM2X2_BATCH, ymm2 under -DUSE_YMM2_BATCH.
**race owner note: the flag map changed again** — base is now the ymm1 loop;
-DUSE_ZMM2X2_BATCH selects the r4 default; -DUSE_YMM1_BATCH is gone (it is
the default; the define is accepted and ignored).

Why zmm2x2 loses on ICX despite 287-vs-505 instructions — and it is NOT
frequency: scaling_cur_freq sampled mid-run reads 3.3 GHz for BOTH kernels
(and for the 512-bit chain step). On ICX every 512-bit FMA/add/shuffle
dispatches on p0(+p1 fused)+p5 only, so zmm2x2's ~132 FMA-class ops/xform
floor at ~66 cycles on two ports with its broadcasts/stores/permilpd
competing for the same pair; ymm1's 256-bit mix fills p0/p1/p5 three-wide.
r4's "should win on the decode-bound node" prediction is falsified: the node
race and r4 board verdicts for zmm2x2 were inside the (large, 18% spread)
scoring noise, and under idle-node interleaved conditions the 256-bit loop
wins by 13%. Lesson for the record: on ICX, instruction count only wins if
the instructions are <=256-bit; a 512-bit "diet" kernel must get UNDER the
256-bit version's port-weighted uop count, not just its instruction count.

### Measured on the NODE this round (a80n0 leased core, interleaved, warm)
| cell | mine | MKL same core | r4 board (mine / MKL) |
|---|---|---|---|
| B=1, m=1 | **0.041-0.042** | 0.054 | 0.0451 / 0.0616 |
| B=512, m=1 | **0.044-0.046** | 0.043-0.044 | 0.0578 / 0.0514 |
| B=1, m=60000 | **0.110** | 0.270 | 0.1104 / 0.2371 |
| B=512, m=600 | **0.058** | 0.227-0.229 | 0.0740 / 0.2291 |

The one losing cell goes from 1.12x behind to ~1.03x — inside the ±4-5%
layout-noise floor race's wisdom established. Wallaby sanity (SPR, pinned
core 101): B=512 0.032 (same as the old ymm2 default — the ymm1 loop tied
ymm2 there in r4 already), B=1 0.028-0.029. No SPR regression.

Correctness (all on the node, driver + check.py): exec rel_l2 2.1-3.7e-16
for B in {1,2,3,5,8,13,512} at L=60 and B=3 at L=12/24/36; strict m=2 gates
4.3e-16 (B=1) and 4.9e-16 (B=512) vs tol 3e-14; chain gates 3.1e-15 (B=1
m=60000), 6.6e-13 (B=512 m=600), 4.1e-15 (B=5 m=100), 1.7e-14 (B=13 m=50),
plus L=12/24/36 chains, all vs tol 1e-10; output bit-identical across runs
(tryout repeatability check at B=1 and B=512). All flag variants
(-DUSE_ZMM2X2_BATCH, -DUSE_YMM2_BATCH, -DUSE_ZMM4, -DCHAIN_V1, -DEXACT_MAP)
build -Wall -Wextra clean.

### What did NOT work / was killed by measurement, with numbers
- **The 512-bit-license hypothesis for the B=1 chain** (rewrite chain step
  emissions/map to 256-bit for higher turbo): killed BEFORE writing it —
  3.3 GHz sustained mid-chain-run, i.e. no license drop to recover. The
  chain stays 512-bit; the Newton chains are latency, not clock.
- **Faster map via shallower Newton** (1 NR instead of 2): killed by
  arithmetic, not measurement — rsqrt14+1NR leaves ~5.6e-9 relative error
  per step against the exact-map reference; the B=512 chain gate already
  sits at 6.6e-13 of the 1e-10 budget, and a per-step 5e-9 injection has no
  contraction headroom to hide in. 2NR stays.
- First cold-session numbers on the leased core read zmm2x2 0.066-0.080 and
  MKL 0.052-0.054 — a fresh core/binary needs 2+ throwaway runs before the
  numbers settle (~+15-40% otherwise). ab.sh's round 1 is always discarded.

### Borrowed, explicitly
- d1_prime r3 / d1_batchlane r4: the /tmp squeue-shim workaround for the
  "dead-looking" live reservation, and the warning that tryout.sh's chain
  detection is broken at multi-case sizes (awk $1==l matches all four L=60
  lines; chained cells must be run manually with --chain/--map/--cin +
  check.py --map-check) — both used verbatim this round.
- d1_pow2 r2 (again): "on ICL every 512-bit shuffle lands on port 5, one of
  the two FMA ports" — r4 used it to justify the p5 diet; this round's A/B
  shows the diet was not enough and the port-weighted count is what decides.

### Next round
1. B=512 m=1 is at ~1.03x of MKL with an all-256-bit kernel at 3.3 GHz.
   The remaining 3% is inside layout noise; if the board still shows a gap,
   the only structural lever left is arithmetic below PFA's ~204 FMA/xform
   in 256-bit form, and I do not know of one. Consider the cell closed
   unless MKL's number moves.
2. If d1_batchlane ports the coset-row chain (they are welcome to — the
   generator recipe is in my r3 section), the B=1 chain margin narrows to
   layout luck; the wp[2][15] spill barrier idea (r3 item 1) is the only
   ~10% candidate left, now testable on the node in an afternoon.
3. Keep ab.sh + the variant builds under build/tryout/d1_composite_ab/ —
   interleaved same-core A/B on the real silicon settled in one session
   what three rounds of wallaby proxies and wisdom forensics could not.
