# GENERALIZE: from eight tuned points to an arbitrary-L 3D FFT library

You are one of a panel of implementers. The previous campaigns produced world-class
kernels for EIGHT fixed cube sizes (6, 8, 13, 17, 23, 36, 45, 64) — beating MKL, FFTW
and ducc0 by 2.7-9.7x per size and a 28-attempt rival AI committee overall. This
campaign builds what those campaigns proved possible: a LIBRARY —

    fft3d_plan* fft3d_create(int L, int batch);      // any 2 <= L <= 128, any B >= 1
    void fft3d_execute(plan, in, out);                // forward c2c double, batched
    void fft3d_chain(plan, x0, c, out, m);            // optional fused map chain

for any L, on x86-64 AVX-512 (Intel-validated: Cascade Lake / Ice Lake / Sapphire
Rapids). The deliverable is NOT a pile of per-L C files. It is:

1. a FACTORIZATION PLANNER: L -> candidate per-axis algorithm chains,
2. CODELET GENERATORS that emit specialized C for any chosen factorization,
3. a PLAN-TIME RACE that times the candidates on the running machine and persists
   per-host wisdom (the fft3d_best create()-race, generalized),
4. exact fp64 arithmetic passing the same two-part gate as always.

This is the method every winning attempt (ours and the rivals') converged on. You are
industrializing it.

## The algorithm classes and who owns them

Each implementer owns a CLASS, not a size. Your entry binary must `support()` exactly
the acceptance sizes your class covers (declining the rest), plus — from round 3 — ANY
size the driver asks for within your class.

| entry | class | acceptance sizes covered | seed material |
|---|---|---|---|
| gen_pfa_small   | PFA of coprime pairs, small     | 10, 12, 15, 20 | L6_pfa, L36/45 two-sweep, warm 3907 AoSoA |
| gen_pfa_large   | PFA of coprime pairs, large     | 40, 50, 100    | L45_pfa 9x5, gt-PFA n1_9 DAG |
| gen_pow2        | 2^k axes                        | 32 (and 16, 64, 128 unscored) | L8_radix8, L64_blocked |
| gen_powp        | p^k axes: CT within prime power | 25, 27, 50, 100 | NEW — the twiddle framework (see below) |
| gen_dense_prime | direct dense prime, p <= 31     | 31 (and helps 10..20 via 5,7-point modules) | L13_direct, L17/L23_matrixsimd, warm 00291a90 gen_a folded composite |
| gen_rader       | Rader for primes                | 31 (crossover fight vs dense) | L13/17/23_rader, warm 00291a90 gen_asm_prime |
| gen_bluestein   | Bluestein fallback, any L       | none scored; must run EVERYWHERE | literature 07 section 6.3 (know its 107-1315x warning: use only as existence fallback) |
| gen_batchlane   | SoA 8-vol/zmm batch-lane engine | 10, 12, 15 at B>=8 | L8_fusedaxes bl8, L13_rader soa8, warm d43251c2 gen.py |
| gen_planner     | factorization search + candidate enumeration | all (as a library layer others call) | — |
| gen_race        | plan-time tuner + wisdom cache  | all (library layer) | fft3d_best choose()/trial() |
| gen_twiddle     | exact-twiddle tables + accuracy audit | all (library layer) | FFTW accuracy notes: "inaccurate twiddles are the most likely reason for FFT inaccuracy" |
| gen_layout      | allocation/layout lib: stagger, 4K-avoidance, huge pages, pencil SoA | all (library layer) | warm prelude_c.py alloc_huge_st, ice notes |

Library-layer owners (planner, race, twiddle, layout) ship their code as headers/TUs the
class owners include; they are scored by ADOPTION — the monitor credits them when class
entries win using their layers. Everyone else links against the newest layer versions
each round.

## The twiddle problem is the campaign's center of gravity

Every prior winner DODGED general twiddles (PFA needs coprime factors; only L=64 used
them, in a fixed schedule). Arbitrary L forces general mixed-radix Cooley-Tukey inside
prime powers (25 = 5x5 with twiddles, 27 = 3x3x3, 100's 4x25 axis). Requirements:
- twiddle tables computed exact (long-double sincospi or argument-reduced), laid out in
  CONSUMPTION ORDER (ice lesson: no in-sweep gathers),
- the whole step must still pass 1.5e-14/step — budget your reassociations,
- table bytes stay trivial at these L (under 3 KiB per size was the fixed-size result);
  spend layout effort, not compression effort.

## The graded call and gates (identical machinery to the ice campaign)

Chained map workload per case (see cases.txt): z = FFT3(x) + c; x <- z/(1+|z|).
Timing: chain seconds, min over runs, spread reported; compile/plan/warmup excluded —
BUT plan time is now also reported (see budget below).
Gates, all three required, same as ice rounds 8+:
1. single call rel L2 < 1e-12 vs numpy;
2. TWO-STEP precision gate: fused chain path at m=2 within 3e-14 (1.5e-14/step — the
   precision contract; catches every shortcut, chaos-free);
3. chain-end within 300x the honest reference divergence measured on the same chain
   (floor 1e-10, {1,3}x10^n grid) — catches gross cheats only; honest drift forgiven.

## Plan-time budget

create() for a NEVER-SEEN (L, B) on a given host: <= 60 s including candidate
generation, compilation and racing. With persisted wisdom (results/wisdom_<host>.json):
<= 50 ms. The race must degrade gracefully: if compilation is unavailable, fall back to
the best precompiled candidate; Bluestein guarantees existence for any L.

## Scoring and the endgame

Rounds 1-5 score the acceptance suite in cases.txt against MKL 2022/2026, FFTW3
(three planners, plan_many), and ducc0 on the reserved Ice Lake node — per-size table,
time-weighted, geomean, worst-case, exactly as always. Development on wallaby via
./tryout.sh (leased node cores); implementers never submit slurm jobs.

**Round 6 is the surprise round**: the monitor draws THREE sizes never announced in any
brief or case file (composites and primes in 14..127), and scores the ASSEMBLED LIBRARY
(fft3d_general trunk = planner + race + all class winners), not your individual entries.
A size that fails to plan, races past the 60 s budget, or misses a gate scores zero for
the whole library. That is the test of generality — build for it from round 1.

Cross-architecture guard: every second round the monitor reruns the suite on Cascade
Lake and Sapphire Rapids (xarch_report.py). Those numbers are advisory (the score is
Ice Lake), but a kernel that wins only on Ice Lake is flagged, and the race must pick a
winning variant on all three — that is what the wisdom cache is FOR.

## Standing rules (unchanged from prior campaigns)

Timing after compile+warmup; multiple runs; batched and B=1 both matter; no FFT library
calls inside our transforms (libraries are baselines only); per-host builds under
build/$(hostname -s)/; maintain your STRATEGY.md every round; read the previous
campaigns' strategies, impl_N sources, and the rival corpora (fft_v4_solutions/,
fft_v5v6_solutions/, fft_warm_solutions/ — especially the generator pipelines, which are
the closest prior art to this campaign's deliverable).

## Cross-architecture advisory (from round 4)

After the round-4 and round-6 Ice Lake leaderboards, the suite is rerun on a Cascade Lake
node (severe 512-bit downclock, 1 MB L2), and once on Sapphire Rapids after round 5 —
reduced sampling, ADVISORY only (the score stays Ice Lake; absolute times never cross
machines). Read `XARCH.md` when it appears: a cell whose winner changes or whose
vs-library ratio degrades >25% on another Intel machine is a portability finding — the
fix is racing variants at plan time (the gen_race layer), not retuning for one host.
The wisdom cache is per-host by design; the advisory run populates a fresh one, which is
exactly what round 6's surprise sizes will do too.

## Baseline upgrade (round 3): fftw3_guru
FFTW now also competes through its guru SPLIT-ARRAY interface with a fused split chain
(deinterleave once per chain, exact map on split data) — the same layout and conversion
amortization your kernels use. "Best library" per cell may tighten from round 3 onward;
your target is whatever the strongest library configuration measures, not the friendliest.

## New literature (rounds 5-6): docs/literature/11-post2020-untested-ideas.md
A six-vein sweep of post-2020 ideas NEVER validated in fast software. Tier 1 is adoptable now:
dual-select FMA twiddle tables (free accuracy), flap-count factorization ranking (2,8-split-radix
is FMA-optimal), constant-per-site twiddle routing (compile-in ALL twiddles for a fixed L), and
GT/Rader-as-vectorization-first. Tier 2 names the structural plays for the weak large-L cells
(stage-as-matrix outer products, two-axes-per-pass fusion, transpose-free ordering). Being FIRST
to make any of these work in performant code is the point — cite the section in your strategy
record when you try one.

## ROUNDS 7-8: spend the queued backlog

The campaign's six scored rounds ended at 3.49x over the best library, 11/11 cells. Your own
strategy records queued the heaviest literature plays without spending them. This extension
exists to spend them:

1. **Two-axes-per-pass fusion at L=100/50/40** (literature 11 Tier 2) — both gen_pfa_large and
   gen_powp records name it as "still the" next lever. L=100 sits at 1.71x, the weakest cell;
   its working set (32 MB in+out) is the one case that spills L3, so a fused y*z pass with
   L2-resident tiles is worth real percent.
2. **Constant-per-site twiddle routing** (Garrido, lit 11 Tier 1) — never attempted. For a fixed
   L the angle-sorted schedule computes OFFLINE; twiddles become compiled-in broadcast constants.
   Try it on one mid-size cell (25/27/32) where twiddle loads are measurable.
3. **Stage-as-outer-product / register-resident stage matrices** (lit 11 Tier 2) — fold twiddles
   into per-stage constant matrices applied by broadcast-FMA. Natural first target: the L<=16
   dense-GEMM crossover claim at 10/12/15.
4. **REFFT-style plan enumeration in gen_race** — tree rotations + codelet permutations as a
   complete plan lattice; memory order tuned independently of arithmetic order.
5. Protect every cell: the two-part gate is unchanged; a regression that fails the gate scores
   nothing regardless of speed.

The surprise-size test (L=21, 44, 96 vs the trunk) is being scored as you start — its findings
land in this brief as an addendum; treat any surprise-size failure as a planner/race bug with
priority over per-cell tuning.

### SURPRISE-TEST ADDENDUM (Aug 25, 09:05 — the r6 test, run properly)
Three sizes never named in any brief or case file — L=21 (contains prime 7: no panel ever
built a 7-point module), L=44 (prime 11: same), L=96 (2^5*3) — were scored against the
assembled trunk (gen_planner/gen_race), cold, full gates:
  L=21: trunk 0.2684 s vs best library 0.5346 (mkl) -> 1.99x WIN, plan 0.04 s
  L=44: trunk 0.3690 s vs 0.4761 (mkl)             -> 1.29x WIN, plan 0.06 s
  L=96: trunk 0.3456 s vs 0.5382 (mkl)             -> 1.56x WIN, plan 0.60 s
All one-step gates exact, all chain drift inside the honest anchor band; gen_bluestein's
existence fallback also ran correctly everywhere. Plan budget: worst case 0.60 s cold vs the
60 s allowance. THE LIBRARY GENERALIZES. Rounds 7-8 implication: the never-built primes are
now measured — a real 7-point and 11-point module (dense folded, lit 11 GT/Rader-first) would
lift L=21/44-class cells from 1.3-2x toward the 3-4x the built classes achieve.

## New tools (round 8): static microarchitecture analyzers — read tools/TOOLS.md
llvm-mca (LLVM 22, icelake-server model), uiCA (uops.info ICL model), and OSACA are now
installed. Per-port dispatch counts and cycles/iteration for any loop, deterministically,
without burning lease slots. Choose schedules with the models; SCORE with the node.

### Tools addendum: HARDWARE COUNTERS are live (Aug 25)
/tmp/perf on a80n0 with tools/pmu.sh — per-port uops, L1 traffic, license-throttle cycles,
measured not modeled. See tools/TOOLS.md section 4.

## ROUNDS 9-10: the counter-directed rounds (read results/PMU_AUDIT.md FIRST)

The PMU audit measured where the remaining time lives. Four avenues, in value order:

1. **Bank the picks (gen_race + every engine with an internal tuner).** The L=25 "regression"
   was plan-time pick instability: r7 and r8 binaries are counter-identical, but a lucky
   create() once picked a path worth 25% (0.1265 vs 0.1681) and nothing persisted it. Route
   every engine-internal pick through the wisdom cache with NOISE-GATED storage (store only
   verdicts whose trial spread is tight; re-race, never trust, a noisy trial). Recovering
   L=25's 0.1265 deterministically is the round's cheapest big win. Prove determinism:
   5 consecutive create() cycles must pick identically.
2. **Two-axes-per-pass fusion at L=100/50/40.** Now proven traffic-bound with numbers:
   L=100 moves ~4x the algorithmic minimum through L1 (2.34G line fills) at only 0.82/cycle
   FMA dispatch; L=50 pushes 77 GB into L2 at 1.07/cycle. Fuse y*z into one L2-resident
   pass (lit 11 Tier 2). Success metric is the COUNTER, not just time: l1d.replacement per
   chain step should drop ~2x. Use tools/pmu.sh before and after.
3. **The champion signature is your dashboard.** gen_rader at L=31 runs 1.60/2.0 combined
   p0+p5 dispatch (IPC 2.15) — that is what done looks like. For your cells, measure
   uops_dispatched.port_0+port_5 per cycle: below ~1.1 with high l1d.replacement means
   traffic headroom; near 1.6 means move on.
4. **Port 1 idles in every kernel** (0.7-2.0G vs port 0's ~10G): 256-bit FP dispatches
   there. Independent side work — map tails, twiddle prep, B%8 remainder volumes as a
   ymm lane-pair — can co-issue nearly free. Also: a 4-lane SoA variant would unlock
   batch-lane layout at L=50 (B=4), where the 8-lane form cannot run.

PMU is live from round start: tools/pmu.sh (if /tmp/perf is missing, the node rebooted —
tell the monitor; re-staging needs one scp but paranoid=2 also needs re-setting by Will).

## ROUNDS 11-12: ALL HANDS ON L=100 (the large-size rounds)

The board is converged everywhere except where it matters most for big problems: the
large cells (40: 2.53x, 50: 2.27x, 100: 1.73x — the weakest, and the only cell whose
working set spills L3). EVERY implementer works the large-size problem this round,
from your own class's angle; cross-class entries at L=40/50/100 are explicitly
encouraged — the race picks winners, your credit is in your strategy record.

**Counters are live from the start** (tools/pmu.sh on the scoring node; both Ice Lake
nodes are PMU-enabled). The protocol is mandatory: baseline counters BEFORE you change
code, counters after, both in your strategy record. The success metrics at L=100:
  - l1d.replacement per chain step (baseline ~100M line-fills/chain = ~6.4 GB through
    L1 for a 32 MB problem: ~4x the algorithmic floor);
  - p0+p5 dispatch per cycle (baseline 0.82; the champion cells run 1.6);
  - and settle the OPEN DISAGREEMENT: gen_pfa_large's r7 accounting says the engine is
    uop-saturated on this host; the audit says 0.82/cycle is headroom. Measure TOTAL
    vector dispatch (p0+p1+p5+p2_3+p4_9 per cycle) against the ~2.1 uops/cycle cap the
    Ice Lake notes describe — whoever settles this with counters decides the round.

Approach menu (all measured-or-published, none tried here; pick your lane):
  1. Two-axes-per-pass fusion: y*z as one L2-resident pass (lit 11 Tier 2).
  2. Transpose-free column-order passes (MDFFT lineage, lit 11 / staging 04).
  3. Order-p matmul passes with the FLOP<->I/O dial (FlashFFTConv shape): fewer, bigger
     DFT-matrix multiplies against register/L2-resident factors.
  4. Within-volume SoA at B=1: 8 PENCILS per zmm lane-slot instead of 8 volumes — the
     batch-lane trick without a batch (nobody has tried it; batchlane owner, this is you).
  5. Layout: THP verification at 32 MB (smaps!), stream-count tiling, 4K-stagger audit of
     the three passes (layout owner).
  6. Port-1 co-issue: the map's arithmetic as 256-bit work overlapping the FFT's 512-bit
     passes (audit finding 4).
Wins at 100 should transfer to 50/40 (same regime, smaller): validate there too. Gates
unchanged. Model with mca/uiCA, MEASURE with pmu.sh, SCORE with the node.

## Standing rule (added after the ~/bin/squeue incident)
NEVER install anything into ~/bin, ~/.local/bin, dotfiles, or any user-global path — it
shadows commands for the human operator too. Host-specific workarounds (like the wallaby
squeue heartbeat shim, which now lives in bench/gen/wallaby_shims/ and is on YOUR PATH
automatically) belong inside the harness tree, documented in your strategy record.

## ROUND 13: close the B=1 small-L gap (quick fix round)

benchFFT (Frigo & Johnson's community harness, now wired at benchfft_ours/) exposed the
one place FFTW still beats this library: SINGLE-VOLUME transforms at tiny L.
Measured on the node (their 5N log2 N mflops convention, B=1):
    L=10: ours 13,440 vs fftw3 23,842  (they win 1.77x)
    L=12: ours 17,112 vs fftw3 29,724  (they win 1.74x)
Cause: our small-L engines are batch-lane designs; the B=1 fallback paths are weak.
TWO NEW SCORED CELLS this round: 10:1:16384 and 12:1:12288 (in cases.txt) — the sweep
now sees what benchFFT sees. Targets: beat fftw3_measure at both (its equivalent chain
pace: ~all entries can compute it from the board's fftw3 rows).

Known-good material:
1. **Within-volume pencil lanes** — the r11 L=100 winner's trick at small scale: 8
   z-pencils of ONE volume per zmm lane-slot (L=10: 100 pencils = 12 lane-groups + tail;
   L=12: 144 = 18 exactly). Same zero-shuffle steady state as batch-lanes, no batch needed.
2. **genfft small codelets** — sota/codelets/n1_10.c and n1_12.c (3 KB straight-line
   DAGs, FFTW's own generator) are in-tree and license-compatible; mine or adopt.
3. FFTW's own small-codelet scheduling is the benchmark: study what 24-30k mflops
   requires — it is NOT memory-bound at 16-27 KB working sets; it is pure schedule.
Everyone else: protect your cells (the full suite still scores). Verify with
benchfft_ours/doit --verify AND the standard gates. Racing: gen_race must route the new
cells' B=1 correctly (batch%8 fallbacks must not silently pick the slow path).

## ROUND 14: harden the WHOLE B=1 curve via the execute() reroute
benchFFT (Frigo-Johnson, single-shot fft3d_execute) is the B=1 metric. r13 fixed the CHAIN
path at 10/12 (2.2-2.7x over libs in the sweep) but execute() stayed on the old r6-sandwich
path and LOSES to fftw3 there (13.3k/17.3k vs 23.8k/29.7k mflops). The gap is systemic, not
local: the current execute() B=1 curve wins elsewhere but only thinly where our chain path
dominates -- L=15 1.20x, L=32 1.32x, L=64 1.16x, L=128 1.03x -- the SAME plumbing seam.

Mission: route fft3d_execute() through the fast within-volume engine the chain path uses,
ACROSS THE WHOLE L RANGE, and beat fftw3 at every B=1 size. The fix is plumbing (execute()
and chain() share the fast engine), named already in gen_pfa_small's r13 record; not new
kernels except where a size never had a B=1 fast path.

Metric & acceptance: benchfft_ours/doit --speed cof<L>x<L>x<L> for L in
10,12,15,16,20,25,27,31,32,40,50,64,100,128 -- MUST beat fftw3 (must-win: 10=23842,
12=29724, the two we currently lose; must-not-regress: all others), plus --verify and the
standard gates on the batched suite. Expectation by regime: compute-bound small/mid
(10-32) gain most from the schedule reroute; large (64-128) are partly B=1 memory-bound --
firm the thin wins, do not expect leaps; use tools/pmu.sh to tell which cell is which.
gen_race + every engine owner: make execute() and chain() share one fast path at B=1.
