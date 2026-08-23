# L36_pfa — strategy record (ICE LAKE panel)

Lineage: this entry is the geom-panel L36_pfa, developed over rounds panel_r2..panel_r11
on Cascade Lake (Gold 5218). That full history — the Good-Thomas 4x9 structure, the n1_9
DFT9 DAG, the prefetch taxonomy pf=0..7, the tuner/hysteresis design, and every dead end —
lives in `bench/geom/strategies/L36_pfa.md`. Do not re-derive it; read it.

## Round ice_r1 (no work done — worker crash)

The ice_r1 agent for this entry died at startup (Bun segfault before its first tool call;
see `results/ice_r1/agents/L36_pfa.log`), so ice_r1 scored the UNMODIFIED geom r11 code on
the new node. It still took first at L=36: 119.163 us/transform (graded chain B=8 m=64),
0.3% ahead of L36_mixedradix (119.530), 1.35x ahead of MKL (160.688). The r8-r11 probes ran
on the ice node and are on the r1 board: p1=63.1 p1z=28.2 p1y=19.9 p2w=20.8 fu=81.0 us at
nv=1 steady state, tuner pick pw=4 mode=inplace pf=0 — same shape as on CLX. No strategy
record was written; this file starts here.

## Round ice_r2 — transpose-free phase 1 (tr=1): lane transposes moved off port 5

### The lever

New machine, new binding resource. On the graded node (Xeon Gold 6326, bare-metal ICX-SP)
the second 512-bit FMA pipe is on PORT 5 — the same and only port that executes 512-bit
shuffles. On CLX (one FMA pipe, port 0) this file's TRNC transposes and codelet SWAPs rode
an otherwise idle port 5 for free; on ICX every shuffle displaces an FMA issue slot.

Counts per volume at PW=4 (FMA-class F, port-5-only shuffles S):
  F = 225,504   (unchanged since r10's n1_9 DAG)
  S = 46,656 TRNC shuffles (phase-1 lane transposes) + 55,404 codelet swaps = 102,060
  two-port floor (F+S)/2 = 163,782 cyc = 56.5 us at the node's sustained 2.90 GHz
  (no AVX-512 downclock on this die — every r1 board entry that measured clk512 got 2.90).

The fix comes straight out of corpus §10's bare-metal proviso: ICX-SP does 2x64B loads/cyc
and **folds broadcasts into the load uop for free** (that was the VM tier's missing
feature). `vbroadcastf64x2 (mem), zmm{k}` is ONE load-port uop, zero port-5 involvement.
So tr=1 phase 1 builds every lane-transposed vector as 4 masked 128-bit broadcast loads
and drops BOTH TRNC passes:

  z-subloop: LD(n) = 4 broadcasts straight from the four in-plane y-rows (lanes = y);
             codelet output stored UNtransposed to pl[ygroup*36+kz], lanes = y,
             as a plain full-width store (was: codelet -> second TRNC -> pl[y][kz]).
  y-subloop: LD(n) = 4 broadcasts re-gathering pl's 128-bit granules into lanes = kz
             on the fly (element (y,kz) sits at pl[(y>>2)*36+kz], lane y&3);
             mid stores unchanged. Phase 2 was already shuffle-free apart from swaps.

Port arithmetic per yb-group: p5-only 201 -> 57 ops; p0/p5 bound 217 -> 145 cyc; the 144
broadcast loads cost 72 load-port cycles, comfortably under the 145 bound (load ports were
near idle). New floor (225,504 + 55,404)/2 = 140,454 cyc = **48.4 us**.

Codegen verified on the .o before ever touching the node (gcc 11.4, -march=icelake-server,
objdump): tr=1 plane body = 288 vbroadcastf64x2 all with memory operands, kmovs hoisted
(3 total), shuffle-class instructions 258 -> 114 = exactly the codelet swaps. The
4-deep merge-mask dependency chain per vector is covered by ~36 independent vectors in
flight.

### Integration

Plan dimension `tr` (PW=4 only — PW=2's TRNC is 2 shuffles/2 vecs and pw2 loses every
tournament anyway): tr=1 twins for pw4 x {inplace, scratch} x pf {0,1,2,7}, instantiated
via the same always_inline + literal-constant-flag specialization as r9 (phase1_tr1_pf0 is
the scored body). tr gets EQUAL cand_rank to its tr=0 twin: kernels compete on raw min
time; the 3% hysteresis band keeps guarding prefetch knobs only. FFT36_TR forces either
kernel (and disables the small-nv list restriction, like the other overrides). Probe p1t
(tr=1 phase 1 alone, nv=1) joins the description string next to p1. Every pw2/NT/pipe/
streaming path is byte-for-byte untouched.

### Measured (tryout.sh = leased core on a80n0, graded chain m=64 --unitary; conditions
are noisier than the monitor's quiet window — r1's scored 119.163 measures ~136 here)

  old build (r11 code)  B=8: min 136.155 us/xform   MKL same run: 210.6 (sd 4.9%, noisy)
  new build             B=8: min 120.825 us/xform   MKL same run: 168.8 (sd 0.03%)
                        B=1: min 106.287 us/xform   MKL same run: 138.4
  correctness: rel_l2 = 3.586e-16 (B=8), 3.591e-16 (B=1); chain check 1.26e-14
  (tol 8e-12); output bit-identical across runs.

The honest A/B is the tuner's own arena (same process, same buffers, interleaved timing):
  pw4-inplace-pf0  tr=0: 111.4 us/vol   tr=1: 102.8 us/vol   -> tr=1 wins by 7.7%
  tuner pick at B=8: pw=4 mode=inplace pf=0 tr=1 (nc=21, all candidates pass correctness)
Expected scored number: ~110-112 us/transform if the 7.7% carries to the quiet window.

### What did not work (with the number that killed it)

- Prefetch on top of tr=1, all flavors, at the graded B=8 chain: pf=1 107.3, pf=2 109.6,
  pf=7 108.9 vs pf=0 102.8 us/vol. The chain's working set (12 MB in+out at B=8) is
  L3-resident and the in-read is one linear HW-prefetchable stream — software prefetch
  remains pure uop tax here, now six rounds running across two microarchitectures. Stop
  re-proposing prefetch for the cache-resident cells.
- (Considered and rejected on paper, not fielded: extract-store for the pl write side —
  vextractf64x2-to-mem is store-port only, but ICX commits just 1x64B store/cyc, so 144
  128-bit stores per group would cost 72 store-port cycles against 36 for the full-width
  stores tr=1 keeps. Broadcast-on-load is strictly better than extract-on-store here.)

### Borrowed

- Corpus §10 (Ice Lake dossier): the bare-metal "broadcasts fold into the load uop"
  proviso and the 2-FMA-pipe port map — the whole basis of tr=1.
- L17_matrixsimd (ice panel): its "extract-store" note pointed at the
  shuffle-vs-memory-port trade; I took the direction but landed on the load side.
- r9's compile-time exec-variant specialization (originally from L45_pfa r8), reused
  verbatim for the tr=1 pf=0 body.

### Next

1. Codelet swaps are now the entire port-5-only term: 55,404/volume = 19.1 us of p5
   minimum against a 48.4 us floor. The only structural attack is split-complex (SoA)
   lanes — kills every SWAP and every VPAIR sign constant, at the cost of a full layout
   rework of both phases (the corpus's 1000f989 "zero-shuffle passes" is exactly this,
   and L64_radix8 already runs split-complex here). Big job; right shape for a dedicated
   round.
2. Read the next leaderboard's p1t probe: nv=1 phase-1 split predicts p1 63.1 -> ~48-52.
   If p1t lands well above that, the residual is load-port/L1-fill, not port 5.
3. The gap from the ~48 us floor to ~103 us in-arena at B=8 is L2/L3 round trips the
   chain forces (746 KB in-read + RFO + writeback per volume, buffers cycle through L3)
   plus the driver's own unitary-scale pass (~1.5 MB RMW per volume, identical for all
   backends, not removable). If anything is left on the table at L=36 it is there, and
   in-place + pf=0 has beaten every alternative structure on both nodes. Treat further
   memory-mode candidates as low-yield.
4. If a future round adds streaming-batch cases for L=36, wire tr=1 into M_PIPE/NT
   before trusting any tr comparison there (currently tr=0-only paths).

## Round ice_r4 — fft3d_chain: the map moves inside the passes

### The task changed

The graded step became `state <- (z+c)/(1+|z+c|)`, z = RAW (unnormalized)
FFT(state), no driver-side unitary scale, and the driver times an exported
`fft3d_chain(plan, x0, c, final_out, m)` weak symbol as the whole m=64-step
unit.  Without it you pay fft3d_execute + a driver-side map pass (MKL,
fallback-only, measured 282-300 us/t on the graded cell in my windows —
r3 it scored 160.7).  fft3d_execute is now correctness-only; everything
this round is the chain.  (There was no ice_r3 section here: the r3 worker
made no changes to this entry; r3 scored the r2 code at 111.568 us/t, 2nd
behind L36_pencilfused 109.619.)

### What was built

1. **Per-volume chaining**: volume b runs all 64 steps before b+1 is
   touched — state T (746 KB), this volume's c and the mid buffer stay
   cache-resident instead of sloshing 11.4 MB through L3 per step.
2. **Lazy map fused into phase 1's loads** (corpus S10 S2; the rivals'
   winning shape): between steps T holds RAW z; phase 1 applies
   map(z+c) inside its tr=1 broadcast-load side and feeds the mapped value
   straight into the z/y transforms.  Each point mapped exactly once;
   divides issue 4-per-DFT4 inside PFA36's stage A.  Step m instead fuses
   the full map into phase 2's store, straight into final_out — no
   epilogue pass, x0 never written.
3. **Map arithmetic** (11 FMA-class ops + 1 SWAP + 1 vdivpd per zmm):
   w = z+c; |w|^2 = VFMA(w,w,1e-300)+SWAP (the additive bias replaces both
   the vmaxpd NaN guard for rsqrt14(0)=inf AND the denormal trap);
   vrsqrt14pd double seed + 2 Newtons (6.1e-5 -> 5.6e-9 -> 4.7e-17; ONE
   Newton fails the 1e-13/step budget, do not tier down); d = VFMA(mm,y,1)
   (saves an add); one exact vdivpd.  Chain drift measured **1.216e-14
   (B=8) / 1.182e-14 (B=1) vs tol 6.4e-12** — 530x margin.
4. **The split variant (sp=1, THE SHIPPED PICK)**: phase 2 maps ODD
   x-planes at its store (c read full-width strided, raw layout), the next
   phase 1 maps EVEN planes at load.  Halves the divider demand per phase:
   phase-1 divider drops 187k -> 93k cyc/vol against ~56 us of phase-1
   port work.  Also built mix=1 (every 4th map vector on a vrcp14pd+2N
   ladder instead of the divide) — stable but ~2% behind sp=1.
5. **c pre-transposed (plan->ct)**: phase 1's map c-operand is ONE aligned
   full-width load in exact consumption order instead of 4 masked
   broadcasts (-108 load uops/group).  Rebuilt only when the caller's c
   pointer changes (lands in the driver's warmup).  ct uses a PADDED plane
   stride (+192 B, planes drift 448 B mod 4096).
6. **Hugepage-backed plan buffers** (mmap 2MB-aligned + MADV_HUGEPAGE,
   posix_memalign fallback; adopted from L64_blocked): fewer dTLB entries
   for the 1.5 MB steady loop AND deterministic mutual mod-4K offsets for
   everything the plan owns (S +576 B, T=S1 +1152 B, ct +0 with its padded
   stride) — see the lottery below.
7. **create() races** {inplace,scratch} x {mix} x {split} short chains
   (m=10, nv<=2) against an exact-map reference chain (pw2 FFT + scalar
   sqrt/div) — a candidate must pass rel L2 < 1e-12 before it can be
   timed; if nothing validates, fft3d_chain runs the exact reference.
   FFT36_CHM/FFT36_CMIX/FFT36_CSP force it.  fft3d_execute and its whole
   tuner are byte-for-byte untouched (single-transform fingerprint still
   3.586e-16).

### Operation count (per volume-step, PW=4)

FFT unchanged: 225,504 FMA-class + 55,404 swaps = 140.5k two-port cycles.
Map: 11,664 zmm x (11 FMA + 1 swap) = 70.0k two-port cycles, + 11,664
vdivpd x ~16 cyc = 187k divider cycles (own unit; the round was about
keeping it hidden).  Two-port floor ~210k cyc = 72.6 us/step at 2.9 GHz.

### Measured (tryout = leased core on a80n0; same-window contrasts only)

- **SHIPPED: B=8 min 129.4-130.4 us/step-transform (sd 0.07-0.13%) across
  4 windows; B=1 129.3-129.7.**  Pick chm=0 (inplace, T-only) mix=0 sp=1.
  MKL same case/core through the fallback: 289-300 us/t (~2.2-2.3x).
  Chain + single outputs bit-identical across runs.
- Variant ladder (forced, interleaved same-window): sp=1/mix=0
  {128.9, 130.4-133.1} ~ sp=0/mix=1 {129.8-133.8} > sp=1/mix=1 131.8
  >> scratch (chm=1) 157.4-159.2 (+18%: the +746 KB mid pushes the steady
  loop from 1.5 to 2.2 MB against 1.25 MB L2) >> the fallback path (never
  shipped): my r3 code through the driver map would sit ~160+.

### The trap that decided the defaults: a buffer-address LOTTERY

sp=0/mix=0 (all 36 divides per group in phase 1) is BIMODAL across
processes running the identical installed variant: 129.6 vs 172.3 us/t
(and 133 vs 168 in other windows) — stable within a process, a coin-flip
across processes.  mix=1 (same loads, 3/4 of the divides) and sp=1 (half)
never showed the mode in ~10 processes each.  Mechanism consistent with
all observations: 4K-alias false dependences between loads and in-flight
stores (offsets vary with per-process mmap placement) insert latency into
the divide inputs; when the phase-1 divider has NO slack (all-divide:
187k cyc demand vs ~164k cyc of other work), those stalls become
unrecoverable divider bubbles; with slack they are absorbed.  The create
arena CANNOT price this (its buffers are not the driver's; it read
pcc=133.9 for sp0/mix0 while the same process's timed region ran 168.5).
Consequences shipped: sp0/mix0 is NOT in the candidate list at all, and
every plan-owned buffer is hugepage-anchored + deliberately skewed so at
least the plan-vs-plan offsets are deterministic.  Driver-buffer offsets
remain per-process; the shipped shapes are insensitive to them.

### What did NOT work / other numbers

- Scratch mid (chm=1): +18% every window.  Working set beats aliasing
  hygiene at this size; in-place won exactly as it has since r5.
- First run of a session reads +13% (146.8-147.9 vs 129.9 immediately
  after, same binary, same pick) — schedutil ramp on a cold core.  Min
  across the monitor's independent processes absorbs it; do not chase.
- tryout.sh is broken this round (L23_matrixsimd found it first): set -u
  kills it reading $W before it is set — invoke as
  `W=<ice>/build/tryout/L36_pfa ./tryout.sh L36_pfa 36 8`; and the remote
  check.py gets a literally-unexpanded '$W/c.bin', so the map-chain check
  always dies inside tryout — run check.py yourself afterwards and do the
  two-run cmp manually.
- gen_input.py needs numpy, absent on the node — generate inputs on the
  login side; beware NFS attribute-cache lag before the node sees them.

### Borrowed this round

- Map shape (rsqrt14 double seed + 2 Newtons + ONE exact vdivpd), the
  d=fma(m,y,1) merge, divider-placement doctrine, vdivpd ~16 cyc/zmm on
  this node, and both tryout bug workarounds: **L23_matrixsimd ice_r4**.
- 1e-300 additive bias replacing NaN-guard + denormal trap, per-volume
  chain order, "do the budget arithmetic, don't tier down": **L17_winograd
  ice_r4** (and PANEL_BRIEF's per-(L,m) arithmetic — at m=64 the answer is
  "2 Newtons, exact divide", same as theirs).
- Hugepage mmap + MADV_HUGEPAGE + touch, with posix_memalign fallback:
  **L64_blocked** (their panel_r7 mechanism, reused for chain buffers).
- Lazy map, 4K-aliasing hygiene, per-volume residency: corpus S10 S2/S3.

### Next

1. The step is ~130 us against a ~73 us two-port floor; the ~40 us gap is
   the memory system (T rw + 750 KB of c per step through a 1.25 MB L2,
   plus the phase2-store -> phase1-load junction).  The one structural cut
   left: SPLIT-COMPLEX lanes (kills the 55k FFT swaps = ~9.5 us of the
   floor AND frees pipe slots the map's Newton ladder competes for — the
   L23 record makes the same compounding argument).  Still the right
   dedicated-round job.
2. Finer split granularities (sp maps planes 2-of-3? k mod 4?) to tune
   divider balance — cheap A/Bs off the existing mapx machinery.
3. If the scored number lands well above ~130, suspect the first-process
   ramp or a driver-buffer draw; check the monitor's per-process spread
   before touching code.
4. The chain race's arena mispredicts aliasing-sensitive contrasts by
   construction.  Keep it as a correctness gate + gross ranking only;
   never re-admit a divider-saturated shape on its say-so.
