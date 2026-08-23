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
