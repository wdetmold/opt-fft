# L64_radix8 — strategy record

Geometry: **L = 64** (64³ = 262144 complex doubles = **4 MiB per volume** — does not
fit L2 on any machine here; B=1 working set in+out+scratch ≈ 12.5 MiB is L3-resident
on the node, batched streams from DRAM).
File: `impl/L64_radix8.c`.  `fft3d_name()` = `L64_radix8`.

---

## Round panel_r6 (first implementation — this geometry is new this round)

### Technique

64 = 8×8, so every axis is **two radix-8 stages with w64 twiddles between them**
(Cooley–Tukey N1 = N2 = 8, DIT: j = j1 + 8·j2, k = k2 + 8·k1;
X[k2+8k1] = DFT8_{j1}( w64^{j1·k2} · DFT8_{j2}( x[j1+8j2] ) )).  The radix-8 module is
the panel's proven **52-instruction / 56-flop codelet** (44 add/sub + 8 FMA, natural
order in and out, only irrational constant C = 1/√2, every ±i free in split-complex) —
taken verbatim from the L8_radix8 record (which took the FMA form from L8_batchsimd r1).
One generic macro instantiates it for `__m512d` and for scalar `double` (the portable
fallback is the same algorithm, same twiddle tables, scalar).

**Layout invariant: the SIMD lanes always hold 8 adjacent z (later k2z) values.**
That makes the y-FFT and x-FFT fully elementwise — 64-point FFTs *across vectors*, zero
shuffles in any butterfly — at the cost of one transpose pair in the z-FFT only.

Split-complex scratch volume SC, slot (x, ky, zb) = `x*SCXS + ky*SCKS + zb*16` doubles
(re vector at +0, im at +8), **both strides padded to an odd number of cache lines**
(SCKS = 136 doubles = 17 lines; SCXS = 64·136+8 = 8712 doubles = 1089 lines).  See the
padding A/B below — this is worth ~25%.

Two passes over the volume (the third is fused):

* **Pass 1 (y-FFT), per (x-plane, z-octet):** deinterleave fused into the stage-1 loads
  (one `vpermutex2var` per vector — measured free next to the FP work); stage 1 does
  DFT8 over y = j1+8t for each residue j1, twiddles by broadcast w64^{j1·k2} from a
  64-entry scalar table, through an 8-KiB L1 line buffer; stage 2 does DFT8 over j1 and
  stores split vectors to SC.  Input reads are 128-B chunks at 1-KiB stride inside a
  64-KiB plane (L2-resident after first touch).  When batch > 1, a variant additionally
  prefetches the **next x-plane** into L2 (`prefetcht1`, 16 lines per stage-1 iteration,
  spread — the L8_fusedaxes fill-buffer-friendly placement, not a burst).
* **Fused pass 2+3, per ky:** first the 8 x-line groups — 64-point FFT over x with
  loads at SCXS stride (the unavoidable transpose-like sweep), **in place in SC**
  through the line buffer (legal: stage 1 consumes all 64 slots into the buffer before
  stage 2 writes back); `prefetcht0` of the next zb column (+16 doubles) on every load.
  Then immediately the 64 **z-lines** of that ky-slab, which is still **L2-hot** — this
  fusion replaced a separate third full-volume sweep and bought ~10% in both regimes.
  z-line (z = 8g + lane): one codelet over the 8 zb register-pairs (g → k2), lane
  twiddle w64^{l·k2} from 14 vector tables (loaded per use, NOT hoisted — see spill
  note), the 24-op non-destructive 8×8 transpose pair (L8_fusedaxes network; its
  SW = swap-lane-bits-1,2 residue is absorbed into the final interleave index vectors
  ILO = {0,8,1,9,4,12,5,13}, IHI = {2,10,3,11,6,14,7,15}), second codelet (l → k1),
  interleave, and a contiguous 1-KiB store of the output row (kx,ky).  Output rows go
  out ky-major (64-KiB stride between consecutive rows; each row is 16 full lines, so
  NT stores stay fill-buffer-clean).
* **Store type (plain vs `vmovntpd`) is self-tuned in `fft3d_create()`** at (a clamp
  of) the real batch size, L8_radix8-style: 3 interleaved rounds, warm-up pass per
  candidate, min of 2 timed executes, NT must win by 2%.  The pick is reported through
  `fft3d_description()` ("tuner pick[B=…]=nt|plain").  Note at B=1 the driver never
  reads `out`, so NT can legitimately win there too by not polluting L3 — let the
  machine decide.

SC is a 4.5-MiB anonymous mmap with `MADV_HUGEPAGE` (kills TLB pressure of the 64-KiB
strided column walks; ~1024 4-KiB pages otherwise).

### Derivation / operation count (per volume)

Lines: 3 axes × 64² = 12288 lines of 64 points.  Per line: 16 codelets × 56 = 896 flops
+ 49 nontrivial twiddles × 6 = 294 → **1190 flops/line**, against split-radix's
4·64·6 − 6·64 + 8 = 1160 (Yavne) — within 2.6%, and the radix-8² form vectorizes with
zero butterfly shuffles, so there is nothing meaningful left in the arithmetic.

Vector instruction bill: 24576 codelets × 52 = 1,277,952 FP + 78,848 vector cmuls
(2 mul + 2 FMA each) = 315,392 FP → **≈ 1.59 M vector FP instructions**.  Shuffles:
65,536 deinterleave + 4096 × (48 transpose + 16 interleave) = **328 K** (p5, hides
under FP).  Loads ≈ 0.5 M, stores ≈ 0.4 M (two ports / one port — not binding).
Scalar flops ≈ 14.8 M real (driver's nominal 5N·log₂N = 23.6 M, so the leaderboard
GF/s over-reads our true rate by ~1.6×, uniformly).

**Node floor estimate:** the Gold 5218 issues one 512-bit FP op/cycle → ≥ 1.6 M cycles
≈ **550 µs at 2.9 GHz** for B=1 if perfectly memory-hidden.  Batched: 8.4 MB/volume of
mandatory DRAM traffic (in + NT out) at the node's ~12 GB/s single-core ≈ **700 µs**
floor — the batched cells will be bandwidth-bound there, unlike wallaby.

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4, tryout.sh, best of ≥3 runs)

Wallaby was strongly bimodal all session (~1.7× between states, e.g. B=1 543 vs 995 µs
across back-to-back runs; sd within a run ≤1%).  Every cross-variant decision below was
therefore made with **in-process A/B harnesses** (same process, alternating variants),
not by comparing tryout runs.  Numbers here are best-of-runs, per transform:

| B | this round | MKL same session | ratio |
|---|---|---|---|
| 1  | **542.9 µs** | 667.2 µs | 1.23× ahead |
| 8  | **512.0 µs** | 845.0 µs | 1.65× ahead |
| 64 | **587.5 µs** | 1624.4 µs | 2.76× ahead |

Correctness: PASS at B = 1, 2, 3, 8, 64; rel_l2 = 4.16–4.46e-16, rel_max ≤ 5.9e-16;
bit-identical re-runs everywhere (tryout's repeatability check).  The portable scalar
path (`-mno-avx512f`) PASSES at 3.3 ms/volume.  Builds warning-free under
`-Wall -Wextra` at native (SPR) and generic x86-64.  Setup 0.02–0.11 s.

**Padding A/B (this is the round's transferable finding, relevant to L64_blocked):**
compiling with `-DSCKS=128 -DSCXPAD=0` (all strides exact powers of two) costs
**~25% in BOTH regimes** — B=1: 683 vs 543 µs fast-state (and 1210 vs 907 slow-state);
B=64: 719 vs 587 µs/vol.  At L=64 the natural x-column stride (64 KiB ≡ 0 mod every
set stride) maps the x-FFT's 8-load groups into a single L1 set, exactly Bailey's
worst case from corpus §04.  Odd-line padding (17-line ky rows, 1089-line x planes)
restores the sets for ~4% memory overhead.  This settles the question the L64_blocked
stub was asked to measure, at least for this pass structure.

### What was tried and did NOT work (numbers from in-process A/B on wallaby)

* **Sequential plane copy + deinterleave into a 64-KiB buffer before pass 1** (to make
  the DRAM reads perfectly sequential): 577 vs 477 µs per pass-1 sweep — the strided
  direct loads were never the problem; the extra 2048 L1 ops/plane were.  Removed.
* **Per-load `prefetcht0 +128 B` (next zb column) in pass 1**: removing it saved
  ~35 µs/volume (455 → 419 µs pass-1 sweep).  The same hint in the x-line pass HELPS
  (~12% at B=8: 651 vs 572 µs/vol without/with) — do not generalize prefetch decisions
  across passes; the x-lines miss to L3 where pass 1 mostly hits L2.
* **Next-plane spread `prefetcht1` in pass 1 at B=1**: 436 vs 426 µs — slightly
  negative when the input is L3-resident; positive (~5%, 264 vs 279) when streaming.
  Hence the runtime gate: prefetch variant only when nvol > 1.
* **Writing the x-FFT output to a small padded slab buffer** (68 KiB, L2-pinned)
  instead of in place into SC — the traffic model said it should save ~9 MB/volume of
  L3 write+writeback; it measured **3–4% SLOWER** in both regimes (B=1: 684 vs 708;
  B=8: 933 vs 968 µs/vol, same process).  In-place stores hit lines the stage-1 loads
  just brought in (no RFO, no extra working set).  Model rejected by measurement;
  reverted.  Do not re-try without new evidence.
* **Hoisting the 14 z-twiddle vectors into registers across the pass-2+3 body**:
  14 pinned + ILO/IHI + C + 16 live data > 32 zmm → gcc spilled (20 rsp references in
  each pass23 function).  Loading each twiddle vector from the 896-B L1-resident table
  at its use site (`CTW(r,i,VLD(tw3r[k]),VLD(tw3i[k]))`) dropped that to 3 (prologue).
* **Comparing tryout runs to decide variants**: worthless this session — wallaby's
  ~1.7× bimodal states (NUMA/neighbor effects; `numactl -N0 -m0` did NOT cure it)
  swamp any 5–15% effect.  Also: timing passes in separate per-pass loops distorts
  cache behavior completely (all-pass1s-back-to-back turns B=8 into pure streaming and
  reverses conclusions).  A/B inside one process, inside the real per-volume sequence.

### Attribution

52-instruction radix-8 codelet and the create-time NT tuner protocol: **L8_radix8**'s
record (codelet FMA form originally **L8_batchsimd** r1).  Non-destructive 24-op
transpose network, the SW lane-residue algebra and SW-composed interleave vectors:
**L8_fusedaxes** r1 via **L8_radix8** r2's write-up.  Spread-not-burst prefetch and its
fill-buffer rationale: **L8_fusedaxes** r2.  Odd-cache-line padding: corpus §04
(Bailey) via the **L64_blocked** stub's brief.  The lanes-always-z 3D structure, the
ky-slab fusion of the x- and z-passes, and the in-place-in-SC x-FFT are mine.

### Node predictions (stated to be scored)

* B=1: **0.55–0.75 ms** (FP floor 550 µs at 2.9 GHz; L3 there is faster relative to
  core than wallaby's, but one FMA port means the 1.59 M FP instructions bind).
* Batched: **0.75–1.0 ms/vol**, bandwidth-bound at the ~700 µs DRAM floor; expect the
  tuner to pick NT there.  If MKL's 64³ node numbers scale from its L=36 showing
  (~22 GF/s nominal), it lands ≈ 1.1 ms — we should lead every cell.

### Next

1. **Read the node numbers and pick strings first.**  If B=1 lands well above 0.6 ms,
   ask the monitor for `perf stat -e cycles,ref-cycles` — the gap is then load-port /
   L3-latency, not FP, and the fix is more MLP (see 2), not fewer instructions.
2. **Two-volume software pipeline at batch**: pass 1 of volume b+1 interleaved with
   pass 2+3 of volume b (double-buffered SC, 9 MiB — still L3-resident on the node).
   The passes have complementary port profiles (pass 1 load-heavy, z-lines
   shuffle/store-heavy), and it halves the exposure to the in-stream's DRAM latency.
3. **Special-case the w8-power twiddles** in stage 1 (j1·k2 ≡ 0 mod 8: (2,4),(4,2),
   (4,4),(4,6),(6,4) per line-group are ±i or C(1±i) forms): saves ~10 FP instr/line
   ≈ 1% — only worth it if the node proves FP-bound at B=1.
4. **A/B `-DSCKS=128 -DSCXPAD=0` on the node** (one flag) — wallaby says 25%; the
   node's L1/L2 are smaller and 8-way/16-way, so it should bite at least as hard.
   Coordinate with L64_blocked, whose whole brief is this question.
5. Do not revisit: the codelet arithmetic (settled at L=8, three ways), the slab
   buffer (measured loss), the pass-1 plane copy (measured loss), or per-run tryout
   comparisons on wallaby (measured meaningless).

---

## Round panel_r7 (no r6 leaderboard existed, so still no node numbers to react to)

### What changed

The algorithm, layout, scratch padding, and all three passes' arithmetic are
untouched (r6 settled them).  This round is entirely about hiding memory latency
and giving the create-time tuner the candidates the **node** will actually want:

1. **slabpf — next-slab T1 prefetch (new, mine; ON when the tuner says so).**
   While the z-line loop emits ky-slab k's output rows, it T1-prefetches ky-slab
   k+1 of SC — the exact 1024 lines that the next round of x-line stage-1 loads
   would otherwise miss to L3 (they are the only L3-latency-exposed reads left in
   pass 2+3; everything else is L1/L2-hit or covered).  16 lines per kx iteration,
   split into two 8-line groups placed against the FP work, gated off for ky=63
   (tail would run past the mmap).  Worth **1.5–2.5% in every regime** under NT
   stores (in-process A/B below); *slightly negative under plain stores*, so it is
   a tuner axis, not hardwired.
2. **pfw — plain stores + prefetchw of the output row PFW_LEAD (default 4) kx
   iterations ahead (new store mode).**  Borrowed from **L6_unrolled r3 / L36_pfa
   r3–r6** (via **L23_matrixsimd**'s write-up): on the node, three rounds of L=36
   verdicts rejected NT stores and kept prefetchw; my r6 tuner only offered
   plain-vs-NT, i.e. no good streaming option on a machine where NT loses.  On
   wallaby pfw predictably loses to NT (874 vs 660 at B=64) — it exists for the
   node tournament.  `-DPFW_LEAD=n` sweepable.
3. **Tuner now covers the {plain, nt, pfw} × {slabpf 0,1} grid** (6 candidates,
   simplest first, 2% bar vs plain/no-pf, 3 interleaved self-warming rounds at
   bt = min(B,4)), and the pick is reported via `fft3d_description()`.
   **Env forcing for the monitor** (protocol borrowed from **L64_blocked** r6):
   `FFT64R_MODE=0|1|2`, `FFT64R_SLABPF=0|1`, `FFT64R_SCPFW=0|1`.
4. **scpfw — pass-1 SC-store prefetchw variant (kept, default OFF, env-only).**
   Write-intent prefetch of the *next x-plane's* SC rows (17 lines per stage-1
   iteration, x<63) to hide the 4.5 MB/volume of SC-store RFOs.  **Measured a
   LOSS on wallaby** (numbers below) — consistent with r6's "pass 1 tolerates no
   extra prefetch" pattern — but the mechanism targets RFO latency the node's
   slower L3 exposes more, so it stays compiled and forcible rather than deleted.

### Operation count

Arithmetic unchanged from r6: 1190 flops/line, ≈1.59 M vector FP instructions
per volume.  Added instructions when enabled: slabpf 1008 prefetches/volume·64
= 64512 T1 prefetches (4% of FP count, on the load ports); pfw ≤ 61440
prefetchw.  Nothing else moved.

### What was measured (wallaby, Gold 6448Y, gcc 11.4; per transform)

All decisions from **in-process A/B** (6 alternating rounds, best kept), per the
r6 lesson; wallaby was again bimodal across processes (~1.7×).  Same-window
tables, best fast-window values:

| B | plain | nt | **nt+slabpf** | pfw+slabpf |
|---|---|---|---|---|
| 1  | 588.7 | 552.3 | **544.0** | 578.4 |
| 8  | 691.8 | 510.3 | **500.3** | 599.9 |
| 64 | 1086.3 | 653.3 | **639.1** | 874.1 |

So vs r6's shipped configuration (nt, no slabpf): −1.5% (B=1), −2.0% (B=8),
−2.2% (B=64), all same-process.  A separate fast-window single-variant run gave
**B=1 512.0 µs** (best ever seen for this file on wallaby).  Best batched:
**500.3 µs/vol at B=8**, 639.1 at B=64 (vs MKL same day: 1381–1882 µs/vol at
B=64, 2.2–2.9×).

tryout.sh (cross-process, window luck applies): B=1 948.0 (slow window; MKL drew
668 in a fast one the same session — ignore cross-process, again), B=3 552.2/vol,
B=8 546.0/vol, B=64 635.5/vol.  PASS everywhere, rel_l2 = 4.16–4.46e-16,
rel_max ≤ 5.9e-16, bit-identical re-runs (output is variant-independent, so the
tuner's pick cannot break repeatability).  Scalar fallback (`-mno-avx512f`)
PASSES, 3.29 ms/vol.  Builds warning-free (`-Wall -Wextra`) at native SPR and
generic x86-64.  Setup 0.06–0.16 s.

### What was tried and did NOT work

* **scpfw (pass-1 SC-store prefetchw)**: B=1 967.4 vs 937.7 (−3.2%, slow-window
  same-process), B=8 523.5 vs 515.2 (−1.6%).  Pass 1 is issue-limited; the store
  buffer was already absorbing the RFOs.  Kept env-gated for the node only.
* **slabpf under PLAIN stores**: B=1 1018 vs 994, B=64 in earlier grid slightly
  negative too.  Only pays when stores don't compete for L2 fill (NT/pfw) — hence
  a tuner axis.  Do not hardwire it on.
* **Not rebuilt, per records**: two-volume software pipeline (node rejected every
  pipeline at L=36 three rounds running; L64_blocked r6 reached the same verdict);
  any x-FFT output buffering (r6 measured the slab buffer 3–4% slower — my
  "fuse x-stage-2 into z-lines through an 8 KB buffer" sketch is the same traffic
  claim and was dropped without spending the round on it); w8-power twiddle
  special-casing (in split-complex FMA form, C(1±i) twiddles cost the same 4 ops
  as a general cmul — only w^16 = −i saves anything, once per line: noise).

### Attribution

prefetchw-instead-of-NT as a store candidate and the row-ahead placement:
**L6_unrolled r3** and **L36_pfa r3–r6** via **L23_matrixsimd**'s pass-3 writeup.
Env/`-D` forcing protocol for monitor experiments: **L64_blocked r6**.
The next-slab T1 prefetch (slabpf) and the pass-1 SC prefetchw experiment (and
its negative result) are mine, this round.

### Node predictions (stated to be scored)

* B=1: tuner picks nt or nt+slabpf; **0.55–0.75 ms** (unchanged band; slabpf's
  couple of % ride on top of whatever the node's L3 latency does to the base).
* Batched: **the interesting cell.**  If the node behaves like its L=36 self, the
  tuner should pick **pfw+slabpf** there (NT rejected, prefetchw kept) — check
  the pick string in the t_*.json descriptions.  If it picks NT anyway, L=64's
  16-full-line sequential rows are different physics from L36's strided NT and
  that is worth a verdict note.  Monitor one-flag sweeps that would settle
  things: `FFT64R_SCPFW=1` (does the node contradict wallaby on pass-1 RFO
  hiding?), `-DPFW_LEAD=8` and `=16` (L36 found the lead mattered −42%
  in-arena), `-DSCKS=128 -DSCXPAD=0` (the r6 padding number on the node, still
  unmeasured there, coordinate with L64_blocked).

### Next

1. Read the first node leaderboard for this geometry; every prediction above is
   falsifiable from the pick strings + one `perf stat -e cycles,ref-cycles`.
2. If node B=1 sits far above 0.6 ms with nt+slabpf picked, the residual is
   x-line L3 latency that slabpf's one-slab lead can't cover: try two-slab lead
   (prefetch ky+2; costs nothing to A/B in-process).
3. If the node's batched cells sit near the ~700 µs DRAM floor already, stop
   spending on batch and put the next round into B=1 (the only cell with
   headroom vs the 550 µs port floor).

---

## Round panel_r8 (first round with node numbers to react to)

### Where things stood after the r7 leaderboard (node, Gold 5218)

B=1 **995.782 µs** (1.20× ahead of mkl_dfti 1197.1), B=2 1022.97 (1.20×),
B=8 1245.37 (1.56× ahead) — fastest in all three cells, promoted.  Node tuner
picks: **plain+slabpf at B=1, pfw+slabpf at B=2 and B=8** — the r7 prediction
(node rejects NT, from the L=36 rule) was right, and the verdict scored it as
such.  What was wrong was the floor band: I predicted 0.55–0.75 ms and landed
at 0.996 (+33%); the verdict's §4 ratio table puts my wallaby→node factor at
**1.83–1.95×**, and instructs next predictions be anchored on that, not on the
port floor.  B=1 is **1.81× above the ~550 µs port floor — the worst floor
ratio on the board** — so the residual is memory schedule, not FP.

### Technique this round

The r7 verdict's §6 names the move for L=64 explicitly: LITERATURE §4.3's
re-opened case, **L2↔DRAM tiling** ("tile so a tile fits L2, then run the
axes inside the tile"; §08 §1.9 measures a 7× bandwidth gap L2 vs single-core
DRAM).  My fused structure runs the x-FFT with stage-1 loads strided across
the whole 4.5-MB SC — L3 misses on the node's 1-MB L2, only partially hidden
by slabpf.  So this round adds a second structure as a **tuner candidate**
("tiled"), leaving the fused one untouched as the default:

* **Pass A, per z-octet slab** (slot (x,ky) = x·TXS + ky·16, TXS = 1032
  doubles = 129 lines odd; slab = 64·TXS+8 = 8257 lines odd, **528 KB —
  L2-resident on the node's 1 MB**; 8 slabs = 4.23 MB, fits inside the same
  hugepage SC mmap): y-FFT for all 64 x (input reads are that zb's 128-B
  chunks at 1-KiB stride; stage-2 slab stores become **dense sequential 8-KiB
  rows**, vs the fused pass 1's scattered 17-line rows), then the x-FFT for
  all 64 ky **in place in the slab** — its 8-load stage-1 groups (stride
  8·TXS = 1032 lines ≡ 8 mod 64 sets and mod 1024 sets: conflict-free) hit
  L2 instead of L3.  A PF variant (batch only, like the fused pass 1)
  T1-prefetches exactly the 16 lines that stage-1 iteration j1 of plane x+1
  will load.
* **Pass B (after all 8 slabs): the z-FFT sweep, kx outer / ky inner**, so
  the 8 slab reads are 8 *sequential* streams (+128 B per iteration each) and
  the output is one sequential write stream (consecutive 1-KiB rows of the kx
  plane) — the most prefetch-friendly shape this geometry allows, vs the
  fused z-lines' dependence on the ky-slab being L2-hot.  The z-line body is
  byte-for-byte the fused one (radix-8 over octets, tw3 lane twiddles, TR8
  pair, radix-8, SW-composed interleave).  A paced T0 read prefetch runs
  PFB_LEAD=8 rows ahead in all 8 slabs (16 lines/iteration = consumption
  rate; worth −1.8% on tiled-nt on wallaby, kept); store modes plain/NT/pfw
  as usual.

Both structures produce **bit-identical output** (same codelets, same
twiddles, same per-line operation order), so the tuner's pick cannot break
repeatability — verified by tryout at B=1/2/8/64 shipped and B=1/2/8 forced
tiled.

Second change, from my own r7 "Next" item 2: **slabpf is now a lead count**
(0/1/2) — lead 2 T1-prefetches ky-slab k+2 while z-lining slab k, giving the
prefetch two slabs of time to land.

Tuner grid is now **{fused}×{plain,nt,pfw}×{lead 0,1,2} + {tiled}×{plain,nt,
pfw} = 12 candidates**, same protocol (3 interleaved self-warming rounds,
min of 2 timed executes, 2% bar vs fused/plain/0).  New env for the monitor:
`FFT64R_STRUCT=0|1`, `FFT64R_SLABPF` now 0|1|2, **`FFT64R_TUNEDBG=1` dumps
the whole 12-candidate in-process table to stderr** — one run of that on the
node answers the structure question even if the pick goes the other way.
`-DPFB_LEAD=n`, `-DPFW_LEAD=n` sweepable.  SC allocation is
max(fused, tiled) so the monitor's `-DSCKS=128 -DSCXPAD=0` padding A/B still
builds.

### Operation count

Arithmetic identical in both structures: 1190 flops/line, ≈1.59 M vector FP
instructions per volume, same shuffle count (tiled moves zero butterflies).
What tiled changes is traffic: fused pays SC RFO+WB from pass 1 (scattered),
an L3 read of all 4.5 MB in the x-lines (slabpf-covered at best), and gets
the z-line reads L2-free; tiled pays the same pass-A RFO (but on dense rows),
gets the x-FFT reads as **L2 hits**, and moves the z-read to a 4.2-MB
sequential L3 sweep.  Net: one full SC round trip less exposure, traded for
sequential-vs-L2-hot z reads.  Which side wins depends on the L2 size and L3
latency — i.e., on the machine, hence a tuner axis, not a decision.

### What was measured (wallaby, Gold 6448Y, gcc 11.4; in-process 12-candidate
tuner tables via FFT64R_TUNEDBG, per-transform µs/vol; cross-process numbers
carry the usual wallaby window luck)

B=1 tables (two processes):

| candidate | proc 1 | proc 2 |
|---|---|---|
| fused plain lead0 | 711.1 | 703.6 |
| fused nt lead1 | 570.5 | 629.3 |
| fused **nt lead2** | **530.3** (pick) | 636.0 |
| fused pfw lead1 | 563.9 | 550.3 |
| fused **pfw lead2** | 577.9 | **547.8** (pick) |
| tiled nt | 635.4 | 623.8 |

B=64 (tuner at bt=4): fused nt lead0/1 **544.1/545.5** (pick nt lead0; an
earlier window gave nt lead1 503.1), tiled nt 669.5, tiled plain/pfw 685/692.

Full tryout runs: **B=1 555.6 µs** (sd 0.12–2.6% across processes; MKL same
sessions 666–1205), **B=8 570.3 µs/vol** (4562.1/8, sd 0.03%; MKL 889/vol →
1.56×), **B=64 626.6 µs/vol** (40103.6/64; MKL 1613–1694/vol → 2.6–2.7×).
Forced tiled: B=1 1117 µs and B=8 1297 µs/vol in slow-ish windows —
cross-process, not comparable; the in-table gap is the honest one (tiled
**15–23% behind fused on wallaby**).  Correctness: PASS at B=1/2/8/64
(shipped picks) and B=1/2/8 forced `FFT64R_STRUCT=1` (incl. `FFT64R_MODE=1`
and `=2`), rel_l2 = 4.460–4.464e-16, rel_max ≤ 5.7e-16, bit-identical
re-runs everywhere.  Both ISA paths compile warning-free under
`-Wall -Wextra` (AVX-512 and generic x86-64).  Setup 0.08–0.50 s (the
12-candidate grid roughly doubled tuner time; still well under a second).

### What was tried and did NOT work

* **Tiled on wallaby**: 623.8 vs 530.3 at B=1, 669.5 vs 544.1 at B=64-arena —
  loses by 15–23% *on this machine*, whose 2-MB L2 + 60-MB fast L3 make the
  fused x-lines nearly free (slabpf-covered L3 hits) while tiled still pays
  its pass-B L3 sweep.  This is precisely L64_blocked r7's st=1 situation
  (wallaby cannot exhibit the node's 1-MB-L2 physics), so it ships as a
  candidate the node tournament decides, NOT as a wallaby verdict.  If the
  node also rejects it, LITERATURE §4.3's L2↔DRAM tiling case is **closed
  for L=64 on both machines** and the record should say so.
* **Pass-B without software read prefetch**: tiled-nt 635.4 vs 623.8 with
  PFB_LEAD=8 — HW streamers alone leave ~2% on the table even for 8 purely
  sequential streams.  Kept (it cannot hurt the node, where the same reads
  are farther away).
* **slabpf lead 2 under plain stores**: 745–761 vs 703–711 lead0 — same
  pattern as r7's lead-1 result: slab prefetch only pays when stores do not
  compete for L2 fill (NT/pfw).  The grid already encodes this.

### Attribution

The tiled structure is the r7 **VERDICT §6 / LITERATURE §4.3** ask (Alappat
et al. / Intel manual / L3-Fusion via §08 §1.9), executed as a tuner
candidate per the panel's standard protocol; the
expect-wallaby-to-reject-it-and-let-the-node-decide reasoning is
**L64_blocked r6/r7**'s pf=3/4 and st=1 precedent.  Two-slab slabpf lead is
my own r7 "Next" item.  Everything else carries r6/r7 attributions
(codelet L8_radix8/L8_batchsimd, transpose+interleave L8_fusedaxes, pfw
L6_unrolled/L36_pfa, padding Bailey §04).

### Node predictions (stated to be scored; anchored on the measured 1.83–1.95×
wallaby→node ratio per the r7 verdict's §4 instruction, not on the port floor)

* **If the node's 1-MB L2 makes the tiling case real**, the tuner picks
  `tiled-pfw` or `tiled-plain` at B=1 and lands **below 950 µs**; batched
  picks tiled-pfw near the ~700 µs DRAM floor.
* **If fused holds**, expect `plain/pfw+slabpf1-or-2` at B=1 at **960–1010 µs**
  (wallaby 530–556 × 1.83–1.95, slabpf2's wallaby gain is 0–7% and may not
  transfer), and the tiling question closes at L=64.
* Monitor asks, in cost order: (1) one run per B with `FFT64R_TUNEDBG=1` —
  the 12-candidate node table settles fused-vs-tiled, the slabpf lead, and
  the store mode in one shot even if the picks look unchanged; (2) if tiled
  is picked anywhere, `-DPFB_LEAD=4/16`; (3) still outstanding from r6/r7:
  `-DSCKS=128 -DSCXPAD=0` (padding on the node) and `FFT64R_SCPFW=1`.

### Next

1. Read the node table.  If tiled wins B=1, the follow-ups are (a) prefetchw
   on pass A's dense 8-KiB slab rows (the RFO stream is now sequential, so
   the fused scpfw negative result does not carry over), and (b) fusing
   pass B of volume b with pass A of volume b+1 at batch (their port
   profiles are complementary and both are stream-shaped) — but only if
   batched sits well above the ~700 µs DRAM model.
2. If tiled loses everywhere, B=1's remaining lever is latency in the fused
   x-lines that slabpf cannot cover (it prefetches the *next* slab while the
   *current* one is being z-lined — the first slab of every volume is always
   cold): a one-slab prologue prefetch before the ky loop is ~20 lines of
   code and untried.
3. Do not revisit: everything on the r6/r7 dead-end lists, and (new) tiled
   as a *wallaby* optimization.

---

## Round panel_r9

### Where things stood after the r8 leaderboard (node, Gold 5218)

B=1 **966.824 µs** (−2.9% vs r7; 1.24× ahead of mkl_dfti 1203.4), B=2 1020.108
(honest 1.21×), B=8 1252.336 (honest 1.55× vs fftw3_patient) — fastest in all
three cells, promoted.  Node picks: **fused-plain+slabpf1 at B=1,
fused-pfw+slabpf1 at B=2/B=8**.  The r8 verdict: **tiled rejected in all three
cells** (as pre-registered — LITERATURE §4.3's L2↔DRAM tiling case is closed
for L=64 on both machines, with L64_blocked supplying the data-dependency
argument for why); slabpf lead-2 declined (lead 1 everywhere); wallaby→node
ratio 1.74–1.83×, inside the panel band, to be used as the prediction anchor.
B=1's floor ratio (1.76×, ~966.8 vs ~550 µs port floor) is still the worst on
the board — the residual is memory schedule, not FP.  Verdict §6 names my next
move verbatim: *"a one-slab prologue prefetch, because slabpf prefetches slab
k+1 while z-lining slab k, so the first slab of every volume is always cold"*,
plus two cheap standing sweeps: a hugepage kill-switch (never swept on the
node's smaller STLB) and `FFT64R_TUNEDBG=1` node tables.

### Technique this round

Arithmetic, layout, padding, both structures: untouched (r6 settled the
kernels, r8 closed the structure question).  Three latency-schedule
mechanisms, all node-targeted:

1. **propf — prologue prefetch (the verdict's named item).**  Two symmetric
   halves, one knob:
   * *Pass 2+3:* before the ky loop, T1-prefetch the first max(1, slabpf)
     ky-slabs of SC (1024 lines = 64 KiB each, via the existing PFS8 pairs).
     slabpf covers slab ky+lead while z-lining ky, so slab 0's x-line loads
     were always demand misses — pass 1 wrote slab 0's low-x lines a whole
     4.5-MB SC sweep earlier, i.e. they sit in L3 on the node (1-MB L2 holds
     only the last ~14 x-planes of SC).
   * *Pass 1 (PF variants only):* burst-prefetch input plane 0 (1024 T1
     lines) before the x loop — the next-plane prefetch covers planes 1..63
     (and at batch the next volume's plane 0), so plane 0 of the first
     volume was the same "first item is cold" hole.  Redundant re-prefetch
     at batch b>0 costs ~0.15 µs of issue per volume — noise, accepted.
   propf is decided by a **create-time A/B on the picked candidate** (3
   interleaved rounds, min of 2 timed executes per side, strict win to stay
   on) — so each machine keeps it only if IT measures a win.  Env
   `FFT64R_PROPF=0|1` overrides.
2. **p1pf — pass-1 next-plane prefetch as a B=1 tuner axis.**  r6 gated the
   pass-1 PF variant to nvol>1 on a single wallaby number (−2.3% at B=1).
   On the node, B=1's input is L3-resident and the node's L3 is *relatively*
   much slower than wallaby's; pass-1 stage-1 loads are 8 chunks at 8-KiB
   stride (a page boundary per load — HW prefetch does nothing across
   pages), so every group is L3-exposed.  Three new tuner rows at bt==1
   ({plain,nt,pfw} × slabpf1 × p1pf=1) let the node overrule r6's wallaby
   gate.  Env `FFT64R_P1PF=0|1`.
3. **FFT64R_NOHP=1** skips MADV_HUGEPAGE on SC — the monitor's hugepage
   sweep, protocol from **L64_blocked**'s FFT64B_NOHP (their r7 wallaby
   number was +3.3–3.7% FOR hugepages; the node's smaller STLB has never
   been asked).

Tuner grid: 12 base candidates (unchanged) + 3 p1pf rows at bt==1 = 15,
then the propf A/B on the pick (18 extra executes).  `FFT64R_TUNEDBG=1` now
prints the p1pf column and the propf A/B line.  Setup stays ≪ 1 s.

### Operation count

Arithmetic identical to r6–r8: 1190 flops/line, ≈1.59 M vector FP
instructions/volume.  Added when enabled: propf ≤ 2048 T1 prefetches/volume
(pass 2+3) + 1024 (pass-1 plane 0); p1pf just re-routes to the existing PF
variant at B=1.  Nothing else moved; **output remains bit-identical across
every variant** (prefetch/store-mode only), so the tuner cannot break
repeatability.

### What was measured (wallaby, Gold 6448Y, gcc 11.4; in-process tables via
FFT64R_TUNEDBG unless stated, per-transform µs/vol)

B=1 table (one slow-window process; same-process comparisons only):
pick **fused nt slabpf1 = 888.2**; p1pf row (nt slabpf1 p1pf1) = 898.4
(**−1.1%** — wallaby declines p1pf, consistent with r6's −2.3%); propf A/B on
the pick: off 931.8 / on 936.3 → **propf=0 on wallaby** (−0.5%, ≈ noise, same
sign at B=8: 892.3 / 896.1).  Expected: wallaby's 2-MB L2 + fast 60-MB L3
make the slab-0 misses nearly free; this machine cannot exhibit the node
physics the mechanism targets, exactly the tiled/st=1 situation from r8 —
which is why propf ships as a per-machine A/B, not a hardwired default.

Full tryout runs (cross-process window luck applies): **B=1 525.6 µs**
(sd 0.41%, fast window — best number ever recorded for this file on wallaby;
other windows 543.3/543.9/553.9/567.1; MKL same sessions 663–884),
**B=8 499.6 µs/vol** (3996.4/8 — ties r7's 500.3 best-ever; MKL 6772–6858/8
= 847–857/vol), **B=64 618.8 µs/vol** (39602/64; MKL 2101/vol → 3.4×).
Correctness: PASS at B=1/8/64 shipped picks and at forced `FFT64R_PROPF=1
FFT64R_P1PF=1` (B=1), `FFT64R_PROPF=1 FFT64R_MODE=2` (B=2), `FFT64R_NOHP=1`
(B=2), `FFT64R_STRUCT=1` (B=2); rel_l2 = 4.460–4.464e-16, rel_max ≤ 5.7e-16,
bit-identical re-runs everywhere.  Scalar fallback (`-mno-avx512f`) PASSES.
Warning-free under `-Wall -Wextra` at native Haswell (scalar path) and
generic x86-64 + `-mavx512f`.

### What was tried and did NOT work

* **propf on wallaby**: −0.5% at both B=1 and B=8 (in-process A/B) — the A/B
  correctly turns it off here.  This is a *wallaby* null, not a node verdict;
  the node runs its own A/B at create time and `FFT64R_TUNEDBG=1` prints it.
* **p1pf on wallaby**: −1.1% in-table at B=1 — r6's gate was right for this
  machine.  Ships as node tournament rows only.
* Nothing else was attempted: the r6/r7/r8 dead-end lists stand (slab
  buffer, plane copy, pipeline, w8 twiddle special-casing, scpfw-on-wallaby,
  tiled-on-wallaby, cross-process tryout comparisons).

### Attribution

The prologue prefetch is the **r8 VERDICT §6** ask (which is my own r8 "Next"
item 2, promoted by the monitor to the round's named move).  The hugepage
kill-switch env and its rationale: **L64_blocked** r6/r7 (FFT64B_NOHP,
+3.3–3.7% wallaby measurement).  The decide-per-machine-inside-the-plan
A/B protocol continues my r6 tuner line, reinforced by the verdict's praise
of **L36_pfa**'s in-plan node probe.  p1pf is mine (a re-litigation of my own
r6 gate, but on the node's physics this time, via the tuner so no machine
pays for the other's answer).

### Node predictions (stated to be scored; anchored on my 1.74–1.83×
wallaby→node band per the r7/r8 verdict instruction)

* **B=1**: expect fused-plain+slabpf1 to hold, **945–975 µs**.  propf's A/B
  should come up ON on the node (slab 0 is a genuine L3 exposure there) and
  is worth **0.3–1%** (1/64 of the x-line read traffic, plus plane-0 if p1pf
  is also picked).  p1pf is the wild card: if a p1pf row is picked, input-L3
  latency was a real exposure and the cell can land at the low end of the
  band; if not picked, r6's gate transfers to CLX and that closes it.
* **B=2/B=8**: fused-pfw+slabpf1 holds (propf rides along if its A/B wins);
  **1000–1025 / 1230–1255 µs/vol** — batch is near its traffic model and I
  spent nothing on it this round.
* Monitor asks, in cost order: (1) `FFT64R_TUNEDBG=1`, one run per B — the
  15-row table + propf A/B line settles p1pf and propf even if the picks
  look unchanged; (2) `FFT64R_NOHP=1`, one run at B=1 and B=8 (the verdict's
  standing hugepage question); (3) still outstanding from r6–r8:
  `-DSCKS=128 -DSCXPAD=0` padding A/B and `FFT64R_SCPFW=1`.

### Next

1. Read the node's propf A/B line and p1pf rows.  If **both** decline on the
   node too, first-item coldness and pass-1 read latency are ruled out, and
   the remaining B=1 residual must be the SC store RFOs (pass 1 writes
   4.5 MB scattered) — the one component nothing has ever hidden; the
   perf-stat ask (`cycles, cycle_activity.stalls_mem_any, offcore RFO
   counters`, B=1) becomes decisive before writing more prefetches.
2. If propf wins but small, try the spread variant: interleave the slab-0
   lines into pass 1's last 8 x-planes (2 lines per stage-1 iteration)
   instead of a burst at the pass boundary — only worth building on a
   measured node win, since wallaby cannot see the effect at all.
3. Do not revisit anything on the r6–r8 dead-end lists; tiled stays closed.

---

## Round panel_r10

### Where things stood after the r9 leaderboard (node, Gold 5218)

B=1 **952.944 µs** (−1.4% vs r8; 1.25× ahead of mkl_dfti 1192.0), B=2 1020.05
→ 1021.050 (1.24×), B=8 1249.923 (1.93× ahead of fftw3_patient) — fastest in
all three cells for the fifth round, promoted.  The verdict's two L=64
findings: **p1pf was picked by the node 3/3 at B=1** (the mechanism my own r6
wallaby gate had killed at −2.3% — wallaby-gate reversed by node physics,
exactly why it shipped as a tournament row), and **propf read a coin flip**
(pro0 in two runs, pro1 in one, its own A/B ±0).  B=1 still sits at **1.73×
above the ~550 µs port floor — the board's worst** — and the verdict names
both the residual and the route verbatim: *"the remaining B=1 residual is the
SC store RFOs — pass 1 writes 4.5 MB scattered, the one component nothing has
ever hidden — and with the counters withdrawn the only route is the entry's
own in-plan create-time A/B applied to a store-mode twin."*

### Technique this round

Arithmetic, layout, padding, both structures, the 15-candidate grid: all
untouched.  The round executes the verdict's named move — **scst, a three-way
store-mode twin for pass 1's SC stores**, A/B'd on the picked candidate at
create time (the propf protocol), so each machine keeps only what IT measures:

* **scst=0** — plain stores (incumbent).  Every SC line missing L1/L2 costs
  an RFO read of the stale line (4.5 MB/volume; at B=1 on the node these hit
  L3, and 16 scattered stores per k2-iteration can fill the store buffer
  faster than the FP work drains it).
* **scst=1** — plain + prefetchw of the NEXT x-plane's SC rows, one 17-line
  ky row per stage-1 iteration.  This is r7's `scpfw`, which wallaby measured
  a loss (−1.6 to −3.2%) and which has sat env-only on the monitor's
  outstanding list for three rounds — promoted into the in-plan A/B so the
  node finally runs it without costing the monitor a flag sweep.
* **scst=2** — NON-TEMPORAL stores (`vmovntpd`), new: the RFO disappears
  entirely.  Legal without any repacking because every SC line is written
  whole (each __m512d store covers exactly one 64-B line — the split-complex
  slot layout guarantees it), so the WC buffers combine cleanly even though
  the slots are scattered.  Cost side of the trade: SC lines are evicted, so
  pass 2's x-line loads re-read them from DRAM — which is exactly the traffic
  slabpf/propf prefetch, so if scst=2 wins its A/B, **the propf A/B is re-run
  under it** (slab-0 coldness is strictly worse under NT; r9's coin-flip
  verdict was taken under plain stores and may flip).  One `sfence` per
  volume drains the WC buffers before pass 2 reads the same lines back
  (same-core data dependence is architecturally safe; the fence just stops a
  straggler buffer flushing mid-pass-2).

Non-plain must beat plain by **1%** (a store-mode change moves traffic, so a
noise-level win must not flip it; the propf A/B keeps its strict-win rule).
All three variants produce **bit-identical output** (prefetch/store-opcode
only), verified by tryout's repeatability check on every forced path.  Env:
`FFT64R_SCST=0|1|2` (forcing), `FFT64R_SCPFW=1` still accepted as an alias
for scst=1 so the monitor's standing one-flag ask from r7–r9 works unchanged.
`FFT64R_TUNEDBG=1` now prints the scst A/B line.  The pick string gains
`+sc%d`.  Setup cost: +27 tuner executes (+12 more only if scst≠0 wins) —
measured 0.14–0.51 s total, unchanged band.

**Considered and NOT built — association-order codelet twins** (the r9
verdict §5 finding: ±3.3–6.2% on CLX from re-association alone at L=6, with
an explicit "propagate to every other unrolled codelet" instruction): the
verdict's named targets are the L1-resident geometries (L=8 dft8s, L=36
DFT4/DFT9, L=13 chunk13) where the core schedule binds.  L=64 is
memory-schedule-bound — 1.73× floor with the FP bill NOT binding, and the
same verdict names my residual as the SC store RFOs, not the schedule.
Racing codelet twins here would also multiply the already-large macro
surface (6 pass-1 × 3 pass-23 × 2 structures).  If the node's scst A/B and
the batch cells both come back flat next round, this is the next candidate,
scoped to the xline body only.

### Operation count

Arithmetic identical to r6–r9: 1190 flops/line, ≈1.59 M vector FP
instructions/volume, same shuffle bill.  scst=1 adds ≤17,408 prefetchw;
scst=2 adds ZERO instructions (same store count, different opcode) plus one
sfence/volume; scst=0 is byte-identical to r9's pass 1.

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4; per transform)

Full tryout runs (cross-process window luck applies as always):
**B=1 503.4 µs min** (sd 4.3%, fast window — best tryout number ever recorded
for this file on wallaby; a later slow-window run gave 586.4 at sd 0.05%),
**B=8 4346.5/8 = 543.3 µs/vol** (second run 545.1), **B=64 40881.4/64 =
638.8 µs/vol**.  MKL same sessions: 791–1216 (B=1), 787–967/vol (B=8),
1831/vol (B=64 → 2.9×).

In-process A/B tables (FFT64R_TUNEDBG, same-process comparisons only):

| | B=1 (slow window) | B=8 |
|---|---|---|
| grid pick | fused-nt+slabpf1+p1pf1 = 588.3 | fused-nt (batch rows) |
| propf A/B | off 587.2 / on 585.4 → **propf=1** | off 538.0 / on 542.2 → propf=0 |
| scst A/B: plain / pfw / nt | **585.2** / 584.4 / 803.8 → **scst=0** | **535.5** / 542.1 / 783.1 → **scst=0** |

So wallaby declines both twins: pfw is inside the 1% bar (−0.1% at B=1, −1.2%
at B=8 — consistent with r7's scpfw loss), and **NT costs +37%/+46% here** —
wallaby's 2-MB L2 + fast 60-MB L3 make the RFOs nearly free while NT pushes
the 4.5-MB re-read to DRAM.  This machine cannot exhibit the node physics the
twin targets (1-MB L2, relatively slow L3, one store port) — the tiled/propf/
p1pf situation again, which is precisely why it ships as a per-machine A/B
and not a wallaby verdict.  p1pf note: the B=1 grid pick was a p1pf row on
wallaby this time too (588.3 vs 592.8 without), consistent with the node's
3/3.

Correctness: PASS at B=1/8/64 shipped picks and at forced `FFT64R_SCST=1`
(B=2), `FFT64R_SCST=2` (B=2), `FFT64R_SCST=2 FFT64R_PROPF=1 FFT64R_P1PF=1`
(B=1), `FFT64R_SCPFW=1 FFT64R_MODE=2` (B=2, alias path); rel_l2 =
4.460–4.464e-16, rel_max ≤ 5.7e-16, **identical rel_l2 across all forced
variants** (bit-identity, as designed), bit-identical re-runs everywhere.
Scalar fallback (`-mno-avx512f`) PASSES at 3.43 ms/vol.  Builds warning-free
under `-Wall -Wextra` at native SPR, generic x86-64 + `-mavx512f`, and
generic x86-64 scalar.  Setup 0.14–0.51 s.

### What was tried and did NOT work

* **scst=1 (pfw) on wallaby**: −0.1% / −1.2% (in-table) — below the bar,
  correctly declined.  A wallaby null for the third time; the node has never
  run it, and now runs it for free inside create().
* **scst=2 (NT) on wallaby**: +37% at B=1, +46% at B=8 — decisively wrong
  for THIS machine, exactly as the traffic model says (L3-resident SC makes
  the RFO cheap and the DRAM re-read expensive).  The node's 1-MB L2 and
  slower L3 change both sides of that trade; its A/B decides next round.
  Do NOT read these numbers as closing the question.
* Nothing else attempted: the r6–r9 dead-end lists stand (slab buffer, plane
  copy, pipeline, w8 twiddle special-casing, tiled-anywhere, cross-process
  tryout comparisons on wallaby).

### Attribution

The pass-1 store-mode twin is the **r9 VERDICT §6 / L=64** named ask,
executed with my own r9 propf in-plan A/B protocol.  scst=1 is my r7 scpfw
promoted from env-only to the A/B; scst=2 (NT-to-scratch, full-line
write-combining on scattered slots) is mine this round.  The
per-machine-decides framing continues the tiled (r8) and propf/p1pf (r9)
precedents; the 1% traffic-change bar follows the grid's 2%-bar rationale
from r6 (L8_radix8's tuner protocol).

### Node predictions (stated to be scored; anchored on my 1.74–1.83×
wallaby→node band)

* **B=1**: base pick fused-plain-or-nt+slabpf1+p1pf1 unchanged →
  **940–970 µs** if the node's scst A/B also reads plain.  If it picks
  scst=1, expect **−0.5 to −2%** (925–955); if it picks scst=2 the RFO
  theory was right at full strength and the cell can move **−3 to −6%**
  (895–935) — NT there also relieves the single store port.  Either
  non-plain pick vindicates the verdict's residual diagnosis even if small.
* **B=2/B=8**: untouched paths, expect **1000–1030 / 1230–1260 µs/vol**.
  scst=2 at batch is a real possibility (the RFOs there compete with the
  input stream for DRAM), worth reading the pick string.
* Monitor asks, in cost order: (1) `FFT64R_TUNEDBG=1`, one run per B — the
  table now ends with the scst A/B line, which is THE datum of this round;
  (2) still outstanding from r6–r9: `FFT64R_NOHP=1` (hugepage sweep) and
  `-DSCKS=128 -DSCXPAD=0` (padding on the node).

### Next

1. Read the node's scst line.  If the node also reads plain in all three
   cells, the SC-RFO theory is dead by direct test on the machine it was
   written for, and the B=1 residual hunt needs the perf-stat ask
   (`cycles, cycle_activity.stalls_mem_any, offcore RFO counters`, B=1)
   before ANY further prefetch/store mechanism is built — three rounds of
   scheduling tweaks have now moved B=1 by ~1.4% total.
2. If scst=2 wins at B=1, try the natural follow-up: since SC re-reads then
   come from DRAM, raise slabpf lead to 2 (already a grid row) and consider
   T2 for the prologue.
3. If everything is flat, the scoped association-order experiment (xline
   body only, three orderings, raced in-plan) is the last untried mechanism
   class with a positive node result this round — see "considered and NOT
   built" above for why it waited.

---

## Round panel_r11

### Where things stood after the r10 leaderboard (node, Gold 5218)

B=1 **949.904 µs** (−0.3% vs r9; 1.25× ahead of mkl_dfti 1191.4; L64_blocked's
split-complex rewrite tied the cell at 952.9), B=2 1026.618 (1.21×), B=8
1262.872 (1.53× honest, per the verdict's §3e margin correction) — fastest in
all three cells on medians for the sixth round, promoted.  **The scst A/B read
`sc0` (plain) 3/3 in every cell** — my own pre-registered criterion says the
SC-store-RFO theory is dead by direct test on the machine it was written for,
and the verdict recorded it that way.  B=1 still sits at **1.73× the ~550 µs
port floor, the board's worst**, propf read a coin flip, and every
memory-schedule mechanism this file has built (slabpf/propf/p1pf/scst/tiled)
is now either adopted-and-small or dead.  The verdict's L=64 item points at
L64_blocked's B=8 residency boundary; for this file it names nothing — my own
r10 "Next" item 3 (the association-order class, scoped to the xline body) is
the only pre-registered move left.

### Technique this round (three changes, all decided per machine)

Arithmetic, layout, padding, structures, the 15-candidate grid, propf/scst:
untouched.

1. **xb — compact x-FFT output buffer, the r6 slab-buffer rejection re-run on
   the node for the first time.**  The x-line pass updates SC IN PLACE, which
   dirties 4.5 MB/volume of SC lines and forces an L2→L3 writeback sweep —
   after scst's death, **the last unaddressed traffic component at B=1**.
   xb=1 sends x-line stage-2 output to a 68-KB buffer instead (slot (kx,zb) =
   kx·XBS + zb·16, XBS = 17 lines odd, kx = k2+8k1 — the exact mapping the
   z-lines read back), and the z-lines read XB rather than SC; SC becomes
   READ-ONLY in pass 2+3 and its 4.5-MB writeback disappears.  The buffer is
   appended to the SC mmap so it shares the hugepages.  r6 measured the
   equivalent 68-KB slab buffer **3–4% slower on wallaby** and wrote "do not
   re-try without new evidence" — the new evidence is r9→r10's p1pf
   precedent: a small wallaby loss (−1.1%) inverted to a 3/3 node pick,
   because wallaby's 2-MB L2 + fast L3 cannot exhibit the node physics
   (1-MB L2, relatively slow L3) that the mechanism targets.  Ships as an
   in-plan create-time A/B on the settled configuration (the scst protocol,
   1% bar since it moves traffic), so wallaby's expected rejection costs the
   node nothing.  Output is bit-identical (same values, same order,
   different intermediate addresses).
2. **fout — the r9 verdict's store-feeding-class law tested at the one L=64
   site it can apply to.**  The law: "on Cascade Lake, store-feeding FMAs
   beat store-feeding adds by 3–6% on identical arithmetic" (L6_pfa's d2 /
   L6_unrolled's f3d, r9).  Inspecting my codelet against it: the radix-8
   final stage already produces its odd outputs by FMA (the d2-winning
   shape), and the even outputs have no irrational factor to fold — so a
   genuine reassociation twin does not exist here, only the anti-law
   direction (hoist the C-muls early, store-feed adds), which the law says
   loses.  What CAN be tested: convert the even outputs with
   **fma(x, 1.0, y), which is IEEE-identical to add(x, y)** (x·1 exact,
   single rounding) — a bit-identical, zero-extra-op opcode-class twin
   (RADIX8F), applied in the x-line stage-2 codelet whose 16 outputs feed
   the L3-bound stores.  This isolates the instruction-CLASS half of the law
   from the graph-shape half: if fout moves the node, the class matters; if
   it reads null there, the L6 effect was graph shape and the law's wording
   over-claims.  In-plan A/B, strict win (zero-traffic change); bit-identity
   means an unstable pick across runs is provenance-harmless — no bit-class
   exposure (the §3a lesson from r10 respected by construction).
3. **Tuner arena clamp raised 4 → 8 volumes.**  B=8 is scored streaming
   64 MB from DRAM, but bt = min(B,4) tuned its picks in a 32-MB arena that
   is half L3-resident on the node — the wrong regime.  bt = min(B,8) tunes
   B=8 where it is scored.  Setup cost measured 1.06–1.15 s at B=8 (was
   ~0.5 s); create() is unscored and libraries run up to 4.8 s.

Mechanics: 4 xline variants ({SC,XB} × {RADIX8,RADIX8F}, macro-generated,
**0 stack spills in all four**, checked by objdump), pass23 gains an XB
compile-time flag (6 variants) and a runtime fout switch per (ky,zb) call
(512 branches/volume — noise).  New env: `FFT64R_XB=0|1`, `FFT64R_FOUT=0|1`;
`FFT64R_TUNEDBG=1` now also prints the xb and fout A/B lines.  Pick string
gains `+xb%d+fo%d`.

### Operation count

Arithmetic identical to r6–r10: 1190 flops/line, ≈1.59 M vector FP
instructions/volume, same shuffle bill.  fout changes ZERO ops (8 add/sub →
8 FMA per stage-2 codelet, same count, same ports on CLX).  xb changes zero
instructions (same store count, different target) and −4.5 MB/volume of
L2→L3 writeback traffic when on, traded for a 68-KB L2-hot store target.
All variants bit-identical — verified: rel_l2 = 4.464e-16 at B=2 for every
forced combination (xb, fout, xb+fout+pfw, xb+nt, xb+scst2), the exact
incumbent fingerprint, and tryout's re-run check passes everywhere.

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4; per transform)

Full tryout runs (cross-process window luck applies as always):
**B=1 520.3 µs min** (fast window, sd 2.7%; slow-window runs 913.9–923.9 at
sd ≤0.23%; MKL same sessions 663.5–1224.9), **B=8 4406.2/8 = 550.8 µs/vol**
(second run 4466.6/8 = 558.3; MKL 977–1081/vol → 1.8–2.0×), **B=64 39394.7/64
= 615.5 µs/vol — best B=64 number ever recorded for this file on wallaby**
(MKL 111978/64 = 1749.7/vol → 2.8×).

In-process A/B lines (FFT64R_TUNEDBG, same-process comparisons only):

| A/B | B=1 (slow window) | B=8 (bt=8 arena, streaming) |
|---|---|---|
| grid pick | fused-nt+slabpf1+p1pf1 = 930.4 | fused-nt+slabpf1 = 528.2 |
| propf | off 924.7 / on 928.1 → 0 | off 560.2 / on 556.4 → 1 |
| scst | plain 924.5 / pfw 942.6 / nt 1021.6 → 0 | plain 561.1 / pfw 558.2 / nt 793.5 → 0 |
| **xb** | inplace 921.2 / **xbuf 963.1 → xb=0 (−4.5%)** | inplace 559.0 / xbuf 573.8 → xb=0 (−2.6%) |
| **fout** | add 923.7 / **fma 922.2 → fout=1 (+0.16%)** | add 560.3 / **fma 556.2 → fout=1 (+0.7%)** |

Readings: **xb loses on wallaby by the same 3–4.5% r6 measured** — as
pre-registered; this machine's fast L3 makes writebacks free while the extra
68-KB store sweep costs.  A wallaby null by design; the node runs its own A/B
inside create() and `FFT64R_TUNEDBG=1` prints it.  **fout won its strict-win
A/B in both regimes, twice** (+0.16% and +0.7%) — weakly positive on SPR,
which was the machine expected to DECLINE it (SPR has dedicated fast FP-add
ports the FMA form gives up); if even SPR prefers FMA store feeds, the CLX
prediction firms up.  Correctness: PASS at B=1/2/8/64 shipped picks and at
all five forced combinations listed above; rel_l2 = 4.460–4.464e-16,
rel_max ≤ 5.7e-16, bit-identical re-runs everywhere.  Scalar fallback
(`-mno-avx512f`) PASSES at 3.14 ms/vol.  Builds warning-free under
`-Wall -Wextra` at native SPR, generic x86-64 + `-mavx512f`, and generic
scalar.  Setup 0.15–1.15 s (upper end is the bt=8 arena at B≥8).

### What was tried and did NOT work

* **xb on wallaby**: −4.5% (B=1), −2.6% (B=8, streaming arena) — r6's
  slab-buffer number reproduced five rounds later on the same machine class.
  Correctly declined by the A/B here.  NOT a node verdict — that was the
  entire point of shipping it as a twin; do not read these numbers as
  closing the r6 question on the node.
* Nothing else attempted: the r6–r10 dead-end lists stand (slab buffer as a
  wallaby optimization, plane copy, pipeline, w8 twiddle special-casing,
  tiled-anywhere, cross-process tryout comparisons on wallaby).
* **Considered and NOT built — a genuine reassociation twin of RADIX8**: the
  codelet's odd outputs are already FMA-final (the d2-winning pole) and the
  even outputs have no multiply to fold, so the only real alternative
  association is the anti-law direction (early C-muls, add-final), which
  costs +4 ops AND a new bit class for a predicted loss.  Documented so the
  next round does not re-derive it: **at L=64 the association-order search
  space is one opcode-class twin (built, = fout), not three graphs.**

### Attribution

The store-feeding-class law and the propagate-it instruction: **r9 VERDICT
§5 via L6_pfa r9/r10 (`fused_d2`) and L6_unrolled r9 (`f3d`/`ab1`)**.  The
fma(x,1,y)≡add(x,y) bit-identical formulation of the twin is mine (keeps the
tuner pool one bit class — the discipline **L36_mixedradix** was forced to
invent and the r10 verdict §3a made panel law).  xb re-tests my own r6 slab
buffer under the **p1pf wallaby-inversion precedent (r9→r10)**; the
in-plan-A/B-decides-per-machine protocol continues r8–r10 (tiled/propf/scst).
The bt=8 arena-fidelity argument follows **L8_batchsimd r9's regime-gated
allocation** finding (§4.5 refinement: tune in the regime you are scored in).

### Node predictions (stated to be scored; anchored on my 1.74–1.95×
wallaby→node band)

* **B=1**: base pick unchanged (fused, plain-or-nt, slabpf1, p1pf1, sc0).
  Branches: (a) if the node's xb A/B also reads inplace, r6's rejection
  transfers to CLX and the in-place-vs-buffer question closes for good on
  both machines — expect **940–970 µs** (flat); (b) if xb is picked, the
  writeback sweep was a real cost wallaby could not see — expect
  **905–945 µs** (−1 to −4%); (c) fout picked with a visible move (≥1%)
  vindicates the law's instruction-class reading at a memory-bound
  geometry — worth **0–2%** on top of either branch (the L6 3–6% was
  measured where the core schedule binds; here it binds maybe a third of
  the time).  My honest central estimate: fout picked, sub-1%, cell lands
  **935–965**.
* **B=2**: untouched paths (bt=2 arena unchanged), **1010–1035 µs/vol**.
* **B=8**: the bt=8 arena is the live change — the picks are now decided
  streaming.  If the pick string differs from r10's (pfw+slabpf1), the
  32-MB arena was mis-tuning the cell and the honest band widens down:
  **1180–1265 µs/vol**; if the pick is unchanged, expect **1230–1265**
  (flat — I spent nothing else on batch).
* Monitor asks, in cost order: (1) `FFT64R_TUNEDBG=1`, one run per B — the
  table now ends with the xb and fout A/B lines, which are THE data of this
  round; (2) still outstanding from r6–r10: `FFT64R_NOHP=1` (hugepage
  sweep) and `-DSCKS=128 -DSCXPAD=0` (padding on the node).

### Next

1. Read the node's xb and fout lines.  Four outcomes, all informative:
   xb picked → the traffic model finally caught a real node cost; follow up
   with NT stores INTO xb's z-line output path re-examined (out-stream RFO
   is then the only writeback left).  xb declined → in-place-vs-buffer is
   closed on both machines, permanently.  fout picked with ≥1% → propagate
   RADIX8F to the pass-1 stage-2 codelet (the other store-feeding site,
   ~20 lines).  fout null → the L6 law was graph shape, not opcode class;
   record it so L=8/L=36 don't chase the class reading.
2. If B=1 is still ~950 with everything above read out, this file has no
   named mechanism left at any regime boundary: the remaining 400 µs above
   the port floor is unattributed L3/L2 machinery that only a PMU can name,
   and `perf_event_paranoid=4` is cluster-wide (L6_unrolled r9 checked).
   The honest next round is then consolidation: adopt whatever L64_blocked's
   B=8 residency work turns up, and stop spending B=1 rounds on blind
   scheduling twins — three rounds of them have moved the cell 1.4%.
3. Do not revisit anything on the r6–r10 dead-end lists; tiled stays closed;
   the association-graph space at L=64 is settled as one-twin-only (above).
