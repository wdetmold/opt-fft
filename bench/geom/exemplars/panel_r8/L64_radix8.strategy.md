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
