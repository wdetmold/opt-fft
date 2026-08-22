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
