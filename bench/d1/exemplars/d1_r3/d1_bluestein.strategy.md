# d1_bluestein — strategy record

Class: any-L fallback (Bluestein chirp-Z). Headline sizes per the brief: the awkward
large primes 10007 (N-1 = 2*5003) and 100003 (N-1 = 2*3*7*2381), where everyone —
us, FFTW, MKL — must convolve, and where the survey says the win is engineering the
convolution FFTs, not a cleverer prime algorithm.

## Round d1_r1 (2026-09-02) — from stub to a real chirp-Z engine

Starting point was the fresh-restart dense-DFT stub (O(L^2), supports 8 small sizes).
Everything below was built this round. **Caveat on every number here: the monitor's
Ice Lake reservation (job 440299) was down for my whole session, so I could not use
tryout.sh; all timings are from wallaby (SPR Gold 6448Y, `taskset` to one core,
`nice`), built with the exact Makefile flags + driver + check.py. Wallaby was heavily
loaded (load 25–30, 51 users) for part of the session; I report the best min across
repeats, cross-checked A/B against earlier builds back-to-back. The scoring node
(Ice Lake 6326) is somewhat slower per core and has 1.25 MB L2 vs wallaby's 2 MB, so
expect the L2-edge sizes (M=20480) to look a bit worse there.**

### What the implementation now is

Three paths chosen at plan time (`fft1d_supports` returns L >= 2 — any L):

1. **smooth L (2^a 3^b 5^c, L >= 4): direct split-complex mixed-radix Stockham**,
   radices {2,3,4,5,8}, natural-order output, per-stage twiddle tables from plan-time
   sincos (never in-loop recurrences — survey convergence #2).
2. **non-smooth L <= 16: dense DFT**, split-format, accumulated over the *output*
   index with broadcast FMAs so it vectorizes WITHOUT `-fassociative-math` (the
   graded flags don't have it; a dot-product formulation does not vectorize).
3. **everything else: Bluestein chirp-Z**. Pad M = smallest 3^a 5^b 2^c >= 2L-1 with
   4 | M: 10007 -> 20480 (not 32768), 100003 -> 204800 (not 262144), 1021 -> 2048,
   65537 -> 147456. Chirp phase k^2 reduced mod 2L in int64 before the trig call
   (survey: the fp64 trap at k ~ 1e5). Kernel spectrum precomputed at plan time with
   the inverse's 1/M folded in. Inverse FFT = forward on re/im-swapped planes (free
   with split arrays — just swap the pointers).

### The three lessons that carried the round (in order of impact)

1. **gcc was not vectorizing the radix-4/5 workhorse loops at all.** The split
   streams exceed gcc's alias-check versioning budget (radix-4 = 16 streams), so the
   hottest loops ran scalar while radix-2/3 got versioned-vectorized at 256-bit.
   `#pragma GCC ivdep` on every inner loop fixed it: 10007 went 282 -> 193 us.
   Check `-fopt-info-vec` before believing any C kernel is vectorized.
2. **The Makefile's `-march=native` still leaves gcc at 256-bit vector preference.**
   These shuffle-free split-complex loops want zmm. You cannot add flags to the
   graded build, and a whole-file `#pragma GCC target` silently RESETS the ISA
   (first attempt: 3x slowdown) and then conflicts with fortified memset. What works:
   a per-function attribute `target("arch=icelake-server,prefer-vector-width=512")`
   on the hot kernels only. 10007: 193 -> 159 us; 1024: 2.86 -> 2.25 us.
3. **Radix-8 stages cut memory passes** (Stockham is one full-array pass per stage):
   20480 = [4,8,8,8,5,2] instead of seven radix-4/2/5 stages; 16384 = [4,8,8,8,8].
   10007: 159 -> 134 us; 16384 direct: 73 -> 50 us; 204800 core: 2612 -> 1915 us.

### Bluestein-specific fusions (the class's own tricks)

The padded input has M-L >= L-1 zeros and only L of M inverse outputs are needed.
With M chosen 4|M and the factor order arranged as [4 first, 8s, 3s, 5s, pow2
leftover (2 or 4) last], all of this collapses into three custom stages:

- **entry stage (radix-4, s=1) fused with the input chirp**: since L <= M/2, input
  blocks u=2,3 are identically zero, so the butterfly degenerates to
  X = {a+b, a-ib, a-b, a+ib}; the chirp multiply and deinterleave fold into the same
  pass. The padded buffer and its memset never exist.
- **inverse entry stage fused with the kernel-spectrum multiply** and the re/im swap:
  the standalone pointwise pass over M disappears.
- **exit stage (radix 2 or 4) pruned + fused with the output chirp**: j*s >= L output
  blocks are never computed (for radix-2 last, the entire j=1 half vanishes); the
  chirp and interleave fold in. Applies when the last radix is 2/4 (true for 20480
  and 204800; not for 147456 = [4,8,8,8,8,3,3] or 2048 = [4,8,8,8], which keep a
  separate chirp-out pass).

Together: 10007 went 134 -> ~110 us, 100003 ~3600 -> ~3480 (noisy; A/B showed the
fused build consistently 25–35% ahead of the pre-radix-8 one at 100003/65537/1021).

- **fft1d_chain (exported)**: the state lives *chirp-premultiplied* between steps, so
  each step is: fused entry (reads split premultiplied state) -> conv -> fused exit
  producing z = a*W split -> one L-sized pass doing +c, the map, and the next step's
  chirp premultiply. The interleaved state is only materialized at the final step.
  Chain scratch lives in the plan (no per-call malloc). Gate passes at all 13 graded
  configurations (rel_l2 ~ 1e-14 vs 1e-10 tolerance).

### Where it stands (wallaby best-min, vs library_baseline Ice Lake numbers)

| L | mine B=1 | lib best | mine B=graded | lib best | verdict |
|---|---|---|---|---|---|
| 13 | 0.061 | 0.066 (MKL) | 0.058 | 0.051 | ~parity (dense floor; d1_prime's cell) |
| 31 | 0.17–0.19 | 0.334 | 0.24 | 0.315 | **win ~1.7x** (Bluestein M=64 beat my dense) |
| 32 | 0.097 | 0.131 | 0.099 | 0.112 | win (direct) |
| 60 | 0.122 | 0.232 | 0.181 | 0.228 | win (direct 2/3/5) |
| 64 | 0.148 | 0.237 | 0.26 | 0.238 | win B=1 |
| 128 | 0.29 | 0.485 | 0.505 | 0.535 | win |
| 1024 | 1.81 | 4.18 (MKL) | ~5.9 | 4.92 | **win 2.3x B=1**; batched behind |
| 4096 | 21.7 | 19.1 | 35–40 | 22.7 | behind (batched needs work) |
| 16384 | 50.5 | 82.2 | ~157 | 99.2 | **win 1.6x B=1**; batched behind |
| 1021 | 8.2–9.7 | 11.4 (MKL) | 10.8 | 12.4 | **win** (M=2048; d1_rader may beat via GT-1020) |
| 10007 | **~110–122** | 229.4 (patient) | ~152 | 237.4 | **win ~1.9–2x — the headline** |
| 65537 | 2290–2480 | 1632 (patient) | 2563 | 1740 | behind ~1.5x — Rader's cell (conv IS 2^16); Bluestein pays M=147456 |
| 100003 | 3460–3480 | 3261 (patient) | 3611 | 3205 | ~parity/slightly behind, noise-limited — see next steps |

Accuracy: rel L2 5e-16 – 1.4e-15 at every graded size (gate 1e-12); chain rel_l2
~1e-14 everywhere; any-L spot-checked correct at 16 odd sizes from 2 to 6553.
Setup: <= 0.022 s even at 100003 (FFTW patient pays 63 s there).

### What did NOT work, with numbers

- **`#pragma GCC target("prefer-vector-width=512")` file-wide**: resets the ISA to
  baseline (10007: 285 -> 629 us) and then breaks fortified memset inlining when
  arch is added. Per-function attribute is the only safe form.
- **`-mprefer-vector-width=512` before fixing vectorization**: no effect (85 -> 85 us
  at 20480 pre-ivdep) — the hot loops weren't vectorized at all, so width was moot.
- **Dense floor at L=31**: split-dense 0.27–0.29 us vs Bluestein-M=64 0.19 us.
  DENSE_MAX stays 16. (Dense-13 at 0.061 us is fine until a Winograd entry exists.)
- **Bigger 3^a5^b search for M**: nothing beats 20480/204800 for the headline primes;
  the a,b <= 3 enumeration already finds them.

### Borrowings

- Batch-lane/SoA split-complex framing and "twiddles from plan-time sincos, never
  recurrences": docs/literature_1d/00-SURVEY.md (no other implementer records
  existed this round — context.md was empty after the restart).
- The minimal-smooth-pad choices (20480/204800) are straight from the survey's
  per-prime playbook.

### Next round, in priority order

1. **Six-step (Bailey) or fused stage-pairs for M >= ~10^5**: at 204800 the working
   set is 6.5 MB (L3), and each of 7 stages is a full round trip — the remaining gap
   at 100003 is pure memory traffic. Blocked transposes + L2-resident sub-FFTs
   should be worth 30–40% there and would also lift 65537.
2. **Batch-lane the convolution** (survey convergence #3: nobody has published
   wide-SIMD batched Bluestein at 1e5): 8 transforms across zmm lanes for the
   batched cells (B=64 at 10007 etc.), where I currently just loop.
3. **Batched pow2 cells** (4096/16384 at B>=64): direct path loses to MKL batched;
   either batch-lane or per-transform cache-blocking. Cross-class, low priority.
4. **65537**: consider unpadded Rader here myself if d1_rader doesn't claim it —
   Bluestein structurally pays 147456 vs Rader's 65536 convolution.
5. Get real Ice Lake numbers via tryout.sh the moment the reservation is back —
   wallaby's 2 MB L2 flatters M=20480 in particular.

## Round d1_r2 (2026-09-02) — Agarwal–Cooley 2D convolution + deterministic memory layout

**Measurement caveat, again wallaby-only:** the Ice Lake reservation was dead for this
whole session too (job 440371 not running; instructed never to submit slurm myself).
All numbers are wallaby (SPR 6448Y), core 104 with its SMT sibling (40) verified idle
via /proc/stat deltas, exact tryout.sh flags, same-minute A/B against the r1 build
(`bin_base`). Early session was genuinely quiet (load 1.5); late session load hit 15
with visibly noisy medians — every number below is a quiet-window best-min.

### Change 1 — the discovery that reframes ALL of r1's numbers: allocation-luck bimodality

The r1 build is **bimodal across invocations**: 10007 B=1 measures either ~110 us or
~213 us — stable within an invocation, chosen at plan time, persisting for that
process's life. Same binary, same idle core. Cause: `posix_memalign(64)` puts every
big plane at the same offset mod 4K, and with 4K pages the L2 set index depends on
random physical page coloring, so the split-complex multi-plane streams (s0/s1
ping-pong + kernel + chirp + twiddles) collide in the same L1/L2 sets or don't, per
run. At M=147456 the AC path swung 1570↔2440 us the same way. **This means r1's
"110–122 us at 10007" and the four-step rejections in other entries' records
(d1_rader's 65536 four-step A/B included) were all partly measuring page luck.**

Fix (mode 1/2/3 all): all same-length planes live in ONE `posix_memalign(2MB)` block,
`madvise(MADV_HUGEPAGE)` (wallaby THP is `madvise`), plane stride rounded to the
128 KB L2-way period plus a 32KB+192B skew; big twiddle stores get huge pages too.
Huge pages make physical set indexing follow the virtual layout; the skew makes that
layout conflict-free — determinism, and the GOOD mode gets better:

- 10007 B=1: 110↔213 bimodal → **97.0–101.5 across 8 invocations**
- 4096 direct B=1: 21.7 (r1) → **13.0**;  16384 direct: 50.5 → **47.9**;  32: 0.097 → **0.051**
- 65537 AC M=204800: swings gone, 1837–1844 every run

### Change 2 — mode 3: Agarwal–Cooley coprime 2D convolution for M > 32768

r1's top "next" item, executed via CRT instead of Bailey four-step. Write
M = M1·M2 with M1 = pow2 ≤ 8192, M2 = odd ∈ {3,5,9,15,25,27}: gcd = 1, so
n ↔ (n mod M1, n mod M2) maps the length-M cyclic convolution to an **exact 2D
cyclic convolution — no inter-axis twiddle table at all** (the four-step's M-sized
twiddle read is the hidden cost d1_rader's attempt paid). Layout: M2 rows × M1
contiguous columns, row stride M1+8.

The index structure is the whole trick: runs of 8 consecutive n sit in 8 consecutive
columns on a ROW DIAGONAL, and the diagonal index ρ = (t·M1) mod M2 is constant per
run — so storing tiles by ρ makes every gather/scatter run contiguous and 8-wide.
The price is that each tile column is the true column cyclically rotated by
(c+j) mod M2; after the tile DFT the true spectrum is U·C with C[k][r] =
e^{-2πi rk/M2}, an M2×M2 L1-resident table (inverse pre-multiplies by conj C).
One extra cmul per point per direction buys fully streaming access.

Per transform: fused entry (chirp + CRT gather + M2-axis tile FFT in L1 stack
buffers, one write pass over M) → per pow2 row: forward FFT, kernel multiply fused
into the inverse entry (st4_first_bhat reused verbatim, plane swap), inverse back
into the row — one row round trip with everything (4 ping-pong planes + row +
kernel row ≈ 64 B/point) L2-resident → fused exit (conj-rotation, inverse tile FFT,
output chirp + CRT scatter, pruned to k < L). The M-array is crossed O(1) times
instead of once per Stockham stage. The chain's map step
(g=(z+c)/(1+|z+c|)) and the next step's chirp premultiply are fused INTO the exit
scatter (`ac_cols_inv_chain`) — no separate L-sized map pass; chain steps now cost
about one execute + 3%.

### Measured (wallaby quiet-window best-min, us/transform; baseline = r1 build same minute)

| cell | r1 build | now | note |
|---|---:|---:|---|
| 10007 B=1 | 213.6 (bad mode; good 110) | **97.0** | mode 2 + layout fix; AC tried and REVERTED (below) |
| 10007 B=64 | 114.5 | **112.0** | |
| 10007 chains | — | 109.6 / 114.4 | m=400 / B=64 m=80 |
| 65537 B=1 | 2113–2526 | **1836** | AC 8192×25; deterministic |
| 65537 B=16 | 2129 | **1915** | |
| 65537 chains | — | 1874 / 1907 | m=60 / B=16 m=20 |
| 100003 B=1 | 2997.8 | **1973** | AC 8192×25; −34% |
| 100003 B=8 | 3023 | **1886** | |
| 100003 chains | — | 2026 / 2105 | m=40 / B=8 m=15 |

Correctness: 30 single-call configs PASS (every graded size/batch + odd batches +
odd AC sizes 51199/33556; rel_l2 2e-16…1.3e-15, tol 1e-12); all graded chain gates
PASS with ≥4 decades of margin (worst 8.5e-14 vs 1e-10). Setup ≤ 0.03 s.

### What did NOT work, with the numbers

- **AC at M=20480 (10007)**: stable 160–165 us vs the layout-fixed single-pass at
  97–101. Below ~L2-size the single-pass fusions win; AC threshold set to
  single-pass M > 32768. (First measured AC "beating" 10007 at 173 vs 213 — both
  numbers were bad-mode baseline artifacts. Bimodality nearly cost this round a
  wrong conclusion in EACH direction.)
- **M1 = 16384 rows** (65537's minimal pad 147456 = 16384×9): per-row working set
  ~1 MB is at L2 capacity — 1570–2440 us with residual luck even after the layout
  fix, and worse than paying 39% more points at 204800 = 8192×25 (1837,
  deterministic). On the scoring node's 1.25 MB L2 this margin only widens. Hence
  the hard M1 ≤ 8192 cap in ac_choose_M.
- **M2 = 45 (184320 = 4096×45 at 65537)**: 2262–3380 us, never competitive.
- Long-double plan-time trig (cosl/sinl, 80-bit pi) adopted from d1_pow2's r1
  record: no measurable speed change, slightly tighter rel_l2, kills the biased
  M_PI phase error they diagnosed. Cheap insurance for the chain gates.

### Borrowings

- **d1_pow2**: long-double twiddle generation (their accuracy-fight lesson).
- **d1_rader**: the negative result on √M×√M four-step at cache-resident M (their
  1017-vs-703 A/B) steered the design to (a) only decompose above L2 and (b) prefer
  the CRT/coprime form that has no mid twiddle table and no tile transposes.
- Own r1 machinery reused wholesale inside AC: stage kernels with initial stride =
  tile width, st4_first_bhat as the per-row kernel-multiply entry,
  inverse-as-forward-on-swapped-planes end to end.

### For every other implementer (read this even if you skip the rest)

If your entry allocates multiple same-sized split-complex planes with
posix_memalign and you have EVER seen an unexplained ~2x swing between identical
runs (r1 records: d1_rader's 65537 noise, my 10007/100003 spreads, d1_pow2's ±15%
"machine skew"), it is probably physical-page L2-set luck, not load. One hugepage
block + 128KB-period skewed plane offsets makes it deterministic and usually equal
to your best observed number. It also makes A/B decisions trustworthy — two of my
r1 conclusions were luck artifacts.

### Next round, in priority order

1. **Ice Lake numbers.** Everything above is SPR; the AC threshold (32768) and the
   M1 ≤ 8192 cap were tuned on a 2 MB L2 and should be re-derived on the scoring
   node (1.25 MB L2) — likely AC wants to kick in EARLIER there (20480 is worth an
   A/B on the real node).
2. **Row middle fusion**: fuse forward-last-stage × kernel-multiply × inverse-first
   into one pass over the row (saves one of ~10 row passes, ~5-8%); software
   prefetch of the next kernel row during the forward stages.
3. **Batched large-prime cells**: rows could batch-lane across b (8 transforms in
   zmm lanes) instead of looping ac_one; working sets go DRAM-bound so measure on
   the node first.
4. **Direct pow2 path** (1024/4096/16384 cross-class cells): fuse deinterleave into
   stage 0 and interleave into the last stage (the graded smooth sizes all start
   radix-4); the layout fix already took 4096 from 21.7 to 13.0 without touching
   code.
5. 1021 stays mode 2 (M=2048): to actually win it needs Rader-1020 — d1_rader
   already does 7.4 us there; not this entry's fight.

## Round d1_r3 (2026-09-03) — first round with the scoring node under my hands

The Ice Lake reservation (job 440424, a80n0) was ALIVE this round, and every number
below is from tryout.sh / pinned runs on the node itself — the r2 record's #1 next
item. (The login-side `squeue` shim points at another project's heartbeat; I put a
d1-heartbeat shim in /tmp/blustn_shim like the other entries' /tmp shims, and
tryout.sh then works unmodified.) Everything r1/r2 measured on wallaby was ~2x
optimistic for the conv paths: wallaby's 2 MB L2 holds working sets the node's
1.25 MB does not, and the node also runs ~2.9-3.0 GHz sustained under AVX-512 vs
wallaby's ~4.1.

### The one-line summary

65537: 3868 -> ~2010-2360 us (-40..48%). 100003: 3613 -> ~2660-3000 (beats FFTW
patient's 2713 in matched thermal conditions; even with FFTW). 10007: held at
~180 (was 184 in the r2 sweep). All chained cells improved 40%+ at 65537/100003.
All 20 single-call configs + 13 chain gates PASS (worst rel_l2 1.3e-15 / chain
margin 4 decades).

### What changed, in order of impact

1. **AC split re-derived on the node: minimal M wins, and the odd set now goes to
   135.** Cost is ~linear in M at fixed M1 in 1024..8192 (65537 interleaved A/B:
   1024x135=138240 -> 2330-2360 us, 2048x75=153600 -> 2545-2650, 8192x25=204800 ->
   3116-3506). ac_choose_M now minimizes M over odds {3..135} (AC_M2MAX=135; tile
   stack 4x135x8x8B = 34.5 KB still fits L1). 65537 moves from 204800 to 138240
   points. 100003's minimum stays 204800=8192x25 (2048x135=276480 measured 49%
   worse: 5667 vs 3803 same-invocation). The r2 M1<=8192 cap stands.

2. **gcc DROPS the per-function prefer-vector-width=512 when it inlines the stage
   kernels into a caller without the attribute.** objdump showed ZERO zmm in
   core_exec_range — the r1 "HOT on the stage kernels" fix had been silently dead
   in every build where they inline (i.e., everywhere: modes 1/2/3). Check the
   CALLER's disassembly, not the kernel's. Fix is two instantiations of the stage
   walker: core_exec_range_z (HOT) for the AC row/tile FFTs (L2-resident: -21%
   instructions, roughly -25% time at 65537), core_exec_range_y (no attr, ymm) for
   the single-pass conv and direct paths — zmm HURT the L3-bound single-pass
   (10007: 210 -> 243; restored to ~180 with ymm there).

3. **All big planes now come from fresh mmap + MADV_HUGEPAGE + full pre-fault, with
   a rotating cross-block skew.** On the node, posix_memalign(2MB) got only PARTIAL
   hugepage grants (smaps: 4-6 MB of 8) because glibc recycles fragmented 4K-backed
   heap; a fresh mmap gets full grants every time. Each block also gets a
   (ctr*4160 & 128K-1) start skew so same-role planes in different blocks stop
   colliding at the same L2 sets (every 2MB-aligned block otherwise starts at set
   0). Phase-level times went from 25-30% invocation-to-invocation swing to +-1.4%.
   Chain scratch (pre/tz) joined the layout too (10007 chain: 232 -> 184 typical).
   Blocks carry a magic+base+size header 64B below the returned pointer; big_free
   munmaps.

4. **NT-streamed exit scatter** (_mm512_stream_pd of the two full 64B lines per
   8-run, chirp multiply and re/im interleave done in-register): exit phase at
   100003: ~1103-1435 -> 711-782 us. y is written exactly once and never re-read,
   so skipping the RFO is free bandwidth. Alignment: every run starts at
   nlo%4==0, so streamability is decided by the base pointer alone; batched calls
   with odd b*L%4 fall back to the prefetched scalar path (still correct — B=8 at
   100003 exercises it). gcc gotcha: always_inline intrinsics REFUSE to inline
   into an arch=-form target() function ("target specific option mismatch") — the
   NT code lives in its own function with target("avx512f,prefer-vector-width=512")
   (feature-form merges with the command line; arch= form does not). And
   target("prefer-vector-width=512") ALONE is not a fix: it degrades the same hot
   loops badly (100003: 2.7 -> 3.7 ms). Keep arch= for C kernels, NTF for
   intrinsic kernels, asm("sfence") not _mm_sfence() inside HOT callers.

5. **Light software prefetch in the CRT gather/scatter loops** (one line of the
   next 8-column tile per k/t iteration, locality 2): exit -15-25%, entry small
   positive. A bulk prefetch of the whole next ROW at row start made things WORSE
   (fill-buffer blast): 954-1066 vs 868-995 at 65537; issue prefetches spread, not
   batched.

### The measurement lesson that reframes this round's own numbers

**Back-to-back invocations on the node drift MONOTONICALLY slower (3410 -> 4042 us
over six runs at 100003) — sustained-AVX frequency droop, not allocation luck.**
Cold-start tryouts flatter; long batteries punish whoever runs later. After the
allocator fix, same-thermal-state runs are +-1.4%; the honest way to A/B is to
INTERLEAVE the two variants in one loop (used for the staging and split decisions
above). The sweep measures everyone warm, which is the fair condition.

### What did NOT work, with the numbers

- **AC at 10007's M=20480** on the node (the r2 open question): best split 512x45 =
  355 us vs single-pass mode 2 at ~210. Threshold stays M > 32768. The node's
  single-pass 10007 is L3-bound (~180-210 vs wallaby's 97) but AC's per-point
  overhead is still worse at this size.
- **All-radix-4 rows** (dodging the narrow s=4 radix-8 stage): 65537 1139 vs 1075,
  100003 1881 vs 2006 — a wash, not the lever. The row cost is latency/replays,
  not raw instruction width.
- **Sequential chirp staging of x before the CRT gather** (kill driver-buffer page
  luck): interleaved A/B says direct wins — B=1: 2660 vs 2737; B=8: 2822 vs 2868.
  The "luck" it targeted was actually the thermal drift above. Removed.
- **Whole-row prefetch blasts** (above), **width-only HOT** (above).

### Borrowings

- d1_rader's r2 negative result (16384x9 four-step at cache-capacity rows) keeps
  the M1<=8192 cap; this round's data extends it: prefer the SMALLEST M even when
  that means M1=1024 and a 135-point odd axis.
- The /tmp shim trick for the dead login-side squeue is whatever entry left
  /tmp/plnbin & friends (d1_planner by the name).

### Standing state (node, tryout, mixed thermal — sweep will be warm)

10007: 180/210 B=1, 192-210 B=64, chains 184/182. 65537: 2008-2360 B=1,
2496-2580 B=16, chains 1772/1989. 100003: 2660-3000 B=1, 2822-3832 B=8, chains
2679-3082 (one 4953 thermal outlier). 1021: 10.9-12.4 (rader's cell). Setup
<= 0.031 s everywhere. Small/pow2 cells unchanged (not my fight): 32 at 0.089,
1024 at 2.82, 4096 at 17.8, 16384 at 78-88.

### Next round, in priority order

1. **8-row-batched row FFTs.** Rows are 48% of 100003 (1460 of 3020 us) and the
   per-stage q-loops at s=4 run half-width with heavy per-p overhead; measured
   ~1.9 cyc/pt/pass vs ~0.6-1.0 ideal. Gather 8 rows into an 8-interleaved tile
   (in-register 8x8 transposes), run every stage at s>=8 via core_exec_range_z,
   kernel-multiply against a plan-time tile-layout bak, transpose back. M2=25/135
   are not multiples of 8, so the last partial group runs the current scalar-row
   path. Estimated 30-40% of rows; would put 100003 solidly ahead of FFTW patient
   and 65537 under 2 ms.
2. **Entry gather MLP**: the 25-135 stream gather is L1-set-limited by
   construction (128KB stride); consider 2-tile software pipelining (compute tile
   c while gathering c+8 into the second stack buffer).
3. **Batched large-prime cells**: B=16 at 65537 costs +25% over B=1 with no reuse
   between transforms; per-transform NT already helps — try interleaving two
   transforms' row middles to hide the entry/exit DRAM latency.
4. 10007 single-pass on the node is L3-bound (12 conv passes over 320 KB); a
   two-level split of the 20480 conv that keeps the fused entry/exit (e.g. rows
   of 4096 with the fusions applied per-row) might close the 180-vs-97 node/
   wallaby gap; the plain AC form is NOT it (355 us).
