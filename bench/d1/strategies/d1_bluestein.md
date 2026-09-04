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

## Round d1_r4 (2026-09-03) — one-pass fused middle, paired-p s=4 radix-8, radix-9 tiles, Newton map

Every number in this section is from the scoring node (a80n0, job 440424) via tryout.sh
or its exact pipeline run manually over ssh (slot lease, core 4). The login-side squeue
shim was dead again — same fix as r3, a /tmp/blustn_shim/squeue keyed to the d1
heartbeat. The node was BUSY with other implementers for the second half of the session
(load 5–14), so decisions were made on same-window INTERLEAVED A/B ratios (the r3
thermal lesson, which held for a fourth round); best-observed numbers are from one quiet
window (load 0) mid-session.

### The one-line summary

Same-window interleaved vs the r3 build: 10007 −21% (124 us B=1), 65537 −11% (1560,
quiet-window 1535), 100003 −11% (2399, quiet-window 2302), 10007 B=64 −17% (147),
65537 B=16 −11% (1767), 100003 B=8 −11% (2726), all chains −5..−22%. The direct pow2
path came along for free: 1024 2.43→1.86, 4096 15.8→13.2, 16384 68.3→59.1. 100003
now beats FFTW patient (2687) at every cell; 65537 closes to ~5–7% behind patient
(1463) from 17% behind. All 26 single-call configs and 10 chain gates PASS
(worst single 1.3e-15, worst chain margin >3 decades); output bit-repeatable.

### What changed, in order of impact

1. **Paired-p zmm radix-8 at the narrow s=4 stage (st8_s4)** — d1_rader's r3
   st16_s4/st3_s4 trick applied to my radix-8. Every conv/row/direct plan here runs
   its second stage at s=4, where the q-loop is 4-wide with per-p scalar twiddle
   loads; r3 had measured rows at ~1.9 cyc/pt/pass against 0.6–1.0 ideal, and this
   stage was the reason. Two p-groups per zmm (lanes 0–3 = p, 4–7 = p+1): pair
   inputs are contiguous 8-double runs, twiddles are pair-broadcast
   (permutexvar of a 128-bit load), outputs recombine to full contiguous zmm
   stores via 128-bit-lane shuffles (0x44/0xEE) — no masked half-stores at all.
   NTF function dispatched from BOTH core_exec walkers (case 8, s==4, m even).
   This was the round's big lever: it is most of the −20% at 10007 and the pow2
   direct-path gains, and applies to fwd+inv in every AC row. The r3 worry that
   zmm hurts the L3-bound single-pass path did NOT materialize for this one stage
   (10007 improved 20% with it dispatched from the ymm walker too).

2. **Fused middle: forward-last x kernel-multiply x inverse-entry in ONE pass
   (stmid2/stmid4, modes 2 and 3)** — executes d1_rader's r3 "next round" idea #1,
   which their record left untried. The key observation: the last forward stage has
   m = 1, so its twiddles are ALL 1, and its output blocks (k = q + j*s) line up
   exactly with the radix-4 inverse entry's reads (p + u*M/4) — for rl=4 they are
   the same index set at p=q; for rl=2 the two butterflies at q=p and q=p+M/4
   produce them. So the spectrum is never materialized: butterfly, R*W with the
   re/im swap, and the inverse stage-0 butterfly happen in registers, saving one
   full plane-pair write + read per transform (mode 2) / per row (mode 3).
   Measured alone (before st8_s4), same-minute interleaved: 10007 156→148 (−6%),
   100003 2590→2533 (−2.5%), 65537 wash. Applies whenever the last radix is 2/4
   (all AC rows; mode-2 M=20480; NOT 2048=[4,8,8,8], so 1021 keeps the old path).

3. **Radix-9 tile stages (st9, AC odd axis only)** — an AC_PHASES env-gated phase
   timer (left in, default off) showed 65537 = entry 745 + rows 547 + exit 610 us:
   the M2=135 tile FFTs dominate, not the rows. st9 does 3x3 Cooley-Tukey in
   registers (6 DFT3s + 4 inner W9 cmuls), turning 135 = [3,3,3,5] into [9,3,5] —
   one fewer full pass over the tile per entry AND per exit. Phase split moved to
   entry 590 + rows 522 + exit 500 (measured under load, so if anything
   understated). 100003 (M2=25) unaffected. Emitted only in the c2 (tile) plan
   via a use9 flag on core_factor, so row/conv/direct factorizations are unchanged.

4. **Newton 2NR map in every mode-2/3 chain exit** (chain_map8_pre/out for the AC
   scatter runs, chain_map_pass for the mode-2 fused chain): rsqrt14+2NR and
   rcp14+2NR on the FMA ports instead of vsqrtpd+vdivpd serializing on the divide
   unit, max(m,1e-300) guarding m=0. Adopted from d1_rader r3 (chain_map8) and
   d1_pow2/d1_batchlane (the shipped "2NR-only fast map"). Chain gates moved in the
   second digit only (10007 m=400: 8.0e-14 vs 1e-10 tol). Chains improved ~9% at
   10007 pre-st8_s4; final chain gains vs r3 are −22%/−21% (10007), −11%/−5%
   (65537), −7%/−15% (100003).

### Measured (a80n0; interleaved same-window old(r3)/new min us, warm rep; plus quiet-window best)

| cell | r3 build | now | quiet best | lib best (r3 lb) |
|---|---:|---:|---:|---:|
| 10007 B=1 | 158.1 | 124.0 | 123.3 | 198.4 patient |
| 10007 B=64 | 177.7 | 147.2 | — | 207.1 patient |
| 65537 B=1 | 1760.7 | 1559.6 | 1535 | 1463.4 patient |
| 65537 B=16 | 1987.8 | 1767.4 | 1718.8 | 1535.6 patient |
| 100003 B=1 | 2707.1 | 2398.8 | 2301.7 | 2686.7 patient |
| 100003 B=8 | 3060.3 | 2726.5 | 2562.9 | 2808.8 patient |
| 10007 ch 1:400 / 64:80 | 173.9 / 185.0 | 136.3 / 146.2 | — | 225.7 / 239.8 |
| 65537 ch 1:60 / 16:20 | 1846.6 / 1977.5 | 1646.7 / 1873.9 | — | 1627.6 / 1740.2 |
| 100003 ch 1:40 / 8:15 | 3063.4 / 3283.7 | 2856.6 / 2795.6 | — | 3113.8 / 3195.6 |
| 1024 / 4096 / 16384 B=1 | 2.43 / 15.8 / 68.3 | 1.86 / 13.2 / 59.1 | — | 1.08 / 19.1(r1) / 32.3 |

Correctness: 26 single-call configs PASS (all graded sizes/batches, odd batches
3/5/2, odd AC sizes 51199/33556, every size st8_s4/st9/stmid touches: 128, 1021,
2048, 8192); 10 chain gates PASS (worst margin 1021 m=2000 at 2.8e-12 vs 1e-9);
bit-repeatable. Setup unchanged (<= 0.03 s).

### What did NOT work / non-results, with numbers

- **AC split re-check at 65537 after st8_s4** (rows got cheaper, so the minimal-M
  choice could have shifted): 1024x135 = 1541/1560, 2048x75 = 1674/1702,
  4096x45 = 2098 — minimal M still wins, margin unchanged. Keep ac_choose_M as is.
- **The fused middle at 65537 was a wash** in both single and chain cells (1705→1710,
  1727→1729 same-minute) — consistent with the phase split: 65537's rows are only
  ~29% of the time, and one pass of ~8 row passes is noise there. It still ships
  (it clearly pays at 10007/100003 and costs nothing at 65537).
- **Newton map at 65537 single-chain was also ~flat** pre-st8_s4 — the exit scatter
  is bandwidth-bound there and the sqrt/div latency was already hidden. The gains
  showed up at 10007 (compute-bound conv) and in the batched chains.
- Not attempted this round, for the record: 8-row-batched row FFTs (r3 #1) — at
  100003's M1=8192 an 8-row tile is ~2–3 MB (past the node's 1.25 MB L2), so the
  design only fits 65537's M1=1024; st8_s4 took the cheaper half of the same win.

### Borrowings (all from d1_rader r3 unless noted)

- st8_s4 paired-p shape = their st16_s4/st3_s4, generalized to radix 8 exactly as
  their r2 record predicted ("generalizes to radix 8").
- The fused middle executes their r3 next-round idea #1 ("fuse the kernel-multiply
  into the LAST forward stage; needs the twiddle algebra checked") — the check is
  that the last stage's twiddles are all 1, which makes it clean.
- Newton 2NR map: d1_rader chain_map8 + d1_pow2/d1_batchlane's shipped 2NR fast map.
- st9's fuse-two-layers-per-pass idea is their st16 at radix 3.

### Next round, in priority order

1. **65537 entry/exit are still 70% of the cell** (entry 590 / exit 500 vs rows 522
   after st9). Ideas: fold the conj-rotation premultiply into the inverse tile
   FFT's FIRST stage twiddles (needs per-cm tables or algebra on C[k][r]);
   software-pipeline the CRT gather two tiles deep (r3 #2, still untried);
   radix-15 tile stage ([9,15] = 2 passes for 135) if register pressure allows.
2. **65537 B=16 (1767 vs patient 1536)**: only 1/4 of batch outputs hit the NT
   scatter (yd alignment cycles mod 4 with L odd — 65537*16B ≡ 16 mod 64). A
   64B-aligned bounce row + streamed copy, or masked-edge NT, would NT all of them.
3. **100003 rows (still 1320 of ~2400)**: two-row interleaving to hide the bak-row
   L3 latency (bak = 3.3 MB at M1=8192, read once per row from L3); or the L1-tile
   radix-16/32 first-stage pair to cut 2 of 9 passes.
4. 10007 B=64: mode-2 exit (st2_last_chirp) NT-store variant for the 19.5 MB
   batched working set — same trick as exit_scatter_nt, alignment cycles mod 4.
5. The AC_PHASES timer is in and env-gated — use it before betting on any phase.

## Round d1_r5 (2026-09-03) — radix-16 schedules by flavor, stmid8, NT batched exits; radix-15 killed by measurement

Numbers are from the scoring node (a80n0, job 440424) via tryout.sh or its exact
pipeline over ssh (slot lease, core 4). The login-side squeue was dead again — the
same /tmp/blustn_shim fix as r3/r4. The node ran LOADED for most of the session
(other implementers, load 3–9), so every decision below is a same-window
INTERLEAVED A/B (one binary, env-gated variants: AC_R15 / BLU_NONT / BLU_NO16);
the absolute numbers are warm-rep minima under that load and should read a bit
better in a quiet sweep.

### The one-line summary

Interleaved vs the r4 build: 10007 −11.5% B=1 (109–110 us), 1021 −11% (7.8),
100003 −3.7% B=1 (2219) and −12% B=8 (2385), 10007 B=64 −24% vs r4-interleaved
(112), 65537 B=16 −5% (1680), 65537 B=1 unchanged (~1530); direct pow2 4096 −9%
(12.1), 16384 −8% (56), 4096 B=256 −13% (15.6), 16384 B=64 −14% (65). Chains:
100003 m=40 2315 (r4 2857, −19%), 65537 m=60 1563–1592, 10007 m=400 121–136,
10007 B=64 m=80 130. All 28 single-call configs and 12 chain gates PASS (worst
single 1.1e-15, worst chain margin >2.5 decades); bit-repeatable on the node.

### What changed, in order of impact

1. **Radix-16 stages (PORTED from d1_rader — their st16/st16_s4 verbatim),
   with a per-flavor factorization.** The port was near-free because their
   kernels are built on my own r1 Stockham conventions (their r1 borrowing
   coming home). The key structural idea their code carries: bounce layer 1
   through an aligned 2 KB stack tile so only 8 lane-vectors are live —
   exactly the fix my radix-15 attempt lacked (below). core_factor now takes
   a flavor:
   - CF_ROWS (AC row FFTs): when the exponent lands on 16^k*8, take it and
     fuse the trailing 8 (new stmid8): 8192 = [4,16,16,8] runs SEVEN passes
     per row instead of nine. 100003 (M1=8192, rows were 41% of the cell):
     B=1 2219 vs 2306 (−3.7%), B=8 2385 vs r4 2726. M1=1024 keeps [4,8,8,4]
     (st8 is pure-register; st16's L1 bounce loses at equal pass count), so
     65537 B=1 is unchanged — its gain this round is elsewhere.
   - CF_CONV (mode-2): 16s only while a 2/4 tail survives for the pruned
     chirp exit + fused middle: 20480 = [4,16,16,5,4] (5 stages, was 6) —
     10007 B=1 109–110 vs 123–125 (−11.5%). 2048 = [4,16,8,4], which for the
     FIRST time gives 1021 a fused middle and pruned fused exit ([4,8,8,8]
     had neither): 7.8 vs 8.6–9.0 (−11%).
   - CF_DIRECT (mode-1 pow2): minimize stages outright — but ONLY for
     a >= 10: 4096 = [4,16,16,4] (12.1 vs 13.3), 16384 = [4,16,16,16] (56 vs
     60.5). 1024 = [4,16,16] measured WORSE (2.22 vs 1.90, +17%): L2-resident,
     pass count moot, the tile bounce is pure cost. Gate a >= 10.
2. **stmid8** — the r4 fused-middle idea one radix level up: a forward-last
   radix-8 butterfly's even-j outputs are exactly the radix-4 inverse entry's
   reads at p = q, its odd-j outputs those at p = q + M/8, so radix-8 DFT +
   8 kernel muls (with the inverse plane swap) + TWO twiddled radix-4 inverse
   entries run in one pass and the spectrum never touches a plane. This is
   what lets CF_ROWS end in 8 at all.
3. **NT-streamed batched final stores (modes 1+2)** when the batched output
   is >= 8 MB: y is written once and never re-read, so streaming stores skip
   the RFO. The mode-2 exit and the mode-1 interleave are CONTIGUOUS, so
   misalignment is solved by peeling scalar head/tail complexes until the
   store address is 64B-aligned and reading all sources loadu — EVERY batch
   member NT-streams, unlike the mode-3 scatter (only 1-in-4 aligned).
   st2_last_chirp_nt + st4_last_chirp_nt (conv) and interleave_nt (direct).
   Measured: 16384 B=64 65.2–72.7 vs 77.4–86.2 (−11..16%), 4096 B=256
   15.6–17.1 vs 18.1–19.7 (−13%), 10007 B=64 best 124.4 vs 132.0 (6 of 8
   reps favored NT ~−5%; two thermal reps went the other way). NT is
   force-disabled inside the chain fallback loops (chain_y is re-read).
4. **Newton-map clamp 1e-300 → 1e-100** (borrowed: d1_batchlane r3 via
   d1_twiddle r4): rsqrt14 of a denormal-range operand is an FP assist
   (~250 cycles/lane). Chain gates moved in the second digit only.

### What did NOT work, with the numbers (both fully measured, both reverted)

- **Radix-15 Good-Thomas tile stage (st15)**: 135 = [9,15] — 5 DFT3s + 3
  DFT5s, internally twiddle-free by the coprime CRT maps, one tile pass
  fewer, and as last stage its output twiddles are all 1. Interleaved A/B at
  65537 B=1: 2126/2130/1897 (st15) vs 1776/1526/1592 ([9,3,5]) — 20–40%
  WORSE. 15 live complex = 30 zmm plus temps overruns the register file and
  gcc's spills cost more than the saved pass. The fix d1_rader's st16 uses
  (L1 tile bounce between layers) would apply here too — untried. Ships
  DISABLED behind AC_R15; the CRT index maps are in the code for whoever
  tries the tiled form.
- **Interior-line NT for the MISALIGNED mode-3 exit scatter** (r4 next-item
  #2): per 128B run, mask-store the head/tail lanes and stream the one fully
  covered 64B line (runtime-index permutex2var, since valignq wants an
  immediate). Catastrophic: 65537 B=16 2864–3133 vs 1665–1746, 100003 B=8
  4492–4799 vs 2492–2553 — 1.7–1.8x SLOWER. Mixing RFO masked stores and NT
  lines per run defeats both write paths. Misaligned members keep the plain
  scatter; code removed, numbers kept here so nobody re-derives it.
- Non-result worth recording: the FIRST 10007-B=64 NT A/B was a perfect wash
  (131.1 vs 131.4) — because the 16 MB gate excluded the cell and both arms
  ran identical code. If an A/B is suspiciously flat, check the gate before
  trusting the conclusion.

### Borrowings

- st16/st16_s4 ported near-verbatim from d1_rader (their r3 kernels; the
  L1-tile two-layer structure is the round's key import). stmid8 extends my
  own r4 stmid2/4 (itself their r3 idea) to the trailing-8 schedule.
- The 1e-100 clamp: d1_batchlane r3, re-confirmed by d1_twiddle r4.
- NT final-stage stores for mode-1 batched: d1_pow2 r3 / d1_twiddle r4
  (their −17% at 16384 B=64 reproduced at −11..16% here); the input-offset
  realignment trick for misaligned batch members is new here.

### Standing state (node, warm reps under load 3–9; sweep should read equal or better)

10007: 109–110 B=1, 112 B=64, chains 121–136 / 130. 65537: ~1530 B=1, 1675–
1687 B=16, chains 1563–1592 / (B=16 not re-timed). 100003: 2219–2225 B=1,
2383–2399 B=8, chain 2315. 1021: 7.8 B=1, 9.2 B=256 (rader's cell). Direct:
1024 1.87, 4096 12.1, 16384 56; 4096 B=256 15.6, 16384 B=64 65.2. Setup
<= 0.031 s. Small cells untouched.

### Next round, in priority order

1. **65537 is now clearly the weak headline** (~1530 vs d1_rader 915, patient
   1463) and its rows gained nothing this round (M1=1024 too short for the
   trailing-8 trick). Two concrete options: (a) stmid16 — fuse a trailing
   radix-16 ([4,16,16] rows = 5 passes vs 7) via the same even/odd-class
   algebra (one 16-butterfly feeds four inverse radix-4 entries); register
   pressure says build it ON the st16 L1-tile pattern, not flat; (b) a tiled
   st15 (the L1-bounce form) to cut the 135-tile from 3 passes to 2 — entry+
   exit are 59% of the cell.
2. **65537 B=16 misaligned members**: the interior-line NT failed, but a
   64B-aligned bounce ROW + full-line NT copy pass was never measured — the
   analysis said roughly break-even vs L3; measure it in a quiet window.
3. **100003 rows** are still 41%: two-row software pipelining of the g-row
   read (the bak row is streamed by stmid8 already); or check whether
   ac_choose_M wants 4096x27=110592 now that rows got cheaper per point at
   M1=8192 but 4096 keeps [4,8,8,4] (the M-minimum vs schedule interaction
   changed this round).
4. AC_PHASES is still in — re-derive the 65537 phase split on a quiet node
   before betting on (1a) vs (1b).

## Round d1_r6 (2026-09-03) — the last 4K-page table, tiled radix-15, radix-64 rows; two prefetch ideas killed by measurement

Numbers are from the scoring node (a80n0, job 440424) over ssh on a leased core,
tryout.sh's exact build. The login-side squeue was dead AGAIN (fourth round) —
same /tmp/blustn_shim fix; note the PATH already carries ANOTHER project's
squeue shim (gen_planner's, keyed to THEIR heartbeat), which is why the failure
looks new each round. The node ran loaded (other implementers, load 0.5–6.4)
for most of the session, with the window level swinging ±12–15% (65537 B=1 read
1485 in the best window and 1680 in a "quiet" one an hour later) — every
decision below is a same-window interleaved A/B or an env-gated variant of one
binary; the absolute table at the end is best-window-observed.

### The one-line summary

The round's real win was a VARIANCE kill, not a mean kill: the AC rotation
table C[k][r] was the last hot array still on 4K pages, and moving it into the
skewed hugepage layout collapsed 65537 B=1 from a 1545/1950 us per-process
bimodality to 1485–1517 over 8 invocations (median == min). 100003 B=8 now
reads 2379–2395 across processes (r5 board: median 3249, best 2377 — the
median should collapse to the best). On top of that: tiled radix-15 (−2–3% at
65537 via [9,15] tiles) and d1_rader's radix-64 rows ([4,64,4] at M1=1024,
wash when quiet / better under load). All 35 single-call configs and 14 chain
gates PASS (worst single 1.1e-15, worst chain margin 2.5 decades), bitwise
repeatable.

### What changed, in order of impact

1. **ctr/cti (the AC conj-rotation table) moved from posix_memalign into a
   planes_alloc hugepage block (new blkD).** At M2=135 this table is 309 KB
   and is read once per tile in BOTH entry and exit; as the lone amalloc'd hot
   array it re-rolled its L2-set coloring every process — measured this round
   as a clean two-mode 1545/1950 lottery at 65537 B=1 (same binary, same core,
   stable within a process). After the move: 1485.2–1517.5 over 8 back-to-back
   invocations, medians equal to mins, and the best mode got BETTER than
   either old mode. Under the 9-run-median scoring this is worth more than any
   kernel work this round. Lesson recorded for everyone: grep your plan for
   the LAST amalloc/malloc'd table that the hot loop touches — r2/r3 hugepaged
   the big planes and the twiddles, and this one 300 KB table quietly kept the
   lottery alive at exactly one size class (100003's M2=25 table is 13 KB and
   never mattered).

2. **Tiled Good-Thomas radix-15 (st15_block), CF_TILE now defaults to
   [9,15].** The r5 flat st15 lost 20–40% to register spills; the fix is
   exactly d1_rader's st16/st64 tile-bounce shape: layer 1 = five DFT3
   columns through an aligned 1.9 KB stack tile (CRT input map
   u = (5n1+3n2) mod 15), layer 2 = three DFT5 rows off the tile (output map
   k = (10k1+6k2) mod 15), zero inner twiddles, <= 10 live lane-vectors per
   sub-loop. M2=135 tiles run 2 passes instead of 3 in both entry and exit.
   Same-window phase A/B at 65537 B=1 (BLU_NO64=1 both arms): entry 637–648
   vs 659–663, exit 569–578 vs 577–590 — about −25–30 us of ~1500 (−2%).
   Small because entry/exit are gather/scatter-bound, not tile-FFT-bound: ONE
   tile pass is only ~20–30 us of a ~640 us entry. Correct at m=1 (135, 65537
   graded; 33556 odd) and m>1 with twiddles (75 = [15,5] via L=19000).
   AC_NO15 restores [9,3,5].

3. **Radix-64 two-layer stages PORTED from d1_rader (r5 st64/st64_s4/dft8v,
   near-verbatim; their kernels are built on my r1 conventions so the port is
   mechanical).** CF_ROWS now takes 64 when the pow2 budget after the entry-4
   is 8..10: M1=1024 = [4,64,4] (5 row passes with stmid4, was 7 with
   [4,8,8,4]), 2048 = [4,64,8], 4096 = [4,64,8,2]. Measured at 65537
   (M1=1024, the only graded user): rows phase 450.7 (64) vs 443.7 (8s) in a
   quiet window — the two saved passes are fully eaten by the L1 tile bounce,
   confirming the r5 st16-at-1024 lesson one radix up. BUT under load 6+ the
   64-form was never worse and dodged the 8-form's bad outliers (610–619 vs
   617–698 over 5 alternations), and it reads less twiddle traffic, so it
   ships default-ON. BLU_NO64 for A/B. M1=8192 (100003) keeps [4,16,16,8] —
   64 cannot shorten that schedule.

### What did NOT work, with the numbers that killed it (both fully reverted)

- **Short-lead exclusive-hint scatter/gather prefetch (d1_rader's r5 −15%
  B=16 trick), as a REPLACEMENT for my long-lead form**: exit_scatter
  prefetching the run 2 t-iterations ahead with write/locality-3, entry
  gather with a batch-gated T0 2 runs ahead. 65537 B=16: 1618 -> 1728 (+7%);
  100003 B=8: 2390 -> 2715 (+13.5%), reproducible 3/3. A t-iteration is ~8
  cmuls — far too short a lead for a DRAM RFO — and my r3 prefetch already
  runs ONE C-BLOCK ahead (the whole tile FFT of lead time). Rader's win came
  from having no prefetch at all on that pass; mine was already load-bearing.
  Lesson: before importing a prefetch result, check the lead time you already
  have, not just the hint.
- **Spread prefetch of the next row's bak inside the stmid loops** (my own r5
  next-item #3): 100003 B=1 went 2900 -> 3440 (+18%). A __builtin_prefetch
  inside a `#pragma GCC ivdep` loop DEVECTORIZES the whole fused-middle pass
  in gcc-11 — and bak is a sequential stream the L2 hardware prefetcher
  already covers. Never put prefetches inside a loop you need vectorized;
  strip-mine or leave it alone.

### Diagnosed but NOT fixed (the top next-round item)

**10007 B=64 is still a per-process lottery: 4 invocations at ~150 us, one at
112.3, same binary, same core, within a minute.** Mode 2's plan buffers are
all hugepaged, so the remaining coin is the DRIVER's 10+10 MB in/out buffers
(4K pages, random coloring vs my fixed-layout L2-resident conv planes; the
r5 board median 149 vs best 111.7 is exactly this two-mode split). My side
cannot re-color the driver's pages at plan time; the known fix is d1_race's
r5 wisdom-referenced first-call probe (re-roll own layout against the real
buffers until the draw is good). Either adopt a light version in-plan
(first-execute probe + one arena re-roll), or accept that d1_race's routing
layer already does this on top of my engine.

### Measured (a80n0, bin at final source; best window / typical loaded window)

| cell | r6 best-window | r5 board (median/best) | lib best (r5) |
|---|---:|---:|---:|
| 10007 B=1 | 112–124 | 110.6 / 109.8 | 200.0 patient |
| 10007 B=64 | 112.3 (good draw; bad 150) | 149.1 / 111.7 | 208.3 patient |
| 10007 ch 1:400 / 64:80 | 181 / 130.7 (loaded) | 152.0 / 131.4 | 230.6 / 237.2 |
| 65537 B=1 | **1485–1518 x8 procs** | 1530.8 / 1521.4 | 1466.9 patient |
| 65537 B=16 | 1609–1677 | 1677.7 / 1666.7 | 1538.7 patient |
| 65537 ch 1:60 / 16:20 | 1525 / 1826 (loaded) | 1624.9 / 1833.4 | 1628.3 / 1763.2 |
| 100003 B=1 | **2218.8 (loaded!)** | 2563.9 / 2267.7 | 2715.5 patient |
| 100003 B=8 | **2379–2395 x5 procs** | 3249.1 / 2376.8 | 2805.7 patient |
| 100003 ch 1:40 / 8:15 | 2319 / 2710 | 2312.7 / 2727.8 | 3116.0 / 3169.4 |
| 1021 B=1 / B=256 | 7.84 / 8.10 | 8.91 / 8.83 | (rader's cell) |
| 4096 / 16384 B=1 | 12.1 / 56.7 | 12.2 / 56.3 | unchanged, not my fight |

Correctness: 35 single-call configs PASS (all graded sizes/batches, odd
batches 3/5, odd AC sizes 51199/33556/27001/19000 exercising every new path:
st15 m=1 and m=5, [4,64,4]/[4,64,8]/[4,64,8,2] rows), worst rel_l2 1.09e-15
(tol 1e-12); 14 chain gates PASS (worst 31:512:1200 at 2.79e-12 vs 1e-9),
all chained outputs bitwise repeatable across processes. Setup <= 0.03 s.

### Borrowings

- st64/st64_s4/dft8v: d1_rader r5, near-verbatim (their r3 borrowing of my
  Stockham core coming home a second time; st16 was the first).
- The tile-bounce idea that fixed radix-15: d1_rader's st16/st64 two-layer
  shape (the r5 record explicitly flagged "the fix d1_rader's st16 uses would
  apply here too — untried"; it applied cleanly).
- The variance-first framing (medians are score): RESCORE_PLAN.md via
  d1_pow2's r5 record; the diagnosis method (N invocations of one binary,
  look for modes not spread) is my own r2 protocol.
- Negative import that saved nothing but is now measured: d1_rader's r5
  short-lead ET0 scatter prefetch does NOT transfer to a loop that already
  has a long-lead prefetch (numbers above).

### Next round, in priority order

1. **10007 B=64 (and any mode-2 batched cell): kill the driver-buffer
   lottery.** Options measured-cheapest first: (a) first-execute probe + one
   arena re-roll (adopt d1_race's ref idea in miniature); (b) strip-mined
   NTA prefetch of x in the entry pass so streamed x stops evicting the L2
   sets that hold the conv planes (must be OUTSIDE the vectorized loop —
   today's devectorization lesson).
2. **65537 entry/exit are still ~70% of the cell** and are gather/scatter
   bound, not tile-FFT bound (a tile pass is only 20-30 us of 640): the
   remaining structural idea is fusing the CRT gather INTO the first tile
   stage (st9 reading x directly via the inverse acrow map) and the rotation
   into the last (st15 storing to g with per-lane C[k] loads) — saves two
   full tile traversals per direction; bin-index algebra is worked out in
   the r6 session notes (bin k = q0/8 + 9*k_ at the [9,15] last stage).
3. **65537 B=16 bounce-row NT for the 3-of-4 misaligned members** — still
   unmeasured (r5 #2), analysis says break-even; only worth a quiet window.
4. If st64 ever looks suspect at M1=2048/4096 odd sizes, BLU_NO64 A/Bs it in
   one invocation; graded sizes only exercise M1=1024/8192.

## Round d1_r7 (2026-09-03) — two well-measured NEGATIVES: blocked entry/exit, and the placement probe; the r6 "lottery" was contention, not page luck

Numbers are from the scoring node (a80n0, job 440424) over ssh on a leased core,
tryout.sh's exact build. The login-side squeue was dead again (fifth round) — the
same /tmp/blustn_shim heartbeat fix as r3–r6. The node was mostly QUIET this
session (load 0.2–1.9, one window with a sibling python3 at 334% CPU), which is
itself the round's headline finding (below). Both changes this round measured as
losses/neutral and ship DISABLED behind env flags; the engine that ships is
functionally the r6 engine, re-confirmed at every graded cell with zero
regression.

### The one-line summary

Nothing got faster, but two plausible ideas were killed with numbers and one r6
diagnosis was overturned: the "10007 B=64 150-vs-112 per-process lottery" and the
"100003 median 2554 vs best 2269" spread that r6/the r6 board attributed to
driver-buffer page coloring are NOT placement luck — on a quiet leased core the
engine is DETERMINISTIC (10007 B=64: 111–113 over 8 invocations; 100003 B=1:
2219–2234 over 10). The r6 bad modes were busy-SIBLING contention and thermal
drift across a loaded 9-run battery (my own r3 thermal lesson). A first-contact
placement probe (adopted from d1_race r4) therefore has nothing to escape and
ships OFF. All 12 single-call + 6 chain gates re-checked PASS (worst single
1.16e-15, worst chain 9.4e-14 vs 1e-10), bit-repeatable.

### What was tried, and the numbers that killed each (both reverted, env-gated)

1. **Blocked entry/exit: NC 8-column tiles per gather/scatter pass (AC_NC=n,
   default OFF via acnc=1).** The r6 phase data said entry/exit are
   gather/scatter-bound, not tile-FFT-bound: the single-tile loops switch
   streams every 128 B, and at M1=8192 the x gather strides exactly the 128 KB
   L2-way period (every run in the SAME sets). The idea: process NC tiles per
   pass so every x run is NC*128 B contiguous, every g/y run NC-wide, NC× fewer
   stream switches and NC× more L2 sets touched per strided visit. Built the full
   thing (ac_entry_blk_x / ac_exit_blk_y / ac_scatter_blk[_nt], NC live tile
   stacks, long-lead prefetch of the next run outside the vectorized loops).
   MEASURED SLOWER in every clean interleaved pair on the node:
   - 65537 B=1 (M2=135, NC=2): BLK 1632/2258/2288 vs single-tile 1487/2145/1714.
   - 100003 B=8 (M1=8192, M2=25, NC=8): BLK8 2592/3685/3700, NC4 2580/3461/2598
     vs single-tile 2391/3258/3258 — old wins every pair.
   Why: the NC live tile buffers (2*NC*M2 lane-vectors on the stack) plus the
   wider gather working set evict the M1-sized row-FFT ping-pong planes from L2,
   and that costs more than the stream-switch reduction saves. The single-tile
   form was already well-tuned. AC_NC=n left in for whoever wants a streaming
   software-pipeline variant (compute tile q while gathering q+1), which is the
   only version that might pay — the naive all-tiles-live form does not.

2. **First-contact placement probe (BLU_PROBE=1, default OFF).** Mechanism from
   d1_race r4, moved in-plan: the driver allocates in/out on 4K pages BEFORE
   fft1d_create and runs discarded warmups, so the first (untimed) execute/chain
   call carries the REAL scored buffers. fft1d_execute/fft1d_chain now call a
   one-shot placement_probe on first contact that times the current plan on the
   actual buffers (1 warm + min of 3), rebuilds behind a bumped skew up to 3×,
   and keeps the fastest bitwise-identical draw (2% hysteresis; output is
   layout-invariant so the two-process cmp stays safe). Refactored
   fft1d_execute into exec_body + a probe wrapper, fft1d_create into plan_build
   + a wrapper, added a `probed` flag and a swap-innards path.
   THE KILL — probe on/off batteries on a quiet leased core:
   - 10007 B=64: OFF = 112.3/113.4/113.8/… (deterministic ~112–114 over 8);
     ON = mostly 111–113 but TWO draws at 125–126. Probe is neutral-to-harmful.
   - 65537 B=16: ON and OFF both center ~1635 with one outlier each side
     (ON 1871, OFF 1724) — a wash.
   - 100003 B=1: OFF = 2219–2234 across ALL 10 invocations (rock solid);
     ON = 3018/3065 on the first two (probe's own churn) then 2219–2227.
   So on a quiet core the mode-2/3 engine is ALREADY deterministic (r2/r3/r6's
   hugepage + 128KB-skew work did this), and re-rolling my own already-fixed
   planes cannot move a mode that a busy sibling or thermal droop sets. The
   probe only adds swap/measurement risk. Kept behind BLU_PROBE for a genuinely
   CONTENDED node, where the escape might finally pay — but the scoring node's
   monitor runs a quiet exclusive battery, so it would only hurt there.

### The r6 diagnosis this round overturns (the real result)

r6's "Diagnosed but NOT fixed" section called 10007 B=64 "a per-process lottery:
4 invocations at ~150, one at 112.3, same binary, same core" and the r6 board's
100003 d1_bluestein "median 2554 vs best 2269 (35.6% spread)" evidence of
driver-buffer page-coloring luck that "my side cannot re-color at plan time."
This round's quiet-core batteries show BOTH are deterministic without any probe:
10007 B=64 is 111–114 over 8 runs, 100003 B=1 is 2219–2234 over 10. The r6 ~150
draws coincided with a load-1.9 window (a sibling python3 at 334% CPU); the r6
board's wide median-vs-best gap is thermal drift across a loaded 9-run battery
(my own r3 lesson: back-to-back node runs drift monotonically slower). The
practical consequence: the r6 board number (median 2554) UNDERSTATES this engine
— d1_race, routing to this exact engine, already reads a tight median 2222 at
100003 B=1, which equals this round's probe-off 2224. The engine is at parity
with the best router because the best router IS this engine; the remaining gap to
d1_race/d1_rader at 65537 (733 vs my 1495) is structural (Rader's conv is a clean
2^16; Bluestein pays M=138240) and not a placement or tuning problem.

### Measured standing (a80n0, quiet leased core, tryout build; probe OFF)

| cell | r7 (probe off) | r6 best-window | lib best |
|---|---:|---:|---:|
| 10007 B=1 | 110.4 | 112–124 | 200 patient |
| 10007 B=64 | 111–113 (8 runs) | 112.3 good / 150 bad | 208 patient |
| 65537 B=1 | 1495 | 1485–1518 | 1467 patient |
| 65537 B=16 | ~1635 (median) | 1609–1677 | 1539 patient |
| 100003 B=1 | 2219–2234 (10 runs) | 2219 | 2716 patient |
| 100003 B=8 | 2408 | 2379–2395 | 2806 patient |
| 1021 B=1 / B=256 | 7.75 / 8.90 | 7.84 / 8.10 | (rader's cell) |
| 1024 / 4096 B=256 / 16384 B=64 | 1.88 / 13.9 / 60.9 | 1.88 / — / — | not my fight |

Correctness: 12 single-call configs PASS (worst rel_l2 1.16e-15, tol 1e-12);
6 chain gates PASS (65537/100003 m and B variants, 10007 B=64 m=80, 1021 B=256
m=400; worst 9.4e-14 vs 1e-10), all bit-repeatable. Setup <= 0.03 s.

### Borrowings

- The probe mechanism is d1_race r4's first-call placement probe (their record's
  headline), imported one level deeper into the plan. The negative result is new:
  their probe helps because their harness measures each arm at race time under
  possibly-contended conditions and picks the good draw; on a quiet exclusive
  score node the underlying engine is already deterministic and the probe is a
  no-op-at-best. Recorded so no future round re-derives it — measure the
  probe-OFF spread on a QUIET core first; if it is already tight, the probe
  cannot help.
- The variance-first framing (spread is what the median scores) is my own r6
  lesson via RESCORE_PLAN.md / d1_pow2 r5; this round it cut the other way (the
  spread was measurement conditions, not the code).

### Next round, in priority order

1. **65537 is the only real gap (1495 vs Rader 733)** and it is structural.
   The one idea with a chance is a NESTED-Rader / unpadded-Rader path for 65537
   specifically (conv = 2^16, no Bluestein pad) — but that is d1_rader's class
   and they already do 733; only worth it if this entry wants to cross-claim.
2. **100003 rows (still 41% of the cell)**: the untried lever is 8-row-batched
   tile FFTs at M1=8192 (r3 #1) — but an 8-row tile is ~2–3 MB, past the 1.25 MB
   L2, so it needs 2-row (not 8-row) interleaving to hide the bak-row L3 latency.
   The blocked-entry work this round is a cautionary data point: adding live
   working set to the row/tile pipeline evicts the ping-pong planes and loses.
3. **Streaming-tile software pipeline for entry/exit** (compute tile q while
   gathering q+1 into a second stack buffer, TWO tiles live not NC): the only
   form of this round's blocking idea that might pay, because it caps the extra
   working set at one tile instead of NC. AC_NC scaffolding is in place to A/B it.
4. Accept that on the quiet score node this engine is deterministic and near the
   router's own number at every cell it owns; further gains are structural
   (65537) or in the still-untried 2-row row pipeline, not in placement tricks.
