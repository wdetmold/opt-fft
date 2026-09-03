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
