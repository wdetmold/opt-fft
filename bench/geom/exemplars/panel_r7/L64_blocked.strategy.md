# L64_blocked — strategy record

Geometry: **L = 64** (262 144 complex doubles = 4.19 MB per volume — the only
geometry on the board whose single volume exceeds every cache but L3).
File: `impl/L64_blocked.c`. `fft3d_name()` = `L64_blocked`.

---

## Round panel_r6 (first implementation — this geometry is new this round)

### Technique

Row–column 3D DFT, interleaved-complex vectors, lanes = PW consecutive **z**
(the contiguous axis), PW ∈ {2, 4} both built from one `#include __FILE__`
template. Every 64-point line is **two radix-8 stages** (64 = 8·8, DIT):

```
X[8c+d] = sum_s W8^{sc} · ( W64^{sd} · sum_a W8^{ad} x[8a+s] )
```

The load-bearing decision is what happens to the **x-axis**. A monolithic
x-pass would read 64 concurrent streams of stride 64 KB — and at L=64 that
stride is exactly one L1 way AND one L2 way (all 64 lines land in a single
set at both levels), on top of being unprefetchable (≥4 KB strides defeat the
streamer) and 2× beyond the L2 streamer's stream count. So the x-transform's
two radix-8 stages are **split across the two passes** and no loop in the
file ever runs more than 8 memory streams, all sequential:

```
pass A   planes visited in stride-8 groups {r, r+8, ..., r+56}:
  per plane p (64 KB):
    z-FFT   lanes = PW y-rows, PWxPW complex-granule register transposes on
            load AND store (the one unavoidable transpose pair, placed
            against L1 data), into a PADDED plane scratch P[y][kz] (~70 KB)
    y-FFT   lanes = PW kz, shuffle-free out of P, stores to the PADDED
            scratch volume mid[p][ky][kz] (64 write streams, stride 1088 B —
            stores scatter for free)
  after the group's 8 planes land in mid (~545 KB, L2/L3-warm):
    x stage 1: DFT-8 across the group IN PLACE on mid (8 sequential read +
            8 sequential write streams at plane stride), twiddled W64^{r·d}
pass B   per octet d: DFT-8 over the 8 CONSECUTIVE mid planes {8d..8d+7}
            (one ~545 KB sequential read run), writing X[8c+d] to out through
            8 sequential plane-streams; stores cached / cached+prefetchw / NT
            (tuner candidates)
```

**Padding (this file's charter from the stub).** mid's z-row stride is 68
complex = 17 cache lines (odd) and its plane stride 4356 complex = 1089 lines
(odd), so gcd(stride, sets) = 1 at L1 and L2 and Bailey's single-set case
cannot form on any scratch access; `-DFFT64B_NOPAD` builds the power-of-two
control. `in`/`out` keep the driver's layout but are only ever touched as ≤8
sequential streams. Cost of the padding: mid is 4.46 MB instead of 4.19.

Buffers: one scratch volume `mid`, reused across the batch (stays
cache-resident; `out` is written exactly once, by pass B, so it is NT-able).
in → mid (pass A, x1 in place on mid) → out (pass B).

Prefetch levels (`pf`), all tuner candidates:
* pf=1: paced T1 read cursor over `in`, one plane's worth per plane
  processed, spread over both pass-A subloops, aimed at the **next plane in
  VISIT order** (the group walks planes at stride 8 — a naive linear +32 KB
  lead would waste half its coverage on a plane read 8 iterations later);
  plus FFT64B_PFN lines/ky of next-volume pre-coverage from inside pass B.
* pf=2: pf=1 + write-intent `prefetchw` on pass B's 8 cold out streams,
  FFT64B_PFWL lines ahead (default 4, `-D`-sweepable).
* pf=3/4: NTA read at consumption rate (4 KB lead, A1 subloop only) with/
  without the prefetchw — fills L1, bypasses L2 on SKX-class cores, so the
  in-read stops evicting the group's mid planes before x-stage-1 needs them.
  Built for the node's 1 MB L2 (group = 545 KB + P 70 KB + streams ≈ its
  whole L2); expected to lose on wallaby's 2 MB L2, and does (below).

Modes: `cached` and `nt` (pass-B out stores). Gates: NT and pf∈{2,3} admitted
only when batch × 8.4 MB > 24 MB (B ≥ 3). Tuner: interleaved rounds,
self-warming (one untimed exec per candidate per round), correctness
interlock vs cands[0] to 1e-13 rel L2, 3% simplest-wins hysteresis, pick
reported in `fft3d_description()`, env + `-D` forcing for the monitor
(`FFT64B_PW/MODE/PF`, `FFT64B_FORCE_*`, knobs `FFT64B_PFH/PFN/PFWL/PFDN`).
Diagnostics: `-DFFT64B_SKIPA/SKIPX1/SKIPB` (timing only, wrong answers).

### Derivation / operation count

DFT-8 module (interleaved-vector currency, per PW lanes): 14 add + 12 FMA =
**26 FMA-port ops + 5 swaps**; only irrational constant 1/√2; ±i and
(1±i)/√2 realized as one `vpermilpd` swap + sign-alternating-constant FMAs
(idiom from L36_mixedradix via L36_pfa). Twiddle cmul = 1 swap + 1 mul +
1 FMA with two pre-splatted 64-B table loads.

Per 64-point line over PW lanes: 16 DFT-8 + 49 cmul = **514 FMA-port ops +
129 swaps**. Per volume: z+y = 2·1024 line-groups at PW=4 → 1.053M FMA-port
ops; x1 = 8192 octets × (26+14) = 328k; x2 = 8192 × 26 = 213k; total
**≈1.59M FMA-port vector ops** + ≈0.67M port-5 shuffles (incl. 262k transpose
shuffles). On the node's single 512-bit FMA pipe at the measured 2.9 GHz
licence clock: **~550 µs/volume of port-0 work — the compute floor**.

Scalar-equivalent check: radix-8² is 4·(4+52) + 48·6 per line-ish ≈ 1264 real
flops/line vs split-radix's 1160; the 8% flop give-up buys two stages instead
of three (fewer L1 round trips through the 4-KB stage buffer) and the shared
DFT-8 codelet. Instructions, not flops (corpus consensus #3).

Memory model per volume (the real budget): DRAM = read `in` 4.19 MB + write
`out` 4.19 (+RFO 4.19 unless NT/prefetchw-covered); cache-side = mid write
4.46 + x1 RMW 8.9 + pass-B read 4.46 ≈ 18 MB through L2/L3. At the node's
published 18.2 GB/s single-core L3 this geometry is **memory-shaped at every
batch size** — the schedule is the design, the codelet is not.

### Layout and SIMD decisions

* Interleaved complex end-to-end, lanes = spectator z. Split-complex was NOT
  tried: L36_pencilfused r2 documents it losing at batch-of-one cube geometry
  (their headline dead end), and B=1 is separately scored here too.
* One transpose pair per element (L36_pencilfused r1's proof adopted), both
  sides placed against L1-resident data inside the z-FFT, using L36_pfa's
  PWxPW `TRNC` verbatim.
* The 64-vector FFT stages through a stack array `H_[64]`; gcc's allocator
  spills what it must (L36_pencilfused r1 #1: do not hand-stage, measured
  42% loss there).
* x-stage-1 twiddle vectors (14 loads) hoisted per group outside the
  ky/zq loops.
* All scratch strides an odd number of lines; every vector load/store 64-B
  aligned at PW=4 (32-B at PW=2); PW=2 NT path pairs z-blocks so every NT
  store completes a full line (L36_pfa's pw2 pairing).

### What was measured (wallaby, Xeon Gold 6448Y Sapphire Rapids; µs per
transform, driver min; wallaby's documented ~2× bimodal placement state was
active in some windows — within-window A/Bs only for comparisons; rel_l2 =
4.165–4.173e-16 and bit-identical re-runs on every run listed)

Auto-tuned full runs vs MKL 2022 same day, same case:

| B | L64_blocked | mkl_dfti | ratio |
|---|---|---|---|
| 1 | **646.1** | 708.9 | **1.10×** |
| 4 | **631.7** | 676.6 | 1.07× |
| 8 | **614.8** | 925.7 | **1.51×** |
| 32 | **777.5** | 1561.6 | **2.01×** |
| 64 | **812.0** | 1847.1 | **2.27×** |

(A later, slower placement window gave B=1 691–697 and B=32 828; the table
rows are the best clean windows, sd ≤ 3.4%.)

Tuner picks on wallaby: B=1 `pw4 cached pf0` (pf1 was 735 vs 747 in-arena but
inside the 3% hysteresis band); B≥8 `pw4 nt pf0/pf1` (NT 700–706 vs best
cached 741–751 in-arena at B=8). pw4 beats pw2 by ~13–15% in every cell on
this 2-FMA-pipe machine. The node (one 512-bit FMA pipe, 1 MB L2, NT-hostile
per three rounds of L=36 verdicts) may re-rank all of this; every variant
stays a candidate and the tournament decides there.

Phase split at B=1 (isolated `-DFFT64B_SKIP*` builds, forced pw4/cached/pf0;
phases see different cache states in isolation, so read as scale not sum):
pass A ≈ 385 (sd 0.8%), x1 ≈ 118 (sd 0.2%), pass B ≈ 407 isolated / ≈230
incremental. Pass A carries 2/3 of the FFT arithmetic plus the transposes and
the cold read — it is the biggest single block everywhere.

**Padding A/B (the charter question), forced pw4/cached/pf0, same window:**

| B | padded (17/1089-line strides) | `-DFFT64B_NOPAD` (16/1024) | |
|---|---|---|---|
| 1 | 692.3 | 699.5 | neutral (window noisy, sd 15–31%) |
| 8 | **1266.7** | 1445.8 | **padding worth 14%** (sd 0.5%, clean) |

So at L=64 the corpus's §04-vs-§05 padding dispute resolves the same way the
working-set arithmetic predicts: irrelevant while the volume set is
cache-slack, a real double-digit loss once the batch streams — and this is
on wallaby's 16-way 2 MB L2; the node's 1 MB should show more. First measured
padding number the panel has for a power-of-two geometry.

### What was tried and did NOT work — with the number that killed it

1. **NTA in-read (pf=3/4) on wallaby**: B=1 894.9 vs 735.2 (pf1), B=8 1067
   vs 751 (pf2) in-arena — a 20–42% loss ON THIS MACHINE, whose 2 MB L2
   holds the 545 KB group with room to spare, so NTA only forfeits L2 hits.
   Kept as gated candidates anyway: the mechanism targets the node's 1 MB
   L2, which wallaby cannot exhibit (same reasoning as L36_pfa r6's pf=3/4,
   whose wallaby-vs-node story matched).
2. **prefetchw lead sweep (FFT64B_PFWL = 1/4/16) at B=32 cached**: 850.7 /
   859.3 / 841.5 µs/vol — indistinguishable on wallaby, whose LFB depth and
   memory latency make one-iteration leads sufficient. Default 4; the knob
   is `-D`-sweepable for the monitor, where the L36 record says this exact
   mechanism was worth −42% in-arena on the node.
3. **Unpadded power-of-two scratch** — the NOPAD control above: −14% at
   B=8. Not a surprise, but now it is a number, not an argument.
4. **The monolithic 64-stream x-pass was NOT built, deliberately.** Reasoned
   from L36_pfa r3's measured 101-vs-58 µs/vol split between 36-stream and
   sequential cold reads, plus the set arithmetic (64 KB stride ≡ one L1 way
   and one L2 way — every x-pencil in ONE set at both levels): 64 streams of
   the pathological stride would be strictly worse than anything L=36 ever
   measured. The stage-split above makes the question moot; if someone wants
   the control number, it is a day of work to write the naive pass.
5. (Bug note for honesty, caught before any measurement: pass B's first
   draft loaded octet planes at 2× the plane stride — planes 8d+2,4,6,…
   instead of 8d+1,2,3,…. Caught by inspection before the first build;
   the numpy check would have caught it too.)

### Attribution summary (round rules: name what you borrowed)

Interleaved-complex spectator-axis lanes, CMUL/±i swap idioms, PWxPW TRNC,
two-width `#include __FILE__` template, z-first transpose-on-load against the
cold buffer, and stores-scatter-for-free: **L36_mixedradix r1 via L36_pfa r2/
L36_pencilfused r3**. Paced T1 read cursor through both subloops, prefetchw
on cold out streams, NTA-at-consumption-rate, next-volume pre-coverage, env/
`-D` forcing, description plumbing: **L36_pfa r3–r6** (prefetchw ultimately
**L6_unrolled r3**). Self-warming interleaved-rounds tuner + correctness
interlock + physics gates: **L36_pencilfused r5** + **L36_pfa r2/r4**.
Simplest-wins hysteresis: **L36_pfa r4**. NT-as-gated-candidate-never-default:
the L=36 node verdicts r3–r5. The visit-order-aware prefetch target, the
8-plane-group x-stage split, and the odd-line-padded scratch measurements:
this file, this round.

### Next

1. **Read the node's tuner picks** off `fft3d_description()` in the t_*.json.
   Predictions, stated so they can be scored: pw=4 everywhere; B=1 `cached
   pf0/pf1`; streaming cells `cached pf2` (the L36 node story: NT rejected,
   prefetchw selected 3/3) — if the node instead keeps NT, L=64's
   sequential-plane NT streams (vs L36's 36-way strided NT) are different
   physics and worth a verdict note. If pf=4 (NTA) wins anywhere, the 1 MB-L2
   residency theory is confirmed and the corpus gains a second NTA data point.
2. **Monitor sweeps that cost one `-D` each**: FFT64B_PFWL ∈ {1,4,16},
   FFT64B_PFH ∈ {2,3}, FFT64B_NOPAD (the padding number on the node's 11-way
   L3 / 16-way 1 MB L2), and forced pw2-vs-pw4 at B=1.
3. **Cross-volume drain hiding** (pipeseq analog: interleave pass A of volume
   b+1 into pass B of volume b): wallaby's L36 evidence says −11%, the node
   rejected every pipeline at L=36. Only worth building if the node's B≥8
   cells land far above the ~730 µs/vol DRAM model (in-read + out-write at
   11.5 GB/s), i.e. if reads and writes demonstrably do not overlap.
4. **Pass A is the largest block (385 of 646 µs at B=1)** and it is 2/3
   compute. The two candidate cuts, in order: fuse x-stage-1 into the tail of
   the group's plane loop at column granularity so its 545 KB re-read
   disappears into L2 hits (hard: needs the last plane's A2 interleaved with
   the group's x1); and a split-radix-flavored 64-point stage pairing to shave
   ~8% of FP ops (low value — the geometry is memory-shaped, not port-bound).
5. **Huge pages on `mid`** (`madvise(MADV_HUGEPAGE)` before the memset in
   create): 4.46 MB of scratch = 1090 4-KB pages touched twice per volume;
   L36's record says THP-by-khugepaged never arrives in time for driver
   buffers, but `mid` is plan-owned and allocated in create(), where a
   deliberate `madvise` + touch CAN get 2 MB pages. One experiment.
6. **B=1 residual**: 646 vs ~550 µs port floor + L3 traffic — the gap is
   mostly un-overlapped L3. The paced-read machinery only covers `in`; a T1
   cursor over pass B's octet reads (8 sequential streams, currently
   hardware-only) is the cheapest untried candidate.

---

## Round panel_r7 (cumulative round; r6 had no leaderboard, so no node numbers exist yet)

### Where things stood

No node verdict on any L=64 entry. On wallaby, my rival **L64_radix8**'s r6
record shows them ahead of me in every regime (their 543/512/588 µs at
B=1/8/64 vs my 646/615/812, different windows but the gap is real). Their two
structural advantages over this file: (a) **two sweeps instead of three** —
they do the full 64-point x-FFT in one strided pass where I pay a separate
x1 RMW pass; (b) **2 MB hugepages on the scratch volume**. This round adopts
both, the second unconditionally, the first as tuner candidates.

### What changed (both adopted from L64_radix8 r6 — attribution per the rules)

1. **mid now lives in a 2 MB-aligned anonymous mmap with `MADV_HUGEPAGE`**,
   madvise'd *before* the faulting memset so THP-madvise kernels back it
   synchronously (4.46 MB = 3 huge pages instead of ~1090 4-KB pages).
   posix_memalign fallback if mmap fails; `-DFFT64B_NOHP` builds the control.
2. **A 2-sweep structure variant (`st=1`) joined the tuner.** Pass A does
   z+y only, planes in NATURAL order (the cold in-read becomes one sequential
   4.19 MB run; the stride-8 visit order only existed to feed x1). A new pass
   B2 does the FULL two-stage 64-point x-FFT per (ky, z-column) — the same
   FFT64V macro the z/y lines use, LD = 64 strided plane loads, ST = 64
   plane-streams straight to `out`. Reads are 64 streams at the odd-line
   padded plane stride (sets clean), one `prefetcht0` per load
   `FFT64B_PFXC` (default 2) columns ahead — L64_radix8's next-column hint,
   +12% at B=8 for them. At PW=4 every out store is exactly one full line, so
   NT is fill-buffer-clean at any stride. st=1 candidates are pw4-only, and
   its NT is NOT gated on streaming (see the set arithmetic below).
   Env/`-D`: `FFT64B_ST`, `FFT64B_FORCE_ST`, `FFT64B_PFXC`.
3. Plumbing: `{pw,mode,pf,st}` in the tuner/description; hysteresis rank
   prefers the incumbent st=0 at ties.

### Operation count

Unchanged in total: st=1 moves x1's 328k FMA-port ops into pass B2 (which
becomes a full FFT64V per line-group, 514 ops), total still ≈1.59M FMA-port
vector ops/volume. What st=1 changes is traffic: it deletes the x1 RMW sweep
(8.9 MB of L2/L3 round trip) at the price of turning pass B's 8-sequential-
stream read into a 64-stream strided read of the same 4.46 MB.

### What was measured (wallaby, Gold 6448Y SPR; the ~2× bimodal placement
state was VERY active this session — MKL same-case swung 670→1337 µs between
runs — so all decisions below are from the tuner's in-arena numbers, which
are same-process A/Bs; rel_l2 = 4.165–4.173e-16 and bit-identical re-runs on
every tryout listed)

Full runs, best clean window: **B=1 665.5 µs** (sd 0.23%; MKL that window
1224.8), a later clean run 673.0 with MKL at 670.2 — B=1 is parity-to-1.06×
against MKL depending on window; **B=8 632.5 µs/vol** (5060.4 µs/call, sd
0.14%; MKL same session 1029–1337/vol → ~1.7–2×); **B=64 859.8 µs/vol**
(MKL 1478–1780/vol, noisy).

In-arena (per-candidate, self-warmed, µs/vol):

| candidate | B=1 | B=8 |
|---|---|---|
| pw4 cached pf0 **st0** | **658.8** | 764.6 |
| pw4 nt pf0 **st0** | (gated) | **623.0** |
| pw4 nt pf1 st0 | — | 628.1 |
| pw4 nt best **st1** | 671.3 (pf1) | 696.7 (pf1) |
| pw4 cached pf0 st1 | 687.3 | **1167.2** |

Tuner picks: B=1 `pw4 cached pf0 st0`, B=8 `pw4 nt pf0 st0` — the incumbent
3-sweep won both regimes ON THIS MACHINE and the shipped picks are unchanged.

**Hugepages: +3.3–3.7% at B=8, ~0 at B=1** (the round's one shipped win).
In-arena nt/pf0/st0: 623.0 (HP) vs 647.2 (`-DFFT64B_NOHP`); full-run minima
across 3 alternating build pairs: 5060–5131 (HP) vs 5234–5448 (NOHP), no
overlap. B=1 is insensitive (scratch is reused and 1090 pages sit in SPR's
big STLB; the node's CLX STLB is smaller — expect more there).

### What was tried and did NOT work — with the number that killed it

1. **st=1 (2-sweep) on wallaby**: B=1 671.3 vs 658.8, B=8 696.7 vs 623.0
   in-arena — the 3-sweep incumbent wins by 2% / 11% HERE. Mechanism: on a
   2 MB-L2 machine the x1 RMW is nearly free (the 545 KB group is L2-warm),
   so st=0 pays little for its extra sweep, while B2's 64-stream strided read
   is latency-exposed in a way the octet pass's 8 sequential streams are not.
   **Kept as candidates anyway**: on the node's 1 MB L2 the x1 group does NOT
   sit L2-warm next to the P buffer and the in-read streams, so the sweep
   st=1 deletes is much more expensive there — this is exactly the
   wallaby-cannot-exhibit-it situation of L36_pfa's pf=3/4. The node
   tournament decides; nothing ships slower if it says no.
2. **st=1 with cached out stores, batched**: 1167 vs 697 (NT) at B=8 —
   catastrophic, as predicted by set arithmetic: B2's out stores are 64 RFO
   streams at 64-KB stride ≡ ONE L1 set (and one L2 set on the node). This is
   why st=1's NT is admitted at every batch size including B=1 (where st=1
   cached still loses to NT by 10–16 µs in-arena: 687.3/692.8 vs 677.4/671.3).
   NT-vs-cached is a *structural* question for B2, not a streaming question.
3. **B2 prefetch lead sweep** (`FFT64B_PFXC` = 1/2/4, st1-nt-only arenas):
   1165 / 709 / 732 µs/vol at B=8. Lead 1 is far too short (or that run ate a
   window flip mid-arena — either way not usable), lead 4 measurably worse
   than 2. Default stays 2; `-D`-sweepable for the monitor.
4. (Bug caught before any measurement: pass B2's first draft named its base
   pointers `s_`/`d_`, which FFT64V shadows with its internal stage loop
   counters — gcc's int-conversion warning caught it at compile time.
   Renamed `xsrc_`/`xdst_`. A reminder that FFT64V owns `s_`, `d_`, `H_`,
   `y*_`, `z*_` as names.)

### Attribution summary

2 MB-hugepage scratch (mmap + MADV_HUGEPAGE + touch in create) and the
full-x strided pass with per-load next-column prefetcht0: **L64_radix8 r6**
(their fused pass 2+3; hugepage idea earlier flagged as untried in L36_pfa r1
item 4 and L8_radix8, but L64_radix8 shipped it first). Everything else
carried from r6 with its original attributions.

### Next

1. **Node picks first**: read `st` off `fft3d_description()` in the t_*.json.
   Predictions, stated to be scored: B=1 `st0 cached pf0/pf1`; if the node's
   1 MB L2 makes x1 as expensive as feared, batched cells flip to
   `st1 nt pf0/pf1`; if the node keeps `st0 nt` everywhere, the 2-sweep idea
   is dead on both machines and should be recorded as such.
2. **Monitor `-D` sweeps worth one flag each**: FFT64B_NOHP (hugepage control
   on the node's smaller STLB — expect >3.7%), FFT64B_PFXC ∈ {2,3,4} if any
   st1 cell wins, FFT64B_NOPAD (still the standing padding control), and the
   r6 list (PFWL, PFH, forced pw2).
3. **If st=1 wins batched on the node**, the follow-up is the transposed-mid
   variant sketched this round but not built: pass A's y-FFT stores to
   mid[ky][p][z] (64 write streams at slab stride — stores scatter for free,
   hugepages kill the TLB cost) so B2's reads become fully sequential
   per-ky slabs. That converts B2's one weakness (strided reads) into
   sequential reads at the cost of moving the scatter to the store side.
4. **Cross-volume drain hiding** stays on the list (r6 item 3), still gated
   on the node showing B≥8 far above the ~730 µs DRAM model.
