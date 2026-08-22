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

---

## Round panel_r8 (first round WITH node numbers: r7 leaderboard exists)

### Where things stood

First node contact, round r7: **behind L64_radix8 in all three cells** —
B=1 1087.5 vs their 995.8 (−9%), B=2 1161.0 vs 1023.0 (−13%), B=8 1308.1 vs
1245.4 (−5%); both of us beat MKL everywhere (my B=8 is 1.49× MKL). Node
tuner picks for this file: **`pw4 cached pf0 st0` in every cell** — so the
paced T1 in-read (pf1), NTA (pf4) and st=1 all lost or tied on the node, and
the r7 verdict formally recorded **st=1 dead on both machines** (its own
r7 prediction, resolved on the dead branch). Two more verdict facts drove
this round: (a) the rival's node picks were `plain+slabpf` at B=1 and
**`pfw+slabpf` at B=2 AND B=8** — their scratch-read prefetch was selected
in ALL THREE cells and prefetchw in both batched cells; (b) my pf=2
(prefetchw) was **gated to batch ≥ 3**, i.e. it was never even a candidate
at B=1 or B=2 — the two cells where I lose worst. The wallaby-vs-node ratio
table (verdict §"calibration") puts this file at 1.63–1.65×, the best
translation ratio at L=64 (the rival is 1.83–1.95×): their wallaby lead is
mostly SPR's second FMA port loving their zero-shuffle split-complex kernel,
so the target to optimize against is the node picture, not wallaby's.

### What changed (all schedule/tuner; the arithmetic and pass structure are untouched)

1. **st=1 left the default tournament.** Dead on both machines per the r7
   verdict; the code and env forcing (`FFT64B_ST=1` / `-DFFT64B_FORCE_ST=1`)
   remain so the monitor can still resurrect it, but its 4 candidates no
   longer cost arena time.
2. **pf=2 (paced T1 in-read + prefetchw out) un-gated to every batch size.**
   The old streaming gate (batch×8.4 MB > 24 MB) silenced it exactly where
   the node evidence (rival's B=2 pfw pick) says it should run. Only pf=3
   (NTA+pfw, never a winner anywhere) keeps the gate.
3. **New pf=5: prefetchw ONLY.** The node rejected pf=1 in every cell, so
   pf=2 may have been losing on its read half while its write half was
   good — pf=5 decouples them (this is what the rival's `pfw` mode is).
4. **New pf=6/7: `pfb`, a scratch-read prefetch** (pf=6 alone, pf=7 with
   prefetchw; also admitted with NT as `nt pf6`). Adopted from
   **L64_radix8 r7's slabpf** (node-selected in all three cells), re-shaped
   for this file's streams: (a) in pass B, 8 T0 prefetches per zb step —
   one line per mid plane-stream, `FFT64B_PFBL` (default 2) padded rows
   ahead of the read cursor, exactly 1:1 with consumption; (b) in x1, the
   same pacing but **write-intent** (the group is RMW'd in place). Rationale
   from the node's cache arithmetic: mid is 4.46 MB so pass B's reads always
   miss the 1 MB L2, and the L2 streamer must retrain at every 4-KB
   boundary; and during pass A a group's working set is 545 KB (group) +
   70 KB (P) + 512 KB (in-octet) ≈ 1.13 MB > L2, so x1's "L2-warm" re-read
   is actually a partial L3 re-read on the node — wallaby's 2 MB L2 cannot
   exhibit either mechanism (same reasoning as r6's pf=3/4).
5. Hysteresis rank extended over the new levels (0 < 1 < 4 < 6 < 5 < 2 <
   7 < 3); `FFT64B_PFBL` is env-independent but `-D`-sweepable.

### Operation count

Transform arithmetic unchanged: ≈1.59 M FMA-port vector ops + ≈0.67 M
port-5 shuffles per volume (port 5 carries no FMA on the node's one-pipe
CLX, so the shuffle bill hides there — this is why the node gap is 9% where
the wallaby gap is 20%). pfb, when enabled, adds ≤131 K prefetch uops per
volume (65 536 pass-B T0 + 65 536 x1 prefetchw, one per line consumed —
~8% of the FP count, on the load ports).

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4; per transform;
rel_l2 = 4.165–4.173e-16 and bit-identical re-runs on every run listed;
AVX2/haswell and generic x86-64 builds warning-free under -Wall -Wextra)

Full tryout runs, same session: **B=1 664.0 µs** (sd 0.30%; MKL same window
663.9 — parity), **B=2 662.8 µs/vol** (sd 1.84%), **B=8 687.9 µs/vol**
(sd 0.34%; MKL 805.6/vol → 1.17×), **B=64 832.6 µs/vol** (MKL 1595/vol →
1.92×). Same picture as r7 — this round was never going to move wallaby.

In-arena (per-candidate, self-warmed, pw4 rows; two consistent runs each):

| candidate | B=1 | B=2 | B=8 |
|---|---|---|---|
| cached pf0 | 676.4 / 677.5 | 677.2 | 760.9 / 784.4 |
| cached pf1 | — | 679.8 | 730.5 / 745.7 |
| cached pf2 | 680.9 / 679.7 | 677.4 | **697.9 / 704.0** |
| cached pf5 (pfw only) | 675.8 / 677.5 | **673.1** | 747.8 / 754.2 |
| cached pf6 (pfb only) | 680.9 / 682.3 | 678.4 | 765.5 / 778.8 |
| cached pf7 (pfb+pfw) | 681.7 / 682.8 | 677.2 | 743.9 / 748.8 |
| nt pf0 | (gated) | (gated) | **626.7 / 634.1** |
| nt pf1 | — | — | 631.8 / 636.7 |
| nt pf6 | — | — | 628.4 / 637.5 |

Shipped picks on wallaby are UNCHANGED (B=1/B=2 `cached pf0` via the 3%
hysteresis, B=8 `nt pf0`) — i.e. the new machinery costs nothing here and
exists purely as node-tournament options. Points of note: pf=2 is clearly
the best cached-store batched candidate on wallaby too (−8% vs pf0 at B=8,
read-pacing + pfw both contributing: pf1 −4%, pf5 −2.5%); pfb is neutral
(±1%) in every wallaby cell, as predicted by the 2 MB-L2 argument; NTA
(pf4 943–1121 at B=8) remains a disaster on wallaby.

### What was tried and did NOT work — with the number that killed it

1. **Trusting a single tuner arena on wallaby**: the first B=1 create of the
   session showed a monotonic 1425 → 677 µs drift down the candidate list
   (pf7 "beating" pf0 by 43%) — a bimodal window flip mid-create, not
   physics; three repeat runs showed all cached candidates within 1%.
   Corollary worth recording: **min-across-rounds inside one create is
   biased on a bimodal machine** (whichever candidate's round lands in the
   fast window wins); on the node (0.3% spread, exclusive) the tournament
   is sound. Wallaby conclusions in this record are from the consistent
   repeat runs only.
2. **pfb on wallaby**: neutral-to-−1% everywhere (e.g. 682 vs 677 at B=1,
   765–779 vs 761–784 at B=8) — kept as candidates anyway, since the
   mechanism (L2-missing scratch reads, streamer retrain at 4-KB
   boundaries, x1's group half-evicted from a 1 MB L2) exists only on the
   node. This is the third instance of the wallaby-cannot-exhibit-it
   pattern in this file's record (r6 pf=3/4, r7 st=1 — the latter died on
   the node too, so the pattern is 1-for-2; stated so it can be scored).
3. Nothing else was built this round: the verdict's L2↔DRAM tiling ask is
   answered by argument below rather than by a new variant.

### The verdict's L2↔DRAM tiling ask (answered by construction, not built)

LITERATURE §4.3's re-opened case is "tile the batch so a tile fits L2, then
run all three axes inside the tile," flagged in the r7 verdict as the
largest untried structural move and meaningful only at L=64. Position taken
in this record: **this file's st=0 structure already IS that construction,
specialized to a volume that cannot fit L2.** The 8-plane group (545 KB,
sized against the node's 1 MB L2) is the tile; z, y and HALF the x-FFT
(stage 1) run inside it — 2⅓ of 3 axes. The un-tileable remainder is
forced by data dependency, not schedule: x-stage-2's octet d needs one
plane from EVERY group (any radix-p² split of the x-DFT has one stage
whose octets span all tiles), so a second sweep of the volume is
irreducible for any within-volume tiling. What CAN still be bought is
latency coverage on that sweep's L2 misses — which is exactly what pfb
(this round) does. If the monitor wants the tiling question quantified
anyway, the number to compare is pass B's incremental cost (~230 µs of
~646 on wallaby at B=1) against the 7× L2↔DRAM bandwidth gap §08 measures:
even deleting pass B entirely could not close more than that.

### Attribution summary

Scratch-read prefetch as a tuner axis (pfb) and the decoupled
prefetchw-only store mode: **L64_radix8 r7** (their slabpf and pfw node
picks; pfw ultimately **L6_unrolled r3 / L36_pfa r3–r6** as before). The
1:1 consumption-rate pacing and write-intent form in x1, the 4-KB-streamer
and group-working-set arithmetic, and the un-gating analysis: this file,
this round. Everything else carried from r6/r7 with original attributions.

### Next

1. **Node picks first** (read `pf` off `fft3d_description()` in the
   t_*.json). Predictions, stated to be scored: B=1 `cached pf0` or
   `cached pf6`; B=2 `cached pf5` or `pf7` (if neither, the rival's B=2
   pfw win does not transfer to my 8-sequential-stream pass B and the B=2
   gap is elsewhere); B=8 `cached pf2/pf5/pf7` or `nt` (NT is 0-for-
   everything on this node across five rounds — if it appears here, that
   is a verdict-worthy anomaly). If pfb is never picked anywhere, scratch-
   read latency is NOT the node gap and the next move is a perf-counter
   ask, not more prefetch.
2. **Monitor `-D` sweeps worth one flag each**: FFT64B_PFBL ∈ {1, 2, 4}
   (if any pf6/pf7 cell wins), FFT64B_PFWL ∈ {1, 4, 16} (never yet swept on
   the node; L36 measured −42% from this lead), FFT64B_NOHP (standing
   hugepage control, expect >3.7% on CLX's smaller STLB), FFT64B_NOPAD
   (standing padding control).
3. **If the gap survives this round's picks**, ask the monitor for
   `perf stat -e cycles,L1-dcache-load-misses,LLC-loads,LLC-load-misses`
   on B=1 forced pf0 vs pf6: it separates "x1/pass B miss L2 as predicted"
   from "the residue is front-end/port pressure," which decides between
   more schedule work and a split-complex kernel rewrite next round.
4. **A split-complex pass A** (the rival's layout) is the only remaining
   structural idea with node evidence behind it, but it forfeits this
   file's distinct-structure value and its B=1 shuffle bill is FREE on the
   node's one-FMA-pipe CLX (port 5 idle) — the case for it is wallaby-only,
   so it stays unbuilt unless a future node round proves port-5 pressure.

---

## Round panel_r9 (cumulative round; reacting to the r8 leaderboard)

### Where things stood

r8 node: **behind L64_radix8 in all three cells, fourth round running** —
B=1 1092.6 vs 966.8 (−13%), B=2 1161.4 vs 1020.1 (−14%), B=8 1304.5 vs
1252.3 (−4%). My node picks: `cached pf0` (B=1), `cached pf0` (B=2),
`cached pf2` (B=8). **pfb took zero picks**, firing my pre-registered
reading: scratch-read latency is NOT the node gap, so r8's prefetch round
was a null and the residual is structural. The verdict's reading of the
rival: their tiled candidate lost everywhere (LITERATURE §4.3's L2↔DRAM
tiling case is closed at L=64), and what actually beats me is their **fused
2-sweep** — the last axis's reads run L2-hot because they happen right
after the pass that produced them, where my pass B re-reads mid ~a full
volume after x1 wrote it (L3-cold on the node), and my x1 RMW is itself
partially L3 on a 1 MB L2.

### Technique this round: st=2, the "x-first" 2-sweep (new tuner structure)

The ledger insight first, because it corrects r8's framing: counting
per-element loads/stores, my 3-sweep and the rival's 2-sweep are THE SAME
(3 loads + 3 stores per element either way. st0: in→reg→mid, mid→reg→mid,
mid→reg→out; theirs: in→SC, SC→SC, SC→out). The fusion win is not fewer
memory ops, it is **temporal locality on the last axis's reads** — theirs
hit L2, my pass B's miss to L3. My dead st=1 (r7) failed because it put the
STRIDED stage last, writing 64 scattered out-streams. The fix is the pass
ORDER, not the sweep count:

```
pass 1  x stage 1 DIRECTLY off the driver's input (this pass IS the cold
        in-read): for group s, DFT-8 across in planes {s, s+8, .., s+56}
        (8 sequential read streams), twiddle W64^{s·d}, store to mid plane
        8d+s (8 sequential write streams at padded stride).  Elementwise:
        zero shuffles beyond the cmul swaps.
pass 2  per octet d: x stage 2 = plain DFT-8 over 8 CONSECUTIVE mid planes
        {8d..8d+7} (one ~545 KB sequential read run) into the octet buffer
        OB (8 padded planes, ~560 KB, reused across octets and volumes →
        recirculates in L2/L3); then per completed output plane 8c+d:
        y-FFT THEN z-FFT straight out of OB to out.
```

The tail is ordered **y-first, z-last** so each side gets the pattern it
can afford: the y-FFT's 64-row scatter reads land on the L2-hot OB (reads
carry no RFO; the 8 loads per DFT8 are independent so MLP covers L2
latency), and the z-FFT's transpose-on-store emits **PW sequential row
streams** to the cold out plane. Math is the same 64=8×8 DIT as ever, just
axis order x→y→z instead of z→y→x; arithmetic count unchanged
(≈1.59 M FMA-port vector ops + ≈0.67 M shuffles per volume).

What st=2 buys ON THE NODE (the machine it is aimed at): st0's x1 RMW
(8.9 MB round trip, partially L3 on a 1 MB L2) and pass B's 4.46 MB
L3-cold re-read both disappear; in exchange OB adds a 8.9 MB round trip
that is **L2-hot by construction** (written and consumed inside one octet
step). Wallaby's 2 MB L2 covers st0's exposure anyway, so wallaby cannot
exhibit the mechanism — the familiar pattern (r6 pf=3/4, r7 st=1, r8 pfb:
now 1-for-3, stated so it can be scored).

Also this round: st0's candidate list trimmed to the levels with a win on
record ({0,1,2,5} cached, {0,1} nt) to make arena room; st=2's NT is NOT
streaming-gated (out is written exactly once by the z tail — structural,
st=1 r7 precedent); new `FFT64B_PFWL2` knob (default 16) for the tail's
prefetchw lead, separate from FFT64B_PFWL so monitor sweeps cannot disturb
st0's node-picked pf2 at B=8. For st=2, pf=6/7 (pfb) means: paced T0 on
pass 1's 8 in-streams, pass 2's mid read, AND a next-column T0 at
consumption rate in the y-FFT (the PFXC idiom).

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4; per transform;
rel_l2 = 4.165–4.173e-16 (shipped) / 4.463e-16 (forced st2, different
reassociation), bit-identical re-runs on every run listed; both ISA paths
warning-free under -Wall -Wextra)

Full tryout runs, shipped picks (unchanged on wallaby: B=1 `pw4 cached pf0
st0`, B≥8 `pw4 nt pf0 st0`): **B=1 646.8 µs** (sd 1.1%; MKL same session
668.5), B=2 674.7/vol, **B=8 631.8 µs/vol** (5054.7/call; MKL 857.4/vol →
1.36×), B=64 814.2/vol (MKL 1941.4/vol → 2.38×).

In-arena (consistent fast-window repeats only, per the r8 bimodality rule;
µs/vol, pw4):

| candidate | B=1 | B=8 |
|---|---|---|
| cached pf0 **st0** | **644–668** | 761–770 |
| cached pf2 st0 | 667–681 | **694–701** |
| nt pf0 **st0** | (gated) | **622–630** |
| cached pf0 st2 | 796–879 | 1144–1180 |
| cached pf5 st2 (lead 16) | 669–686 | 813–823 |
| nt pf0 **st2** | **662–693** | **690–698** |
| nt pf6 st2 | 664–720 | 698–712 |

So st=2's best form sits 2.5–4% behind st0 at B=1 and ~11% behind at B=8
ON THIS MACHINE, and ships purely as the node's option — exactly the r7
st=1 shape, except this time the mechanism (L3-cold last-axis reads) is
one the node has already demonstrated via the r8 leaderboard gap itself.

### What was tried and did NOT work — with the number that killed it

1. **z-first tail (first draft of st=2)**: wrote out as a 64-row scatter
   (64 write streams at 1 KB stride, RFO-unprefetchable) — in-arena B=1
   748.1 vs st0's 663.5 (−13%). The y-first/z-last rewrite plus prefetchw
   moved st=2's best to 662–671. Curious detail for the record: the
   scatter draft's *pf0* was faster than the stream version's pf0 (748 vs
   ~860 — 64 scattered RFO misses self-provide MLP where 4 serial row
   streams stall), but its best-with-prefetch was far worse; st=2's tail is
   RFO-bound either way, which is why pf5/NT dominate its cached-pf0 by
   20% (875 → 686/690).
2. **pfb's y-column T0 prefetch on wallaby**: pf7 697–708 vs pf5 685–696,
   nt-pf6 698–720 vs nt-pf0 686–708 — neutral-to-−2%, OB is L2-hot here
   and the extra 64 uops/FFT64V buy nothing. Kept gated behind pf=6/7 for
   the node, where OB will half-spill to L3 (560 KB OB + 545 KB octet
   read > 1 MB L2).
3. **FFT64B_PFWL2 = 32**: 675–705 vs lead-16's 669–681 — past the knee;
   16 ships as the default. (Lead 16 lines = exactly one out row = the
   next yb-group's rows one full group ahead, ~500+ cycles.)
4. **Not built, consciously**: fusing x2 into the y-FFT's load side to
   delete OB (each OB element IS consumed exactly once, but x2's DFT8
   produces all 8 c-planes per column while the y-pass consumes one plane
   at a time → 8× recompute; and iterating x2 column-major to match would
   turn pass 2's sequential mid read into st=1's strided disease). OB is
   the honest price of the fusion.

### Operation count

Arithmetic identical to r6–r8 (≈1.59 M FMA-port vector ops/volume); st=2
reorders it across passes: pass 1 = x stage 1 (306 of 1542 vec-op-units
per element-line), pass 2 = x stage 2 + y + z. Traffic per volume, node
model: st0 = in 4.19r + mid 4.46w + RMW 8.9 (L2/L3) + mid 4.46r (L3) +
out 4.19w; st2 = in 4.19r + mid 4.46w + mid 4.46r + OB 8.9 (L2-hot) +
out 4.19w. Same totals, ~9 MB of exposure moved from L3 to L2.

### Attribution summary

The fused-last-pass structure (and the evidence that it is what beats me):
**L64_radix8 r6–r8** (their fused pass 2+3, node picks all three rounds).
The x-first inversion making the fusion compatible with my z-lane
interleaved layout, the y-first/z-last tail ordering, the OB construction,
and the same-ledger analysis are mine, this round. Next-column T0 in the
y-pass: **L64_radix8 r6** (PFXC) via my own passB2. prefetchw/NT machinery
carried with original attributions (L6_unrolled r3 / L36_pfa r3–r6).

### Node predictions (stated to be scored)

* If the L3-exposure theory is right, the tuner picks `st2` (mode nt or
  cached-pf5, likely pf7 if OB spills) in at least the B=1 and B=2 cells
  and lands **≤ 1030 µs at B=1** (st2-wallaby 662 × my 1.55–1.65×
  translation ratio; st0 would re-land ~1085–1095). Batched: st2-nt ~
  690 × 1.64 ≈ 1130 at B=8 would beat the rival's 1252.
* If the node keeps `st0 cached pf0/pf2` everywhere, the fused-tail idea
  does not transfer to my layout, st=2 joins st=1 in the dead list, and
  the honest next move is the split-complex kernel rewrite (last resort:
  it forfeits this file's distinct structure).
* Monitor sweeps worth one flag each, in value order: `FFT64B_FORCE_ST=2`
  + `FFT64B_TUNEDBG`-equivalent (`FFT64B_VERBOSE=1`) at B=1/B=8 — one run
  each settles the structure question even if the picks look unchanged;
  `-DFFT64B_PFWL2=4/32` if any st2-pf5/7 cell wins; the standing r6–r8
  list (FFT64B_NOHP, FFT64B_NOPAD, FFT64B_PFWL for st0-pf2).

### Next

1. Read the node picks. If st2 wins anywhere, the follow-ups in order:
   (a) next-octet T1 pre-coverage during the tail's compute (the rival's
   slabpf translated: pass 2's mid read is sequential but its first octet
   per volume is always cold); (b) a pass-1 mid-store prefetchw variant
   (the write side of pass 1 is 8 RFO streams; the rival's scpfw failed
   for THEIR pass-1 shape, mine is different).
2. If st2 loses everywhere on the node, stop paying its arena cost
   (env-gate it like st1) and take the split-complex question seriously —
   L64_radix8's node numbers are the standing evidence that the shuffle
   bill is NOT free even on one-FMA-pipe CLX, contra my r8 §4 reasoning.
3. Standing: do not rebuild st=1, the monolithic 64-stream x-pass, NTA
   in-reads on wallaby, or per-run tryout comparisons on wallaby (bimodal
   windows; in-arena repeats only).

---

## Round panel_r10 (reacting to the r9 leaderboard and verdict)

### Where things stood

r9 node: **behind L64_radix8 in all three cells, fifth round running** —
B=1 1098.4 vs 952.9 (−13%), B=2 1180.0 vs 1021.1 (−13%), B=8 1304.2 vs
1249.9 (−4%). My r9 bet **st=2 was DECLINED 3/3** (node kept `cached pf0
st0` in every cell; B=1 read 1098 against my pre-registered "≤1030 if the
L3-exposure theory is right"), so st=2 joined st=1 in the dead list, exactly
per my own falsification branch. The verdict's instruction was verbatim:
*"L64_blocked should be judged on the split-complex rewrite it has deferred
for two rounds, or the slot moved: st=1 and st=2 are both dead, and
L64_radix8's node numbers are standing evidence that the shuffle bill is
not free even on a one-FMA-pipe part."* My r8 §4 reasoning (port 5 is idle
on CLX, so interleaved swaps are free) is thereby dead: the ~0.42 M-cycle
node gap at B=1 is the same order as my extra ~0.34 M port-5/front-end
shuffle uops vs the rival's split-complex kernel.

### Technique this round: **st=3 "split-sc"**, the split-complex rewrite

New structure in the tournament, adopted wholesale in shape from
**L64_radix8 r6–r9** (attribution below; this is the round's explicit
assignment, not a convergent accident — the panel brief says take what
wins). Currency: **split-complex, SIMD lanes = 8 adjacent z**; every
butterfly and twiddle is elementwise across vectors — **zero shuffles in
the arithmetic**:

```
pass 1   per x-plane p: y-FFT (64 = 8x8 DIT, two DFT8S stages through an
         8-KB line buffer), deinterleave fused into the stage-1 loads
         (2 vpermt2pd per point), stage-1 twiddles W64^{s*d} by scalar
         broadcast; split (re,im) vector pairs (each one full 64-B line)
         stored to SC[x][ky][zb], strides SCKS=136 doubles (17 lines),
         SCXS=8712 (1089 lines) -- odd-line Bailey padding, same values my
         r6 padding A/B and the rival's -25% NOPAD control both endorse.
         SC reuses the existing 2-MB-hugepage mmap (fits MIDDBL exactly).
pass 2+3 fused, per ky: x-FFT over the 64 planes IN PLACE in SC (loads at
         SCXS stride through the line buffer -- all 64 loads complete
         before writeback, so in-place is legal; prefetcht0 of the next zb
         column on every load, the rival's +12% hint); then the 64 z-lines
         of the ky-slab the writeback just made L2-hot: DFT8S across the 8
         slot pairs (g->k2), lane twiddles W64^{l*k2} from 14 vector
         tables, 8x8 transpose pair (2x24 two-source shuffles), second
         DFT8S (l->k1), re/im interleave (16 shuffles), contiguous 1-KB
         row store to out (cached / NT / +prefetchw, tuner axes).
```

The split radix-8 codelet is the panel's 44-add/sub + 8-FMA form
(L8_batchsimd r1 via L8_radix8), rederived and re-verified here; the only
irrational constant is 1/√2. st=3 prefetch axes (my pf field, st3
semantics): `1` = p1pf (pass-1 next-plane T1, 16 lines per stage-1 group —
the mechanism the node picked 3/3 for the rival at B=1), `6` = slabpf
(T1 next-ky slab, 17 lines per kx step during z-lines — node-picked in
every rival cell r7–r9), `5` = pfw (out-row write-intent FFT64B_PFWL3=4
rows ahead — the rival's batched pick), `7/8/9` = the combinations. NT not
streaming-gated (out written exactly once — structural). st=3 is
AVX-512-only; the pw2/portable path remains st=0. **st=2 left the default
tournament** (env-only via FFT64B_ST=2, like st=1); st=0 stays as the
correctness reference and per-machine fallback.

### Operation count

FMA-port work unchanged (the split DFT8S is 52 ops per 64 lane-points =
the same 0.8125 ops/point as the interleaved codelet; twiddle 4 ops per 8
points = same 0.5). What changes is the **shuffle bill: 65 536
deinterleave + 4096×(48 transpose + 16 interleave) = 328 K/volume, half
the interleaved path's ≈0.67 M**, and none of it on any butterfly's
critical path. Traffic: in 4.19r + SC 4.5w (RFO) + SC 4.5r strided
(slabpf/L2) + SC 4.5w in-place (hits just-loaded lines, no RFO) + slab
re-read L2-hot + out 4.19w — two sweeps, same ledger as the rival.

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4; per transform;
rel_l2 = 4.460–4.464e-16, rel_max ≤ 5.7e-16, bit-identical re-runs on every
run listed; both ISA paths warning-free under -Wall -Wextra)

Full tryout runs, clean windows (sd ≤ 0.5% unless noted):

| B | this round (st=3 picked) | r9 best (st=0) | MKL same session | vs MKL |
|---|---|---|---|---|
| 1 | **535.0** (sd 0.04%) | 646.8 | 689.9 | 1.29× |
| 2 | **523.0**/vol (sd 11%, window) | 674.7 | 667.6/vol | 1.28× |
| 8 | **538.0**/vol (sd 0.05%) | 631.8 | 1097.5/vol | 2.04× |
| 64 | **694.7**/vol (sd 1.1%) | 814.2 | 1619.4/vol | 2.33× |

**−15 to −17% in every cell**, the largest one-round move this file has
recorded. In-arena (same-process, self-warmed):

| candidate | B=1 | B=8 |
|---|---|---|
| pw4 cached pf0 st0 (old incumbent) | 1199.7* | — |
| pw4 cached pf5 st0 | 689.3 | — |
| pw4 cached pf0 **st3** | **547.4 (pick)** | 772.9 |
| pw4 cached pf1 st3 (p1pf) | 545.7 | 729.3 |
| pw4 cached pf8 st3 (slab+p1pf) | 572.4 | 727.6 |
| pw4 cached pf9 st3 (slab+pfw+p1pf) | 574.9 | 610.3 |
| pw4 nt pf0 st3 | 539.2 | 535.9 |
| pw4 nt pf6 st3 (slabpf) | **536.6** | **528.9 → pick nt pf0** |

(*st0's arena rows drifted high in that create — first-touch window effect;
the cross-run full numbers above are the honest st0 baseline.) Picks: B=1
`cached pf0 st3` (3% hysteresis pulled NT's 2% edge back to the simpler
cached — well-aimed for the node, which has never picked NT at L=64), B≥8
`nt pf0 st3`. Wallaby's NT-love and prefetch-indifference are the familiar
2-MB-L2 story; the node candidates that matter (`cached pf8` ≈ the rival's
B=1 pick, `cached pf7/pf9` ≈ their batched pick) are all in the arena.

Regression checks: forced FFT64B_ST=0/1/2 all still PASS at B=2 (4.173/
4.173/4.469e-16); `if (st)` → `if (st == 1)` dispatch tightened so st=3
can never fall into the st=1 path on a non-AVX-512 build.

### What was tried and did NOT work

Nothing failed this round — the round was one large planned rewrite, and
it landed. For the record: no wallaby cell wanted p1pf or slabpf by more
than the hysteresis band (545.7/536.6 vs 547.4/539.2), consistent with
every previous wallaby-cannot-exhibit-it precedent (r6 pf=3/4, r8 pfb, the
rival's own p1pf at −1.1% here vs node-picked 3/3). The node tournament
decides; the mechanisms are all candidates there.

### Attribution (per the panel rules, and this round it is the whole point)

The split-complex currency, lanes-are-8z layout, deinterleave-fused pass-1
loads, in-place strided x-FFT through the line buffer, next-column
prefetcht0, fused L2-hot z-line tail with transpose pair + interleave-on-
store, slabpf, p1pf, and pfw-instead-of-NT: **L64_radix8 r6–r9** (their
node picks are the evidence base; codelet lineage **L8_batchsimd r1 via
L8_radix8**; transpose-network idea **L8_fusedaxes r1**, my 24-op
formulation via gcc generic shuffles is new but equivalent; pfw ultimately
**L6_unrolled r3 / L36_pfa r3–r6**). Odd-line padding values: my own r6
A/B + their SCKS/SCXS numbers (identical conclusion, both recorded). The
{pw,mode,pf,st} tournament, hysteresis, env forcing, hugepage mapping:
carried from my r6–r9 with original attributions. What remains mine in the
kernel: the FFT64S two-stage macro with pluggable LD/ST/prefetch hooks (one
body serves pass 1, in-place x-lines, and any future structure), and
keeping st=0 as an always-on correctness interlock for the new path.

### Node predictions (stated to be scored)

* Wallaby→node ratio for this file has been 1.63–1.70×; st=3's shape is
  the rival's, whose ratio is 1.74–1.83×. Band: **B=1 890–980 µs**
  (535×1.66 to 535×1.83) with pick `cached pf0/pf1/pf8 st3`; **B=2
  920–1010**; **B=8 940–1130/vol** with pick `cached pf7/pf9 st3` (if NT
  appears on the node at L=64 for the first time, that is verdict-worthy).
  Anything ≤ 980 at B=1 beats the rival's r9 952.9 only at the low edge —
  parity is the realistic outcome; the point is deleting the 13% structural
  deficit and letting the schedule axes (already theirs) fight it out.
* If the node keeps `st0` anywhere, the split-complex theory is wrong for
  my pass shapes and the shuffle-bill diagnosis needs a rethink — record
  it loudly.
* Monitor sweeps worth one flag each: `FFT64B_VERBOSE=1` at each B (the
  16-row table settles every axis in one run); `-DFFT64B_PFWL3=1/8/16` if
  any pfw cell wins; the standing FFT64B_NOHP / FFT64B_NOPAD controls.

### Next

1. Read the node picks. If st=3 wins, the follow-ups in value order:
   (a) **propf** (prologue prefetch of ky-slab 0 + input plane 0 — the
   rival's r9 item; their node A/B read a coin flip, but my pass-1 store
   pattern differs, so it is untested for this file); (b) a create-time
   store-mode A/B twin for pass 1's SC RFOs (the verdict names pass-1
   scattered stores as the rival's un-hidden residual — mine are 2
   full-line vectors per slot, so `movntpd`-to-SC + early T2 re-read is
   NOT obviously wrong here the way it was for their 17-line rows);
   (c) retire st=1/st=2 code entirely if the monitor agrees the record
   suffices.
2. If st=3 ties the rival, the differentiator to build next is batched:
   nobody has beaten the ~700 µs DRAM model at B≥8; a two-volume software
   pipeline is on every dead-list, but a **batch-paired pass 1** (volume
   b's pass 2+3 interleaved with b+1's pass 1 at ky-slab granularity)
   has never been costed with split currency and complementary port
   profiles.
3. Standing dead list unchanged: st=1, st=2 (as defaults), NTA in-reads,
   monolithic 64-stream x-pass, unpadded scratch, cross-process wallaby
   comparisons.
