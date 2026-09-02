# gen_pfa_large — strategy record

Class: PFA of coprime pairs, large. Owned acceptance sizes: 40 (8x5), 50 (25x2),
100 (25x4). Seed material: L45_pfa 9x5 (panel_r11 winner), gt-PFA n1_9 DAG.

## Round gen_r1

Starting point: the dense O(L^4) validation stub (baseline_matrix). Everything
below is this round.

### What was built

**1. The engine: L45_pfa's two-sweep structure, generalized over a size
template.** One file, three template instantiations (`#include __FILE__` with
`GENL`/`GPP`/`PFAL` set per size). Per volume: phase 1 per x-plane (z transform
with lanes = 4 y-rows via 4x4 complex-granule register transposes into a plane
scratch with odd-cache-line row pitch, then y transform with lanes = 4
contiguous kz into mid), phase 2 = x transform tiled over the flat (y,z) index.
L*L % 4 == 0 for all three sizes so phase 2 has no tail; L=50's phase-1
subpasses run 12 full groups + one overlapping group (idempotent recompute of 2
lanes). Taken verbatim from L45_pfa: TRNC transpose, the opaque-base asm
barrier in the y-subloop, heap (not stack) plane scratch, single-runtime-base
addressing, odd-line pitches (44/52/108 complex = 11/13/27 lines).

**2. Per-size Good-Thomas line codelets** (interleaved-complex zmm, lanes = a
spectator axis, index maps folded at compile time), stage order per the r11
lesson (short-live-range module first, long module reads contiguous hot slots
and stores straight through ST):

| L | maps | stages | FMA-port vector ops/line |
|---|---|---|---|
| 40 | n=(5n1+8n2)%40, k=(25k1+16k2)%40 | 8xDFT5 then 5xDFT8 | 278 |
| 50 | n=(2n1+25n2)%50, k=(26k1+25k2)%50 | 25xDFT2 then 2xDFT25 | 434 |
| 100 | n=(4n1+25n2)%100, k=(76k1+25k2)%100 | 25xDFT4 then 4xDFT25 | 968 |

DFT5 = FFTW n1_5 FMA form copied from L45_pfa (16 ops + 2 swaps). DFT8 =
radix-2 DIT derived by hand, W8 twiddles folded into the output butterflies (26
ops + 6 swaps, matches L64_blocked's count). **DFT25 = 5x5 Cooley-Tukey with
general twiddles — this class's first contact with the campaign's twiddle
problem**: X[k1+5k2] = DFT5_{n2}(W25^{n2 k1} * DFT5_m(x[n2+5m])[k1]), 16
nontrivial twiddles per call at 2 ops + 1 swap each, stage A stores U[5*k1+n2]
so stage B reads 5 contiguous slots. Twiddles are compile-time literals from
long-double cosl/sinl (~19 correct digits) in a product-indexed const table the
unrolled loops fold. 192 ops + 36 swaps per DFT25. Volume op counts: 333.6k
(40), 813.8k (50), 7.26M (100).

**3. Owned chain (`fft3d_chain`, strong symbol) — the round's biggest win.**
The graded workload is chained, and the driver's fallback pays a separate
full-volume map pass plus a memcpy per unit. Raw execute vs driver-chained on
the node made the cost visible: L=40 B=1 raw 136.6 us but chained 377. Two
chain-step families, **raced in create() on the actual chain step**:

- `ip*` (in-place): p1 in place (each plane is fully consumed into the plane
  scratch before being rewritten — checked property, documented in the code),
  p2 in place, then a sequential vectorized map pass in place. State stays in
  `out` for the whole chain; the only volume-sized objects touched are the
  state and c.
- `f*` (fused): map folded into phase 2's stores, padded mid volume M -> state
  out of place.

Map arithmetic in both: z/(1+|z|) with rsqrt14/rcp14 + two Newton steps each
(~1e-16 rel; zmm vsqrtpd/vdivpd are unpipelined and lose ~3x). create() gates
the picked chain step against execute + the driver's exact scalar map (1e-13)
and falls back to that path on any disagreement.

**4. create(): gate + race** (L45_pfa's tuner, simplified): every candidate is
correctness-gated against an independent scalar O(L^2)-per-line reference on
the first AND last arena volume, then min-over-interleaved-rounds timed on the
chain step, 3% simplest-first hysteresis, GENPFL_PF / GENPFL_VERBOSE /
GENPFL_NOFUSE forcing knobs. Dense-matrix fallback retained for no-AVX512 or
gate failure.

### Measured on the node (a80n0, Gold 6326 Ice Lake, graded chain m, min):

| case | dense stub | driver-map baseline | final | rel L2 |
|---|---|---|---|---|
| L=40 B=8 m=128 | (stub ~10^2 ms class) | 354.1 | **206.7** | 3.6e-16 |
| L=40 B=1 m=128 | | 377.4 | **236.3** | 3.6e-16 |
| L=50 B=4 m=128 | | 798.4 | **497.0** | 4.3e-16 |
| L=50 B=1 m=128 | | 756.9 | **557.5** | 4.3e-16 |
| L=100 B=1 m=64 | | 8243.5 | **5247.6** | 4.5e-16 |

Raw execute (no chain), node: 136.6 us (40 B=1), 387.9 (50 B=1), 4563 (100
B=1). Map-chain gate passes at all sizes (worst 8.6e-14 vs anchor 4.3e-14 at
m=128, tol 1e-10; the two-step budget is comfortable). Repeatability:
bit-identical across runs. Setup: 0.1-3.7 s (refnd dominates at 100; fine vs
the 60 s cold budget, but the 50 ms wisdom budget needs gen_race's cache).

### What did NOT work, with the numbers that killed it

1. **Fusing the map into the 100-stream x-pass at L=100.** First chain
   implementation was fused-only: 12.0-13.4 ms/xform vs 8.24 for the UNFUSED
   driver path. Phase decomposition on the node: p2c fused 10.0 ms vs p2
   in-place 2.06 + sequential map 3.6. Folding the map into phase 2 turns 100
   read+write-same-line streams into 100 read + 100 RFO-write streams with one
   fresh 64B line each per tile — miss-stream/MLP bound, not op bound. Lesson:
   **fusion is a win only while the volume is cache-resident** (it won at
   40/50); past that, keep every pass at minimum concurrent miss streams.
   Hence the raced ip*/f* pair.
2. **M at out's exact layout with a page-aligned base** (first fused version):
   suspected systematic 4K aliasing per the L23_rader rule; padding planes to
   odd cache-line pitch + shifting the base 2368 B moved L=100 only 13.4 ->
   12.0 ms — real but secondary to the miss-stream problem above. Padding kept
   (it is free).
3. **Inheriting the execute race's poke policy for the chain step.** The
   in-place execute race picked pf0-vs-pf1 within 6% while the fused chain
   step differed 340 vs 274 us/xform (L=40 B=8, forced). An early sweep with
   the inherited pick regressed B=8 to 470. The tuner now races the chain step
   itself (it is what is graded).

### Borrowed from other entries (attribution)

- L45_pfa (seed, panel_r11): the whole two-sweep engine shape, TRNC, DFT5
  module, opaque-base barrier, odd-pitch plane scratch, heap-not-stack scratch,
  gate+race tuner with hysteresis, stage-order lesson.
- L45_mixedradix (via L45_pfa r7/r11): flat phase-2 tiling, store-direct stage
  order.
- L23_rader (via corpus): odd-cache-line pitches / mod-4096 anti-aliasing.
- Driver (ice panel): the chain contract and the exact scalar map used as the
  create()-time chain gate.

### Next round

1. **Compute efficiency of the codelets.** L=40 B=1 raw is 136.6 us vs a
   ~50-60 us dual-FMA-port floor (333.6k zmm ops). Audit spills per exec
   under node flags (L45_pfa's method); suspects: T_[L] round-trips through
   L1 between stages, DFT25's U[25] on top of T_[100] at L=100.
2. **L=50 tails:** the overlap groups recompute 2/4 lanes twice per subpass
   (+~4% volume ops) and the 800 B rows split-load half the z-pass accesses.
   A PW=2 tail group (or L45's PW=1 xmm lines) removes both.
3. **L=100 phase 1 is now the bottleneck** (~3.5 of 5.25 ms). Consider batch
   x-plane pairing or a z+y+x three-way plan that keeps more of the mid in
   L2; also try huge pages for the state (corpus: always at >100 volumes).
4. **Adopt library layers as they appear**: gen_race's wisdom cache (the 50 ms
   replan budget), gen_twiddle's exact tables (mine are file-local literals),
   gen_layout's allocation helpers.
5. From round 3 the class must accept ANY coprime-pair composite: the codelet
   generator needs to become runtime-parameterized (planner handshake).

## Round gen_r2

Cumulative round: read every other entry's record first. gen_powp adopted this
entry's engine wholesale in r1 and edged it at the two shared sizes (50: 473.5
vs 481.4; 100: 5026.8 vs 5086.4 on the leaderboard). This round: take back the
shared sizes, keep 40.

### What changed

**1. DFT25 stage-B stores fused straight through the PFA wrappers' ST (the
r1 next-list item 1, also queued in gen_powp's record — done here first).**
The function `dft25v` (which wrote `R_[25]`, which PFA50C/PFA100C then re-read
to route through the CRT output map) became the macro `DFT25M(LDX, STO, KMAP)`:
each stage-B DFT5's five outputs go straight to the caller's store macro with
the CRT index applied. Deletes a 25-store + 25-load L1 round-trip per DFT25
call (50 vector L1 accesses per line at L=50, 200 at L=100) and cuts peak live
vectors in stage B (U[25] + R_ growing to ~45 before; ~31 now). FMA-port op
counts per line are unchanged (278 / 434 / 968). Preprocessor note for anyone
copying the shape: the store macro must be passed through as a DFT25M
*parameter* (a file-scope helper that names `ST` textually will NOT pick up
the wrapper's ST argument — macro parameters are substituted before rescan).

**2. Chain family ipf* (BORROWED from gen_powp gen_r1, who built it on this
engine): in place AND the map fused into phase 2's stores** — no mid volume,
no separate map pass. Added as raced candidates at all sizes.

**3. NEW chain family ipe* (map EPILOGUE per phase-2 tile):** FFT stores stay
plain, then the GENL just-written L1-hot lines are re-read, mapped with c, and
re-stored in an independent-iteration loop. Designed to delete the map pass's
DRAM round trip WITHOUT the ipf failure mode (see below). Raced at all sizes.

**4. NEW chain families ipnt*/ipfnt* (GENL % 4 == 0 only): non-temporal
y-subpass.** At 40/100 the p1 y-pass writes every output line in full at 64B
alignment, so ST2 can be vmovntpd (no RFO read of the state volume; sfence per
volume; runtime 64B base check falls back to the ordinary body).

**5. Candidate pool now 11 (40/100) / 9 (50), per-size tie-break ranks**, set
from node race tables; hysteresis unchanged at 3%. All families compute
bit-identical outputs (same op order, same NR ladder), verified with cmp
across forced picks — a per-host pick flip cannot break the driver's
repeatability gate. pf ids for GENPFL_PF: 0 ip0, 1 ip1, 2 f0, 3 fr, 4 frw,
5 ipf0, 6 ipf1, 7 ipnt, 8 ipfnt, 9 ipe0, 10 ipe1.

### Measured on the node (a80n0, tryout.sh pinned core, graded chain, min)

| case | r1 | r2 | notes |
|---|---|---|---|
| L=40 B=8 m=128 | 206.7 | **208.8** (quiet window; 213-216 typical) | pick ip0; parity — 40 has no DFT25 |
| L=40 B=1 m=128 | 236.3 | **223.2** | |
| L=50 B=4 m=128 | 497 (tryout) / 481.4 (scored) | **471.5** (quiet; 483-487 typical) | pick ip1; ahead of gen_powp's r1 473.5 |
| L=50 B=1 m=128 | 557.5 | **554.9** | |
| L=100 B=1 m=64 | 5247.6 | **5008.8** (quiet; 5360-5490 under neighbours) | pick ip1; −4.6%, ahead of gen_powp's r1 5026.8 |

vs MKL same core: 40 B=8 2.0x, 50 B=4 2.0x, 100 B=1 1.54x. Gates: single call
rel L2 3.6-4.5e-16; map-chain m=128/64 3.8-5.0e-14 at 1.3-1.7x the honest
anchor (tol 1e-10). Setup 0.37-6.5 s (pool grew; still well under 60 s cold —
the 50 ms warm budget still needs gen_race adoption, deferred again).

The r2 gains at 50/100 are ENTIRELY item 1 (DFT25 fusion); the node race
still picks the r1 ip* family everywhere. Items 2-4 are pool insurance (see
below for why that is not nothing).

### What did NOT work, with the numbers that killed it

1. **ipf at L=100: 11,253 vs ip1's 6,421 us/vol in the same race** (+75%!).
   Worse than the r1 miss-stream account predicts. Mechanism: each fused
   store waits on a ~40-cycle rsqrt14/rcp14 NR dependency chain; 100 such
   gated stores per tile choke the store buffer and collapse the x-pass's
   memory-level parallelism (the pass lives on ~100 misses in flight). At 50
   ipf is only ~1% behind ip (549.6 vs 542.7) — latency hides when the
   volume is L3-resident. So gen_powp's 50-win over me was NOT their ipf
   pick per se; adopting it did not move my 50 number. My DFT25 fusion did.
2. **ipe (map epilogue) at L=100: 8,163 vs 6,421** (+27%). Decoupling the map
   from the FFT stores was not enough: the epilogue's c reads add 100
   plane-stride miss streams per tile and its latency chains still sit
   between consecutive tiles' load bursts. At 50: 578 vs 543 (+6%). At 40 it
   ties ip (245.7 vs 246.8). Lesson recorded: on this node, ANYTHING
   interleaved into the 100-stream x-pass loses to a separate, perfectly
   sequential map pass — the sequential pass is BW-bound with its 4.5M map
   ops fully hidden (48 MB at ~20 GB/s ≈ 2.4 ms, ops ≈ 0.8 ms).
3. **NT stores at L=100: ipnt 7,299 vs ip1 6,421** (+14%). The state volume
   (16 MB) half-fits the 24 MB L3, so killing its L3 presence for phase 2's
   reads costs more than the deleted RFO. At L=40 B=8 quiet: ipfnt 417 vs
   auto 209 us/xform — disastrous when L3 is yours. BUT under neighbour
   contention ipfnt WON its race at 40 (250.3 vs ip0's 273.2): NT's value is
   contention-dependent, which is precisely why these stay raced candidates
   (and why the xarch hosts may pick them) rather than being deleted.
4. **Pre-RA scheduling on the L=50 family** (gen_powp's −5%-at-25 attribute
   trick, A/B'd at 50): 662 vs 472 us/xform, **+40%**. Killed. Their result
   does not transfer even to the size whose codelet shares the 5x5 CT.
5. The r1 record's "L=50 overlap tail costs ~8% extra ops" was miscounted:
   13 codelet calls per subpass is the true minimum however the tail is
   shaped (12.5 is not achievable with 4-wide vector lanes; a ymm tail
   codelet runs the same NUMBER of vector ops at half width). The only real
   cost is split loads/stores on the 800 B row stride, which no tail scheme
   fixes. Removed from the next-round list.

### Borrowed, plainly

- **gen_powp gen_r1**: the ipf chain family (their design, my engine), kept
  in the pool despite losing here — their wallaby data shows it winning on
  other hosts. Also their tryout.sh workaround notes (squeue PATH, manual
  check.py legs), which saved round trips.
- **gen_pfa_small gen_r1**: reinforced "do not put prefetch uops in
  issue-bound passes" and the fast-map ladder both entries already share.
- The DFT25 fused-store item was independently queued by both this entry (r1
  next-list #1, as "audit spills / T_ round-trips") and gen_powp (their #3);
  implementation here is mine.

### What I would do next (ranked)

1. **Adopt gen_race's gr_pick/wisdom cache** (deferred twice now): create()
   re-races every process — 6.5 s at L=100 vs the 50 ms warm budget the brief
   sets. The layer's key `entry/tag/L/B-bucket` + candidate-signature scheme
   fits this entry's pool as-is. Do this FIRST next round; it is also what
   pins picks across the driver's two-process repeatability cmp on hosts
   where families are NOT bit-identical.
2. **L=100 phase 1** is still ~2/3 of the step and now the only structural
   target left on the node: try plane-PAIR processing in p1 (two x-planes
   per sweep halves scratch traffic per byte streamed and doubles the
   sequential read burst from the state), and huge pages via gl_map_huge for
   the plane scratch + M.
3. **Round-3 generality**: DFT25M's (LDX, STO, KMAP) shape is exactly the
   generator interface the planner needs — a two-stage GT-PFA line codelet
   is (module A, module B, fold maps). Emitting PFA{n1xn2}C for any coprime
   pair with modules in {2,3,4,5,7,8,9,25} is now mostly mechanical.
4. **B=1 at 50/100 x-pass**: the race arena at nv=1 shows 10-15% wobble
   between pf0/pf1 pokes; a poke-distance knob (1 vs 2 lines) is a
   30-minute add via gr_pick_value once gen_race is adopted.

## Round gen_r3

Standings into the round: led all three owned sizes (40: 201.0; 50: 473.0 vs
gen_powp's 474.0; 100: 4947.5 vs their 5031.3). This round's plan: delete the
map pass's DRAM round trip at 100 (the one structural target left), adopt
gen_race's wisdom cache (deferred twice, and the brief's 50 ms warm budget is
a requirement this entry did not meet), and put down the round-3
any-size-in-class payment.

### What changed

**1. NEW chain family ipm* (deferred map, map-on-NEXT-step's-loads) — built,
raced, and it LOSES on this node (kept in the pool; see below).** The idea:
ipf (map in p2's stores, +75% at 100) and ipe (per-tile epilogue, +27%) failed
because the NR ladder sat in the miss-bound x-pass; a separate sequential map
pass costs a full state read + RFO + write per step (~32 of ~112 MB at
L=100). ipm instead leaves the state as the RAW FFT output z' and applies
map(z' + c) inside the NEXT step's phase-1 z-subpass loads (every element is
loaded there exactly once, before TRNC): the ladder gates LOADS feeding
compute — thousands of independent chains the OoO window can run ahead of —
not stores. fft3d_chain runs step 1 as plain execute, steps 2..m as p1m+p2,
one trailing map_vec; same op order as map_vec, so outputs stay bit-identical
to ip*. Measured (node race table, same window): l100-ipm1 6036.6-6107.6 vs
l100-ip1 5458.9-5545.5 us/vol (+11-12%); at 50 the driver chain with a
(wrongly) picked ipm0 ran 574.8 vs ~500 window-adjusted ip (+15%). Suspected
mechanism (consistent, not PMU-verified): the ladder's SWAP + max + rsqrt14 +
rcp14 uops land on port 5, which the z-subpass's TRNC (2 shuffles/vector) plus
the codelet's swaps already saturate — the standalone map pass pays those same
ops where NOTHING competes for port 5 and DRAM hides them. With stores (ipf),
epilogue (ipe), and loads (ipm) all measured losers at 100, **the map
placement axis on this node is exhausted: a separate, perfectly sequential
map pass is optimal for this engine. Do not revisit.** ipm stays raced
(rank behind ip*) as cross-arch insurance — SPR's second shuffle port could
flip it.

**2. The race arena had a bias that the new family exposed, now fixed: race
IN PLACE with a DISTINCT c buffer.** The r1/r2 race passed tin as both the
state and the c field, and ran cur -> dst out of place. For ip* that was a
fair-enough proxy; for ipm it halved the apparent read traffic (its
back-to-back state+c loads hit the SAME line), so the first r3 race picked
l50-ipm0 while the real chain regressed 574.8 vs ~500. TRIAL now runs
cfn(tout, tout, tcf, ...) — in place, separate 0.1-scaled c arena — exactly
the graded steady state (fft3d_chain calls cfn(out, out, c)). With the fix
the race re-picks ip* everywhere and its verdicts match the driver. Lesson
recorded: **race the graded workload including its STREAMS — buffer aliasing
in a race arena is a thumb on the scale.**

**3. gen_race ADOPTED (their r2 "worked patch" offer, my #1 deferred item):**
tune()'s verdict — winner name + chain-gate outcome, e.g. "l100-ip1.ch" — is
persisted via gr_wisdom_get_str/put_str under
gen_pfa_large/chain3/L<L>/B<bucket>#<gr_sig over candidate names>. Wisdom hit
= install with no arena, no refnd, no race: **measured warm create() 0.001 s**
(brief budget 50 ms; cold is 0.25-9.0 s). Only gate-passed verdicts are
stored; all families are bit-identical so a pick pin can never break
repeatability (it structurally guarantees it, per gen_race's rationale).
GENPFL_PF/GENPFL_NOFUSE bypass wisdom both ways; GEN_RACE_NO_WISDOM/
REFRESH/WISDOM honored by the layer. Following gen_powp's r2 protocol I
STRIPPED all gen_pfa_large/ keys from results/wisdom_a80n0.json at round end
(and kept it valid JSON): the monitor's process 1 cold-races in its full-quiet
window and stores its own verdict; process 2 hits it. Absent entries are
deliberate.

**4. NEW size L=80 = 16x5 (round-3 any-size-in-class duty, unscored).** The
clean candidate: 80%4==0 and 6400%4==0, so no tails anywhere. PFA80C = 16 x
DFT5 into T_[16*k2+n1], then 5 x DFT16 storing straight through ST via
k = (65 k1 + 16 k2) % 80 (65 = 1 mod 16, 0 mod 5). DFT16M = 4x4 CT in the
DFT25M fused-store shape: 8 CMULC twiddles (products d*k1 in {1,2,3,6,9},
long-double literals) + one exact -i (1 mul + swap), 81 FMA-port ops. pl
pitch 84 complex = 21 lines, odd. Validated on the node: rel L2 4.078e-16,
bit-repeatable, chain gate passes (pick l80-ip1). Raw m=1 B=8 execute is
2871.5 us vs MKL 2437 — the un-chained streaming call is not this engine's
shape (no graded case exists at 80; a round-6 draw would be chained, where
the owned chain and the race do the work). supports() now returns
40/50/80/100.

**5. NEW knob candidates ip2 (p2 poke distance 2 lines) and ipr1 (map pass
walking BACKWARD to reap p2's L3-hot tail).** p2's pf parameter is now a
distance. Both lose-or-tie at 100: ip2 5480.0 vs ip1 5545.5 (+1.2%, inside
the 3% tie band -> hysteresis keeps ip1), ipr1 5566.8 (wash — p2's own
read+write cycle already turns the state over, so the tail-hot residue is
worth ~nothing). Kept raced; the distance knob may matter on CLX/SPR.

Pool is now 15 candidates per size (13 at 50: no NT). pf ids: +11 ipm0,
12 ipm1, 13 ip2, 14 ipr1.

### Operation count

Unchanged at 40/50/100 (278/434/968 FMA-port vector ops per line; the ipm
family moves the ~18-op/vector map from its own pass into p1, total per step
identical). NEW L=80: 16x16 (DFT5) + 5x81 (DFT16) = 661 ops/line, 3 x 1600 x
661 = 3,172,800 per volume.

### Measured on the node (a80n0, leased core via tryout.sh, graded chain, min; same-window MKL 2022 alongside)

| case | r2 | r3 | same-window MKL | pick |
|---|---|---|---|---|
| L=40 B=8 m=128 | 208.8 (quiet) | **211.3** | 458.2 | ip0 |
| L=40 B=1 m=128 | 223.2 | **215.1** | 440.8 | ip0 |
| L=50 B=4 m=128 | 471.5 (quiet), 483-487 typ | **483.2** | 950.1 | ip1 |
| L=50 B=1 m=128 | 554.9 | **530.7** | 926.7 | ip1 |
| L=100 B=1 m=64 | 5008.8 (quiet, MKL 7807) | **5374.6** (MKL 8330; same 0.64 ratio) | 8329.9 | ip1 |
| L=80 B=8 m=1 (new, unscored) | — | 2871.5 execute | 2437.1 | ip1 |

Batched 40/50/100 are window-parity with r2 (this round's windows ran 4-6%
slow by the MKL yardstick); B=1 improved: 40 -3.6%, 50 -4.4% — the in-place
race arena ranks candidates honestly at nv=1 where the old out-of-place
arena wobbled 10-15%. Setup: cold 0.25-9.0 s (pool grew), warm 0.001 s
(wisdom). Gates, all graded cases, manual (tryout's remote map-check leg
still dies on the '$W/c.bin' quoting bug): single 3.6-4.5e-16 (tol 1e-12);
two-step m=2 2.361e-15 (50) / 2.721e-15 (100) vs tol 3e-14; full chains
3.804e-14 (40, anchor 2.612e-14) / 5.028e-14 (50, 2.922e-14) / 4.181e-14
(100, 2.416e-14), tol 1e-10; bit-repeatable across processes everywhere.

### What did NOT work, with the numbers that killed it

1. **ipm at 100: 6036.6-6107.6 vs ip1's 5458.9-5545.5 us/vol (+11-12%); at
   50: 574.8 vs ~500 us/xform (+15%).** See item 1 for the mechanism and the
   axis-exhausted conclusion. The traffic model (delete 32 of 112 MB ->
   -25%) was wrong because p1 at 100 is NOT purely DRAM-bound — its port-5 /
   issue budget is where the ladder lands.
2. **The race picked ipm before the arena fix** (l50-ipm0 stored, driver ran
   574.8 vs the 513.3 the fixed race's ip0 pick ran minutes later in the
   same window class). Do not race a chain step against an aliased c buffer.
3. **ipr1 (reverse map): 5566.8 vs 5545.5 at 100** — the L3-tail-reap
   hypothesis is worth ~0. **ip2 (poke 2): 5480.0 vs 5545.5** — real but
   inside the tie band; hysteresis correctly refuses to encode it.
4. Harness note for everyone: **the round rotation flipped impl -> impl_3
   MID-SESSION** (10:02, while this entry was being edited); six edits landed
   in a file that was rotated out from under the editor and silently
   vanished from the new working copy. If your file looks half-edited at
   round start, diff against impl_2/ before assuming you dreamed it.
   tryout.sh: the '$W/c.bin' map-check quoting bug is still there (run
   check.py yourself on the shared FS); reserve.sh --status still needs the
   slurm PATH shim on wallaby.

### Borrowed, plainly

- **gen_race gen_r2**: gr_wisdom_get_str/put_str, gr_keyf/gr_bucket/gr_sig —
  the whole warm-create path (their r2 next-list #1 was exactly this
  handshake). Also the tie/noise doctrine language.
- **gen_powp gen_r2**: the round-end wisdom-strip protocol (and its
  monitor-facing rationale), and the reminder that wisdom can pin a
  noisy-window pick (why the strip matters).
- **gen_powp gen_r1 / gen_pfa_small**: the "measure the map variant on YOUR
  engine" doctrine — applied in reverse here: the ladder that is optimal in
  a bare pass is what kills it inside a port-5-saturated pass.
- DFT16M reuses this entry's own DFT25M fused-store shape (gen_powp adopted
  it in their r2; the 16-point specialization is new here).

### What I would do next (ranked)

1. **L=100 p1 is the only residue left** (~2.6 ms of the 5.0 ms quiet step
   with p2 ~2.1 and map ~1.5 overlapping): the next real move is a PMU
   session (port 5 vs DRAM attribution) before any more speculative
   candidates — three families died on an unverified bottleneck model this
   round.
2. **Generality: 75 = 3x25 next** (needs a DFT3 module + gen_powp's
   odd-L^2 stash tail in p2), then 7/9/11/13 modules unlock 56, 63, 72, 88,
   99, 104, 112, 117 — DFT25M/DFT16M's (LDX, STO, KMAP) shape is the
   generator interface; coordinate with gen_planner on who serves what at
   round 6.
3. **Chain-time execute for unscored raw calls at 80**: the m=1 streaming
   execute loses to MKL; if round-6 grading ever times raw execute, add an
   out-of-place p1 variant with NT stores raced on that path.
4. **Cross-arch**: when the xarch report lands, check whether ipm/ip2 flip
   on SPR (two shuffle ports) — that is what they are in the pool for.

## Round gen_r4

Standings into the round: led 40 (202.4), tied 50 (474.0 vs gen_powp 473.7),
trailed 100 by 1.4% (5089.0 vs 5021.0). Plan: adopt the volume-major chain
schedule this entry alone never took (the one structural item three other
records agree on), and take one more shot at the L=100 map placement with the
granularity lesson from ipm's post-mortem.

### What changed

**1. VOLUME-MAJOR chain schedule (ADOPTED from gen_dense_prime / gen_rader /
gen_layout — the corpus-consensus shape all three records ship and my r1-r3
step-major fft3d_chain ignored).** Volumes are independent in the chain
algebra, so fft3d_chain now runs ALL m steps on one volume before touching the
next: the per-step working set drops from the whole batch's state+c (16 MB at
40 B=8 / 50 B=4, shared-L3 hostage) to one volume's state+c slice (2-4 MB,
L2/L3-resident for m-1 of m steps). Per-volume op order unchanged — outputs
bit-identical to step-major, so the pick machinery and repeatability are
untouched. L=100 is B=1 and unaffected. This alone: 40 B=8 m=128 went 211.3
(r3) -> **188.8** us/xform at sd 0.03% (MKL 423.8 same window — 2.24x, was
2.0x), and 50 B=1 went 530.7 -> **482.9**.

**2. NEW chain family ipp* (plane-granularity deferred map) — and it WINS at
L=100 (pick l100-ipp1).** Same schedule as r3's ipm (state holds raw FFT
output z' between steps; step 1 plain execute; one trailing map_vec) but the
map runs as map_vec's perfectly sequential per-plane PREPASS into an
L2-resident scratch plane (M's base, 160 KB at 100) that p1's z-subpass then
consumes. Same op order as map_vec — bit-identical. Rationale: ipm and ipp
have IDENTICAL per-step traffic accounting (~80 vs ip's ~112 MB at 100), and
ipm still lost 11-16% — so ipm's loss is pure fine-grain interference (the
ladder's port-5/latency footprint inside the granule-load stream), not
traffic. ipp keeps the traffic cut and pays the interference only at the
plane seam (~50k-uop granularity, invisible to the OoO window). Measured,
paired alternating runs (see below): quiet **4923-4974 vs ip1's 5099-5119**
(-3.4%), busy windows **5803-6039 vs 6558-6874** (-11%): the smaller DRAM
footprint is also contention armor. Race table (same session): ipp1 5180,
ip2 5341, ipp0 5357, ip0 5373, ip1 5406, ipr1 5497, ipm1 6032, ipe1 7089,
ipnt 8009, ipf1 11151, f0 13401. pf ids 15 (ipp0) / 16 (ipp1); raced at all
sizes (at 40/50 it ties ip within noise; race + wisdom arbitrate per host).

**3. The create() race now times the VOLUME-MAJOR schedule** (per volume: one
unmeasured warm step, then R timed steps on that volume, volumes outer; R =
8/6/4 by volume bytes <=2/<=8/>8 MiB, min over NR=3-4 interleaved rounds).
Racing the old step-major shape would rank candidates on a working set the
graded chain no longer has (at resident sizes the fused families' economics
change — gen_powp's residency rule). Wisdom tag bumped chain3 -> chain4 so a
stale step-major verdict can never be installed.

### Operation count

FMA-port vector ops per line unchanged (278 / 434 / 661 / 968 at 40/50/80/
100). ipp moves the map's ~18 ops/vector from its own pass into the per-plane
prepass (total per step identical to ip*) and adds one L2 plane round-trip
(write + strided read of mp, 160 KB) per plane; per-step DRAM accounting at
100 drops from ~112 to ~80 MB (state read once, y-subpass writes land on
prepass-warmed lines, no separate map pass).

### Measured on the node (a80n0, leased core, graded chain, min; windows this session swung +-15% — see lesson 1)

| case | r3 | r4 | notes |
|---|---|---|---|
| L=40 B=8 m=128 | 211.3 | **188.8** (sd 0.03%; MKL 423.8 same window, 2.24x) | volume-major; ip0 vs ipp0 paired = wash (185.9-196.6) |
| L=40 B=1 m=128 | 215.1 | **213.5** | parity |
| L=50 B=4 m=128 | 483.2 | **491.7** raw, ~476 window-adjusted (MKL 981.1 vs r3's 950.1) | ip1/ipr1/ipp1 paired all within noise (471.1-493.8) |
| L=50 B=1 m=128 | 530.7 | **482.9** (-9%) | |
| L=100 B=1 m=64 | 5374.6 (busy) / ~5009 quiet-class | **4923** quiet paired best; 4923-4974 quiet reps vs ip1 5099-5119; busy 5803-6039 vs ip1 6558-6874 | pick l100-ipp1; -3.4% quiet, -11% busy |
| L=80 B=8 m=1 (unscored) | 2871.5 | 2989.6 (window) | execute path unchanged |

Gates, all graded cases, manual (tryout's map-check leg still has the
'$W/c.bin' quoting bug): single 3.58-4.52e-16 (tol 1e-12); two-step m=2
2.090e-15 (50) / 2.721e-15 (100) vs tol 3e-14; full chains 3.804e-14 (40) /
5.028e-14 (50) / 4.181e-14 (100) — EXACTLY r3's values (bit-identical algebra
across the schedule change, as designed); single + chain outputs bit-identical
across two processes at all three sizes. Setup: cold 0.19-9.2 s (pool now 17
candidates; 60 s budget), warm wisdom ~1 ms. Round end: all 9 gen_pfa_large
keys STRIPPED from results/wisdom_a80n0.json (r3 protocol — the monitor
cold-races in its full-quiet window; absent entries are deliberate).

### What did NOT work / nearly went wrong, with the numbers

1. **I nearly killed ipp on a window-drift artifact.** Sequential forced A/B
   (ip1 for ~40 s, then ipp1, minutes apart in one lease) read ip1 5532 vs
   ipp1 6204 (+12%) — the OPPOSITE of the race's verdict — because this
   session's windows drift +-15% inside a single lease (same lease, ip1 read
   5099 first and 6319 last). Tight ALTERNATION (ip1, ipp1, ip1, ipp1 ...,
   samples=2 each, compared pairwise) settled it: ipp1 wins 4 of 5 pairs,
   -2.8/-3.7/-12/-11.5/+2.8%. Lesson for everyone: on this node, one-after-
   the-other forced runs are no longer an A/B; alternate within the lease
   and compare adjacent pairs, or trust the race's interleaved min-of-rounds
   (which was right here all along).
2. **ipp at 40/50 is a wash vs ip** (40: 185.9-196.5 vs 188.4-196.6 paired;
   50: all three families 471-494 in the same wobble) — the deleted map pass
   is L3-resident traffic there, worth ~nothing, exactly as the r2 ipe data
   predicted. Kept raced; hysteresis prefers ip on ties.
3. **L=50 B=4 batched did not improve** beyond window adjustment (491.7 raw
   vs 483.2 r3, MKL 3% slower same session): its 16 MB step-major set
   apparently already rode the L3 well enough on quiet windows. The B=1 -9%
   and the 40 B=8 -11% are where volume-major pays. No number killed
   anything — but do not claim 50 B=4 as a win on this evidence.

### Borrowed, plainly

- **gen_dense_prime / gen_rader / gen_layout (all since their r1/r2)**: the
  volume-major chain schedule and its residency rationale ("each volume runs
  all m steps cache-hot" — gen_layout r2's words). This entry was the last
  chain owner still step-major; adopting it was this round's cheapest win.
- **gen_powp gen_r3**: their ipm-at-100 post-mortem ("doubles phase-1
  compute; phase 1 is only ~half miss-bound") sharpened the ipp design
  question — same traffic as ipm, interference moved to plane seams — and
  their residency rule motivated re-racing families under the new schedule.
- ipp itself is new here (a granularity fix to my own r3 ipm), as is the
  volume-major race arena shape.

### What I would do next (ranked)

1. **Cross-arch (XARCH.md lands after this round's board)**: ipp's smaller
   DRAM footprint should shine on CLX (severe 512-bit downclock = more
   traffic-bound); verify the per-host race flips to it there, and check the
   40/80 NT variants on CLX as r2 predicted.
2. **Generality: 75 = 3x25** (DFT3 module + odd-L^2 stash tail from
   gen_powp), then 7/11/13 modules for 56/63/88/99/104/112/117 — the round-6
   surprise draw is the reason; coordinate with gen_planner on routing.
3. **L=100 p1 residue**: with the map pass gone (ipp), the step is p1m'+p2;
   the PMU session (port 5 vs DRAM attribution) is STILL the right next move
   before more candidates — this round's win came from granularity, not from
   a verified bottleneck model.
4. **L=50 B=4**: the one graded case volume-major did not move; if gen_powp
   pulls ahead there, the lever is probably a 2-volume-pair schedule (4 MB
   working set, still L3-safe) or their soa ideas at B=4 via 2-lane packing.

## Round gen_r5

Standings into the round: led 40 (188.7), led 50 B=4 by 1.4% (466.0 vs
gen_powp 472.9), DEAD TIE at 100 (4827.6 vs 4828.2 — 0.65 us). gen_powp's r4
adopted this entry's volume-major schedule and ipp family wholesale, so the
shared cells needed new structure, not another schedule shuffle. Session
baseline re-measured first (same windows as the r5 numbers below): 40 B=8
189.3 (MKL 404.3), 50 B=4 466.9 (MKL 947.8), 100 B=1 4778.2 (MKL 7730.4).

### What changed

**1. PAIR-PACKED map ladder (`map_step_pair`) — the round's win.** The map
z/(1+|z|) ran its ~19-op NR ladder (rsqrt14/rcp14 + two Newton steps each)
on vectors where every |z|^2 sits DUPLICATED in both complex lanes — half
the ladder lanes always computed nothing new. Now the 8 distinct |z|^2 of a
vector PAIR pack into one zmm (2 shuffles), ONE ladder serves both vectors,
and the reciprocals unpack pair-duplicated (2 shuffles): per pair of vectors
38 arith + 2 shuffles -> 21 arith + 4 shuffles, a 45% cut in map arithmetic.
BIT-IDENTICAL per element (q_re+q_im is IEEE-commutative; max/rsqrt14/rcp14/
FMA are elementwise; the NR expressions are verbatim), so all chain families
still agree bitwise, and every r4 gate number reproduced EXACTLY (see gates
below) — the strongest possible regression check. Applied in map_vec,
map_vec_rev, and the ipp prepass; the in-stream users (ipm granule loads,
ipf/f*/ipe stores) keep map_step_v. Why it pays: at 40/50 the step is
compute-bound and map ops were 46%/41% of the total (304k of 638k vector
ops at 40; 594k of 1408k at 50). Odd-count spans (L=50's 625-vector planes)
take a one-vector map_step_v tail.

**2. NEW c-stream L3-bypass families ipq1/ipk1/iqn1 (pf 17/18/19), aimed at
100.** Theory: per ipp step at 100, state+c = 32 MB streams through the
24 MB NON-INCLUSIVE L3; c has zero within-step reuse, so caching it evicts
the state that p2 and the next prepass re-read (~32 MB/step of avoidable
DRAM). ipq1 = ipp1 with PREFETCHNTA on the prepass c stream; ipk1 = ipp1
with CLFLUSHOPT on c lines one iteration after use (architectural semantics
where NTA fill policy is implementation-defined); iqn1 = ip1 + NTA-c map
pass. All compute bit-identical values. Verdict this session: **ipk1 WON
the interleaved create() race at 100 (4822.4 vs ipp1 4916.3, ip0 4920.1)**
— first direct evidence the c-pollution theory has teeth — but LOST all 4
held-lease adjacent pairs vs ipp1 (+1.3/+3.9/+18/+21%, worst in hot
windows; suspected cost is clflushopt's cross-core invalidation traffic
under load, ~250k flushes/step). NTA lost outright: ipq1 5539.8 (-12%),
iqn1 5395.7 — PREFETCHNTA is the wrong bypass mechanism on this ICL part.
All three stay raced (ranked last): CLX/SPR may flip them, and ipk1's race
win says the quiet-window economics are borderline-real. Wisdom tag
chain4 -> chain5.

**3. Rank reorder at 100 ONLY (gen_powp's r4 doctrine, their exact move):
ipp1/ipp0 to ranks 0/1 ahead of ip*.** The pair-packed map narrowed the
ipp1-vs-ip1 race gap under 1% and the 3% hysteresis reverted the pick to
ip1 (this session's first cold race stored l100-ip1). Held-lease paired
minima say ipp1's QUIET floor is lower — 4657-4668 vs ip1's 4842 this
session, on top of r4's paired quiet -3.4% / busy -11% — and the score is
measured on full quiet. A margin that shrinks under the tie band falls to
the measured winner. Verified post-reorder: the cold race now installs
l100-ipp1.ch.

### Operation count

FFT vector ops per line unchanged (278/434/661/968 at 40/50/80/100). Map
arithmetic per volume-step: 40: 304k -> 168k; 50: 594k -> 328k; 100 (ipp
prepass): 4.75M -> 2.63M; shuffles double to 4 per pair (port 5, but the
sequential map paths have no port-5 competition).

### Measured on the node (a80n0, leased cores via tryout.sh; session-first
### baselines above were taken in the same window class, minutes apart)

| case | r4 (same session) | gen_r5 | delta | same-window MKL |
|---|---|---|---|---|
| L=40 B=8 m=128 | 189.3 | **162.8** (sd 0.05%) | -14% | 416.0 (2.56x) |
| L=40 B=1 m=128 | 213.5 (r4 board) | **185.8** | -13% | 441.0 |
| L=50 B=4 m=128 | 466.9 | **426.6** (sd 0.02%) | -8.6% raw, window ran +6.7% HOT by MKL | 1010.9 (2.37x) |
| L=50 B=1 m=128 | 482.9 (r4 board) | **419.2** | -13% | 928.7 |
| L=100 B=1 m=64 | 4778.2 | paired r5-ipp1 beats r4-ipp1 **4/4**: 4737/5072/4817/4914 vs 5197/5283/6357/5103 (-4 to -9%); session quiet floor **4657** | ~-4% | window swung 7730-8913 |
| L=80 B=8 m=1 (unscored) | 2989.6 (r4) | 2916.2 | execute path untouched | 2471.4 |

Gates, all graded sizes, by hand (tryout's '$W/c.bin' map-check quoting bug
is STILL there): single 3.58-4.52e-16 (tol 1e-12); two-step m=2 1.857e-15
(40) / 2.361e-15 (50) / 2.721e-15 (100) vs tol 3e-14; full chains 3.804e-14
(40) / 5.028e-14 (50) / 4.181e-14 (100) — all three EXACTLY r4's values
(bit-identity across the map rewrite, as designed); outputs bit-identical
across two processes at 40/50/100. Setup: cold 0.3-10.4 s (pool now 20/18
candidates; 60 s budget fine), warm wisdom ~1 ms. Round end: all
gen_pfa_large keys stripped from results/wisdom_a80n0.json (r3 protocol).

### What did NOT work / incidents, with the numbers

1. **PREFETCHNTA on the c stream: ipq1 5539.8 vs ipp1 4916.3 (+12.7%),
   iqn1 5395.7 vs ip1 4953.4 (+8.9%)** in the same race. Either NTA lines
   on this part still fill a level p2 does not benefit from, or the L1-only
   fill re-misses before the demand load. The FLUSH variant is the working
   mechanism for the same idea (ipk1 won that race) — do not spend another
   round on NTA prefetch here.
2. **ipk1's race-vs-paired contradiction** (race winner by 1.9%; loses 4/4
   held-lease pairs by up to +21% in hot states). Doctrine to record: the
   interleaved race can flatter a candidate whose cost is SUSTAINED-LOAD
   snoop/invalidation traffic; held-lease adjacent pairs are the arbiter.
   Kept raced and ranked last — the monitor's full-quiet race decides, and
   xarch may genuinely flip it.
3. **I corrupted results/wisdom_a80n0.json** doing the round-end strip with
   a bare python rewrite: ignored the .lock file AND the nested
   {host, format, entries} layout; a concurrent gen_planner write got
   orphaned mid-file and the file was unparseable until repaired. Fixed
   under flock, the stray (newer) gen_planner/tree/L27 entry merged back,
   my keys stripped from d['entries']. Lesson for EVERYONE: take
   fcntl.flock on wisdom_<host>.json.lock for ANY hand edit, and edit
   ['entries'], not the top level.
   Timeline note for the monitor: after my 20:36 repair the file held 46
   entries (gen_race/gen_powp/gen_planner/mine); at 20:42 a concurrent
   gen_planner session rewrote it to 7 gen_planner/*/g8 keys only (their
   new key schema). My locked strip afterwards removed exactly my one
   remaining key. Absent entries are legitimate per protocol, but the
   40-entry drop at 20:42 was not this entry's write.
4. **Window drift nearly mislabeled the round twice**: the first r5 tryout
   at 100 read 5043 vs the 4778 baseline (-5.5% "regression") — the pick
   had silently gone to ip1 via the tie band AND the window had shifted;
   the held-lease pairs (r5 vs r4 alternating, same core) showed r5 winning
   4/4. And a later within-lease A/B swung +-8% between reps 3 and 4.
   gen_batchlane r4's protocol (alternate within ONE lease, compare
   adjacent pairs, read minima) was load-bearing for every conclusion this
   round.

### Borrowed, plainly

- **gen_batchlane gen_r4** (via gen_powp r4): the held-lease alternation
  protocol.
- **gen_powp gen_r4**: the rank-first-for-the-measured-winner doctrine,
  applied here to ipp at 100 exactly as they applied it.
- **gen_layout gen_r4**: the RFO/LFB attribution mindset behind the
  c-pollution hypothesis. Their NT-store fix itself does NOT transfer to
  this engine (my p2 stores are read-modify-write on just-read lines and
  the y-subpass lands on prepass-warmed lines; gen_powp r4 reached the
  same conclusion) — recorded so nobody re-derives it a third time.
- The pair-packed ladder and the flush-based c bypass are new here.

### What I would do next (ranked)

1. **XARCH: ipk1 on CLX/SPR.** Smaller LLC footprint, no dependence on ICL
   NTA policy; CLX's 1 MB L2 + downclock make the traffic economics
   sharper. The race + per-host wisdom arbitrate — check the advisory
   report's picks before touching ranks again.
2. **The FFT codelets are now the whole step at 40/50** (map cut from 46%
   to ~30% of ops): the four-round-old PMU/spill audit of p1's Zv/Wv/T_
   staging (80-300 live vectors per line at 100) is finally the top lever.
3. **Generality for round 6**: 75 = 3x25 (DFT3 module + gen_powp's odd-L^2
   stash tail), then 7/11/13 modules for 56/63/88/99/104/112/117;
   coordinate with gen_planner on routing before the surprise draw.
4. **L=50 B=4 2-volume-pair schedule**: still unmeasured by anyone; the
   cell is now mine by 46 us but gen_powp's record queues the same idea.

## Round gen_r6

Standings into the round (r5 board): led 40 (160.2 vs next entry 281.5), led
100 by 1.9% (4531.4 vs gen_powp 4618.7), trailed 50 B=4 by 1.3% (421.0 vs
415.6). But round 6 is the SURPRISE round: the monitor draws three
never-announced sizes in 14..127 and scores the assembled library, not the
acceptance cells. The class duty this round is therefore COVERAGE — my r3-r5
next-lists queued it three times ("75 = 3x25, then 7/11/13 modules") and this
round pays it in full. The scored sizes' generated code ships UNCHANGED
(verified at the instruction level, see below).

### What changed

**1. SIXTEEN new class-coverage sizes: every two-stage coprime composite in
14..127 with modules in {2,3,4,5,7,8,9,11,13,16,25} and L >= 44** — 44, 48,
52, 55, 56, 63, 65, 72, 75, 77, 88, 91, 99, 104, 112, 117. (Sizes below 44
are gen_pfa_small's class; announced sizes excluded; the remaining composite
holes 60/84/90/96/105/108/120/126 need three-factor GT or DFT27/DFT32
modules — see next-list.) Each is a lean instantiation of the same two-sweep
GT-PFA engine with maps n = (N2 n1 + N1 n2) % L, k = (A k1 + B k2) % L,
A === 1 mod N1, 0 mod N2, B === 1 mod N2, 0 mod N1, long module second
storing straight through ST. All 16 maps and CRT constants were validated
against numpy BEFORE writing the C (scripts inline in the session; the same
brute-force harness is 20 lines and reusable).

**2. New modules.** `DFTODDM` — ONE generic direct symmetric odd-N DFT macro
(a_j = x_j + x_{N-j}, b_j = x_j - x_{N-j}; X_k = R_k -+ iI_k) serving
N = 3, 7, 9, 11, 13 from exact long-double cos/sin tables indexed by
(j*k) % N (the C25T compile-time-fold trick; index 0 occurs at composite N,
e.g. j=k=3 at N=9 — tables carry the full 0..N-1 range). Works for
composite odd N (used for 9). 7/33/52/75/102 FMA-port ops at
N = 3/7/9/11/13. `DFT8M` — PFA40C's stage-2 radix-2 DIT DFT8 refactored
into a reusable fused-store macro (PFA40C itself untouched — its code is
scored). Both take LDX *and* STO as macro parameters plus a separate
compile-time index-map parameter (IMAP/KMAP) — the r2 rescan lesson means a
file-scope helper may do index MATH but must never name the wrapper's LD/ST.

**3. GEN_LEAN instantiation mode.** Coverage sizes emit only the six
families that win on this node (ip0/ip1/ip2/ipr1 + ipp0/ipp1) and compile
the two heavy bodies (p1body, p2) once as noinline functions: ~4 codelet
expansions per size instead of ~50. Full build with 20 sizes: ~60 s on the
node — tryout-compatible. The four scored sizes keep the full pool and
always_inline bodies. tune()/create()/supports() became table-driven
(g_sizes: factors, GPP pitch, pool per row); wisdom tag stays chain5 (new
sizes get their own keys via the pool signature).

**4. Odd-L machinery (first odd L in this entry).** Phase 2 gets a stash
tail for NPL % 4 == 1 (BORROWED from gen_powp's odd-L^2 stash idea): the
last 4 flat columns are stashed raw before the in-place sweep, full tiles
cover 0..NPL-2, one final tile transforms the stash into NPL-4..NPL-1 — the
3 recomputed columns read RAW stashed inputs, so the in-place recompute is
idempotent. Phase 1's overlap groups already handled GENL % 4 in {1,2,3}.

**5. BUG found by the create() chain gate, fixed: odd-L map truncation.**
map_vec's vector count VDL/8 TRUNCATES at odd L (2L^3 % 8 == 6): the last 3
complex of every volume were never mapped, and the ipp prepass (2L^2 % 8 ==
2) read one uncopied complex per plane. The execute path has no map, so
check.py PASSED while the chain was wrong — the gate stored `l75-ip1.x`
(chain refused, silent fallback to execute+scalar map) which is what made
me look. Fix: `map_span_tail` — one masked (maskz_loadu/mask_storeu) ladder
step for the 1-3 leftover complex, zero-filled lanes kept finite by the
existing 1e-300 max guard. All 13 lean verdicts now store `.ch`. Lesson:
**when a span length is not a multiple of the vector granule, grep every
`/ 8` at the callsites, not just the loop tails** — and trust the `.x`
suffix in wisdom as a red flag, it is the gate speaking.

**6. Codegen-drift regression on the SCORED sizes, caught and fixed.**
Adding 16 instantiations doubled the translation unit; gcc's unit-growth
inlining budget flipped, and map_vec — which r5 compiled as
`map_vec.constprop.N` clones (constant trip counts) CALLED by the chain
fns — became inlined into every chain fn instead. Held-lease alternation at
100 B=1: the drifted build lost 4/4 pairs by ~+0.9% (medians 4766-4781 vs
4716-4741). Fix: per-size noinline wrappers FN(map_vol)/FN(map_pln) with
compile-time counts — the constprop clone reproduced by construction,
immune to the inliner's mood; map_vec_rev (single caller) restored to
always_inline as r5 had it. After the fix: xc_ipp1_100, xc_ip0_40,
xc_ip1_50, x_pf1_100 are INSTRUCTION-IDENTICAL to the r5 object (objdump
diff), and the A/B rematch shows r6 winning 3/4 pairs at 100 (floor 4726 vs
r5 4741-4946 that window), dead tie at 50 (433.0 vs 433.6), wash over 5
pairs at 40 (min 163.8 vs 164.7). Only the bottom-ranked ipq1/ipk1
insurance candidates still differ in size — accepted. Lesson for everyone:
**growing a shared TU flips GCC inline/clone decisions in code you did not
touch; diff `nm -S` per function against the previous round's object before
trusting any timing, and pin hot callsites structurally (hand clones), not
with global attributes.** (Also: `#if VDL % 8` fails silently-loudly — a
(size_t) cast is illegal in preprocessor arithmetic; use `#if GENL % 2`.)

### Operation count

Scored sizes unchanged (278/434/661/968 ops/line at 40/50/80/100). New
lines (stage1 + stage2 FMA-port vector ops): 44: 388, 48: 355, 52: 512,
55: 551, 56: 446, 63: 661, 65: 718, 72: 650, 75: 751, 77: 888, 88: 886,
91: 1143, 99: 1247, 104: 1154, 112: 1095, 117: 1594. Odd-L adds one stash
tile per p2 (GENL vec loads + 1 codelet) and 1-3 masked map elements per
volume.

### Measured on the node (a80n0, leased cores; B=1 raw execute, min, with
### same-window MKL 2022 — the graded chain at these sizes is round-6's
### business, where the owned chain's deleted map pass adds its usual edge)

| L | mine | MKL | ratio | pick (chain gate) |
|---|---|---|---|---|
| 44 | 236.7 | 359.4 | 1.52x | ip1.ch |
| 48 | 327.2 | 337.9 | 1.03x | (16x3; MKL loves 3-smooth) |
| 52 | 426.7 | 619.7 | 1.45x | ip1.ch |
| 55 | 576.2 | 790.7 | 1.37x | ip1.ch |
| 56 | 503.3 | 679.6 | 1.35x | ipp0.ch |
| 63 | 820.0 | 1244.6 | 1.52x | ip1.ch |
| 65 | 1036.0 | 1358.0 | 1.31x | ip1.ch |
| 72 | 1118.2 | 1318.8 | 1.18x | ipp0.ch (B=8: 1794.8 vs MKL 1739.1) |
| 75 | 1401.5 | 2066.5 | 1.47x | ip1.ch |
| 77 | 1942.7 | 2491.5 | 1.28x | ip2.ch |
| 88 | 2640.4 | 3104.8 | 1.18x | ipp0.ch |
| 91 | 3501.9 | 4216.2 | 1.20x | ip1.ch |
| 99 | 4418.3 | 6322.9 | 1.43x | ip1.ch |
| 104 | 5765.6 | 6711.1 | 1.16x | ipp1.ch |
| 112 | 9064.7 | 8260.3 | **0.91x — the one loss** | ipp1.ch |
| 117 | 10497.1 | 12053.5 | 1.15x | ipp1.ch |

Scored sizes, held-lease A/B r5-vs-r6 binaries (same core, alternating,
min): 100 B=1 r6 4726-4749 (r5 4716-4946 across windows, verdict wash after
the fix); 50 B=4 433.0 vs 433.6 (tie); 40 B=8 wash over 5 pairs (163.8 vs
164.7 min). Winner functions instruction-identical — the r5 numbers stand.

Gates (final source, node): single call 3.5-5.0e-16 at all 20 sizes (tol
1e-12); two-step m=2 at 75: 3.010e-15 (tol 3e-14); full owned chains
m=8 at 75: 2.377e-14 (anchor 5.480e-15, tol 1e-10) and m=6 at 117:
5.938e-15 (anchor 3.941e-15) — both through the odd-L tail and (117) the
ipp prepass; bit-repeatable across processes everywhere. Setup: cold
0.19-3.5 s over the 16 new sizes (60 s budget; refnd dominates at 117),
warm wisdom ~1 ms. Round end: all gen_pfa_large keys stripped from
results/wisdom_a80n0.json under flock (a concurrent session had again
rewritten the file mid-day, exactly the r5 incident pattern — my strip
found only my own keys present and left valid {host, format, entries}).

### What did NOT work / incidents, with the numbers

1. **The odd-L map truncation** (item 5 above): would have shipped a chain
   that silently falls back to execute+scalar at every odd L (the r1-era
   driver-map baseline cost ~1.6x at comparable sizes). Cost of detection:
   one wisdom read. The gate machinery, not the test suite, caught it.
2. **The unit-growth codegen drift** (item 6): +0.9% at 100, 4/4 pairs —
   would have handed the tightest cell on the board (0.65 us at r4!) back
   to gen_powp. Two failed fixes first: always_inline on map_vec (wrong
   direction — reproduces the inlined form), plain noinline (kills the
   constant-trip clones; generic runtime-bound loop). The hand-clone
   wrapper is the one that reproduces r5 exactly.
3. **L=112 loses to MKL by 9%** (9065 vs 8260 raw execute). 112 = 2^4 x 7
   is MKL's home turf; my DFT16-second GT line (1095 ops) is not enough.
   Candidates for next round: 4x4x7 three-stage line, or routing 112 to
   gen_pow2-style radix machinery. Do NOT count 112 as a safe cell.
4. tryout's '$W/c.bin' map-check quoting bug is STILL there (fifth round);
   run check.py by hand for chained validation.

### Borrowed, plainly

- **gen_powp**: the odd-L^2 stash tail idea for p2 (their record, my
  implementation in this engine).
- **gen_batchlane gen_r4 (via everyone)**: held-lease same-core alternation
  — it caught both the +0.9% drift and its fix's verification.
- **gen_race**: the wisdom layer's `.x`/`.ch` verdict suffix — this round it
  doubled as a correctness alarm.
- **gen_pfa_small gen_r6 (concurrent)**: their gfactor widening list showed
  the class-boundary overlap early; sizes < 44 left to them, the trunk race
  arbitrates the overlap band. DFTODDM's direct symmetric form is textbook
  (validated vs numpy); DFT8M is my own PFA40C refactor.

### What I would do next (ranked)

1. **Three-factor GT lines** (or composite modules DFT10/DFT15 = internal
   2x5/3x5 GT) unlock the remaining composite holes 60, 84, 90, 105, 120,
   126 — with DFT27/DFT32 modules covering 96/108. That closes the class
   completely for any future draw.
2. **112**: three-stage 4x4x7 or a pow2-grade DFT16; it is the only cell
   losing to a library.
3. **DFT11/DFT13 are O(h^2) direct** (75/102 ops); Rader or Winograd forms
   would roughly halve them if 44/52/55/65/88/91/99/104/117 ever become
   scored cells — not worth it for a library-existence round.
4. **XARCH**: the lean pools carry ipp0/ipp1; check the advisory picks on
   CLX (smaller L2 sharpens the prepass economics) before touching ranks.
5. The scored-size next-list stands from r5: PMU attribution of p1 at 100,
   and the 50 B=4 two-volume-pair schedule (still unmeasured by anyone).

## Round gen_r7

Standings into the round (r6 board): led 40 (160.6 vs next 237.0), led 100
(4570.3 vs gen_powp 4617.5), trailed 50 B=4 by 1.1% (420.0 vs 415.5).  The
rounds-7/8 extension brief names this entry's queued backlog explicitly:
two-axes-per-pass fusion at 100/50/40 (lit 11 Tier 2).  Also on the desk at
round start: gen_powp's r6 record (they ranked MY ipk1 c-flush bypass FIRST
at 100 on 5/5 held-lease pairs and built a "challenger playoff" in their
race), the SPR advisory flagging the 100 winner flipping to gen_powp there,
and the surprise-round results (trunk won all three draws; L=44 B=8 rode
this class's engine through gen_race at 480.4 vs MKL 620.3 — no planner
or coverage bug to firefight, the r6 coverage sizes did their job).

### What changed

**1. CHALLENGER PLAYOFF in tune() (ADOPTED from gen_powp gen_r6, their
design, ~35 lines).**  The R-step race trials systematically under-read
candidates whose benefit accrues ACROSS steps — the c-flush bypass pays its
flush cost in-step and collects on later steps' state re-reads; gen_powp
measured ipk1 +5.3% in a 4-step cold race vs -0.6..-16% in 5/5 graded
pairs.  When the rank-0 live candidate is not the trial best but within
15%, the two are re-judged on long steady runs (24 steps/arm/round at
<= 8 MiB volumes, 12 above; 3 rounds, alternating) and the results feed
tc[] via min() — fair to both arms; the 3% simplest-first hysteresis still
decides.  Portable by construction: it runs wherever the race runs, so the
CLX/SPR cold races get the same long-horizon fairness without any
host-specific rank hacks.  Setup cost measured: cold create at 100 went
~4.4 -> 4.8 s (60 s budget untroubled); warm wisdom stays ~1 ms.

**2. ipk1 rank-first at 100: TRIED (gen_powp's r6 evidence) and REFUTED on
this engine — ipp1 keeps rank 0, ipk1 moves to rank 2.**  Held-lease
same-core alternation, L=100 B=1 m=64, five pairs (ipk1 first in each
pair): 5288.1/4836.6, 4843.2/5149.3, 5775.0/4938.0, 4804.5/4697.9,
4805.9/4645.5 us/xform — **ipp1 wins 4/5, min-of-mins 4645.5 vs 4804.5
(-3.3%)**, matching my r5 session (4/4 to ipp1) and CONTRADICTING
gen_powp's 5/5 to ipk1 on their side of the shared shell.  Boundary
recorded for everyone: the c-flush-vs-plain-prepass verdict at 100 is
ENGINE-SPECIFIC (likely the flush uops interact differently with each
engine's prepass/p1 port mix); do not import the rank, import the race
machinery that measures it per host — which is what item 1 does.  ipk1 at
rank 2 (ahead of plain ip*) keeps it the preferred bypass insurance for
CLX/SPR wisdom races, where it has been trial-best repeatedly.

**3. ipk1 extended to the 16 LEAN coverage sizes** (pool 6 -> 7 candidates,
rank last): the L >= 91 volumes are 12-26 MB — the same beyond-L3
c-pollution regime as 100.  p1pmk moved out of the non-lean guard and
gained a masked odd-plane tail (`#if PLNDL % 8` map_span_tail — the r6
odd-L truncation lesson applied BEFORE the bug this time; compiles out at
the scored sizes).  Forced-pick verification at the two odd lean sizes:
GENPFL_PF=18 chains m=8 pass at 75 (rel 2.377e-14 — EXACTLY r6's ip1 value,
i.e. bit-identical family algebra through the new tail) and 117 (6.889e-15).
Pool signatures change, so every lean size gets fresh wisdom keys by
construction.  Wisdom tag chain5 -> chain6 (pool order + race shape both
changed; no stale verdict can replay).

**4. Two-axes-per-pass fusion (lit 11 Tier 2, section "Few-pass 3D +
fusion") — worked through and CLOSED WITHOUT CODE; the accounting is the
deliverable.**  This engine has fused two axes per DRAM pass since r1:
phase 1 transforms z AND y of each x-plane through the L2-resident plane
scratch (one state read, one state write for two axes).  The literature's
proposed re-cut (pass A = z only in place; pass B = y+x fused over kz-slab
tiles, 640 KB L2 buffer at 100) has BIT-IDENTICAL DRAM accounting per ipp
step at L=100: pass A 16r state + 16r c + 16w state, pass B 16r + 16w —
the same ~80 MB/step as today's map-prepass+z+y | x split.  A one-pass
3-axis step is impossible at volume >> cache (the x-stage DFT25 alone
couples 25 planes; full 3D data dependence), so two passes is the floor
and two-axes-per-pass is already the optimum pass shape.  The ONLY
reducible term is pass B's DRAM share via L3 custody of the state across
the step — and that is precisely the c-bypass family built in r5 (ipk1/
ipq1/iqn1), now playoff-protected per host.  gen_powp: this closes the
"coordinate so only one of us burns the round" item from your r5/r6
next-lists — the accounting transfers to your engine (your phase 1 is the
same two-sweep shape); the per-host question is only ipk1-vs-ipp1, which
your playoff and mine now both measure honestly.

### Operation count

Unchanged everywhere: 278/434/661/968 FMA-port vector ops per line at
40/50/80/100, r6 coverage counts as listed in the r6 section.  The playoff
moves no arithmetic; lean ipk1 adds none (flushes only).  Scored-size
generated code verified UNCHANGED at the instruction level (below).

### Measured on the node (a80n0, leased cores via tryout.sh, graded chain
### min, same-window MKL 2022 alongside)

| case | r6 board | gen_r7 (this window) | same-window MKL | pick |
|---|---|---|---|---|
| L=40 B=8 m=128 | 160.56 | 172.1 (window) | 406.7 (2.36x) | ip0.ch |
| L=40 B=1 m=128 | — | 185.5 (r5-class: 185.8) | 441-class | ip0.ch |
| L=50 B=4 m=128 | 420.02 | 437.9 (window) | 955.7 (2.18x) | ip1.ch |
| L=50 B=1 m=128 | — | 433.9 | — | ip1.ch |
| L=100 B=1 m=64 | 4570.27 | **4634.0 min, MKL 7872.3 same window (ratio 0.589 — the best ratio this cell has printed)** | 7872.3 | ipp1.ch |
| L=75 B=1 m=1 (lean, unscored) | — | 2330.0 exec | 2071.3 | ip1.ch |
| L=117 B=1 m=1 (lean, unscored) | — | 8869.7 exec | 10990.3 | ipp1.ch |

Windows this session ran 4-7% hot by the MKL yardstick at 40/50 (406.7 vs
the 380-class quiet MKL); the codegen identity check below is the real
regression armor.  Gates, by hand (tryout's '$W/c.bin' map-check quoting
bug is STILL there, sixth round): single call 3.578-5.035e-16 at every size
measured (tol 1e-12); two-step m=2 at 100: 2.721e-15 (tol 3e-14) — the
EXACT r3-r6 value; full chains 8.557e-14 (40 B=1 m=128), 2.538e-14 (50
B=1), 3.804e-14 (40 B=8), 4.181e-14 (100 m=64) — all historical values,
bit-identical algebra; chained output bit-repeatable across independent
processes at 100.  Setup: cold 0.25-4.8 s, warm wisdom ~1 ms.  Round end:
all gen_pfa_large keys stripped from results/wisdom_a80n0.json under flock
(the file had again been wholesale-rewritten by concurrent create() races
mid-session — my earlier L40/B8 verdict vanished on its own, the r5/r6
incident pattern; absent entries are deliberate per protocol).

### What did NOT work / verification notes, with the numbers

1. **ipk1 rank-first at 100: refuted, 4/5 pairs to ipp1** (numbers in item
   2).  The cross-engine contradiction with gen_powp r6 is the finding:
   rank evidence does NOT transfer even between engines sharing the shell;
   race machinery does.
2. **The r6 codegen-drift check needs address normalization to be usable.**
   nm -S sizes matched r6 for all 13 hot functions, but raw
   objdump-text diffs showed 764-2470 changed lines — ALL of it
   intra-function jump-target addresses shifted by the 16 new lean
   functions moving .text around.  After stripping addresses/target labels
   (sed on the operand column), xc_ipp1_100, xc_ip0_40, xc_ip1_50 and
   x_pf1_100 are INSTRUCTION-IDENTICAL to the r6 object.  Protocol update
   for everyone: `nm -S` first (cheap), then normalized objdump; a raw
   objdump diff on a grown TU will cry wolf every time.
3. The challenger playoff CAN still install ipk1 in a noisy window (my
   first cold race this session did exactly that before the rank revert;
   the verdict gate-passed, so it is a performance coin-flip, not a
   correctness risk).  With ipp1 at rank 0 the quiet-floor evidence
   (4645 vs 4804) says the monitor's full-quiet race installs ipp1.
4. tryout's map-check leg still dies on the '$W/c.bin' quoting bug AND its
   repeatability cmp never runs at chained sizes (it sits after the failed
   check in the && chain) — run both by hand.  reserve.sh --status still
   needs the slurm PATH shim on wallaby
   (/opt/software/slurm-19.05.8.1-cuda-11.8/bin).

### Borrowed, plainly

- **gen_powp gen_r6**: the challenger-playoff design, adopted nearly
  verbatim into tune() (their soa-playoff sibling does not apply — no SoA
  arena here), and their ipk1-at-100 evidence, which motivated the try
  that my pairs then refuted.  Both directions of that exchange are the
  system working as intended.
- **gen_batchlane gen_r4 (via everyone)**: held-lease same-core
  alternation, again the arbiter for the round's one rank decision.
- **gen_powp r2-r6 / this entry r3-r6**: the round-end wisdom-strip
  protocol, executed under flock per my own r5 incident rule.
- The two-axes-per-pass closure accounting, the lean ipk1 odd-tail
  extension, and the normalized-objdump protocol update are mine.

### What I would do next (ranked)

1. **PMU attribution of p1 at 100** (queued since r3, now four rounds):
   with the map placement, schedule, ranks and traffic accounting all
   exhausted, port-5-vs-DRAM attribution of the remaining ~2.5 ms of p1 is
   the only honestly unknown quantity left in the weakest cell.  Do it
   before writing ANY new candidate.
2. **Three-factor GT lines** (or composite modules DFT10/DFT15, DFT27/
   DFT32) for the remaining class holes 60/84/90/96/105/108/120/126 — the
   surprise round drew 96 and the trunk had to serve it from another class.
3. **112 = 16x7 still loses to MKL** (r6: 0.91x raw execute) — three-stage
   4x4x7 line or a pow2-grade DFT16 are the candidates.
4. **50 B=4 two-volume-pair schedule**: still unmeasured by anyone, cell
   still a ~1% coin flip with gen_powp.
5. **XARCH**: SPR flagged the 100 winner flipping to gen_powp; the playoff
   + rank-2 ipk1 are this round's answer — check the next advisory's picks
   before touching anything else.

## Round gen_r8

Standings into the round (r7 board): led all three cells — 40 (159.96 vs
next entry 224.9), 50 B=4 (413.96 vs gen_powp 415.07, 0.3%), 100 (4475.3 vs
4549.6, 1.7%). The extension brief's backlog for this entry was already
spent (two-axes closed in r7; gen_pow2's r7 killed constant-per-site
routing; gen_powp's r7 showed 3-shear twiddles are split-complex-only and
the lifted DFT5 loses on interleaved slot engines — both read before
touching anything, both saved a burn). What remained was my own #1 item,
queued since r3: ATTRIBUTION of p1, for which round 8 shipped the static
analyzers (PMU locked at paranoid=4). This round did the attribution
properly, built the fix it pointed at, measured it, and REVERTED it — the
shipped code is instruction-identical to r7 (.text byte-compared). The
attribution table and the machine calibration are the round's deliverable.

### 1. Machine calibration (leased-core rdtscp microbench, portcal2.c)

llvm-mca's icelake-server model dispatches ALL 512-bit FP to port 0 at
1/cycle. That contradicts the Gold 6326 spec (2x512-bit FMA), so before
trusting any model output I measured (TSC-timed dependent/independent
chains, mins of 3, core clock calibrated off the latency-1 shuffle chain;
core ran ~3.26 GHz):

- zmm FMA: latency 4, throughput **2.0/cycle** (P=12 chains: 6.0 cyc/iter,
  P=16: 8.0) — two real 512-bit pipes, ports 0+5.
- zmm shuffle (vshufpd): latency 1, throughput **1.0/cycle** (port 5).
- Mixed 8 indep FMA + K indep shuffles per iter: K=4/8/12 -> 6.0/8.0/12.0
  cycles — EXACTLY the p0/p5 model (shuffles steal FMA slots 1:1 past 2
  uops/cyc).

So: **llvm-mca ICX port tables are wrong on this SKU** (FMA port0-only) —
use mca only for relative schedule questions. uiCA could not arbitrate:
its install is incomplete (instructions.xml download from uops.info times
out; no outbound net). The microbench source is left for everyone at
build/tryout/gen_pfa_large/portcal2.c.

### 2. Phase attribution (rdtscp counters on a dev copy, graded chains, forced picks)

Per-iteration CORE cycles (TSC x1.138) vs the two candidate floors —
"ALU" = (FP + shuffles)/2 on the calibrated 2-port model, "fetch" = loop
body bytes / 16 B/cyc (MITE legacy decode; these bodies are 5.3-17.8 KB,
far past the DSB's ~384x32B-window capacity):

| cell/pick | phase | meas | ALU floor | fetch floor | share of step |
|---|---|---|---|---|---|
| 100/ipp1 | prepass | 55.9k/plane | 15.6k | (DSB loop) | 35% |
| 100/ipp1 | zsub | 1410/yg | 768 | 1111 | 22% |
| 100/ipp1 | ysub | 898/zg | 568 | 770 | 14% |
| 100/ipp1 | p2 | 1881/tile | 568 | 832 | 29% |
| 50/ip1 | zsub | 720/yg | 357 | 506 | } p1 52% |
| 50/ip1 | ysub | 461/zg | 253 | 334 | } |
| 50/ip1 | p2 | 434/tile | 253 | 369 | 19% |
| 50/ip1 | map | 429k/step | 164k | (DSB loop) | 29% |
| 40/ip0 | zsub | 403/yg | 242 | 331 | — |
| 40/ip0 | ysub | 200/zg | 162 | 216 | (DSB-resident: ALU+23%) |
| 40/ip0 | p2 | 229/tile | 162 | 220 | — |
| 40/ip0 | map | 196k/step | 84k | (DSB loop) | — |

Sums reproduce the measured steps (100: 16.1M cyc = 4.92 ms). Readings:
prepass/p2 at 100 are DRAM (known); the 50-map runs at L3 BW (~14.7 B/cyc,
3 streams x 2 MB); and the transform subpasses sit at **the node's ~2.1
vector-uops/cycle global dispatch cap** (zsub@100: 2971 total uops / 1410
cyc = 2.11; the cap TOOLS.md documents as the models' blind spot), well
above the 2-FMA port floor. The four-rounds-old "port 5 vs DRAM?" question
about p1 has the answer NEITHER: **total uops**.

### 3. The frontend-diet experiment — built, measured, REFUTED, reverted

The fetch-floor pattern above (everything over ~4 KB tracks bytes/16, and
the ONE loop that fits the DSB — ysub@40 at 3.4 KB — runs at ALU+23%)
made MITE decode the prime suspect. Fix built: PFA50C/PFA100C stage loops
ROLLED (stage-2's 2x/4x DFT25M copies behind a real k2_ loop with the CRT
store indices from 25-entry byte tables K50IDX/K100IDX; stage 1 wrap-split
into affine ranges; TRNC granule loops rolled per-size via GENROLL) —
bit-identical op order, index math on idle ports 1/6. Code shrank exactly
as designed at 100 (xc_ipp1_100 43.7 -> 11.1 KB; subpass bodies ~3.7 KB,
zsub+ysub fit the DSB together); at 50 gcc PEELED the 2-trip k2_ loop
despite `#pragma GCC unroll 1` (19.6 -> 15.8 KB only).

Held-lease alternation (same core, samples 3, min per arm), outputs
cmp-verified bit-identical both sizes:

- L=50 B=4 m=128: r7 432.5/432.5/437.0/441.3 vs r8 483.6/485.1/478.5/483.6
  — **r8 LOSES 4/4 pairs, +11%.**
- L=100 B=1 m=64: r7 4597/4665/4592/4770 vs r8 4735/4730/4730/4749 —
  r7 3/4, min 4592 vs 4730, **+3%** (r8 notably steadier, 4730±10, but the
  floor is what scores).
- Phase re-measure on the rolled build: zsub/ysub at 100 UNCHANGED within
  the window (1410->1479, 898->952 in a ~10% hot window) despite the 4x
  code cut; at 50 zsub 720 -> 858 (+19%) — the rolled form's extra
  index/staging uops cost real cycles under the uop cap while the fetch
  saving bought nothing.

Conclusion, recorded so nobody re-runs it: **MITE fetch is NOT the binder
even at 17.8 KB bodies; the unrolled r1-r7 codelets already minimize total
uops, and total uops are the currency on this node.** The fetch-floor
correlation in the table was a coincidence of both floors scaling with op
count. Reverted; the only r8 source change is the header documenting this.
(.text of the shipped build byte-identical to the r7 binary via objcopy
cmp — stronger than the r7 normalized-objdump protocol.)

### 4. Closed by arithmetic (no code): load-TRNC-into-stage-1 fusion

The one remaining uop cut visible in the attribution was the Zv staging in
zsub (200 uops/yg at 100). Fusing the load-transpose into stage 1 requires
consuming whole transposed granules in registers; stage-1 GT strides are
25 (z = 4n1 + 25n2), so granule g's four elements feed DFT4s #g, #g-6,
#g-12, #g-18 (mod 25) — a systolic pattern whose live set IS the full T_
staging. No cut exists. With map placement (r3), schedule (r4), map ops
(r5), c-bypass (r5-r7), two-axes (r7), frontend (r8) and staging (r8) all
measured or closed, I consider this engine SATURATED on this host at all
three scored cells.

### Measured on the node (final ship build, one warm window, sd 2.7-4.6%)

40 B=8: 161.3 min / 50 B=4: 421.2 / 100 B=1: 4832 (hot window) — code is
instruction-identical to r7, so the r7 board values stand. Gates, fresh,
all three cells: single 3.582/4.336/4.522e-16 (tol 1e-12); two-step m=2
1.857/2.361/2.721e-15 (tol 3e-14) — the EXACT r3-r7 values; full chains
3.804e-14 (40, anchor 2.612e-14) / 5.028e-14 (50, 2.922e-14) / 4.181e-14
(100, 2.416e-14), tol 1e-10; chains bit-repeatable across processes.
Round end: all 3 gen_pfa_large chain6 keys stripped from
results/wisdom_a80n0.json under flock (entries sub-dict, file left valid,
60 foreign entries untouched).

### What did NOT work / incidents, with the numbers

1. **The rolled-codelet frontend diet** (item 3): +11% at 50 (4/4 pairs),
   +3% at 100 min-of-mins. Killed by the numbers above.
2. **`#pragma GCC unroll 1` does not stop gcc 11.4 from PEELING a 2-trip
   constant loop** (the 50 k2_ loop) — it honored the same pragma on the
   4-trip loop at 100. If you need a tiny-trip loop to stay rolled,
   restructure; don't trust the pragma.
3. uiCA is NOT usable yet (setup2.log shows instructions.xml timeouts);
   llvm-mca ICX FMA ports are wrong (item 1). OSACA untested here.
4. tryout's '$W/c.bin' map-check quoting bug: STILL there (seventh round).

### Borrowed, plainly

- **gen_batchlane gen_r4 (via everyone)**: held-lease same-core
  alternation — again the arbiter that killed the round's candidate.
- **gen_powp gen_r7 / gen_pow2 gen_r7 / gen_batchlane gen_r7**: their
  negative results (3-shear interleaved, constant-per-site routing, lifted
  DFT5 on slot engines) — read first, nothing re-derived.
- **tools/TOOLS.md's ~2.1 uops/cycle cap note (the monitor's)**: the
  attribution confirmed it at loop level; it is the single most predictive
  number for this engine.
- The calibration microbench, phase-counter dev harness
  (build/tryout/gen_pfa_large/{portcal2.c,inst8.c,ab.sh,gates.sh}) and the
  llvm-mca-ICX-port finding are mine — take them.

### What I would do next (ranked)

1. **Nothing on this host's scored cells** — the engine is saturated (item
   4); any further candidate must first show a TOTAL-UOP reduction on
   paper, or it will lose like ipm/ipe/ipf/rolled did.
2. **SPR xarch**: two shuffle ports change the p0/p5 balance and the DSB
   is bigger — the GENROLL/rolled variant (this round's corpse) and ipk1
   are exactly the insurance candidates the per-host race exists for; the
   rolled variant's code is in this record's history if SPR wants it.
3. **uiCA**: drop uops.info's instructions.xml into ext/tools/uiCA/ from
   any machine with net access and rerun setup — the ICL model would have
   settled item 1 without a lease slot.
4. **The 50-map's L3-BW share (29% of the cell)**: the only non-saturated
   term left anywhere; every placement variant is already measured against
   it (r2/r3/r5) — a genuinely new idea would have to cut the 3-stream
   traffic itself, and c is read-once-per-step by contract.

## Round gen_r9

Standings into the round (r8 board): led 40 (159.71 vs next entry 235.3),
50 B=4 416.64 and 100 B=1 4567.72 with the gen_race trunk fractionally
ahead at both (410.90 / 4562.29) — riding this engine's own pick, so the
class effectively holds all three cells. The r9 brief is counter-directed:
of its four avenues, avenue 2 (two-axes fusion) was closed by accounting in
my r7, the r8 attribution declared the engine uop-saturated on this host,
and avenue 1 — BANK THE PICKS — is both the brief's named cheapest win and
a hole my own r7 record admits ("the challenger playoff CAN still install
ipk1 in a noisy window"). This round spent itself on avenue 1, plus a
microbench that arbitrates avenue 4 (port-1 co-issue) for everyone.

**HARNESS NOTE (round-defining): the Ice Lake node was unreachable the
entire session.** The icehold (438854) sat PD "Resources" behind another
user's ~2-day jobs on both axxxl nodes; gen_dense_prime and gen_pfa_small
hit the same wall (their r9 records). Everything below is wallaby (SPR
Gold 6448Y login host) plus bit-identity arguments — and for THIS round's
deliverable that is the harsher test: proving pick determinism on an
unpinned 49-user login host with 2x load swings is strictly harder than on
the node's leased quiet cores. Node re-proof is queued first thing.

### What changed

**1. NOISE-GATED PICK + WISDOM STORAGE (avenue 1, the deliverable).**
tune() now keeps every candidate's per-round trial times (tr[][]) and uses
their relative spread (max-min)/min as the noise measure, with two rules:

- **UPSET RULE**: a trial winner may displace the rank-0 prior only if
  (a) both spreads are tight (<= GEN_TIGHT = 10%), (b) its margin over
  rank 0 exceeds max(the larger spread, GEN_UPSET_MIN = 6%), AND (c) it
  holds >= 6% again on one FRESH long-run alternation (chain_longrun — the
  r7 playoff's evidence shape, now factored into a shared helper) that it
  has not seen. Otherwise the pick reverts to rank 0. The rank order IS
  the banked held-lease evidence of r4-r8; quiet-floor gaps among the
  leading families are 1-3% (coin-flip band, belongs to the prior), while
  the genuine cross-host upsets this pool exists for measured 9-16% (ipk1
  on CLX/SPR per gen_powp, NT under contention per my r2) and clear the
  bar comfortably.
- **STORAGE RULE**: only tight, non-reverted verdicts persist to wisdom. A
  reverted upset is a forced default, not a measurement — installed for
  the current plan, never stored, so the next create() re-races ("re-race,
  never trust, a noisy trial"). Engine picks have been routed through
  gen_race wisdom since r3; this closes the noisy-verdict hole the PMU
  audit named.

All families remain bit-identical, so neither rule can affect correctness,
gates, or cross-process repeatability. Wisdom tag chain6 -> chain7 (race
shape + storage policy changed; no stale verdict replays — the r7
doctrine). GENPFL_VERBOSE now prints per-round times, spreads, and the
storage decision. My two leftover chain6 keys were stripped from
results/wisdom_a80n0.json under flock (round protocol; file left valid, 15
foreign entries untouched).

**2. DETERMINISM PROVEN — the brief's requirement, on the harsher host.**
Harness build/tryout/gen_pfa_large/det5.c (left for everyone): N
consecutive COLD create() cycles under GEN_RACE_NO_WISDOM=1, comparing the
installed pick. On wallaby: **5/5 identical at 40 B=8 (ip0), 50 B=4 (ip1),
100 B=1 (ipp1), and 75 B=1 (ip1, odd-L lean path)**. The negative control
matters more: my FIRST gate design (margin > max(spread, 3%) only — no
floor, no confirmation) was NOT deterministic, flipping
ip0/ipp1/ip0/ip1/ip1 across 5 cycles at 40 B=8. Mechanism: within-create
spread underestimates BETWEEN-create window drift, so each cycle's upset
looked locally decisive. The 6% floor plus the fresh-evidence confirmation
killed it. Lesson recorded: **a noise gate keyed only to in-window spread
cannot deliver determinism; it needs a margin floor calibrated to
between-window drift and a confirmation on evidence the challenger has
not seen.**

**3. Avenue 4 (port-1 co-issue) ARBITRATED — negative on SPR, harness
ready for ICL.** portcal3.c (build/tryout/gen_pfa_large/, portcal2's
TSC-calibrated method) measures 8 indep zmm FMA chains + K indep ymm
side-streams. Wallaby/SPR result, core cycles/iter: ZMM8 baseline 4.00;
8ZMM+K ymm FMA = **(8+K)/2 exactly** (K=2: 5.02, K=4: 6.00, K=8: 8.00,
K=12: 10.00); same for ymm MUL (6.00/8.00) and ymm in-lane SHUF
(6.14/8.00); pure ymm FMA runs 2/cycle (P=8: 4.00 — no third port even
unmixed). **256-bit FP steals 512-bit FMA slots 1:1 when zmm is in
flight** — consistent with p0+p1 fusing into the 512-bit pipe, exactly the
design documented for ICL too. The PMU audit's "port 1 idles, side-work
could co-issue nearly free" hope is dead on SPR and predicted dead on the
node; portcal3 on a leased ICL core settles it in 30 seconds when the
reservation returns. Do not build ymm side-work candidates before that
run.

**4. Everything else UNTOUCHED, verified.** nm -S function-size diff of
the full TU vs impl_8: the ONLY differences are tune (changed) and
chain_longrun (new) — all ~104 generated FFT/chain functions are
size-identical, so the r6 unit-growth codegen-drift trap did not fire.

### Operation count

Unchanged everywhere (278/434/661/968 FMA-port vector ops per line at
40/50/80/100; r6 coverage counts stand). The confirmation adds 2 x PS
chain steps at create() time only when an upset is pending displacement —
zero cost in the common case, ~0.5-2 s worst case, budget untroubled
(cold create measured 0.3-2.4 s here; warm wisdom hit ~1 ms, verified
setup=0.000 s on a second L=100 create).

### Measured on WALLABY (SPR login host — ADVISORY; the node was down, the
### monitor's scoring pass is the ICX measurement)

Gates, shipped source, all EXACTLY the historical values (bit-identical
algebra, tune()-only change): single 3.582/4.336/4.522e-16 at 40/50/100
(tol 1e-12); two-step m=2 1.857/2.361/2.721e-15 (tol 3e-14); full chains
3.804e-14 (40 B=8 m=128, anchor 2.612e-14) / 5.028e-14 (50 B=4 m=128,
2.922e-14) / 4.181e-14 (100 m=64, 2.416e-14), tol 1e-10; chain outputs
bit-repeatable across processes at all three cells; odd-L (75) create +
chain gate pass. Wallaby chain context (not scored): 40 B=8 115.9 / B=1
224.8; 50 B=4 303.4 / B=1 314.3; 100 B=1 2895.0 us/xform — SPR runs this
engine 1.3-1.6x faster than the ICX board numbers, matching the
xarch_spr_r5 advisory.

### What did NOT work / incidents, with the numbers

1. **Spread-only noise gate: refuted by its own determinism test** (5
   cycles: ip0/ipp1/ip0/ip1/ip1 at 40 B=8). Numbers and mechanism in item
   2 above. Fixed with the 6% floor + confirmation; 5/5 after, four cells.
2. First verbose run on wallaby read 90%+ spreads (unpinned churn) and the
   gate correctly reverted + refused storage — then a quiet run minutes
   later read 0.2-8% spreads and stored. Both behaviors are the design.
3. **Node unreachable all session** (icehold 438854 PD behind ~2-day
   foreign jobs; reserve.log full of "sbatch: command not found" from the
   wallaby-side guard — it cannot resubmit without the slurm PATH shim).
   Everything above ships on wallaby evidence + bit-identity; node
   re-proof and portcal3-on-ICL are queued.

### Borrowed, plainly

- **PMU_AUDIT.md (the monitor's)**: avenue 1 verbatim — the noise-gated
  storage requirement and the 5-cycle determinism bar.
- **gen_race (since r3)**: the wisdom substrate. The gate lives
  engine-side because the verdict semantics (rank prior, playoff,
  confirmation) are engine-specific; the storage contract is unchanged.
- **gen_dense_prime gen_r9 (concurrent)**: the "tag wallaby-advisory
  verdicts as pending-ICX" doctrine, applied throughout this section.
- **My own r7 playoff / r8 portcal2**: chain_longrun is the playoff arm
  factored out; portcal3 is portcal2's method pointed at avenue 4.

### What I would do next (ranked)

1. **Node re-proof when the reservation returns**: 5 cold creates per
   scored cell (expect rank-0 picks, tight verdicts stored), then one
   graded-shape tryout per cell to confirm board parity.
2. **portcal3 on a leased ICL core** — settles avenue 4 for every entry.
3. **Nothing on this host's scored cells** without a paper total-uop cut
   (the r8 saturation verdict stands; this round deliberately shipped
   zero arithmetic changes).
4. **XARCH**: the noise gate is precisely what the CLX/SPR wisdom races
   needed — genuine >= 6% upsets (ipk1's regime) still bank per host;
   1-3% coin flips no longer can. Check the next advisory's picks.

## Round gen_r10

Standings into the round (r9 board, scored on a81n2 — a NEW node, same Gold
6326 SKU): led 40 (160.17) and 100 (4521.7 vs gen_powp's ipk1 4699.4), but
50 B=4 read **437.06 vs gen_powp's 417.4 (+4.7%)** — a regression from my
own r8 board 416.64 despite r9 being a tune()-only, bit-identical round.
The r9 scoring window banked my `l50-ip1` while gen_powp's rank rule banked
`l50-ipp1` (their trial us 416.2), so the round opened on one question: is
ipp1 now genuinely faster at 50, with my r9 noise gate's 6% upset floor
structurally unable to discover it (a real 4.7% gap can never displace the
rank-0 prior — by design the RANKS must carry sub-6% truths)? The node
reservation was ALIVE all session (the r9 "queued-busy" NOTICE is stale),
so this became the node-verification round my r9 record queued.

### What was measured (a81n2, leased cores; zero code changes shipped —
### the r10 header note is the only source edit, generated code
### instruction-identical to r7/r8/r9)

**1. The 50-cell anomaly is NOT a pick error; the ranks are right.**
Held-lease alternation, one binary, forced picks, L=50 B=4 m=128, 5 pairs
(ip1 first): 463.9/426.4, 417.5/421.6, 414.6/420.8, 410.0/434.4,
412.5/416.0 — **ip1 wins 4/5 warm pairs, min-of-mins 410.0 vs ipp1's
416.0 (-1.4%)**. Full family sweep against an interleaved ip1 control
(ctrl range 409.4-420.4): ip0 +2%, ip2 tie (414.0-415.0), ipr1 tie,
ipp1 428.2/429.3 (+3-4%), ipp0 421.5/423.3 (+2%), ipm1 476.6/492.6
(+16%), ipk1 482.8/487.6 (+17%), ipe1 521.7/525.9 (+27%), **ipf1
597.2/600.8 (+45%)**. So the banked `l50-ip1.ch` verdict is CORRECT, the
scored 437 was a sustained-slow scoring-slot artifact (its 1.2% run spread
means the slot was uniformly ~5% slow — per-process THP/page/frequency
luck, not reproducible in any lease today; my floors sat at 410 while
same-window MKL held board parity), and gen_powp's 417.4-with-ipp1 was a
normal window on an engine whose ipp1 ties ip1 within noise. NO rank
change; do not widen the noise gate to chase it. **Record correction: ipf1
at 50 degraded from r2's -1% to +45%** — the r5 pair-packed ladder made
the standalone map pass ~2x cheaper while ipf's in-stream stores still pay
map_step_v per store AND gate the store buffer; nobody should trust the r2
number for any host.

**2. 100 B=1 sweep vs ipp1 control (ctrl 4501-4734): ipnt 7380/7396
(+64%!), ip1 +5%, ipk1 4704/4713 (+4%, matching r7's 4/5-to-ipp1), ipp0
tie.** This killed the round's one candidate build idea before it was
built: an "ippnt" hybrid (ipp prepass + NT y-subpass stores) aimed at a
naive 96->80 MB/step DRAM cut. Two independent kills: (a) NT y-stores
lose 56% within their own family on this memory system (worse than r2's
+14% — do not revisit NT at 100 on ICL), and (b) the traffic
re-accounting shows **ipp1 is ALREADY at the 80 MB/step floor**: phase 1
is per-x-plane, so the y-subpass writes exactly the plane lines the
prepass just read (160 KB, L2/L3-hot) — there is no DRAM RFO to delete.
With traffic at floor and zsub/ysub at the ~2.1 uops/cyc dispatch cap
(r8), 100 is saturated from both directions. 40 parity: ip0 floor 159.8
vs board 160.2, ip1 +2%, ipp1 +3% — correct.

**3. Determinism node-proof (r9 queue item 1, the brief's acceptance
test, now on scoring silicon).** det5 on a81n2 under GEN_RACE_NO_WISDOM=1:
**5/5 identical cold picks at all four cells** — 40 B=8 ip0, 50 B=4 ip1,
100 B=1 ipp1, 75 B=1 ip1 (odd-L lean path) — matching the banked scoring-
window verdicts exactly. The r9 wallaby proof now holds where it counts.

**4. portcal3 on ICL (r9 queue item 2 — the avenue-4 arbiter, panel
deliverable).** On a leased a81n2 core: zmm FMA P=8/12/16 -> 4.36/6.00/
8.00 cyc/iter (2/cycle); ymm-only FMA the same (2/cycle, no third port);
**8 zmm + K ymm FMA = (8+K)/2 exactly** (K=2: 5.05, K=4: 6.00, K=8: 8.00,
K=12: 10.00); same for ymm MUL. **256-bit FP steals 512-bit FMA slots 1:1
on the scoring node, identical to SPR: the PMU audit's "port 1 idles,
side-work could co-issue nearly free" is dead on ICL too.** (One nuance:
ymm SHUF at K=4 read 6.08 — the p1-share of shuffle dispatch may ride
along, but FP arithmetic does not.) Nobody should spend a round on ymm
side-work candidates. PMU cross-check on my cells (tools/pmu.sh, /tmp/perf
is staged on a81n2, paranoid=2): port 1 dispatch 0.13/0.27/1.90G vs port
0's 3.9/4.7/6.1G at 40/50/100; LLC misses at 50 negligible (5.1M) = L3-
resident as the audit said; 100 shows 132M LLC misses = the DRAM-bound
prepass+p2.

### Operation count

Unchanged everywhere (278/434/661/968 FMA-port vector ops per line at
40/50/80/100; r6 coverage counts stand). Zero arithmetic changes; nm/objdump
identity holds by construction (comment-only edit).

### Measured on the node (a81n2, leased cores; windows swung +-8% and one
### late lease ran +30% hot by the MKL yardstick — floors quoted)

| case | best this session | same-window MKL | pick (banked) |
|---|---|---|---|
| L=40 B=8 m=128 | **159.8** | 425.7 (2.66x) | ip0.ch |
| L=40 B=1 m=128 | **176.3** | — | ip0 |
| L=50 B=4 m=128 | **410.0** | 965.8 (2.36x, that window's ctrl ~442) | ip1.ch |
| L=50 B=1 m=128 | 465.0 (hot window; r7-class is ~434) | — | ip1 |
| L=100 B=1 m=64 | **4501.2** | 7798.7 (1.73x) | ipp1.ch |

Board parity confirmed at all three scored cells (r9 board: 160.17 /
[437.06 anomaly, true floor ~410-417] / 4521.7). Gates, full manual pass
(tryout's '$W/c.bin' map-check quoting bug is STILL there, eighth round):
single 3.582/4.336/4.522e-16 (tol 1e-12); two-step m=2 1.857/2.361/
2.721e-15 (tol 3e-14); full chains 3.804e-14 (40, anchor 2.612e-14) /
5.028e-14 (50, 2.922e-14) / 4.181e-14 (100, 2.416e-14), tol 1e-10 — all
EXACT r3-r9 values; chain outputs bit-identical across forced picks (cmp).
Setup: warm wisdom 0.001 s on the node (banked verdicts). Wisdom protocol:
the a81n2 chain7 verdicts written by the r9 SCORING window are left BANKED
(adopting gen_powp r9's retirement of the round-end strip for scoring-
window verdicts — banking is the design; r10's scoring warm-hits them
deterministically). All dev runs this round were forced-pick or NO_WISDOM
and stored nothing — verified by dumping the file post-session.

### What did NOT work / what was killed without being built

1. **ippnt (prepass + NT y-stores) at 100**: killed by measurement before
   writing code — ipnt's +64% (and +56% within-family) plus the RFO
   re-accounting above. The 80 MB floor is already reached by ipp1.
2. **Re-ranking ipp first at 50** (the round's opening hypothesis from the
   r9 board): refuted 4/5 pairs + sweep. The scored 437 was environmental.
   If an r10 scored number at 50 again reads ~435+ while a shared-shell
   rival reads ~417 in the same window, the anomaly reproduces ONLY under
   scoring conditions — hand the monitor this record and ask for a
   THP/free-page-state A/B of the scored process; it is not addressable
   from engine code.
3. Nothing else was attempted: no paper total-uop cut exists (map ladder
   is 2+2 NR minimal for the 1.5e-14 contract with 14-bit seeds; staging
   closed by arithmetic in r8; traffic at floor at 100; every pass-
   structure variant freshly measured a loser at 40/50/100).

### Borrowed, plainly

- **gen_powp gen_r9**: the banked-verdict doctrine (retire the round-end
  strip for scoring-window verdicts) — adopted; also their r9 standings
  note that flagged the 50-cell pick divergence worth investigating.
- **gen_batchlane gen_r4 (via everyone)**: held-lease same-core
  alternation with interleaved controls — again the arbiter for every
  verdict this round.
- det5/portcal3 are my own r9 harnesses, now run where they were meant to
  run; sweep10.sh (interleaved-control family sweep) is new this round and
  left in build/tryout/gen_pfa_large/ for anyone.

### What I would do next (ranked)

1. **Nothing on this host's scored cells** — the saturation verdict now
   rests on measurements from BOTH Ice Lake nodes (a80n0 r8, a81n2 r10):
   ranks verified, traffic at floor, dispatch at cap, determinism 5/5.
2. **Class coverage holes** (60/84/90/96/105/108/120/126 via three-factor
   GT or DFT27/DFT32; 112's 0.91x vs MKL) — the only structural work left
   in class if the campaign continues past r10.
3. **XARCH**: ipk1 and the r8 rolled-variant corpse remain the CLX/SPR
   insurance; the noise gate banks genuine >=6% per-host upsets, and
   portcal3 should be re-run on CLX before anyone considers ymm side-work
   there.
4. **For the monitor**: /tmp/perf is staged and working on a81n2
   (paranoid=2); tryout's map-check '$W/c.bin' quoting bug is eight rounds
   old — one sed on the CH= line fixes both it and the repeatability leg
   it blocks.

## Round gen_r11

Standings into the round (r10 board, a81n2): led all three cells — 40
(159.58), 50 B=4 (418.46 vs gen_powp 417.20, a wash), 100 (4529.4 vs
gen_powp 4659.8; 1.73x vs MKL, the board's weakest cell and this round's
all-hands target). The r11 brief made the counter protocol mandatory and
put a bounty on settling the open disagreement: my r7/r8 "the engine is
uop-saturated" vs the audit's "0.82 p0+p5 per cycle is headroom".

### 1. The disagreement is SETTLED with counters (the round's bounty)

tools/pmu.sh on the graded L=100 B=1 m=64 chain, a80n0 leased core,
samples-16-minus-samples-4 delta = 768 pure steady-state steps, per step:

| counter | per step | per cycle |
|---|---|---|
| cycles | 14.93M | — |
| p0 / p5 dispatch | 5.79M / 6.76M | 0.84 combined |
| p1 dispatch | ~0 (2.0G whole-process = setup only) | 0.00 |
| p2_3 / p4_9 | 3.63M / 1.79M | 0.36 |
| TOTAL vector dispatch (p0+p1+p5+p2_3+p4_9) | 17.97M | **1.20** |
| l1d.replacement | 1.81M (=116 MB through L1) | — |

**Both sides were right about different granularities.** Step-averaged
total dispatch is 1.20/cycle — far under the ~2.1 cap, real headroom
(p0+p5 = 0.84 reproduces the audit's 0.82). But the r8 phase attribution
also stands: zsub/ysub run AT the ~2.1 cap. The engine is
**PHASE-saturated, not step-saturated**: the compute passes sit at the
dispatch cap with DRAM near-idle, and the memory passes (prepass 35%, p2
29%) idle the ports. The step-average headroom exists only while the
memory phases run. Corollary that decided everything else this round: 80
MB/step at the measured ~19-20 GB/s single-core streaming ceiling is
~4.0 ms of the 4.53 ms step — **L=100 is ~88% DRAM-BW-bound end to end**,
so overlap can reclaim at most the compute-only residue and nothing can
"prefetch around" the bandwidth itself.

### 2. What shipped: the ipa family (prefetch-ahead), pf id 20

New chain family xc_ipa1 = ipp1 with the NEXT plane's STATE stream
T1-prefetched during the current plane's compute subpasses: every
z-subpass granule load is mirrored by one prefetch of the same address
one plane ahead (GENL^2/4 lines/plane, 4 per zg iteration, in the load
STAGING loop — never inside the codelet). p1body_a/p1pma; plane index
wraps at the volume end (warms p2's first tiles). Values, op order and
outputs are BIT-IDENTICAL to ipp1/ip* (prefetches are hints) — verified
by cmp across forced picks; every historical gate value reproduced
exactly (see gates below). Raced at 40/50/80/100 (pool 21/19/21/21,
NCMAX 21; lean sizes unchanged). Dev knobs, all compiled out by default:
GENPFA_PFCON (c-next mirror), GENPFA_CFLX (c-flush in compute),
GENPFA_HINT, GENPFA_P2POKE, GENPFA_NOPFS.

Evidence at 100, and the rank decision: state-only ipa1 was trial-best in
**5/5 cold interleaved races** (margins 0.8-1.8% over ipp1: 5079.6 vs
5159.7, 4813.1 vs 4874.6, 4730.3 vs 4767.6, 4811.8 vs 4900.4, 5242.2 vs
5303.8 us/vol) and won **4/6 held-lease pairs** (-3.6/-7.9/+2.3/-4.5/
-0.3/+0.7%; min-of-mins 4754.7 vs 4768.5). A consistent sub-6% quiet-floor
winner cannot displace rank 0 through the r9 noise gate BY DESIGN — the
rank must carry it (the r5 doctrine, third application): **ipa1 is rank 0
at 100 only**; ipp1 sits at rank 1 one hysteresis band away for any host
where the prefetches cost more than they hide. Determinism re-proven with
the new pool: 5/5 cold creates pick l100-ipa1 — including under the
round's all-hands node load (trial spreads 22-60%!), where every noisy
upset correctly reverted to rank 0 and refused storage. 50 B=4 still
picks ip1, 40 B=8 picks ip0 (ipa raced, not preferred — at the resident
sizes the prepass barely misses, so there is nothing to hide; expected
and confirmed).

### 3. What did NOT work, with the numbers that killed it (8 variants)

All measured at 100 B=1 m=64 against interleaved/adjacent ipp1 controls:

1. **ipa with BOTH streams (state + c-next in the ysub codelet's LD2):
   +5.3-7.1%, 0/5 pairs.** PMU: p2_3 +0.51M uops/step (exactly the 5000
   prefetches/plane), l1d.replacement UNCHANGED — the damage is L2-level.
2. **c-next mirror alone, inside the codelet: +2.5-3.3%.** 100 prefetch
   uops inside the uop-cap-bound PFAL expansion — the ipm lesson again,
   now with its own number. NEVER add uops inside the codelet passes.
3. **c-next mirror moved to the store staging loop (out of the codelet):
   +1.5-2.5%, 0/3 pairs.** Even in the "free" slot, the second stream
   loses: 320 KB/plane of incoming prefetch evicts the prepass-warmed
   state-x lines from L2 before ysub writes them (RFO where none was),
   and the swpf counters show the round trip: state-only prefetch already
   ADDS 16.5M LLC misses/run (l2_rqsts.swpf_miss +65M, prefetched lines
   evicted and re-fetched). One stream is the L2's limit here.
4. **T2 hint: +3.9-4.7%.** T2 fills L2 on this part, same pollution.
5. **c-flush-in-compute (ipk1's L3-custody play with the CLFLUSHOPT cost
   moved from the BW-critical prepass into the compute staging loops):
   +4-8% with state-pf, +8% flush-only.** The flush cost is intrinsic to
   the uop/snoop machinery, not to where it is paid. Combined with
   r5/r7/r10 this CLOSES the c-bypass axis on ICL: NTA prefetch (+12.7%,
   r5), flush-in-prepass (+4%, r7/r10 pairs), flush-in-compute (+4-8%,
   here). The 16 MB/step c stream is untouchable on this host.
6. **p2 poke distance 4 on the ipp schedule: +8-24%.** Way past the
   sweet spot; drops/pollution. Poke 2: wash (+0.4/-0.3/+1.2%), matching
   r3's ip2 result on the ip schedule. Distance 1 stays.
7. **state+poke2 combined: wild (+/-8% swings), no signal over state-only.**
8. Windows during the all-hands rounds swing far worse than r4-r10
   (ctrl 4661-5147 in one lease; late session 2x with every implementer
   hammering the same node): 2-sample A/Bs are worthless — interleaved
   race minima + >=5 pairs with samples>=3 were the only usable evidence
   shapes this session.

### 4. Operation count

FFT/map arithmetic unchanged everywhere (278/434/661/968 FMA-port vector
ops per line at 40/50/80/100; r6 coverage counts stand). ipa1 adds
GENL^2/4 prefetch uops per plane (250k/step at 100 — measured +0.25M
p2_3 dispatch/step, l1d.replacement unchanged at 2.02M/step). Codegen
identity: xc_ipp1_100, xc_ip0_40, xc_ip1_50, x_pf1_100 verified
INSTRUCTION-IDENTICAL to the r10 object (nm -S + objdump normalized with
full hex masking — raw jump-target hex sneaks past the r7 sed recipe;
mask every >=4-digit hex token). Only collateral: the lean p2_65
constprop clone split (unscored size, bit-identical algebra, accepted).

### Measured on the node (a80n0, leased cores; early-session quiet
### windows for the board-shape numbers, late session unusable)

| case | quiet best this session | same-window MKL | pick |
|---|---|---|---|
| L=40 B=8 m=128 | 166.3 (window ~3% hot) | 411.5 (2.47x) | ip0.ch |
| L=40 B=1 m=128 | 185.1 | 446.1 | ip0 |
| L=50 B=4 m=128 | 451.2 raw, ~421 window-adj (MKL ran 7% hot) | 1035.0 (2.29x) | ip1.ch |
| L=50 B=1 m=128 | 419.1 | 931.4 | ip1 |
| L=100 B=1 m=64 | ipa1 4730.3 (race-round floor; pairs min 4754.7) vs ipp1 4767.6-class | 7729.9-7887.1 | **ipa1.ch (new)** |

Gates, final binary with the ipa1 pick installed, all EXACT historical
values (bit-identity through the new family + rank change): single
3.578-4.522e-16 (tol 1e-12); two-step m=2 1.857/2.361/2.721e-15 at
40/50/100 and 3.010e-15 at 75 (tol 3e-14); full chains 3.804e-14 (40 B=8,
anchor 2.612e-14) / 5.028e-14 (50 B=4, 2.922e-14) / 4.181e-14 (100,
2.416e-14) / 2.377e-14 (75 m=8), tol 1e-10; chain outputs bit-identical
across processes and across forced picks (cmp); L=80 exec 4.078e-16.
Setup: cold 0.25-5.1 s, warm wisdom ~1 ms. Round end: all 7 gen_pfa_large
keys (6 dev-window chain7 + 1 stale chain6) stripped from
results/wisdom_a80n0.json under flock, 11 foreign entries untouched; the
pool signature changed anyway, so no stale verdict can replay. NOTE: the
a80n0 icehold lapsed mid-session (job 438856 hit its 10 h limit);
./reserve.sh re-claimed a80n0 as job 438881 with the slurm PATH shim
(/opt/software/slurm-19.05.8.1-cuda-11.8/bin) exported — reserve.sh
--status still needs it on wallaby, ninth round of the tryout '$W/c.bin'
map-check quoting bug, run check.py by hand.

### Borrowed, plainly

- **The monitor's r11 brief / PMU_AUDIT**: the total-dispatch-vs-cap
  framing that this round measured, and the mandatory before/after
  counter protocol (followed: baseline and after tables above).
- **gen_batchlane gen_r4 (via everyone)**: held-lease alternation with
  interleaved controls — under this round's node load it was the ONLY
  usable timing method.
- **gen_powp r6 / my r5-r10 ipk1 line**: the c-bypass hypothesis whose
  third and (on ICL) final refutation is item 3.5 above.
- **My own r8 calibration + r4/r5 rank doctrine**: reused verbatim.
- The ipa prefetch-ahead family, the phase-vs-step saturation
  reconciliation, and the "one incoming stream is the L2's limit"
  boundary are new here — take them.

### What I would do next (ranked)

1. **XARCH**: ipa1 vs ipp1 is exactly the kind of per-host coin the
   CLX/SPR races exist for (CLX's 1 MB L2 makes the prefetch pollution
   worse; SPR's bigger L2 makes it cheaper). ipk1 remains rank 3
   insurance there. Check the advisory picks before touching ranks.
2. **L=100 is now closed from three directions on this host**: uops
   (r8), traffic (r10), and overlap (this round — the residue is the
   ~12% non-DRAM window and one stream of it is now banked). The only
   lever that could move the cell materially is more DRAM bandwidth per
   core (THP/page-policy/layout — gen_layout's lane, brief item 5) or
   less traffic per step, and c is read-once-per-step by contract.
3. **Class coverage holes** (60/84/90/96/105/108/120/126 three-factor GT;
   112's 0.91x vs MKL) — unchanged from r10, still the only structural
   class work left.
4. **Protocol note for everyone**: normalized-objdump must mask BARE hex
   jump targets (any >=4-digit hex token), not just <sym+off> — the r7
   recipe cries wolf on any function whose .text moved.

## Round gen_r12

Standings into the round (r11 board): 40 led (160.03-class), 50 B=4 a wash
with gen_powp, and 100 LOST — gen_batchlane's within-volume SoA took the
cell at 4072.3 (trunk-routed) vs this entry's 4554.7, with gen_powp (4465.2,
THP re-home) also ahead. The r11 cross-entry record changed the ground truth
under this entry's feet three ways: (1) gen_dense_prime's ubcap microbench
showed the "~2.1 uops/cycle cap" does not exist (the node sustains 3.0; the
real wall is a ~1.12/cyc POOLED 512-bit L1 access ceiling) — my r8
saturation closure "was computed in the wrong currency" (their words, aimed
at this entry); (2) gen_pfa_small measured MY engine's demand LLC misses at
~19-28 MB/step and called the cell cache-latency-bound, contradicting my
r11 "88% DRAM-BW-bound" corollary; (3) gen_pow2 shipped a one-sweep fused
chain step at L=128 (-14%, -61% demand DRAM reads) with a transfer note
naming this cell: "the 100 = 4x25 axis split gives tiles of 4 or 25 planes
and the same label algebra applies." This round spent itself on exactly
those two adoptions — one shipped, one refuted with counters.

### What changed

**1. THP RE-HOME (ADOPTED from gen_layout gen_r11's zero-copy recipe via
gen_powp gen_r11, who measured -2.0% 5/5 pairs on the sibling shell).** At
vbytes >= 8 MiB per volume (100 graded; lean 80/88/91/99/104/112/117; 40/50
OFF — gen_powp's measured threshold, STLB covers 2 MB volumes) create()
allocates a prefaulted gl_map_huge arena (gen_layout lib, now #included)
holding a state volume S and a staged-c volume CV. The deferred-map (dm==1)
chain runs steps 1..m-1 with state in S — step 1 reads x0 directly, the
trailing map writes the caller's out directly, so the re-home costs ZERO
extra copies; c is memcpy'd once per volume-chain into CV (~0.35%/call,
gen_powp's cv-wins-3/3 verdict re-confirmed by riding it). The scoring host
runs THP=madvise on kernel 5.15 (gen_layout r11's smaps finding): driver
posix_memalign buffers are 4K-backed at any size, only MADV_HUGEPAGE-at-
fault arenas are huge. Race fidelity (gen_powp's move): at re-home sizes
the race arena's tout/tcf ARE S/CV, so trials, playoff and confirmation run
in the graded cache+TLB regime and trial 4K phases are gl_arena_take-
deterministic. Wisdom tag chain7 -> chain8 (race regime changed at every
re-home size). GENPFL_NOREHOME is the A/B control.
Node evidence: differential PMU (samples-10 minus samples-2, 512 steps)
dtlb_load_misses.walk_completed 6910 -> 99 per step (-98.6%; ports and
l1d.replacement flat — bit-identical work); held-lease alternating pairs
8/11 across two sessions, quiet-window session: 5/6 pairs, min-of-mins
4416.7 vs 4490.5 (-1.6%). Matches gen_powp's -2.0% and gen_layout's
"small, real, ~free, composes with everything."

**2. ONE-SWEEP FAMILY ipw1 (pf 21, dm 2) — BUILT, MEASURED, REFUTED on both
hosts; kept raced at rank last.** The adoption gen_pow2's transfer note
asked for: split the x transform's two GT stages across the chain-step
seam. State between steps holds x-stage-1 (DFT4) custody t-planes, slot
25*k2 + n1 = t[n1][k2] (class-major); one body sweep = head DFT25 per
CONTIGUOUS 25-plane class (4 MB, 25 read streams vs p2's 100) into a class
buffer CB carved from the mid volume M, then map(+c) prepass + z + y per
output plane written IN PLACE into the class block the head just consumed
(slot 25*k2 + (19k)%25; 19 = 4^-1 mod 25 — the CRT makes the quartet's four
slots exactly {n1, 25+n1, 50+n1, 75+n1}), with the next step's DFT4 fused
into class 3's plane loop as each quartet completes. In-place custody,
zero ping-pong (gen_pow2's digit-map rule); prologue = z+y+DFT4 of x0
quartet-fused; epilogue = head + map straight to the caller's out. Paper
DRAM: 48-72 MB/step vs the two-sweep 80. Same FMA count, same per-element
vector-access count, and every per-element expression is PFA100C/p1body/
map_vec verbatim — outputs BIT-IDENTICAL to ip*/ipp/ipa (cmp-verified),
every historical gate value reproduced exactly.

### What did NOT work, with the numbers that killed it

1. **ipw1 at 100 on the node: +16-25%.** Held-lease pairs (ipw first):
   6925.9/4468.2, 5932.3/4734.0, 5871.9/6127.8, 6893.6/5362.6,
   6879.5/5457.8 — 4/5 to ipa1; three cold interleaved races 6879/6678/5155
   vs ipa1 5460/5491/4410 (the quiet race: ipw 5155.3 at sd 0.2% — uniformly
   slow, not noise). Wallaby/SPR agrees: 3570 vs ipa1 3033 (+18%).
2. **The mechanism, from the differential counters (and this is the round's
   real finding):** under ipa1 the graded step's demand LLC-load-misses are
   **5.9K lines/step = ~0.4 MB** — the 80 MB/step DRAM stream is ~fully
   hardware-prefetch-covered UNDER the passes. There is no demand-DRAM cost
   for a fusion to delete: L=100 on this engine is gen_pow2's L=64 regime
   (their own +2% fusion loss, "traffic nobody was paying for"), NOT their
   L=128 regime (42M demand misses/step). Meanwhile the sweep ADDS one
   volume-level pass (the tail DFT4 re-read/re-write that p2's in-register
   T_ staging never paid) plus the CB L3 round trip: l1d.replacement 2.04M
   -> 2.51M lines/step (+23%), LLC-loads 7.1K -> 55.0K. **My r11 "L=100 is
   ~88% DRAM-BW-bound" corollary is hereby refuted by my own engine's
   counters**; the r11 cross-entry disagreement settles on gen_pfa_small's
   side — the cell is prefetch-fed and latency/overlap-bound. The
   effective-bandwidth ledger still closes (80 MB / 4.43 ms = 18 GB/s,
   gen_batchlane's 4059 = 19.7 GB/s on the same compulsory traffic): the
   winner runs the SAME traffic closer to the ceiling; nobody at this size
   wins by moving less of it.
   Boundary for everyone, stated once: **check LLC-load-misses (demand)
   before building any traffic-structure candidate — l1d.replacement and
   paper MB/step both lie about what the machine is actually waiting for**
   (gen_pow2 said this in r11; this entry now has the cleanest
   demonstration: 0.4 MB/step demand under an 80 MB/step stream).
3. ipw1 kept raced at rank LAST (the ipm/ipf precedent: measured losers
   stay as xarch insurance — CLX's inclusive 24.75 MB L3 probably makes it
   worse, but the race, not a guess, should say so). It cannot displace
   ipa1: sub-rank + the r9 noise gate. Anyone tempted to "fix" it by fusing
   the map into the head's stores: that deletes 2 of the ~4 added
   volume-level accesses, not the tail pass — the sign does not flip.
4. Re-home cycles DIFFERENTIAL read +0.7M/step in one churning window while
   the pairs said -1.6% — gen_rader's r11 caveat re-confirmed: cycles
   differentials need timing runs; only port/l1d/TLB counters difference
   cleanly.

### Operation count

FFT/map arithmetic unchanged everywhere (278/434/661/968 FMA-port vector
ops per line at 40/50/80/100; r6 coverage counts stand). ipw1 has the same
FMA and vector-access counts per element as ipp1 with the T_ register
staging exchanged for the CB round trip (the measured +0.47M line
fills/step above). Re-home moves no arithmetic; +one 16 MB memcpy per
volume-chain (CV). Codegen identity: xc_ipa1_100, xc_ipp1_100, xc_ip0_40,
xc_ip1_50, x_pf1_100 nm -S SAME size as the r11 object; collateral confined
to unscored lean constprop clones (p2_56/p2_65 split, the accepted r11
pattern).

### Measured on the node (a80n0, leased cores; all-hands windows early,
### one quiet window late — floors quoted with same-window MKL)

| case | ship (quiet window) | same-window MKL | pick |
|---|---|---|---|
| L=40 B=8 m=128 | **159.7** (sd 0.05%) | 404.3 (2.53x) | ip0.ch |
| L=50 B=4 m=128 | **410.9** (sd 0.08%) | 944.6 (2.30x) | ip1.ch |
| L=100 B=1 m=64 | **4428.8** min (median hit a churn burst) | 8463.5 (**1.91x — this cell's best ratio yet for this entry**) | ipa1.ch + re-home |

Gates, ship build, all EXACT historical values (bit-identical algebra
through both changes): single 3.582/4.336/4.522e-16 at 40/50/100 (tol
1e-12); two-step m=2 at 100: 2.721e-15 (tol 3e-14); full chains 3.804e-14
(40 B=8 m=128) / 5.028e-14 (50 B=4) / 4.181e-14 (100 m=64) / 2.377e-14 (75
m=8) / 5.938e-15 (117 m=6) / 4.413e-15 (80 B=8 m=4) / 3.539e-15 (112 m=4);
100 B=2 m=4 off-case 5.223e-15. Chain outputs bit-identical across forced
picks (ipw vs ipa1 vs ipp1) and across rehome/norehome. Determinism with
the new pool: 5/5 cold creates at 100 B=1 install l100-ipa1 (bitwise-
identical outputs) with every noisy upset correctly reverted at 10-32%
spreads and nothing stored. Setup: cold 1.4-5.1 s (60 s budget), warm
wisdom ~1 ms. Round end: all gen_pfa_large keys stripped from
results/wisdom_a80n0.json under flock (1 dev-window chain8 + 3 stale
chain7; 105 foreign entries untouched) — dev keys never bank, the
monitor's scoring window writes its own chain8 verdicts.

### Borrowed, plainly

- **gen_layout gen_r11**: the whole zero-copy THP re-home recipe +
  gl_map_huge/gl_arena/gl_thp_bytes (their library, now #included —
  GEN_LAYOUT_LIB_ONLY), and the madvise/kernel-5.15 finding.
- **gen_powp gen_r11**: the 8 MiB gate, the cv verdict, arena-before-
  wisdom, and trial-regime fidelity (race arena = plan arena), taken
  nearly verbatim.
- **gen_pow2 gen_r11**: the one-sweep design (digit-map custody, "a fused
  operand's layout must match the fused consumer's walk order") AND the
  L=64-vs-L=128 regime boundary + the LLC-loads/LLC-load-misses
  discriminator that explained my negative result. Both directions of
  their transfer note were load-bearing.
- **gen_pfa_small gen_r11**: the differential-PMU method (panel standard)
  and the cache-latency-bound reading of this cell, which my counters now
  confirm; **gen_dense_prime gen_r11**: the corrected currency (512-bit
  access ceiling) that reopened the r8 closure enough to make this round's
  experiment worth running; **gen_rader gen_r11**: the cycles-differential
  caveat, re-confirmed; **gen_batchlane gen_r4 (via everyone)**: held-lease
  alternation, the arbiter for every verdict again.

### What I would do next (ranked)

1. **The cell now belongs to overlap quality, not traffic quantity**: my
   engine runs 80 MB/step at ~18 GB/s effective, gen_batchlane's winner at
   ~19.7. The residue vs them (~8%) is the compute-phase DRAM idle my ipa1
   only nibbled (~1%) and the r11 kill list bounds ("one incoming stream
   is the L2's limit"). Honest options left on this engine: convert
   compute-phase staging round trips to the untouched 256-bit access class
   (gen_dense_prime's ask — a scheduling experiment on zsub/ysub worth one
   mca+node window), or concede the cell to the within-volume-SoA shape
   and make sure the trunk routes it (it does since gen_race r11).
2. **XARCH**: re-home is size-gated, not host-gated — CLX/SPR verify via
   the per-host race as usual; ipw1's rank-last slot exists exactly so the
   CLX race can print its number once.
3. **Class coverage holes** (60/84/90/96/105/108/120/126 three-factor GT;
   112's 0.91x) — unchanged from r10/r11, still the only structural class
   work left.
4. **For whoever builds at large L next**: the two boundaries this round
   nailed down — (a) demand LLC misses, not paper MB or l1d.replacement,
   decide whether a traffic restructure can pay; (b) a pass "deleted" by
   splitting another pass in two is a net ADD; only fusions that keep
   pass count flat (gen_pow2's, where stage-2 and stage-1 share one tile
   visit) can win, and on this engine the tile that closes under both GT
   stages is the whole volume.

## Round gen_r13

Standings into the round (r12 board): led 40 (159.25, 2.55x MKL) and 50 B=4
(413.4, 2.29x); 100 B=1 at 4562.5 scored (1.12x vs the cell winner —
gen_batchlane's within-volume SoA holds the cell, as conceded in r12). The
r13 brief is a QUICK round aimed at the benchFFT-exposed B=1 small-L cells
(10:1, 12:1) — gen_pfa_small/gen_batchlane's lane, not this class
(supports() declines 10/12 by construction; verified no exposure). Duty
here: protect 40/50/100, and spend the one queued lever from my own r12
next-list — gen_dense_prime r11's "convert staging round trips to the
256-bit access class" re-accounting — before declaring the engine closed.

### What changed

**CODE UNCHANGED** (r10 precedent: the impl header note is the only edit;
all generated FFT/chain code is instruction-identical to r12 — a final
node rebuild from the edited source reproduces every gate digit).

**The round's work is a CALIBRATION CLOSURE: the 256-bit access-class
lever at L=100 is measured DEAD, both directions, without engine
surgery.** gen_dense_prime's r11 ubcap measured the load side (zmm loads
ceiling ~1.124/cyc, ymm loads 2.0/cyc, zmm ld+st POOL at ~1.17/cyc) and
suggested my zsub/ysub staging could escape the pooled 512-bit ceiling by
converting round trips to 256-bit accesses ("a DIFFERENT budget").  Their
own note flagged it unmeasured on the store side.  Built the missing
kernels (r13dev/ubst.c + ubst_ldzy.c, rdtscp min-of-7 on a leased a80n0
core; ldz reproduces ubcap's pmu-measured 1.125/cyc exactly, so TSC ==
core cycles at this license level):

| kernel | shape | result |
|---|---|---|
| ldz | 64 dead zmm ld | 1.125/cyc (ubcap control, reproduced) |
| stz | 64 zmm st | 1.00/cyc = 64 B/cyc |
| sty | 128 ymm st, same bytes | 2.00/cyc = 64 B/cyc — no byte gain |
| stx | 64 x (ymm-lo st + vextractf64x4-hi-to-mem) | 128 cyc vs sty's 64: **extract-to-mem is NOT shuffle-free; the split pair runs at HALF the plain-ymm rate** |
| ldzstz | 32 zmm ld + 32 zmm st | 1.142/cyc pooled = **73 B/cyc** (gdp's 1.17 reproduced) |
| ldzsty | 32 zmm ld + 64 ymm st | 64.0 cyc = **64 B/cyc — WORSE than all-512b** |
| ldzy | 32 zmm ld + 64 ymm ld | 66.75 cyc vs 60 predicted by a weighted single pool, ~32 by separate budgets |

**The model that fits all eight kernels plus gdp's r11 table: ICL-SP L1
runs ONE ~2-slot/cycle access budget, loads and stores, where a 512-bit
access costs ~1.75 slots and a 256-bit access 1 slot** (ld-only 2/1.78 =
1.125; pooled ld+st 2/1.75 = 1.142; ymm-only 2.0; ldzsty (32x1.75+64)/64
= 1.875 ≈ 2).  There is NO separate 256-bit budget to escape into — for
loads or stores — and 512-bit accesses are the most byte-efficient
currency in the shared pool (72-73 B/cyc vs 64).  Consequences:
1. My r8 minimum-uop staging closure was computed in a currency that
   turns out to give the same answer in gdp's corrected currency: the
   all-512b schedule already maximizes staging bytes/cycle.  zsub/ysub
   have no access-class lever.  **L=100 on this engine is now closed from
   four directions: uops (r8), traffic (r10/r12), overlap (r11), and
   access class (here).**
2. For the panel: nobody should build ymm-staging candidates on this
   host, load-side OR store-side — and vextractf64x4-to-memory
   specifically is a trap (2x cost vs plain ymm stores; whoever needs a
   zmm split must pay a register extract + two ymm stores, and per item
   above it still loses bytes/cycle).  gdp's feed-ratio doctrine
   (<0.56 zmm ld/FMA for FMA saturation) is unaffected.

### Operation count

Unchanged everywhere (278/434/661/968 FMA-port vector ops per line at
40/50/80/100; r6 coverage counts stand). No new families; pool, ranks and
wisdom tags exactly r12's.

### Measured on the node (a80n0, leased cores; ordinary round windows)

| case | this session | same-window MKL | pick |
|---|---|---|---|
| L=40 B=8 m=128 | **159.77** (r12 ship 159.7) | 404.1 (2.53x) | ip0.ch (warm) |
| L=50 B=4 m=128 | **419.4** | 988.5 (2.36x; window ~4% hot by the MKL yardstick) | ip1.ch (warm) |
| L=100 B=1 m=64 | **4430.1** (r12 ship 4428.8); rebuild-from-edited-source window 4453.7 | 7755.9 (1.75x) | ipa1.ch + re-home (warm) |

Gates, all EXACT historical digits (bit-identity, code untouched):
singles 3.582/4.336/4.522e-16 (tol 1e-12); two-step m=2 1.857/2.361/
2.721e-15 (tol 3e-14); full chains 3.804e-14 (40 B=8) / 5.028e-14 (50
B=4) / 4.181e-14 (100), tol 1e-10; chain outputs bit-identical across
independent node runs at all three sizes.  Setup: warm wisdom 1-15 ms
everywhere (all runs hit the monitor's r12 chain8 verdicts).  Wisdom
protocol: NO keys added or changed this round (diffed against a
round-start snapshot, r13dev/wisdom_snapshot_r13start.json) — nothing to
strip; the monitor's 9 scoring-window keys left untouched.

### What did NOT work, with the numbers that killed it

The round's whole experiment IS the negative result, banked above at
microbench cost instead of engine-surgery cost: the planned "ipx"
split-store family (ymm-half stores for the zsub pl stores, ysub mx
stores, p2 ST3) was designed, then killed BEFORE implementation by
ldzsty = 64 B/cyc vs ldzstz = 73 B/cyc and stx = 2x stz.  Estimated
saved: a full p1body_s/p1body_as/p2s clone set plus race plumbing that
could only have lost ~12% on the staging phases.  Measure the mechanism
before building the candidate — this entry's r3 lesson, finally applied
in the cheap order.

### Borrowed, plainly

- **gen_dense_prime gen_r11**: ubcap.c (kit reused as the control and
  extended), the corrected-currency framing, and the explicit ask that
  this entry re-run the zsub/ysub accounting.  Their speculative
  "256-bit is a different budget" clause is hereby refuted with their
  own method — the productive kind of cross-entry disagreement.
- **The r12 boundary doctrine** (demand-LLC-misses before traffic
  candidates) kept this round from re-opening any traffic idea.

### What I would do next (ranked)

1. **The only lever left at 100 for THIS entry is structural adoption:
   within-volume pencil lanes (gen_batchlane's r11 winner, named again in
   the r13 brief as known-good material).**  A full SoA engine rewrite
   was out of scope for a quick round; the trunk routes gen_batchlane at
   100 (gen_race r11), so the library cell is covered either way.  If a
   future round makes large-L individual cells decisive again, that
   rewrite is the move — budget a full round for it.
2. **XARCH**: the weighted-access-budget model is ICL-SKU-specific;
   CLX/SPR may genuinely have 2/cyc 512-bit loads (SPR does on paper).
   The per-host race already covers this; anyone tempted to generalize
   this round's "no ymm staging" rule off-host should re-run ubst there
   first (it is in r13dev/, argv-selectable, 30 s per host).
3. **Class coverage holes** (60/84/90/96/105/108/120/126 three-factor GT;
   112's 0.91x) — unchanged from r10-r12, still the only structural
   class work left.
4. Harness: tryout's '$W/c.bin' map-check quoting bug is now ELEVEN
   rounds old (manual check.py battery in this round's session log runs
   all three gates + repeatability in one ssh; steal it).
