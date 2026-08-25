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
