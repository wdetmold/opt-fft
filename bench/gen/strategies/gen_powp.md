# gen_powp — strategy record

Class: p^k axes, Cooley–Tukey within prime powers, general exact twiddles — the
campaign's "center of gravity" per the brief. Owned acceptance sizes: 25 (=5²),
27 (=3³), 50 (=2·5²), 100 (=2²·5²). Graded cases: 25:16:256, 27:16:200,
50:4:128, 100:1:64. Note gen_pfa_large also covers 50 and 100 (their PFA
25x2 / 25x4 needs the same DFT25); 25 and 27 are this entry's alone.

## Round gen_r1

### Where this entry started
The round-0 stub was the dense O(L^4) row-column floor. Everything below is new
this round.

### What was built

One C file, a per-size self-include template, four line codelets:

1. **Engine shell adopted WHOLESALE from gen_pfa_large's gen_r1 entry** (who in
   turn credit L45_pfa panel_r11, L45_mixedradix, L23_rader): row-column 3D
   DFT, lanes = 4 interleaved complex per zmm on a spectator axis. Phase 1 per
   x-plane: z-transform with 4x4 complex-granule register transposes (TRNC)
   into plane scratch at an odd-cache-line row pitch (GPP = 28/28/52/108
   complex), then y-transform (lanes = 4 kz, contiguous) out of the scratch.
   Phase 2: x-transform tiled over the flat (y,z) index. Heap plane scratch,
   opaque-base asm barrier in the y-subloop, create()-time gate against an
   independent scalar O(L²)-per-line reference (first AND last arena volume,
   1e-13 rel L2), interleaved min-of-rounds race with 3% simplest-first
   hysteresis, dense fallback of last resort. Their DFT5 (FFTW n1_5 FMA form),
   DFT4, DFT25 (5x5 CT), PFA50C, PFA100C, and NR map ladder are verbatim.

2. **NEW: the DFT27 = 3x9 Cooley–Tukey line codelet.** Stage A: 9 x DFT3 +
   16 nontrivial W27 twiddles into U[9·k1+n2]; stage B: 3 x DFT9, where DFT9
   is itself a 3x3 CT with 4 nontrivial W9 twiddles (keeps live ranges at ≤15
   vectors — no 27-body monolith). DFT3 is 6 FMA-port ops + 1 swap. Line cost:
   218 FMA-port ops + 55 swaps. All twiddles are compile-time literals from
   long-double cosl/sinl (~19 digits), indexed by compile-time products so
   every access folds after unrolling. Per-volume op count: 3·729·218 = 476,766
   vector FMA-port ops (in lanes of 4).

3. **NEW: odd prime-power geometry support in the engine.** 25 and 27 break two
   of the template's divisibility assumptions gen_pfa_large never faced:
   L % 4 != 0 in phase 1 (their overlap-group trick generalizes: the last
   y/z-group re-computes 4−L%4 lanes, stores idempotent) and — the real
   novelty — **L² % 4 = 1 in phase 2**, where an overlapping tail tile is NOT
   idempotent in place (its inputs coincide with the previous tile's outputs).
   Fix: stash the tail tile's GENL input vectors in registers/stack BEFORE any
   tile runs; the tail recomputes from the stash and its stores repeat the
   overlap columns' values exactly. Costs GENL vector loads per plane-pass —
   noise. This is what lets one template serve 25/27/50/100.

4. **Owned chain (fft3d_chain, weak symbol in the driver), THREE step families
   raced at create():**
   - `ip*` (BORROWED from gen_pfa_large's mid-round update, measured by them
     first): everything in place — p1 in place per plane, p2 in place (with
     the stash tail at 25/27), then a SEQUENTIAL vectorized map pass. The
     state volume is the only volume-sized object touched besides c.
   - `ipf` (NEW): in place AND the map fused into phase 2's stores — no mid
     volume, no separate map pass. Built on the reasoning that at 25/27 the
     volume is L2-resident (250/315 KB vs 1.25 MB L2) so the separate map
     pass is pure extra traffic, while gen_pfa_large's miss-stream argument
     against fusing only bites when the volume streams (L=100).
   - `f*`: map fused into phase 2 routed through the padded mid volume M
     (+64 B/plane, odd line count — their measured L=100 4K-alias fix).
   Map is rsqrt14/rcp14 + two Newton steps each (~1e-16 rel; zmm
   vsqrtpd/vdivpd are unpipelined) — the campaign-standard ladder. The picked
   chain step is gated at create() against execute + the driver's scalar map;
   any mismatch falls back to the always-correct execute+scalar path.
   The 25³/27³ volumes are not multiples of 8 doubles, so map_span has an
   exact scalar tail (1–3 complex).

5. **Per-size pre-RA scheduling**: `optimize("schedule-insns","sched-pressure")`
   as a function attribute on the L=25 family only (gen_batchlane's SCHED15
   trick; the 5x5 CT body holds 25+ live vectors and is pressure-bound).

### Measured on the node (a80n0 Ice Lake, pinned core, graded chain, min over samples)

| case | this entry (chain µs/xform) | MKL 2022 same case | ratio |
|---|---|---|---|
| L=25 B=16 m=256 | **50.0** (typ 50–52) | 122.6 | **2.45x** |
| L=27 B=16 m=200 | **66.1** (typ 66–68) | 146.0 | **2.21x** |
| L=50 B=4 m=128  | **478** (typ 478–497) | 966.7 | **2.02x** |
| L=100 B=1 m=64  | **5000** (typ 5000–5500) | 8037.9 | **1.61x** |

B=1 chains (not graded except 100): L=25: 48.7 vs MKL 133.1 (2.73x);
L=27: 68.3 vs 165.8 (2.43x); L=50: 615 vs 940 (1.53x).
Run-to-run wobble between tryout invocations is up to ±8% at L=100 (other
implementers on neighbouring cores); within a run sd < 1% on a quiet core.

Wallaby (dev host, same source): 38.8 / 47.8 / 343 / 3296 µs at the graded
batches; B=1 68.7 / 90.4 / 503 / 3296. Note wallaby INVERTS several picks
(f* wins 25/27 there too, but ipf wins 50 and ip1 wins 100) — the per-host
race is doing real work; do not hardcode a family.

Race picks on the node: 25: f0/fr (59.4 µs/vol vs ipf 63.5, ip 65.9);
27: fr/f0 (65.4/66.1 vs ipf 69.7, ip 73); 50: ipf; 100: ip1
(race: ip1 ~3637–6017 vs f0 ~4518–10235 across hosts).

Gates, all sizes, node: single call rel L2 = 3.6–4.5e-16 (tol 1e-12);
two-step 1.5–2.7e-15 (tol 3e-14, >10x margin); full graded chains 3.1–5.0e-14
at 1.2–1.7x the honest anchor (tol 1e-10); bit-repeatable across runs.
Setup (create incl. gate+race of 6 candidates): 0.07–4.6 s (60 s budget).

### What did NOT work / went wrong first (with the number)

* **Shipping only the fused-through-M chain family** (my first cut, copied
  from the gen_pfa_large source as it stood at round start): L=100 chain was
  **12,729 µs vs their concurrent 5,030** — I was 2.5x slow on a shared size
  because their file had moved on mid-round to the in-place ip* family. Their
  measured mechanism (recorded in their header): fusing the map into the
  100-stream x-pass doubles the miss-stream count to 100 reads + 100 RFO
  writes with one fresh line each per tile; in-place + a sequential map pass
  keeps the state the only volume-sized object and won 2x. Adopting ip* took
  L=100 from 12,729 → ~5,000 and L=50 from 789 → 478. Lesson: re-read your
  seed entry's CURRENT source before closing a round; this panel iterates
  concurrently.
* **Global `-fschedule-insns -fsched-pressure`**: L=25 52.2→49.6 (−5%) but
  L=100 5061→7472 (**+48%**) and L=27 +2%. Same shape as gen_batchlane's
  measurement (−13% at 15, +17% at 10/12). Hence the attribute on the
  25-family only; the −5% at 25 is inside the ±5% node noise and unproven —
  kept because it is provably inert at the other three sizes.
* My `ipf` variant did NOT win 25/27 on the node as designed (f* beat it by
  ~7%: the M round trip at 25/27 is L2-resident and nearly free, and ipf's
  in-place p2 mixes loads and RFO stores on the same lines where f* streams
  them apart). It DID win L=50 on wallaby and is within noise of ip* at 50 on
  the node. Kept in the pool: 6 candidates cost ~0.5 s of create() and the
  race chooses per host.
* tryout.sh still has the `$W`-before-assignment bug (workaround: export
  W=<gen>/build/tryout/<name> first) and its remote map-check/MKL legs still
  don't run (peers documented it; I ran check.py and MKL by hand over the
  shared FS — scripts in build/tryout/gen_powp/{final_check,mkl_bench,b1_bench}.sh).
  Also `reserve.sh --status` needs `export PATH=/opt/software/slurm-19.05.8.1/bin:$PATH`
  on wallaby, and remote ssh commands land in $HOME — cd explicitly (I lost
  three round trips to this).

### Borrowed, plainly
- **gen_pfa_large (gen_r1)**: the entire engine shell + race/gate machinery,
  DFT5/DFT4/DFT25 codelets, PFA50C/PFA100C, padded mid volume, NR map ladder,
  and (mid-round) the ip* in-place chain family with its miss-stream
  rationale. This entry is substantially a derivative of theirs plus the
  DFT27 codelet, the odd-L²-tail stash, the ipf family, and the 25/27 tuning.
- **gen_batchlane**: the per-size sched-pressure function attribute.
- Campaign-wide (via all three peer records): fast NR map, map fused at
  last-axis stores where residency allows, no software prefetch in
  issue-bound passes (not re-tested here; the pf knobs that survive are
  stream pokes in the miss-bound x-pass, which their race keeps).

### Operation count (vector FMA-port ops per volume, lanes of 4)
L=25: 3·625·192 = 360,000 (+2 granule transposes/elt in phase 1's z-pass);
L=27: 3·729·218 = 476,766; L=50: 3·625·434 ≈ 813,750 (+~8% phase-1 overlap);
L=100: 3·2500·968 = 7,260,000.

### What I would do next (ranked)
1. **SoA-8 batch-lane engine for 25/27 at B=16** (two groups of 8 volumes in
   zmm lanes, split complex, zero shuffles — gen_batchlane/gen_pfa_small's
   proven 3–4x-vs-MKL structure; twiddle stages become extra broadcast-FMA
   operands between stages, exactly as gen_pfa_small's record suggests for
   this entry). The current interleaved engine pays a SWAP per twiddle cmul
   and 2 TRNC transposes per element; the batch-lane form pays one pack/unpack
   per chain. Expected: 25/27 from ~2.3x to ~3.5x MKL. This is the round-2
   headline item.
2. **L=100 residency**: state (16 MB) + c (16 MB) exceeds the 24 MB L3; the ip
   chain still streams. Candidates: interleave c into the state layout (one
   stream instead of two, gen_pfa_small's L=20 idea), or a z-split/plane-major
   state layout so each phase touches longer contiguous spans. Instrument
   first if PMU ever appears.
3. **DFT25 stage-B fused stores for 50/100** (my DFT25C macro already stores
   straight through ST; PFA50C/PFA100C still round-trip R_[25] per DFT25 —
   ~50 extra L1 accesses per line, ~10% of line ops). Needs a macro-plumbing
   rework of the PFA wrappers; measure before keeping.
4. **From round 3 (any size in class)**: the template is already parameterized
   by (GENL, GPP, line codelet); adding p^k sizes means emitting a CT line
   codelet per size — 32=2^5 (gen_pow2's, if the planner routes it here),
   49=7², 121=11², 125=5³, 81=3⁴, 64/128. A generic two-stage CT generator
   (radix from {2,3,4,5,7,8,9,25,27}) over this template covers every p^k ≤
   128. Coordinate with gen_planner/gen_race on who serves what.
5. **A/B the 3-FMA lifting twiddle form** (LITERATURE §08 §6.3) in dft25v —
   budget an hour, not a round (ice measured 11.9% op cut → 0.8% time).

## Round gen_r2

Standings into the round: led all four owned sizes in gen_r1, but
gen_pfa_large's r2 (already landed when this round started) retook the shared
sizes by a hair (50: 471.5 vs my 473.5; 100: 5008.8 vs 5026.8) via their DFT25
fused-store rework. This round: ship my r1 headline item (the SoA-8 batch-lane
engine for 25/27), take their DFT25M back, and adopt the library layers.

### What was built (three things, plus one A/B)

**1. SoA-8 batch-lane chain engine for L=25/27 at batch % 8 == 0 — the round's
headline (−33% at 25, −31% at 27).** Structure ADOPTED from gen_pfa_small
gen_r2 / gen_batchlane gen_r1 (ice bl8 lineage) exactly as both records
invited: 8 volumes fill the zmm lanes, one interleaved site arena (site =
re[8]|im[8], 128 B), plane stride padded to ≡ 2 mod 32 sites (642/738 for
25/27, plane bytes ≡ 256 mod 4096), zy sweep per x-plane then an x-pass with
the graded map fused in registers into the final-stage stores, pack x0/c once
per chain, unpack once. ZERO shuffles inside the transform.

The prime-power NOVELTY (nobody in the corpus had twiddles inside this
structure): a general-CT pencil needs a digit permutation somewhere, and a
buffered natural-order pencil would hold 50–54 live vectors (gen_pfa_small's
r1 spill lesson). Instead the pencil is the classic IN-PLACE pair — DIF
(natural in → digit-reversed out) on even-numbered steps, DIT (digit-reversed
in → natural out) on odd ones — whose every stage reads and writes the SAME
slot set (no buffering, no self-sort pass, ≤16 live vectors). The digit
reversal is an involution per axis, so alternating DIF/DIT is a steady state;
the map is elementwise, so it commutes with the layout provided **c is packed
in both site layouts once per chain** (digit-reversed copy for DIF steps,
natural for DIT). Unpack chooses by m's parity. Twiddles are the SAME exact
product-indexed tables as r1, applied at the stores as 4 broadcast FMAs.
Codelets: 25 = two stages of 5xDFT5 (gen_pfa_small's 4-constant Winograd
split core, 34 ops) + 16 W25 twiddles → 404 FMA-port vector ops per pencil
per 8 volumes; 27 = three stages of 9xDFT3 (12 ops) + 28 W27/W9 twiddles →
436. Map ≈ 20 FMA ops + 1 vdivpd per site. I verified the slot algebra and
twiddle placement against numpy BEFORE writing the C (all four
pencil-direction combinations) — see "what went wrong" for why that was not
quite enough. Offered as a 7th raced candidate ("soa", rank last: must beat
the simplest interleaved family by >3%); on the node it won by 54% (25) and
48% (27). Not offered at batch % 8 != 0 (graded 25/27 are B=16).

**2. DFT25M fused-store (ADOPTED from gen_pfa_large gen_r2, verbatim macro
shape).** dft25v + the R_[25] round-trip in PFA50C/PFA100C replaced by their
DFT25M(LDX, STO, KMAP); stage-B DFT5 outputs go straight through the wrapper's
ST with the CRT map applied. Their preprocessor note transfers: pass the store
macro as a PARAMETER.

**3. gen_race + gen_layout adoption.** The create()-race verdict is persisted
per host via gr_keyf/gr_sig/gr_wisdom_lookup/store (key
`gen_powp/chain2/L<L>/B<bucket>#<sig>`, sig over the post-GENPWP_PF-filter
candidate name set). Warm create() is a file read: **measured 2–4 ms** (50 ms
budget; cold 0.15/0.19/0.61/4.2 s at 25/27/50/100). Wisdom is also what makes
the soa family SAFE: its split-complex arithmetic is NOT bit-identical to the
interleaved families, so an unpinned noise flip between the driver's two
repeatability processes would flag NOT REPEATABLE — gen_race's documented
rationale, confirmed here (two-process cmp bit-identical at both soa sizes).
Wisdom-hit installs skip the gates (that is the budget); only picks whose
chain gate passed are ever stored. gen_layout: gl_arena (THP, staggered
mod-4096 phases) for the three SoA buffers; gl_pack8/gl_unpack8/gl_tr8x8 for
the natural pack/unpack (the permuted variants are my loops around their
gl_tr8x8).

**4. Map reciprocal A/B in the SoA x-pass**: gen_pfa_small's ONE exact vdivpd
vs my rcp14+2NR ladder. Control-first pairs on the node: div wins at 25
(34.23 vs 34.94–35.67; second pair 31.99 vs 35.60) and ties at 27 (48.31 vs
48.72). Shipped div as the SoA default (also better-rounded); knob
-DGENPWP_MAPRCP keeps the ladder compilable. Note this REVERSES
gen_batchlane's r2 result on their engine (−8% for rcp) and confirms
gen_pfa_small's on theirs — the divider-idle argument is engine-specific;
measure on your own x-pass. The interleaved families keep the r1 ladder
untouched.

### Measured on the node (a80n0, leased core via tryout.sh, graded chain, min)

| case | r1 | r2 | same-window MKL 2022 | ratio | pick |
|---|---|---|---|---|---|
| L=25 B=16 m=256 | 48.8 | **32.6** (windows: 32.0–35.7) | 125.0 | **3.83x** | soa |
| L=27 B=16 m=200 | 66.1 | **45.6** (windows: 45.6–48.7) | 149.5 | **3.28x** | soa |
| L=50 B=4 m=128  | 473.5 | **469.9–485.1** (window-bound) | 945.8–966.0 | **2.0x** | ip1 |
| L=100 B=1 m=64  | 5026.8 | **5072** (quiet; 5726 busy) | 7807 | **1.54x** | ip1 |

B=1 (ungraded at 25/27): 48.7 / 68.1 us — unchanged r1 interleaved path (soa
not offered). B=8 off-case at 25: 33.7 us (soa, single group). L=100 windows
this round were violent: gen_pfa_large's binary measured 7905 in the same
window where mine read 6270 (their quiet number is 5009); the 5072 above is
from the calmest window I got. Gates, all sizes: single call 3.6–4.5e-16
(tol 1e-12); graded chains 3.1–5.0e-14 at 1.1–1.7x the honest anchors (tol
1e-10); soa create()-gate = the whole m=2 chain on 8 volumes against
refnd + the driver's exact scalar map (exercises pack, DIF, reversed-c map,
DIT, natural-c map, unpack, and all 8 lanes in one shot); outputs
bit-identical across independent node processes.

### What did NOT work / went wrong, with the number

* **The DIT27 stage-2 twiddle carried a constant offset my stage macro
  dropped.** W27^{n1(k1+3k2)} is nonzero at k2=0 (exponent n1*k1); the
  D3STAGE macro hardcoded output 0 as twiddle-free (true for the other five
  stage shapes, whose exponents are proportional to the output index). The
  numpy simulation was CORRECT and the C translation was not — the create()
  m=2 gate caught it (l27-soa raced OUT), and a 1-D pencil unit test
  (#include the impl TU, drive the static pencils against a scalar DFT)
  pinned it in minutes: p27_dit rel L2 1.3e-1, other three 4.4e-16. Fix: J0
  parameter. Lesson: when a macro "obviously" specializes a verified formula,
  unit-test each instantiation direction anyway — the gate told me WHICH
  pencil, the unit test told me WHERE.
* **Wisdom can pin a noisy-window pick.** My first L=100 cold race ran in a
  busy window and stored ip0 as a "tie" that a cleaner re-race put 6% behind
  ip1. Handled: GEN_RACE_REFRESH re-race in a calmer window (final stored
  pick: ip1, margin 3.6%), and at round end I STRIPPED all gen_powp/ lines
  from results/wisdom_a80n0.json so the monitor's scoring run cold-races in
  its full-quiet window and stores its own verdict (its process 2 then hits
  process 1's entry — repeatability preserved). Monitors: absent entries are
  deliberate, not a failure.
* rcp14 reciprocal in the SoA fused x-pass: loses 2–10% to one vdivpd at 25
  (numbers above). Kept only as the -DGENPWP_MAPRCP control.
* tryout.sh note (update to my r1 list): the `$W`-before-assignment BUILD bug
  is fixed this round, but the remote check.py leg still receives a
  literally-quoted `'$W/c.bin'` (single quotes inside `$(...)`) →
  FileNotFoundError '/c.bin', and the `&&` chain then skips the
  repeatability cmp. Run check.py --map-check by hand on the shared FS and
  cmp two runs' outputs yourself (gen_layout r2 documents the same).

### Borrowed, plainly

- **gen_pfa_small (r1+r2)**: the whole SoA site-arena engine shape (128 B
  sites, padded planes, zy sweep + fused-map x-pass, in-place slot modules),
  the D5 Winograd split core verbatim algebra, and the vdivpd map verdict.
- **gen_batchlane (r1)**: bl8 lineage of the same structure; plane-pad and
  pack-once/unpack-once discipline.
- **gen_pfa_large (r2)**: DFT25M fused-store macro (their implementation of
  the item both our r1 records queued); their ipf/ipe/NT L=100 post-mortems
  are why I did not spend this round re-litigating the 100 x-pass.
- **gen_race (r1)**: wisdom cache machinery + the repeatability rationale.
- **gen_layout (r1+r2)**: gl_arena THP arena, gl_tr8x8/gl_pack8/gl_unpack8.
- The DIF/DIT in-place alternation with dual-layout c and the parity unpack
  is NEW here (nothing in the corpus runs general twiddles inside the
  batch-lane structure).

### Operation count (vector ops per pencil per 8 volumes, SoA path)

L=25: 2 stages x 5 DFT5 (34) + 16 twiddles (4) = 404 FMA-port; L=27: 3 stages
x 9 DFT3 (12) + 28 twiddles (4) = 436 FMA-port; zero shuffle-port ops inside
the transform (24 shuffles per 4 sites in pack/unpack, twice per chain, plus
two c packs at chain start — amortized over m >= 200 steps). Map: ~20 FMA +
1 vdivpd + 1 rsqrt14 seed per site. Interleaved-engine counts unchanged from
r1 (DFT25M moves L1 traffic, not FMA ops).

### What I would do next (ranked)

1. **Generalize the DIF/DIT SoA pencil pair over any p^k** (round 3's
   any-size-in-class rule): D5STAGE/D3STAGE are already (base, step,
   twiddle-exponent) parameterized; a DFT7/DFT2 module and a table generator
   give every p^k <= 128 with the same in-place property. The twiddle
   exponent's constant offset (the J0 bug) must be part of the generator's
   contract. Coordinate with gen_planner on routing (gen_pow2 owns 2^k).
2. **L=50 at off-case B >= 8 via the same soa family** (50 = 2x25 —
   the DFT2 stage is trivially in-place): matters if round 6 draws a
   composite with a 25- or 27-like axis at batch, or the planner routes
   50-at-B=8 here. The graded B=4 case cannot use it.
3. **SoA x-pass software-pipelining at 27**: the x-pass is now ~60% of the
   step at 27 (27 plane-stride streams + c + RFO); try two-column
   interleaving BUT heed gen_pow2's vfft32x2 register-overflow number (+15%)
   — the 27-pencil holds ~16 live vectors, two columns ~32: right at the
   cliff. Measure, do not assume.
4. **L=100**: still gen_pfa_large's phase-1 problem; watch their r3 record
   (plane pairing, huge pages) and take what wins rather than duplicating.
5. If the xarch guard flags the soa picks on CLX/SPR: the wisdom race
   already re-decides per host — verify the margins there before touching
   code.
