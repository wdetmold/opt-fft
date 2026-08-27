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

## Round gen_r3

Standings into the round: led 25/27 outright (soa picks, 32.6/45.6), tied 50
(469.9 vs gen_pfa_large 473.0 r2 board), trailed 100 by 1.7% (5072 vs 5009).
Round-3 rule now in force: supports() must take ANY size in class.  This
round: adopt gen_pfa_large's brand-new r3 ipm chain family + their two race
fixes, and ship the odd-p^k class (49/81/121/125) for round 6.

### What was built (three things, plus two A/Bs that died)

**1. ipm* deferred-map chain family (ADOPTED from gen_pfa_large gen_r3 -- read
out of their impl_3 source mid-round; their record was not yet written).**
The map z/(1+|z|) is applied to the NEXT step's phase-1 z-subpass LOADS
(map(z' + c) right after the granule load, before TRNC); the state buffer
holds raw FFT output between steps; fft3d_chain runs step 1 = plain execute,
steps 2..m = p1m+p2ip, one trailing map_span.  Bit-identical to ip* op order.
Raced as ipm0/ipm1 at 50/100 and ipm0 at the new sizes; dm picks gate the
full m=2 chain COMPOSITION at create() (their shape).  Their two race fixes
came along, both mandatory for honest ipm times: a DISTINCT tcf c-buffer
(tin-as-c let ipm's state+c loads share lines and halved its apparent
traffic -- their measured mis-pick), and trials now run IN PLACE on tout
(cfn(tout,tout,tcf), the graded steady state) instead of tin->tout.

**2. Round-3 class duty: the odd-p^k sizes 49 (7x7), 81 (9x9), 121 (11x11),
125 (5x25).**  Same engine template, four new line codelets: DFT7K/DFT11K
conjugate-pair-fold modules (33/75 FMA-port ops; full mod-n cos/sin tables
carry the sine signs), DFT9K = 3x3 CT reusing the compile-time W9 tables,
DFT49C/81C/121C = two-stage p x p CT, DFT125C = 5x25 with stage B through
DFT25M.  New-size twiddles are RUNTIME tables computed once at create() in
long double (gen_pfa_small r3's precedent for the exactness rule), stored as
{cos} + {sin,-sin} 16 B pairs so a twiddle costs VSPLAT + one
_mm512_broadcast_f64x2 (no per-use sign build).  Lite template instantiation
(GENLITE: ip0/ipf/f0/ipm0 only) keeps the added compile tolerable (whole
file: 61 s).  Preprocessor trap for the record: DFT25M declares U_[25]
internally, so DFT125C's stage array must be T_ (the PFA50C rule) -- caught
at desk-check, worth stating for the next codelet author.

**3. ipm execute-pairing bug, found and fixed on the node.**  I first paired
ipm candidates with the x_pf* (through-M) executes, copying gen_pfa_large's
table shape.  At L=81 the race rightly picked the ipm0 CHAIN (3596 us/vol vs
ip0 3870) -- and every m=1 execute then rode x_pf0 at 8383 us vs 6060 for
x_ip0 (f* executes crater at streaming sizes: f0 12798 vs ip0 4924 at 100).
All ipm rows now ride x_ip0/x_ip1.  Lesson: the execute fn rides the CHAIN
winner, so pair every chain candidate with the best execute of its working
set, not its phase structure.

### Measured on the node (a80n0, leased core via tryout.sh + hand runs, min)

| case | r2 | gen_r3 | same-window MKL 2022 | pick |
|---|---|---|---|---|
| L=25 B=16 m=256 | 32.6 | **32.40** (34.2-34.3 in later, busier windows) | 121.0-122.6 | soa |
| L=27 B=16 m=200 | 45.6 | **45.03** (47.5-47.8 busier) | 144.4-144.7 | soa |
| L=50 B=4 m=128  | 469.9 | **482.8-486.6** (window-bound; race ip1 580.7 vs ipm1 588.6 in the same elevated window) | 951.9 | ip1 |
| L=100 B=1 m=64  | 5072 | **5070-5123** | 7857 (sd 11.6%, busy) | ip0 |

Scored sizes are r2-level: the picks did not change, and the windows this
session ran 3-6% hot (MKL steady, our numbers wobbling together with peers').
B=1 chains (ungraded): 25: 42.93, 27: 59.71, 50: 484.5 -- vs r1/r2's
48.7 / 68.1 / 615.  The B=1 gains are the race fixes, not new kernels: the
old out-of-place tin->tout trial arena mis-ranked the in-place families at
B=1; the steady-state in-place trial ranks them as graded.

New sizes (node): L=49 B=8 execute 600-602 us (MKL 606 -- parity), chain
m=32 588 us/xform; L=81 B=8 execute 6060 (MKL 3770 -- 1.6x behind at plain
execute) but B=2 chain 3283 us/xform (ipm0; MKL execute alone is 3770 -- in
the graded chain shape we lead); L=121 B=1 execute 12909 (MKL 11721), chain
m=8 14158; L=125 B=1 execute 14369 (MKL 14987 -- ahead), chain 15183.
Wallaby (dev host, same source): graded chains 24.26 / 33.98 / 343.4 /
3258.9; new-size executes 325 / 1743 / 7144 / 7744 us (wallaby is simply a
faster machine at streaming sizes; per-host race REQUIRED).

Gates, all eight sizes, node: single call 3.6-5.0e-16 (tol 1e-12); graded
map-chains 3.1-5.0e-14 at 1.1-1.7x honest anchors; new-size chains
5.6e-15-1.2e-14 at 1.05-1.3x anchors; ALL bit-repeatable across independent
processes.  Setup: cold 0.13 s (25) to 7.2 s (125) [60 s budget]; warm
wisdom 1-7 ms [50 ms budget].  Round-end: all 12 gen_powp wisdom entries
STRIPPED from results/wisdom_a80n0.json (r2 practice -- monitor cold-races
in its quiet window; absent entries are deliberate).

### What did NOT work, with the number that killed it

* **ipm at L=100 on the node**: ipm0 5708 / ipm1 5484 vs ip0 4924 / ip1 4939
  us/vol (+11-16%), calm window, distinct-c race.  Mechanism: p1m's map is
  ~21 FMA-port ops per loaded vector = ~52k extra ops/plane against p1's own
  ~48k -- it DOUBLES phase-1 compute, and phase 1 at L=100 is only ~half
  miss-bound, so the "free under the misses" argument does not close.  At
  L=50 ipm ties ip (588.6 vs 580.7).  At L=81 it WINS (-7%): ~850-op line
  against the same per-plane traffic leaves more miss headroom.  The family
  stays in the pool -- the per-host, per-size race is the arbiter, and
  wallaby/CLX/SPR may disagree with Ice Lake.  gen_pfa_large: your ipm
  looks like a loser at 100 on my op counts too; check yours before
  shipping it as a default.
* **sched-pressure on the soa_step wrappers** (gen_batchlane r3 ships it at
  -5.7..-10% on their register-explicit pencils): on MY soa engine it LOSES
  32% at 25 (34.3 -> 45.4) and 22% at 27 (47.8 -> 58.3), control-first
  same-window pairs, MKL steady at 121/144.  Their win is specific to
  spill-heavy codelets.  Knob -DGENPWP_SOASCHED kept for cross-arch races.
* **Dead-store audit of the soa pencils** (gen_batchlane r3's objdump
  method): NOT a problem here -- gcc already forwards ~106 stage-1->stage-2
  loads and DSEs ~61 stores per soa_step (244 loads / 239 stores emitted vs
  350/300 naive).  No barrier surgery warranted; audit cost 10 minutes and
  is worth repeating each round.
* The x_pf* execute pairing (above): L=81 m=1 went 8383 -> 6060 us on the
  repair.

### Borrowed, plainly

- **gen_pfa_large (gen_r3, their working source)**: the whole ipm idea and
  p1m shape, the deferred-chain schedule, the m=2 composition gate, the
  distinct-tcf race fix, and the in-place steady-state trial fix.
- **gen_pfa_small (gen_r3)**: runtime long-double tables as the sanctioned
  exactness route; their consumption-order-c negative result kept me from
  reordering the new tables.
- **gen_batchlane (gen_r3)**: the asm store-count audit method (applied,
  clean verdict) and the warning that sched-pressure wins are
  codelet-specific (confirmed, opposite sign here).
- The {sin,-sin} pair-table + broadcast_f64x2 twiddle form and the GENLITE
  reduced-template instantiation are new here.

### Operation count (vector FMA-port ops per line, lanes of 4, new sizes)

L=49: 14 x DFT7K(33) + 36 tw(2) = 534;  L=81: 18 x DFT9K(32) + 64 tw = 850;
L=121: 22 x DFT11K(75) + 100 tw = 1850;  L=125: 25 x DFT5(16) + 96 tw +
5 x DFT25M(192) = 1552.  Per volume x3 axes as ever.  Scored sizes unchanged
from r2 (ipm moves the map's ~21 ops/vec from its own pass into p1's loads;
FMA total identical).

### What I would do next (ranked)

1. **L=81 execute gap (1.6x behind MKL)**: try 81 = 27x3 reusing the tuned
   DFT27C machinery (26 twiddles/stage vs 64) or a 3^4 four-stage form;
   also give the lite sizes the ip1/pf1 poke variants -- the 81 chain
   already leads MKL, so this only matters if round 6 draws 81 and scores
   execute-heavy shapes.
2. **Read gen_pfa_large's r3 record when it lands**: their ipm verdicts at
   40/80 and any L=100 phase-1 traffic work supersede my guesses; take what
   wins, as ever.
3. **SoA-8 chain at 27, x-pass two-column pipelining**: still the one
   untried item from r2 (register cliff at ~32 live vecs); 27 remains the
   weakest scored ratio (3.2x vs 3.8x at 25).
4. **B % 8 != 0 batches at 25/27**: lane-replicate remainder volumes
   (gen_batchlane r1) would extend the soa engine to any B >= 8; matters if
   round 6 draws a p^k with an awkward batch.
5. **Planner handshake**: tell gen_planner supports() now takes
   49/81/121/125 so the trunk routes p^k draws here instead of Bluestein
   (we beat it 5-15x at these sizes).

### Harness notes (unchanged bugs, one new script)

tryout.sh's remote map-check leg still gets the unexpanded '$W/c.bin' and
its && chain then skips the repeatability cmp; build/tryout/gen_powp/
r3_final.sh runs the full graded + B=1 + new-size sweep with map-checks and
two-process cmps by hand on a leased core (bash r3_final.sh <core>).  squeue
on wallaby still needs the slurm PATH shim.  Off-case sizes: tryout accepts
any L (m defaults to 1), which is how the new sizes were driven.

## Round gen_r4

Standings into the round: led all four owned sizes (25: 32.21, 27: 44.81,
50: 473.68 by 0.07% over gen_pfa_large, 100: 5021.0 by 1.4%).  But
gen_pfa_large's r4 (already landed): volume-major chain + a new ipp family
that reads 4923 at 100 -- doing nothing would have lost both shared sizes.
This round is deliberately a fast-follow: take their two structural items,
verify them on MY engine with the honest A/B protocol gen_batchlane r4
published, and re-rank where the evidence says so.

### What was built (three adoptions, one rank decision)

**1. VOLUME-MAJOR chain schedule (ADOPTED from gen_pfa_large gen_r4, who
took it from gen_dense_prime / gen_rader / gen_layout).**  fft3d_chain now
runs ALL m steps on one volume before the next, for both interleaved chain
paths (deferred and plain): per-step working set drops from the whole
batch's state+c to one volume's slice (0.5 MB at 25 -- L2-resident -- to
4 MB at 50 B=4).  Per-volume FFT op order unchanged; outputs bit-identical
to r3's step-major at 50/100 (VDv % 8 == 0); at 25/27 the per-volume
map_span tail moves 1-3 complex per volume from the vector ladder to the
exact scalar map (gates unaffected, and the graded 25/27 picks are soa).
The soa engine was group-major from birth -- untouched.

**2. ipp* plane-prepass deferred-map family (ADOPTED from gen_pfa_large
gen_r4, read from their landed impl_4 source).**  The ipm schedule (state
holds raw FFT output between steps; step 1 plain, one trailing map_span)
but the map runs as map_span's perfectly sequential per-plane prepass into
an L2-resident scratch plane (M's base, 10-250 KB at these sizes) that a
factored p1body then consumes.  Their mechanism transfers exactly as
advertised: ipm and ipp have identical per-step traffic, and my r3 ipm
still lost 11-16% at 100 -- the loss was the ladder's port/latency
footprint inside the granule-load stream, and ipp pays it only at plane
seams.  Raced as ipp0/ipp1 at 50/100, ipp0 at 25/27 and the lite sizes
(pf ids 9/10; ipp rides the x_ip* executes per my r3 pairing lesson --
NOTE gen_pfa_large pairs ipp with x_pf* in their table; executes differ
per engine, pair with your own best).

**3. The create() race times the VOLUME-MAJOR shape (their race fix): per
volume one unmeasured warm step, then R timed steps in place on that tout
volume (R = 8/6/4 by volume bytes <=2/<=8/>8 MiB), min over 4 interleaved
rounds; tc[] is now per VOLUME-step.**  Wisdom tag chain2 -> chain4 so no
stale step-major verdict can install.

**4. Rank reorder at 50/100 (own decision, from paired evidence).**  The
first node cold race scored ipp1-at-100 a 1.4% "tie" and hysteresis kept
ip0.  Same-core tight alternation (ONE held slot lease, alternating forced
GENPWP_PF binaries, gen_batchlane r4's protocol) settled it: ipp1 wins 4
of 5 pairs, quiet floors 4905-4929 vs ip0's 5110-5160 (-4.0..-4.5%); the
tie verdict was window drift, exactly gen_pfa_large's lesson 1.  So at
50/100 (and ipp0 to rank 1 at the lite sizes) ipp is now ranked FIRST: a
busy-window race margin that shrinks under the 3% hysteresis must fall to
the measured winner, not the simpler loser.

### Measured on the node (a80n0, leased cores; this session ran BIMODAL --
### two sustained core states ~6-14% apart, MKL steady, exactly
### gen_pfa_small r4's method note; quiet-state minima quoted, hot in parens)

| case | r3 | gen_r4 | same-window MKL 2022 | pick |
|---|---|---|---|---|
| L=25 B=16 m=256 | 32.21 | **31.95** (34.2 hot state) | 121.2 | soa (3.79x) |
| L=27 B=16 m=200 | 44.81 | **47.6-47.8 every window this session** -- code bit-identical to r3, MKL steady 144.5-145.0; the quiet floor remains r3's 45.0 | 144.5 | soa |
| L=50 B=4 m=128  | 473.68 | **478.3** (484.5 hot; MKL 953.8 vs r3-board 947.2 -- window-adjusted ~ r3) | 953.8 | ipp0 |
| L=100 B=1 m=64  | 5021.0 | **4861** (4899 second window; paired-A/B floor 4905-4929 vs ip0 5110-5160) | 7801 | ipp1 (1.60x) |

B=1 chains (ungraded): 25: 42.79 (pick ipf), 27: 59.85 (ip1; one hot-core
tryout read 68.0 -- core-hop, not code), 50: **476.1** (ipp0; r3 484.5, and
B=1 now ~= B=4).  New sizes, same m as r3's finals: 49 B=8 m=32 **548.5**
(r3 588, -6.7%, pick ipp0), 81 B=2 m=16 **3037.7** (r3 3283, -7.5%, ipp0),
121 m=8 14103 (ipm0 keeps it), 125 m=8 15258 (ipm0).  Wallaby: 100 B=1 m=4
ipp1 3100 vs ip1 3331 (-6.9%); 50 B=4 m=8 ipp1 325.7 vs ip1 339.3 (-4%).

Gates, all eight sizes, node, by hand (tryout's map-check leg still has the
'$W/c.bin' quoting bug): single call 3.6-5.0e-16 (tol 1e-12); two-step m=2
1.40/1.56/2.36/2.72e-15 at 25/27/50/100 (tol 3e-14, >10x margin); graded
chains 3.10/3.47/5.03/4.18e-14 at 1.1-1.7x honest anchors (tol 1e-10);
new-size chains 5.6e-15-1.2e-14; ALL bit-repeatable across independent
processes.  Setup: cold 0.38-4.1 s (pools now 8-10 candidates; 60 s
budget); warm wisdom 1-6 ms (50 ms budget).  Round end: all gen_powp
entries STRIPPED from results/wisdom_a80n0.json (r2/r3 protocol).

### What did NOT work / boundaries, with the numbers

* **ipp does NOT win everywhere**: at 121/125 ipm0 stays ahead (wallaby:
  121: ipm0 7835 vs ipp0 8396; 125: ipm0 8349 vs ipp0 8545) -- at ~1850-op
  lines phase 1 is compute-fat enough that the in-stream ladder hides, and
  ipp's extra L2 plane round-trip is pure cost.  Family stays raced, race
  arbitrates, exactly like ipm-at-100 in r3.
* **The rank reorder has a measurable worst case**: at 50 B=1 in one hot
  window the race stored ipp0 with margin -2.9% (an ip variant was faster
  there); rank-first means we eat <=3% when ipp genuinely trails inside
  the hysteresis band.  Accepted: both scoring-relevant cells (50 B=4, 100
  B=1) show ipp ahead in paired A/Bs, and the CLX advisory should favor
  ipp harder (smaller footprint = contention armor, their r4 busy-window
  -11%).
* **gl_tr8x8_c2i (gen_layout r4's new exit primitive) NOT adopted**: my r2
  soa unpack already loads re/im pairs as alternating tr8x8 rows, which IS
  the fused shape -- 48 shuffles per 8 sites either way; and pack/unpack
  runs twice per graded chain (amortized over m >= 128).  No win to take;
  recorded so nobody re-derives it.
* **gen_layout r4's NT stores (-19% on their L=100 floor) NOT applicable
  here**: my p2ip stores are read-modify-write on lines the codelet just
  read (RFO is real), the y-subpass writes land on prepass-warmed lines by
  design, and gen_pfa_large's race table already shows an NT variant
  (ipnt 8009 vs ipp1 5180) cratering in this engine shape.
* **One hot-core tryout read 27-B=1 at 68.0 us (+14% vs r3)** and nearly
  looked like a regression; the held-lease rerun read 59.85.  tryout.sh
  acquires a fresh lease per invocation and hops cores (gen_batchlane r4's
  finding) -- for ANY conclusion, hold one lease and alternate, or rerun.

### Borrowed, plainly

- **gen_pfa_large (gen_r4)**: the volume-major chain schedule (transitively
  gen_dense_prime / gen_rader / gen_layout), the whole ipp idea and its
  plane-seam granularity rationale, and the volume-major race arena fix
  with the chain-tag bump.  This round is mostly their round, ported and
  re-verified; the rank-first-for-ipp decision and the lite-size ipp
  verdicts (wins 49/81, loses 121/125) are mine.
- **gen_batchlane (gen_r4)**: the same-core held-lease alternation
  protocol -- it re-decided the pick at 100 (the race's tie was wrong) and
  acquitted a false regression at 27 B=1.
- **gen_pfa_small (gen_r4)**: the bimodal-core-state method note (min-of-
  mins per config, control-first pairs) -- used to read every number above.

### Operation count

FMA-port vector ops per line unchanged at all eight sizes (192/218/434/968
scored, 534/850/1850/1552 lite).  ipp moves the map's ~21 ops/vector from
its own pass into the per-plane prepass (per-step total identical to ip*)
and adds one L2 plane round-trip (write + re-read of M's base, 10-250 KB)
per plane; per-step DRAM accounting at 100 drops from ~112 to ~80 MB.
Volume-major moves no arithmetic at all.

### What I would do next (ranked)

1. **XARCH.md lands after this round**: verify the per-host race flips --
   ipp should widen on CLX (traffic-bound), the soa picks need checking on
   SPR; the knobs and the wisdom race are already in place, race, don't
   retune.
2. **L=27 remains the weakest scored ratio** (3.0-3.2x vs 3.8x at 25): the
   x-pass two-column pipelining item is now three rounds old and untried
   (register cliff at ~32 live vecs); one honest same-core A/B, one
   afternoon, then close it either way.
3. **L=50 B=4**: gen_pfa_large's suggestion for whoever moves first -- a
   2-volume-pair schedule (4 MB, L3-safe) or soa-style 2-lane packing at
   B=4.  Nobody has measured either; the cell is a dead heat.
4. **Lite sizes at execute-heavy shapes**: 81 execute is still 1.6x behind
   MKL (chain leads); 27x3 reuse of DFT27C machinery is the standing idea
   if round 6 draws 81.
5. **Planner handshake unchanged**: supports() covers 25/27/49/50/81/100/
   121/125; keep gen_planner routing p^k draws here, not to Bluestein.

## Round gen_r5

Standings into the round (r4 board): led 25 (32.100) and 27 (44.281)
comfortably; TRAILED 50 by 1.5% (472.873 vs gen_pfa_large 466.040); dead
heat at 100 (4828.216 vs their 4827.566).  gen_pfa_large's r5 was already
landed in impl_5 when this round started: a pair-packed map ladder and three
c-stream L3-bypass chain families.  Both apply to my engine nearly verbatim
(we share the shell); this round takes both, plus one idea of my own for the
SoA x-pass divider from gen_layout's new r5 primitive.

### What was built (three things)

**1. PAIR-PACKED map ladder in map_span (ADOPTED from gen_pfa_large gen_r5,
their map_step_pair, verbatim).**  The sequential map paths ran the ~18-op
NR ladder on vectors where each |z|^2 sits duplicated in both complex lanes
-- half the ladder lanes computed nothing new.  Two vectors' 8 distinct
|z|^2 now pack into one zmm (2 shuffles), ONE ladder runs for both, and the
reciprocals unpack pair-duplicated (2 shuffles): ~21 arithmetic ops + 4
shuffles per pair vs 36 + 2 before, a ~35-40% cut in map-pass uops.
BIT-IDENTICAL per element (q_re + q_im commutes; everything else is
elementwise), so it flows into every ip* map pass, the ipp prepass, and the
deferred-chain trailing span at all eight sizes with zero numerics risk.
This is the round's headline: **L=50 dropped 9.9%** (472.9 -> 426.1) -- the
map is ~40% of step arithmetic at the L3-resident sizes, exactly
gen_pfa_large's accounting -- and every B=1 chain fell 6-16%.

**2. c-stream L3-bypass families ipq1/ipk1/iqn1 at 50/100, ipq0 at the lite
sizes (ADOPTED from gen_pfa_large gen_r5).**  c is read once per step; its
only reuse is across steps, so caching it is pollution wherever the
VOLUME-MAJOR working set (state+c per volume at B=1, per batch at small B)
exceeds the 24 MB L3.  ipq* = ipp* with the prepass c fetched PREFETCHNTA;
ipk1 = CLFLUSHOPT-ing clean c lines a pair behind (architectural semantics
where NTA fill policy is implementation-defined); iqn1 = ip1 with the NTA-c
map pass.  All compute bit-identical values.  Ranked LAST; pf ids 11/12/13
(ipq0 = 14); wisdom tag chain4 -> chain5.  Node verdicts this session:
at 100 B=1 (32 MB/volume-chain) **ipk1 won the cold race trial 5312 vs ipp1
5367** and the 3% hysteresis correctly handed the install back to ipp1; a
4-pair held-lease alternation (ipk1=pf12 vs ipp1=pf10) split 2-2 in a
violently bimodal window with min-of-mins 4721 vs 4762 (-0.9% for ipk1) --
NOT enough to justify rank-first (my r4 ipp reorder needed 4/5).  Left
ranked behind ipp1: if ipk1 wins by >3% in the monitor's quiet window it
installs itself.  At 50 B=4 they lose big as predicted (ipq1 630.5 / iqn1
613.1 / ipk1 548.4 vs ip1 468.9): the batch's c IS the L3 reuse set there.
Surprise worth recording: **ipq1 (NTA) is WORSE than plain ipp1 at 100**
(5950 vs 5367) while ipk1 (flush) is better -- on this ICL part the NTA
prefetch actively hurts the prepass (dropped-from-L1-too-early, or prefetch
distance fighting the ladder); the flush variant is the one that works.
gen_pfa_large: check which of your three actually carries the win on your
side before reading the family as confirmed.

**3. PAIRED-vdivpd SoA x-pass map at 25/27 (reciprocal-product trick from
gen_layout gen_r5's gl_map16 / gen_dense_prime r4's queued idea, adapted to
the split-complex site).**  In split complex the ladder lanes are all
distinct (nothing to pack), but the final divide is shareable: the map-fused
final stages now map output pairs through q = 1/(d0*d1), r0 = q*d1,
r1 = q*d0 -- ONE vdivpd per two sites.  x-pass divider occupancy: 25 vdivpd
-> 15 per pencil at 25 (5 outputs/group: single + pair + pair), 27 -> 18 at
27 (3 outputs: single + pair); cost 3 vmulpd per pair; ~1-2 ulp on the
reciprocal against a ~60 ulp/step budget (measured: graded-chain rel L2
moved 3.10e-14 -> 3.145e-14 at 25, i.e. nothing).  Same-core held-lease
A/Bs, clean windows: 25: 33.56/33.74/33.76 vs 34.17/34.26/34.38 (3/3 pairs,
**-1.6%**); 27: 44.26/44.44/44.63 vs 44.84/44.98/45.23 (3/3, **-1.3%**).
Knob -DGENPWP_NOMAPPAIR restores the r2-r4 one-div-per-site form for
cross-arch races.  Implementation note: the final-stage macros became
D5STAGEMP/D3STAGEMP and the pencil generators take a final-stage macro
parameter (D5FS_MAP/D3FS_MAP) instead of a store macro -- the twiddle-store
path (SSTW stages) is untouched.

### Measured on the node (a80n0 core 4, QUIET window, r5_final.sh battery;
### sd 0.01-0.4% within runs; same-window MKL not re-run, r4 references)

| case | r4 board | gen_r5 | delta | pick |
|---|---|---|---|---|
| L=25 B=16 m=256 | 32.100 | **31.420** | -2.1% | soa (paired-div) |
| L=27 B=16 m=200 | 44.281 | **44.567** | wash (quiet r4 floor was 44.8-45.0; the A/B says the code is -1.3%) | soa (paired-div) |
| L=50 B=4 m=128  | 472.873 | **426.098** | **-9.9%** | ipp1 (ip1 fastest trial, 3% hysteresis) |
| L=100 B=1 m=64  | 4828.216 | **4802.410** | -0.5% board-to-board; paired floors 4721 (ipk1) / 4762 (ipp1) vs r4's 4905-4929 | ipp1 (ipk1 raced, see above) |

B=1 chains (ungraded): 25: **39.96** (r4 42.79, -6.6%), 27: **50.35**
(r4 59.85, -15.9%), 50: **416.87** (r4 476.1, -12.4%; B=1 is now faster
than B=4 -- pure pair-packed-map effect on the in-place/prepass families).
New sizes, same shapes as r3/r4 finals: 49 B=8 m=32: **491.2** (r4 548.5,
-10.4%); 81 B=2 m=16: **2868.0** (r4 3037.7, -5.6%); 121 m=8: **13960**
(r4 14103); 125 m=8: **15661**.  Lite-size bypass verdicts (hot window):
ipq0 loses at 49 (696 vs ip0 552) and 81 (3517 vs 3140) -- volume-major
makes the per-volume working set L3-resident there -- is a close 2nd at 121
(15194 vs ipp0 15138) and WINS at 125 (16022 vs 16139) by 0.7%.

Gates, all eight sizes, node, by hand (r5_final.sh; tryout's map-check leg
still has the '$W/c.bin' quoting bug): single call 3.6-5.0e-16 (tol 1e-12);
two-step m=2 1.42/1.55/2.36/2.72e-15 at 25/27/50/100 (tol 3e-14, >11x
margin); graded chains 3.15/3.10/5.03/4.18e-14 at 1.1-1.7x honest anchors
(tol 1e-10); new-size chains 5.6e-15-1.2e-14; ALL bit-repeatable across
independent processes.  Setup: cold 0.30-2.82 s scored, 4.1-4.5 s lite
(60 s budget); warm wisdom unchanged ms-scale.  Round end: all gen_powp
wisdom entries STRIPPED from results/wisdom_a80n0.json (r2-r4 protocol).

### What did NOT work / went wrong, with the numbers

* **NFS-stale source builds (harness trap, cost ~40 minutes).**  tryout.sh
  compiles on the node from the shared FS; my first three "successful"
  builds compiled a MINUTES-old view of gen_powp.c -- the L=50 binary had
  10 candidates when the source had 13, while md5sum on both hosts already
  agreed by the time I compared.  Symptom that caught it: the verbose race
  printed no ipq/ipk/iqn rows.  Rule: after editing, `strings bin | grep
  <new-candidate-name>` BEFORE trusting any number from a rebuilt binary.
* **A busy core poisons the race AND the wisdom pins it.**  My first cold
  race ran on a contended core: every trial read 20-30% high and soa
  mis-ranked 42.8 us/vol against its true 34.8 (interleaved candidates
  evict the SoA arena and i-cache between trial rounds; 256-step graded
  chains hide this, 2-step trials do not) -- the race stored l25-ip0 and
  the graded chain shipped 41.1 us instead of 31.4.  Stripped the entry,
  re-raced on a live core: soa wins its slot back (34.8 vs ip0 46.7).
  This is gen_batchlane r4's core-hop lesson biting the RACE itself, not
  just A/Bs; the monitor's quiet-window race is the real arbiter, and the
  strip-at-round-end protocol is what protects it.
* **ipq (NTA) at 100: +11% vs ipp1** (5950 vs 5367, same cold race that had
  ipk1 at 5312).  The NTA c-prefetch is not a free L3 bypass on this host;
  only the CLFLUSHOPT variant delivers.  Kept raced for CLX/SPR.
* **ipk1 rank-first at 100: evidence insufficient** (2-2 pairs, -0.9%
  min-of-mins, bimodal window).  Not reordered; revisit with the monitor's
  quiet number or XARCH data.
* The wisdom JSON is rewritten wholesale by concurrent implementers'
  create() races all session -- entries I greped one minute were gone the
  next.  Do not fight it mid-session; strip your own keys at round end and
  trust the tag/sig machinery for staleness.

### Borrowed, plainly

- **gen_pfa_large (gen_r5, their landed impl_5 source)**: map_step_pair
  verbatim (the round's headline win at 50 and every B=1 cell), and the
  whole ipq/ipk/iqn c-bypass family design with its ranked-last discipline.
  The lite-size ipq0 instantiation, the NTA-is-worse-than-flush-at-100
  verdict, and the 50/49/81 volume-major-residency loss accounting are mine.
- **gen_layout (gen_r5)**: the gl_map16 reciprocal-product idea, re-derived
  for split-complex sites where only the divide is shareable (their
  pair-compression of |w|^2 does not apply -- soa lanes are already
  distinct).  Credited also to gen_dense_prime r4 who queued the trick.
- **gen_batchlane (gen_r4)**: the held-lease same-core alternation protocol,
  again, for every A/B verdict above.

### Operation count

FFT FMA ops unchanged at all eight sizes (192/218/434/968 scored,
534/850/1850/1552 lite per line).  map_span: ~21 arith + 4 shuffles per
vector PAIR (was 36 + 2), all ip*/ipp/deferred paths.  SoA x-pass map:
divider ops 25 -> 15 (L=25) / 27 -> 18 (L=27) per pencil, +3 vmulpd per
paired output, ladders unchanged.  ipq/ipk/iqn move no arithmetic (hints
and flushes only).

### What I would do next (ranked)

1. **L=27 x-pass two-column pipelining** -- now FOUR rounds on this list,
   still the weakest scored ratio.  Note the paired-div map raised final-
   stage register pressure (~6 more live vecs); the ~32-zmm cliff is closer
   than it was in r2.  One honest same-core A/B, then close it either way.
2. **100 rank decision**: re-run the ipk1-vs-ipp1 pairs in a quiet window
   (this session never got one at 100); if ipk1 holds -1% or better across
   4+ pairs, rank it first, else leave.  Watch gen_pfa_large's r5 record
   for their side's verdict.
3. **Tier-2 structural at 100** (literature 11): two-axes-per-pass fusion
   is the only idea on the table that changes the 100 traffic floor rather
   than shaving the map; it is a rewrite of phase 1+2 scheduling -- budget
   a full round, coordinate with gen_pfa_large so only one of us burns it.
4. **Round 6 readiness unchanged**: supports() covers the eight sizes,
   lite pools now carry a large-L bypass candidate, cold create 4.5 s worst
   case.  If the draw lands a p^k not in {49,81,121,125}, nothing serves it
   better than Bluestein -- a generic two-stage CT line-codelet emitter
   over this template remains the (unbudgeted) fix.

## Round gen_r6

Standings into the round (r5 board): led 25 (31.352) and 50 (415.637 vs
gen_pfa_large 420.957); trailed 100 by 1.9% (4618.710 vs 4531.445) -- and
the L=27 cell shipped **51.444 where ~44.5 was available**: the monitor's
quiet-window cold race installed l27-ip0 (wisdom: 58.63 us/vol trial, margin
-0.6%, "tie"), i.e. soa was mis-ranked OUT in the scoring race itself.  That
is my r5 busy-core race lesson biting the monitor's own window, and it set
this round's agenda: the race must measure what the chain is graded on.
Round 6 is also the surprise round (three sizes in 14..127, library-scored);
supports() already covers every p^k in range that is not 2^k.

### What was built (two race fixes, one measured rejection, one rank flip)

**1. SOA-vs-INTERLEAVED PLAYOFF in the create() race (own fix; the round's
headline).**  Mechanism of the wound: interleaved trials evict the SoA
arenas (7.6 MB at 27) and i-cache between interleaved trial rounds, and a
2-step soa trial re-pays that refill EVERY round where the graded 200-step
chain pays it once per chain.  Fix: after the short min-of-rounds trials,
the soa candidate and the best interleaved candidate are re-timed
head-to-head in alternation -- 24 steady steps per arm per round, min over
3 rounds -- and the playoff numbers feed back into tc[] with min() on the
interleaved side (the playoff can only make the comparison FAIRER to the
incumbent); soa still must clear the 3% simplest-first hysteresis.  Verified
on the node: in a busy window the playoff read soa 46.65 vs ip1 51.10 at 27
(soa installs, +8.7% clear) and soa 36.4 vs 46.5 at 25; the graded chain
then shipped 45.25 at 27 -- vs the r5 board's 51.44, a ~12% recovery on a
scored cell without touching kernel code.  Setup stays 0.41-0.58 s at
25/27 (60 s budget).  Wisdom tag chain5 -> chain6 so the mis-picked
l27-ip0 verdict (or any short-trial verdict) can never replay.

**2. Rank-0 CHALLENGER playoff at the non-soa sizes (same disease,
interleaved case).**  Rank order encodes graded-shape held-lease evidence,
but a 4-step trial dilutes benefits that accrue in steady state -- ipk1's
c-flush at 100 keeps the L3 clean for LATER steps' state re-reads, and the
same cold race that timed my 5/5-winning ipk1 at +5.3% would never let the
3% hysteresis reach it.  When the lowest-rank live candidate differs from
the trial best by < 15%, the two are re-judged on long steady runs (24
steps per arm per round at <= 8 MiB volumes, 12 above; 3 rounds,
alternating).  Observed at 50: playoff ipp0 424.9 vs ip1 414.9 -- inside
the 3% band, ipp0 keeps the slot (r4's accepted <= 3% worst case,
re-observed and still acceptable).

**3. TWO-COLUMN map-fused x-pass at 27: built, measured, REJECTED (closes
the item queued since r2).**  p27m2 processes two adjacent flat columns per
pencil: column B's stage loads precede column A's stores in program order
(st is a runtime stride -- the compiler cannot alias-reorder), so 12 site
loads are in flight per stage visit, and the map stage pairs divides ACROSS
columns (3 vdivpd per 6 sites vs 4 in-column).  Quiet-window held-lease
pairs (MKL steady at 144.57), 27 B=16 m=200: **two-column 47.12-47.39 vs
single-column 45.18-45.57 min (+4.4%), single wins 4/5 pairs** -- the
doubled straight-line body costs more in the front end than the paired
loads and the saved divide buy; the x-pass streams were already covered by
the OoO window.  Shipped x-pass = the r5 single-column form (bit-identical
chain values); two-column kept opt-in behind -DGENPWP_XCOL2 for cross-arch
races (its build passes all gates: m=200 chain 3.303e-14, m=2 1.581e-15).
gen_batchlane r6's DFT5X2-at-15 ILP verdict (+1% for fusing) does NOT
transfer to this engine -- pair-fusion is codelet-shape-specific, the same
boundary as sched-pressure in my r3.

**4. ipk1 ranked FIRST at 100 (settles my r5 open question).**  5/5
held-lease pairs at L=100 B=1 m=64, one core, window running quiet -> busy:
ipk1/ipp1 mins 5282/5317, 5027/5399, 5606/6355, 5499/6577, 5458/5686
(-0.6% to -16%) -- the margin WIDENS under contention: flushing clean
c lines frees L3 that contention squeezes (gen_pfa_large r4's
smaller-footprint-is-contention-armor, confirmed on my side).  The r5
criterion ("-1% or better across 4+ pairs") is met.  50 ranks unchanged
(bypass still loses at B=4: the batch's c IS the L3 reuse set there).

### Measured on the node (a80n0, held leases via slot_lease, mixed windows)

| case | r5 board | gen_r6 | pick | note |
|---|---|---|---|---|
| L=25 B=16 m=256 | 31.352 | **32.103** (sd 0.33%) | soa (playoff) | window; code = r5 |
| L=27 B=16 m=200 | 51.444 (mis-pick) | **45.254** (sd 0.26%) | soa (playoff) | ~-12%: the playoff recovery |
| L=50 B=4 m=128  | 415.637 | **424.218** (sd 0.27%) | ipp0 (<=3% rank rule) | window; code = r5 |
| L=100 B=1 m=64  | 4618.710 | **4530.426** (sd 0.35%) | ipk1 (rank 0) | -1.9%: matches gen_pfa_large's r5 cell |

B=1 chains: 25: 40.055, 27: 50.453, 50: 486.5 (hot, sd 3.9%).  New sizes,
same shapes as r5 finals: 49 B=8 m=32: 509.6; 81 B=2 m=16: 2994.9;
121 m=8: 14419.8; 125 m=8: 15804.6.  Gates, all eight sizes: single call
3.6-5.0e-16 (tol 1e-12); two-step m=2 1.422/1.549/2.361/2.721e-15 at
25/27/50/100 (tol 3e-14, >11x margin); graded chains 3.145/3.098/5.028/
4.181e-14 at 1.05-1.7x honest anchors (tol 1e-10); new-size chains
5.6e-15-1.2e-14; ALL bit-repeatable across independent runs.  Setup: cold
0.41-4.6 s (playoffs included; 60 s budget); warm wisdom unchanged
ms-scale.  -Wall -Wextra: only the pre-existing unused xc_ipq0_50/100
(ipq0 is pooled at lite sizes only).  Round end: all gen_powp keys
STRIPPED from results/wisdom_a80n0.json (r2-r5 protocol; note the file is
rewritten wholesale by concurrent create() races -- strip, wait, re-verify).

### What did NOT work / boundaries, with the numbers

* **Two-column x-pass at 27: +4.4%** (47.12-47.39 vs 45.18-45.57, 4/5
  quiet pairs) -- see item 3.  Do not re-derive: pair-fusion of stage-2
  groups is an ILP win only on register-explicit spill-heavy codelets
  (gen_batchlane's), not on in-place slot pencils whose stages already
  stream through memory.
* **ipk1's short-trial number is not its graded number**: +5.3% in the
  4-step cold race vs -0.6..-16% in 5/5 graded m=64 pairs.  Any candidate
  whose win comes from cross-step cache state (bypass families, deferred
  maps at the margin) is systematically under-read by short trials; that
  is what the challenger playoff is for.  If a future family's benefit
  needs even longer horizons than 24 steps, raise PS before adding rank
  hacks.
* The challenger playoff at 50 re-confirms ipp0-vs-ip1 as a genuine <= 3%
  coin flip (424.9 vs 414.9 this window); the rank rule eats <= 3% there
  by design and the CLX/SPR advisory is the reason it stays.

### Borrowed, plainly

- **gen_batchlane (gen_r6)**: the DFT5X2/DFT7X2 pair-fusion idea motivated
  the two-column try; opposite verdict on my engine, recorded above so the
  boundary is explicit.  Their held-lease alternation protocol remains the
  measurement standard for every number in this section.
- **gen_dense_prime (gen_r6)**: their custody and lazy-map negative
  results (both +2..13% inside cache levels the OoO already covers) are
  the same mechanism class as my ipm-at-100 loss; I did NOT re-litigate
  z-into-x fusion or in-stream maps this round because their numbers
  already close those doors on L3-resident engines.
- **gen_pfa_large (gen_r4/r5, transitively)**: the c-bypass family whose
  ipk1 variant now leads 100, and the contention-armor account its 5/5
  playoff pairs confirm.
- The playoff machinery (both flavours) is new here; monitors and rivals
  are welcome to it -- it is ~60 lines in tune() and the mechanism
  (short trials re-pay refills the graded chain amortizes) applies to any
  entry racing a cache-hungry candidate against lean ones.

### Operation count

Unchanged at all eight sizes (192/218/434/968 scored, 534/850/1850/1552
lite FMA-port vector ops per line; soa 404/436 per pencil per 8 volumes).
The playoffs move no arithmetic; the shipped chain paths are bit-identical
to r5 at every size.

### What I would do next (ranked)

1. **Round-6 surprise draw**: if a p^k in {25,27,49,81,121,125} is drawn,
   the playoff protects the pick; nothing to do but watch.  If the trunk
   routes a p^k at batch % 8 == 0 and B >= 8 through me at 25/27-like
   sizes, soa serves it; other p^k at odd batches ride the interleaved
   families.
2. **Two-axes-per-pass fusion at 100** (literature 11 tier 2) is still the
   only lever that changes the 100 traffic floor; a full-round rewrite,
   coordinate with gen_pfa_large so only one of us burns the round.
3. **XARCH follow-up**: the XCOL2 and NOMAPPAIR knobs plus the per-host
   race are the portability levers; if SPR/CLX flip the 27 verdict, race
   an XCOL2 build there rather than retuning Ice Lake.
4. **L=81 execute** remains 1.6x behind MKL (chain leads); 27x3 reuse of
   DFT27C is the standing idea if a draw makes execute-heavy shapes matter.

## Round gen_r7

Standings into the round (r6 board): led 25 (31.971), 27 (44.544) and 50
(415.524 vs gen_pfa_large 420.018); trailed 100 by 1.0% (4617.510 vs their
4570.267).  The surprise test (r6 addendum) won all three unseen sizes, so no
planner/race fire to fight.  The rounds-7/8 brief says: spend the queued
literature backlog.  Mine was two op-diet items for the SoA engine at 25/27
(the two cells nobody shares with me); the round's ONE piece of luck is that
gen_pfa_large's r7 header closed the two-axes-per-pass item (brief item 1,
named for both of us) WITHOUT CODE before I burned anything on it -- see
"borrowed".

### What was built (one new technique, one adoption-and-rejection, knobs)

**1. 3-SHEAR LIFTED TWIDDLE ROTATIONS at the SoA twiddle stores (literature
08 6.3 -- queued in my r1 record as "budget an hour"; six rounds later it is
the round's headline).**  In split complex a twiddle W^j is a pure plane
rotation of (re, im) -- no lane swaps anywhere -- so the classic lifting
factorization finally has a home with zero shuffle tax: the 4-op twiddled
store (2 vmul + 2 vfma, r/i independent) becomes THREE FMA-port ops,
u = re + T*im; im' = im - S*u; re' = u + T*im' with T = S/(1+C).  The
catch nobody in the corpus has dealt with: tan(theta/2) blows up near
theta = pi, and our exponent sets (j to 16/25 and 16/27) cross it.  Fix:
HALF-TURN REDUCTION -- for C < 0 the rotation factors through
rot(theta - pi) and both output negations fold into FNMSUB opcodes
(u = re - T*im; im' = -S*u - im; re' = -T*im' - u, T = S/(1-C)) -- every
case is exactly 3 FMA-family instructions, all signs inside constants or
opcodes, |T| <= 0.944 across all 24 table entries.  Tables (C25T/C27T/C9T
_TS tangent + _HT half-turn flag) are compile-time long-double literals
generated AND fp-verified by build/tryout/gen_powp/gentw3.c (applies the
double-rounded shears to unit vectors against C/S, all <5e-16); the _HT
branch folds after unrolling (j is a literal).  Covers every twiddled
store in both engines' SoA pencils incl. the XCOL2 two-column path
(TWST2 routes through the same TWROT3).  16 twiddles/pencil at 25,
28 (W27+W9) at 27: pencil FP 404 -> 388 (-4.0%) and 436 -> 408 (-6.4%).
Depth per twiddle rises 2 -> 3 dependent FMAs -- a latency-for-port trade
covered by the 5/9 independent slot groups per stage; -DGENPWP_NOTW3
restores the r6 form for cross-arch races.  NOT bit-identical to r6
(~2 ulp vs ~1 per twiddle against a 1.5e-14/step budget): all gates
re-run, margins unchanged (numbers below).

**2. LIFTED DFT5 v-pair (ADOPTED from gen_batchlane gen_r7, their DFT5VPAIR
verbatim) -- BUILT, MEASURED, DEFAULT OFF.**  sin(2pi/5) = phi*sin(pi/5)
exactly, so the Winograd v-pair factors through u = sa - PHI*sb (-2 ops,
-2 live temps per DFT5; 10 DFT5s per 25-pencil).  On THEIR engine: -0.8..
-1.0%.  On MINE: lift-only vs r6 is a -0.3% wash (31.271 vs 31.355
min-of-mins), and ON TOP of the shears it LOSES ~0.3% -- shears-only beat
shears+lift 7/8 held-lease pairs at 25 (mins 30.868/30.904/30.964/30.994
vs 31.047/31.082/31.116/31.158).  Mechanism, recorded so nobody re-derives:
the lift serializes v1/v2 through u; gen_batchlane's stage-2 groups expose
2-4 independent DFT5s per pencil PLUS eight batch lanes of ILP per op,
while my in-place slot pencil has exactly 5 independent groups per stage
feeding depth-3 twiddle stores -- there is not enough surrounding ILP to
hide the extra serial link.  Same boundary class as sched-pressure (r3)
and two-column (r6): batchlane-engine wins are codelet-shape-specific.
Ships as -DGENPWP_LIFT5 opt-in; their r7 note says CLX's weaker FMA
throughput should widen the lift's win, so the knob is a real cross-arch
race candidate, not a corpse.

### Measured on the node (a80n0, ONE held slot lease per battery, soa forced
### GENPWP_PF=6 + cold race per invocation, control first; MKL same core)

A/B, r6 arithmetic (bin_c7 = -DGENPWP_NOLIFT5 -DGENPWP_NOTW3, verified
bit-identical to the r6 ship binary's chains) vs shipped r7:

| case | r6 arith (4-pair mins) | r7 ship | verdict |
|---|---|---|---|
| L=25 B=16 m=256 | 31.676-31.870 | 30.868-31.564 (t3 arm) | shears-only wins; -1.6% min-of-mins (30.87 vs 31.36 in the isolation round), 4/4 + 7/8 pair records above |
| L=27 B=16 m=200 | 44.378-45.120 | 43.075-44.293 | -1.0..-3.3%, 4/4 pairs (-2.9% min-of-mins; 27 is pure shears -- no DFT5 in its pencils) |

MKL 2022 same core, same windows: 25: 127.6, 27: 146.6 us (ship ~4.1x /
~3.4x in dev windows).  Ship battery (r7_final.sh, one lease, window ran
HOT -- sd 6-7%, treat as gate evidence not timing): graded chains
34.06 / 44.31 / 438.5 / 5211.3 at 25/27/50/100; B=1 40.02 / 50.55 / 421.2
at 25/27/50 (r6: 40.06 / 50.45 / 486.5-hot); new sizes 49 B=8 m=32 496.9,
81 B=2 m=16 3031.4, 121 m=8 14858.8, 125 m=8 16255.1.  50/100 and all
B=1/new-size paths are the interleaved engine -- code untouched, chain
rel_l2 identical to r6 (5.028e-14 / 4.181e-14 at 50/100).

Wallaby (dev host, ship build, GEN_RACE_NO_WISDOM): 25 B=16 23.878 (r6
source same-session control: 24.147, -1.1%), 27 B=16 32.572; B=1 31.35 /
74.79 (interleaved, unchanged; wallaby B=1 medians were noisy).

Gates, all eight sizes, node (r7_final.sh by hand; tryout's map-check leg
still has the '$W/c.bin' quoting bug): single call 3.6-5.0e-16 (tol 1e-12);
two-step m=2 1.467/1.624/2.361/2.721e-15 at 25/27/50/100 (tol 3e-14, >18x
margin -- the shears moved 25's from 1.422e-15 and 27's from 1.549e-15,
i.e. ~5% of a margin we have 18x of); graded chains 3.061e-14 (25 --
BETTER than r6's 3.145) / 3.147e-14 (27) / 5.028e-14 / 4.181e-14 at
1.05-1.7x honest anchors (tol 1e-10); new-size chains 5.6e-15-1.2e-14;
ALL bit-repeatable across independent runs; ship binary cmp-identical to
the measured A/B arms at both sizes.  Setup: cold 0.47-5.9 s (60 s
budget); warm wisdom unchanged ms-scale.  -Wall -Wextra: the 16
pre-existing unused-candidate warnings only, in ship AND both knob builds
(the shear tables carry __attribute__((unused)) for the NOTW3 build).
Round end: all gen_powp keys STRIPPED from results/wisdom_a80n0.json and
_wallaby.json (r2-r6 protocol; NOTE the wisdom file is {"host",...,
"entries":{...}} -- strip inside "entries", a flat-JSON filter silently
strips nothing).

### What did NOT work / went wrong, with the numbers

* **The lifted DFT5 as a default** (item 2 above): -0.3% wash alone,
  +0.3% on top of shears, 7/8 pairs against.  Kept as -DGENPWP_LIFT5.
* **Build-verification trap (cost ~20 minutes, worth recording):** piping
  a warning-heavy gcc through `head -30` SIGPIPE-killed the compiler
  mid-link, leaving the OLD binary in place with exit status masked by
  the pipeline; the stale r6 binary then ran with warm wisdom and printed
  a perfectly plausible 24.147 us.  The r5 rule (strings-marker check
  before trusting any rebuilt binary) caught it -- the marker is now a
  description-string token ("r7 3-shear twiddles").  Corollary: never
  pipe the build; capture stderr to a file and echo gcc's own $?.
* **First unpinned wallaby race read 30.6 us at 25** (vs 23.6-23.9
  pinned-window truth) -- the busy-core race-poisoning lesson (r5) on the
  dev host; wallaby numbers in this record are GEN_RACE_NO_WISDOM re-runs.
* 3-shear twiddles for the INTERLEAVED engine's CMULC: declined by
  arithmetic, not measured -- each shear mixes re/im lanes, which in
  interleaved layout costs a SWAP per shear (3 shuffles + 3 FMA vs the
  current 1 shuffle + 2 FMA).  Strictly worse on port 5; do not re-derive.

### Borrowed, plainly

- **Literature 08 6.3 (Gustafsson ARITH-24 lifting)**: the 3-shear
  rotation; the half-turn reduction with FNMSUB-folded negations and the
  split-complex-store placement are new here.  This is the round's
  literature spend, and per the corpus the first performant 3-shear
  twiddle implementation in the campaign.
- **gen_batchlane (gen_r7)**: the DFT5VPAIR lift, taken whole with their
  exact PHI/KL5 constants; opposite default verdict on my engine, boundary
  recorded above.  Their held-lease protocol remains under every number.
- **gen_pfa_large (gen_r7, their impl header, read mid-round)**: the
  two-axes-per-pass CLOSURE -- our shared shell already transforms two
  axes per DRAM pass through the L2 plane scratch, no cut removes the
  second pass (the x-stage couples 25 planes), and the real 100-lever is
  step-custody of the state, which is ipk1 (already my rank 0 at 100
  since r6; their side refutes ipk1-rank-first on THEIR engine, 4/5 to
  ipp1 -- per-engine verdicts, both stand, the race arbitrates).  Reading
  their header before coding saved this round from burning on brief item
  1; that is what the cumulative-context round is FOR.
- gen_pow2 appears to be spending brief item 2 (Garrido constant-per-site
  routing) at 32 (their genconst/cps_tables work in build/tryout); I did
  not duplicate it at 25/27 -- my SoA twiddles are already compile-time
  broadcast constants, so the routing claim has nothing left to buy here.

### Operation count

SoA pencil FP: 25: 404 -> 388 (16 twiddled stores at 3 ops, was 4);
27: 436 -> 408 (28 stores).  Twiddle depth 2 -> 3 dependent FMAs (latency
trade, ILP-covered).  With -DGENPWP_LIFT5: further -20 at 25 (10 DFT5s x
-2), measured net negative on ICL.  Interleaved engine, map ladders,
divides, pack/unpack: unchanged (192/218/434/968 scored, 534/850/1850/
1552 lite per line).  The shipped 50/100/B=1/new-size chain paths are
bit-identical to r6.

### What I would do next (ranked)

1. **XARCH races for the two new knobs**: GENPWP_LIFT5 (CLX may flip it --
   gen_batchlane's prediction) and GENPWP_NOTW3 (if any host's front end
   hates the depth-3 chain).  The wisdom race decides per host; nothing to
   hand-tune.
2. **L=100 is again the only trailed cell** (-1.0%): two-axes is closed
   (gen_pfa_large's accounting), ipk1 rank-0 confirmed by this round's
   fresh node race (margin 3.5%).  What remains is their r7 item 3 kind of
   territory (lean-size bypass) -- nothing for my scored cells; watch
   their record when it lands for anything 100-shaped.
3. **If a surprise-style draw lands p^k at batch % 8 == 0**, soa + the
   playoff serve it with the r7 shears for free (the tables cover W25/W27/
   W9 only -- a NEW p^k soa size would need its tables added to gentw3.c;
   the interleaved fallback needs nothing).
4. **The 27 x-pass remains ~60% of its step**: with twiddle ops now cut,
   the next 27 lever is memory-shape, not arithmetic -- and r6's
   two-column rejection plus gen_dense_prime's custody negatives say the
   OoO window already covers it.  I consider 27 saturated on this host;
   protect, don't chase.

## Round gen_r8

Standings into the round (r7 board): led 25 (30.882, 3.90x MKL) and 27
(43.966, 3.28x) outright; trailed gen_pfa_large at 50 by 0.27% (415.066 vs
413.958) and at 100 by 1.7% (4549.6 vs 4475.3) -- both inside the 3-6%
window spreads those cells print, i.e. shared-shell coin flips.  The r8
brief's new item is the static-analyzer tooling ("choose schedules with the
models; SCORE with the node"), and mid-round the monitor turned the PMU ON
(perf at /tmp/perf, paranoid=2, tools/pmu.sh -- announced via the new NOTICE
broadcast).  This round is therefore the tools round: every number below is
model + counter attribution, one model-refuted candidate ships as a
cross-arch knob, and the shipped chain arithmetic is BIT-IDENTICAL to r7 at
every size (verified below), so all r7 gate numbers carry over exactly.

### What was built / measured (one candidate, one attribution pass)

**1. n1_5 SPLIT-COMPLEX DFT5 (-DGENPWP_N15): BUILT, MODEL-REFUTED, NODE-
CONFIRMED REFUTED, DEFAULT OFF.**  The interleaved DFT5M has used FFTW's
n1_5 KIG factorization since r1 (B1 = KS5*(u1 + KIG*u2), B2 = -KS5*(u2 -
KIG*u1); the KS5 scale folds into the +-i cross FMAs).  Ported to the SoA
site layout it is 32 FP ops per DFT5 vs the Winograd split core's 36 --
"-10% FP per 25-pencil", and the PMU (below) says the 25 cell is ~68%
port-bound, so it looked like the round's headline.  The objdump/llvm-mca
audit killed it BEFORE the first lease slot: every folded output FMA is
destructive, and its addend (tp/tq) is live in the +- partner output, so
gcc must copy -- +72 zmm reg-reg movs per 25-pencil (34 -> 106), and ICL
does NOT eliminate vector movs at rename.  Port-0/5 uops per zy pencil:
442 (W5) -> 474 (N15), +7%.  Node A/B (one held lease, soa forced
GENPWP_PF=6, cold race GEN_RACE_NO_WISDOM=1, control first, 6 pairs):
**W5 31.445-31.749 vs N15 32.360-32.710 us/xform -- N15 loses 6/6 pairs,
+2.9% min-of-mins.**  PMU delta (34-vs-2-sample runs, per group-step):
N15 cycles +3.2%, instructions +4.6%, and port-2/3 (load) uops +26% --
at -march=native gcc turned the pressure into spill reloads rather than
movs; same tax, different port.  THE LESSON, stated for the whole panel:
**in split complex, "FP op count" is the wrong metric -- PORT UOPS is the
right one.**  The Winograd core's +- pairs ride NON-destructive
vaddpd/vsubpd, which is why it beats every fewer-ops DFT5 rewrite on ICL
(this also re-explains the r7 LIFT5 wash from the port side, and why the
r7 3-shear twiddles DID win: the shear asm shows the mov count constant,
563 -> 538 p05 uops at 27, a real cut).  Kept as a race knob for SPR,
whose Golden Cove cores DO eliminate vector movs (there n1_5 is 368 vs
442 p05 uops, -17% -- a genuine cross-arch candidate, like LIFT5-on-CLX).

**2. Attribution of all four scored cells with the r8 tools (llvm-mca
static counts + PMU counters once they went live).**  Method: gcc
-march=icelake-server -S with LLVM-MCA-BEGIN/END markers around each inner
loop; instruction-class counts from the marked regions; PMU by DELTA of a
34-sample and a 2-sample run (subtracts create/race/pack exactly).
Numbers, per pencil / per 4-lane line, shipped arithmetic:

| loop | FP ops | shuffles | spills+reloads | zmm movs | p05 uops |
|---|---|---|---|---|---|
| 25 zy pencil (p25 plain)  | 408 | 0 | 27+27 | 34 | 442 |
| 25 x pencil (p25m, map)   | 813 + 40 div-class | 0 | 35+35 | 88 | ~941 |
| 27 zy pencil (p27 plain)  | 408 | 0 | 55+55 | 130 | 538 |
| 27 x pencil (p27m, map)   | 840 + 45 div-class | 0 | 70+70 | 177 | ~1017 |
| 100 p1 z-line (ipk1)      | 968 | 569 (400 TRNC + 169 swaps) | 229+229 | 234 | ~886 crit |
| 100 p1 y-line / p2 x-line | 968 | 169 | 78-93 each way | 237-244 | ~687 crit |

PMU ground truth (a80n0 core 5, graded shapes, min-window):
- **L=25 soa**: 834k cycles/group-step at ~3.30 GHz effective; p0+p5
  dispatch = 1.13M uops/group-step (matches the static count exactly);
  max-port utilization **68%** -- the engine is two-thirds port-bound,
  one-third L3-stream slack.  FPDivider ~240 cyc/x-pencil vs ~470 port
  cyc: the r5 paired-div verdict stands, divider is hidden.
- **L=27 soa**: 1.189M cycles/group-step; utilization **60%**;
  l1d.replacement 155k lines/group-step = 9.9 MB = state x3 passes + c --
  the structural floor, nothing pathological.  27's extra slack vs 25 is
  the bigger arena (2.55 vs 2.05 MB) streaming L2<->L3; confirms the r7
  "saturated, protect" verdict with a number.
- **L=100 ipk1**: 16.7M cycles/volume-step; max-port (p5, carrying the
  TRNC shuffles + FMA share) utilization **45%**; cycle_activity.
  stalls_mem_any = 23% of cycles; core at 512-bit license-2 ~100% of
  cycles; and **LLC misses only ~9-14 MB/step** -- the c-flush custody
  WORKS: the naive 80 MB/step is mostly served from L3, DRAM traffic is
  c-dominated.  This closes gen_pfa_large's r7 next-list #1 (p1
  attribution) from the counter side, for both shared-shell engines: p1's
  z-subpass is shuffle-heavy (886 vs 687 port-critical cycles/line, 400
  of 569 shuffles are the TRNC granule transposes) but HIDDEN under the
  L3-stream slack -- a TRNC diet would buy little; the cell's floor is
  memory custody, which ipk1 already takes.  Do not chase 100 with
  arithmetic.

**3. Ship verification (the point of a final round): the r8 default build's
graded chains are cmp-BIT-IDENTICAL to the r7 ship binary at 25/27/50/100**
(forced same picks, same inputs), despite the D5SC refactor to Z-named
outputs -- same expression trees, only schedule; gcc even dropped 16 spill
lines at 25.  Full gate battery re-run on the node anyway: single call
3.604/3.725/4.336/4.522e-16 (tol 1e-12); two-step m=2 1.467/1.624/2.361/
2.721e-15 (tol 3e-14 -- the exact r7 values); graded chains 3.061/3.147/
5.028/4.181e-14 at 1.1-1.7x honest anchors (exact r7 values); repeatable
bit-identical across independent runs.  Dev-window graded mins this
session: 31.4-31.8 (25), 44.8 (27), 419.5 (50), 5081-hot (100); MKL same
core at 25: 120.9.  All gen_powp wisdom keys stripped from
results/wisdom_a80n0.json at round end under flock (r2-r7 protocol).

### What did NOT work / tool-state findings (record before anyone re-burns)

* **The n1_5 DFT5 as a default: +2.9%, 6/6 pairs against** (item 1) --
  refuted by model first, node second, both agreeing.  Total analyst cost
  before the lease: one -S compile and four greps.
* **uiCA is BROKEN on this filesystem**: ext/tools/uiCA has no instrData/
  module -- setup2.log shows the uops.info instructions.xml download timed
  out (no outbound net from the cluster); uiCA.py dies on import.
  **OSACA is also broken**: osaca-pkg/osaca/ contains only data/parser/
  semantics, no top-level modules.  Only llvm-mca of the three advertised
  static tools actually runs.  And llvm-mca's icelake-server model puts
  essentially ALL 512-bit FP on port 0 (vaddpd zmm RThroughput 1.0; a
  0-shuffle split-complex pencil shows P0=408, P5=61) -- Gold 6326 has TWO
  512-bit FMA pipes, so take mca's per-port table as a uop CENSUS, rebalance
  p0/p5 by hand, and treat only RELATIVE comparisons as meaningful.  Now
  that /tmp/perf is live, counters are ground truth (my measured port sums
  matched the hand-rebalanced census within 2% at 25 -- the census method
  is sound).
* **The build/tryout file sweeper**: mid-session, every executable and .sh
  in build/tryout/gen_powp/ vanished (bin, bin_ship, the r3-r7 battery
  scripts, .err logs) while data .bin files survived.  Nothing of mine
  depended on them, but a binary you measured an hour ago may not exist
  now -- rebuild before every A/B (the r5 strings-marker rule now has a
  second reason), and keep battery scripts out of build/tryout (mine now
  pipes over ssh stdin from /tmp).
* tryout.sh's remote map-check leg still dies on the '$W/c.bin' quoting
  bug (seventh round); the gates above were run by hand on the node.
  reserve.sh --status still needs the slurm PATH shim on wallaby, and
  tryout.sh now FAILS SILENTLY ("no live reservation") when invoked
  without it -- export PATH=/opt/software/slurm-19.05.8.1/bin:$PATH first.
* **One fresh data point on the r5/r6 race-poisoning mode**: a late-session
  tryout cold race (busy neighbors) stored l25-ip0 as a "tie" (ip0 trial
  46.77 us/vol, soa playoff margin -0.7%) and the graded chain shipped
  41.1 us where soa's same-day quiet number is 31.4-31.8 -- the r6 playoff
  narrows the window but does not close it under real neighbor contention
  (soa's ~6 MB arena is squeezed harder than ip's in-place set; under
  contention the race's flip is arguably HONEST for that environment).
  Key stripped per protocol; the monitor's acquire-all quiet window is the
  designed arbiter and has picked soa on every scored board since r6.  Do
  NOT "fix" this by pre-storing a soa verdict -- that would game the race
  the whole campaign is built on.

### Borrowed, plainly

- **The r8 tools themselves** (monitor's llvm-mca install + mid-round PMU
  enablement): this round's entire method.  The marked-region mca recipe is
  TOOLS.md's; the 2-vs-34-sample PMU delta that subtracts create()/pack is
  mine, take it.
- **gen_pfa_large (gen_r7)**: their next-list #1 (p1 attribution at 100)
  set item 2's target; answered here with counters for both our engines.
- **gen_batchlane (gen_r7/r8)**: their lifted DFT5 and their r8 decision to
  NOT spend the analyzers on bit-identical known-good cells -- the port-uop
  census now explains their lift's engine-specificity mechanistically (it
  is not "ILP", it is the mov/spill tax of destructive FMAs vs surrounding
  register pressure).  Their DFT11/22/33/44/55 build is the round's real
  class win; nothing of it applies to p^k, but their "verified slot tables
  generated in Python before C" discipline is what my gentw3.c already
  does for twiddles -- convergent, worth naming.
- **gen_pfa_small (gen_r8)**: their rotation split chain makes B%8
  remainders fast at generic coprime sizes; NOT adopted here because my
  interleaved B=1 paths are already raced engines (40.0 us at 25 B=1 vs
  their-class lane-replication disasters), and my scored batches are
  multiples of 8.  Noted so the planner routes p^k-at-B<8 draws to my
  interleaved families with confidence, not to a port of their split chain.

### Operation count

Unchanged everywhere; shipped chains bit-identical to r7 (192/218/434/968
scored, 534/850/1850/1552 lite FMA-port vector ops per line; soa 388/408
per pencil in the record-lineage count).  Census correction from the asm
audit: the "34-op" Winograd split DFT5 EMITS 36 (the +-cross adds), so the
25 zy pencil is 408 emitted FP, not 388 -- lineage counts since r2
undercount by 2/DFT5; keeping both bases explicit here so nobody
"optimizes" the phantom 20 ops again.  With -DGENPWP_N15: 408 -> 368
emitted FP but +72 zmm movs (ICL net +32 p05 uops, +2.9% measured;
SPR-candidate only).

### What I would do next (ranked)

1. **Nothing on ICL kernels.**  25: 68% port-bound at the op floor the r7
   shears set; 27: 60%, memory-shaped, saturated (three independent
   verdicts now); 50/100: shared-shell coin flips sitting on L3/DRAM
   custody floors the c-bypass families already take.  The library-scoring
   risk is race/window luck, and the playoff machinery is the protection.
2. **XARCH: race the three DFT5 forms per host** -- W5 (ICL default),
   LIFT5 (CLX candidate, gen_batchlane's prediction), N15 (SPR candidate,
   Golden Cove mov elimination).  All three are compiled-in knobs; the
   wisdom race decides; nobody should hand-tune.
3. **If any future round re-opens arithmetic**: audit with the port-uop
   census FIRST (mca marked regions + the grep set in this section), and
   count destructive-FMA copy pressure before counting FP ops.
4. **PMU follow-ups for whoever owns the weak cells**: at 100 the
   interesting number is LLC-miss bytes/step (~9-14 MB measured vs 32 MB
   state+c) -- if a future idea claims a traffic cut, demand its
   longest_lat_cache.miss delta, not its DRAM accounting.

## Round gen_r9

Standings into the round (r8 board): **L=25 shipped 41.025 where 31.4 was
available (-29.8%)** — and the monitor's PMU audit proved it was NOT code:
r7 and r8 binaries are counter-identical; the scoring window's own cold race
banked `l25-ip0` as a -0.97% "tie" from a contended window.  Worse, that
poisoned entry was STILL LIVE in wisdom_a80n0.json at r9 start
(`gen_powp/chain6/L25/B16#6bb92654 -> l25-ip0`), and impl_9 carried the r8
source unchanged — same candidate names, same sig, same key — so an
untouched entry would have WARM-HIT it and replayed the 41 µs regression in
the r9 scoring window.  27: led (43.357).  50/100: shared-shell coin flips
(421.7 vs gen_pfa_large 416.6; 4596.6 vs 4567.7).  The r9 brief's avenue 1
(bank the picks, noise-gated) names this entry's wound as the round's
cheapest big win, and the NOTICE says the node is queued-busy (both Ice Lake
nodes held by other users; our hold first in queue, worst case Aug 27) —
tryout node runs impossible, so this is the pure-logic round avenue 1 was
advertised to be.  Kernel arithmetic is untouched; only tune() changes.

### What was built (three mechanisms, one file, all in tune())

**1. WISDOM TAG chain6 → chain7.**  The immediate fire: the r8 window's
poisoned l25-ip0 verdict (and every other chain6 short-horizon verdict) can
never replay.  This alone recovers the L=25 cell if the r9 scoring race runs
clean — everything else makes "runs clean" much more likely and makes the
result durable.

**2. ADAPTIVE NOISE-GATED TRIALS.**  Every deciding measurement now runs
until its evidence is demonstrated or provably sufficient, capped:
- *Floor-stability metric* `pwp_spread3`: spread of an arm's 3 smallest
  round times, (s3-s1)/s1 — min-of-mins methodology (gen_batchlane r4's
  held-lease protocol) applied to the race's own evidence.  <3 samples =
  1.0 (undemonstrated).
- *Base race*: the r4 min-of-rounds loop (4 rounds) now records per-round
  times (the trial body factored into `pwp_trial_once`, measurement
  unchanged); afterwards the two LEADERS buy up to 6 extra rounds until both
  floors are stable (leaders recomputed each round — a settling floor can
  change who leads; also-rans' exact times decide nothing).
- *Both playoffs* (soa-vs-interleaved and rank-0 challenger): 3 rounds
  minimum as before, now extended up to 9 until both arms' floors are
  stable OR the margin dwarfs the noise (margin >= 2Q — more rounds cannot
  flip the verdict).  This is the r6 playoff's missing half: the r6 fix made
  the comparison fair in shape; r9 makes it run until it is fair in fact.
- Q of the deciding measurement (playoff arms if one ran, else the leaders'
  base trials) feeds the store gate.  Tolerance GENPWP_NQTOL, default 5%.

**3. QUALITY-MARKED BANKED VERDICTS.**  tight = (Q <= tol) OR (margin >=
2Q).  A tight verdict stores as a plain candidate name — BANKED, honored
until the tag/sig machinery re-keys it (r10's scoring warm-hits r9's tight
verdict: cross-round determinism, the brief's "recover 0.1265
deterministically").  A verdict still noisy after the extension stores as
`name~q<pct>@<unixtime>` — PROVISIONAL: the lookup honors it only within
GENPWP_NQHORIZON (default 1800 s), long enough to pin the driver's two
repeatability processes (seconds apart — the reason not-storing was never an
option; an unpinned noisy race would flip between processes and flag NOT
REPEATABLE on the non-bit-identical soa family), short enough that the next
scoring window RE-RACES a coin flip instead of replaying it.  "Re-race,
never trust, a noisy trial" — the brief's words, implemented as an expiry.
The r2–r8 round-end wisdom-strip protocol RETIRES for scoring-window
verdicts: banking tight verdicts is now the design.  (My own future NODE dev
sessions: I will still strip my dev-session keys at round end — a leased
core next to active neighbors can produce a SUSTAINED-bias window that reads
tight-but-wrong, my r8 "arguably honest for that environment" observation;
the scoring window holds all 24 slots and is the arbiter.  The spread gate
catches fluctuating noise, not sustained bias; nothing from a dev host can.)

### Measured on WALLABY (SPR Gold 6448Y login host, 51 users, taskset core
### 108 — the node was queued-busy ALL ROUND; these are correctness +
### determinism numbers and cross-arch signal, NOT scored numbers)

Determinism (the brief's acceptance test), 5 consecutive create() cycles:
- **Cold** (GEN_RACE_REFRESH=1, race every time): 25: 5/5 `l25-soa`
  (margins +24.9..+32.1%, Q 0.1–12%); 27: 5/5 `l27-soa` (+19.5..+22.9%);
  50: 5/5 `l50-ipp0` (rank rule, margins -0.7..-1.9% inside hysteresis);
  100: 4/5 `l100-ipp1`, one `l100-ipp0` at +3.3%/Q 3.1% — an honest
  >hysteresis window verdict between two near-tied ipp variants on a busy
  login host (wallaby has always inverted picks vs the node; r1 record).
- **Deployment path** (wisdom on): 5/5 identical picks, warm setup
  2.0–4.5 ms (50 ms budget) — identical BY CONSTRUCTION once cycle 1 banks;
  this is what makes the acceptance test hold on any host.
- **Provisional lifecycle** (forced via GENPWP_NQTOL=0.0001 at 50, where
  margins ~1% cannot dwarf noise): stored `l50-ipp0~q0@1787711924`; warm
  hit 2 ms inside horizon (repeatability pinned); GENPWP_NQHORIZON=1 +
  sleep 2 → create() re-raced (1.26 s) and re-stored tight.  At 25 the
  same forcing could NOT produce a noisy verdict: playoff self-extended to
  6 rounds and margin 27.6% >= 2×8.7% kept it tight — soa's dominance is
  never bankable-away by the gate, which is the point.

Gates, all eight sizes, wallaby (node re-run belongs to the monitor's
scoring pass): single call 3.60–5.04e-16 (tol 1e-12); two-step m=2
**1.467/1.624/2.361/2.721e-15** at 25/27/50/100 — the EXACT r7/r8 values
(shipped chain arithmetic bit-identical to r8); graded chains
3.061/3.147/5.028/4.181e-14 (exact r8 values) at 1.09–1.73x honest anchors;
lite sizes m=2 2.0–2.9e-15; two-process repeatability cmp-identical at
25/27 (soa picks, wisdom-pinned).  Setup: cold 0.31–0.56 s (25/27),
1.1–2.1 s (50/100), worst 4.64 s (121) [60 s budget]; the adaptive
extensions cost nothing when windows are clean (3 playoff rounds, 0 extra
base rounds — observed in most runs above).  Dev timings for the record
(SPR, quiet-ish): 24.5 / 33.6 / 306 / 2990 µs graded; B=1 30.0 / 38.2 /
305 µs at 25/27/50.  -Wall -Wextra: the 16 pre-existing unused-candidate
warnings only.  Build marker token: "r9 noise-gated banked verdicts".

### What did NOT work / boundaries, with the numbers

* **The spread gate cannot catch sustained bias** (stated above, so nobody
  oversells this mechanism): a window that is consistently 30% slow for one
  arm reads tight.  Defenses layered instead: the scoring window's full
  quiet, the dev-key strip discipline, and the horizon on anything that
  wobbled.  If a tight-but-wrong verdict ever banks in a scoring window,
  GEN_RACE_REFRESH (monitor-side) or a tag bump remains the manual purge.
* **L=100 cold determinism is 4/5 on wallaby**, and that is CORRECT
  behavior: run 4's ipp0-by-3.3% beat both the hysteresis band (3%) and the
  2Q rule (Q 3.1%).  Two genuinely near-tied candidates on a drifting host
  produce honest alternating verdicts; banking exists precisely so one
  verdict is then pinned.  Do not "fix" this with a wider hysteresis — at
  the node the r6 5/5 evidence separates ipk1 from ipp1; wallaby's ipp0/
  ipp1 tie is a different host's truth.
* Avenue dispositions (whiteboard, node-dependent, recorded so r10 spends
  zero time rediscovering): **Avenue 2** (two-axes y×z fusion at 100/50) is
  re-opened by the audit's counters (L=100: 2.34G l1d.replacement lines =
  ~4x algorithmic minimum, p0+p5 0.82/cyc; L=50: 77 GB into L2, 1.07/cyc)
  DESPITE gen_pfa_large's r7 shell-level closure — the counters say the
  traffic is there even with the L2 plane scratch; the success metric is
  l1d.replacement ~2x down, not wall time.  Full-round scope, needs the
  node, coordinate with gen_pfa_large so only one of us burns it.
  **Avenue 3**: my cells' signatures are already in the r8 record (25: 68%
  port-bound — near done; 27: 60%, memory-shaped; 50/100: traffic, see
  avenue 2).  **Avenue 4** (port-1 co-issue): struck at the whiteboard for
  the soa engine — the map ladder is on the x-pass critical path, twiddles
  are compile-time broadcast constants, and the graded 25/27 batches are
  B=16 (no B%8 remainder volumes to route as ymm side work).  The real
  port-1-adjacent play is the audit's **4-lane SoA variant at L=50 B=4**
  (256-bit FP dispatches p0+p1, no 512-bit license, and it unlocks
  batch-lane layout where the 8-lane form cannot run) — a new engine
  build, node-dependent, queued as the r10 candidate for the weakest
  shared cell.

### Borrowed, plainly

- **PMU_AUDIT.md (monitor)**: the diagnosis this round implements — the
  audit's counter-identity proof of r7≡r8 and the "noise-gated storage"
  prescription are its avenue 1, verbatim.  The provisional-with-expiry
  mechanism (pin repeatability now, re-race next window) and the
  margin>=2Q sufficiency escape are new here.
- **gen_batchlane (gen_r4)**: min-of-mins floor methodology, now applied
  by the race to its own rounds (pwp_spread3).
- **gen_dense_prime / gen_pfa_small (gen_r9, read before starting)**: the
  node-is-down round shape — static validation + wallaby correctness,
  "pending ICX" tags on anything timing-shaped; also their avenue-1
  dispositions ("no internal picks to bank") confirmed this entry is where
  avenue 1 lives: mine is the panel's biggest internal tuner (8–13
  candidates, two playoff flavours).
- **gen_race (gen_r2 lib)**: unchanged wisdom machinery; the quality marker
  rides inside the stored winner NAME so no gen_race API change was needed
  (their file is not mine to edit) — gen_race owner: if you standardize a
  quality field, I will migrate off the ~suffix.

### Operation count

Unchanged everywhere (192/218/434/968 scored, 534/850/1850/1552 lite
FMA-port vector ops per line; soa 388/408 record-lineage per pencil).  The
shipped chains are bit-identical to r8/r7 at every size (two-step and
graded drifts reproduce r8 to the last digit).  tune() cost: +0–6 base
rounds and +0–6 playoff rounds, only when the window is noisy; observed
cold setups 0.31–4.64 s vs the 60 s budget.

### What I would do next (ranked)

1. **When the node hold lands: nothing to re-tune by hand.**  The scoring
   run's own chain7 cold race (now noise-gated) banks the verdicts; verify
   on the r9 board that L=25 recovered ~31.4 and that wisdom_a80n0.json
   holds plain-name (tight) gen_powp/chain7 entries afterwards.  If any
   entry is provisional (~q marker), the window was noisy — tell the
   monitor, and the r10 run re-races it by design.
2. **4-lane SoA at L=50 B=4** (audit avenues 2+4 combined for my weakest
   shared cell) — engine build + node A/B; success metric l2_lines_in
   down AND wall time, champion signature p0+p1+p5.
3. **Two-axes fusion at 100**: coordinate with gen_pfa_large FIRST (their
   r9 record was not yet written this round); the counters re-opened what
   their r7 header closed, and only one of us should burn the round on it.
4. **XARCH**: the three DFT5 knobs (W5/LIFT5/N15) and NOTW3/XCOL2 remain
   compiled-in race candidates; with banking live, each host's first quiet
   race now settles its pick permanently — check the CLX/SPR advisories
   pick up the banked verdicts rather than re-rolling.

## Round gen_r10

Standings into the round (r9 board, scored on a NEW node — a81n2, same Gold
6326 SKU as a80n0): led 25 (31.054 — the r9 banking design recovered the r8
wound exactly as intended), 27 (43.607), 50 (417.400 vs gen_pfa_large
437.059); **L=100 shipped 4807.4 where ~4550 was available (-6.3% vs
gen_pfa_large's ipp1 at 4521.7, shared shell, run spread 6.5% vs their
2.0%)**.  Cause, read straight out of wisdom_a81n2.json before touching any
code: `gen_powp/chain7/L100/B1 -> l100-ipk1, tie=1, us=4699.06,
margin=-0.01596` — the scoring window's own challenger playoff put ipp1
AHEAD of rank-0 ipk1 by 1.6% (a tight, banked verdict), and the 3% rank
hysteresis handed the slot back to ipk1 anyway.  The rank prior was my r6
a80n0 evidence (5/5 held-lease pairs); it is HOST evidence, and it was
imported onto a different host against that host's own measurement.  Worse,
the banked plain-name verdict would have WARM-HIT in the r10 scoring window
and replayed the loss (banking working exactly as designed, on a wrong
premise).  Kernel arithmetic is untouched this round (bit-identical to
r7/r8/r9 — the historical gate drifts reproduce to the last digit); all
four changes are in tune().

### What was built (four mechanisms, all in tune(); marker "r10
### playoff-authoritative picks")

**1. A DECIDED challenger playoff is AUTHORITATIVE between its two arms.**
When the interleaved playoff reaches its noise-gated exit (Q <= tol or
margin >= 2Q), the loser can no longer take the slot back through the 3%
rank hysteresis (implemented as: if `best == po_w && pick == po_l`, pick
reverts to the playoff winner).  Rank priors encode one host's held-lease
history; a decided playoff is the RUNNING host's long-horizon measurement
and outranks it — per-host truth is what the whole race layer is for.
Third candidates, undecided (still-noisy after 9 rounds) playoffs, and the
soa family's deliberate 3% hurdle are all unchanged.

**2. The challenger playoff runs even when rank 0 IS the trial best.**  The
arms are then the two trial LEADERS (rank-0/best vs runner-up-by-trial).
Without this, 3/5 cold races on a81n2 put ipk1 first in the contaminated
short trials (see item 3) and installed it with no long-horizon check at
all — the override of item 1 never even got evidence to act on.

**3. Trial/playoff CACHE-REGIME FIDELITY at volumes past L3 scale (> 8 MiB):
base trials warm 3 steps (was 1); playoff arms warm 6 (was 2) and time
PS=24 steps (was 12).**  Mechanism, measured before/after on a81n2: the
candidates alternate on the SAME tout/tcf arenas, so a c-custody arm (ipk1
CLFLUSHOPTs the entire 16 MB c stream every step) hands its successor a
c-cold cache hierarchy — and at 100 the successor is ALWAYS ipp1 (table
order), so the measurement systematically charged ipp1 for ipk1's flushes.
With the r9 shape the in-race playoff read ipk1 4775-5151 vs ipp1
5363/4805 (ipk1 "winning") in the same session where held-lease GRADED
m=64 pairs read the opposite: **ipp1 wins 5/6 pairs, floors 4538-4619 vs
ipk1's 4712-4780 (-2.2..-4.9%)** — the graded shape is the ground truth the
race exists to predict.  One warm step cannot re-converge 16 MB of c; six
steps at the arm's own regime can.  Small volumes keep the r6/r9 shape
exactly (their arenas are L2/L3-resident; nothing to converge).

**4. Wisdom tag chain7 -> chain8**, so the r9 window's banked l100-ipk1
hysteresis tie can never replay.  (gen_race independently bumped its own
tags enggate9->enggate10 etc. this round; no API change, my GEN_RACE_LIB_ONLY
include is unaffected.)

### Measured on the node (a81n2 — the r9/r10 scoring host; the reservation
### landed this round, first node access since r8)

The decisive A/B (held lease, one core, GENPWP_PF forced, cold race, full
graded m=64 chains, samples 5-6, order alternated across pairs), run in a
quiet-ish early window: **ipp1 4538.2/4554.3/4561.7/4609.5/4619.2/4871.5 vs
ipk1 4712.1/4746.6/4771.2/4772.8/4774.1/4779.7 — ipp1 wins 5/6 pairs,
min-of-mins 4538 vs 4712 (-3.7%)**.  This matches the r9 board (their ipp1
4521.7) and the banked playoff margin's direction; it is the OPPOSITE of
a80n0's r6 5/5.  Window boundary, recorded honestly: later in the session
the node went busy (12 implementers active, load ~6.6) and the same graded
pairs compressed to a tie (ipp1 4742/4759 vs ipk1 4795/4769) — contention
armor compresses the gap here rather than widening ipk1's lead as on a80n0.
The design outcome is NOT "ipp1 is hardcoded": the quiet scoring window
races cold under chain8, the faithful playoff decides on that window's
long-horizon evidence, and banking pins it.

Verification battery (all on a81n2, mixed-to-heavy contention — timing
numbers are gate evidence, not floors):
- Gates: single call 3.604/3.725/4.336/4.522e-16 at 25/27/50/100 (tol
  1e-12); two-step m=2 **1.467/1.624/2.361/2.721e-15** (tol 3e-14 — the
  exact r7/r8/r9 values, arithmetic bit-identical); graded chains
  3.061/3.147/5.028/4.181e-14 at 1.09-1.73x honest anchors (exact r9
  values); all four sizes repeatable (cmp-identical across independent
  processes, m=2 and graded m).  Lite sizes 49/81/121/125: m=2 gates
  2.017/2.869/2.687/2.861e-15, cold setups 2.2-8.5 s under contention.
- Pick determinism (3 consecutive GEN_RACE_REFRESH cycles per size, busy
  window): 25 3/3 l25-soa (margins +32.8..+37.4%, Q 0.1-0.2%); 27 3/3
  l27-soa (+19.0..+21.4%); 50 3/3 l50-ipp1 (rank rule, runner inside the
  accepted <=3% band at -1.2..-2.1%, Q 0.0%).  100 in the busy window: 5/5
  ipk1 (that window's honest truth — see boundary above).
- B=1 interleaved path exercises the new leader playoff (l25-ip0 45.47 vs
  l25-ip1 45.31, decided; near-tie semantics unchanged) and passes its
  m=256 chain gate.
- Dev-session chain8 keys STRIPPED from wisdom_a81n2.json under flock at
  session end (r9 protocol for dev-window verdicts: a contended window can
  read tight-but-biased; the scoring window's cold race is the arbiter).
  The r9 chain7 scoring verdicts were left untouched (dead keys to the r10
  binary; scoring-window property).
- -Wall -Wextra: exactly the 16 pre-existing unused-candidate warnings.
- MKL same-core references during the battery: 121.3 (25), 147.5 (27) —
  ~3.9x / ~3.4x in dev windows.

### What did NOT work / boundaries, with the numbers

* **Item 1 alone was insufficient** (my first cut): with only the
  authoritative override, 3/5 cold races at 100 never ran a playoff (ipk1
  was trial best) and the 2/5 that did read ipk1 AHEAD (5151 vs 5363; 4801
  vs 4805) — the override obediently installed the loser of the graded
  truth.  The fidelity fixes (items 2+3) are what make item 1 mean
  something.  Lesson, stated for the panel: **an authoritative decision
  rule is only as good as the measurement it authorizes; fix the
  measurement's cache-regime fidelity BEFORE strengthening the rule.**
* **The playoff cannot out-measure its window.**  In the heavy-contention
  window even the graded pairs tie (4742/4759 vs 4795/4769); no create()-
  time machinery can recover a quiet-window verdict from a busy window.
  Layered defenses unchanged: quiet scoring window + chain8 cold race +
  banking + dev-key strip.
* **Cold create at 100 hit 37 s under peak contention** (tryout, 12
  implementers active; quiet-window estimate ~6 s, r9 was ~2 s).  Still
  inside the 60 s budget, but the adaptive extensions + longer playoff eat
  margin under load — if a future round adds candidates at 100, trim the
  also-ran pool (f0/fr/frw have never won a 100 race in ten rounds) before
  extending horizons further.
* tryout.sh's remote map-check leg still dies on the '$W/c.bin' quoting bug
  (eighth round); gates above were run by hand on the node.  reserve.sh
  still needs the slurm PATH shim on wallaby.

### Borrowed, plainly

- **gen_pfa_large (gen_r9)**: their determinism lesson — "a noise gate
  keyed only to in-window spread cannot deliver determinism; it needs a
  margin floor calibrated to between-window drift and a confirmation on
  fresh evidence" — is why the override fires only on DECIDED playoffs and
  why rank priors were not simply swapped (their ranks already had ipp1
  first at 100; on my engine the same truth arrived via wisdom forensics).
  Their r9 upset rule (6% floor + fresh-evidence confirmation) is the
  stricter cousin of my item 1; mine can be looser because the playoff
  itself is already the long-horizon confirmation.
- **The monitor's r9 board + wisdom_a81n2.json**: the entire diagnosis was
  read out of the banked verdict record — banking picks (r9's design)
  turned a mystery regression into a one-line root cause.  That is the
  strongest argument yet for the avenue-1 machinery.
- The cache-regime-contamination mechanism (item 3) is new here; any entry
  racing a cache-custody candidate (NT stores, CLFLUSHOPT, NTA prefetch)
  against a caching one on SHARED trial buffers has the same exposure —
  gen_pfa_large's ipq/ipk/iqn pool at 40/50/100 most directly.  Take the
  warm-step fix.

### Operation count

Unchanged everywhere (192/218/434/968 scored, 534/850/1850/1552 lite
FMA-port vector ops per line; soa 388/408 record-lineage per pencil).
Shipped chains bit-identical to r9/r8/r7 at every size.  tune() cost: the
leader playoff now runs at essentially every interleaved cold race (+0.3 s
quiet at 50, +2-3 s quiet at 100); warm wisdom path unchanged (ms-scale).

### What I would do next (ranked)

1. **Verify on the r10 board** that L=100 recovered to ~4550 (parity with
   gen_pfa_large) and that wisdom_a81n2.json holds a plain-name chain8
   l100-ipp1 (or a genuinely-measured ipk1 if the quiet window says so —
   either is the design working; a provisional ~q marker means the scoring
   window was noisy, tell the monitor).
2. **4-lane SoA at L=50 B=4** (r9 avenue 2+4 hybrid) remains the one
   unexplored engine idea for my cells, and gen_pfa_large's r9 portcal3
   result (256-bit FP steals 512-bit slots 1:1 on SPR, predicted same on
   ICL) has KILLED its port-1 co-issue rationale — the only remaining
   angle is the batch-lane layout itself (zero shuffles vs TRNC) at half
   vector width, which the r8 port census prices as a net loss (2x ops vs
   +569 shuffles/line saved).  Consider it closed unless portcal3-on-ICL
   surprises.
3. **Two-axes fusion at 100**: the counters (2.34G l1d lines, 0.82/cyc
   p0+p5) still say traffic headroom exists, but two shell-level closures
   (gen_pfa_large r7 accounting, my r8 attribution: the z-subpass shuffle
   cost hides under L3-stream slack) stand against it.  Only worth a round
   if someone produces a paper schedule with fewer L1 round trips, not
   just fewer passes.
4. **If any host ever shows the playoff and the graded shape disagreeing
   AFTER item 3** (check by forcing GENPWP_PF pairs in the same window):
   the next fidelity lever is running the playoff arms on SEPARATE c
   buffers (one per arm), which removes the cross-arm custody coupling
   entirely at the cost of 16 MB more race arena.

## Round gen_r11 (all hands on L=100)

Standings into the round (r10 board, a81n2): led 25 (30.863), 27 (43.723 —
gen_race's warm-hit of my own pick edged me by noise), 50 (417.201);
L=100 4659.795 at 10.9% run spread vs gen_pfa_large's 4529.429 — the r10
playoff/banking machinery landed the right family (ipp1-class) and the
residual gap is window noise on a shared-shell engine.  The r11 brief: all
hands on L=100, counters mandatory before/after, and settle the
uop-saturation disagreement with data.  Node: a80n0 (reservation 438881).

### Counter protocol FIRST — and this engine's answer to the open question

Method: forced-ipp1 (GENPWP_PF=10, GEN_RACE_NO_WISDOM=1), whole-process
counters at --samples 2 and --samples 10, SUBTRACT — a 512-chain-step delta
with create()/gates/warmup cancelled out (ADOPTED from gen_pfa_large's r11
768-step delta method).  tools/pmu.sh for ports, /tmp/perf directly for the
dtlb set.  Per 512 steps, r10-lineage binary (baseline) vs r11 ship:

| counter (512-step delta)        | r10 baseline | r11 ship | delta |
|---|---|---|---|
| cycles                          | 10.60G (window-elevated) | 7.87G (quiet) | (windows differ; A/B below is the time evidence) |
| p0 / p1 / p5                    | 3.246G / 0.020G / 3.841G | 3.232G / 0.023G / 3.826G | flat |
| p2_3 / p4_9                     | 2.834G / 1.030G | 2.838G / 1.041G | flat |
| l1d.replacement                 | 1.031G (2.02M lines/step) | 1.031G | flat |
| dtlb_load_misses.walk_completed | 1.282M (~2.5K/step) | 26.8K | **-98%** |
| dtlb_load_misses.walk_active    | 83.5M cyc (~163K/step, ~0.8%) | 0.96M | **-99%** |
| dtlb_load_misses.stlb_hit       | 207.7M (~405K lookups/step) | 1.95M | **-99%** |

**Uop-saturation verdict for this engine at L=100: total vector dispatch
(p0+p1+p5+p2_3+p4_9) = 10.97G / 10.60G cycles = 1.04 uops/cycle
(quiet-window after: 1.39), p0+p5 0.67–0.90 — NOWHERE NEAR the ~2.1
all-port cap.**  This corroborates the audit's "0.82 = headroom" reading
and gen_pfa_large's r11 1.20 step-average on the shared shell, and
coexists with gen_bluestein's 2.03-at-cap: the cap is real (their engine
hits it) but PER-ENGINE — this shell at 100 is traffic/DRAM-bound, so uop
deletion does not pay here and effective bandwidth is the only lever.

### What was built: THP re-home of the chain-hot streams (ADOPTED from
### gen_layout gen_r11, their adoption recipe for this entry, verbatim)

gen_layout measured the mechanism: a80n0 runs THP=madvise on kernel 5.15,
so the driver's posix_memalign buffers get ZERO huge pages at any size (no
MADV_COLLAPSE before 6.1; khugepaged irrelevant in-process) — the L=100
chain streams state (16 MB) + c (16 MB) through 4K pages every step.  My
baseline numbers above price that at ~2.5K completed walks + 163K
walk-active cycles + 405K STLB-hit lookups per step.  The change, all in
tune()/fft3d_chain (kernel arithmetic untouched):

1. At vbytes >= 8 MiB (100 graded; 81/121/125 lite; 25/27/50 gated OFF —
   their per-volume working sets are STLB-covered) the plan owns a
   gl_map_huge arena (2 MiB pages, prefaulted at create) with two buffers
   STV/CV.  fft3d_chain gates PER CALL on gl_thp_bytes(out) < 50%
   (gen_layout's new primitive — verify, don't assume; a THP=always host
   disables the whole mechanism for free): steps then run with the state
   in STV, c staged once per volume into CV (one 16 MB copy vs m=64
   4K-paged re-reads), and the LAST write of every volume exits to the
   driver's out directly — trailing map_span for the deferred families,
   last-step dst for the plain ones.  ZERO extra state copies; the state
   only changes address.  Values BIT-IDENTICAL (memcpy is exact, same
   arithmetic in the same order): proven by cmp of full m=64 chain + single
   outputs, ship vs GENPWP_NOREHOME control, and cv vs GENPWP_NOCV.
2. TRIAL-REGIME FIDELITY (the r10 lesson applied forward): at re-home
   sizes the race's tout/tcf ARE the arena buffers, so every trial and
   playoff runs in the cache+TLB regime the graded chain now has — and
   trial 4K phases become gl_arena_take-deterministic (576 B stagger)
   instead of malloc luck.  Wisdom tag chain8 -> chain9 (a chain8 verdict
   was measured in the 4K regime and must re-race).  The arena is
   allocated BEFORE the wisdom lookup so a warm-hit plan re-homes exactly
   like the cold-raced plan that banked the verdict.
3. Knobs: GENPWP_NOREHOME (A/B control arm, also kills the arena),
   GENPWP_NOCV (state-only re-home).  soa (25/27), execute, and the dense
   fallback untouched.

### Measured on the node (a80n0, held-lease same-core alternating pairs,
### forced ipp1, graded L=100 B=1 m=64; sd 0.04–0.2% windows)

Ship vs GENPWP_NOREHOME control: **4590.3/4593.1/4595.8/4594.1/4594.4 vs
4698.2/4686.3/4682.7/4678.5/4693.6 — 5/5 clean pairs to the re-home,
-1.8..-2.3% (~-2.0%), min-of-mins 4590.3 vs 4678.5**.  A sixth pair's
window went busy (ship sample sd 5.45%) and is discarded as noise, recorded
honestly.  cv-vs-NOCV (is the once-per-chain c copy worth it): cv
4520.8/4422.2/4417.4 vs nocv 4704.4/4451.0/4500.5 — **cv wins 3/3**
(c is half the 4K-page stream; the 16 MB copy at ~0.35% of the call buys
back more than it costs, same verdict as gen_layout's demo).  cv stays the
default.  Full-pool cold-race runs in quiet windows: **4439.5 and 4417.4
us/xform** session bests (MKL same session 7813–7893); the r10-lineage
baseline tryout read 4844.2 in its own window.  Note the time win (~2%)
EXCEEDS the raw walk-active accounting (~0.8%): the 405K/step STLB-hit
lookups and 4K-boundary prefetch-stream breaks do not show in walk_active
— the inverse of gen_layout's finding on their slower engine (their win
was SMALLER than the counters); recorded so the next adopter prices both
directions.

Off-case and parity: 100 B=2 (multi-volume re-home loop) 4606.8, PASS.
25: 40.97 / 27: 51.42 / 50: 436.57 in busy dev windows — code paths
untouched below the gate (see boundary note below on the 25 race).  Lite
smokes: 81 B=2 execute 3022.6 (setup 2.97 s), 125 B=1 execute 15272
(setup 9.08 s) — both PASS, both now re-home their chains.  Setup at 100:
4.1–4.3 s cold full-pool quiet, 10.3 s at B=2 under contention (60 s
budget; the arena prefault is ~10 ms of that).  Warm wisdom unchanged
(ms-scale).

Gates (hand-run on the node; tryout's map-check leg still ships the
'$W/c.bin' quoting bug, ninth round): single 4.522e-16 (tol 1e-12);
two-step m=2 **2.721e-15** (tol 3e-14); graded m=64 chain **4.181e-14**
vs honest anchor 2.416e-14 (tol 1e-10) — the EXACT r7–r10 digits, as a
bit-identical change must read; single + chain outputs cmp-identical
across independent processes and across all three knob arms.

### What did NOT work / boundaries, with the numbers

* **The dev-window race banked l25-ip0 AGAIN — and the session protocol
  caught it.**  My 25/27/50 parity tryouts (fresh leases, 12 implementers
  active) cold-raced on contended cores; the 25 race banked l25-ip0 as a
  TIGHT -0.5% "tie" (46.6 us/vol trial) — the exact r8 wound shape: a
  SUSTAINED-bias window reads tight to the spread gate (my r9 record's
  documented boundary, now observed live under chain9).  All gen_powp/
  chain9 dev keys were stripped from wisdom_a80n0.json under flock, twice
  (mid-session and at end).  The scoring window's all-24-slot quiet race
  is the arbiter, as designed; monitors: absent entries are deliberate.
* **Uop deletion at 100 is confirmed dead for this engine** (1.04–1.39
  all-port vs the 2.1 cap): gen_bluestein's spill-diet lever does not
  transfer here.  Conversely my traffic-bound verdict does not transfer to
  THEIR engine — measure your own, the round's lesson in one line.
* The re-home does NOT apply at 50 B=4 (2 MB volumes, ~2K-page per-volume
  working set, STLB-covered): gated off by size, nothing to win there —
  recorded so nobody sells THP as a small-L lever on this shell.
* Harness: inline multi-line ssh commands land in $HOME and silently run
  against the wrong CWD (cost three round trips AGAIN despite my own r1
  note) — the reliable pattern is a script file under build/tryout/ that
  cd's itself, then `ssh node "bash <abs-path> <core>"`.  r11_pmu.sh /
  r11_ab.sh / r11_nocv.sh / r11_chk.sh in build/tryout/gen_powp/ are the
  reusable set.

### Borrowed, plainly

- **gen_layout (gen_r11)**: the entire mechanism — the THP=madvise
  finding, gl_thp_bytes, the zero-copy re-home recipe ("gate on
  gl_thp_bytes(final_out) < half, run steps in a gl_map_huge volume, exit
  the last step to final_out"), and the cv staging verdict.  This round is
  their finding adopted at the cell it was aimed at; the trial-regime
  extension (race tout/tcf on the same arena) and the warm-path-safe
  allocation order are mine.
- **gen_pfa_large (gen_r11, read from their impl source)**: the
  samples-delta counter method, the 88%-DRAM-bound step accounting and the
  heavier-overlap kill list (both-streams/T2/c-flush-in-compute all lose)
  that scoped this round to bandwidth-side work, saving me from re-testing
  their negatives on the shared shell.
- **gen_bluestein (gen_r11)**: the per-engine cap framing ("at ~2.1
  all-port with p0+p5 < 1.6 only deleting uops pays") — used here in the
  contrapositive: at 1.04 all-port, deletion pays nothing and bandwidth is
  everything.
- **gen_batchlane r4 / gen_pfa_small r4** (standing): held-lease same-core
  alternating pairs and min-of-mins reads for every number above.

### Operation count

FMA-port vector ops per line unchanged at all eight sizes (192/218/434/968
scored, 534/850/1850/1552 lite; soa 388/408 per pencil).  The re-home adds
ONE c-volume memcpy per volume-chain at re-home sizes (16 MB at 100, ~0.35%
of the call, measured to buy back more than it costs) and zero uops
anywhere else — port deltas flat to <1% in the table above.  Chain outputs
bit-identical to r7–r10 at every size.

### What I would do next (ranked)

1. **Verify on the r11 board** that L=100 lands ~4.4–4.6 ms class (parity
   or better vs gen_pfa_large — if they adopt the same re-home, the cell
   converges again and the remaining gap is genuinely zero on this shell)
   and that wisdom_a80n0.json holds plain-name chain9 verdicts from the
   scoring window.
2. **Below the 80 MB/step DRAM floor** the only remaining structural idea
   is cross-step tiling of the x-pass with the next step's plane pass —
   closed twice at the whiteboard (full barrier between a streaming axis
   and any plane pass); do not reopen without a paper schedule that
   actually cuts DRAM volume, not passes.
3. **XARCH**: the re-home is self-gating per host (gl_thp_bytes measures,
   the race re-decides under chain9) — check the CLX/SPR advisories see
   huge-backed arenas at all (their THP mode is unverified) before reading
   any flip as code.
4. If gen_race standardizes a quality field or a THP-regime marker in
   wisdom entries, migrate the ~q suffix and note the regime there.

## Round gen_r12 (all hands on L=100, round 2)

Standings into the round (r11 board, a80n0): led 25 (30.857), 27 (43.484),
50 (413.898 vs gen_pfa_large 417.960; gen_race 410.975 ahead by routing);
**L=100: 4465.201, and gen_batchlane's brand-new WITHIN-VOLUME SoA engine
took the cell at 4072.3 (via gen_race routing; their own quiet number
4059)** -- ~9% past me, past gen_pfa_large (4554.7), past everything.  Their
r11 record is explicit about the mechanism, and it is MY OWN r8 census they
cite as pricing the prize: the interleaved shell pays 569 shuffles + ~458
spill slots per z-line plus a separate map pass, and their lanes-are-
x-planes layout deletes all of it while keeping the same 80 MB/step DRAM
floor.  My r10 record had left 100 open "only if someone produces a paper
schedule with fewer L1 round trips, not just fewer passes" -- they produced
it and measured it.  This round I take it: the cumulative round working
exactly as the brief intends.

### What was built: the "l100-wv" candidate (ADOPTED from gen_batchlane
### gen_r11, close to verbatim)

Their within-volume engine, ported into this entry as a 14th raced L=100
chain candidate:

1. **Layout**: 8 zmm lanes = 8 x-planes of ONE volume; 13 slabs (slab 12 =
   4 replicated pad lanes, never unpacked), row stride ZP100 = 101 sv, slab
   stride SLST100 = 10114 sv (256-mod-4096 house rule), site =
   re[8]|im[8].  z-pencils (stride 1 sv) and y-pencils (stride 101 sv) are
   elementwise; both sweeps run per slab (~1.3 MB, two-axes-per-pass); the
   x-pass gathers each (y, 8z) column through 26+26 trans8 into a 104-sv
   stack scratch, runs the map-fused pencil, scatters back.  c packs once
   per chain into x-consumption order (their lanex discipline -- I took
   their measured convention and their store-at-slot-lanex[j] fix verbatim
   rather than re-earning their half-session debugging loss).
2. **Pencil**: PFA(4 x 25), their numpy-verified slot tables verbatim;
   DFT25 = 5x5 CT through a 25-sv L1 scratch with 9 compiled-in w25
   broadcast constant pairs (lit 11 Tier 1), lifted DFT5 v-pair (their
   engine default; this codelet shape is theirs, so their lift verdict
   travels with it -- NOT my in-place slot pencils where lift lost in r7),
   stage-2 DFT4s naturally in place via safe placement.  2016 vector FP
   per pencil per 8 lanes, zero shuffles in the sweeps.  Their map8 ladder
   (rsqrt14 + 2 Newtons, one exact vdivpd tail).
3. **Race integration (the part that is mine)**: wv is gated at create()
   by an m=2 whole-chain COMPOSITION gate (pack, two full steps, unpack vs
   a refnd-gated execute + the driver's exact scalar map -- exercises
   pack_vol, both sweep strides, the trans8 bracket, the lanex c pack, the
   fused map, unpack); trialed on its OWN gl_map_huge arena (graded regime
   by construction -- the r11 trial-fidelity rule); and decided by a NEW
   wv-vs-best-interleaved long-horizon playoff that runs AFTER the r10
   challenger playoff settles the interleaved slot (24 steps/arm/round,
   6 warm for the >L3 interleaved arm, adaptive r9 noise gate, min() feeds
   back only on the interleaved side).  Ranked LAST: like soa, the
   non-bit-identical engine must clear the 3% simplest-first hysteresis.
   Wisdom sig covers the pool change at 100 (stale chain9 verdicts miss);
   all other sizes' pools, protocol, and code paths are untouched.
4. NOT ported: their L=50 within-volume form -- their own r11 measured it
   14% BEHIND my ipp at the L3-resident cell ("do not spend r12 trying to
   close it with knobs", their record); and their in-flight r12 one-sweep
   fused step (BL_FUSE100, gen_pow2 r11's step-boundary x-split) -- theirs
   to land, unproven at round start, and racing two engines' worth of new
   code in one round is how gates get missed.

### Measured on the node (a80n0, reservation 438881; quiet + busy windows)

- **Graded L=100 B=1 m=64: 4074.2 us/xform (sd 0.08%, MKL 7773.9 same
  window) and 4070.0 (sd 0.13%) in a second quiet window** -- vs the r11
  board's 4465.2 (-8.8%) and gen_batchlane's quiet 4059 (parity within
  window noise).  Cold setup 5.25-5.29 s (60 s budget); warm wisdom hit
  0.015-0.046 s.
- Verbose cold race, calm window: challenger playoff ipp0 4282.3 vs
  rank-0 ipk1 4628.6 (decided); **wv playoff l100-wv 4001.4 vs l100-ipp0
  4422.4 us/vol (-9.5%, 3 rounds)**; race table wv 4001.4, ipp0 4282.3,
  ip0 4366.7, ipp1 4367.6, ip1 4427.8, ipm/ipq/iqn 5163-5364, f* 6740+.
  Banked verdict (before the dev-key strip): l100-wv, TIGHT, margin +6.9%.
- Held-lease alternating pairs, BUSY window (load 9-11): wv
  4951.2/4765.5/4891.3 vs forced ipp1 5098.7/5092.6/5425.9 -- **wv wins
  3/3 (-2.9..-9.9%)**; contention COMPRESSES wv's margin (the DRAM-bound
  engine loses more to bandwidth contention than ipp1's smaller-footprint
  form; same shape as the r6/r10 window-boundary notes -- recorded so a
  busy scoring window's narrower margin surprises nobody).
- **Counter protocol (mandatory), forced-family 512-step deltas
  (samples 10 - samples 2), same busy window**:
  | counter/512 steps | ipp1 (r11 pick) | wv (r12) | delta |
  |---|---|---|---|
  | cycles            | 10.20G | 8.45G  | -17% (window-elevated both) |
  | l1d.replacement   | 1.031G (2.01M lines/step) | 0.844G (1.65M/step) | **-18%** |
  | p0 / p5           | 3.24G / 3.82G | 2.80G / 3.74G | p5 ~flat: trans8 ~= TRNC+swaps |
  | p2_3 / p4_9       | 2.86G / 1.04G | 2.16G / 1.28G | loads -24% (spill diet), stores +23% (scratch) |
  | instructions      | 10.29G | 9.24G  | -10% |
  | stalls_mem_any    | 24% of cyc | 39% of cyc | wv is MORE purely memory-shaped |
  | dtlb walk_active  | 3.7M | 5.6M | ~0.07% both (huge pages both: ch_ar re-home / wv arena) |
  The l1d.replacement drop is the round's brief metric moving the right
  way; total vector dispatch 1.18/cyc (wv) vs 1.08 (ipp1) against the
  ~2.1 cap -- both still traffic-bound, consistent with r11's verdict.
- **Gates, all hand-run on the node** (tryout's map-check leg still ships
  the '$W/c.bin' quoting bug, tenth round): single call 4.522e-16 (tol
  1e-12, interleaved execute unchanged); **two-step m=2 3.090e-15** (tol
  3e-14, ~10x margin -- exactly gen_batchlane's number, same arithmetic);
  **graded m=64 chain 4.422e-14 vs honest anchor 2.416e-14** (1.83x, tol
  1e-10 -- also their number); single AND chain outputs bit-repeatable
  across independent processes (wisdom-pinned, the soa rationale); **B=2
  off-case PASS** (multi-volume wv loop; chain 2.970e-14 vs anchor
  4.268e-14).
- Parity, untouched sizes (busy dev windows, gate evidence): 25 B=16
  31.78 (soa warm-hit, single 3.604e-16), 27 B=16 44.54 (3.725e-16),
  50 B=4 418.76 (4.336e-16), 81 B=2 execute 2807.7 (5.038e-16) -- the
  exact historical single-call digits at every size.
- -Wall -Wextra: exactly the 16 pre-existing unused-candidate warnings
  (the new candpl field got explicit initializers everywhere).
- Round end: all gen_powp/chain9 L100 keys (including my banked l100-wv),
  the L50 key my busy-window smoke may have overwritten, and my L81 smoke
  key STRIPPED from wisdom_a80n0.json under flock (r9-r11 dev-key
  protocol; the 25/27 warm-hits stored nothing).  The scoring window
  cold-races the new 14-candidate pool and banks its own verdict --
  the wv playoff is what makes that race faithful.

### What did NOT work / boundaries, with the numbers

* **A busy-window graded reading of 10584 us** during the gate battery
  (load ~11, 12 implementers active) looked like a 2.6x regression; the
  MKL control read 8701 in the same window (vs 7773 quiet) and the paired
  A/B held -- window, not code.  At a DRAM-bound cell under all-hands
  contention, NEVER read an unpaired number (r4's lesson, bandwidth
  edition).
* The first battery ALSO showed B=2 cold setup at 25.8 s under that load
  (quiet: ~5 s): the 60 s budget still holds but the margin thins when 12
  implementers race concurrently -- same observation as r10, now with one
  more candidate in the pool.  The also-ran trim (f0/fr/frw have never won
  a 100 race in eleven rounds) remains the lever if this ever binds.
* Two banked-verdict subtleties handled, worth stating: (1) my dev tryout
  banked l100-wv TIGHT from a genuinely quiet window -- stripped anyway;
  the protocol is strip-all-dev-keys, not strip-when-unsure.  (2) The wv
  arena is allocated before the wisdom lookup, so a warm-hit l100-wv plan
  re-homes exactly like the cold-raced plan (the r11 allocation-order rule
  applied to the new engine).

### Borrowed, plainly

- **gen_batchlane (gen_r11)**: essentially everything in the wv engine --
  the within-volume layout, trans8/lanex discipline (their measured
  semantics and slot-store fix taken verbatim, saving their half-session
  loss), the PFA(4x25) module with safe placement, the DFT25-through-L1-
  scratch shape, the compiled-in w25 constants, the lifted DFT5, their
  map8 ladder, the fused-map-in-x-pass verdict (their BL_EPI100 race), and
  the c consumption-order pack.  Their record's transfer notes (50 loses,
  clflushopt declined on LLC-miss data, NT stores inapplicable) were
  honored as written -- zero re-derivation spent.
- **My own machinery kept**: the r6/r9/r10 playoff + noise-gate + banking
  stack (extended with the wv playoff stage), the r11 THP re-home (still
  protects every interleaved candidate on hosts where they win), LDU/STU/
  VSH primitives instead of their LD/ST/SH.
- The r12 lesson in one line, for the record: **when a rival's engine wins
  your cell and their record names your own census as the mechanism,
  porting it faithfully beats improving it speculatively** -- every
  deviation I allowed myself was one their record explicitly licensed.

### Operation count

wv at 100: 3900 pencils/volume-step (2600 zy + 1300 x) x 2016 vector FP
per pencil per 8 lanes, zero shuffles in the sweeps; x-pass 1248 shuffles
per 100-site column (26+26 trans8) + 100 map8 ladders per x-pencil (~15
ops + 1 vdivpd each); pack/unpack/c-pack once per 64-step chain
(amortized ~1.5%).  Interleaved families, soa 25/27, and all lite sizes:
unchanged (192/218/434/968 scored, 534/850/1850/1552 lite per line; soa
388/408 per pencil) -- their chain outputs reproduce the r7-r11 gate
digits bit-exactly.

### What I would do next (ranked)

1. **Verify on the r12 board** that L=100 lands ~4.07 ms class and
   wisdom_a80n0.json holds a plain-name chain9 l100-wv from the scoring
   window (a provisional ~q or an interleaved pick means the window was
   noisy or contended -- either is the machinery working; check the margin
   in the entry before assuming a bug).
2. **Watch gen_batchlane's r12 one-sweep fused step (BL_FUSE100)**: their
   CT-10x10 step-boundary x-split targets ~50 MB/step DRAM vs this
   engine's ~84 -- if their record lands with numbers, the same
   fuse-across-steps question opens for my wv port (the PFA pencil
   provably cannot tile across steps; theirs is the equal-radix answer).
   That is the one remaining structural lever at 100.
3. **Their next-list item 1** (software-pipeline the x-column loads;
   z-interleave the y-sweep to shorten slab reuse distance) applies
   verbatim to my port now -- worth one lease if r13 exists; also their
   CLX note (1.29 MB slab vs 1 MB L2) predicts the xarch advisory flags
   wv there, and the race + 3% hurdle already arbitrate that per host.
4. **50 stays interleaved** on their own measurement; 25/27 stay soa
   (within-volume is for B < 8 by construction -- their r11 next-list
   item 3, same conclusion).
