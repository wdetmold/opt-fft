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
