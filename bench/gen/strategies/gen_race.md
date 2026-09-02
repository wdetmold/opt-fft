# gen_race — plan-time race + per-host wisdom cache (library layer)

Scored by ADOPTION. This record doubles as the layer's user manual: if you own
a class entry, "How to adopt" below is written for you. The layer generalizes
`seed_fft3d_best`'s `choose()/trial()` — the machinery every prior winning
campaign converged on for machine-dependent kernel choice (their L=64 pair
inverts 4.3x between Haswell and Cascade Lake; the Ice Lake grading winner
differed from the Cascade Lake winner at three of eight sizes).

## Round gen_r1

### What shipped

`impl/gen_race.c` was rewritten from the dense stub into two parts (the same
adoption pattern gen_layout established this round — the two layers compose):

1. **The library** (top of the file): all-`static inline`, `gr_`-prefixed,
   zero link footprint, libc-only. Adopt with

   ```c
   #define GEN_RACE_LIB_ONLY
   #include "gen_race.c"          /* impl/ is the include dir */
   ```

   Compiles clean with `-Wall -Wextra` (gcc 11, wallaby and a80n0); unused
   pieces cost nothing (`static inline` never warns).

2. **A demo entry** (below the guard, compiled only when this file is the
   entry TU): any-L (2..128) dense row-column DFT carrying THREE genuinely
   different contraction variants — `tile4x8` (k-quad x 8-wide register tile,
   split-real), `kcj_scalar` (k-outer complex scalar), `jk_axpy` (j-outer
   accumulate) — picked per (L, B-bucket, host) by `gr_pick` in create().
   Deliberately O(L^4) floor class: the layer's living test bench, NOT a
   contender. What it proves under the real driver is the layer's whole
   contract: cold create() races and persists, warm create() is a file read,
   and the driver's two-process repeatability cmp passes BECAUSE wisdom pins
   run 2 to run 1's winner (variants round differently, so an unpersisted
   noise-flip between processes would flag NOT REPEATABLE — persistence is
   correctness infrastructure here, not just speed).

### The API (what each piece is, and the lesson it encodes)

| function | what | lesson / source |
|---|---|---|
| `gr_time_run` | warmups, inner-rep calibration until a sample clears `min_sample_us`, min over samples, spread reported | the driver's own grading discipline, so plan-time decisions rank the way scores rank |
| `gr_race` | race N thunks (optional setup/teardown); all candidates within `noise_rel` (default 2%) of the fastest form a tie group and the LOWEST-INDEX member wins | seed fft3d_best's L8/L36 lesson verbatim: "selecting on batch there would encode measurement noise" — put your primary first and a noise-level rival cannot displace it |
| `gr_pick` | `gr_race` behind the wisdom cache; hit = no candidate even built | the 50 ms warm budget; measured 4 ms on the node including the shared-FS read |
| `gr_keyf` / `gr_bucket` / `gr_sig` | key `"entry/tag/L<L>/B<bucket>"` + FNV signature of the candidate-name list appended | changed candidate set next round ⇒ different key ⇒ stale wisdom misses instead of silently replaying |
| `gr_wisdom_lookup/store` | `results/wisdom_<host>.json` (env `GEN_RACE_WISDOM` overrides; falls back to the campaign's absolute results/ if cwd differs); flock + write-temp + rename | 12 implementer binaries share this file; atomic rename means a torn write can never corrupt it (validated: 4 concurrent writers, file stays valid JSON) |
| `gr_pick_value` | integer-knob sweep (tile widths, block sizes, map variants) over the same machinery | gen_dense_prime's "race for the map variant + BC/tile knobs" ask is this one call |
| `gr_pick_plan` | whole-plan race over N `{create,execute,destroy}` vtables: 32 MB-capped race batch (the seed's cap), deterministic fill, non-planning candidates skipped, candidate 0 = existence fallback | the generalized `fft3d_best choose()/trial()`; this is the call the round-6 assembled library (planner enumerates → race picks) sits on |

Env pins (for measuring something else, mirroring `FFT3D_BEST_NO_RACE`):
`GEN_RACE_NO_RACE=1` (primary), `GEN_RACE_FORCE=name`, `GEN_RACE_REFRESH=1`,
`GEN_RACE_NO_WISDOM=1`, `GEN_RACE_WISDOM=path`, `GEN_RACE_VERBOSE=1`.

### How to adopt (60 seconds)

```c
#define GEN_RACE_LIB_ONLY
#include "gen_race.c"

/* in create(), race an in-plan variant on YOUR graded workload: */
struct myctx { my_plan *p; int variant; } a = {p,0}, b = {p,1};
static void run_v(void *s){ struct myctx *c = s; my_chain_step(c->p, c->variant); }

char key[GR_KEY_MAX];
gr_keyf(key, sizeof key, "gen_powp", "ypass", L, gr_bucket(batch));
gr_cand c[2] = { {"tile2",0,run_v,0,&a},      /* primary FIRST (tie doctrine) */
                 {"tile4",0,run_v,0,&b} };
p->variant = gr_pick(key, c, 2, NULL, NULL);  /* NULL = default opts */
```

Two rules that matter:
* **Race your graded workload** (your chain step), not a proxy — the thunk is
  yours, so nothing forces a proxy.
* **Allocate race-only buffers in a `setup()` callback**, not before
  `gr_pick` — on a wisdom hit setup() never runs and your create() stays a
  file read. (The demo does exactly this; copy its `demo_setup` shape.)

### What I measured on the node (a80n0 Ice Lake, leased core via tryout.sh, graded chain workload)

Demo entry (dense floor — the numbers prove the LAYER's budgets, not speed):

| case | chain µs/xform | setup (cold) | setup (warm, wisdom hit) | rel L2 single | map-chain |
|---|---|---|---|---|---|
| L=12 B=64 m=600 | 93.25 (sd 0.11%) | 0.149 s | **0.004 s** | 3.26e-16 | 5.43e-14 (anchor 3.89e-14) PASS |
| L=12 B=1 m=600 | 111.84 (sd 0.58%) | 0.006 s | — | 3.26e-16 | 3.11e-14 (anchor 5.80e-14) PASS |
| L=100 B=1 m=64 | 395,614 | **9.26 s** | — | 7.03e-16 | — |

* Plan budget: worst cold case measured (L=100, three dense candidates raced
  at ~0.3 s/execute) is 9.26 s vs the 60 s budget; warm create is 4 ms vs the
  50 ms budget, shared-FS wisdom read included. Small-L cold races are
  milliseconds (L=12 B=1: 6 ms).
* The race's verdicts on the node: `tile4x8` beats `kcj_scalar`/`jk_axpy` by
  21% (L12/B64), 18% (L12/B1) — real margins, no tie flags, stable picks.
* Repeatability: out.bin AND end-of-chain state bit-identical across two node
  processes (run 2 create() = wisdom hit = same variant, same rounding).
* Wisdom persisted at `results/wisdom_a80n0.json`; entries carry winner, time,
  margin, tie flag, and the candidate-set signature in the key.
* Library self-tests (wallaby): knob sweep picks the true fastest value with
  wisdom hit on re-run; `gr_pick_plan` skips a create()-refusing candidate and
  picks the faster planner; 4 concurrent writers leave valid JSON;
  `GEN_RACE_FORCE`/`NO_RACE` pins verified through the driver.

### Operation count

Library: zero instructions in any hot path — everything runs at plan time.
Demo per volume: 3·L⁴ complex MACs (dense floor by construction); the tile
variant does them as k-quad × 8-wide split-real FMA tiles, the scalar and axpy
variants exist to give the race genuinely different shapes to choose among.

### What did NOT work (with the number that killed it)

* **First cut treated a NULL thunk state as "candidate will not plan", which
  silently skipped every candidate whose ctx was legitimately NULL** — the
  race returned -1 and gr_pick fell back to candidate 0 with no wisdom write
  (smoke test: 2-candidate race "won" by the fallback with us=0). Fix:
  NULL-state-means-skip only applies when a setup() callback exists. If you
  adopt and see `us=0, margin=0` in your info struct, you hit this class of
  bug — the info struct is deliberately honest about it.
* **Micro-workloads get dead-code-eliminated and every candidate ties at
  ~3 ns**: my first library self-test "raced" empty loops and produced a
  winner with NEGATIVE margin (tie group, stability pick — the logic worked,
  the workload was gone). Not a library bug, but a warning for adopters: if
  your thunk's result is unused, gcc may delete your race subject; a
  `volatile` sink fixed it. The demo races the real execute, which cannot be
  deleted.
* tryout.sh still has the panel-documented `$W`-before-definition bug (line
  36) and the unexpanded `'$W/c.bin'` in the remote check.py call — worked
  around exactly as gen_batchlane/gen_dense_prime recorded (export W first,
  run the map-check by hand on the shared FS, run the repeatability cmp with
  absolute paths over ssh). gen_dense_prime's extended `~/bin_shim/squeue`
  heartbeat shim works unchanged.

### Borrowed, plainly

* `seed_fft3d_best/fft3d_best.c`: the whole design center — `choose()/trial()`
  warm/min timing shape, the 32 MB race-batch cap, the deterministic
  `2654435761u` input fill, the `NO_RACE`/`FORCE` pin pattern, and above all
  the noise doctrine ("two candidates within 1–4% alternating with batch is
  measurement noise — don't encode it"), which became the tie-group rule.
* **gen_layout** (this round): the `*_LIB_ONLY` include-adoption pattern,
  `static inline` for warning-free partial adoption, and the record-as-manual
  format of this file.
* **gen_dense_prime** (this round): the k-quad × wide-tile register-blocking
  shape for the demo's `tile4x8` variant (their 155→33 µs/pass axpy-vs-tile
  measurement is why the demo's axpy variant loses everywhere — it exists as
  a real, known-bad contrast), and the bin_shim/tryout workarounds.
* **gen_pfa_small / gen_batchlane** (this round): the harness-bug workarounds
  confirmed, and the reminder that races must run the GRADED workload (their
  chain-owns-the-map lesson is why gr_cand takes an arbitrary thunk instead of
  an execute signature).

### What I would do next (gen_r2)

1. **Adoption is the score**: wire gr_pick into the first class entry that
   wants it. Concrete offers already on the table: gen_dense_prime's map
   variant (`div` vs `rcp` — their measured 1.5% split is exactly tie-group
   territory, and cross-arch it may flip) and BC/tile knobs via
   `gr_pick_value`; gen_pfa_small/gen_planner's round-3 "who serves an
   unlisted size at which B" via `gr_pick_plan`.
2. **Cross-arch wisdom**: the file is already per-host; when the monitor's
   Cascade Lake / Sapphire Rapids reruns start (every second round), verify a
   wisdom file from one host never leaks to another (it can't — hostname is
   in the path — but the xarch report should show per-host winners diverging,
   which is the layer's whole justification; capture that table here).
3. **Race-time confidence upgrade**: re-race on margin<noise ties every Nth
   create instead of pinning forever (cheap: the tie flag is already stored);
   only if an adopter shows a real flip-flop case.
4. **Plan-cache layer**: wisdom currently caches the CHOICE; for round 6's
   60 s budget the expensive part may become candidate GENERATION (planner) —
   coordinate with gen_planner on caching the chosen factorization string in
   the same file (the format has room: it's a JSON object per key).

## Round gen_r2

### What changed

**1. Library (the adoption surface) — additive only, nothing an r1 adopter
sees breaks:**

* **`GR_NAME_MAX` 64 → 128, fixing a real latent bug**: `gr_pick` and
  `gr_pick_plan` used 64-byte winner-name buffers, but `gr_wisdom_lookup`
  REJECTS a stored name that does not fit (returns miss). gen_planner's
  canonical tree names run to 96 chars, so any adopter keying on them would
  have re-raced every single create() forever while the wisdom file looked
  perfectly healthy — a silent, permanent cache miss. Found while wiring the
  new demo (below); fixed before it ever bit anyone.
* **String-valued wisdom**: `gr_wisdom_get_str(key, out, cap)` /
  `gr_wisdom_put_str(key, val)` — cache an arbitrary short string (no
  quotes/spaces/braces, <= 127 chars) under your own key, same file, same
  flock+rename atomicity, honors `GEN_RACE_NO_WISDOM` / `GEN_RACE_REFRESH`.
  This is gen_planner's requested "default pick = last raced winner" hook
  (their r1 next-list #5) and my r1 next-list #4 (plan-string cache), in two
  calls. Validated: 91-char value round-trips and survives merges with other
  keys; `-Wall -Wextra` clean under `GEN_RACE_LIB_ONLY` (gcc 11).
  Namespace note: gr_pick keys always end in `#<8 hex>`, so raw string keys
  cannot collide with race keys.

**2. Demo entry — rebuilt as the round-6 trunk in miniature.** The r1 demo
was a deliberately-slow O(L^4) dense test bench (last on every leaderboard
row, 30-80x behind winners). It is now the composition the campaign is
building: **gen_planner's library enumerates candidate trees for L
(`GEN_PLANNER_LIB` include, exactly as their record invites), and `gr_pick`
races the top <= 8 trees ON THE GRADED CHAIN STEP** (`pln_p3d_step`, one
volume in place), persisting the winner per (L, B-bucket, host). Execute and
the fused `fft3d_chain` run through the winner's `pln_p3d` engine; the
two-step chain-ownership gate (fused vs execute + exact scalar map, 1e-12) is
run once cold and its verdict cached through the new string-wisdom API, so
warm create() never re-runs it. Race buffers and engines are built lazily in
setup(): a wisdom hit builds NOTHING except the winning engine.

### Measured on the node (a80n0 Ice Lake, leased core via tryout.sh, graded chain, min us/xform)

| case | r1 | r2 | speedup | setup cold -> warm |
|---|---|---|---|---|
| L=10  B=64 m=1000 | 48.6 | **6.26** | 7.8x | 0.007 s |
| L=12  B=64 m=600  | 93.2 | **10.50** | 8.9x | 0.015 s |
| L=12  B=1  m=600  | 111.8 | **11.86** | 9.4x | 0.010 s |
| L=15  B=32 m=600  | 223.1 | **21.49** | 10.4x | 0.118 s |
| L=20  B=32 m=256  | 633.1 | **46.17** | 13.7x | 0.015 s |
| L=25  B=16 m=256  | 1569.6 | **105.19** | 14.9x | 0.010 s |
| L=27  B=16 m=200  | 2094.4 | **154.52** | 13.6x | 0.011 s |
| L=31  B=16 m=140  | 3545.4 | **533.38** | 6.6x | 0.017 s |
| L=32  B=8  m=250  | 3888.1 | **238.04** | 16.3x | 0.021 s |
| L=40  B=8  m=128  | 9325.9 | **507.06** | 18.4x | 0.047 s |
| L=50  B=4  m=128  | 23750.8 | **1187.39** | 20.0x | 0.052 s |
| L=100 B=1  m=64   | 391379.6 | **9538.88** | 41.0x | 0.726 s -> **0.002 s** |

Beats MKL at L=31 (533 vs 849). Gates: single call rel L2 2.9-5.2e-16
everywhere; map-chain m=600/140/64 checked by hand (tryout's map-check leg
still has the `$W` quoting bug): 5.0e-14 / 3.5e-14 / 5.5e-14 vs anchors
3.9/2.3/2.4e-14, tol 1e-10 — PASS; two-step m=2 gate 2.2-2.9e-15 vs tol
3e-14 — PASS; repeatability cmp identical across processes (wisdom pins run 2
to run 1's tree, which matters MORE now: different trees round differently).

**The race earns its keep over the model pick** (wisdom file receipts):
L=10 picked `c2(d5)` = cand[1], 15.9% over the model-best — exactly the L=10
model miss gen_planner's r1 record owned up to; L=100 picked `c4(c5(d5))` =
cand[1], 3.7% (their record predicted this flip too); L=31 model and race
agree on `rad31(c3(c5(d2)))` with 10.7% margin; L=12 is a genuine tie (0.6%)
and the tie doctrine keeps the model pick — stability, not noise-encoding.

### Operation count

Library: still zero instructions in any hot path — everything at plan time.
Demo: whatever the winning planner tree costs (mixed-radix/Rader/PFA class,
O(L^3 log L)-flavored), not my arithmetic to count — the demo's contribution
is CHOOSING it per host and pinning it. Warm create() = one wisdom read (ms)
+ one engine build (~ms) + one cached gate read; 2 ms measured at L=100
against the 50 ms budget (was 726 ms cold, vs the 60 s cold budget).

### What did NOT work (with the number that killed it)

* The r1 wname[64] buffers vs 96-char planner names (above): not a
  measurement, a discovered landmine — the failure mode is a wisdom file that
  LOOKS populated while every create() silently re-races (cold-cost forever,
  and worse, a noise flip between the driver's two processes would show NOT
  REPEATABLE). If your adopted layer keys on long names, update to r2.
* tryout.sh's reservation check needs `~/bin_shim/squeue` on PATH (documented
  by gen_dense_prime in r1, still true) and the remote map-check leg still
  dies on the unexpanded `'$W/c.bin'` — same manual workaround as r1.

### Borrowed, plainly

* **gen_planner (this round, the big one)**: the entire execution engine —
  `pln_enumerate` candidate trees + `pln_p3d_build/exec/step` — adopted via
  their `GEN_PLANNER_LIB` include exactly as their record invites; also their
  create()-gate shape (fused chain vs execute + exact scalar map) and the
  volume-major `fft3d_chain` structure, copied from their entry section.
  Credit where due: most of the raw speedup above is their engine; my layer's
  contribution is the measured pick (+3.7-15.9% where the model missed), the
  per-host persistence that makes warm create 2 ms and repeatability
  structural, and the cached gate.
* **gen_pfa_large r1**: "race the CHAIN STEP, not raw execute — the two order
  candidates differently." The demo races `pln_p3d_step`, not `pln_p3d_exec`.
* **gen_pfa_small/gen_batchlane r1**: harness workarounds reconfirmed.

### What I would do next (gen_r3)

1. **Adoption handshake with gen_pfa_large** (their #1 next item, deferred
   twice): their pf-id pool + 0.37-6.5 s re-race per process is exactly
   `gr_pick` + lazy setup(); offer a worked patch in their record's terms.
2. **gen_planner wisdom hook**: now that string wisdom exists, their
   standalone entry can persist `picked` under their own key and skip their
   in-process re-race — coordinate so the two entries' keys stay disjoint.
3. **Sub-tree diversity racing**: when planner emits top-2 sub-candidates per
   composite child (their #3), the candidate count grows past 8 — raise
   NK_MAX and let the budget/deadline logic prune, or race hierarchically.
4. **Cross-arch wisdom table**: when the monitor's Cascade Lake / Sapphire
   Rapids reruns land, capture per-host winner divergence here (the layer's
   whole justification); verify rankings, expect rad31-vs-bs and c2/c5 flips.

## Round gen_r3

### What changed

**1. Library API: FROZEN.** No adopter's r2 record asked for a new primitive,
and three entries now `#include` this file (`gen_powp` since r2;
`gen_planner` and `gen_pfa_large` adopted in r3 — see below), so churn is
now a cost I charge to others. Zero changes to any `gr_*` signature or to
the wisdom file format. (Doctrine borrowed from gen_layout's r3 section,
who articulated it first: "churn in a layer others include is its own
cost".)

**2. Demo entry, change 1 — engine-generation SALT on the wisdom tags**
(`chain` → `chain3`, `chaingate` → `chaingate3`, new `tile3`). This is the
round's most important lesson for every wisdom adopter: gen_planner's r3
engine (AVX-512 leaves, register-tiled folded dense) reordered the candidate
trees, but the candidate NAMES did not change, so `gr_sig` alone could not
invalidate the r2 verdicts. The receipt that proves the hazard: the node's
r2 wisdom pinned L=31 to `rad31(c3(c5(d2)))`; the fresh r3 race picks **d31
at an 84% margin** (their register-tiled fold made dense-31 the winner —
their own r2 record predicted the d31/rad31 crossover was kernel-dependent).
Replaying stale wisdom would have silently cost ~2x at a scored size while
the wisdom file looked perfectly healthy. Rule now in the header comment:
**if your engine generation changes and your candidate names don't, salt
your key tag.** Also flipped: L=12 `c4(d3)` → `c3(d4)`, L=40 kept `c5(d8)`
but with a fresh 7.1% margin.

**3. Demo entry, change 2 — second-stage TILE race.** After the tree race,
a second `gr_pick` races the winning tree's scratch-row width {32, 16, 64}
(names `t32`/`t16`/`t64`, per (L, B-bucket, host), primary = engine default
32 per the tie doctrine), built through gen_planner's documented
`GEN_PLANNER_TILE` env hook in setup() (deliberately NOT a copy of
pln_p3d_build's body — the env hook survives their internal changes; the
set/restore is create()-time single-threaded). Planner's r3 hardcodes 32
("the r2 crossover vanished") — and the race CONFIRMS 32 at 9 of 11 sizes,
but found two real, non-tie wins: **t16 at L=31 (+3.9%)** and **t64 at L=50
(+2.3%)**. Exactly the knob class the cross-arch guard exists for; now it
is measured and persisted per host instead of trusted. Cold cost: two extra
engine builds + timings (~5-10 ms at small L); warm cost: one wisdom read.
`picked[]` widened to GR_NAME_MAX+8 so `name@t<w>` never truncates a
96-char planner name.

### Measured on the node (a80n0 Ice Lake, leased core via tryout.sh, graded chain, min us/xform)

| case | r2 | r3 | vs r2 | MKL 2022 | vs MKL |
|---|---|---|---|---|---|
| L=10  B=64 m=1000 | 6.30  | **4.74**  | -25% | 4.69  | 0.99x |
| L=12  B=64 m=600  | 10.46 | **6.70**  | -36% | 7.91  | **1.18x** |
| L=15  B=32 m=600  | 21.43 | **14.97** | -30% | 16.73 | **1.12x** |
| L=20  B=32 m=256  | 45.88 | **27.97** | -39% | 58.58 | **2.09x** |
| L=25  B=16 m=256  | 104.9 | **69.50** | -34% | 121.1 | **1.74x** |
| L=27  B=16 m=200  | 153.7 | **105.6** | -31% | 144.7 | **1.37x** |
| L=31  B=16 m=140  | 531.0 | **200.8** | -62% | 849.7 | **4.23x** |
| L=32  B=8  m=250  | 234.5 | **128.6** | -45% | 176.9 | **1.38x** |
| L=40  B=8  m=128  | 500.1 | **281.3** | -44% | 412.3 | **1.47x** |
| L=50  B=4  m=128  | 1171.6| **790.3** | -33% | 956.6 | **1.21x** |
| L=100 B=1  m=64   | 9501.6| **6279.3**| -34% | 7921.8| **1.26x** |

Beats MKL at 10 of 11 cases now (ties at L=10, 4.74 vs 4.69). B=1 ==
batched per transform (volume-major chain, no remainder): 5.42 (10), 69.30
(25), 196.6 (31), 764.2 (50). Setup: cold worst 0.43 s at L=100 (budget
60 s); warm 5-7 ms measured at L=12/L=100 (budget 50 ms) = three wisdom
reads + one engine build.

Gates, shipped binary, run on the node: single-call rel L2 2.9e-16..5.2e-16
at all 11 cases + all four B=1 cases; map-chain at graded m PASS at
12/25/31/100 (4.9e-14 / 3.4e-14 / 2.5e-14 / 5.5e-14 vs anchors
3.9/2.8/2.3/2.4e-14, tol 1e-10); two-step m=2 gate PASS at the same four
(9.5e-16..3.0e-15 vs tol 3e-14); out.bin bit-identical across two node
processes at all four (wisdom pins run 2 to run 1's tree+tile — with trees
AND tiles both raced this is now doubly load-bearing).

### Operation count

Library: unchanged, zero instructions in any hot path. Demo: the winning
planner tree's cost on gen_planner's r3 engine; my contribution is the
measured (tree, tile) pick per (L, B-bucket, host) and the persistence that
makes warm create ~6 ms and repeatability structural.

### What did NOT work / friction (with numbers)

* **The tile race is 9/11 confirmation, 2/11 win** — t32 everywhere except
  L=31 (t16 +3.9%) and L=50 (t64 +2.3%), and two of the t32 verdicts are
  ties with NEGATIVE margin recorded (-1.3% at L=10, -1.4% at L=40: the
  default won on the tie doctrine, not the stopwatch). If you copy this
  pattern, expect mostly-confirmation; the value is the exceptions plus the
  cross-arch insurance, and the cold cost (~2 engine builds) is only worth
  it because the verdict persists.
* **GEN_RACE_FORCE spans stages awkwardly**: with two chained gr_picks,
  FORCE=d31 pins the tree but the tile race still runs (tile names don't
  match), and FORCE=t16 pins the tile but not the tree. Fine for dev,
  documented here; a compound force syntax is API churn I'm not spending
  during a freeze.
* **tryout.sh's check.py leg still dies on the unexpanded `'$W/c.bin'`**
  (line 48; the `$W` build bug from r1 is fixed, this one is not) — and
  because the ssh command is &&-chained, its failure also SKIPS the
  repeatability cmp. All map-checks, two-step gates and cmp runs above were
  run manually over ssh; the harness PASS line only covers the single-call
  gate. Budget one extra manual pass for this until it's fixed.
* `reserve.sh --status` still needs `~/bin_shim/squeue` on PATH on wallaby
  (gen_dense_prime r1, still true in r3).

### Borrowed, plainly

* **gen_planner r3 (again the big one)**: their AVX-512 engine is most of
  the raw speedup in the table above — same honest split as r2: their
  kernels, my measured pick + persistence. Also their `GEN_PLANNER_TILE`
  dev knob, promoted here into a raced-and-persisted plan parameter.
* **gen_layout r3**: the API-freeze doctrine, adopted verbatim.
* **gen_pfa_small r3**: "the div-vs-rcp choice is a property of the
  SURROUNDING CODELET... A/B it in place, never adopt on faith" — their
  lesson is this round's tile receipts in miniature, and the reason the
  demo races the default before trusting it.

### Adoption status (the score)

* **gen_planner r3**: entry now persists its raced tree per (host, L) via
  `gr_wisdom_get_str/put_str` (key `gen_planner/tree/L<L>`), storing only
  gate-passed picks. Their warm create skips the race on a file read.
* **gen_pfa_large r3**: adopted (their r2 next-list #1, third time's the
  charm) — tune() verdict cached via string wisdom, `GEN_RACE_*` pins pass
  through.
* **gen_powp r2**: still on gr_keyf/gr_sig/gr_wisdom_lookup/store for the
  soa-variant race.
* Key namespaces verified disjoint: my keys are `gen_race/*`; theirs carry
  their own entry prefix; gr_pick keys always end `#<8 hex>`.

### What I would do next (gen_r4)

1. **Salt audit for adopters**: gen_planner's `gen_planner/tree/L<L>` key
   has the same stale-wisdom hazard my chain keys had (engine changes, names
   don't). Offer them a one-line fix (version their key or value) before the
   r4 engine changes bite them the way r2→r3 would have bitten me.
2. **Cross-arch wisdom table** (carried again): when the every-second-round
   CLX/SPR reruns land, capture the per-host winner divergence here.
   `tile3/L31=t16` and `chain3/L31=d31` are my predicted flip candidates.
3. **Race across class ENGINES**: the round-6 trunk should race
   planner-trees vs gen_batchlane/gen_powp/gen_pfa_* engines per (L,B) via
   gr_pick_plan — blocked only on class entries exposing `*_LIB_ONLY`
   includes (none do yet; this is an open invitation, the vtable API is
   already shipped as gr_plan_cand).
4. **Hierarchical sub-tree racing** if planner ships enumeration diversity
   (their carried #3): race sub-decompositions within the winning root
   instead of raising NK_MAX.

## Round gen_r4

### What changed

**1. Library: the race is now INTERLEAVED (sample-major), the round's one
substantive library change — and it is the whole panel's finding, not mine.**
Three r4 records reported independently that candidate-major timing on this
node is broken: gen_batchlane ("tryout A/B pairs hop cores... states differed
by 10-25%"), gen_pfa_small ("the leased core spent most of this session
flipping between two sustained states ~13.5% apart"), gen_pfa_large ("one-
after-the-other forced runs are no longer an A/B; alternate within the lease
and compare adjacent pairs"). My gr_race was candidate-major — time candidate
0 to completion, then candidate 1 — i.e. exactly the confounded protocol, run
at plan time on every adopter's behalf. Now: every candidate is set up,
warmed and rep-calibrated first (that pass is its sample 0), then the
remaining samples run as round-robin ROUNDS over all live candidates,
min-of-rounds per candidate. Same signatures, same gr_opts, same total work,
same tie doctrine — API freeze intact; adopters get drift-immunity on
recompile. `GEN_RACE_SEQ=1` restores the r3 order for A/B-ing the racer
itself. One behavioral note documented in the header: all candidates' setup()
states now COEXIST during the race — share big race buffers through ctx (the
demo's demo_share shape), not per-candidate.

**The receipt that the change matters** (node, L=12 B=64, two fresh
no-wisdom races each way, same lease): SEQ run 1 picked `c3(d4)` as a -0.5%
"tie", SEQ run 2 picked `c4(d3)` at 2.1% — a flip-flop, and with wisdom ON
whichever ran first would have been pinned per host. INTERLEAVED picked
`c4(d3)` both times (margins 2.5%, 3.7%), and c4(d3) is genuinely faster
(4.99-5.03 us shipped vs ~5.2 through the r3 pick). Stability double-check at
L=31: two fresh interleaved races both pick d31@t16 with reproducing margins
(tree 57.3/57.4%, tile 4.1/3.5%).

**2. Demo: wisdom salt bumped chain3/tile3/chaingate3 -> chain4/tile4/
chaingate4** — mandated twice over by my own r3 rule: gen_planner's engine
changed again (fused register-resident CT codelets for n<=25, new cost
model) with candidate names unchanged, AND the race methodology changed
(an interleaved verdict must not be presumed comparable to a candidate-major
one). Receipts that the bump was load-bearing again: the L=12 pick flipped
c3(d4) -> c4(d3) (+2.0% real margin), L=100 flipped c4(c5(d5)) ->
c5(c4(d5)), and the L=40 tile flipped t32(tie) -> t16 (+2.2% NON-tie) on the
fused engine.

**3. Demo: tree race widened NK_MAX 8 -> 12, samples 3 -> 4.** gen_planner's
r4 enumeration emits each CT root with its runner-up child tree — their
comment says "sub-tree diversity for the race", i.e. built explicitly for
this layer (my r3 next-list #4, delivered from their side). Truncating at 8
would have discarded exactly those candidates. samples=4 because under
bimodal cores each candidate should be sampled in >=4 temporally-separate
states; cold cost stays trivial (setup 0.021-0.077 s at 10-50, 0.72 s at
L=100, vs 60 s budget).

**4. Fixed a latent truncation hazard my own -Wall audit caught**: the
chaingate wisdom key buffer was GR_KEY_MAX (160) but
"gen_race/chaingate4/L<L>/" + a maximal 96-char tree name + "@t<w>" can pass
it. Widened to GR_KEY_MAX+64 (still inside the lookup needle buffer). Both
build modes (entry / GEN_RACE_LIB_ONLY) now compile -Wall -Wextra clean
again; the one remaining warning in the entry TU is a -Wrestrict inside
gen_planner.c line 1779 (their in-place fused-codelet call — deliberate on
their part, their file, flagged here so nobody burns time on it).

### Measured on the node (a80n0, leased core via tryout.sh, graded chain, min us/xform; single runs each — per this round's collective lesson I trust deltas here only where MKL moved with them)

| case | r3 board | r4 | MKL same window | vs MKL | note |
|---|---|---|---|---|---|
| L=10  B=64 m=1000 | 4.718 | **3.483** | 4.672 | 1.34x | r3 was an MKL tie; now a win |
| L=12  B=64 m=600  | 6.634 | **4.993** | 7.77 | 1.56x | pick flipped to c4(d3) |
| L=12  B=1  m=600  | — | **5.732** | 8.32 | 1.45x | |
| L=15  B=32 m=600  | 14.785 | **11.767** | 16.76 | 1.42x | |
| L=20  B=32 m=256  | 27.826 | **24.224** | 58.51 | 2.42x | |
| L=25  B=16 m=256  | 69.317 | **60.129** | 121.0 | 2.01x | |
| L=27  B=16 m=200  | 104.54 | **86.257** | 146.9 | 1.70x | |
| L=31  B=16 m=140  | 195.63 | **196.16** | 858.4 | 4.38x | flat; d31@t16 re-confirmed fresh |
| L=32  B=8  m=250  | 127.31 | 132.24 | 185.7 (r3: 176.1) | 1.40x | window +5% hot (MKL moved with it) |
| L=40  B=8  m=128  | 269.78 | 282.93 | 426.5 (r3: 405.9) | 1.51x | window +5% hot; tile now t16 |
| L=50  B=4  m=128  | 772.61 | **645.33** | 1002.6 | 1.55x | |
| L=100 B=1  m=64   | 6198.3 | **5504.7** | 7769 | 1.41x | quieter window; 5907 in a hotter one |

Most of the raw speedup at 10-27 is gen_planner's fused CT codelets (same
honest split as every round: their kernels, my measured pick + persistence);
the pick's own contribution this round is the non-tie race wins — L=12
c4(d3) +2.0%, L=20 c5(d4) +2.3% (model's #2), L=27 t32 +7.6%, L=31 d31
+56%/t16 +4.9%, L=40 t16 +2.2% — and the flip-flops that interleaving
stopped from being pinned.

Gates, shipped binary, run manually on the node (tryout's check.py leg still
dies on the unexpanded `'$W/c.bin'`, unchanged since r1): single-call rel L2
2.9-4.7e-16 at all 12 cases above (tol 1e-12); map-chain at graded m PASS at
12/25/31/100: 5.195e-14 / 3.402e-14 / 2.462e-14 / 3.489e-14 vs anchors
3.9/2.8/2.3/2.4e-14 (tol 1e-10); two-step m=2 gate PASS at the same four:
9.9e-16 / 1.5e-15 / 1.7e-15 / 2.7e-15 (tol 3e-14); single AND chain outputs
bit-identical across independent node processes at all four. Warm create
2 ms measured (three wisdom reads + one engine build).

**Round end: all 35 gen_race/* keys stripped from results/wisdom_a80n0.json**
(gen_pfa_large's r3 protocol, now evidently campaign-wide — after my strip
the entries block was EMPTY, so every other adopter had already stripped
theirs). The monitor cold-races in its full-quiet window with the
interleaved racer; absent entries are deliberate. Both my C parsers handle
the empty-entries file (verified).

### Operation count

Library: unchanged, zero instructions in any hot path; the interleave
reorders WHEN plan-time samples are taken, not how many. Demo: the winning
planner tree's cost on gen_planner's r4 fused engine; my contribution is the
measured (tree, tile) pick per (L, B-bucket, host) under a drift-immune
protocol, and the persistence that makes warm create 2 ms and repeatability
structural.

### What did NOT work / honest boundaries

* **L=32 and L=40 read above their r3 boards (+4-5%)** — but MKL read +5%
  in the same windows, so I claim window heat, not regression. If the r4
  board disagrees, these two cells are where to look first.
* **The interleaved race slightly mis-models the graded cache pattern**: in
  the graded chain one engine runs m steps back-to-back (its plane scratch
  stays L2-hot); interleaved, each candidate's plane is evicted by rivals
  between rounds. All candidates pay the same tax, so ranking should
  survive; noted because a very close race between engines of very
  different scratch size could in principle tilt. Not observed this round
  (all margins reproduced across protocols or were honest ties).
* **The >32-candidate path stays candidate-major** (the fixed state arrays):
  documented in the code; nobody races >32 today.

### Borrowed, plainly

* **gen_batchlane r4 / gen_pfa_small r4 / gen_pfa_large r4** (jointly): the
  same-core interleaved A/B protocol — this round's entire library change is
  their measurement finding institutionalized at plan time, credited in the
  header. gen_pfa_large's "trust the race's interleaved min-of-rounds" is
  now literally what gr_race does.
* **gen_planner r4**: the fused-CT engine (most of the raw speedup at
  L<=27) and the sub-tree diversity enumeration that NK_MAX=12 consumes.
* **gen_pfa_large r3**: the round-end wisdom-strip protocol.

### Adoption status (the score)

* gen_planner (string wisdom, tree key), gen_pfa_large (tune() verdict
  cache), gen_powp (keyf/sig/lookup/store for their soa race) — all
  unchanged by this round's library change (their calls don't route through
  gr_race). Standing offer: gen_powp's hand-rolled soa-variant timing loop
  would get the interleaving for free by switching to gr_pick — their r3
  record's own bimodality complaints are the argument.
* **Flagged again for gen_planner (r3 next-list #1, now urgent)**: their
  `gen_planner/tree/L<L>` string key is still unversioned. Their name-match
  fallback catches renamed candidates but NOT an engine change under
  unchanged names — precisely what their r4 fused codelets just did. Their
  r3-era pinned trees will silently replay against the r4 engine on any
  host whose wisdom wasn't stripped. One-line fix on their side: salt the
  key ("tree4") or version the value.

### What I would do next (gen_r5)

1. **Cross-arch (XARCH.md lands after this round)**: the interleaved racer
   is exactly what the CLX/SPR advisory needs — capture the per-host winner
   divergence table here at last (carried three rounds). Predicted flips:
   d31 tile (t16 is a register/L2 effect), the c4/c5 root order at 100,
   and gen_layout's NT threshold class of knobs generally.
2. **Race across class ENGINES via gr_pick_plan** (carried; the round-6
   trunk's missing piece): still blocked on class entries exposing
   `*_LIB_ONLY` includes — none do as of this writing. Fourth open
   invitation; the gr_plan_cand vtable has been shipped since r1.
3. **Wisdom-strip as API**: three entries now hand-strip their keys at round
   end. A `gr_wisdom_drop_prefix("gen_race/")` would make the protocol one
   call and flock-safe for everyone; small, additive, freeze-compatible.
4. **Tile race candidate set per engine generation**: t16's win moved from
   {31} to {31,40} when the engine fused; if planner's engine changes again,
   consider racing {32,16,48,64} once per host rather than trusting the
   pair.

## Round gen_r5

### What changed

**1. Library: ONE additive call, `gr_wisdom_drop_prefix(prefix)`** (my r4
next-list #3). Round-end wisdom strip as a single flock-safe read-filter-
rename; returns the count dropped. Three entries hand-roll this protocol
every round (gen_pfa_large r3 started it; by r4 end the entries block was
empty because everyone had stripped by hand) -- now it is one call that
cannot tear the shared file. Self-tested on wallaby (drop 2 of 4 keys,
foreign prefixes untouched, file valid and writable after; empty-drop
no-op), then dogfooded for this round's own strip (60 entries, file valid,
0 gen_race keys left). Everything else in the API: FROZEN, unchanged
signatures, r3 doctrine.

**2. Demo entry: gen_planner's NEW split-group batch-lane engine raced in
as extra arms.** Their r5 (in-progress file, adopted the day it appeared)
adds `pln_s8_*`: 8 volumes of a batch packed site-major SPLIT-complex --
batch is the vector dimension, zero shuffles, no masked tails, the map
ladder sees 8 sites/zmm -- with levels s1 (fused whole-axis CT), s2
(two-pass CT), s3 (dense fold). At batch >= 8 my create() now probes
`pln_s8_build` per enumerated tree x level (cap 4 arms, planner's own
heuristic) and enters them in the SAME gr_pick tree race as candidates
"<tree>@s<lev>". Normalization: a group thunk is one `pln_s8_step` (8
volumes); pv thunks run 8 volume-major steps per call -- equal work, each
in its real chain pattern. Probes run before gr_pick so the candidate name
list (hence gr_sig) is deterministic; group race buffers (16*vol doubles
x 3, shared by ALL group arms per the r4 coexistence rule) allocate lazily
in setup(), so a wisdom hit builds nothing. A group winner gets its own
two-step gate (pack -> 2 group steps -> unpack vs per-volume execute +
exact scalar map, verdict string-cached like the pv gate) and fft3d_chain
runs pack -> m steps -> unpack per 8 volumes, B%8 remainder per-volume
(planner's own shapes, adopted). Group-gate failure drops to the pv path,
never ships.

**3. Salt bump chain4/tile4/chaingate4 -> chain5/tile5/chaingate5**
(mandated by my r3 rule: planner's engine generation changed again -- twice
during this session; their file went 124951 -> 130517 bytes while I raced).
Tile race now runs only for pv winners (the group engine has no tile knob;
p3 then only serves execute() + remainder).

### Measured on the node (a80n0, tryout leased cores, graded chain, min us/xform)

| case | r4 board | r5 | delta | MKL same window | vs MKL | pick (race receipt) |
|---|---|---|---|---|---|---|
| L=10  B=64 | 3.483 | **1.458** | -58% | 4.562 | 3.1x | c2(d5)@s1 |
| L=12  B=64 | 4.993 | **2.511** | -50% | 7.738 | 3.1x | c3(d4)@s1 |
| L=12  B=1  | 5.732 | **4.992** | -13% | 7.316 | 1.5x | pv (no group at B<8) |
| L=15  B=32 | 11.767 | **5.830** | -50% | 16.458 | 2.8x | c3(d5)@s1 (+2.5% non-tie) |
| L=20  B=32 | 24.224 | **19.934** | -18% | 58.967 | 3.0x | c4(d5)@s1 |
| L=25  B=16 | 60.129 | **40.337** | -33% | 124.394 | 3.1x | c5(d5)@s1 (+50.7% margin!) |
| L=27  B=16 | 86.257 | 86.591 | flat | 145.216 | 1.7x | pv c3(c3(d3)) beat its @s2 by 3.1% |
| L=31  B=16 | 196.16 | **144.95** | -26% | 856.881 | 5.9x | d31@s3 (+39.3% margin) |
| L=32  B=8  | 132.24 | 135.26 | ~flat | 186.124 | 1.4x | pv c4(d8)@t16 (s8 tie, -0.3%) |
| L=40  B=8  | 282.93 | 279.99 | -1% | 429.194 | 1.5x | pv c5(d8)@t16 (s8 tie, -0.06%) |
| L=50  B=4  | 645.33 | 644.63 | flat | 964.107 | 1.5x | pv (no group at B<8) |
| L=100 B=1  | 5504.7 | 5479.5 | flat | 8374.796 | 1.5x | pv c5(c4(d5)) |

Same-core interleaved receipt (held slot lease core 3, 4 alternating rounds,
L=12 B=64 m=600): shipped c3(d4)@s1 2.486-2.528 vs GEN_RACE_FORCE'd r4-style
pv c4(d3) 5.005-5.109 -- **2.02x, all four rounds**. The pick's contribution
vs blind adoption: the race REJECTED the group engine at 27 (+3.1% pv),
32/40 (honest ties, pv primary kept), i.e. the batch-lane form is NOT a
uniform win and the per-(L,B,host) verdict is exactly what this layer sells.

Gates, shipped binary, run manually on the node (tryout's check.py leg still
dies on the unexpanded '$W/c.bin', unchanged since r1): single-call rel L2
2.8e-16..4.5e-16 at all 12 graded cases (tol 1e-12); map-chain at graded m
PASS at 12/27/31/100 + mixed B=12 at L=10 (4.9e-14 / 3.1e-14 / 2.6e-14 /
4.3e-14 / 8.6e-14 vs anchors 3.9/2.6/2.3/2.4/9.1e-14, tol 1e-10); two-step
m=2 gate PASS at the same five (9.2e-16..2.6e-15 vs tol 3e-14); single AND
chain outputs bit-identical across independent node processes at all five --
the B=12 case exercises group-of-8 + 4-volume pv remainder end to end.

Budgets: warm create 2 ms measured at 27/31 (instrumented phase timers:
enumerate 0.0 / s8 probes 0.1 / tree pick 0.7 / gates-cached 1.0 ms) -- the
0.08-0.17 s "setup" the driver prints on chain runs is chain-buffer prep it
charges every entry, not create(). Cold create worst measured: 0.68 s at
L=100, 1.43 s at the round-6-style L=127 B=8 probe (vs 60 s budget), which
also PASSes (rel L2 1.0e-15) and beats MKL on raw execute (43.9 vs 50.6 ms).
Unseen-composite probe L=14 B=8: plans, passes, 0.022 s cold.

### Operation count

Library: unchanged, zero instructions in any hot path. Demo: the winning
arm's cost on gen_planner's r5 engines -- group arms are op-for-op the pv
tree (their fmsub/fmadd pairs match fmaddsub rounding per lane) with zero
shuffle uops and the batch dimension as the vector dimension; my
contribution is the measured (tree, form, tile) pick per (L, B-bucket,
host) and the persistence that keeps repeatability structural.

### What did NOT work / honest boundaries

* **The group arms lose or tie at 27/32/40** (numbers above) -- expected
  boundary, not a failure: at vol >= 20k the group working set (16*vol
  doubles vs 2*vol) trades L2 residency for lane width. Recorded so nobody
  assumes @s wins everywhere.
* **gen_planner's file was mid-refactor twice during this session** (their
  pw_leaf_run/pln_fusedw_run grew a fused-c arg with call sites trailing;
  ~20 min of broken AVX-512 builds). Validated my side against a
  NULL-patched scratch copy, re-verified against their real file once it
  stabilized -- shipped state compiles and passes against their 20:55
  snapshot. If their final r5 file changes arm relative speeds again, the
  round-end strip already guarantees the monitor cold-races fresh verdicts.
* **Execute() (non-chain) stays per-volume even at B>=8** -- MKL beats my
  execute at L=14 (18.5 vs 4.3 us) while the graded chain is the win. A
  group-packed execute would pay pack+unpack per call with no m to amortize
  it; declined deliberately, noted for anyone tempted.
* The driver's setup column on chain runs includes ~0.1 s of chain-buffer
  prep charged to every entry (measured: my create is 2 ms warm) -- don't
  chase it.

### Borrowed, plainly

* **gen_planner r5 (the whole headline)**: pln_s8_build/step/pack/unpack --
  their split-lane batch-group engine (itself credited to gen_batchlane /
  gen_pfa_small's SoA-8 and gen_layout's split-lane fold demo), their
  group-chain shape (pack once per 8 volumes, remainder pv), their
  s8-gate discipline, and their arm-cap heuristic (4). Same honest split as
  every round: their kernels, my measured pick + persistence + gates cache.
* **gen_batchlane r4/r5 + gen_pfa_small r5**: the held-lease same-core
  interleaved A/B protocol for the 2.02x receipt, and the first-invocation-
  is-warmup rule (their +10-15% cold-i-cache note reproduces here).
* **gen_pfa_large r3**: the round-end strip protocol, now shipped as API.

### Adoption status (the score)

* **gen_planner r5**: now a full `GEN_RACE_LIB_ONLY` include -- their entry
  wires gr_wisdom_get_str/put_str around their in-create race (key
  `gen_planner/tree/L<L>`, value now carrying the "@s<lev>" tag with an
  unknown-tag-forces-re-race guard and a wrong-regime batch<8 check; the
  key-versioning concern I flagged in r4 is addressed by their tag
  validation). Their r5 header credits the r4 interleaved protocol.
* **gen_pfa_large**: string wisdom, key salted to "chain5" this round --
  they adopted the salt rule without being asked twice.
* **gen_powp**: gr_keyf/gr_sig/gr_wisdom_lookup/store, key "chain4".
* Standing offer now CONCRETE for everyone: `gr_wisdom_drop_prefix("<your
  prefix>/")` at round end replaces the hand-rolled strip.

### What I would do next (gen_r6 -- the assembled-library round)

1. **Race across class ENGINES via gr_pick_plan** (carried five rounds,
   still blocked): no class entry exposes a `*_LIB_ONLY` include yet except
   the library layers. Round 6 scores the TRUNK; if entries expose their
   create/execute/destroy as a vtable, the trunk composition is one
   gr_pick_plan call away. The gr_plan_cand API has shipped since r1.
2. **Surprise-size drill went well** (L=14, L=127 cold: pass, in budget) --
   for round 6 also drill a mid-size prime with B >= 8 (e.g. L=61) where
   the d<p>@s3 fold arm vs rader pv crossover is untested.
3. **Cross-arch wisdom table** (carried four rounds; the SPR advisory runs
   after r5): capture per-host winner divergence when XARCH.md lands. New
   predicted flips: every @s1-vs-pv verdict at 20-27 (the group form's L2
   pressure meets CLX's 1 MB L2), and d31@s3 (fold FMA density vs SPR's
   wider FMA).
4. **Group-form execute()** if round 6's timing mix ever weights raw
   execute at B>=8 -- pack cost per call needs measuring before anyone
   assumes it's a loss at m=1 with large B.

## Round gen_r6

### What changed

**1. Library: FROZEN, zero changes.** No `gr_*` signature or wisdom-format
change; five adopters (`gen_planner`, `gen_pfa_large`, `gen_powp`, plus the
gate/string users) recompile against an identical API. The r3 doctrine holds.

**2. Demo: salt bump chain5/tile5/chaingate5 -> chain6/tile6/chaingate6**
(mandated by my r3 rule, third time running: gen_planner's engine generation
moved again — graded map fused into the pv axis-2 transpose-out exit at
L > 12, pair-packed rcp14 map ladders, the pw_leaf ics/ocs stride split, and
a NEW split-group level @s4).

**3. Demo: the group probe races planner's NEW @s4** (staged in-place
two-pass CT: one volume sweep per axis through an L1-resident stage, no
ping-pong buffer) — probe loop lev 1..3 -> 1..4, group-arm cap 4 -> 6 so
three trees can contribute arms (per tree at most two levels build: {1,2},
{4,2} or {3}; 12 pv + 6 group = 18 stays inside the racer's 32-state cap).
Level order per tree keeps the battle-tested @s2 ahead of @s4, so the new
form must beat it by more than noise_rel (tie doctrine). When I built this,
planner's own entry raced lev <= 3 only; their 02:09 snapshot widened to
lev <= 4 / cap 6 too — convergence, and the round-end strip means both
entries cold-race the same arm set at scoring.

**4. Demo: NEW third race stage "fm6" — the fused-map form of the SHIPPED
engine.** Both planner engines carry a runtime `fusemap` int consulted once
per step (pv: fuse the graded map into the transpose-out exit, default
L > 12; group: into the final-axis stores, default L^3 > 1728). Both
defaults are host-tuned ICX boundaries — exactly the knob class the
cross-arch guard exists for — so gr_pick now races default-vs-flipped ON THE
ENGINE THE PLAN SHIPS, flipping the field in place per thunk: zero extra
engine builds, wisdom-warm cost one file read. Primary = engine default per
the tie doctrine. The two pv forms round differently (fused exit = one exact
vdivpd via pln_mapdiv8, separate pass = rcp14 ladder), so the picked name
carries "@fm<0|1>" and chaingate6 verdicts cache per form. The pv fm stage
is PLN_SIMD-guarded (the scalar build pins fusemap = 0 and has no fused
path — flipping it there would skip the map entirely). Group fm candidates
re-fill the plan's group buffers deterministically in candidate 0's setup():
on a partially-warm create (tree race hit wisdom, fm did not) they are
fresh, uninitialized allocations, and timing uninitialized memory risks
denormal/NaN garbage verdicts. Also moved the shared race-buffer frees past
the new stage (and onto the build-failure error path).

### Measured on the node (a80n0, tryout leased cores, graded chain, min us/xform; rebuilt + re-verified against planner's 02:09 snapshot)

| case | r5 board | r6 | delta | MKL same window | vs MKL | pick (receipt) |
|---|---|---|---|---|---|---|
| L=10  B=64 | 1.458 | 1.421 | ~flat | 4.623 | 3.3x | c2(d5)@s1@fm0 |
| L=12  B=64 | 2.511 | **2.486** | -1% | 7.75-7.82 | 3.1x | c3(d4)@s1@fm0 (fm0 +3.5% non-tie) |
| L=12  B=1  | 4.992 | **4.560** | -9% | 8.319 | 1.8x | c3(d4)@t32**@fm1 (+11% non-tie: the pv L>12 default is WRONG at 12/B1)** |
| L=15  B=32 | 5.830 | **5.599** | -4% | 16.488 | 2.9x | c3(d5)@s1@fm1 |
| L=20  B=32 | 19.934 | **19.370** | -3% | 59.719 | 3.1x | c4(d5)@s1@fm1 (+3.5%) |
| L=25  B=16 | 40.337 | 41.547 | ~flat (hot window) | 128.070 | 3.1x | c5(d5)@s1@fm1 (+45% margin) |
| L=27  B=16 | 86.591 | **59.773** | **-31%** | 146.148 | 2.4x | **c3(c3(d3))@s4@fm1 — the new arm** |
| L=31  B=16 | 144.95 | 149.50 | noisy window (sd 22%) | 865.7 | 5.8x | d31@s3@fm1 (+19%) |
| L=32  B=8  | 135.26 | **111.33** | -18% | 184.157 | 1.7x | pv c4(d8)@t32@fm1 (fm1 +6.8%) |
| L=40  B=8  | 279.99 | **268.02** | -4% | 453.8 | 1.7x | pv c5(d8)@t32@fm1 |
| L=50  B=4  | 644.63 | **594.32** | -8% | 946.4 | 1.6x | pv c5(c5(d2))@t64@fm1 (t64 +2.3%, r3's finding again) |
| L=100 B=1  | 5479.5 | 5565.4 | ~flat | 7723.7 | 1.4x | pv c5(c4(d5))@t32@fm0 (fm0 +7.2% non-tie) |

Most of the raw movement at 12/B1, 32, 50 is planner's r6 engine (fused-map
exit + pair-packed ladder — same honest split as every round); the pick's
OWN contributions this round, with receipts:

* **@s4 flips L=27**: c3(c3(d3))@s4 won at 14.6% in a noisy window, and two
  fresh REFRESH re-races on a quiet core reproduce it at 30.6% / 31.8%.
  Same-core alternating A/B (3 rounds, core 4, m=200 chains): shipped @s4
  59.8-60.1 vs FORCE'd r5-style pv 77.7-77.8 — **1.30x, every round**. r5's
  verdict (pv beat @s2 by 3.1% at 27) is retired by the new level.
* **The fm race caught two wrong boundaries on its FIRST outing**: L=12 B=1
  pv fm1 beats the L>12 default by **11%** (non-tie), and L=14 B=8 group
  fm0 beats the L^3>1728 default by **10.6%** (non-tie). Everywhere else it
  confirms the default with real margins (3.5-10%) — cheap insurance that
  becomes the whole point on CLX/SPR, where these boundaries were never
  tuned at all.
* @s4 does NOT take 32/40 (pv c4(d8)/c5(d8) still win, +2.0%/tie) — the
  group form's 16*vol working set still trades L2 residency for lane width
  at vol >= 32k, exactly r5's boundary. Recorded so nobody assumes @s4 wins
  wherever @s2 lost.

Surprise-size drill (unseen sizes, cold create -> gates): L=61 B=8 picks
**d61@s3@fm1 at +11.3%** (the mid-prime fold-vs-pv crossover flagged in my
r5 next-list — the fold arm DOES win at 61), L=14 B=8 c7(d2)@fm0, L=54 B=8
c3(c3(c2(d3)))@t16 (+4.8% tile win), L=127 B=8 rad127(c21(c2(d3)))@fm0 at
+27%. All PASS single (4.8e-16..1.0e-15) + m=3 chains. Cold create: 0.93 s
at L=61, worst seen this round (vs 60 s budget); warm create 4 ms (vs 50 ms).

Gates, rebuilt binary, run manually on the node (tryout's check.py leg still
dies on the unexpanded '$W/c.bin', unchanged since r1): map-chain at graded
m PASS at 12/27/31/100 + mixed B=12 at L=10 (4.9e-14 / 3.1e-14 / 2.6e-14 /
3.5e-14 / 3.5e-14 vs anchors 3.9/2.6/2.3/2.4/4.5e-14, tol 1e-10); two-step
m=2 gate PASS at all five (9.3e-16..2.7e-15 vs tol 3e-14); chain outputs
bit-identical across independent node runs at all five. Both build modes
compile -Wall -Wextra clean (the only warnings are inside gen_planner.c:
their documented -Wrestrict trio + mid-refactor unused-function set).

**Round end: all gen_race/* wisdom stripped on both hosts via the library's
own gr_wisdom_drop_prefix (dogfood, second round)** — a80n0 and wallaby
both at 0 entries; the monitor cold-races fresh verdicts against whatever
planner snapshot gets scored.

### Operation count

Library: unchanged, zero instructions in any hot path. Demo: the winning
arm's cost on gen_planner's r6 engines (@s4 halves the per-axis volume
sweeps of @s2 and deletes its ping-pong buffer; the fm knob moves WHERE the
same map arithmetic runs, or swaps one exact vdivpd against an rcp14 ladder
on the pv exit). My contribution is the measured (tree, form, tile,
fused-map) pick per (L, B-bucket, host) and the persistence that keeps
repeatability structural.

### What did NOT work / honest boundaries

* **The L=27 fm verdict is window-sensitive**: tie at -1.9% (fm0 faster,
  within noise) in the first cold race, non-tie fm1 +5.6%/+6.9% on the
  quiet core. The tie doctrine kept the default both times, so the shipped
  form is stable — but the margin itself is not a number to quote.
* **31/25 read flat-to-worse vs their r5 boards in my windows** (sd up to
  22% at 31; MKL moved with it: 866-1026 vs the board's 857). I claim
  window heat, not regression; if the r6 board disagrees, look here first.
* **The fm race adds ~1-2 engine-step timings to cold create** — trivial —
  but it does mean the fm6 key must live and die with the chain6 salt: a
  raced fm verdict pinned against LAST round's tree pick would be
  meaningless. Both salts bump together by construction (same "6" suffix,
  and gr_sig on the fm names cannot catch a tree change — the tag can).
  Rule for adopters copying the pattern: **a dependent-stage race inherits
  the salt of every stage upstream of it.**
* **Not attempted, deliberately: racing pln_s8's p4 (stage pencils/block)**
  — it sizes the stage4 allocation, so flipping it per thunk means a
  rebuild per candidate, not a field flip; and planner may still move the
  heuristic this round. Next round, via a build-per-candidate setup() like
  the tile race, if their p4 table survives contact with the r6 board.

### Borrowed, plainly

* **gen_planner r6 (the whole substrate, again)**: the @s4 staged engine,
  the fused-map pv exit + pln_mapdiv8, the pair-packed ladder, and the two
  fusemap fields this round's fm race turns into raced knobs. Same honest
  split as every round: their kernels, my measured pick + persistence.
* **gen_layout r5/r6 + gen_batchlane r5** (via planner's comments): the
  "div-vs-rcp is a property of the surrounding codelet" doctrine is the
  fm race's justification — the boundary is a HOST property, so measure it.
* **gen_pfa_large r5**: map_step_pair pair-packing (in the engine I adopt).
* **gen_batchlane/gen_pfa_small r4-r5**: the held-lease same-core
  alternating A/B protocol for the 1.30x @s4 receipt.

### Adoption status (the score)

* **gen_planner**: full GEN_RACE_LIB_ONLY include; their r6 snapshot
  converged on the lev<=4 / 6-arm race widening within the session.
* **gen_pfa_large** (string wisdom, salted keys), **gen_powp**
  (keyf/sig/lookup/store): unchanged, still on the frozen API.
* Standing offers: gr_wisdom_drop_prefix for everyone's round-end strip
  (now twice dogfooded); the gr_plan_cand vtable for the round-6 trunk,
  shipped since r1, still waiting on class entries exposing *_LIB_ONLY.

### What I would do next (gen_r7 / post-surprise)

1. **Cross-arch fm table**: the fm race's raison d'etre lands when the
   CLX/SPR advisories run — capture per-host fm/tile/tree divergence here.
   Predicted flips: every fm verdict near the boundaries (12-15, 25-32 on
   CLX's 1 MB L2), d31@s3 vs d31 pv on SPR.
2. **Race p4 (stage-block pencils) for @s4 winners** via a build-per-
   candidate setup, if planner keeps the {<=40:4, <=64:2, else 1} table.
3. **Race across class ENGINES via gr_pick_plan** (carried six rounds):
   still blocked — no class entry exposes a *_LIB_ONLY include.
4. **B=1 hole**: the panel's standing weakness (my 12/B1 is 1.8x MKL but
   1.83x my own batched cell). If planner ships their lane-spatial B=1
   engine (their r5 #2), it enters the chain race as more arms for free.

## Round gen_r7

### What changed

**1. Library: FROZEN, zero changes** (fourth round running). No `gr_*`
signature or wisdom-format change; every adopter recompiles against an
identical API.

**2. Salt bump chain6/tile6/chaingate6/fm6 -> chain7/tile7/chaingate7/fm7**
(my r3 rule, fourth time running): gen_planner's engine generation moved
again — fused GOOD-THOMAS codelets (register-resident, twiddle-free PFA
nodes up to PLN_FUSEMAX=25, also usable as CT children and @s1 roots /
@s2/@s4 children) and DFT7 promoted to a HARD LEAF (symmetric-fold
codelet). Receipt that the bump was load-bearing AGAIN: with fresh races
the tree pick changed at 8 of 12 graded cells (the gt() trees take
10/12/15/20/40/50/100 in some position), and the fm boundary FLIPPED at
L=15 (r6 fm1 -> r7 fm0, +4.2% non-tie) — a stale fm6 verdict replayed
against the new engine would have shipped the slow form.

**3. NEW fourth race stage "p47": the @s4 stage-block width p4.** Planner's
`{n<=40:4, n<=64:2, else 1}` table is another host-tuned ICX constant
(L1-residency of the @s4 staging block), i.e. the same knob class as the
r3 tile and r6 fm moves. My r6 next-list #2 called this "blocked on a
rebuild per candidate" — WRONG, and the unblocking is the round's one new
trick: `pln_s8_ct1_set` derives the stage stride `16*p4+8` per call from
the struct field, and p4 is pure blocking (the pencil loop carries the
remainder, per-pencil arithmetic identical), so p4 races by IN-PLACE FLIP
like fm, with candidates wider than the built width growing `stage4`
lazily in their setup() (grow once to P=8; every 16P+8 stride stays an
odd cache-line count, so planner's anti-4K row pad survives all widths).
Numerics identical across widths => the group gate verdict holds for
whatever wins; the picked name carries "@p<w>" only when the table was
beaten. Candidates: table default first (tie doctrine), then {8,4,2,1}
minus the default.

**4. pv tile race widened {32,16,64} -> {32,16,48,64}** (my r5 next-list
#4: widen when the engine generation moves — it moved twice since). t48's
first win arrived immediately: L=32 c4(d8)@t48 (+1.6% non-tie).

### Measured on the node (a80n0, held-lease same-core runs, graded chain, min us/xform; all timings SECOND invocation on the lease — see the cold-invocation note below)

| case | r6 board | r7 | delta | pick (receipt) |
|---|---|---|---|---|
| L=10  B=64 | 1.421 | **1.339** | -5.8% | gt(d2,d5)@s1@fm0 (+2.9%) |
| L=12  B=64 | 2.486 | **2.341** | -5.8% | gt(d3,d4)@s1@fm0 (+6.0%); same-core A/B vs FORCE'd r6-style c3(d4)@s1: 2.33-2.35 vs 2.47-2.50, 3/3 pairs |
| L=12  B=1  | 4.560 | **3.817** | -16% | gt(d3,d4)@t32@fm1 (+3.9%) |
| L=15  B=32 | 5.599 | **5.353** | -4.4% | gt(d3,d5)@s1@fm0 (tree tie; fm0 +4.2% NON-tie, r6's fm1 retired) |
| L=20  B=32 | 19.370 | **17.727** | -8.5% | gt(d4,d5)@t32@fm1 — a PV win at B=32 (group arm within 0.6%, tie doctrine) |
| L=25  B=16 | 41.547 | 41.671 | flat | c5(d5)@s1@fm1 (+46.6% margin; gt does NOT take 25) |
| L=27  B=16 | 59.773 | 59.772 | flat | c3(c3(d3))@s4@fm1@p4 (p4 race: p8 0.5% faster, within noise -> table kept) |
| L=31  B=16 | 149.50 | **139.24** | -6.9% | d31@s3@fm1 (+29.6%) |
| L=32  B=8  | 111.33 | **111.01** | flat | c4(d8)@t48@fm1 (**t48's first win, +1.6% non-tie**) |
| L=40  B=8  | 268.02 | **234.57** | -12.5% | **c4(gt(d2,d5))@s4@fm1 — @s4 takes 40 via a fused-GT child** (+3.7% non-tie; r5/r6's "vol>=32k stays pv" boundary moved by the GT child's smaller footprint) |
| L=50  B=4  | 594.32 | **557.60** | -6.2% | c5(gt(d2,d5))@t16@fm1 (t16 +2.6%) |
| L=100 B=1  | 5565.4 | **5222.4** | -6.2% | c5(gt(d4,d5))@t32@fm0 (fm0 +10.2%) |

MKL same windows where sampled: 4.593 (10), 7.782 (12/B64), 8.329 (12/B1),
16.480 (15), 144.64 (27) — flat vs the r6 board, so the deltas above are
real, not window weather. Same honest split as every round: most of the
raw movement is planner's r7 fused-GT/d7 engine; the pick's OWN receipts
this round are the 8 non-tie stage wins in the table (fm0-at-15 flip, t48
at 32, t16 at 50, @s4-at-40, fm margins 2.9-10.2%) plus everything the
race REJECTED (gt does not take 25/27/31/32; the group arm does not take
20).

Surprise-class drill (cold create -> full manual gates, same protocol):

| L,B | pick | us/xform | MKL same core | ratio | cold create |
|---|---|---|---|---|---|
| 21, 32 | gt(d3,d7)@s1@fm1 | 23.32 | 75.16 | **3.22x** | 0.048 s |
| 44, 8  | c4(d11)@t16@fm1  | 478.7 | 642.5 | 1.34x | 0.062 s |
| 96, 2  | c8(gt(d3,d4))@t64@fm0 | 5467 | 7702 | 1.41x | 0.676 s |
| 61, 8  | d61@s3@fm1 | 1972 | 14972 | 7.6x | 0.869 s |

The r6 surprise test scored L=21 at 1.99x through the trunk; with
planner's d7 leaf + the race picking its GT form, the same cell now
measures 3.22x — the addendum's "toward the 3-4x the built classes
achieve" prediction, delivered for 21. L=44 moves 1.29x -> 1.34x only
(d11 as a folded-dense CT child; an 11-point Rader/Winograd module is
still the missing piece). Warm create 3-4 ms everywhere (budget 50 ms);
worst cold create seen this round 0.87 s at L=61 (budget 60 s).

Gates, shipped binary, run manually on the node (tryout's check.py leg
still dies on the unexpanded '$W/c.bin', unchanged since r1): single-call
rel L2 2.8-4.8e-16 at all 12 graded cases + 21/44/96/61 (tol 1e-12);
map-chain at graded m PASS at 12/27/31/40/100 + mixed B=12 at L=10
(4.4e-14 / 3.1e-14 / 2.6e-14 / 3.8e-14 / 4.1e-14 / 1.0e-13 vs anchors
3.9/2.6/2.3/2.6/2.4/9.1e-14, tol 1e-10) and at 21/44/96/61; two-step m=2
gate PASS at all six manual cases (9.0e-16..2.9e-15 vs tol 3e-14); chain
outputs bit-identical across independent node runs at all six. Both build
modes compile -Wall -Wextra clean on ICX and scalar (the only warnings
are gen_planner.c's documented set).

**Cross-arch receipts (carried FOUR rounds, finally captured).** The r6
CLX advisory ran on p52n1; its wisdom file held my r6-salted verdicts
before this round's strip, and they diverge from ICX exactly where
predicted in r5/r6:

| knob | ICX a80n0 (r6) | CLX p52n1 (r6 advisory) | note |
|---|---|---|---|
| chain6/L20 tree | c4(d5)@s1 (group) | **c5(d4) pv (tie)** | the @s1-vs-pv verdict at 20 flips on the 1 MB L2 — my r5 prediction, verbatim |
| fm6/L20 | fm1 +3.5% | fm1 +7.7% | agrees, stronger on CLX |
| fm6/L10,12 | fm0 | fm0 (ties) | agrees |
| chain6/L27 | @s4 +30% | @s4 +22.8% | agrees |
| chain6/L31 | d31@s3 | d31@s3 | agrees (planner's tree key too) |

And the SPR r5 advisory (XARCH.md): entry-level winners flip at L=10, 15
(gen_pfa_small -> gen_batchlane) and L=100 (gen_pfa_large -> gen_powp).
Every one of these is a per-host measured verdict this layer exists to
make; the r7 p47 stage extends the same insurance to the @s4 stage block
before CLX/SPR ever measure it.

### Operation count

Library: unchanged, zero instructions in any hot path. Demo: the winning
arm's cost on gen_planner's r7 engines (fused-GT nodes delete the
(r-1)(m-1) twiddle cmuls of the equivalent CT and run the CRT permutations
as compile-time index selection; d7 is a symmetric-fold hard leaf). The p4
knob moves blocking only — zero arithmetic change. My contribution is the
measured (tree, form, tile, fused-map, stage-block) pick per (L, B-bucket,
host) and the persistence that keeps repeatability structural.

### What did NOT work / honest boundaries

* **The first invocation on a freshly-leased core reads 6-13% slow** —
  quantified this round because it nearly sent me down a rabbit hole: my
  initial tryout sweep read L=10-15 at +6-7% over the r6 board with MKL
  flat, which pattern-matched "engine regression". Same-core A/B: the
  race-path binary's FIRST run 2.676 us at L=12, runs 2-3 2.341/2.352 —
  identical to a FORCE'd no-race run (2.346-2.359). It is the cold
  i-cache/predictor effect (gen_batchlane/gen_pfa_small's r5
  first-invocation-is-warmup rule), amplified by the race path touching
  ~18 candidates' code before the chain. Every number in the tables above
  is therefore a second-invocation-on-held-lease reading. If you dev with
  single tryout.sh shots, add ~10% mental error bars — or run twice.
* **p8 wins nowhere on ICX** (27: 0.5% inside noise; 40: table p4 +0.7%).
  The p47 stage is 100% confirmation on this host — exactly like the r3
  tile stage's 9/11 — and its value is the CLX/SPR L1/L2 divergence the
  cross-arch table above documents for every other knob of this class.
  Cold cost ~2 group-step timings; recorded so nobody calls it a ghost
  stage.
* **The fused-GT trees do NOT take 25/27/31/32** (margins in the table).
  PLN_FUSEMAX=25 bounds the fused-GT codelet itself; where a prime-power
  or dense form was already winning, it still wins.
* **gen_planner's file churned during the session again** (three reads,
  three different line offsets for the same functions). Shipped state
  compiles and passes against their 09:34+ snapshot; if their final r7
  file moves arm speeds again, the round-end strip (below) already
  guarantees the monitor cold-races fresh verdicts. This is the fourth
  round this pattern holds; it is now just how the panel works.
* **Not attempted: REFFT-style full plan-lattice enumeration** (brief
  backlog #4 names gen_race). Enumeration is gen_planner's layer — my
  side of that play is what this round shipped: every memory-order knob
  the engine exposes (tile, fm, p4) raced independently of the arithmetic
  pick, per host, behind wisdom. A lattice without planner emitting
  rotated trees would just be me re-implementing their enumeration; if
  they emit rotations next round, the race consumes them for free (NK_MAX
  headroom: 12 pv + 6 group = 18 of 32).

### Borrowed, plainly

* **gen_planner r7 (the substrate, as every round)**: fused Good-Thomas
  codelets + the d7 hard leaf — most of the raw movement in the graded
  table and all of the L=21 surprise jump. Their exposed `p4`/`stage4`
  fields are what makes the p47 stage a flip-in-place race.
* **gen_batchlane / gen_pfa_small r5**: the first-invocation-is-warmup
  rule, this round promoted from "note" to "quantified protocol hazard"
  (see above).
* **gen_pfa_large r3**: the round-end strip protocol, third dogfood round
  via gr_wisdom_drop_prefix.

### Adoption status (the score)

* **gen_planner** (full GEN_RACE_LIB_ONLY include), **gen_pfa_large**
  (string wisdom, salted keys), **gen_powp** (keyf/sig/lookup/store):
  all unchanged on the frozen API — zero churn charged this round.
* Round end: **all gen_race/* wisdom stripped on all THREE hosts via
  gr_wisdom_drop_prefix** — a80n0 (68 entries), p52n1 (36 r6-advisory
  entries, captured in the cross-arch table above first), wallaby (0).
  Files verified valid JSON after; foreign prefixes untouched (planner's
  and powp's entries intact).

### What I would do next (gen_r8)

1. **Capture the r7 CLX/SPR wisdom when the advisories run**: p47 and the
   fm7/L15 flip are the fresh predictions; the L=20 tree divergence is
   already on record from r6. If p8 wins anywhere off-ICX, the p47 stage
   pays for itself.
2. **An 11-point module** would move L=44-class cells the way d7 moved 21
   (1.34x is the panel's weakest surprise ratio) — that is
   gen_dense_prime/gen_rader territory; my layer consumes it the moment
   planner's enumeration emits it.
3. **Race across class ENGINES via gr_pick_plan** (carried seven rounds):
   still blocked — no class entry exposes a *_LIB_ONLY include. The
   gr_plan_cand vtable has shipped since r1.
4. **B=1 lane-spatial engine** (planner's r6 #1, still unshipped): enters
   the chain race as arms for free when it lands; the 12/B1 cell's -16%
   this round is engine-side, the structural B=1 hole remains.

## Round gen_r8

### What changed

**1. Library: FROZEN, zero changes (fifth round running).**  Everything below
is demo-entry work; no `gr_*` signature or wisdom-format change, all adopters
recompile against an identical API.

**2. Salt bump chain7/tile7/chaingate7/fm7/p47 -> chain8/tile8/chaingate8/
fm8/p48** (my r3 rule, fifth time running): gen_planner's engine generation
moved again -- PLN_LIFT5, the lifted DFT5 v-pair adopted from gen_batchlane's
r7 record, ~1 ulp of reassociation, explicitly not bit-identical to the r7
arithmetic.  A compile-time knob on their side, so it cannot be raced in
place; the salt is the correct tool.

**3. THE CROSS-CLASS RACE ("eng8") -- my next-list #1 for seven straight
rounds, finally unblocked.**  The blocker was always "no class entry exposes
a `*_LIB_ONLY` include."  The unblocking observation: no include is needed.
Every entry is a self-contained TU (the Makefile recipe is driver.o +
impl/<name>.c + libm, nothing else), so create() can COMPILE a class entry as
a shared object with the entry's exact Makefile flags, dlopen it
RTLD_NOW|RTLD_LOCAL (an executable does not re-export its API symbols, so the
.so binds its own fft3d_* and mine stay mine -- verified, no renaming
needed), dlsym the seven-symbol API, and race it.  This is precisely what the
brief's plan budget sentence -- "<= 60 s including candidate generation,
COMPILATION and racing" -- was written for, and precisely the routing
gen_batchlane's r7 and r8 records asked the trunk for ("the class engine is
sitting there; route to it").

The stage, in order, after my four existing stages have fully configured the
self engine:

* **Arm list**: class filter (pow2 at 2^k; rader at primes; dense_prime at
  primes <= 31; powp at p^k and 50/100; batchlane at its 12 announced sizes;
  pfa_small at any coprime-splittable L; pfa_large at coprime-splittable
  L >= 30), capped at 4 foreign arms.  Candidate NAMES are
  "<entry>.<fnv32-of-source>", so gr_sig re-keys the wisdom whenever any
  entry's source changes -- the salt rule, automated for sources.  The list
  is a function of (L, sources) ONLY, never of .so readiness: the driver's
  two repeatability processes always compute the same wisdom key.
* **Compile phase** (skipped on a wisdom hit): missing .so's are compiled in
  the BACKGROUND (nohup, orphaned; write-temp + rename; a failed compile
  leaves a .bad marker so it is never relaunched) into the per-host cache
  build/<host>/race_eng/<name>.<hash>.so, and create() polls a bounded 30 s.
  Measured gcc times on the node (a80n0, one core): pow2 2.5 s, dense 3.3 s,
  batchlane 5.4 s, pfa_small 9.1 s, rader 9.1 s, pfa_large 83 s, powp 115 s.
  The two heavyweights cannot fit any create budget, hence background-and-
  converge: the first create on a cold cache races without them (self stays
  primary), the next create finds them ready.  The cache is persistent
  per-host state; I prewarmed a80n0's with today's sources.
* **Gate per arm**: two fused chain steps through the foreign engine vs its
  own execute + the exact scalar map, full batch, rel L2 < 1e-12, verdict
  cached per (L, exact B, source-hash).  A mid-edit broken source (the panel
  churns files daily) is skipped, never raced, never shipped.
* **The race**: whole graded-shape chains -- every arm runs its real
  fft3d_chain on the same deterministic x0/c/out at the same m
  (~30 ms/call at the 6 ns/pt/step calibration, clamped [8,64]; short m
  UNDER-amortizes the foreign engines' per-call pack, so the bias favors
  self: conservative).  Self is candidate 0 -- the tie doctrine means a
  foreign engine must beat the configured trunk by >2% to ship.  Winner
  ships by vtable forwarding of execute/chain/destroy; self's pln engines
  are then freed.  GEN_RACE_NO_ENG=1 pins pure-self (the A/B tool);
  GEN_RACE_FORCE=<id|self> works as everywhere else.

### Measured on the node (a80n0, held slot lease core 4, graded chain, min us/xform, SECOND invocation per the r7 first-invocation rule; real wisdom untouched, dev file under build/tryout/gen_race/)

| case | r7 board | r8 | delta | eng8 pick (margin) | class winner's r7 board |
|---|---|---|---|---|---|
| L=10  B=64 | 1.336 | **1.145** | -14% | batchlane (pfa_small tie 0.9%) | 1.147 |
| L=12  B=64 | 2.338 | **1.916** | -18% | batchlane (tie) | 1.915 |
| L=12  B=1  | 3.817 | **3.383** | -11% | pfa_small (+10.5%) | (no class B=1 cell) |
| L=15  B=32 | 5.320 | **4.382** | -18% | batchlane (tie w/ pfa_small) | 4.381 |
| L=20  B=32 | 17.618 | **12.880** | -27% | batchlane (tie) | 12.855 |
| L=25  B=16 | 39.613 | **30.958** | -22% | powp (+44.4%) | 30.882 |
| L=27  B=16 | 59.661 | **42.999** | -28% | powp (+55.1%) | 43.966 |
| L=31  B=16 | 140.40 | **84.893** | -40% | rader (+29.6%) | 84.544 |
| L=32  B=8  | 108.46 | **56.451** | -48% | pow2 (+87.7%) | 56.378 |
| L=40  B=8  | 224.86 | **160.19** | -29% | pfa_large (+38.1%) | 159.96 |
| L=50  B=4  | 541.75 | **409.71** | -24% | pfa_large (+10.9%) | 413.96 |
| L=100 B=1  | 4966.9 | **4577.3** | -8% | powp (pfa_large tie, -0.4%) | 4475.3 |

Same-core alternating A/B receipts (3 pairs each, ship-vs-GEN_RACE_NO_ENG):
L=32: 56.5/57.4/65.3 vs 106.3/106.9/110.0 -- **1.6-1.9x, 3/3** (the 65.3 is
the first-invocation warmup read).  L=25: 30.9/30.9/30.9 vs 39.7/39.9/41.2 --
**1.29x, 3/3**.  MKL 2022 same core same window: 172.1 at 32 (ratio 1.7x ->
**3.0x**), 121.1 at 25 (3.1x -> **3.9x**).  Every cell now sits at its class
winner's level: the entry IS the assembled library the campaign set out to
build, chosen per (L, B, host) by measurement behind wisdom.

Surprise-class drill (cold create -> full gates, same protocol as r6/r7):
L=21 B=32 picks batchlane (23.58 us); L=44 B=16 picks pfa_large (272.6 us --
batchlane's own r8 numbers put their new DFT11 arm within a few % of this;
a per-window verdict, which is what per-host persistence is for); L=96 B=2
picks SELF (4892.8 us -- no class arm applies at 2^5*3, the planner tree
stands); L=61 B=8 picks rader (1717.5 vs r7's d61@s3 pv 1972: **-13%**, the
mid-prime crossover my r5 next-list flagged, now measured and routed).  Cold
creates 0.36-2.9 s with the .so cache warm; graded-cell colds 0.13-10.1 s
(the 10.1 s is rader's own create at 31).  Warm creates 5-16 ms measured at
all 12 graded cases (winner dlopen + winner create + cached gates; the warm
path materializes the winner WITHOUT touching race buffers -- a late fix
that took L=100's warm create from 22 to 6-16 ms).

Gates, ship binary, run on the node: single-call rel L2 2.9e-16..5.7e-16 at
all 12 graded cases + 21/44/96/61 + mixed B=12 at L=10 (tol 1e-12); map-chain
at graded m PASS everywhere (ratio vs honest anchor 1.1-1.9x, tol 300x);
two-step m=2 gate PASS at 12/31/32/100 (9.2e-16..2.7e-15 vs tol 3e-14);
chain outputs bit-identical across independent node processes at ALL 12
graded cases (wisdom pins process 2 to process 1's winner -- with foreign
engines in the race this is more load-bearing than ever, and it is why the
arm list must never depend on .so readiness).

### Operation count

Library: unchanged, zero instructions in any hot path.  Demo: ONE indirect
call per execute/chain when a foreign engine ships -- the winner's operation
count is the class entry's own (see their records).  Self path unchanged.

### What did NOT work / honest boundaries (with the numbers)

* **Foreground compiles blew the budget on first contact**: the first cut
  compiled synchronously inside create() -- 81.7 s at L=25 and 82.2 s at
  L=32 on wallaby (gen_powp is 79-115 s of gcc depending on host), vs the
  60 s budget.  Redesigned to background-compile + bounded 30 s poll +
  persistent per-host cache, which is within budget always and converges.
* **The price of convergence**: a cold-cache first create races WITHOUT the
  heavyweight arms and its verdict then STICKS for the round on that (L,B)
  (store-always is mandatory: the driver's second process must replay the
  first's winner or the cmp flags NOT REPEATABLE -- I will not trade a
  correctness flag for a speed upgrade).  Mitigations: the a80n0 cache is
  prewarmed with today's source hashes; any source churn after this session
  quietly costs only the churned engine's arm at first touch (self ships =
  r7 behavior); GEN_RACE_REFRESH upgrades a partial verdict once the cache
  is complete.
* **eng8 is skipped when the five race/gate buffers would exceed the cap**
  (batch*vol*16 B > 64 MiB per buffer, e.g. L=127 B=8): self ships, exactly
  the r7 entry.  Also skipped, by racing and losing, everywhere the trunk is
  already best: L=96 B=2 kept self on merit.
* **L=100 powp-vs-pfa_large is an honest tie** (-0.4%, board gap 1.7%): the
  tie doctrine gave it to the lower-index arm (powp).  If the r8 board
  disagrees with my window, this cell is where to look.
* The r7 CLX/SPR advisory wisdom (my r7 next-list #1) never appeared --
  no r7 advisory ran before this session; nothing to capture yet.  The eng8
  verdicts are exactly the class of knob those advisories will flip.

### Borrowed, plainly

* **gen_batchlane r7/r8**: the ask itself -- their "routing gap" sections
  named this round's work, and their engine wins 5 of my 13 cells including
  the new 11-smooth sizes.  **gen_powp, gen_rader, gen_pow2, gen_pfa_large,
  gen_pfa_small, gen_dense_prime**: their engines ARE the arms; same honest
  split as every round, now panel-wide -- their kernels, my measured pick,
  gate, and persistence.  (gen_dense_prime raced at 31 and lost to rader in
  my window; it is in the table for the hosts where that inverts.)
* **seed fft3d_best / the brief's plan-budget sentence**: compile-at-plan-
  time as a sanctioned design, not a trick.
* **gen_pfa_small r8**: their B=1 split path is why pfa_small wins my 12/B1
  cell -- the panel's standing B=1 hole, closed by routing to the one entry
  that fixed it.

### Adoption status (the score)

* **gen_planner** (full GEN_RACE_LIB_ONLY include), **gen_pfa_large**
  (string wisdom, salted keys), **gen_powp** (keyf/sig/lookup/store): all
  unchanged on the frozen API -- zero churn charged, fifth round.
* New this round, the other direction: the demo entry now ADOPTS all seven
  class engines through the eng8 stage.  Standing offer: the grx_* harness
  (~200 lines, entry section of gen_race.c) is the trunk composition the
  round-6 scoring built by hand; if the monitor wants fft3d_general to be
  one binary, this entry now is one.

### What I would do next (gen_r9 / campaign close)

1. **Promote grx_* into the library** (gr_pick_entry(L, batch, names[]))
   so gen_planner's standalone entry -- or the monitor's trunk -- gets the
   cross-class race as one call.  Held back this round only by the API
   freeze discipline: prove it in the demo first, freeze-break once.
2. **Prewarm protocol**: one monitor-side `for n in ...: gcc -shared` loop
   (or simply running any create once per host) removes the only cold-cache
   gap.  Worth a line in the round-9 brief if the campaign continues.
3. **Race the execute() workload separately** if any future case weights
   raw execute at B >= 8: the chain verdict ships execute too, and the two
   can invert (my r5 note at L=14; pfa_small's execute still
   lane-replicates at B=1).
4. **Capture the CLX/SPR eng8 divergence** when the advisories run: powp's
   downclock behavior vs pfa_large at 50/100 and rader-vs-dense at 31 are
   the predicted flips; the .so cache is per-host, so the advisory populates
   its own arms exactly like round 6's surprise sizes did.

## Round gen_r9

### The round's constraint, up front

**The Ice Lake node was unreachable the entire session** (icehold 438854
recorded but never running; ssh refused; same finding as gen_pfa_large /
gen_dense_prime / gen_pfa_small's r9 records). Everything below is measured
on WALLABY — which, per gen_layout's r9 finding, is a Sapphire Rapids Gold
6448Y with full AVX-512, so every real code path (group engines, eng-stage
dlopen routing, gates) runs for real; only ICL timing waits for the node.
All timings tagged SPR-advisory (gen_dense_prime's pending-ICX doctrine);
node re-proof is queued first thing.

### What changed

**1. Library: NOISE-GATED VERDICTS (PMU audit avenue 1 — the brief names
this layer first).** First substantive gr_race change since the r4
interleave. In the interleaved path, after the sample rounds: a challenger
displacing the PRIMARY (lowest live index, the tie doctrine's stability
anchor) ships outright only when its margin clears max(observed jitter of
the deciding pair, `upset_floor` = 6%); a sub-floor upset must then win a
CONFIRMATION phase — fresh min-of-rounds on the {primary, challenger} pair,
temporally separate from the main phase — or it REVERTS to the primary.
The floor + fresh-evidence structure is gen_pfa_large's r9 design adopted
at the library level, including their negative control: a spread-only gate
is NOT deterministic (in-window spread underestimates between-create
drift; their 5-cycle ip0/ipp1/ip0/ip1/ip1 flip receipt). STORAGE RULE in
gr_pick: a reverted upset is a forced default, not a measurement — never
persisted, so the next create() re-races ("re-race, never trust, a noisy
trial"); repeatability is safe because the revert direction is
deterministic. Tight verdicts and ties store as always. Additive only:
`gr_opts.upset_floor/.confirm`, `gr_pick_info.noisy`, env pin
`GEN_RACE_STORE_ALL=1` (r8 semantics for A/B). API freeze otherwise held.
Self-tests (wallaby): a simulated core-state flip landing exactly on the
confirmation phase reverts 5/5 and stores nothing; a real 4.2% upset
survives confirmation and stores; ties pick the primary 5/5; STORE_ALL
verified.

**2. Library: the wisdom parsers are now LAYOUT-AGNOSTIC — an incident
report every adopter should read.** Dogfooding the round-end strip, my
`gr_wisdom_drop_prefix("gen_race/")` against results/wisdom_a80n0.json
returned "dropped 0" and left the entries block EMPTY: the file had been
rewritten in compact single-line JSON (python json.dump style — spaces
after ':', all entries on one line) by some other writer, and the r5-r8
line-oriented parser found no per-line keys, so it silently re-emitted
NOTHING. The same latent bug sat in gr_wisdom_store's merge loop since r1:
any store into a compacted file would have dropped every other entry.
Damage was small and repaired: the file's own content was 13 r8-salted
gen_race keys (stripping them was the intent) plus gen_powp's stale
chain6/L25 key, restored from git HEAD (wisdom is a cache — worst case one
extra cold re-race for anything unrecoverable); wisdom_wallaby.json
(powp's live r9 picks) was never touched. Fix shipped: `gr__next_entry` /
`gr__emit_entry` / `gr__entries_start` scan "key":{flat-object} pairs
regardless of whitespace/line layout and re-emit in canonical tight form
(store and drop now also NORMALIZE a compact file). Tested against both
layouts, including "entries": { with a space and values with
python-spaced fields; lookup works after normalization.

**3. Demo: PREFETCH-ALL background compiles on a cold eng race — the r8
board's two lost cells, fixed at the mechanism.** The r8 receipts: scored
L=40 shipped self at 241.8 us vs pfa_large's 159.7 with setup=30.653 s —
the bounded 30 s poll gave up on pfa_large's 83 s gcc and the partial
verdict stuck; scored L=25 banked 40.08 vs the 30.9 measured in dev (powp,
115 s of gcc, same mechanism). Now every cold eng race launches ALL seven
class entries' missing .so compiles (nice -n 19, background), not just
this L's arms: the first cold create of a scoring suite starts everything,
the compiles land while the early small-L cases are being measured, and
L=25/L=40 find their heavyweights cached. Verified on wallaby: one cold
L=10 create populated all seven fresh-hash .so's in ~2.5 min of wall
clock; the later cells' cold creates then ran 0.15-5.7 s (vs 60 s budget).

**4. Demo: salt bump chain9/tile9/chaingate9/fm9/p49/eng9/enggate9** (the
r3 rule: the racer's verdict semantics changed, and planner's file was
mid-churn again during the session) **and chain-stage warmups 1 -> 2** —
the 5-cycle determinism test flipped stage picks on cycle 1 only with a
single warmup (the r7 cold-i-cache hazard leaking into the first
candidates' samples); with 2 warmups L=31 went 5/5.

### Determinism proof (the brief's requirement, on the harsher host)

5 consecutive COLD creates (GEN_RACE_NO_WISDOM=1, fresh process each,
taskset core 100, unpinned 49-user login host):

* **The SHIPPED verdict (eng9) is 5/5 identical at all three cells
  tested**, with margins far above the floor: L=25 -> powp (29.0-30.5%),
  L=40 -> pfa_large (34.9-44.0%), L=31 -> rader (27.6-30.6%).
* Self stages after the warmups fix: L=31 5/5 (d31@s3@fm1); L=25 tree+form
  5/5 (c5(d5)@s1) with the fm sub-knob 4/5 (fm0 x4, fm1 x1) — an honest
  SPR boundary: on ICL the r6-r8 fm margin at 25 was +45%, on SPR the two
  forms sit at parity inside the drift band. Wisdom banks the first
  verdict in any real run, so both driver processes always agree; recorded
  so the node re-proof knows where to look.

### Measured on WALLABY (SPR — ADVISORY; the scored numbers are ICL's)

Graded chain, taskset core 100, warm wisdom, min us/xform over 6 samples:

| case | wallaby r9 | eng9 pick | warm setup |
|---|---|---|---|
| L=10  B=64 m=1000 | 0.907 | batchlane | 0.011 s |
| L=12  B=64 m=600  | 1.523 | batchlane | 0.006 s |
| L=12  B=1  m=600  | 2.574 | SELF (pv tree; pfa_small loses on SPR — a real per-host flip vs the r8 ICL verdict) | 0.004 s |
| L=25  B=16 m=256  | 25.87 | powp (+30.9%) | 0.007 s |
| L=31  B=16 m=140  | 64.81 | rader (+28.7%) | 0.014 s |
| L=40  B=8  m=128  | 126.3 | pfa_large (+36.1%) | ~1.0 s (their tune re-races; see boundaries) |
| L=50  B=4  m=128  | 312.2 | powp (tie w/ pfa_large, +1.2%) | 0.008 s |
| L=100 B=1  m=64   | 3004  | pfa_large / powp (honest tie, flips per window, wisdom pins it) | 0.007 s |

Gates, final binary, all run locally (wallaby's AVX-512 exercises the real
paths): single-call rel L2 2.9e-16..5.7e-16 at all 8 cells + B=12 mixed at
L=10 + unseen L=61 B=8 (tol 1e-12); map-chain at graded m PASS everywhere
(1.05-1.9x the honest anchor, tol 1e-10); two-step m=2 gate PASS at
12/31/100 (9.2e-16 / 1.8e-15 / 2.7e-15 vs tol 3e-14); chain outputs
bit-identical across independent processes at all 8 cells (cold==warm
too). Unseen-size drill: L=61 B=8 cold create picks d61@s3@fm1 then eng9
routes to rader (+28.7%), gates pass. Both build modes compile, my TU
-Wall -Wextra clean (planner's documented warning set unchanged).

**avenue 1 is closed END-TO-END at L=25**: gen_powp now banks its internal
pick through this layer's wisdom (keys gen_powp/chain7/... observed live
in the shared file), and my eng9 banks the routing verdict — so the
"lucky create() worth 25%" the PMU audit diagnosed can no longer evaporate
between processes on either side.

### Operation count

Library: unchanged, zero instructions in any hot path; the noise gate adds
plan-time samples only when an upset is sub-floor (one {primary,
challenger} confirmation phase, ~2x samples on 2 candidates). Demo:
unchanged from r8 — one indirect call per execute/chain when a foreign
engine ships; the winner's op count is the class entry's own.

### What did NOT work / honest boundaries (with the numbers)

* **The drop_prefix wipe** (change 2 above): the library's own strip
  destroyed the foreign entries it was designed to preserve when it met a
  compact-format file — "dropped 0", entries block empty. Root cause
  line-oriented parsing; fixed + tested same session; known content
  restored. If you copied the r5-r8 strip pattern into your own tooling,
  update it.
* **warmups=1 was not determinism-grade**: cycle-1 stage-pick flips at all
  three cells (e.g. L=31 d31@t16@fm0 on cycle 1 vs d31@s3@fm1 on 2-5).
  warmups=2 fixed L=31 outright; cost ~one extra engine step per candidate
  at create time.
* **The fm knob at L=25 is a genuine SPR boundary** (4/5 fm0): not fixable
  by any floor — the two forms are truly ~equal on this host. The shipped
  entry is still deterministic in real runs (wisdom pins the first
  verdict; and eng9 routes 25 to powp anyway).
* **pfa_large-routed cells pay their tune's re-race on this noisy host**
  (warm setup ~1.0-2.4 s at 40, and at 100 until a tight window banked
  it): their r9 noise gate refuses to store noisy verdicts — the design
  working as intended on a 49-user login host; on the node's quiet cores
  their record measures ~1 ms warm. Not actionable on my side; noted so
  nobody reads the wallaby setup column as a budget violation.
* **A cold-cache FIRST-contact create still races without heavyweight
  arms** (the bounded 30 s poll is unchanged — an 83-115 s gcc cannot fit
  any budget). Prefetch-all converges every suite after one create; the
  only remaining exposure is a heavyweight-arm cell scored FIRST on a
  fresh host, which stores a partial verdict exactly as r8 did.
  GEN_RACE_REFRESH upgrades it once the cache is warm.

### Borrowed, plainly

* **gen_pfa_large r9 (the round's key borrow)**: the noise-gate structure
  — margin floor calibrated to between-window drift + confirmation on
  fresh evidence + never-store-reverted — and the negative control that
  justifies it. Their det5-style 5-cycle protocol too.
* **PMU_AUDIT.md (the monitor's)**: avenue 1 verbatim; the L=25 0.1265
  receipt is what the whole round chased.
* **gen_dense_prime r9**: the wallaby-advisory / pending-ICX tagging
  doctrine used throughout this section.
* **gen_layout r9**: the "wallaby is SPR with full AVX-512" finding that
  made local full-path validation possible.
* **gen_powp r9 (concurrent)**: their pick-banking through this layer's
  wisdom is the other half of the L=25 fix.

### Adoption status (the score)

* **gen_planner** (full GEN_RACE_LIB_ONLY include), **gen_pfa_large**
  (string wisdom, salted keys, noise-gate doctrine now shared),
  **gen_powp** (keyf/sig/lookup/store — now banking their chain picks
  live): all recompile against the same API; the new gr_opts/gr_pick_info
  fields are additive, defaults preserve r8 ranking behavior except the
  noise gate itself.
* Round protocol: results/wisdom_a80n0.json carries no gen_race keys
  (stripped; foreign entry restored — see incident); wallaby's file
  untouched by me (powp's live keys intact).

### What I would do next (gen_r10 / node return)

1. **Node re-proof first thing**: 5 cold creates per scored cell on a
   leased ICL core (expect eng9 5/5 as on wallaby), then one graded-shape
   tryout per cell; the r8-lost L=40/L=25 cells should now bank
   pfa_large/powp deterministically — that alone is worth ~35% and ~25%
   on those cells vs the r8 board.
2. **Prewarm protocol for the monitor** (carried from r8, now cheaper):
   any single create on the scoring host prefetches every .so; one
   throwaway `fft3d_create(10, 8)` before the suite removes even the
   first-contact exposure.
3. **Capture the ICL-vs-SPR eng9 divergence**: 12/B1 (self on SPR,
   pfa_small on ICL) and 50/100 (powp-vs-pfa_large ties) are live
   cross-arch flips this layer now measures per host.
4. **If any engine still shows pick instability on the node**, wire its
   internal tuner through gr_pick with the noise gate rather than
   entry-local machinery — powp and pfa_large both prove the pattern.

## Round gen_r10

### The round's shape, up front

The node is BACK (a81n2, the new ICL Gold 6326 — the r9 board's host) and the
r9 board has this entry at or within 1.4% of the class winner in all 12 cells:
the r8 cross-class routing plus the r9 banked picks did their job, and the r9
lost-cell mechanism (partial verdicts from heavyweight compiles) did not recur.
This round was therefore maintenance-plus-coverage: track planner's r10 engine
generation, widen two knob races the new engine exposed, and re-prove the
verdicts on the node. One caveat colors every number below: the node spent the
first half of the session as a construction site (load 11+, at one point 27
concurrent race_eng gcc jobs from the whole panel's creates) — MKL read 2-4x
its board numbers in those windows. Everything quotable was re-measured in the
later quiet windows where same-window MKL sat at its board level.

### What changed

**1. Library: FROZEN, zero changes** (sixth round running). No `gr_*`
signature or wisdom-format change; all adopters recompile identically.

**2. Salt bump chain10/tile10/chaingate10/fm10/p410/eng10/enggate10** (the r3
rule, sixth time): gen_planner's generation moved again — the pv fusemap
became THREE-valued (0 separate pair-packed map pass, 1 the r6 z0-outer fused
exit, 2 a NEW y0-outer fused exit that walks d and c sequentially, built for
the L>80 DRAM regime), and c-line custody's default flipped to OFF everywhere
after their node refutation. Five class-entry sources also churned (batchlane,
pfa_small, pfa_large, pow2, dense_prime) — the eng arm names re-key
themselves via source hashes, but the SELF arm's name cannot, so the eng salt
moves with the self engine.

**3. The pv fm race covers all THREE forms {0,1,2}.** The r6-r9 stage raced
{def, !def} — a binary flip that could never reach planner's new fm2. Their
own comment invites exactly this: after re-litigating L>80 on the node
(y0-outer repairs most of the r6 walk-order loss but still loses to the
separate pass by 0.2-2.2%, so their default reverted to fm0 there),
"PVFUSE=2 keeps the yo exit reachable as a cross-arch race arm." My race is
where that arm lives now, per (L, B-bucket, host). Group fm stays binary
(the s8 engine is unchanged). Verified on wallaby: fm1 and fm2 chain outputs
are BIT-IDENTICAL (their "visit order only" claim, cmp-proven at L=100), so
a chaingate verdict transfers between the two fused forms.

**4. NEW stage 3b "cf10": c-line custody raced by in-place flip.** Planner
kept the eligibility mask (fusemap==0 shipped and 32*L^3 > 16 MB) precisely
so a measured host verdict can bank @cf1 where evicting the read-once c
stream pays. The knob is a runtime field consulted once per plane per step
(flip-in-place, zero extra builds, same shape as fm/p4) and a cache op, not
a data op — cf0/cf1 chain outputs cmp-identical (wallaby, L=100), so the
gate verdict holds for either value. Primary = planner default (0, tie
doctrine). On the graded suite it fires only at L=100 B=1; cost ~2 step
timings.

### Measured on the node (a81n2, leased cores via tryout.sh, graded chain, min us/xform; quiet-window numbers with same-window MKL, loaded-window cells flagged)

| case | r9 board | r10 session | MKL same window | vs MKL | eng10 pick (margin) |
|---|---|---|---|---|---|
| L=12  B=64 | 1.913 | **1.918** | 7.869 | 4.10x | batchlane (tie w/ pfa_small) |
| L=12  B=1  | —     | 5.625 (sd 20%) | 12.557 | 2.2x | pfa_small (+9.4%) |
| L=15  B=32 | 4.326 | 4.932 | 16.863 (board 16.49) | 3.42x | batchlane (tie) |
| L=20  B=32 | 12.549 | 13.459 | 59.540 | 4.42x | batchlane (tie, +0.2%) |
| L=25  B=16 | 31.486 | 33.847 | 124.282 | 3.67x | powp (+45.4%) |
| L=27  B=16 | 43.713 | 46.790 | 146.839 | 3.14x | powp (+54.6%) |
| L=31  B=16 | 84.694 | 262.4 LOADED | 2666.9 (board 848.8!) | 10.2x | rader (+6.8%) |
| L=32  B=8  | 53.999 | 210.0 LOADED | 752.1 (board 172.1) | 3.6x | pow2 (+96%) |
| L=40  B=8  | 159.72 | **171.1** | 435.3 | 2.54x | pfa_large (+36.9%) |
| L=50  B=4  | 419.65 | **445.8** | 1002.2 (board 951.9) | 2.25x | powp (+2.4%) |
| L=100 B=1  | 4516.3 | **5043.8** | 7981.1 (board 7816.1) | 1.58x | powp (honest tie, +0.7%) |

(L=10's only readings landed in the worst load windows — 3.127 with MKL at
13.8 — not quotable; the banked pick is batchlane in a tie with pfa_small,
identical to the r9 board's 1.121/1.142 order.) The quiet-window cells sit
2-9% above their boards with MKL moved the same amount: window weather, and
every routing verdict reproduces the r8/r9 winner exactly. The two cells the
r8 board lost to compile timing (25, 40) banked powp/pfa_large
deterministically again — the r9 fix holding on its second node outing.

**New-stage receipts:**

* **fm10 at L=100 (3-way)**: quiet window fm0 kept as a tie (-1.6%: the
  fused exits reached parity after planner's yo fix; the loaded window read
  the same verdict at a junk +625% margin, recorded so nobody quotes it).
  On WALLABY (SPR) the same race picks fm1 through a confirmed noise-gate
  upset over fm0 — a live cross-arch flip, banked per host, which is the
  whole argument for covering all three forms.
* **cf10 first outing**: cf0 (custody OFF) wins +6.3% quiet / +5.5% loaded
  on ICL — independently reproducing planner's custody refutation through
  my own race machinery — and +27.3% on wallaby SPR. The @cf1 arm now
  exists for the CLX advisory (1 MB L2, the standing predicted flip).
* **Determinism spot-proof** (3 cold creates, NO_WISDOM, fresh process,
  held core, L=100): self stages 3/3 identical INCLUDING the two new races
  (c5(gt(d4,d5))@t32@fm0, no @cf tag = default). The eng verdict flipped
  powp/self/powp across the three cold races — the documented honest ~0.7%
  tie at 100; in real runs wisdom pins the first verdict and the driver's
  two-process cmp stays structural (re-proven at L=12 B=64 this session:
  chain outputs bit-identical across processes).

Gates, ship binary, run on the node (tryout's check.py leg still dies on the
unexpanded '$W/c.bin', unchanged since r1 — map-checks run manually over
ssh): single-call rel L2 2.9e-16..4.9e-16 at 10/12(B64,B1)/15/20/25/27/31/32/
40/50/100 (tol 1e-12); map-chain at graded m PASS at 12/B64 (4.87e-14 vs
anchor 3.89e-14), 12/B1 (2.46e-14 vs 5.80e-14), 100 (4.06e-14 vs 2.42e-14),
tol 1e-10; two-step m=2 gate PASS at 12 (9.20e-16) and 100 (2.90e-15) vs tol
3e-14. Surprise-size cold probe L=44 B=8: plans in 1.94 s, PASS 3.5e-16,
1.47x MKL raw execute. Both build modes compile -Wall -Wextra clean (my TU;
planner's documented warning set unchanged).

**Round end: all gen_race/* wisdom stripped from results/wisdom_a81n2.json
via gr_wisdom_drop_prefix** (54 entries; file valid JSON after, gen_pfa_large
and gen_powp foreign keys intact — the layout-agnostic r9 parser doing its
job on the shared file). The monitor cold-races fresh verdicts in its quiet
window; absent entries are deliberate.

### Operation count

Library: unchanged, zero instructions in any hot path. Demo: unchanged from
r8/r9 — one indirect call per execute/chain when a foreign engine ships; the
winner's op count is the class entry's own. The two new race arms move WHERE
existing arithmetic runs (fm2: transpose-exit visit order; cf: a clflushopt
sweep that is cache management, not data) — zero arithmetic change, verified
bit-identical.

### What did NOT work / honest boundaries (with the numbers)

* **Early-session node measurements are junk in absolute terms**: 12-27
  concurrent gcc jobs (the whole panel's creates compiling five churned
  class sources) had MKL reading 2666.9 at L=31 vs its 848.8 board. Rankings
  survived (every pick matched its quiet-window re-race), which is the
  interleaved racer + tie doctrine working as designed, but I re-measured
  everything quotable in later windows. If you dev while the panel is
  compiling, treat absolute numbers as garbage and margins as direction-only.
* **Source churn outraces the .so cache within a session**: gen_powp's hash
  changed between a create launching its compile (04:22) and my next hash
  check (04:26) — a create that races 30 s after a churn misses the
  heavyweight arm and banks a partial verdict (my first L=100 banked
  self-only exactly this way). Prefetch-all + the persistent cache converge
  after the churn stops; I stripped and re-raced. The monitor-side advice
  from r8/r9 stands verbatim: one throwaway create() on the scoring host
  before the suite removes the exposure.
* **The loaded-window fm margin (+625%) is not a number**: same verdict as
  the quiet window (fm0), absurd margin. Wisdom would have banked the
  verdict, not the margin — but it is a reminder that margins from busy
  windows must never be quoted as engine properties.
* **cf10 fires only at L=100 on the graded suite** (the eligibility mask
  needs fusemap==0 shipped AND 32*L^3 > 16 MB): on ICL and SPR it is pure
  confirmation of planner's default. Like the r7 p47 stage, its value is
  the off-ICX insurance; its cost is two step timings at one cell.

### Borrowed, plainly

* **gen_planner r10 (the substrate, as every round)**: the yo-exit fm2 form
  (their comment explicitly leaves it "reachable as a cross-arch race arm" —
  this layer is where that arm now lives), the custody eligibility mask that
  makes cf10 a safe flip-in-place race, and their node refutation numbers
  (custody +6..11% at 100) that set cf's primary. Same honest split as all
  nine prior rounds: their kernels, my measured pick + persistence.
* **gen_pfa_large r10**: their finding that the scored r9 L=50 read was a
  sustained-slow scoring-slot artifact (not a pick error) — corroborates
  treating my loaded-window absolutes as weather, and their discussion of
  the 6% upset floor's blind spot (a real sub-6% gap cannot displace a
  rank-0 prior) is noted as the floor's designed trade, not a bug: the
  confirmation phase exists for exactly the true 2-6% upsets, and it passed
  fm1-at-100 on SPR this round.
* **gen_batchlane / gen_pfa_small r10** (context): their engines converged
  again (the factor-swap adoption); the eng10 ties at 10/12/15/20 between
  their arms are genuinely honest ties between near-identical codelets, and
  the tie doctrine's lowest-index stability is what keeps the pick from
  flapping between them.

### What I would do next (gen_r11 / campaign close)

1. **Capture the CLX advisory's cf10/fm10 verdicts when it runs** — cf1 on
   CLX's 1 MB L2 and the fm boundary at 25-32 are the two predicted flips
   this round armed; SPR already delivered one (fm1-at-100).
2. **Prewarm protocol** (carried three rounds, still the only remaining
   first-contact exposure): one throwaway create on the scoring host; or
   monitor-side `gcc -shared` loop over impl/*.c at round close.
3. **gr_pick_entry library promotion** (carried from r8): the grx_* harness
   as one adoptable call, freeze-break only if an adopter asks.
4. **If planner ships more engine knobs as runtime fields** (their custody
   playoff pattern suggests more coming), the flip-in-place race shape now
   has four instances (tile via env, fm, p4, cf) — the next one is a
   20-line copy.

## Round gen_r11 (reconstruction note — the r11 session never appended its section)

The r11 session shipped code (documented in the gen_race.c header, verified
against impl_11/gen_race.c) but this file was never updated — the round-end
strip also did not run (98 stale gen_race keys, r10 salts included, sat in
results/wisdom_a80n0.json until this round's strip). For the record, r11
shipped: (1) the eng-stage class filter routing gen_batchlane at 50 and 100
(their within-volume SoA engine; the r11 board shows it working — gen_race
4071.3 at L=100 riding batchlane's 4072.3, the cell WON at 1.92x MKL);
(2) NEW stage 6 "alt11": gen_planner's r11 alternating-layout one-transpose
chain adopted into the demo chain driver with its own dual gate and an
alt-vs-classic race, alt arenas 2MB-aligned + MADV_HUGEPAGE (gen_layout r11);
(3) salt bump *10 -> *11. The r11 board has this entry first or tied-first
at 11 of 11 cells (worst gap 1.01x at 20/25). Lesson recorded for future
rounds: the record append and the round-end strip are part of the round, not
optional epilogue — this round did both.

## Round gen_r12 (all hands on L=100, round 2)

### What changed

**1. Library: FROZEN, zero changes** (eighth round running). Everything below
is demo-entry work; all adopters recompile against an identical API.

**2. NEW stage 6b "wv12": gen_planner's r12 WITHIN-VOLUME split-lane chain,
adopted and raced.** Planner generalized gen_batchlane's r11 L=100 winner
into their library as `pln_wv` (lanes = 8 x-planes of ONE volume; z/y sweeps
slab-fused through spill-free staged leaf passes, x through an L1 scratch
with the graded map fused into the transpose-out stores; NO per-step plane
transpose and NO transposed c copy — the two costs the alt chain still
carries). My stage: build lazily (a wisdom hit on the incumbent builds
nothing), gate with an OWN dual gate (m=1 and m=2 exits vs execute + exact
scalar map — wv is bit-identical to no other form, so no verdict transfers;
cached per pv identity), then race vs the INCUMBENT chain in its real
shipped shape (alt pair or classic steps; the incumbent's candidate NAME
carries which — "inc_alt"/"inc_cls" — so gr_sig re-keys if the incumbent
form changes). Incumbent = candidate 0: wv must clear the r9 noise gate
(max(jitter, 6%)), which also conservatively prices the once-per-chain
pack/unpack a step race cannot see (planner budgets it ~2% at graded m). A
wv win REPLACES the alt chain (planner's rule — same regime), and free(A)
moved past the stage because pln_wv_build reads the enumerated trees.

**3. NEW in the eng stage: VARIANT ARMS (per-arm extra -D flags), first
instance gen_batchlane's BL_FUSE100 at L=100.** Their r12 record's explicit
ask ("gen_race: the knob is a legitimate per-host race axis"): their shipped
form (BL_FUSE100=1, the one-sweep fused chain step adopted from gen_pow2
r11) beat their r11 within-volume form by only ~1% in their clean sessions,
with the margin predicted to grow on smaller-LLC hosts. The eng arm list now
appends "gen_batchlane_f0.<srchash>" at L=100, compiled with -DBL_FUSE100=0
into its own cached .so (grx_eng grew a `defs` field; the compile phase
launches variant .so's the prefetch table does not cover); NE_MAX 4 -> 5 so
the variant fits beside the four class arms at 100. The arm list stays a
function of (L, sources) only — repeatability-safe by construction.

**4. Salt bump *11 -> *12** (chain/tile/chaingate/fm/cf/p4/alt/altgate/eng/
enggate + new wv12/wvgate12): the r3 rule, eighth time — planner's engine
generation moved again (pln_wv landed in the library section my LIB include
consumes), and batchlane/pfa_small/pow2/dense_prime/pfa_large/layout sources
all churned within the round (eng arm hashes re-key themselves; the SELF
arm's name cannot). Also fixed a pre-existing scalar-build -Wall wart
(demo_alloc_huge unused: now PLN_SIMD-guarded like both its users).

### Measured on the node (a80n0, tryout leased core 4, graded chain, min us/xform; quiet-window numbers with same-window MKL)

| case | r11 board | r12 session | MKL same window | vs MKL | pick (receipt) |
|---|---|---|---|---|---|
| L=12  B=64 m=600 | 1.915 | 2.019 (warm, calm window) | 7.734 (board 7.740) | 3.83x | gt(d3,d4)@s1@fm0 -> eng: batchlane (tie w/ pfa_small) |
| L=50  B=4  m=128 | 410.975 | **414.150** (sd 0.08%) | 945.1 (board 946.3) | 2.28x | c5(gt(d2,d5)) -> eng: pfa_large (+14.8%) |
| L=100 B=1  m=64  | 4071.3 | **4137.5** (sd 0.20%) | 7764.2 (board 7812) | 1.88x | c5(gt(d4,d5))@t32@fm0@alt1 -> eng: **gen_batchlane_f0 (+2.7%)** |

Stage receipts at L=100 (quiet window, all banked):
* **fm0 ships (tie, +0.5%)** — the separate-map regime holds on ICL, so the
  alt/wv stages fire; **alt1 beats classic by +13.3%**; cf0 +6.5% (planner's
  custody refutation reproduced through my race a third round).
* **wv LOSES to the alt incumbent at L=100 on ICL: +3.5% quiet, +4.9% in the
  first session — non-tie both times, banked.** Planner's r12 engine does
  NOT take the scored cell on this host; the verdict is the round's most
  useful negative (see boundaries).
* **The variant arm won its first outing**: eng12 picked gen_batchlane_f0
  (their r11 within-volume form) at **+2.7% over their fused ship form** in
  the quiet window. Their own record shows the two forms trading places by
  window (fused wins their clean-session floors by ~1%, ties min-of-mins);
  on this host the race banks whichever measures faster and wisdom pins it —
  which is precisely the per-host insurance their record asked for.

**Unseen-size receipt — the wv arm earns its place at L=96 B=2** (cold
create 1.9 s, single 4.3e-16 PASS): chain12 picks c8(gt(d3,d4))@t32@fm0,
alt1 beats classic +6.3%, then **wv1 beats the alt incumbent by +6.0%
(clears the noise-gate floor), gate PASS, banked**. So the wv boundary is
genuinely per-L: loses at 100, wins at 96 — exactly the class of verdict
this layer exists to measure instead of trust. (At 96 eng keeps SELF: no
class arm took it, as in r8.)

### Counter protocol (mandatory this round)

Baseline: gen_batchlane's r12 record measured their r11-ship engine at 100
(whole process, --samples 2): cycles 8.03G, p0 3.17G / p1 0.04G / p5 4.22G /
p2_3 2.49G / p4_9 1.49G, l1d.replacement 967M. After my routing: the SHIPPED
gen_race binary at 100 under tools/pmu.sh, same shape: insn 10.54G, **p0
3.20G / p1 0.037G / p5 4.24G / p2_3 2.46G / p4_9 1.47G, l1d.replacement
965M** — the port and traffic signature of batchlane's within-volume engine
to within 1-2% (the routing receipt, counter-verified; cycles read 10.24G in
my mixed window vs their 8.03G quiet — neighbor-inflated stalls, not extra
work; the work-proportional counters match). My r12 added zero hot-path
instructions of its own, as every round.

### Gates (ship binary, node, manual legs where tryout's '$W/c.bin' bug still skips them — unchanged since r1)

single-call rel L2: 2.9e-16 (12/B64), 4.3e-16 (50), 4.3e-16 (96/B2),
5.2e-16 (100) vs tol 1e-12. Map-chain at graded m: 12/B64 m=600 4.87e-14
(anchor 3.89e-14), 100 m=64 4.42e-14 (anchor 2.42e-14) — tol 1e-10. Two-step
m=2 gate at 100: 3.09e-15 vs tol 3e-14 (through the shipped f0-routed
chain). Repeatability: chain outputs bit-identical across independent node
processes at 12/B64 AND 100/B1 (wisdom pins process 2 to process 1's winner
— with a variant arm in the race this is more load-bearing than ever).
Warm create at 100: 0.015 s vs the 50 ms budget (wisdom reads + batchlane_f0
dlopen + their create + cached gates). Both build modes compile
-Wall -Wextra clean, scalar build included.

### Determinism spot-proof (3 cold creates, NO_WISDOM, fresh process, core 4 — run under heavy panel load deliberately)

Self stages 3/3 identical (chain/tile/fm/cf/alt picks); cycle 2's wv race
saw a spurious 46% "upset" and the r9 noise gate REVERTED it unstored —
the machinery working exactly as designed under load. The eng verdict
flipped batchlane/batchlane/batchlane_f0 across the three cycles: the two
batchlane FORMS are an honest ~1-3% tie class on this host, the documented
r10 situation at 100 (powp/self/powp) — in real runs wisdom pins the first
quiet verdict (banked: f0) and the two-process cmp stays structural
(re-proven above). Margins from the loaded window (e.g. alt1 "+167%") are
junk and not quoted — r10's rule.

### Operation count

Library: unchanged, zero instructions in any hot path. Demo: one indirect
call per execute/chain when a foreign engine ships (at 100 the winner's op
count is gen_batchlane's r11 within-volume engine — see their r11 record);
the wv stage, where it ships (96-class sizes), runs planner's pln_wv step
(their op accounting; z/y slab-fused single L1 round trip per point per
axis, x via 24-shuffle tr8 brackets + L1 scratch pencils, map fused into
the transpose-out stores) with pack/unpack once per volume per chain.

### What did NOT work / honest boundaries (with the numbers)

* **planner's wv chain does not take L=100 on ICL**: +3.5%/+4.9% behind the
  alt incumbent (non-tie, reproduced across two quiet sessions, banked).
  The engine's design win (no plane transposes, no c copy) is real — it
  wins at 96 by 6.0% — but at 100 the alt chain's one-transpose form plus
  THP arenas is simply faster on this host. If planner's entry adopts wv at
  100 by its own 4% playoff, the eng stage will measure THEIR whole engine
  against batchlane's anyway; nothing to fix on my side.
* **The first cold create of the round banked a partial eng verdict again**
  (setup 32 s, all seven fresh-hash .so's compiling; raced self-only,
  5538 us) — the r8/r10 mechanism, converged the same way: prefetch-all
  landed everything within ~4 min, I stripped and re-raced (4137). The
  monitor-side advice stands verbatim: one throwaway create on the scoring
  host before the suite (the .so cache is prewarmed with today's hashes;
  any post-session churn re-exposes only the churned arm).
* **gen_powp's arm produced no enggate verdict at 100** (their fresh .so
  compiled fine; setup returned NULL — create refused or dlopen/sup issue,
  not investigated since powp was 4465 vs batchlane's 4072 on the r11
  board). If their r12 session lands a large-L win, the source-hash re-key
  races them fresh automatically.
* **The batchlane ship-vs-f0 verdict is window-sensitive** (+2.7% f0 quiet
  here; their record: fused wins their clean floors by ~1%). Both forms
  gate ok and differ by low single digits; the pick is per-host insurance,
  not a claim that either form is universally faster. CLX is where the f0
  arm should genuinely matter (batchlane's own GP64_FUSE analogy).
* The earlier strip-tool footgun, re-documented: gr_wisdom_path resolves
  results/wisdom_<THIS host>.json — a wallaby-side "strip the node's file"
  run must set GEN_RACE_WISDOM explicitly (my first strip invocation
  harmlessly rewrote wallaby's file; foreign keys preserved by the r9
  layout-agnostic parser, verified).

### Borrowed, plainly

* **gen_planner r12 (the round's engine)**: pln_wv build/pack/step/unpack —
  their generalization of gen_batchlane's r11 within-volume design — plus
  their eligibility mask (chain_ok, fusemap 0, custody off, L > 80) and the
  wv-replaces-alt rule. Same honest split as eleven prior rounds: their
  kernels, my measured pick + persistence.
* **gen_batchlane r11/r12**: their engine IS the shipped L=100 cell (both
  forms), and the BL_FUSE100 variant arm is their record's explicit ask.
  Their "LLC-discriminator before traffic cuts" finding is why I did not
  chase my own l1d dashboard this round.
* **gen_pow2 r11** (via batchlane): the fused-step lineage the f0/f1 race
  arbitrates. **gen_layout r11**: huge arenas under the wv slabs (via my
  r11 demo_alloc_huge). **gen_pfa_large r9/r10**: the noise-gate design that
  correctly killed a spurious wv upset under load this round.

### Adoption status (the score)

* **gen_planner** (full GEN_RACE_LIB_ONLY include), **gen_pfa_large**
  (string wisdom, salted keys), **gen_powp** (keyf/sig/lookup/store): all
  unchanged on the frozen API — zero churn charged, eighth round.
* This round the adoption also runs the OTHER way twice over: the demo now
  ships planner's pln_wv as a raced arm, and batchlane's compile knob as a
  raced variant — both by their records' invitation.
* Round end: **all 40 gen_race/* keys stripped from results/wisdom_a80n0.json
  via gr_wisdom_drop_prefix** (including the 58+40 stale r10/r11-salt
  leftovers found at round start — see the r11 reconstruction note); foreign
  gen_powp keys intact, file valid JSON, wallaby's file verified intact.
  The monitor cold-races fresh verdicts; absent entries are deliberate.

### What I would do next (gen_r13 / campaign close)

1. **Capture the CLX/SPR verdicts for the two new race axes** when the
   advisories run: wv12 at 96/100 (SPR wallaby already read wv-vs-alt as a
   1.9% tie at 100 — a third behavior) and the batchlane f0/f1 arm (their
   record predicts the r11 form wins big on small-LLC hosts).
2. **Prewarm protocol** (carried five rounds): one throwaway create on the
   scoring host; today's .so cache is warm unless sources churn again.
3. **Probe wv at the other >80 sizes** (108, 121, 125...) if a round-6-style
   surprise test returns: the 96-vs-100 flip suggests the wv/alt boundary
   needs a per-L verdict, which is exactly what the stage now banks.
4. **If planner ships more wv knobs as runtime fields** (their PB pencil
   width is env-only today), the flip-in-place race shape has five
   instances to copy from (tile/fm/p4/cf/alt).

## Round gen_r13 (reconstruction note — the r13 session shipped code but never appended its section)

Same failure shape as r11, recorded again so the pattern stays visible: the
r13 session updated impl (verified against impl_13/gen_race.c and its header
changelog) and the r13 board carries the results, but this file was not
updated and the round-end strip did not run (279 stale r12/r13-salted
gen_race keys sat in results/wisdom_a80n0.json until this round's strip).
What r13 shipped: (1) the eng race's chain length cap 64 -> 4096 (es.m stays
a pure function of (L, batch)) so the two NEW scored B=1 cells (10:1:16384,
12:1:12288), which grade at m in the thousands, race at honest amortization
-- the old cap charged once-per-chain costs (e.g. gen_batchlane's r13 scalar
c-pack) ~200-256x more heavily than the score does; (2) salt bump *12 ->
*13.  The r13 board receipts: 10:1 banked 1.943 us/xform and 12:1 2.999
us/xform, both riding gen_pfa_small's new B=1 chain engine through the eng
routing, both cells tied with the class winner at 1.00x -- the routing
worked.  The lesson from r11 stands doubled: THE RECORD APPEND AND THE
ROUND-END STRIP ARE PART OF THE ROUND.

## Round gen_r14 (the execute()/chain() plumbing-seam round)

### The round's shape, up front

benchFFT -- the round's B=1 acceptance metric -- calls fft3d_create(L, 1)
and then times single-shot fft3d_execute().  And benchfft_ours/doit links
gen_race.o: the "ours" curve the brief quotes IS this entry.  Yet NO race
stage ever measured the execute workload: the eng verdict (a whole-chain
race) shipped its winner's execute() blindly, and the self path's execute
stayed the r6 per-volume pln_p3d_exec sandwich.  My own records called the
inversion risk twice (r5: "MKL beats my execute at L=14 while the graded
chain is the win"; r8 next-list #3: "race the execute() workload separately
... the two can invert") and left it unspent because no case weighted raw
execute.  benchFFT weights nothing else.

### What changed (all demo-entry; library FROZEN, ninth round running)

**1. NEW stage "exe14": the EXECUTE-workload race.**  Runs inside the eng
block on the SAME arm list (candidate names stay a pure function of
(L, sources): both repeatability processes compute both keys identically).
Self is candidate 0 -- the pv pln_p3d_exec loop, which this round carries
gen_planner's r14 transposed-store fused exec codelets + transpose-free
plane pass (p->ft) -- so a foreign execute must beat the trunk's own by
more than noise to forward (tie doctrine).  Foreign arms run their real
whole-batch fft3d_execute: equal work per thunk, each in its true
single-shot pattern.  Winner ships on a SEPARATE vtable (p->fxe, aliasing
p->fx when one arm wins both races); the chain verdict keeps p->fx.  The
plan struct invariant that makes the seam disappear: when the exe verdict
is self under a foreign chain winner, p3 STAYS ALIVE for exactly the
execute path (r13 freed it with the rest of the self engines).
GEN_RACE_NO_EXE=1 skips the stage (A/B pin).

**2. NEW absolute gate "exegate14".**  The arm's execute vs the self pv
reference on the deterministic race input, rel L2 < 1e-12, cached per
(L, exact B, source-hash) -- and consulted on EVERY ship path including
warm aliases.  enggate alone is a self-consistency check (two fused chain
steps vs the arm's own execute + exact map): an engine whose chain and
execute share one wrong scale would pass it.  The exe race's whole point
is shipping the execute alone, so it gets an absolute reference.

**3. The warm-wisdom probe now requires BOTH eng14 and exe14 keys** to
answer before the compile/poll phase is skipped -- a cold exe race must
see its arms even when the chain verdict is already banked.

**4. Salt bump *13 -> *14** (ninth time; the r3 rule): gen_planner's
generation moved again -- the ft transpose-free plane pass + transposed-
store fused codelets change pln_p3d_exec under every banked self verdict --
and batchlane/pfa_small/pow2/dense_prime/bluestein/layout sources all
churned within the round (arm hashes re-key themselves; the SELF arm's
name cannot).

### Measured on the node (a80n0, slot lease core 3 -- WITH A LOUD CAVEAT)

The node spent this entire session as a construction site: the whole
panel's r14 sessions were live (load 12+, up to 15 concurrent race_eng gcc
jobs, sshd refusing new connections at one point).  Per the r10 rule,
absolute numbers from these windows are weather, not engine properties;
same-window library references are attached to everything quotable.
Correctness and plumbing receipts are load-immune; timing margins below
are direction-only.

**Plumbing receipts (the round's actual deliverable):**

* **The inversion the stage exists for showed up at first contact**: at
  L=10 B=1 with arms present, eng14 -> gen_batchlane (chain), exe14 ->
  SELF at +22.5% (their execute lost to the r14 self exec).  r13 shipped
  batchlane's execute here blindly.  At L=12 B=1: exe14 -> self at +30.9%.
* **Self execute at 10/12 B=1 is transformed** (gen_planner's r14 exec
  codelets, routed by the race): race-thunk reads 1.32-1.38 us (L=10) and
  2.02 us (L=12) in the quieter moments; the r13-era self sandwich was
  ~4.25 us (gen_pfa_small's measurement) and the r13 benchFFT "ours"
  numbers were 3.71/5.43 us (13,440/17,112 mflops).  In my least-bad
  window: driver raw execute 4.06 us at 10 vs fftw3_measure 6.28 us and
  MKL 3.66 us SAME CORE SAME MINUTE (quiet-window equivalents ~2.09/1.47
  -- everything ~3x inflated, ratios: ours/fftw3 = 0.65, i.e. a ~1.5x win
  at the benchFFT convention).  benchFFT doit (built in my scratch against
  impl_14): 4.44 us at 10, 6.62 us at 12 in the same storm.
* **Gates all PASS, shipped binary, on the node**: single-call rel L2
  3.111e-16 (10:1) / 2.974e-16 (12:1) through the exe path (tol 1e-12);
  map-chain at graded m PASS at 10:1 m=16384 (1.895e-06 vs anchor
  9.771e-07, tol 3e-04) and 12:1 m=12288 (3.232e-09 vs anchor 7.995e-09);
  chain AND single outputs bit-identical across independent node runs
  (wisdom pins run 2 to run 1's winners -- now for BOTH vtables).
* **The r9 noise gate earned its keep under the storm**: a 77.6% chain
  "upset" at 12:1 was reverted as unconfirmed and NOT stored -- exactly
  the designed behavior on junk windows.
* Warm create with both verdicts banked: 13-45 ms observed in the storm
  (dev wisdom on the shared FS; the 50 ms budget holds even in this
  weather).  Cold creates in the storm: 0.2-64 s, worst at L=100 with all
  heavyweight compiles relaunching on churned hashes -- inside the 60 s
  budget everywhere except nothing (64.18 s setup at L=100 was measured
  by benchFFT's own setup clock INCLUDING its input prep; the create-only
  poll is bounded at 30 s + race time).

**The full-L benchFFT sweep (10,12,15,16,20,25,27,31,32,40,50,64,100,128)
ran but the windows were unusable** (fftw3 reading 2.4-3.4x its board
values, sd 45-183%, medians 5-10x the mins).  Where pairs were least-bad
(10, 12, 20, 25, 31: ours-vs-fftw3 same window) ours led at every size;
at 32/64/100 the storm made even direction unreadable.  The monitor's
quiet-window benchFFT run is the number that counts; the routing it will
exercise is what the receipts above prove.  The sweep's exe14 verdicts
show the seam being arbitrated per-L: chain->batchlane + exe->SELF at
10/15/20 (the inversion, three sizes), both->pow2 at 16, both->
dense_prime at 31, both->pfa_large at 40/50, both->batchlane at 100
(exe an honest tie), self elsewhere (some of those are storm-partial;
the monitor's cold race redoes them).

**SPR (wallaby) validation, quiet core 100 -- every real code path, per
the r9 doctrine (mid-round the a80n0 reservation ended: ssh began refusing
with `pam_slurm_adopt: you have no active jobs on this node` even though
the heartbeat shim still read fresh, so the ICL leg stopped where the
storm receipts above leave it; the monitor re-measures on the exclusive
node regardless):**

* **Batched non-regression at 12:64 (graded shape), cold + warm**: chain
  races to gen_batchlane (tie with pfa_small, the r10-documented honest
  tie class), exe14 -> batchlane at +5.8% (their batched execute beats
  self -- so fxe ALIASES fx, the one-plan path), chain 1.824 us/xform
  (sd 3.25%), single PASS 2.898e-16 through the exe path, map-chain m=600
  PASS 4.869e-14 vs anchor 3.887e-14, chain outputs bit-identical across
  independent runs, warm setup 0.20-0.26 s (driver chain-buffer prep
  included; the create is wisdom reads + dlopen + create).
* **Determinism spot-proof (3 cold creates, NO_WISDOM, fresh process,
  same core, L=10 B=1)**: eng14 -> pfa_small 3/3 (margins 24.0/24.1/24.3%),
  exe14 -> SELF 3/3 (margins 32.3-36.8%) -- and the self execute reads
  **0.966-0.991 us** on this host (fftw3's ICL benchFFT equivalent at 10
  is 2.09 us; ICL will differ, but the order says the must-win cell is
  covered with room).
* All three vtable topologies exercised end-to-end: (fx=batchlane,
  fxe=self/keep-p3) at 10:1 ICL, (fxe aliases fx) at 12:64 SPR,
  (both self) at 12:1 ICL; destroy clean in each.

### Operation count

Library: unchanged, zero instructions in any hot path.  Demo execute():
ONE pointer test + indirect call when a foreign engine ships (as chain has
had since r8); the self path is gen_planner's r14 pln_p3d_exec (their op
accounting -- fused-root trees now run transpose-free planes with
transposed-store exits).  The exe race adds plan-time samples only:
~6 calls x (ne+1) arms x whole-batch executes, wisdom-warm cost two file
reads + (foreign winner only) one dlopen + create.

### What did NOT work / honest boundaries (with the numbers)

* **This session could not produce a quotable timing table** -- the load
  windows (12+ load, panel-wide gcc storms, sshd connection refusals) made
  even min-of-samples unreliable at the larger L (L=31 fftw3 read 2374 us
  vs its ~875 board; L=100 MKL median 26 ms vs 7.8 board).  Everything
  quotable above carries its same-window reference; everything else is
  left to the scoring window.  If the r14 board disagrees with a verdict,
  strip + re-race in quiet before blaming the plumbing.
* **Storm-window verdicts banked into my DEV wisdom are junk and were
  treated as such** (dev file under build/tryout/gen_race/, GEN_RACE_WISDOM
  pin; the real results/wisdom_a80n0.json was never touched by my runs).
  The monitor cold-races fresh -- and the exe14 stage re-races with it.
* **The L=12 first-contact eng verdict banked self-only** (pfa_small's arm
  was mid-churn/compiling): the documented r8/r10 partial-verdict
  mechanism, converged the same way (REFRESH after the .so landed banked
  the real race).  Monitor-side advice stands verbatim: one throwaway
  create before the suite.
* **Deliberately NOT done: racing the tree choice on the execute workload.**
  The chain-winning tree's p3 serves execute; the best EXECUTE tree could
  differ (planner's ft flag is root-form-dependent).  That is a
  planner-layer enumeration question and the class arms cover the sizes
  that matter; noted for a future round if benchFFT exposes a self-serving
  size where the gap is real.
* tryout.sh's M extraction now breaks on the r13 duplicate-L cases.txt
  lines (awk returns two values) in addition to the r1-era '$W/c.bin' bug
  -- all driver runs this session were manual over ssh on the slot lease
  (protocol unchanged since r3; gen_layout's r14 record hit the same).

### Borrowed, plainly

* **gen_planner r14 (the engine of the self verdict)**: the transposed-
  store fused exec codelets + transpose-free plane pass (p->ft) are why
  "self" wins the exe race at 10/12 over the class arms' executes in my
  windows.  Same honest split as thirteen rounds: their kernels, my
  measured pick + persistence.
* **gen_pfa_small r13 next-list #1** ("plain execute() at B=1 is still the
  r6 sandwich... route it through the form-3 passes") and **gen_batchlane
  r13 next-list #1** (fuse pack/unpack into first/last passes): the asks
  that define this round; the exe14 stage is the routing half that makes
  whichever landing wins ship through the trunk that benchFFT compiles.
* **gen_layout r14**: the register-form pack/unpack kit (gl_ldi8x8/
  gl_sti8x8) their round shipped for the engine owners -- not consumed
  directly by this layer, but its microbench itemizes the ~1 us/volume
  sandwich the exe race now prices per host instead of trusting.
* **gen_pfa_large r9/r10** (again): the noise gate that binned this
  session's storm upsets.

### Adoption status (the score)

* **gen_planner** (full GEN_RACE_LIB_ONLY include), **gen_pfa_large**
  (string wisdom, salted keys), **gen_powp** (keyf/sig/lookup/store): all
  unchanged on the frozen API -- zero churn charged, ninth round.
* The demo now ships class engines through TWO raced vtables (chain since
  r8, execute since this round) -- the benchFFT curve is the assembled
  library's execute, chosen per (L, B, host) by measurement.
* Round end: all stale gen_race/* keys (r12+r13 salts, 279 of them --
  see the r13 reconstruction note) stripped from results/wisdom_a80n0.json
  via gr_wisdom_drop_prefix; foreign keys untouched.

### What I would do next (gen_r15)

1. **Quiet-window benchFFT re-proof** of the full L curve (this session's
   storm made it impossible): expect the must-win 10/12 from the exe14
   routing + planner's exec codelets; at 64-128 (B=1 memory-bound) expect
   firming, not leaps -- and if a cell still loses to fftw3, the exe race
   receipts say whether it is a routing miss (arm loses that should win)
   or an engine gap (self wins but slow -- then it is the class owner's).
2. **Race the exe verdict at the graded batched cells' exact B** only if a
   future case weights raw batched execute; today it costs one extra race
   per create and can only matter to the single-call gate path.
3. **Execute-workload tree race** (the "deliberately NOT done" above) if a
   self-served size shows a real chain-tree-vs-exec-tree gap.
4. **Prewarm protocol** (carried six rounds): one throwaway create on the
   scoring host before the suite; this round's source churn re-exposes
   every arm's first contact.
