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
