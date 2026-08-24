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
