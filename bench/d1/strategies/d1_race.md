# d1_race — strategy record

## Round d1_r1 (2026-09-02, post-restart real r1)

### Starting point
The impl_0 stub (dense DFT, `fft1d_supports()` = 0 — supported nothing). No prior 1D
rounds survived the harness loss. But the 3D campaign's gen_race exists at
`bench/gen/impl/gen_race.c` with nine rounds of accumulated racing doctrine, and the
mission statement says "generalize gen_race" — so this round was a port + one genuinely
new mechanism, not a from-scratch design.

### What was built (all in impl/d1_race.c, ~1100 lines)

**1. The library layer** (adoption-scored, `#define D1_RACE_LIB_ONLY` + `#include
"d1_race.c"`, everything static, prefix `gr_`, no symbols leak). Ported VERBATIM from
gen_race's gen_r9-era core, with 1D pins:
  - `gr_time_run` / `gr_race`: interleaved sample-major timing (gen_r4 — candidate-major
    timing lets 10-15% core-state drift decide races), lowest-index tie doctrine,
    gen_r9 noise gate (sub-floor upsets must confirm on fresh evidence or revert;
    reverted upsets never persisted).
  - `gr_pick` behind per-host wisdom at `results/wisdom1d_<host>.json` (absolute path,
    never cwd-relative — a binary run from another campaign's dir must not cross-write;
    file name deliberately distinct from gen's `wisdom_<host>.json`). Keys carry the
    FNV signature of the candidate-name list; flock + write-temp + rename; the gen_r9
    layout-agnostic entry parser (the line-oriented one wiped compacted files once).
  - `gr_pick_value` (integer-knob sweep), `gr_pick_plan` (whole-plan vtable race),
    `gr_wisdom_get_str`/`put_str` (string wisdom), `gr_wisdom_drop_prefix` (round-end
    strip protocol).
  - Env pins renamed D1_RACE_* (NO_RACE / FORCE / REFRESH / NO_WISDOM / WISDOM /
    VERBOSE / SEQ / NO_ENG). `gr_bucket` cap raised 128 → 512 (1D graded batches).
  - Do NOT include this and gen_race.c in one TU — same gr_ prefix by design.

**2. The demo entry** — the gen_r8 "eng" cross-class race brought to 1D on day one,
because unlike gen_r1 the siblings here are already real:
  - supports every L in [2, 2^20]. Own trust-anchor engine in-file: scalar Stockham
    DIF pow2 (natural-order, ping-pong, manual complex mul — no __muldc3 NaN branches)
    + Bluestein for any L (pad M = pow2 ≥ 2L−1, chirp k² reduced mod 2L in int64,
    1/M folded into the kernel spectrum at plan time). Deliberately simple; it is the
    gate reference and existence fallback, not a contender.
  - At plan time: hash impl sources for {d1_pow2, d1_batchlane, d1_composite, d1_prime,
    d1_rader, d1_bluestein, d1_planner, d1_twiddle}, compile each to a per-host .so
    (cached by source hash under `build/race1d/<host>/`, .bad markers stop re-compiling
    failures, tmp+rename is concurrency-safe), dlopen RTLD_LOCAL (+ -Wl,-Bsymbolic so
    two .so's defining the same fft1d_* symbols stay self-contained; the driver binary
    exports nothing, so no interposition either way).
  - **NEW vs gen_race: the FORK-ISOLATED gate.** gen_r8 gated dlopened arms in-process;
    here every arm's first contact (create + execute + 2-step chain, rel L2 vs the
    in-file reference < 2e-12) runs in a forked child with a 15 s SIGKILL watchdog.
    Motivation: this is a live round — a sibling mid-edit can crash, hang, or be a
    dense-supports-everything stub that takes 40 s per execute at L=100003; none of
    that may take my binary down or blow the plan budget. Verdict cached per
    (L, B, source-hash) via string wisdom, so the fork cost is paid once per source
    version per cell.
  - Stage "exe.r1": whole-batch fft1d_execute raced, self = candidate 0 (tie doctrine's
    stability anchor). Winner ships by vtable forwarding.
  - Stage "chn.r1": the graded map chain raced at an HONEST m — the graded m for this
    (L,B) read from cases.txt, work-capped at ~4e6 points/unit (the gen_r13 lesson:
    a short race chain overcharges once-per-chain costs). Candidate 0 = exec-winner +
    the driver's exact map (so chain can never regress below the exec race); other
    candidates = every gated arm exporting a native fft1d_chain. The c0 name carries
    the exec winner's name, so an exec-winner change re-keys the chain verdict.
  - Wisdom pins run 2 of the driver's two-process repeatability cmp to run 1's choice
    (different winners round differently — an unpersisted flip would show NOT
    REPEATABLE; gen learned this at gen_r8).

### Borrowed, explicitly
- gen_race (3D campaign): the entire library core, the eng-stage architecture, the
  wisdom/signature/salt doctrine, env-pin set. This entry IS gen_race generalized.
- gen_r4 (gen_batchlane/gen_pfa_small/gen_pfa_large records): interleaved racer.
- gen_r9 (gen_pfa_large design): noise gate + never-store-reverted-upsets.
- gen_r13: honest-m chain racing.
- The fork-isolated gate + watchdog is new here (gen never had mid-round dlopen of
  actively-edited sources).

### What it routes to (wallaby, SPR 6448Y, single nice'd core — the Ice Lake
reservation was down all session: job 440299 not running, and implementers must not
submit slurm jobs, so no scoring-node numbers this round)

All cells PASS check.py (rel L2 1.4e-16 … 1.4e-15, tol 1e-12); chained cells PASS the
map-chain gate (3e-14 … 9e-14, tol 1e-10); every cell repeatable across two processes.

| cell | ships exec | ships chain | min µs/xform |
|---|---|---|---|
| 13 B=1 | d1_prime | d1_prime | 0.017 |
| 32 B=1 | d1_pow2 | d1_pow2 | 0.024 |
| 128 B=1 | d1_pow2 | d1_batchlane | 0.078 |
| 1024 B=1 | d1_pow2 | d1_pow2 | 1.08 |
| 13 B=512 (m=2000) | d1_prime | d1_prime | 0.016 |
| 128 B=512 | d1_pow2 | — | 0.120 |
| 4096 B=256 | d1_pow2 | d1_pow2 | 11.3 |
| 16384 B=64 | d1_pow2 | d1_pow2 | 62.3 |
| 1021 B=8 / B=256 | d1_bluestein | d1_bluestein | 16.1 / 8.77 |
| 10007 B=8 | d1_bluestein | d1_bluestein | 120 |
| 65537 B=4 | d1_planner (unpadded Rader) | loop | 1205 |
| 65537 B=16 | d1_rader | loop | 1120 |
| 100003 B=2 | d1_bluestein | d1_bluestein | 3318 |

Setup: ~5 s the first-ever create on a host (compiles all eight .so), 0.4–5 s per
cold (L,B) cell (gate + two races), 4–60 ms warm (wisdom hit + one dlopen + one arm
create). All inside the 60 s cold / soft warm budget, and setup_seconds is not timed.

The round's live-fire validation, unplanned: sibling sources CHURNED under me during
the session (two different d1_batchlane hashes appear in my own test log an hour
apart, and d1_rader grew 65537 support mid-session and immediately took that cell
from d1_planner in a fresh race). Hash-carrying candidate names re-keyed the wisdom
automatically each time — zero stale verdicts observed. That is the r2→r3 gen lesson
(gr_sig defeated by unchanged names) discharged by construction.

### What did not work / rough edges, with numbers
- Self engine at L=100003 is ~75 ms/vector (scalar Bluestein at M=262144), so the
  exe-race primary eats several seconds of the race budget at the largest cells; the
  5 s race deadline caps it (min one calibrated sample per candidate). Not fixed this
  round — self only needs to exist, not to win.
- The gate threshold is 2e-12 (diff vs my reference, both sides ≤1e-12 vs exact by
  contract; measured entries sit at 1e-16..1e-15, so the margin is ~3 orders both
  ways). A sibling that is *subtly* wrong at ~5e-12 would pass my gate and fail the
  checker under my name. Accepted risk: such an entry fails its own cells too and
  dies in its own round.
- No Ice Lake numbers (reservation down). The .so cache is per-host, so the scoring
  node cold-compiles on its first create — ~5 s, once.

### Next round
1. Per-arm variant knobs (gen_r12's -D mechanism): race BL_FUSE-style flags where a
   sibling's record asks for it.
2. Race the B=1 non-chained cell on a SINGLE-SHOT execute thunk (benchFFT semantics)
   separately from the whole-batch thunk — gen needed exe14 for exactly this seam.
3. Round-end `gr_wisdom_drop_prefix("d1_race/")` protocol once the campaign adopts it.
4. If adopters appear (d1_planner is the natural first), split the lib into a doc'd
   header section; keep the salt-bump discipline (exe.r1 → exe.r2 on any engine change).
5. Lane the self Bluestein (8-wide split-complex) only if the fallback ever actually
   ships anywhere — so far it never does.

## Round d1_r2 (2026-09-02)

### Context
r1 shipped the ported gen_race core + fork-gated cross-class demo entry; the a80n0
leaderboard showed the router tracking its best gated sibling within noise almost
everywhere. So this round is three targeted mechanism improvements, not a redesign.
Siblings churned again under me all session (d1_prime/d1_pow2/d1_rader/d1_bluestein
all faster than r1); hash-carrying candidate names re-keyed everything automatically,
zero stale verdicts observed — second live-fire round for that design.

### What changed (impl/d1_race.c)

**1. Flattened dispatch — what ships is exactly what was raced.** r1's
`fft1d_execute` went driver → forward fn → branch on `cx->xarm` → two dependent
loads into the arm struct → indirect call; the race, meanwhile, timed the arm's
BARE execute. Now `fft1d_create` caches `{exec_fn, exec_arg}` and
`{chain_fn, chain_arg}` in the plan's FIRST cache line and `fft1d_execute`/
`fft1d_chain` are each a single unconditional indirect call. Self and the loop-chain
fallback sit behind adapters with the same signature, so there is no branch anywhere
on the scored path.

**2. Variant lanes — gen_r12's -D knob race, adopted (r1 next-round item 1).**
The arm roster is now a table of `{base, extra_cflags, suffix, exe_only}`; a variant
compiles the same sibling source with extra flags into its own per-host .so, mixes
the flag string into the FNV hash (distinct wisdom identity, re-keys on source OR
flag change), and races as a separate candidate. First lane:
`d1_composite -DUSE_ZMM4` ("+zmm4"), which d1_composite's r1 record explicitly kept
for a cross-host A/B (zmm 4-transform batched kernel, statistical tie with ymm2 on
wallaby SPR — the open question is the scoring node). `exe_only=1` keeps it out of
the chain race since the knob only touches the batched execute. Verified end to end
on wallaby: compiles, fork-gates e1c1 at L=60, races; base ymm2 won here by 18%
(as their record predicted for SPR) — the a80n0 race decides for real. Adding a
future lane is one table row.

**3. exe.r1 → exe.r2: self sits out the race at big cells.** r1 rough-edge, now
fixed: scalar self at L=100003 is ~75 ms/vector; as candidate 0 it ate ~3 s of the
5 s race budget and squeezed the actual contenders' samples exactly where
rader/bluestein margins are thin. Now self joins the race only when its estimated
cost (pow2: L, else 12L for the 3-pass padded Bluestein, times B) is ≤ 2^20 points;
it remains the unconditional ship-time fallback (no gated arm ⇒ self ships, as
before). Effect at L=65537 B=16: plan 4.5 s → 1.9 s and the shipped min improved
856 → 779 µs on the same core (cleaner race). Salt bumped exe.r1 → exe.r2 per the
discipline; chn.r1 kept (its stage is mechanically unchanged, and its c0 name
re-keys via the exec winner).

### Measured (wallaby SPR 6448Y, single nice'd core 100; Ice Lake reservation
job 440371 down again, so no scoring-node numbers — a80n0 will re-race fresh since
every sibling hash changed). All cells PASS check.py (rel L2 1.4e-16 … 1.4e-15,
tol 1e-12); chained cells PASS the map-chain gate; two-process repeatability
identical at 32/512-chain.

| cell | ships | min µs/xform (r1 record) |
|---|---|---|
| 13 B=1 | d1_prime | 0.010–0.016 bimodal core state (0.017) |
| 13 B=1 m=200k chain | d1_prime | 0.031 |
| 13 B=512 | d1_prime | 0.009 |
| 31 B=1 / B=512 | d1_prime | 0.032 / 0.035 |
| 32 B=512 m=1000 chain | d1_pow2 | 0.026 |
| 60 B=1 / B=512 | d1_composite | 0.033 / 0.031–0.038 |
| 128 B=1 / B=512 | d1_pow2 | 0.070 / 0.098 |
| 1024 B=1 | d1_pow2 | 1.048 (1.08) |
| 4096 B=256 | d1_pow2 | 9.24 (11.3) |
| 16384 B=1 / B=64 | d1_pow2 | 27.6 / 38.4 (62.3) |
| 1021 B=1 / B=256 | d1_rader (took it from bluestein) | 7.43 / 8.48 (16.1/8.77) |
| 10007 B=8 | d1_bluestein | 115 (120) |
| 65537 B=1 / B=16 | d1_rader | 800 / 779 (1205/1120) |
| 100003 B=8 | d1_bluestein | 3042, chain m=15 3187 |

Most of the raw improvement is the siblings getting faster — that is the router
working as designed, plus items 1–3 keeping its own tax at zero.

### What did not work, with the number that killed it
- **Chasing the last print-quantum at L=13 B=1.** Interleaved 5-round A/B of old
  forwarding vs flattened dispatch vs BARE statically-linked d1_prime:
  0.015 / 0.016 / 0.016 µs. The flattened path exactly matches the bare arm (the
  correct target — dispatch tax zero), but the OLD forwarding "beat" the bare arm
  by one quantum, which is impossible for a forwarding layer: binary code-layout
  luck, not dispatch. Moving the dispatch fields front-of-struct did not change it
  (placement after the ~1 KB ctx vs before: same reading). Kept the flattening for
  race-path/ship-path equivalence; do not chase 1 ns layout noise on a login node.
- **First measurements of the session read 0.010–0.011 µs at L=13 B=1**, later
  stable at 0.015–0.016: wallaby core-state/turbo drift between runs, exactly the
  gen_r4 within-lease drift the interleaved racer exists for. Trust interleaved
  A/Bs only.

### Borrowed, explicitly
- gen_r12 (3D campaign): the -D variant-lane mechanism (item 2).
- d1_composite's r1 record: the USE_ZMM4 cross-host question — their measurement,
  my machinery arbitrates it per host.

### Next round
1. More variant lanes as sibling records request them (one table row each);
   candidates: d1_planner R8_THRESH sweep via gr_pick_value if planner becomes
   competitive anywhere.
2. Round-end `gr_wisdom_drop_prefix("d1_race/")` strip protocol, once the campaign
   adopts it.
3. If an adopter appears (d1_planner is still the natural first), split the lib
   into a documented header section.
4. The chain race still always times candidate 0 (loop over exec winner) even when
   a native chain exists and won last round on the same hashes — a warm-wisdom hit
   already skips it, so this is only cold-cell cost; measure whether it matters
   before touching it.

## Round d1_r3 (2026-09-03)

### Context: read the scoring node's own wisdom file before changing anything
The r2 leaderboard showed d1_race trailing its best gated sibling at exactly the
memory-bound cells: 10007 B=64 single 258 vs d1_bluestein's 195.8 (1.32x), 16384
B=1 chain 80.1 vs d1_pow2's 69.6, 4096 B=256 16.7 vs d1_pow2's 13.5. Before
"fixing the race", I read `results/wisdom1d_a80n0.json` — the node's own r2
verdicts — and it split the problem in two:

1. **4096 B=256: a genuinely wrong verdict.** exe.r2 picked d1_twiddle at
   18.6 us/x; pow2 standalone runs 13.5 warm (its cold/warm gap at a 32 MiB
   working set is huge — driver spread 48%). Interleaved sampling is right for
   core-state drift but wrong for cache state past L2: each candidate's sample
   runs on caches the previous candidate just evicted, while the driver scores
   every backend alone and WARM (min over repeated runs).
2. **10007 B=64 and 16384 B=1 chain: the verdicts were RIGHT and the ship was
   slow.** The race picked bluestein (16590/64 = 259 us/x) and pow2's native
   chain — and the race's number matches the driver's shipped measurement to
   1%, at 2% spread. The same source standalone ran 32% / 15% faster. So the
   arm-in-my-process was steadily slower than the arm-alone: its plan was
   created MID-RACE, with ~7x the cell's bytes of race buffers live and every
   rival arm allocating around it, so its scratch lands in a completely
   different (conflict-prone) heap/page layout than the standalone binary the
   leaderboard compares against.

### What changed (impl/d1_race.c)

**1. warm_each — warm-verdict sampling (fixes #1).** New `gr_opts.warm_each`:
one untimed rewarm run of the SAME candidate immediately before every timed
sample (interleaved sample loop and the noise-gate confirm phase). Verdicts now
reflect each arm in the warm state it is actually scored in. Cost <=2x race
time (still inside the 5 s / 8 s budgets); negligible at small L where a sample
is thousands of reps. Both stages race with it; salts bumped exe.r2 -> exe.r3,
chn.r1 -> chn.r2 so the node re-races everything under the new protocol.
Library default stays 0 (API-compatible for adopters).

**2. Post-race replan on a clean heap (fixes #2).** After both stages: destroy
EVERY arm plan (winners included), free the seven race buffers (MB-sized, so
glibc munmaps them immediately), then re-create only the winner plan(s) on the
emptied heap — approximating the standalone binary's allocation environment.
Skipped on wisdom-warm creates (`cx->dirty` flag: race buffers were never
allocated, so the single winner plan already sits on a clean heap and warm
setup cost is unchanged). If a re-create fails the arm goes dead and the
self/loop fallbacks ship, so the existence guarantee is untouched.

**3. Two variant lanes siblings asked for** (one table row each, exe_only):
  - `d1_composite -DUSE_ZMM2X2` ("+zmm2x2") — their r2 next-round item 1 asks
    for exactly one on-node ymm2-vs-zmm2x2 A/B (SPR: zmm2x2 loses 0.037 vs
    0.032 because 512-bit shuffle/FMA share p0+p5 there; ICL's port mix
    differs). On wallaby at L=60 B=8 it already won its race by 2.1%.
  - `d1_planner -DLANE_MAX_N=16384` ("+lane16k") — their r2 asks to re-derive
    the 8-lane batched cap (tuned on wallaby's 2 MB L2) on the scoring node;
    the knob is #ifndef-guarded so a -D lane can raise it.
  Kept +zmm4: the a80n0 wisdom shows it WON the L=60 B=1 cell in r2 (0.055 vs
  base 0.078 at race time) while base kept B=512 — the r2 cross-host bet paid,
  and it is the round's proof the variant-lane mechanism earns its keep.

### Measured (wallaby SPR 6448Y, nice'd core 100, load ~26 — no Ice Lake
reservation again, job 440424 dead; numbers are machinery checks, the node
re-races fresh). All PASS check.py (rel L2 1.4e-16 … 1.1e-15, tol 1e-12);
chained cells PASS the map-chain gate; repeatable across two processes at
13/32-chain/32-batched-chain.

| cell | ships | min us/xform (r2 wallaby) |
|---|---|---|
| 13 B=1 | d1_prime | 0.009–0.010 (0.010–0.016) |
| 31 B=512 | d1_prime | 0.031 (0.035) |
| 32 B=1 m=100k chain | d1_twiddle chain | 0.065 |
| 32 B=512 m=1000 chain | — | 0.027 (0.026) |
| 60 B=8 | d1_composite+zmm2x2 | 0.028 |
| 4096 B=256 | d1_pow2 (r2 node verdict was twiddle) | 9.59 (9.24) |
| 10007 B=64 | d1_bluestein | 114 |
| 16384 B=1 m=250 chain | d1_pow2 native chain | 45.5 |
| 100003 B=8 | d1_bluestein, chain=loop (tie pick) | 2347 (3042 node) |

The 10007 B=64 in-process-vs-standalone A/B on wallaby (3 alternating runs
each): bluestein alone 104–127, inside d1_race 112–139 — no steady gap under
login-node noise; the decisive test is the quiet node, where the r2 gap was a
steady 32%.

### What did not work / lessons, with the number
- **`gcc … | head -20 && echo BUILD_OK` reported success on a failed build.**
  head exits after 20 lines, gcc dies on SIGPIPE mid-compile, and BUILD_OK
  keyed on head's exit status — a check that could not fail. I then "measured"
  a stale r2-era binary from the same dev dir until the exe.r1 salt in its
  verbose output gave it away. Verify the binary's salt string (`strings bin |
  grep exe.`) before trusting any dev number.
- **"Warm setup regressed to 2.2 s" was not a regression.** strace showed a
  fresh d1_pow2 compile on every run: its implementer was editing the source
  live, every run saw a new hash, the candidate-list signature re-keyed, and
  the full re-race + compile re-ran — the churn design working exactly as
  intended. Third live-fire round for hash-carrying names; zero stale verdicts.

### Borrowed, explicitly
- d1_composite's r2 record: the USE_ZMM2X2 kernel and the explicit request for
  an on-node A/B (lane 2); their in-file knob discipline makes the lane free.
- d1_planner's r2 record: the LANE_MAX_N re-derivation request (lane 3).
- The warm_each idea is the driver's own scoring protocol (warmup + min over
  repeated runs) imported into the race; the diagnosis came from reading the
  scoring node's wisdom file against the leaderboard, which I recommend to any
  routing entry: the node tells you exactly what it chose and at what measured
  cost.

### Next round
1. Read wisdom1d_a80n0.json again first: check exe.r3 verdicts at 4096 B=256
   (should now be pow2) and whether the shipped 10007 B=64 / 16384-chain
   numbers closed on the standalone arms. If the replan did NOT close the gap,
   the next suspect is .so-vs-binary code layout (test: -falign-functions=64,
   or a second compile that statically prelinks the winner arm at plan time).
2. If +lane16k or +zmm2x2 win anywhere on the node, tell their owners the
   verdict so they can flip defaults; drop lanes whose question is settled
   (zmm4 stays only if it still splits B=1 from B=512).
3. Round-end `gr_wisdom_drop_prefix("d1_race/")` strip protocol, still pending
   campaign adoption.
4. Chain race cold-cost item from r2 still open, still unmeasured.

## Round d1_r4 (2026-09-03)

### Context: the verdicts are right, the shipped instance is what lags
Did r3's next-round item 1 first: read `results/wisdom1d_a80n0.json` against the
r3 leaderboard. exe.r3 fixed the wrong verdicts (4096 B=256 now picks pow2;
10007 B=64 and 16384 B=1 chain shipped at 1.00x). What remains is a residual
per-cell gap between the SHIPPED arm and the same source standalone, with BOTH
signs:
- slower in-process: L=32 B=512 exec 0.0176 vs d1_pow2 0.0154 (race-time,
  shipped and leaderboard all agree at 9.0 us/call — the instance genuinely
  runs 14% slower); L=16384 B=64 exec 70.8 vs 54.4 (30%); L=31 chain 0.0619 vs
  d1_batchlane 0.0546, L=60 chain 0.1252 vs d1_composite 0.1105, L=4096 B=1
  chain 12.5 vs d1_pow2 11.05 (all ~13%).
- FASTER in-process: L=1024 B=1 1.53 vs pow2 1.62, L=16384 B=1 45.7 vs 52.9,
  L=65537 B=16 1183 vs d1_rader 1340.
Both signs at fixed spread ⇒ per-process placement/layout luck, frozen when the
plan's scratch was allocated (and, for the KB-working-set chain cells where
data placement cannot matter, frozen at .so link time: code placement). The r3
replan-on-clean-heap re-rolls that luck ONCE, blind.

### What changed (impl/d1_race.c)

**1. First-call placement probe (the round's mechanism).** Two facts from
driver.c make the fix possible: the driver allocates in/out/pong BEFORE
fft1d_create and always runs discarded warmup units before calibrating, so the
first execute/chain call (a) is never timed and (b) passes the REAL scored
buffer addresses. fft1d_create now installs one-shot trampolines in the
flattened dispatch; on first contact the probe re-creates the winner arm's plan
up to 3 more times, each behind heap spacers (a small odd-sized malloc shifts
the sbrk arena tail, a large odd-sized one shifts the mmap base and with it the
2 MiB THP phase of every big scratch block), times every instance on the actual
driver buffers (~60 us/sample, min of 3, one warm lead-in), and keeps the best
with 2% hysteresis. Losers destroyed immediately; expensive creates (>150 ms)
stop the probe after one candidate; `D1_RACE_NO_PROBE=1` disables. After the
probe the dispatch is re-flattened to the bare winner fn+plan — race-path/
ship-path equivalence unchanged.
  REPEATABILITY GUARD: the driver's two-process check compares outputs
BITWISE, and a plan-time-adaptive arm could round differently per instance. So
the original plan survives the probe, and a replacement ships only if its
output checksum (FNV over the full output buffer) is bit-identical to the
original's; otherwise the probe reverts. Verified repeatable at 65537 B=16 and
100003 B=8 (both with replacements in play) and 31-chain.
  Measured on wallaby (probe's own before→after, real-buffer us/call):
16384 B=1: 29.9→27.6 (-8%); 16384 B=64: 2330→2272 (-2.5%); 60 B=1 exec:
0.0316→0.0284 (-10%); 13 B=1: 0.0233→0.0206 (-12%); 65537 B=16 COLD:
12160→11060 (-9%). The headline is the WARM path: on a wisdom-warm create at
65537 B=16 the single re-created plan landed in a bad mode — 22560 us/call,
2.0x the good mode — and the probe recovered it to 11560. That is exactly the
r3 leaderboard failure at 16384 B=64 (setup 0.013s = warm create, no race, 30%
slow, nothing to catch it). A/B after the change, alternating 3 rounds at
16384 B=64: pow2 standalone 36.2/35.6/35.0, d1_race 34.8/34.7/35.0 — the
router now matches or beats its arm.

**2. +al64 alignment lanes** (`-falign-functions=64 -falign-loops=32`) for
d1_pow2, d1_batchlane, d1_composite — the three bases owning the gapped
KB-working-set chain cells, where the only placement left to re-roll is code
placement, fixed per link. A second draw of that luck as ordinary variant
lanes; the per-cell race keeps the faster draw (exe_only=0: the chain cells
are the motivation). On wallaby the lanes already took cells in their first
races: pow2+al64 won 16384 B=64 exec, batchlane+al64 won the 32 B=512 chain.

**3. Lazy p->tmp.** The loop-chain FFT scratch (a full cell-sized allocation)
was allocated up front in every create; now only when the loop fallback
actually ships (carm==NULL). An exec-only process no longer carries an extra
cell-sized block the standalone arm binary lacks.

**4. Salts** exe.r3→exe.r4, chn.r2→chn.r3 (roster change re-keys via gr_sig
anyway; the bump keeps this round's verdicts legible in the wisdom file).

### Measured (wallaby SPR 6448Y, nice'd core 100, load ~0; Ice Lake
reservation job 440424 dead again — numbers are machinery checks, the node
re-races fresh). All cells PASS check.py (rel L2 1.2e-16 … 1.2e-15, tol
1e-12); chained cells PASS the map-chain gate (3e-15 … 9.6e-15, tol 1e-10);
two-process repeatability identical at 65537/100003/31-chain.

| cell | ships | min us/xform (prior wallaby) |
|---|---|---|
| 13 B=1 | d1_batchlane | 0.017 |
| 13 B=512 | d1_prime | 0.007 (0.009) |
| 31 B=1 m=100k chain | d1_prime chain | 0.043 |
| 32 B=512 | d1_batchlane | 0.012 |
| 32 B=512 m=1000 chain | d1_batchlane+al64 chain | 0.023 (0.027) |
| 60 B=1 m=60k chain | d1_composite chain | 0.085 |
| 1024 B=1 | d1_pow2 | 1.022 (1.048) |
| 4096 B=256 | d1_twiddle (fresh warm verdict) | 7.68 (9.59) |
| 16384 B=1 | d1_pow2 | 27.2 (27.6) |
| 16384 B=64 | d1_pow2+al64 | 34.7 vs pow2 standalone 35.0–36.2 |
| 65537 B=16 | d1_planner (took it from rader) | 684 (779) |
| 100003 B=8 | d1_bluestein | 1584 (2347) |

Probe wall-clock cost: 65537 B=16 warm run total 0.69 s; 100003 B=8 warm
4.9 s (dominated by the arm's own create, not the probe); small cells ~ms.
Setup is not scored; all inside budget.

### What did not work / lessons, with the number
- **Repeated my own r3 build-check mistake before catching it**: piped gcc
  through `head -5` and read BUILD_RC=0 off head with no binary produced
  (SIGPIPE kills gcc, head exits 0). My own r3 lesson, ignored for one
  command. Redirect to a log file; the salt-string check
  (`strings bin | grep exe.`) then confirmed the right binary.
- **Probe hysteresis matters more than tries.** With 3 samples at ~60 us each,
  a 2% swap threshold produced zero false swaps across every cell tested
  (probes that found nothing reported "kept" consistently across reruns);
  an earlier draft without hysteresis flapped between equal instances.
- The 4096 B=256 verdict on wallaby is now d1_twiddle at 7.68 us/x (pow2's r3
  number here was 9.59): a fresh warm_each race verdict, not drift — twiddle's
  source churned this session. The node decides for real.

### Borrowed, explicitly
- The probe is the driver's own scoring conditions (real buffers, warm, min
  over samples) imported one level deeper than r3's warm_each: from race-time
  into ship-time. Diagnosis again came from reading the scoring node's wisdom
  file against the leaderboard (recommended to any routing entry).
- gen_r12's -D variant-lane mechanism reused verbatim for the +al64 lanes
  (three table rows, zero new machinery).

### Next round
1. Read wisdom1d_a80n0.json + leaderboard: did the probe close 32 B=512 and
   16384 B=64 to 1.0x of the standalone arm? If a gap SURVIVES the probe, the
   remaining suspect is code placement at cells too big for +al64 to cover —
   consider +al64 lanes for rader/bluestein/twiddle.
2. If +al64 wins cells on the node, tell the owners (their standalone builds
   could adopt the flags outright); drop lanes whose question is settled.
3. Probe currently re-rolls placement only for the WINNER arm; at cells where
   the race margin is <5% the runner-up in a good placement could beat the
   winner in a bad one. Racing top-2 arms through the probe would catch that;
   measure whether any node cell is actually that close first.
4. Round-end wisdom-strip protocol and the r2 chain-race cold-cost item:
   still open.

## Round d1_r5 (2026-09-03)

### Context: the verdicts are right; the MEDIANS lose the cells
r4 next-round item 1 first: read `results/wisdom1d_a80n0.json` against the r4
leaderboard.  Every exe.r4/chn.r3 verdict matches the leaderboard's fastest
arm — the routing layer has nothing left to fix in WHO it picks.  What loses
cells now is the shipped instance's run-to-run placement mode, visible as
median-vs-best splits at fixed verdicts:
- 100003 B=8 single: shipped 1.37x on the median (3398) with the BEST run at
  1.00x (2474 = d1_bluestein standalone).  Two of three runs drew bad.
- 4096 B=256 single: spread 57.9% (median 13.0, best 9.3 vs pow2 10.4).
- 16384 B=1: spread 28.2%; 65537 B=16: 40.9%; 10007 B=64: 39.2%.
The r4 probe re-rolled placement blind, up to 3 draws, and KEPT whatever it
got: a fresh process has no idea what a good draw looks like, and at
expensive-create cells it stopped after one draw.

### What changed (impl/d1_race.c)

**1. Wisdom-referenced probes (the round's mechanism).**  After every
first-call probe, the best shipped instance's us/call — measured on the REAL
driver buffers, warm, min-of-3, identical methodology in every process — is
stored in wisdom per (cell, arm@srchash): keys `pref.r1` (exec), `cref.r1`
(native chain), `crefL.r1` (loop chain).  The next process reads the ref and
(a) STOPS IMMEDIATELY if its first draw is within 6% — the common case is now
cheaper than r4's always-3-draws; (b) otherwise keeps re-rolling: up to 6
heap-spacer data draws (r4: 3), then up to 2 ALT TEXT MAPPINGS — dlopen a
byte-copy of the arm's .so (dlopen dedups by (dev,inode): a hardlink or
symlink returns the SAME mapping, it must be a real copy) so the code lands
at a fresh address, re-create the plan there, and rebind the arm's function
pointers if that instance wins.  Data spacers cannot move text; the small
L1-resident cells that ship 13% slow in 2 of 3 node runs (31/60 chains,
where data placement is irrelevant) are per-process CODE placement luck, and
this is the first mechanism that can re-roll it.  Expensive creates
(>150 ms) keep drawing only while a ref says the current draw is bad; 6 s
hard deadline; refs update monotonically downward (a bad process must never
raise the bar for the next).  The r4 bitwise-output guard is unchanged and
applies to alt-mapping instances too (same code bits => identical rounding).
- Measured, wallaby (SPR 6448Y, nice'd core 100, load ~0.5): 16384 B=64 run 1
  probe replaced 2224->2062 us/call and stored the ref; runs 2-3 measured
  2050/2072, hit the ref window, kept instantly (zero extra creates, warm
  setup 7-10 ms).  65537 B=16 chain probes recovered 42.5k->39.9k (-6%) and
  43.8k->39.5k (-10%) in successive processes.  100003 B=8 chain: cref
  replaced 65.7k->55.4k us/call (-16%); shipped 1669/1673 us/xform in two
  processes (bitwise REPEATABLE), warm setup 0.024 s.  Forced-alt test
  (planted unreachable ref 1000 at 16384 B=64): 6 data + 2 alt draws, .alt
  copies created and dlopen'd, original kept, output repeatable — the
  worst-case path is bounded and correct.

**2. Pre-gate outside the race budget (live-churn bug, found and fixed this
round).**  A broken mid-edit d1_bluestein (two successive hashes both gating
e0) burned the 15 s fork-gate watchdog TWICE inside the exec race's 5 s
deadline at 100003 B=8; every later candidate's setup was starved, the only
valid arm (planner) never raced, gr_race returned -1, and SELF — the scalar
Bluestein fallback — shipped a scored cell, then flipped winners across
processes (NOT REPEATABLE: the driver's chained unit ends with an execute,
so the exec engine is in the chain output).  Fix: when the wisdom answer for
the exact (key, candidate-signature) is absent, all candidate arms are gated
and materialized BEFORE gr_pick, outside any deadline (`d1r_wisdom_has`
mirrors gr_pick's key construction, so the warm path still materializes only
the winner); plus a degenerate-race fallback that ships the first READY arm
rather than silently self.  On a frozen-snapshot round a broken sibling now
costs setup wall-clock once per (L,B,hash), never a race verdict.

**3. Loop-fallback chains now probe (r4 gap, plus a flatten bug).**  When no
native chain ships, the chain trampoline was never installed (flatten routed
carm==NULL straight to the loop), so loop-chain cells got no placement
re-roll at all — and a chained cell may never call fft1d_execute before the
timed region.  The tramp now installs for the loop case too and probes the
EXEC arm through the loop thunk under `crefL.r1`.  Verified by forcing the
loop verdict in dev wisdom at 100003 B=8: crefL fired, replaced 60.3k->58.6k.

**4. Roster trimmed to the lanes that won on the node** (r4 item 2 — owners,
these are your verdicts): DROPPED d1_composite+zmm2x2 (base composite kept
L=60 B=512 on a80n0; SPR already said no — the zmm2x2 question is settled
both hosts), d1_planner+lane16k (LANE_MAX_N=16384 took no batched cell on
a80n0), d1_batchlane+al64 and d1_composite+al64 (base won every contested
cell on the node; the r4 wallaby wins did not transfer).  KEPT
d1_composite+zmm4 (won L=60 B=1 again in r4 — on ICL the zmm4 batched kernel
is the B=1 pick while base keeps B=512) and d1_pow2+al64 (won 16384 B=64
exec and both 1024 chains on the node — pow2's owner could adopt
`-falign-functions=64 -falign-loops=32` outright).  Fewer candidates =
tighter races; the roster change re-keys signatures anyway.

**5. Salts** exe.r4->exe.r5, chn.r3->chn.r4.

### Measured (wallaby, nice'd core 100, load ~0.5; all machinery checks —
the node re-races fresh).  All tested cells PASS check.py (rel L2 1.2e-16 …
1.3e-15, tol 1e-12); chained cells PASS the map-chain gate; two-process
bitwise repeatability at 32-chain / 65537-chain / 100003-chain / NO_PROBE.

| cell | min us/xform (r4 wallaby) | notes |
|---|---|---|
| 13 B=1 | 0.009 (0.017) | |
| 32 B=512 | 0.012 (0.012) | |
| 32 B=512 m=1000 chain | 0.023 (0.023) | |
| 60 B=8 | 0.023 (0.028) | |
| 128 B=512 | 0.084 | |
| 4096 B=256 | 6.87-6.95 x3 procs (7.68) | ref hit instantly warm |
| 16384 B=1 | 23.9-24.8 x3 (27.2) | |
| 16384 B=64 | 31.9-32.5 x3 (34.7) | run1 probe -7%, then ref hits |
| 65537 B=16 m=20 chain | 543-597 | chain probe -6%/-10% |
| 100003 B=8 m=15 chain | 1669-1673 x2 (r4 node 2857) | cref -16%; big sibling gains too |

Much of the raw improvement is siblings getting faster this session
(bluestein/pow2/rader all churned under me — hash-carrying names re-keyed
everything, zero stale verdicts, fifth live-fire round for that design).

### What did not work / closed questions, with the number
- **The steady L=64 B=512 chain gap does NOT reproduce on wallaby**: 3
  alternating rounds, d1_batchlane standalone 0.047/0.047/0.047 vs inside
  d1_race 0.047/0.047/0.047 — identical.  On the node it was a steady 15%
  (race-time = ship-time = 0.0782 vs standalone 0.0677, 0.1% spread).  A
  steady-across-processes gap cannot be address luck (ASLR already varies
  addresses), so alt mappings will not fix it and the ref mechanism cannot
  see it (the ref is the best IN-PROCESS instance; the standalone number is
  invisible to my binary by construction).  Remaining suspects: -fPIC
  codegen or the deterministic relative layout of driver buffers vs plan
  scratch in a binary that dlopens.  Node-only; needs an on-node experiment.
- **r4 item 3 (probe the top-2 arms where the race margin is <5%) is closed
  by design, not by measurement**: the bitwise-output guard means a probe
  can never ship a DIFFERENT arm than the raced winner (different algorithm
  => different rounding => checksum mismatch => revert).  Cross-arm
  second-guessing would need driver-visible tolerance, which repeatability
  forbids.  Not pursued further.
- **Ships-self incident** (see item 2): the number that exposed it — run 1
  shipped exec=self at 100003 B=8 and the exe.r5 wisdom entry showed
  winner=planner margin=0 written by run 2 only; two bluestein gate entries
  (both e0) from successive hashes dated the churn.

### Borrowed, explicitly
- The wisdom-ref idea is the driver's own protocol pushed one level further:
  r3 imported warm-min-over-runs into the race, r4 into ship time, r5 makes
  the measurement PERSISTENT so separate processes share one bar.  Diagnosis
  again came from reading the scoring node's wisdom file against the
  leaderboard (recommended, again, to any routing entry).
- Alt-text mappings: the inode-dedup gotcha is documented dlopen behavior;
  the mechanism is new here (nothing in gen_race re-rolled code placement).

### Next round
1. Read wisdom1d_a80n0.json pref/cref entries against the r5 leaderboard:
   (a) cells whose shipped median now sits at ~1.0x their own ref and the
   arm standalone -> done; (b) cells at ref but still above standalone (the
   L=64 B=512 chain shape) -> structural .so penalty: propose an on-node
   A/B of -fPIC vs a static-link harness to the monitor, or accept; (c)
   cells still above ref on the median -> raise D1R_PROBE_ALT_TRIES /
   deadline, the draws are the bottleneck.
2. If broken-sibling e0 gates appear in the node wisdom (frozen snapshot
   with a dead arm), confirm the pre-gate kept the races intact.
3. Wisdom-strip protocol and the r2 chain-race cold-cost item: still open.

## Round d1_r6 (2026-09-03)

### Context: the verdicts and instances are right; the STATISTIC is wrong
r5 next-round item 1 first: read `results/wisdom1d_a80n0.json` (exe.r5/chn.r4 +
the pref/cref refs) against the r5 leaderboard.  Two distinct failure shapes
remained, and both are the same root cause — the race and the probe optimize a
min-like statistic while the driver scores the MEDIAN of >=20 ms samples and
the leaderboard ranks the median of per-process medians:
1. **Close verdicts break the wrong way.**  Exec TIES at 13/32/64/1024 B=512
   went to the lowest index (pow2/batchlane) while the driver's medians had
   prime/planner/twiddle 14–21% faster (e.g. 32 B=512: shipped pow2 0.0176 vs
   planner 0.0154; 1024 B=512: shipped pow2 2.12 vs twiddle 1.74 at 0.9%
   spread).  16384 B=64 picked twiddle on a min-based 7.6% margin that LOSES
   on medians (pow2 44.4 vs twiddle 50.0).  The tie doctrine was designed for
   noise robustness, but at cells where the driver separates arms by 14% it
   costs real cells.
2. **Probes accept burst-fast/steady-slow draws.**  At 10007 B=1 the r5 ref
   (111.8 us = bluestein standalone) coexisted with a shipped MEDIAN of 142.6
   (best 109): min-of-3 60 us bursts hit the fast mode of an instance whose
   steady 20 ms aggregate is slow.  Same shape at 65537 B=16 (median 1000 vs
   best 888.6), 100003 B=1 (3004 vs 2267), 16384 B=64.
Plus one verdict-quality bug: the chain race's 4e6-POINT work cap raced L=60
B=512 at m=130 against a graded 600 (a 20 ms unit!), and the m-sensitive
batchlane-vs-composite verdict went to batchlane (shipped 0.0748) while
composite standalone runs 0.0605.  Points are the wrong cap unit across
regimes — small-L chains cost ~1 ns/point, big-L memory-bound ~8.

### What changed (impl/d1_race.c)

**1. The probe measures the driver's statistic.**  `d1r_probe_time` is now the
MEDIAN of 5 samples, each a reps-loop sized to >=250 us (was min-of-3 x 60 us).
Ref keys bumped pref.r1→pref.r2, cref.r1→cref.r2.m<mp>, crefL likewise (old
min-based refs are unreachable bars for median draws — they would burn every
draw every process).  Measured: the probe median now matches the driver's
scored median within 0.3–1.6% at every cell I checked (10007 B=1: probe 66.5,
driver median 67.6; 65537 B=16: probe 9500/16, driver 592).

**2. Ship-time runner-up arbitration (the round's mechanism).**  gr_race now
reports the runner-up index (`pi.ridx`, new gr_pick_info field); both stages
persist it per verdict ("<fullkey>/r2" = "name|margin", only when tie or
margin<20%).  On first contact — real driver buffers, discarded warmup — a tie
or sub-12%-margin verdict is re-decided: the winner's probed median vs the
runner-up under the IDENTICAL protocol (same reps/mp/buffers; the runner gets
one heap-spacer redraw if it starts behind, so placement luck does not decide
the arbitration).  Flip needs 5% hysteresis; the verdict entry is rewritten
and a "<fullkey>/xc" marker settles the question ONCE per (cell, roster) —
later processes warm-load the arbitrated verdict, preserving the driver's
two-process bitwise repeatability (the flip completes during run 1's warmup,
before any scored or compared output).  REPEATABILITY GUARDS, both verified:
  - exec arbitration NEVER runs in a process where the chain ran first (an
    exec flip re-keys chain verdicts via the loop-c0 name; flipping after this
    cell's chain was timed would make its later runs race a different chain
    than run 1 shipped).  In chained processes execute is only called after
    the timed region, so nothing is lost; the persisted /r2 lets the SINGLE
    cell's first process arbitrate even when the cold race happened in a
    chained process (cell order is not ours to choose).
  - forced-flip test (doctored dev wisdom: stored winner=planner, runner=pow2,
    marker cleared): run 1 measured 6.835 vs 6.163, FLIPPED, rewrote the
    verdict; run 2 warm-loaded the flip; outputs bitwise identical.
Found-and-fixed during dev: the "no runner recorded" sentinel was `margin<0`,
but a TIE's stored margin is legitimately a small NEGATIVE (runner sampled
faster inside the noise window) — every tie cell silently skipped arbitration.
Sentinel now -1e9.

**3. Honest chain-race m (time-capped off the exec verdict).**  m_race =
min(graded, max(4e6-points cap, 25 ms / (1.6 x exec_us))), where exec_us is
the exec verdict's stored us/call — deterministic across processes (the chain
key embeds the m).  L=60 B=512 now races at the full graded m=600 (was 130),
32 B=512 at 1000 (was 244), 13 B=512 at 2000 (was 600), 16384 B=64 at 7 (was
3); the expensive cells keep their old caps (100003 B=8 stays m=4).  The
chain PROBE likewise rises toward the driver's m under a ~60 ms/call cap
(mp derived from the stored chain-verdict us, so refs stay comparable; ref
keys carry the mp).

**4. -fno-semantic-interposition on the .so compile.**  The arm recipe now
matches the Makefile exactly (verified side by side) plus this one PIC repair:
with plain -fPIC gcc treats exported functions as interposable — no direct
intra-TU calls, no inlining of them — even under our -Wl,-Bsymbolic, and PIC
codegen is the ONLY remaining structural .so-vs-binary difference (r5 closed
placement; the steady node-only in-process gaps at 32/64 B=512 exec and the
64 B=512 chain are the candidate beneficiaries).  The full recipe string is
folded into EVERY arm's wisdom hash, so recipe changes re-key the .so cache
and all verdicts (a stale same-name .so would otherwise survive).  No wallaby
regression anywhere; the node decides whether it closes the steady gaps.

**5. Variant lanes: ALL DROPPED (owners, these are your verdicts).**
d1_composite+zmm4 took L=60 B=1 in r2/r4 but lost it to BASE composite in r5
(exe.r5 verdict); d1_pow2+al64 won 16384 B=64 exec + both 1024 chains in r4
and NOTHING in r5, and the alt-text-mapping probe now re-rolls code placement
at ship time, which is what the lane was for (pow2's owner can adopt
-falign-functions=64 -falign-loops=32 outright if they want the deterministic
version).  Fewer candidates = tighter races at exactly the close cells this
round targets.

**6. Salts** exe.r5→exe.r6, chn.r4→chn.r5.  D1_RACE_FORCE now also matches by
name prefix (hash-free forcing in dev).

### Measured (wallaby SPR 6448Y, nice'd core 100, load 0.2–5; Ice Lake
reservation job 440424 dead again — numbers are machinery checks, the node
re-races fresh).  All tested cells PASS check.py (rel L2 1.2e-16 … 1.4e-15,
tol 1e-12); chained cells PASS the map-chain gate; bitwise repeatability
verified at 32 single / 32-chain / 100003-chain / 65537 / the forced-flip
path.

| cell | min us/xform (r5 wallaby) | notes |
|---|---|---|
| 13 B=1 | 0.009–0.016 (0.009) | bimodal core state as always |
| 32 B=512 | 0.012 (0.012) | xcheck: pow2 6.19 vs planner 6.32 -> keep |
| 32 B=512 m=1000 chain | 0.022 (0.023) | |
| 60 B=8 | 0.027 (0.023) | login-node neighbor noise |
| 60 B=512 m=600 chain | 0.042 | race at graded m=600; xcheck composite 12.69ms vs batchlane 12.82ms/unit |
| 1024 B=512 | 1.372 | xcheck pow2 vs pow2+al64 pre-trim |
| 10007 B=1 | 66.9–67.6, sd 0.03% | probe median = driver median +-1.6% |
| 16384 B=64 | 32.06 (31.9–32.5) | xcheck pow2 2055 vs twiddle 2053 -> keep (node will split) |
| 65537 B=16 | 584–592 | cold wall 4.4 s, warm setup 10 ms |
| 100003 B=8 m=15 chain | 1667 (1669–1673) | xcheck native 56.7ms vs loop 57.8ms -> keep |

### What did not work / lessons, with the number
- **The margin<0 sentinel bug** (above): first live run showed a clean tie at
  32 B=512, r2 stored, and NO xcheck line — the exact mechanism the round was
  built for, silently skipped.  Caught because the smoke test greps for the
  xcheck print, i.e. a check that COULD fail.
- **One probe reading of 12.07 us/call vs ref 6.19 at 32 B=512, "kept"** — a
  neighbor landed on core 100 (login-node load was 5 at session start).  All
  draws were equally slow so the ref could not be reached; the driver's own
  numbers in that run were unaffected.  Interleaved A/Bs and refs behave
  correctly under neighbor noise; single absolute probe readings on wallaby do
  not mean anything.
- Sibling churn re-keyed rosters twice mid-session (three exe sigs for one
  cell in my dev wisdom, each with its own /r2 + /xc, all settling
  independently) — sixth live-fire round for hash-carrying names, zero stale
  verdicts, and the arbitration marker correctly re-opens on every roster
  change.

### Borrowed, explicitly
- The whole round is again the driver's own protocol imported one level
  deeper: r3 put warm-min-over-runs into the race, r4 real buffers into ship
  time, r5 made the bar persistent, r6 makes the STATISTIC the driver's
  (median of long samples) and lets it arbitrate between arms, not just
  instances.  Diagnosis, as in r3–r5, came from reading the scoring node's
  wisdom file against the leaderboard — still recommended to any routing
  entry.
- The chain-m time cap is the gen_r13 honest-m lesson finished properly
  (points were never the right unit; the exec verdict's own measured time is).

### Next round
1. Read wisdom1d_a80n0.json /xc markers against the r6 leaderboard: count
   flips and check each flipped cell's median vs the arm standalone.  If a
   flip went the WRONG way (shipped median > runner standalone), the 5%
   hysteresis or the single-redraw fairness needs tightening.
2. Check whether -fno-semantic-interposition closed the steady in-process
   gaps (32/64 B=512 exec vs planner standalone, 64 B=512 chain vs batchlane
   standalone).  If not, the last suspect is the .so data-vs-text layout
   itself; propose a static-prelink A/B to the monitor.
3. If the node's exe.r6 verdicts at 4096/16384 show twiddle-vs-pow2 flips per
   regime, consider per-regime verdict keys (single vs chained cells share
   the (L,B) exec verdict today; the loop-c0 chain already re-keys, exec does
   not).
4. Wisdom-strip protocol and stale build/race1d .so cleanup: still open.

## Round d1_r7 (2026-09-03)

### Context: grading is SHARDED across two nodes, and the arbitration was unfair
r6 next-round item 1, done first — and it needed BOTH wisdom files, because the
round taught me something structural I had missed: sweep.sh shards the cases
across two grading nodes (a80n0 + a81n2, per-host wisdom each), so a cell's
single and chained variants can land on different nodes.  Six a80n0 exec
verdicts had a stored /r2 runner and NO /xc marker — not a bug: their single
variants were graded on a81n2, where the /xc exists.  (Corollary that stays
true: a node that only ever grades the CHAINED variant of an (L,B) can never
arbitrate its exec verdict, because exec arbitration in chain-first processes
is forbidden by the repeatability design.  Shard-dependent, not fixable from
inside my binary; noted for the monitor.)

With the a81n2 file in hand, the r6 leaderboard decomposed into three real
failure modes:
1. **The arbitration was UNFAIR to the runner.**  The winner gets a full
   wisdom-referenced placement probe; r6 gave the runner one blind redraw.
   a81n2 L=60 B=1: xc=keep composite — driver medians had batchlane (the
   stored runner, race margin 3.4%) at 0.0422 vs shipped 0.0489, 14% lost.
   Same shape at 16384 B=64 (kept twiddle 50.1, pow2 44.2) and 10007 B=1
   (kept bluestein, planner 20% faster standalone).  And the one r6 chain
   FLIP (a81n2 4096 B=256) went the WRONG way — flipped to the arm whose
   single instance drew lucky (driver medians 6% the other way).  r6
   next-round item 1 predicted exactly this repair.
2. **The probe statistic still wasn't the driver's.**  My 250 us samples vs
   the driver's --min-sample-ms 20: on a81n2 the L=60 B=1 pref (0.0592, five
   250 us early-process bursts) and the driver's median from the SAME
   process (0.0489) disagree by 20%.  Every close verdict inherited that
   error bar.
3. **Instance recovery failed in most processes at memory-bound cells.**
   10007 B=1 shipped a 138 us MEDIAN over a proven-reachable 111 us ref
   (best run 109); 16384 B=64 median 50 vs ref 40 us/xf.  Eight draws per
   process, all in the same allocator mode, all missing a bar another
   process had reached — the draws were re-rolling ADDRESSES, never the
   allocation MODE.

### What changed (impl/d1_race.c)
**1. Fair arbitration (fixes #1).**  d1r_xcheck_measure is now a full
d1r_probe of the runner toward the runner's OWN per-cell ref (pref.r3 /
cref.r3.m<mp> / crefL.r3.m<mp>) — data draws, allocator modes, alt text
mappings, identical statistic — instead of one measurement plus one blind
redraw.  Best-instance vs best-instance is the only comparison that predicts
the driver's cross-backend medians.  The probe also persists the runner's
ref, so a later flip-back starts from a proven bar.  Flip hysteresis stays
5%; cost is bounded by the probe's own 6 s deadline and the >150 ms
expensive-create brake, paid once per (cell, roster).
**2. Probe samples are the driver's samples (fixes #2).**  One timed probe
sample now aggregates >= 20 ms (D1R_PROBE_SAMPLE_US = the driver's
--min-sample-ms 20; reps cap raised 8192 -> 2^20 so tiny L actually reaches
it), median of 5.  Measured on wallaby: probe median vs the same process's
driver median now agrees to ~2-3% at 10007 B=1 (64.4-65.6 vs 65.2-67.7) and
16384 B=64 (29.0 vs 30.0 us/xf), where r6's own refs sat 20% off at small L.
**3. Allocation-MODE draws (fixes #3).**  Probe draws now cycle glibc
M_MMAP_THRESHOLD through {32M, 128K, 2M, 8M} before each re-create (restored
after), so an arm's mid-sized scratch re-rolls between the sbrk arena and
dedicated mmaps — page-table locality, THP eligibility and conflict phase vs
the driver's pre-allocated buffers all differ, and heap spacers can never
cross that boundary.  Draw budget raised: 8 data draws under a ref (was 6),
3 alt text mappings (was 2); dl_extra parking 6 -> 10 (alt-open degrades
safely to "stop drawing" when full).
**4. Salts** exe.r6->exe.r7, chn.r5->chn.r6, refs pref.r2->pref.r3,
cref.r2->cref.r3, crefL.r2->crefL.r3 (the statistic changed, so r6 refs are
not comparable bars).  Op count: unchanged — this entry ships whatever the
fastest gated sibling ships.

### Measured (wallaby SPR 6448Y, nice'd core 100, load ~0.3; Ice Lake
reservation job 440424 dead again — machinery checks only, the node re-races
fresh).  All tested cells PASS check.py (rel L2 1.2e-16 … 1.4e-15, tol
1e-12); chained cells PASS the map-chain gate; bitwise repeatability at
13 / 60 / 32-chain / 16384 B=64 / 65537 B=16 / 100003-chain.

| cell | r7 wallaby | r6 wallaby | notes |
|---|---|---|---|
| 13 B=1 | 0.009 | 0.009-0.016 | probe cost at tiny L: 0.58 s whole run |
| 32 B=512 m=1000 chain | 0.022 | 0.022 | chain xcheck ran fair, keep |
| 60 B=1 | 0.028-0.030 | — | live FLIP composite->planner on run 1 (9%); run 2 re-raced under churned sig, arbitrated fresh, outputs bitwise identical |
| 10007 B=1 | 65.2-67.7 x3 procs, sd <=0.7% | 66.9-67.6 | ref hit or 1 replace every proc |
| 16384 B=64 | 29.5-30.0 x3 | 32.06 | probe median = driver median +-3% |
| 65537 B=16 | 587-590 | 584-592 | probe -3% replace, ref hit run 2 |
| 100003 B=8 m=15 chain | 1675-1683 | 1667 | chain probe -17% replace run 1 |

The headline live-fire result: at L=60 B=1 the cold race picked composite by
4.3% and the FAIR arbitration flipped to planner (0.0298 vs 0.0326 probed
medians) — the exact r6-failure shape (thin-margin verdict, runner better on
the score statistic), caught by the mechanism built for it, with the flip
persisted and run 2 consistent.

### What did not work / lessons, with the number
- **Two attempts at a forced chain-flip test were orphaned by sibling churn
  before the binary could read the doctored key** (pow2's hash changed twice
  in ~3 minutes; each new exec winner re-keys the chain sig via the loop-c0
  name).  And the chosen cell was doomed anyway: batchlane-vs-twiddle at
  32 B=512 probe 2.2% apart, below the 5% flip hysteresis.  Accepted
  coverage: the flip DECISION code is r6's (verified by r6's forced test and
  a81n2's live 4096 B=256 flip); this round changed only the measurement
  feeding it, which every keep-path run exercised; and the exec flip fired
  live at L=60.  Lesson: under live churn, doctored-wisdom tests must target
  wide-margin cells and run within seconds of the doctoring.
- **A NO_PROBE run read 0.043 us/xf where the probed path ships 0.022** at
  the 32 B=512 chain — that is the un-recovered create-time instance plus my
  doctored verdict, i.e. exactly what disabling the probe means; not a bug.
- gr_wisdom files now carry six generations of dead salts (chn.r1.* etc.,
  ~60% of the file); harmless but the strip protocol (open since r2) is
  overdue.

### Borrowed, explicitly
- The whole round continues the r3-r6 program: import the driver's scoring
  protocol one level deeper.  r7 completes it — SAME statistic (20 ms-sample
  medians), SAME treatment for both arms of every arbitration.
- The sharding discovery came from reading BOTH nodes' wisdom files against
  the leaderboard (extending my own r3 recommendation: read the scoring
  node's wisdom — plural, now).
- M_MMAP_THRESHOLD cycling is new here; the diagnosis (same-mode draws
  cannot reach another process's ref) is r6's median-vs-ref gap data.

### Next round
1. Read both nodes' pref.r3/xc entries vs the r7 leaderboard: (a) did the
   fair arbitration flip L=60 B=1 / 16384 B=64 / 10007 B=1 to the arms the
   driver medians favor, and did the flips land at ~1.0x of the new arm's
   standalone median?  (b) did the mmap-mode draws close the median-over-ref
   gaps (10007 B=1 was 138-over-111)?  If medians still sit over reachable
   refs, the next lever is madvise(MADV_HUGEPAGE)-style hints, which need
   the arm's cooperation — talk to owners.
2. The steady L=128 B=1 chain gap (shipped batchlane native chain 0.1719 vs
   its standalone 0.1537, 3% spread — steady, so not placement) is the last
   .so-vs-binary structural suspect.  Propose to the monitor a one-off A/B:
   my binary vs a static-link harness of the same arm source on the node.
   -fno-semantic-interposition (r6) did not close it.
3. Exec verdicts on a chain-only shard never arbitrate (see context).  If
   the r7 shard split leaves a lagging exec cell in that state, consider
   letting the SINGLE-cell process of the OTHER node's shard... it cannot —
   different wisdom file.  Only the monitor can co-locate a cell's two
   variants on one shard; mention it.
4. Wisdom-strip protocol and stale build/race1d .so cleanup: still open,
   now also bloating every read-modify-write of the wisdom file.

## Round d1_r8 (2026-09-04)

### Context: r7 was never scored, and the rivals all grew first-call probes
Two structural facts shaped the round.  (1) The r7 sweep never ran — no r7
leaderboard exists, the node wisdom still ends at exe.r6, and the rounds
machinery moved on ("d1_r7 looks stalled... starting d1_r8 anyway"), so my
whole r7 stack (fair arbitration, driver-statistic probes, mmap-mode draws)
reaches its first scored round together with r8's changes.  (2) Every rival's
r7 record ships an IN-FILE ONE-SHOT FIRST-CALL PROBE adopted from this entry
(prime/rader/batchlane in full, twiddle's carve probe), and those fire inside
MY timed windows.  Measured on wallaby core 100 (dlopen the cached .so, time
call 1 vs call 2): d1_prime at 13 B=1 first call 237 ms vs 0.1 us steady;
13 B=512 360 ms vs 4 us; d1_rader at 65537 B=16 286 ms vs 9.4 ms; batchlane
55 us (cheap).  Two places that bites: (a) a cold race's warmups pay ~0.05-
0.36 s per arm INSIDE gr_race's 5 s deadline — 3-5 arms of that starves the
later candidates' samples (the d1_r5 gate-starves-the-race incident, back in
probe form); (b) my placement probe CALIBRATED reps off the plan's first
call, so a fresh sibling plan read as ~0.3 s/call, reps clamped to 1, and
every timed sample collapsed to a single call — the d1_r6 short-sample
disease back through a side door, plus ~0.3 s per draw of pure sibling-probe
re-runs inside the 6 s probe deadline.

### What changed (impl/d1_race.c, 217 diff lines vs impl_7)
**1. Sibling one-shot probes absorbed outside every timed window.**  The
pre-gate loops (both stages, cold path only) now make one untimed
execute / chain-at-m_race call per gated arm right after materialization —
their probe fires there, outside gr_race's deadline, and their placement
pick is made at the same m the race then times.  d1r_probe calibrates reps
off the plan's SECOND call (first contact absorbed untimed); the probe-time
warm loop already absorbed re-created draws' probes by construction.
**2. Wisdom GC (the strip protocol, open since d1_r2, finally shipped).**
Once per process, with the current roster names in hand: drop every
d1_race/ key from a dead tag generation, and every gate/ref/verdict whose
key or stored winner names a roster label at a stale source hash.  Keys not
under d1_race/ are never touched (adopters' keys are theirs).  a80n0's file:
131 KB / ~1100 entries (688 stale-hash gate verdicts, ~330 dead-salt) ->
12.9 KB; wallaby dropped 355 entries.  Every gr_wisdom lookup reads the
whole file and every store rewrites it under flock, so this was a per-cell
tax — and unbounded growth would eventually have hit gr__read_all's 16 MB
cap and silently disabled wisdom.
**3. Salts** exe.r7->exe.r8, chn.r6->chn.r7, refs pref/cref/crefL .r3->.r4
(the calibration fix changes the sample sizing wherever a sibling probe
inflated it, so old refs are wrong-statistic bars), now single-site
#defines (D1R_TAG_*/D1R_REF_*) with the GC's live-list built from them.
Op count: unchanged — this entry ships whatever the fastest gated sibling
ships.

### Measured ON THE NODE (a80n0, leased core 3 — first time in this entry's
### history; the 440424 hold was alive; the stock squeue shim still lies, so
### /tmp/d1race_shim_r8 per prime's r3 recipe)
| cell | node r8 | note |
|---|---|---|
| 13 B=1 | 0.014-0.015 us, sd 0-3% | r6 board best 0.0163; warm setup 8-9 ms |
| 13 B=1 chain m=200k | 0.034 | bitwise repeatable x2 procs |
| 16384 B=64 | 37.5, sd 0.03% | r6 board best 44.2 (pow2); ships the r7 siblings' gains |
| 10007 B=1 | 108.9, sd 0.07% | r6 shipped median was ~138 over a 111.8 ref |
| 100003 B=8 | 2388, sd 0.34% | setup 4.3 s cold |
| 100003 B=8 chain m=15 | 2424 | planner native chain raced fresh (5.7% over loop:bluestein), fair arbitration keep, setup 7.8 s cold, repeatable |
First-ever cold create on the node (compile all 8 sibling .so's + races +
GC) was 31 s at 13 B=1 — inside the 60 s cold budget; warm setups 8 ms-7.8 s
depending on cell cost.  Wallaby checks: 60 B=8 0.027-0.028 with a live
exec-TIE arbitration (batchlane kept over planner, probed medians 1.7%
apart, under the 5% hysteresis); 32 B=512 chain min 0.022 = r7's number,
two-process bitwise repeatable, D1_RACE_NO_PROBE output bitwise identical.
All cells PASS check.py (rel L2 1.4e-16 … 1.04e-15, tol 1e-12); chained
cells PASS the map-chain gate (worst 2.6e-12 vs 1e-10 at 32:512:1000).

### Live-fire notes / what did not work
- Sibling churn re-keyed the 13 B=1 chain sig mid-session (two sigs, each
  with its own /r2 + /xc, each arbitrated independently, outputs bitwise
  identical across the flip boundary) — seventh live-fire round for
  hash-carrying names, zero stale verdicts.
- Hit batchlane's r7 ssh-cwd trap exactly once (relative path over ssh lands
  in $HOME); their fix applies — cd explicitly in the ssh command.
- tryout.sh's chain detection is still broken for multi-row cases.txt
  (mangles L/B/m vars); chains were driven manually over ssh, as every
  sibling has done since r4.
- r7's next-round item 1 (read the r7 board against the r7 wisdom) was
  unactionable: no r7 board exists.  The r8 board is the first scored test
  of BOTH rounds' mechanisms — read it with that in mind.

### Borrowed, explicitly
- The absorption diagnosis exists because prime/rader/batchlane/twiddle's r7
  records document their probes' shape and cost precisely (their adoption of
  my probe came back as my round's main hazard — cumulative rounds cut both
  ways); the ssh-cwd hygiene is batchlane r7; the build-your-reference-from-
  the-impl-symlink trap (rader/twiddle r7) was read BEFORE building anything
  — my A/Bs this round compare against persisted refs and board numbers, not
  an impl_7 rebuild.

### Next round
1. Read the r8 board + BOTH node wisdom files (a80n0/a81n2): (a) did the
   fair arbitration + absorption land the close cells (60 B=1, 16384 B=64,
   10007 B=1, 32/64 B=512) at ~1.0x of the best arm's standalone median?
   (b) count /xc flips and check each against the standalone arm.  The
   chain-only-shard exec-arbitration hole (r7 context) is still only
   fixable by the monitor co-locating a cell's variants on one shard.
2. If sibling first-call probes get heavier still, make the probe's draw
   budget adaptive on measured per-draw cost (currently only the >150 ms
   create brake and the 6 s deadline bound it).
3. Stale build/race1d .so cleanup (dozens of dead-hash .so files per host
   dir) — same shape as the wisdom GC, keyed on current hashes; low stakes
   but the dirs grow every churn.
4. The steady .so-vs-binary chain gap (r7 item 2) may have dissolved with
   planner/batchlane's r7 chain rewrites — re-check on the r8 board before
   proposing the static-link A/B to the monitor.
