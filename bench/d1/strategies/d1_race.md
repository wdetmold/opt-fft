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
