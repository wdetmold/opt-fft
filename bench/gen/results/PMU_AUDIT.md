# After-the-fact PMU audit (a80n0, /tmp/perf, 2026-08-25 evening)

Counter runs: 6 samples + warmup of the graded chain per case, default event set +
l2_lines_in.all + longest_lat_cache.miss. Raw output preserved in the session log.

## 1. The "L=25 r8 regression" is NOT a code regression — it is plan-time pick instability
r7's gen_powp rebuilt from impl_7 and r8's binary produce IDENTICAL counters today
(cycles 6.164G vs 6.171G, every port within 0.1%, both ~0.168 s/chain) — yet r7's SCORED
number was 0.1265. Both source trees contain the fast path; the entry's INTERNAL plan-time
tuner picked it during r7's scoring window and picked the slow variant in r8's window and
in both of today's rebuilds. The engines' internal tuners are per-run nondeterministic and
NOT persisted through gen_race's wisdom layer (the wisdom file holds only 3 entries).
**Library fix, high value: route every engine-internal pick through the wisdom cache with
noise-gated storage (store only picks whose trial spread is tight), so a lucky/unlucky
create() can never cost 25% again.**

## 2. The weak large-L cells are traffic-bound, now with numbers
- L=100 (1.74x): l1d.replacement 2.34G lines = ~150 GB of L1 traffic across the run, ~4x
  the algorithmic minimum; 65 GB into L2; 16 GB from DRAM. Combined 512-bit-FMA-capable
  dispatch (p0+p5) only 0.82/cycle of the 2.0 ceiling, IPC 1.26. Pure traffic problem.
- L=50 (2.30x): fits L3 (LLC misses negligible) but 77 GB into L2; p0+p5 = 1.07/cycle.
  L2-traffic-bound; also B=4 excludes 8-lane SoA (a 4-lane variant is untested).
**The Tier-2 two-axes-per-pass fusion play is confirmed as the right investment: cutting
pass count directly attacks the dominant counter at both cells.**

## 3. The champion signature (what "done" looks like)
gen_rader at L=31: IPC 2.15, p0+p5 dispatch = 1.60/cycle = 80% of the dual-512-bit-FMA
ceiling, traffic minimal. Cells below ~1.1/cycle dispatch with high l1d.replacement have
schedule/traffic headroom; cells near 1.6 are close to done on this microarchitecture.

## 4. Port 1 idles everywhere (0.7-2.0G vs port 0's 9.6-10.3G)
On Ice Lake-SP, 512-bit FP occupies ports 0+5; port 1 serves scalar/256-bit FP. Every
kernel leaves it idle. Any genuinely independent 256-bit side-work (map tails, twiddle
prep, checksum accumulation) could co-issue there at near-zero cost — a micro-opportunity
nobody has touched.

## Verdict on the tooling
Two r8 implementers used the counters in-round (gen_powp's traffic ledger "155k lines/
group-step = 9.9 MB = state x 3 passes + c"; gen_rader's PMU-delta schedule comparisons;
license-downclock cleared at 2.88 GHz). Retrospectively, one evening of counter runs
re-diagnosed a 25% mystery regression, quantified the remaining headroom per cell, and
produced the target signature for future rounds. The PMU should be armed from round 1 in
any future campaign.
