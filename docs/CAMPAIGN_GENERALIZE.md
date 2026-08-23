# The GENERALIZE campaign: an arbitrary-L 3D FFT library

Status: DRAFTED, not armed. Harness at bench/gen/ (brief, cases, config). To launch,
follow "Arming" below.

## Goal

`fft3d_create(L, batch)` for any 2 <= L <= 128, any B >= 1, double-complex 3D c2c,
x86-64 AVX-512, beating the best of {MKL 2022, MKL 2026, FFTW3-best-planner, ducc0} at
every acceptance point under the two-part correctness gate, with plan-time <= 60 s cold
/ <= 50 ms with persisted per-host wisdom.

Expectation from the fixed-size data: prime-heavy sizes 4-10x over best library,
smooth/large sizes 1.5-3x. Envelope is Intel AVX-512 (CLX/ICX/SPR measured); AMD is
untested-but-compilable, ARM/AVX2 are out of scope (ports, not tunes).

## Why this is now a bounded build

Every winning attempt of the fixed-size era (ours and all 28 rival reconstructions)
converged on generator-emitted C + plan-time racing. The assets in-repo: our per-class
winning kernels for every factorization class except general prime-power CT; three
complete rival generator pipelines; fft3d_best's create()-race and per-host builds; the
machine models for three Intel generations; the calibrated gate machinery. The one
genuinely new component is the general mixed-radix twiddle framework (prime-power axes),
which is why it has a dedicated owner and is called out as the campaign's center of
gravity in the brief.

## Acceptance suite (bench/gen/cases.txt)

11 sizes never tuned by any prior round, spanning the factorization classes:
10, 12, 15, 20 (PFA small) | 25, 27, 32, 50, 100 (prime-power/2^k CT) | 31
(dense-vs-Rader crossover) | 40 (PFA large) | 100 also = the memory-pressure case
(32 MB working set vs 54 MB L3). Chain lengths target ~0.4 s of MKL per case;
the round-1 monitor calibrates m once on the scoring node, then the suite freezes.

## Rounds (6, cron-owned, same runner)

- r1: scaffolding round. Planner + race + layout + twiddle layers v0; every class entry
  gets its acceptance sizes RUNNING and gate-passing (speed secondary). Monitor
  calibrates and freezes m.
- r2-r3: class deepening. Dense-vs-Rader crossover measured (p = 29, 31 both ways);
  prime-power CT framework hardened; batch-lane engine generalized. Cross-arch check
  after r2.
- r4: accuracy + portability round. Twiddle audit at 1.5e-14/step on all suite sizes;
  race must select winning variants on CLX and SPR too (advisory xarch after r4).
- r5: integration. fft3d_general trunk assembled (planner + race + class winners);
  on-demand generation for unlisted L proven end-to-end; wisdom cache semantics frozen.
- r6: surprise round. Monitor draws three sizes from 14..127 (never announced anywhere),
  scores the TRUNK only. Plan-budget violations or gate misses score zero. This is the
  acceptance test of the whole campaign.

## Roster

12 named entries (8 class owners + 4 library layers) — see the brief's table. Library
layers are scored by adoption. Implementers on claude-fable-5 via wallaby
(claude_remote.sh); monitor on claude-opus-5; scoring on a reserved Ice Lake node
(axxxl, reservation trick, slot leases, monitor-only scoring windows).

## Arming (when ready — NOT done yet)

1. Copy driver.c, check.py, gen_input.py, sweep.sh, leaderboard.py, slot_lease.sh,
   tryout.sh, submit.sh, reserve.sh, Makefile from bench/ice/ (they are L-generic);
   point sweep.sh at bench/gen/cases.txt.
2. Seed bench/gen/impl_0/ with: fft3d_best/ (race skeleton), the class seed kernels
   named in the brief's table, and stub entries per roster name.
3. reserve.sh on axxxl; then the cron line, mirroring ice:
   CLAUDE=$G/gen/claude_remote.sh flock -n /tmp/fft_gen.lock \
     $G/geom/run_rounds.sh --harness $G/gen --resume
4. State file: echo "1 6" > bench/gen/results/.rounds_state (after a --dry-run).

## Open decisions for Will

- L upper bound 128 vs 256 (256 triples the pow2/memory work; the mgpu campaign may be
  the better home for L >= 128).
- Whether B sweeps (1..2048) join the acceptance scoring or stay a per-round xarch-style
  advisory (the batch-selection lesson says: at least advisory from r2).
- AMD validation round (a Zen node would make the "arbitrary machine" claim honest for
  x86; without one we ship "Intel-validated, AMD-untested").
