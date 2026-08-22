# L23_rader — multicore strategy record

Phase-1 history (how the serial kernel got its shape, and the closed list of
streaming nulls that must not be rebuilt) lives in
`../../geom/strategies/L23_rader.md`. This file starts at the multicore phase.
Phase-1 baselines used below: node (CLX, panel_r11 scored) B=1 47.469 µs,
B=4 49.678, B=128 64.793 µs/vol; wallaby (SPR, driver-level) 21.35 / 23.34 /
24.3 µs per volume.

## Round mt_r1

### What changed

The arithmetic is untouched: same conjugate-folded Rader pair (297 vector FP
ops per line-group, 409 zmm chunks per volume, 943 kflop), same two-sweep
pinned-constant kernel, same X-first pass order, same 1064-double t1 plane
stride, sine-table-folded -i rotation. Every volume's DAG is identical no
matter which thread runs it, so output is bit-identical to the serial exec at
every team size and for every tuner cell (cmp-VERIFIED, see below). What was
added is a threading layer, and what was *deleted* is every phase-1 exec
variant the verdicts had already declared a null (rp-t1, deferred-Z, both
pipelining schedules, pf=1, krn_il, P-parking, X-last) — the MT file races
modes, not corpses.

1. **Range execs** `(plan, in, out, b0, b1, slot)` over per-thread page-sized
   scratch slots (t1 24472 + pb 1104 + NT staging plane 1064 doubles, padded
   to 53 pages = 217 KiB), allocated once and **first-touched by the owning
   thread** in a full-width parallel region in `fft3d_create()` — NUMA-local
   scratch, no false sharing by page construction, and the region spins up
   the OpenMP pool so the first timed execute creates no threads.
2. **mode=batch**: team of min(batch, T) threads, thread t owns the contiguous
   volume block [nb·t/T, nb·(t+1)/T), ranges computed from the team OpenMP
   actually delivers; zero synchronisation except the join. pf=2 (in-pass X
   prefetch) and pw=1 (write-intent out-plane prefetch) kept as per-thread
   knobs.
3. **mode=batchNT**: same, but each finished plane is staged and streamed to
   `out` with NT stores. This was a four-round phase-1 null; it was raced
   again on the argument that with 32 cores sharing DRAM, the RFO third of
   the out traffic is a bigger slice of a scarcer resource. **The null
   flipped**: on wallaby at nv=403 (true streaming), batchNT pf2 = 1.74 µs/t
   vs plain-batch best 2.00 — the round's one transferable finding (other
   entries whose NT variants lost single-threaded should re-race them).
4. **mode=fused** (batch ≤ nslots): omp-for over all batch·133 X chunks
   (chunk i of volume v writes whole distinct 64-B lines of slot v's t1; the
   overlapping tail chunk rewrites bit-identical values, benign cross-thread),
   implicit barrier — the one true global dependency, since every X chunk
   writes one (y,z) column of *all 23* kx-planes — then omp-for `nowait` over
   all batch·23 (v,x) planes, each Y→Z on the executing thread's own pb.
   At B=1 this is the intra-volume split.
5. **Deterministic joint-cell hysteretic tuner** (phase-1 r9 mechanism, cells
   extended to (mode, width, team, pf, pw)): one canonical list per regime,
   two fixed-order sweeps, per-cell min, >2% to displace, licence-honest
   1.5 ms warmups. Heads by the fastest-known-head rule: B=1 serial (the
   node-measured phase-1 pick), 1<B<32 one-volume-per-thread, B≥32 full-team
   batch pf2 pw1. team=16 cells sit in the B≥32 list as the node's
   two-socket question (driver first-touches in/out single-threaded → one
   socket owns them; 16 close-bound threads = one CLX socket, all-local).
   Small batches tune at the TRUE batch (the decomposition is a function of
   it), streaming batches on the L3-relative arena. Since all cells are one
   bit class, no pick can change the output — the pick lottery is
   structurally harmless here, unlike phase-1 r8.
6. Env knobs: `L23R_FORCE` (cell index), `L23R_TEAM`, `L23R_PF`, `L23R_PW`,
   `L23R_VERBOSE`. Tuner telemetry (pick + list-head µs/t, arena size) baked
   into the description string as before.

### Operation count

Per volume unchanged from panel_r11: 3·529 line-groups × 297 vector FP ops =
943 kflop, 409 zmm chunks (flat layout only now). The MT layer adds zero
arithmetic; fused adds two team barriers per call, batch adds one fork/join.

### What was measured on wallaby (Gold 6448Y SPR, 32 threads close/cores, driver-level, shared login node)

| case | phase-1 serial (same machine) | mt_r1 | per-vol | speedup | parallel efficiency |
|---|---|---|---|---|---|
| B=1 | 21.35 µs | **7.65–8.1 µs** (fused team=23) | 7.65 µs | 2.8× | 12% on 23 threads |
| B=3 | ~22.8 µs/vol | 16.5 µs/call | 5.5 µs | 4.1× | — |
| B=8 | ~23.3 µs/vol | 24.2 µs/call | 3.02 µs | 7.7× | 97% on 8 threads |
| B=16 | ~23.3 µs/vol | 47.6 µs/call (batch-16 pf2 pw1) | 2.98 µs | 7.8× | 49% on 16 threads |
| B=32 | ~24.3 µs/vol | 45.0 µs/call | 1.41 µs | 17× | 54% on 32 |
| B=33 | — | 68.4 µs/call (2-vols/thread boundary, PASS) | 2.07 µs | — | — |
| B=128 | 24.3 µs/vol | **140.8 µs/call** (batchNT) | **1.10 µs** | **22×** | **69% on 32** |
| B=512 | 24.3 µs/vol | 892 µs/call (batchNT) | 1.74 µs | 14× | 44% |

Correctness: PASS rel_l2 = 3.77e-16 … 3.81e-16 at B = 1, 3, 4, 8, 16, 32,
33, 128, 512; repeatable (bit-identical across runs) everywhere.
Bit-class verification: forced cells batch / batch+pf2pw1 / fused-32 /
fused-16 / serial / batch-w2 all cmp-identical on full B=3 outputs.

Key tuner tables (one quiet-ish run each):
* B=1 (nv=1): serial 32.07; fused t32 9.83, **t23 9.07 ← kept**, t16 10.48,
  t8 13.46, t4 23.24. The 23-plane phase sets the natural team.
* B=16 (nv=16): batch-16 pf0 3.35, **pf2 pw1 2.96 ← kept**; fused t16 3.41,
  t32 6.15 (cross-thread t1 handoff: X writers ≠ plane readers, so fused
  loses to thread-private t1 once every thread has a whole volume).
* B=512 (nv=403): batch t32 pf2pw1 2.07, pf0pw1 2.00; **batchNT pf2 1.74 ←
  kept** (−13% vs plain best); batch t16 3.49 (wallaby is one socket at 32
  threads, so t16 just halves the compute — the node must re-ask this);
  serial 32.05.

Where the missing cores went, honestly: at B=128 the call moves ≥128 × 0.95
MB (in + out + RFO-or-NT + 2×t1) in 141 µs ≈ 550 GB/s aggregate including
cache reuse — bandwidth-shared territory, and exactly where NT's RFO savings
pay. At B=1 the 7.7 µs decomposes as ~1.4 µs of longest-thread work plus
~6 µs of GOMP fork + one barrier + join on a *shared* login node (L13_direct
measured the same 4–5 µs GOMP region cost); the fused split still buys 2.8×
because L=23's 21 µs of work dwarfs it — the opposite balance from
L13_direct's B=1 (2.5 µs of work, split lost). Both results support the same
rule: intra-volume splitting pays iff work ≫ region cost.

### What did NOT work / observations with numbers

1. **fused at team=32 loses to team=23 at B=1** (9.83 vs 9.07): the plane
   phase has exactly 23 tasks, so threads 24–31 only help the X phase and
   then idle into the barrier; the wider fork costs more than the X-phase
   help. Kept team=23 in the canonical list ahead of 32? No — list order is
   head-first only; both are raced, hysteresis decides per machine.
2. **fused at mid batch** (B=16): 6.15 µs/t at t32 vs 3.35 for plain
   batch-16 — volume v's t1 is written by many threads and read by others
   (dirty-line migration every plane), where batch mode keeps t1 L2-private.
   Fused is only for batch < team.
3. **The second fused barrier was pure waste**: `nowait` on the plane loop
   (join is the sync) took B=1 from 8.68 to 7.69 µs, −11%. One barrier ≈
   1 µs at team 23 on this machine.
4. **The raw-ssh trap** (documented five times in the phase-1 record, and I
   did it AGAIN, twice): one-off remote commands must be
   `ssh wallaby 'source ~/fft/env.sh; cd ~/fft/bench/mt; …'` with absolute
   paths, `test -s` before any cmp, `&&`-chained.

### Borrowed this round (attributions)

**L13_direct mt_r1** (the only other MT record when this round started): the
whole threading-layer design — range execs with per-thread slots first-touched
by their owner inside a create()-time full-width region, ranges computed from
the delivered team, contiguous volume blocks, one-volume-per-thread as the
mid-batch head, and the "B=1 splits iff work ≫ region cost" framing that
predicted my fused win. From my own phase-1 lineage: krn_ts, pf=2/pw=1,
NT-copy machinery, the joint-cell hysteretic tuner, fastest-known-head rule,
licence-honest warmups, L3-relative arena sizing.

### Node prediction (pre-registered)

Picks: B=1 fused (team 23 or 32 — CLX has 2×16 cores, so team 16 may also
surface: it is all-one-socket there); B=4 fused t32; B=128 **batchNT** if the
node's DRAM contention resembles wallaby's, else batch pf2 pw1. Levels are
guesses across a µarch gap: B=1 8–15 µs (fork cost on the idle pinned node
should be *below* wallaby's), B=128 1.7–2.6 µs/vol (node phase-1 streaming
was 2.7× wallaby's per-vol time). The t16-vs-t32 field in the B≥32 tuner
telemetry is the round's NUMA instrument — read it before believing any
UPI story.

### Next

1. **Persistent spin-pool** (pthreads pinned to the harness's cores,
   sense-reversing barrier, created in fft3d_create) — the B=1 cell is
   ~80% synchronisation; cutting fork+barrier+join from ~6 µs to ~1 µs is
   worth ~3× there and helps every batch < 8. L13_direct's next-round item
   too; whoever builds it first, the other should copy. Only build it on
   node confirmation that the fork cost survives on idle pinned hardware.
2. If the node's B≥32 telemetry shows t16 ≥ t32 (UPI-bound): don't cap the
   team — make the far socket's threads stage their volumes' `in` through
   local scratch (one extra 190 KiB local read converts every remote load
   into a single remote stream), and consider NT for the far socket only.
3. Race a fused variant whose X phase assigns chunks so that the thread that
   writes t1 columns of plane x is the one that later reads plane x — a
   chunk→plane affinity map could cut the mid-batch dirty-line handoff and
   make fused competitive at 2 vols/thread (the B=33-boundary 2.07 µs/vol
   is the target).

## Round mt_r2

### What changed

The arithmetic is again untouched (same conjugate-folded Rader pair, 297
vector FP ops per line-group, 943 kflop/volume); output is bit-identical to
mt_r1 in every mode (cmp-VERIFIED below). What changed is the dispatch:

1. **Persistent spin pool replaces every execute-time OpenMP region** —
   adopted whole from **L23_matrixsimd mt_r1**, which measured the thing my
   own r1 record only inferred: one GOMP fork+barrier+join costs 6.2–8.2 µs
   at T=8..32 on wallaby, and a central fetch-add barrier ~2.5–5 µs.
   `fft3d_create()` reads the harness's thread→CPU map from one throwaway
   OMP region (`sched_getcpu()` per thread, so the pool pins exactly where
   `OMP_PROC_BIND=close / OMP_PLACES=cores` would put threads, with their
   duplicate-CPU check for unbound runs), spawns 31 pinned pthreads once,
   and `execute()` publishes a job and release-stores one generation word.
   Barriers are their flag-array design: per-thread padded arrival flags,
   tid 0 always collects and broadcasts one release word; epochs 2·gen
   (mid) / 2·gen+1 (join) tolerate a varying team, and the join needs no
   release. Workers never sleep. OpenMP never runs after create().
2. **Fused mode got a dedicated t1 arena** (batch < 32; batch·24472
   doubles), first-touched by a pool job with the SAME static plane
   partition the plane loop uses — reader-owned pages, also from
   L23_matrixsimd. mt_r1 kept fused t1 in slot v (all of volume v's pages
   on thread v's node); that plus the OMP fork is why fused lost the whole
   mid-batch range last round. The X-pass overlap tail chunk is now folded
   into the last independent item (132 items, the last runs 2 chunks) so no
   t1 line has two writers across threads.
3. **Modes and tuner mechanism unchanged** (serial / batch / batchNT /
   fused; joint-cell hysteretic tuner, two sweeps, per-cell min, >2%
   displacement, 1.5 ms licence warmups; batch and batchNT share one range
   exec via an `ntc` plan flag). Heads updated to measurement: B≥32 head is
   now **batch pf0 pw0 — the NODE's mt_r1 pick** (its tuner: 3.40 vs 5.51
   µs/t for the old pf2 pw1 head; both prefetch knobs LOSE at 32-thread
   streaming there, all four combos stay raced, batchNT stays raced as
   wallaby's winner); B=1 and 2≤B<32 heads are fused-on-pool.
4. Env knobs unchanged (`L23R_FORCE/TEAM/PF/PW/VERBOSE`); description
   string now says `pool` and carries the same pick/inc telemetry.

### Operation count

Per volume unchanged from panel_r11: 3·529 line-groups × 297 vector FP ops
= 943 kflop, 409 zmm chunks. Parallel overhead per execute: one release
store + T arrival stores per barrier; fused = mid + join, batch = join
only; zero extra FP work, zero OpenMP.

### Measured on wallaby (Gold 6448Y SPR, 32 threads close/cores, driver-level, shared login node)

| case | mt_r1 | mt_r2 | per-vol | vs phase-1 ST (21.35–24.3) | parallel eff |
|---|---|---|---|---|---|
| B=1   | 7.65 µs | **4.93–5.36 µs** (fused t32) | 4.9–5.4 | 4.2× | 13% on 32 |
| B=3   | 16.5 µs/call | 11.22 µs/call | 3.74 | 6.1× | — |
| B=8   | 24.2 µs/call | 19.95 µs/call | **2.49** | 9.4× | 29% |
| B=16  | 47.6 µs/call | 34.07 µs/call | **2.13** | 11× | 34% |
| B=32  | 45.0 µs/call | 40.95 µs/call | **1.28** | 19× | 59% |
| B=33  | 68.4 µs/call | 64.7 µs/call | 1.96 | — | — |
| B=128 | 140.8 µs/call | 163.3 µs/call this window; in-tuner pick 1.27–1.28 µs/t | ~1.28 | 19× | 59% |
| B=512 | 892 µs/call | 907 µs/call; tuner batchNT pf2 **1.66** µs/t (r1: 1.74) | 1.77 | 14× | 43% |

(B=128/512 windows are not comparable across days on this shared node; the
in-tuner tables are the like-for-like numbers.)

Correctness: PASS rel_l2 = 3.767e-16 … 3.808e-16 at B = 1, 3, 8, 16, 32,
33, 128, 512; repeatable (bit-identical across runs) everywhere.
Bit-class: all 7 forced cells cmp-identical on full outputs at B=3 AND
B=128 (serial / batch / batchNT / fused / team variants / w2).

Key tuner tables (one run each):
* B=1 (nv=1): fused t32 **5.08 ← kept**, t23 5.67, t16 8.20, t8 10.60,
  w2-fused 5.80, serial 32.1. The OMP-era t23>t32 inversion is gone — the
  flag barrier grows so slowly with T that the 9 extra X-phase threads win.
* B=8: fused t32 **4.22 ← kept**, batch-8 5.02, batch-8 pf2pw1 5.11,
  batchNT-8 7.35. Fused now beats one-volume-per-thread at EVERY 2≤B<32
  (B=16: 2.41 vs 2.66) — mt_r1's opposite verdict (6.15 vs 3.35 at B=16)
  was the fork+slot-t1 tax, not the decomposition.
* B=512 (nv=403): batchNT pf2 **1.66 ← kept**, batchNT pf0 1.75, batch
  pf2pw1 1.88, pf0pw1 1.98, pf0pw0 2.71, t16 3.32–5.11. Wallaby still
  loves NT and pf2 at true streaming; the node picked the exact opposite
  (pf0 pw0 plain) in r1 — that is why the head is the node's pick and
  everything else is raced.
* B=128 (nv=128, fits wallaby's 60 MiB L3): plain batch 1.27–1.28 for all
  four knob combos, batchNT 1.59 — the L3-resident RFO inversion
  L23_matrixsimd documented, reproduced.

### What did NOT work / observations with numbers

1. **Raw-ssh trap, sixth appearance, now with a twist**: three identical
   failed attempts because the remote command lacked `cd
   ~/fft/bench/mt` before `python3 gen_input.py`. Fix that actually held:
   put the dev loop in a script on the shared filesystem and
   `ssh wallaby /abs/path/script.sh` — no quoting, no cwd surprises.
2. No performance regressions found this round: every scored regime
   improved or matched on wallaby. The B=128 window number (163 µs) is
   higher than r1's 140.8 but the in-tuner per-transform (1.27 vs r1's
   ~1.10 µs/t implied) is window noise on a shared node — flagged, not
   explained away: the node will arbitrate.

### Borrowed this round (attributions)

**L23_matrixsimd mt_r1** (my direct rival at this geometry), wholesale:
the persistent pinned spin pool, the flag-array mid/join barriers with
generation-derived epochs, the sched_getcpu OMP-map pinning with the
unbound-run check, the reader-partitioned fused t1 arena first-touch, and
the folded overlap-tail X item. Their record's GOMP fork (6.2–8.2 µs) and
fetch-add barrier (~2.5–5 µs) measurements saved me a round of
rediscovery, exactly as the brief intends. My r1 record had the spin pool
as next-round item 1 and said "whoever builds it first, the other should
copy" — they built it first; copied, and it composes with my modes: my
fused-on-pool mid-batch numbers (2.49 at B=8, 2.13 at B=16) now beat their
r1 wallaby marks (2.77 at B=8), and batchNT-on-pool keeps my streaming
edge (1.66 vs their 1.51 is theirs at nv=416; at the node's B=2048 I won
r1 6.05 vs 7.24).

### Node prediction (pre-registered)

* B=1: fused t32 or t23; expect 6–10 µs (r1: 14.15; matrixsimd's pool got
  11.84 — same sync design, same work, so parity ± the t1-arena
  difference). If the node's cross-socket barrier is the cost, t16 shows
  up; that cell is the instrument.
* B=128: batch or batchNT near 2.0–2.3 µs/vol (r1: 2.35 with the fork
  tax); the node's 22 MiB L3 makes B=128 true streaming, so the pick
  should match its B=2048 pick, not wallaby's L3-resident plain-batch.
* B=2048: pick stays batch pf0 pw0 (the r1 node head), 5.5–6.0 µs/vol —
  the pool only removes one fork per call here.

### Next

1. **Two-level (socket-tree) barrier** if the node's B=1 pick lands at
   t16: tid 16 collects socket-1 flags, tid 0 collects socket-0 + tid 16
   (L23_matrixsimd's next-round item 2; whoever builds it first...).
2. **Far-socket staging for streaming on the node**: if B=2048 telemetry
   shows t16 ≈ t32, socket-1 threads should stream `in` through local
   scratch and/or NT-store only their half — untestable on wallaby (32
   close threads = one socket there), so build it only against node
   telemetry.
3. Wire pf into the fused X items (each thread's 4–5 chunks at B=1 read
   cold strided lines; worth ~0.2–0.4 µs if it lands) — unmeasured this
   round, shared-node noise would have swallowed it.

## Round mt_r3

### What changed

The arithmetic is untouched for the third round running (same conjugate-
folded Rader pair, 297 vector FP ops per line-group, 943 kflop/volume);
every new cell is ownership-or-dispatch only and the bit class is intact
(cmp-VERIFIED below). Four changes, all aimed at what the mt_r2 node data
said was wrong:

1. **Aggregate-cache tuner arena** — the round's must-fix. mt_r2's node
   loss at B=2048 (7.167 vs L23_matrixsimd's 5.934) was not the kernel:
   my streaming arena sized 2.5x ONE L3 gave nv=148 = 58 MiB, inside the
   node's **76 MiB aggregate** (32x1 MiB L2 + 2x22 MiB L3), so the tuner
   priced out-RFOs at L3 speed and picked plain batch over NT (its table:
   2.59 plain vs the NT cell it never believed; driver reality 7.17).
   `l23r_tune_nv` now sizes off `nthr*L2 + nsock*L3` x3.5, cap 640
   volumes (node: nv=640 = 243 MiB; wallaby: same cap). BORROWED whole
   from L23_matrixsimd mt_r2, who borrowed the lesson from L6_pfa mt_r1.
2. **Two-level socket-tree barrier**, raced per-cell (`bar=1` in the cell
   tuple): the far package's lowest tid collects its socket's arrival
   flags locally and posts ONE flag that tid 0 reads across UPI; the
   release relays through a second word (`rel2`) so far threads never
   spin on tid 0's release line. One remote flag read + one remote
   release read per episode instead of 16 of each. Package map from
   sysfs `physical_package_id` of the pinned CPUs (BORROWED:
   L23_matrixsimd mt_r2's socket map); nsock=1 degenerates to the flat
   mt_r2 barrier, bit-for-bit. This was both L23 entries' listed-but-
   unbuilt next item ("whoever builds it first, the other should copy")
   — built here first; matrixsimd should take it back next round.
   Target: the node's B=1, where BOTH entries' scored pick is fused
   T=16 at ~11.6-11.9 us because the flat 32-wide scan crosses UPI.
3. **Weighted near/far volume split** for the range exec (`wt=1` 4:3,
   `wt=2` 5:3, near:far), raced at B>=32, two-socket only: the driver
   first-touches in/out on thread 0, so far-socket threads pay UPI in
   both directions per volume; giving socket-0 threads more volumes
   equalizes finish times iff UPI (not socket-0 DRAM) is the binding
   constraint. Evolves my r1 next-item 2; related to matrixsimd's r1
   next-item 4 (NUMA-asymmetric split). Ownership-only, bit class safe.
4. **pf=2 wired into the fused X items** (my r2 next-item 3): prefetch
   the NEXT item's 23 lines, one per 8464-B-strided x-plane. Raced at
   B<32.

Cell list housekeeping: streaming head stays the node's plain pf0 pw0
(its B=128 winner); dropped the two never-winning mixed plain-prefetch
combos, team=24, and the streaming serial telemetry row (at nv=640 it
was ~0.25 s of setup for a number phase 1 already gives us). B=1 head is
now fused T=16 flat — the node's actual scored pick — so the tree cells
must displace an honest incumbent by >2%. New env knobs: `L23R_BAR`,
`L23R_WT`, and `L23R_FAKESOCK` (DEV ONLY: pretends the top half of the
team is a second package so wallaby can exercise the tree/weighted code
paths). Description string now carries bar/wt/nsock.

### Operation count

Per volume unchanged from panel_r11: 3*529 line-groups x 297 vector FP
ops = 943 kflop, 409 zmm chunks. Tree barrier: same store count per
episode, one extra relay store by the collector; weighted split adds one
<=32-iteration integer prefix loop per thread per call. Zero FP added.

### Measured on wallaby (Gold 6448Y SPR, 32 threads close/cores, one socket, shared login node; somewhat noisy windows, sd 1.6-4.4%)

| case | mt_r2 | mt_r3 | per-vol |
|---|---|---|---|
| B=1    | 4.93-5.36 us | 5.74-6.24 us (fused T32 flat kept; window sd 3.5-4.4%) | 5.7-6.2 |
| B=8    | 19.95 us/call | 20.5 us/call | 2.57 |
| B=128  | in-tuner 1.27-1.28 | 166.9 us/call, in-tuner plain pf2pw1 1.46 | 1.30 |
| B=512  | 907 us/call, tuner batchNT 1.66 | 852.9 us/call, tuner batchNT 1.49 | 1.67 |
| B=2048 | — | 4425 us/call (sd 0.19%), tuner nv=640 batchNT pf2 1.83 | 2.16 |

Setup at B=2048 with the nv=640 arena: 0.268 s — well within budget.
Parallel efficiency vs wallaby phase-1 ST (21.35/24.3): B=1 3.7x (12% on
32), B=2048 11x (35%). Wallaby cannot show a win for ANY of this round's
changes (one socket: tree and weighted degenerate, the honest arena only
re-confirms the NT pick it already made at nv>=416) — parity here is the
expected result, and the round's test is the node's pick telemetry.

Correctness: PASS rel_l2 = 3.767e-16 ... 3.808e-16 at B = 1, 3, 8, 128,
512, 2048; repeatable (bit-identical across runs) everywhere.
Bit-class: with L23R_FAKESOCK=1 (tree barriers and weighted splits
actually engaged, team split 16/16), ALL forced cells cmp-identical on
full outputs — 8 cells at B=1, 9 at B=3, 10 at B=128.

### What did NOT work / neutral observations with numbers

1. **Tree barrier on one socket is free but useless, as designed**:
   fakesock tree cells at B=1 read 5.39-6.19 across a window whose flat
   cells read 5.73-6.35 — the relay hop costs nothing measurable at
   same-socket latencies. No wallaby verdict is possible; the cell
   exists for the node.
2. **fused-pf at B=1 (wallaby)**: 5.79 (tree window) vs 5.73 flat — a
   wash here, exactly as r2 predicted shared-node noise would make it.
   Raced for the node, where `in` is cold on one socket.
3. The streaming serial row was costing ~0.25 s of setup at nv=640 for
   telemetry phase 1 already provides — dropped, documented here so the
   next round doesn't wonder where it went.
4. Raw-ssh trap, seventh appearance, AVOIDED for once: all remote work
   ran through two scripts on the shared filesystem
   (`~/tmp_l23r3/{dev,cmp}.sh`) invoked as `ssh wallaby /abs/path`. Zero
   quoting failures this round. The scripts pattern from my r2 item 1
   works; keep it.

### Borrowed this round (attributions)

* **Aggregate-cache arena sizing** (nthr*L2 + nsock*L3, x3.5, cap 640):
  L23_matrixsimd mt_r2, who took it from L6_pfa mt_r1 and the mt_r1
  verdict §5. Their raw node JSONs (nv=640, pick nt1 pf0, scored 5.934)
  are the direct evidence it fixes my exact B=2048 failure.
* **sysfs socket map** (physical_package_id per pinned CPU): L23_matrixsimd
  mt_r2.
* **Full-NT-beats-asymmetric-NT**: L23_matrixsimd mt_r2 raced nt=2 (far-
  only) and nt=3 (near-only) on the node and full nt=1 won — I did NOT
  rebuild those, per the brief.
* The tree barrier itself is new code but was jointly specified in both
  L23 records' next-round items since mt_r1.

### Node predictions (pre-registered)

* **B=2048**: all three processes tune at nv=640 (243 MiB, honestly past
  the 76 MiB aggregate) and pick batchNT (pf0 or pf2) — expect
  **5.7-6.1 us/vol**, closing the 7.17-vs-5.93 gap to ~parity with
  matrixsimd. If a weighted cell (wt=1/wt=2) displaces plain-team NT,
  UPI was the binding constraint and 5.4-5.8 is possible; if weighted
  loses, socket-0 DRAM is the wall and next round goes to input staging.
* **B=128**: plain batch holds (47.5 MiB < 76 MiB aggregate keeps RFOs
  cache-priced); 2.2-2.3 us/vol, pick pf0pw0 or pf2pw1, possibly
  plain-wt1 by a few percent.
* **B=1**: the walk asks one clean question: tree-T32 (and tree-T23) vs
  the T=16-flat incumbent. If the flat cross-socket scan+release was the
  cost, tree-T32 lands **10.3-11.3 us**; if the cost is fused-t1 data
  movement (half the reader-partitioned pages sit on socket 1 while X
  writers span both), the tree does not move it and T=16 stands at
  ~11.6-11.9. Either outcome settles the mechanism; read bar/ns in the
  description string.

### Next

1. If tree-T32 wins B=1: relay the DISPATCH too (far threads currently
   acquire-spin on the gen word tid 0 writes — same remote line pattern
   the barrier just fixed); a per-socket gen relay is ~20 lines.
2. If weighted wins B=2048: tune the ratio (race 7:5 and 2:1); if it
   loses, build far-socket input staging (stream far volumes' `in`
   through local scratch — converts 23-line strided UPI reads per plane
   into one sequential pull; matrixsimd r2 next-item 1, still unbuilt).
3. If B=1 is latency-closed at T=16 either way, write the measurement
   down and stop spending rounds on it: the cell is ~1.4 us of work
   behind a ~10 us sync+imbalance floor that two independent kernels
   now measure identically.

## Round mt_r4

### Where mt_r3 landed on the node (read off results/mt_r3 + VERDICT)

Scored: B=1 **11.865 us — won** (a 0.8% tie over matrixsimd's 11.956; pick
fused T=16 flat, tuner 11.51-11.91); B=128 **2.300 — lost by 1.4%**
(another tie; pick plain batch T=32 pf0pw0 in 2 of 3 processes, pf2pw1 in
one, tuner 2.23-2.38); B=2048 **5.810 — won** (a 1.4% tie; pick batchNT
static T=32 pf0, tuner pick 5.48 vs plain inc 6.44 at nv=640 — the
borrowed aggregate arena fixed mt_r2's 7.17 exactly as pre-registered,
+1.23x).  My three r3 bets all resolved NEGATIVE on the node, per my own
pre-registered readings: **bar=0 and wt=0 in every one of my nine node
description strings** — the tree barrier displaced nothing (so B=1's cost
is fused-t1 data movement / imbalance, not the flat barrier scan, and the
cell is latency-closed: two kernels and a purpose-built tree barrier now
measure the same ~11.6 floor), and the weighted split lost (socket-0 DRAM
or the schedule, not UPI equalization, is the streaming wall).  The
verdict's L=23 order: both entries sit at **65-67 GB/s, the LOWEST
streaming bandwidth of any geometry**, while the L=36 entries reach
137-151 GB/s on the same node, and what separates them is the
sequential-read discipline (verdict 4.3/6); "stop building placement
instruments" (team width and page migration are dead, four independent
refutations).

### What changed (memory schedule + idle-thread hygiene; DAG untouched)

1. **Staged sequential input (`si` = 0/1/2), range exec**: with si on, a
   thread copies its volume's `in` (24334 doubles, 190 KiB) sequentially
   into a new page-aligned slot region (vs; slot grows 53 -> 101 pages)
   and runs the X pass off the copy.  Same compulsory bytes, but ONE
   hardware-prefetchable stream per thread instead of the X pass's 23
   interleaved 8464-B-strided page streams (32 threads x 23 = 736 open
   DRAM page streams was the suspected 65 GB/s wall).  si=2 stages
   far-socket threads only.  This was my unbuilt next-item since mt_r1
   ("idea 2"), r3 next-item 2's else-branch; built per the verdict's
   explicit L=23 port order.
2. **Paced next-volume input prefetch (`pv` = 0/1), range exec — my own
   variant with zero extra stores**: during volume v's compute-heavy
   plane phase, prefetcht1 volume v+1's `in` sequentially, ~133 lines
   per plane (3043 total).  Same stream conversion as si, no copy tax;
   the X pass then reads mostly L2 hits.  Distinct from the
   node-rejected pf=2, which issued the same strided pattern
   just-in-time against the X pass's own demand loads; pv issues a
   sequential pattern into idle fill-buffer slots during compute.
3. **Parked non-participants**: once the pick is final (end of create,
   after env overrides), pool threads with tid >= team — which join no
   barrier and run no work — poll the generation word with
   nanosleep(100us) instead of pause-spinning; a catch-up loop tolerates
   any generation backlog, so this cannot cost a participant anything.
   The node's scored B=1 pick (fused T=16) leaves 16 workers spinning
   all run; verdict 4.4 collects three entries blaming busy spinners for
   all-core clock drag (clk512 2.29 vs 2.89 GHz).  `L23R_PARK=0`
   disables; `pk=` in the description string.
4. **Cell-list surgery on node evidence**: dropped the tree-barrier and
   weighted-split cells (bar/wt 0-for-9 in my own r3 node strings; code
   kept, env-forceable), dropped the streaming team=16 rows (verdict 5).
   Streaming list is now: plain / plain-pf2pw1 (the two node B=128
   picks), batchNT static (node B=2048 pick), batchNT-pf2 (wallaby), the
   si/pv ladder (NT+si1, NT+pv1, NT+si1+pv1, plain+si1, plain+pv1,
   NT+si2 two-socket only), batchNT-w2 (NEW: the 256-bit licence-clock
   question on the node's 1-FMA CLX — never raced by either L23 entry),
   plain-w2 fallback.  B=1 list: fused T16 head (node's pick), fused T
   flat (wallaby's), fused T16 pf2, serial, w2 — five cells, per my r3
   item 3: the cell is latency-closed, stop spending on it.
   New env: `L23R_SI` (0/1/2), `L23R_PV`; description string gains
   si/pv/pk.

### Operation count

Per volume unchanged from panel_r11: 3*529 line-groups x 297 vector FP
ops = 943 kflop, 409 zmm chunks.  si adds one bit-exact 24334-double
memcpy per volume when engaged (zero FP, L2-resident destination); pv
adds <= 3043 prefetcht1 per volume; park adds nothing to any participant.

### Measured on wallaby (Gold 6448Y SPR, 32 threads close/cores, ONE socket, shared login node)

| case | mt_r3 | mt_r4 | per-vol |
|---|---|---|---|
| B=1    | 5.74-6.24 us | **5.49-7.21 us** across windows (fused T32 flat kept) | 5.5-7.2 |
| B=8    | 20.5 us/call | 20.7 us/call | 2.59 |
| B=33   | — | 65.0 us/call | 1.97 |
| B=128  | 166.9 us/call | **166.7 us/call** (pick plain w4 static) | 1.30 |
| B=512  | 852.9 us/call | 829.8 us/call | 1.62 |
| B=2048 | 4425 us/call | **4110-4341 us/call** (pick batchNT pf2, as r2/r3) | 2.01-2.12 |

Key tables and A/Bs (all on one socket, where si/pv's mechanism does not
exist — parity-or-loss here was the pre-registered wallaby outcome):
* B=2048 in-tuner (nv=640, us/t): NT-pf2 **1.98 <- kept**, NT 2.12,
  NT+pv 2.15, NT+si+pv 2.31, NT+si 2.38, plain-pf2pw1 2.63, plain 3.01,
  plain+si 3.29, plain+pv 3.22, w2 rows 3.2+.  Driver-level forced:
  NT-si0 4147, **NT-pv1 4375 (+5.5%)**, **NT-si1 4559 (+9.9%)** — si is
  pure copy tax without UPI/DDR4 page thrash, matching L23_matrixsimd's
  independent +8.7% measurement this same round.
* B=128 in-tuner (nv=128, L3-resident here): plain **1.29 <- kept**,
  plain+pv 1.35, plain+si 1.58, NT rows 1.63-1.85.  One flagged driver
  window (same shape matrixsimd flagged): forced plain-si1 175.9/177.1
  min/median (sd 1.2%) vs plain ref 275.9/315.5 (sd 13%) — B=128
  windows on this shared host swing >10%; the walk row exists and the
  node decides.
* B=1 park A/B at forced fused T=16 (the node's shape, 16 idle workers):
  PARK=1 6.868/6.913 vs PARK=0 6.867/6.882 (min/median) — **a wash on
  wallaby**, where SPR clocks per-core.  The node's CLX package
  governor is the actual question; pk= in the description will answer.

Correctness: PASS rel_l2 = 3.767e-16 ... 3.808e-16 at B = 1, 8, 33, 128,
512, 2048; repeatable (bit-identical across runs) everywhere.
Bit-class: forced-cell cmp on full outputs, all IDENTICAL — at B=128
{plain ref, plain+si1, plain+pv1, NT+si1, NT+si1+pv1, w2, and with
L23R_FAKESOCK=1 both NT+si1 (all-stage) and NT+si2 (far-half stages)};
at B=1 {fused T32, fused T16 pk1, fused T16 pk0}; at B=2048 {NT si0,
NT si1, NT pv1}.  The copy is bit-exact and pv is semantics-free, so
the bit class is intact by construction AND by measurement.

### What did NOT work / neutral observations with numbers

1. **si and pv on one socket lose or tie, as designed-for**: +9.9% and
   +5.5% driver-level at B=2048.  If the NODE also rejects the whole
   si/pv ladder, the strided input read is exonerated as the 65 GB/s
   wall and the next suspect is the compute clock or claim granularity
   — that closes the question either way, which is the point.
2. **Parking measures ~0 on wallaby** (6.868 vs 6.867 forced T16) — not
   claimed as a win; it is free by construction and exists for the
   node's clock question.  matrixsimd measured ~1% on the same host.
3. Raw-ssh trap, eighth appearance, avoided again: all remote work ran
   through `~/tmp_l23rader_r4/dev.sh` invoked as `ssh wallaby /abs/path`
   (NOTE: `~/tmp_l23r4/` is matrixsimd's dir this round, not mine —
   check ownership before reusing a dev dir name).

### Borrowed this round (attributions)

* **Staged sequential input** (the knob design, si=2 far-only shape, and
  the wallaby copy-tax numbers that set expectations): **L23_matrixsimd
  mt_r4**, who ported the sequential-read discipline of the >=100 GB/s
  mt_r3 entries (L36_pencilfused's paced read cursor, L36_mixedradix's
  sntp) per the verdict's L=23 order.  Both L23 records carried the idea
  since mt_r1; they built it first this round — per the standing rule,
  copied.  pv (prefetch-only conversion, no copy) is my own addition on
  top; if it wins on the node it answers their "interleave the copy"
  next-item for free.
* **Parked non-participants**: L23_matrixsimd mt_r4 (park design with
  parkfrom-after-pick), L36_pfa mt_r3 (nap-after-1ms), verdict 4.4.
* **What I did NOT rebuild, on the node's own verdict**: my tree barrier
  and weighted splits (my r3, 0-for-9), team-width rows (verdict 5,
  four refutations), dyn volume claiming (matrixsimd r3: dy0 in every
  node pick).

### Node predictions (pre-registered)

* **B=2048**: the si/pv ladder asks ONE question — is the 23-way strided
  input read the 65 GB/s wall?  If si1/si2 wins: **4.6-5.4 us/vol**; if
  pv wins instead, the same at zero copy tax (and pv should then beat
  si).  If the whole ladder loses to NT static at ~5.8, the read
  pattern is exonerated and the residual is compute clock (watch the
  new NT-w2 row) or write-side/claim-granularity.  Both entries ship
  si this round, so the cross-check is built in: if their si wins and
  mine loses (or vice versa) the difference is schedule, not mechanism.
* **B=128**: plain static holds (2.23-2.30) unless plain+si or plain+pv
  displaces — the cell is half cache-resident on the node (47.5 MiB vs
  76 MiB aggregate), so I expect a smaller si effect than at B=2048;
  2.0-2.3 either way.
* **B=1**: fused T16 pick again, 11.5-12.0, now with pk=1 (16 far-socket
  workers napping).  Under ~11.3 = parking bought real clock and the
  panel rule ("never leave busy spinners next to a scored team") gets
  my latency-cell datum; flat = the 2.29 GHz was governor/licence, the
  null goes in the record, and the cell stays parked for good.

### Next

1. Read si/pv/w2 off the node strings.  If si won: tune the staging
   (half-volume staging to halve L2 footprint; NT loads are useless on
   WB memory, skip).  If pv won: race pacing depth (prefetch 2 volumes
   ahead at 66 lines/plane).  If both lost: claim granularity >1 volume
   for UPI stream continuity (matrixsimd's next-item), or accept the
   ceiling and write it down.
2. If B=1 moved with pk=1, propose the panel-wide spinner rule with the
   latency-cell number; if not, the cell is closed — say so and stop.
3. If NT-w2 embarrasses NT-w4 on the node (licence clock), consider a
   w2 si/pv pairing next round; do not pre-build it.
