# mt_r4 — monitor's verdict

Measured on `p55n3`, Intel Xeon Gold 5218 (Cascade Lake, 2×16 cores, AVX-512 with one
512-bit FMA unit, 1 MiB L2/core, 22 MiB L3/socket, DDR4-2666), 32 of 32 threads,
`OMP_PROC_BIND=close`, governor `powersave`, gcc 11.4.0, slurm job 438579.
Round mt_r3 was measured on the **same host with the same binding** (job 438566), so every
round-over-round comparison below is machine-clean.

Scoring is min-of-three-processes. Where a min is drawn from a high-spread sample I say so;
several of this round's "wins" are inside their own run spread.

---

## 1. Headline per geometry

Panel best vs best library, per cell. `×` is the panel entry's advantage over the fastest
library at that cell.

### L = 6 (volume 216)

| cell | panel best | best library | × |
|---|---|---|---|
| non-batched | **L6_pfa 0.219 µs** (38.2 GF/s) | mkl_dfti 0.372 µs | **1.69×** |
| batched B=4096 | **L6_pfa 0.009 µs/t** (37.94 µs/call, 904 GF/s) | mkl_dfti 0.021 µs/t (86.77 µs/call) | **2.29×** |
| batched B=65536 | **L6_pfa 0.039 µs/t** (2570.7 µs/call, 214 GF/s) | fftw3_estimate 0.121 µs/t (7949.3 µs/call) | **3.11×** |

The panel now wins every L=6 cell. B=65536 was the panel's loss in mt_r3 and is now its
best streaming result: 452,984,832 B of compulsory in+out traffic in 2570.7 µs =
**176 GB/s**, the highest sustained bandwidth any entry has reached on this node in any
round (see §5 — this displaces L=36's 150 GB/s as the panel's bandwidth record and proves
150 is not the machine's wall).

Caveat, stated because the margin depends on it: fftw3_patient scored **0.066 µs/t** at this
same cell in mt_r3 and **0.122 µs/t** here — the same binary on the same host, a 1.85× swing
from planner nondeterminism (FFTW alternates a 4.3 ms and an 8.0 ms plan shape within one
process). Against FFTW's r3 lucky plan the honest margin is **1.69×**, not 3.11×. Both
numbers are real; the 3.11× is this round's measurement and the 1.69× is the defensible
floor.

### L = 8 (volume 512)

| cell | panel best | best library | × |
|---|---|---|---|
| non-batched | **L8_fusedaxes 0.553 µs** (41.7 GF/s) | mkl_dfti 0.664 µs | **1.20×** |
| batched B=2048 | **L8_fusedaxes 0.026 µs/t** (53.80 µs/call, 877 GF/s) | mkl_dfti 0.037 µs/t (75.06 µs/call) | **1.40×** |
| batched B=32768 | L8_fusedaxes 0.170 µs/t (5560.6 µs/call) | **fftw3_patient 0.159 µs/t** (5225.1 µs/call) | **0.94× — LOST** |

L=8 B=32768 is **the only cell in the whole panel, at any geometry, that a library still
holds**, and it has held it for four consecutive rounds. Our 0.170 µs/t is 96 GB/s;
fftw3_patient's 0.159 is 103 GB/s, delivered by a 32-thread plan. See §6.

### L = 17 (volume 4913)

| cell | panel best | best library | × |
|---|---|---|---|
| non-batched | **L17_matrixsimd 6.220 µs** (48.4 GF/s) | mkl_dfti 23.484 µs | **3.78×** |
| batched B=256 | **L17_matrixsimd 0.760 µs/t** (194.6 µs/call, 396 GF/s) | fftw3_estimate 3.902 µs/t (999.0 µs/call) | **5.13×** |
| batched B=4096 | **L17_winograd 1.237 µs/t** (5065.4 µs/call, 244 GF/s) | fftw3_patient 3.951 µs/t (16182.4 µs/call) | **3.19×** |

L=17 remains the panel's largest margin over every library at every cell.

### L = 36 (volume 46656)

| cell | panel best | best library | × |
|---|---|---|---|
| non-batched | **L36_mixedradix 23.531 µs** (153.8 GF/s) | mkl2026_dfti 38.065 µs | **1.62×** |
| batched B=32 | **L36_mixedradix 5.159 µs/t** (165.1 µs/call, 701 GF/s) | mkl_dfti 7.428 µs/t (237.7 µs/call) | **1.44×** |
| batched B=512 | **L36_mixedradix 9.936 µs/t** (5087.2 µs/call, 364 GF/s) | fftw3_patient 18.716 µs/t (9582.3 µs/call) | **1.88×** |

Read the B=512 result as a **tie, and score L36_pfa as the trustworthy holder**:
L36_mixedradix 9.936 µs/t at **30.3% run spread** against L36_pfa 9.954 µs/t at **0.2%**.
The 0.18% gap is 1/168th of the winner's own spread. The same caution applies at
non-batched (mixedradix 23.531 at 12.2% spread vs pfa 25.638 at 1.0% — on worst-process
rather than best-process, pfa wins) and at B=32 (13.5% vs 1.4%).

---

## 2. What changed since mt_r3

### L = 6 — the round's biggest genuine gain

| cell | mt_r3 | mt_r4 | |
|---|---|---|---|
| B=1 L6_pfa | 0.211 | 0.219 | −3.8%, inside r3's own 4.0% spread |
| B=4096 L6_pfa | 37.746 µs | 37.940 µs | flat |
| **B=65536 L6_pfa** | **0.079 µs/t** | **0.039 µs/t** | **2.02× faster; cell taken from FFTW** |
| B=65536 L6_unrolled | 0.096 µs/t | 0.073 µs/t | 1.32× faster |

Both L=6 entries pre-registered this cell and both hit. L6_pfa predicted "~39 ns/vol or
better if the fast mode materialises under the dwell"; its description reads
`exre=fused_pf_nt_xa_d2,T=32 0.0394us/vol (plan 3pass_pf_nt_xa_d2,T=32)` and the scored
number is 0.039. L6_unrolled predicted "if the node returns to the L6_pfa band (~79 ns/vol)
… (a) is confirmed"; it read 73 ns after replacing its race pool with a deterministic
install (`rd=pool` → `rd=det`).

No L=6 regression. **L6_unrolled's relative position worsened** (1.21× behind pfa at
B=65536 in r3, 1.87× behind now) for a reason that is this round's most important structural
datum — see §5.

### L = 8 — flat, plus one crash

| cell | mt_r3 | mt_r4 | |
|---|---|---|---|
| B=1 L8_fusedaxes | 0.551 | 0.553 | flat |
| B=2048 L8_fusedaxes | 53.496 µs | 53.800 µs | flat |
| B=32768 L8_fusedaxes | 0.172 | 0.170 | flat (−1.2%) |
| B=32768 L8_radix8 | 0.174 | 0.174 | flat |
| B=2048 / B=32768 L8_batchsimd | 0.027 / 0.175 | **SIGSEGV** | regression to a crash — §3 |

Three independent mechanisms were built at L=8 this round and all three measured zero:
grouped L2-staged batch tiling (L8_fusedaxes), the 4K-alias residue model plus permuted
iteration order (L8_radix8), and L8_batchsimd's AA pool runners (which never ran). Both
surviving entries landed inside their own pre-registered null branches. Detail in §5.

### L = 17 — one cell flat, one entry regressed 1.56×

| cell | mt_r3 | mt_r4 | |
|---|---|---|---|
| B=1 L17_matrixsimd | 5.976 | 6.220 | −4.1%, mild regression (spreads 1.8% / 0.8%) |
| B=256 L17_matrixsimd | 0.756 | 0.760 | flat |
| B=4096 L17_winograd | 1.219 | 1.237 | flat (−1.5%); its one mechanism did not install |
| B=4096 L17_matrixsimd | 2.911 | 2.905 | flat — and that is the refutation, §5 |
| **B=4096 L17_rader** | **1.289** | **2.012** | **REGRESSION 1.56×, 44.5% spread** |
| B=1 L17_rader | 6.955 | 7.319 | −5.2% regression |

### L = 36 — one large gain, one regression

| cell | mt_r3 | mt_r4 | |
|---|---|---|---|
| B=1 L36_mixedradix | 23.027 | 23.531 | flat; 12.2% spread (r3: 12.3%) — the lottery survived |
| B=32 L36_mixedradix | 5.362 | 5.159 | +3.8%, retakes the cell from pfa |
| B=32 L36_pfa | 5.212 | 5.294 | −1.6% |
| **B=512 L36_pfa** | **19.322** | **9.954** | **1.94× faster; ties the leader at 0.2% spread** |
| B=512 L36_mixedradix | 9.896 | 9.936 | flat, but spread 0.9% → 30.3% |
| **B=512 L36_pencilfused** | **10.861** | **11.678** | **REGRESSION 7.5%, outside its own predicted bands** |

### The other four geometries (measured, not headline)

| geometry | change | verdict |
|---|---|---|
| L=13 | direct B=1 5.868→5.672; rader B=512 0.302→0.286, B=8192 0.983→0.929 | improved; but L13_rader B=1 spread 42.0% and its description reads `pick=p2`, the pick its own record calls "impossible under the strict rule" |
| L=23 | rader B=1 11.865→11.080 (+6.6%); rader B=2048 5.810→5.941 (−2.3%) | net improvement; `si=0` everywhere — see §5 |
| L=45 | **mixedradix B=256 47.687→32.553 (1.46× faster)**; pfa flat everywhere | the round's second-largest gain |
| L=64 | **blocked B=1 128.691→102.933 (1.25×), B=8 77.127→66.496 (1.16×, takes the cell from MKL)**; **radix8 B=1 136.342→153.147 (−12.3%, now loses to mkl_dfti 152.700)** | blocked overperformed all three of its predicted bands; radix8 regressed at B=1 |

---

## 3. Failures, crashes, and things that must not survive on a fast number

`build_errors.txt` is **empty** — all 19 panel sources plus `baseline_matrix` compiled.
All 19 appear in the round's backend list. Every scored row reads `ok` with rel-L2 between
1.4e-16 and 4.5e-16 against the NumPy reference (tolerance 1e-12); **no entry produced a
wrong answer at any cell it completed.** `baseline_matrix` is absent from the largest batch
of every geometry, which is the harness's uniform cost cap, not a failure.

### L8_batchsimd — CRASHED, root-caused, disqualified

`failures.txt`, all six batched attempts:

```
L8_batchsimd L=8 B=2048  run=1,2,3 exited 139
L8_batchsimd L=8 B=32768 run=1,2,3 exited 139
```

139 = SIGSEGV. It survives only at B=1 (0.554 µs, correct, second by 0.2%) because that
path is serial and never enters the pool.

This is a **regression the round introduced**, and the cause is in the diff. mt_r4 added two
AA pool runners (`R_AA_PS0`, `R_AA_N_PS0`) that dereference a new per-worker arena array:

- `impl_4/L8_batchsimd.c:1588` — `double *aab = pl->waab[tid];`, guarded only by the
  comment "Candidates are gated on the arenas at plan time, so `waab` is never NULL here."
- `impl_4/L8_batchsimd.c:1664` — `pl->waab[tid] = aab;` is the **only** assignment to
  `waab` in the file, and it lives inside the pool-worker startup function.
- `impl_4/L8_batchsimd.c:1563` — the declaration promises "`[0]` = the plan's `aab`, set by
  `fft3d_create` after pool creation". **That assignment does not exist anywhere in the
  file** (`grep waab` returns five hits, none of them it).
- Pool workers are created for `t = 1 .. nmax-1`; `mt_exec_raw` runs rank 0's share on the
  caller's thread via `mt_run_job(pl, 0, scr0)`.

So with an AA runner installed, rank 0 reads `waab[0] == NULL` (the pool is `memset` to
zero at creation), computes `bl = 0`, and stores through
`ascr = (double *)NULL + k*8` — an address in the first page. The candidate gate is
`if (p->aab)`, which tests the *plan's* arena, not rank 0's, so it does not prevent this.
Deterministic, which is why it reproduced 3/3 at both batched cells.

**Two independent disqualifiers under CURATION.md:** it crashed, and
`strategies/L8_batchsimd.md` has **no mt_r4 section at all** — the record ends at mt_r3
while 168 diff lines of new code went in. CURATION is explicit that entries whose strategy
record is missing are not promoted, because the record is what makes the code useful later.
Not promoted. The fix is one line in `fft3d_create`, and mt_r5 should also add an assertion
rather than a comment.

### Nothing else crashed, hung, or went missing

But three results should not be quoted as clean wins, on evidence:

1. **L36_mixedradix's three L=36 firsts** are all drawn from high-spread samples
   (12.2% / 13.5% / 30.3%). At B=512 the 0.18% margin over L36_pfa is 1/168th of the
   winner's own spread. Its record pre-registered "the lottery is dead by construction"
   for B=1 after dropping cross-package split teams; the spread went 12.3% → 12.2%. The
   lottery is **not** dead, and the mechanism blamed in r3 (an 18-thread team straddling
   sockets) was therefore not the whole cause.
2. **L17_rader's 2.012 µs/t at B=4096 carries 44.5% spread**, against 0.2% in mt_r3.
3. **L13_rader's 5.949 µs at B=1 carries 42.0% spread**, and the installed pick (`p2`) is
   one its own pre-registration declares unreachable under the new strict rule. The rule is
   not doing what its author believes; that is a bug to find, not a number to celebrate.

And on the library side, for symmetry: fftw3_patient's plan choice is unstable enough to
move a headline (L=6 B=65536: 0.066 → 0.122 between rounds, 1.85×; L=13 B=8192: 0.603 at
**110.2%** spread in r3 → 0.780 here; L=36 B=32 84.5% spread this round). Margins measured
against fftw3_patient at deep cells should be quoted with that uncertainty attached.

---

## 4. Claimed numbers versus measured, and where the machine explains it

The implementers develop on `wallaby`, a Xeon Gold 6448Y (Sapphire Rapids, full-clock
AVX-512, two 512-bit FMA units, 2 MiB L2/core, DDR5), and are scored here on Cascade Lake
(one 512-bit FMA unit, 1 MiB L2, DDR4-2666). MKL alone spans 2.9× between those machines,
so an absolute-time mismatch is expected and is not evidence of anything. What matters is
whether the *mechanism* transferred.

**Machine difference, benign — absolute times only.**

| entry / cell | claimed (wallaby) | measured (node) | ratio |
|---|---|---|---|
| L6_pfa B=1 | 0.133 µs | 0.219 µs | 1.65× |
| L6_pfa B=4096 | 0.0061 µs/t | 0.009 µs/t | 1.48× |
| L6_pfa B=65536 | 0.0337 µs/t | 0.039 µs/t | 1.16× |
| L8_radix8 B=1 | 0.324 µs | 0.574 µs | 1.77× |
| L8_radix8 B=32768 | 0.0749 µs/t | 0.174 µs/t | 2.32× |
| L8_fusedaxes B=32768 | 0.0741 µs/t | 0.170 µs/t | 2.29× |
| L17_matrixsimd B=1 | 4.05–4.27 µs | 6.220 µs | ~1.5× |
| L17_rader B=1 | 4.99 µs | 7.319 µs | 1.47× |
| L36_mixedradix B=512 | 7.20 µs/t | 9.936 µs/t | 1.38× |
| L36_pfa B=512 | 7.33–7.41 µs/t | 9.954 µs/t | 1.35× |
| L64_blocked B=1 | (band 110–129) | 102.933 µs | faster than predicted |

The L=8 streaming cells are the widest gap (2.3×) and it is exactly where it should be:
that cell is DRAM-bound and wallaby has DDR5 against the node's DDR4-2666. Every one of
these is attributable to the machine and none of them is a discrepancy in the sense that
matters.

**Where the machine difference is NOT a good excuse, because the mechanism inverted.**

1. **L17_winograd's `ntb`.** Its forced A/B on wallaby measured the per-kx-block NT
   staging beating the plain path by **20–26%** at every variant, and its full tuner picked
   `h8+ntb` 3/3 at **0.836–0.845 µs/vol**. On the node the entry's own arena rejected it:
   the description reads `nt=0, ntb=0` and the cell scored **1.237** — no change from r3's
   1.219. Its pre-registration is unambiguous about what that means: "anything with ntb=0
   AND nt=0 at >5% margin would mean CLX NT store throughput under this access pattern is
   the binding constraint." The DDR5-vs-DDR4 difference explains the *magnitude* of a
   transfer loss; it does not explain a 20–26% SPR win becoming a >5% CLX loss. This is a
   real machine-dependent inversion of an NT-store mechanism and should be recorded as
   such, not as noise.
2. **L8_radix8's alias model.** The line-residue model predicted 118 → 110 blocked loads
   per volume at θ=56 and the permuted order 83; wallaby (which the record correctly says
   *hides* 4K-alias stalls) showed nothing. The node's own on-buffer governor priced it:
   `govm{mth-1f-nt-pfs@raw=0.1740, mth-1f-nt-pfs@56=0.1739, lock=raw}` — **0.06%**. The
   model is not miscalibrated for CLX so much as measuring a non-effect at this cell.
3. **L36_mixedradix's `sntx`.** Predicted 9.0–9.6 µs/vol if the paced read-channel
   prefetch installed, 9.8–10.1 if it tied and `sntp` reinstalled. Measured 9.936, i.e. the
   tie branch — so by the record's own stated criterion, "the overlap-the-read-channel-with-
   the-NT-drain idea is dead at L=36." Same conclusion on both machines; no attribution to
   hardware needed.

**Reporting gap worth fixing in the harness, not in anyone's code.** The leaderboard's
`backends:` column carries one description string per entry, and it is whichever process
wrote last. For three entries this round it is the *wrong cell's* string, and the round's
pre-registered telemetry is therefore unreadable from the results: L8_fusedaxes published
its B=2048 arena (no `gNN` rows, so the grouped-variant ranking at B=32768 — the whole point
of its round — is invisible), L36_mixedradix published its B=1 string (no `spA=`/`sxA=`, so
the sntp-vs-sntx arena prices are invisible), and L45_pfa published its B=1 string (no `mts`
rows). All three entries did what the round asked and instrumented it; the harness dropped
the instrument. One string per (entry, cell) would fix it.

---

## 5. Which LITERATURE.md §4 open question moved

### §4.3 — "Is axis fusion worth 3× or 3%?", and specifically the L2↔DRAM re-opening

This is the question the round was aimed at. §4.3's own re-opening note names the untested
case precisely: every prior panel experiment fused across an L1↔L2 boundary (2.6× bandwidth
gap); the untried construction was **tile the batch so a tile fits L2, then run all three
axes inside the tile**, across the L2↔DRAM boundary where the gap is 7×, and it calls that
"the largest untried structural move on the board."

Six entries ported it. **The construction was rejected by measurement at every geometry
where it was newly built, on this machine:**

| geometry | port | node result |
|---|---|---|
| L=8 | L8_fusedaxes: batch-axis groups of G volumes, per-thread L2 group arena, pure-read phase A then pure-NT-write phase B, ± team phase barrier | 0.170 vs r3's 0.172 — **parity**, its own branch (a) |
| L=13 | L13_direct: staged (`st`) and staged+NT (`sh`) X-pass reads | `gov{tf:615, th:1175, st:2245, nt:1046, lock=tf}` — staging is **3.6× worse**; `tf` locked |
| L=17 | L17_winograd: per-kx-block NT staging to delete the `out` RFO | `nt=0 ntb=0`; cell flat at 1.237 |
| L=23 | both entries independently: sequential input staging (`si`) | `si=0` in both descriptions; forced driver runs **+9.9%** (rader) and **+8.7%** (matrixsimd) |
| L=36 | L36_mixedradix: paced T2 read cursor overlapping the NT drain (`sntx`) | tied `sntp` in-arena; cell flat at 9.936 |

The two cells that *did* move — L=36 B=512 (1.94×) and L=45 B=256 (1.46×) — moved for a
different reason, and both records say so plainly: **install the wide sustained shape
deterministically and dwell in it, instead of racing a cold probe.** L36_pfa's diagnosis is
the cleanest statement of it: a one-execute probe of the wide team reads 24.4 µs/vol against
the 9.9 the same shape sustains, so "the choice probe-race-vs-pin is not instrumentation
hygiene, it decides which regime the scored loop runs in." L45_mixedradix's governor is the
corroboration: `gov{fr0=0, fr=18}` — pages did spread mid-run under a napped pool and an OMP
region, where r3 read `fr=0`.

**So §4.3's answer for the L2↔DRAM gap on Cascade Lake: pass count and tiling are worth
approximately nothing (0 to −260%), while the process's steady-state streaming regime is
worth up to 1.94× at the same pass count and the same compulsory traffic.** That extends the
r3 finding (store order beat pass count 18% to 5%) one level down the memory hierarchy, and
it means Tolmachev's avoided-passes-times-bandwidth-gap rule, which survived r3 with numbers
attached, does **not** predict this boundary: the 7× gap is real and buys nothing here
because the DDR4 controllers were already saturated at the finer grain.

### §4.5 — "Padding: does L=8 need it, and where?" — closed at the allocation level

L8_radix8 did exactly what §4.5 asked for ("measure by construction at several
`(scr − out) mod 4096` values") without needing the unavailable
`ld_blocks_partial.address_alias` counter. It pinned the residue to its model's optimum
(θ=56 batched, θ=58 at B=1), raced it against the raw layout on the caller's real buffers,
and published: `govm{@raw=0.1740, @56=0.1739, lock=raw}` — **0.06%**, and B=1 landed 0.574
against a predicted 0.570–0.578 "if θ does nothing". Its own falsifier fired.

**Answer: on Cascade Lake, controlling the 4K-alias residue by construction at L=8 is worth
0.06% at the streaming cell and nothing at B=1.** ducc0's `make_noncritical()` guard remains
cheap insurance and costs nothing, but the 7% residual to fftw3_patient at L=8 B=32768 is
**not** aliasing, and §4.5 should stop being the named suspect there. One live thread
remains: the permuted iteration order (`1fp`) beat the unpermuted one in the node's own arena
(0.172 vs 0.175, 1.7%) but was not installed — that is §4.6's schedule-search question, not
§4.5's, and it is a free 1.7% next round.

### The finding that was not on §4's list, and matters most

At L=6 B=65536, on the same node in the same round, at identical compulsory traffic:

- **L6_pfa**, pinning `3pass_pf_nt_xa_d2` at **T=32** with a ~3 s create-time dwell and no
  team-width race: **0.039 µs/t = 176 GB/s**.
- **L6_unrolled**, whose cold on-buffer re-race chose **T=16**: **0.073 µs/t = 95 GB/s**
  (`exre=3pass_nt_pf,T=16 73.1ns`).

That is **1.85× from team width alone**, in the direction the mt_r3 verdict closed the
question *against* — r3 ruled the wide team 19–35% slower from four independent on-buffer
races and ordered the shrink race deleted at L=6. L6_pfa deleted the race and pinned wide
anyway; the pin is what won the cell. The mechanism is now explicit: **a cold, millisecond-
scale race cannot price a regime that takes seconds of sustained wide streaming to appear,
and it systematically selects the narrow team, which is roughly half speed.**

This also retires the panel's other favourite explanation. L17_matrixsimd removed the spin
pool from its streaming process entirely, and its clock probe recovered exactly as
pre-registered (`clk512` 2.29 → **2.89 GHz**, `disp=omp` in the description) — while the
time did not move at all: **2.911 → 2.905 µs/t**, 0.3% spread, still 2.35× behind
L17_winograd running the same ~1240 lines of arithmetic. Its own pre-registration: "if time
stays ~2.9 with clk512 recovered, the clock was a symptom and the residue is elsewhere."
Meanwhile L17_rader went the other way (`dsp=omp` → `dsp=pool`) and lost 1.56×. The rule
"never leave a spin pool alive at a streaming cell" is confirmed where a pool spins *during*
the scored run (L=6 unrolled, 1.32×; L=17 rader, 1.56× in reverse) and refuted as an
explanation for L17_matrixsimd's 2.35×. **The depressed AVX-512 clock was a correlate, not
the cause.**

---

## 6. The single highest-value thing mt_r5 should attack, per geometry

**L = 6 — the non-batched codelet's port utilisation.** All three cells are won and
B=65536 is at 176 GB/s, ~80% of a two-socket DDR4-2666 sustained ceiling; there is little
left there. B=1 is single-threaded, 4752 flops in 0.219 µs ≈ 635 cycles at 2.9 GHz — under
half of one core's FMA throughput, against a cell nobody contests within 1.69×. That is the
only place at L=6 with 2× of headroom against a hardware floor rather than a memory wall.

**L = 8 — re-price the wide team at B=32768 with a pin and a dwell.** This is the panel's
only lost cell, fftw3_patient holds it with a **32-thread** plan at 103 GB/s, and our best
is T=16 at 96 GB/s. The width conclusion both L=8 entries are relying on
("team width is SETTLED … wide 19–35% slower") comes from create-time and on-buffer races
of exactly the kind L=6 just showed to be blind to the wide regime — L8_radix8's own node
arena reads `mth-1f-nt-pfs=0.175` against `mt-1f-nt-pfs=0.209` and its author shrank the
width ladder on the strength of it. Rerun that comparison as L6_pfa did it: pin T=32, dwell
seconds, no race. Three L=8 hypotheses died this round (placement, aliasing, L2-staged
grouping); this is the one the round's own best result says was closed prematurely. Second
priority: restore L8_batchsimd (`waab[0]`).

**L = 17 — diff L17_matrixsimd's create/execute against L17_winograd's, at the same engine
and the same dispatch.** Three entries now run the same engine at B=4096 and measure
1.237 / 2.012 / 2.905 µs/t — a 2.35× spread from plumbing alone. Dispatch shape and clock
are both eliminated as the explanation (§5), which is exactly the fallback matrixsimd
pre-registered. This is a bounded, cheap, two-file comparison with a 2.35× prize, and it is
still the panel's most actionable single gap. It also subsumes the L17_rader repair: revert
its dispatch flip and the 1.56× regression goes with it.

**L = 36 — socket-local NT streams at B=512.** The two leaders are tied at 150 GB/s and the
read-channel-overlap idea is dead by its author's own criterion, so the record's reading
that this is "the same 150 GB/s wall" is the thing to attack — and this round refutes it:
**L6_pfa reached 176 GB/s on the same node, same round, same in+out traffic pattern.** So
L=36 B=512 has at least 1.18× of demonstrated, not hypothesised, headroom, and both L=36
records independently name the same next move: make each socket's threads write their NT
streams to their own memory controllers. L6_pfa's dev measurement points the same way from a
third direction (2.15× from `OMP_PROC_BIND=spread` over `close` at its deep cell). Secondary
at L=36: L36_mixedradix's B=1 lottery is still 12.2% wide after the fix that was supposed to
kill it.

---

## 7. What to keep

Applying CURATION.md in its stated order.

**Fastest correct entry per geometry (rule 1)** — L6_pfa, L8_fusedaxes, L17_matrixsimd
(B=1, B=256) and L17_winograd (B=4096), L36_mixedradix, plus L13_direct/L13_rader,
L23_rader/L23_matrixsimd, L45_pfa, L64_blocked. Where two entries hold different cells of
one geometry, both are the fastest entry somewhere and both are kept.

**Structurally different runner-ups that are close (rule 2)** —
*L6_unrolled*: within 1.4% at B=1 and 3.6% at B=4096, a distinct unrolled-ymm codelet, and
this round it is the control that isolates the 1.85× team-width effect (§5) — the panel needs
both halves of that pair on the reading list.
*L8_radix8*: within 4% / 4% / 2%, a distinct 52-instruction radix-8 codelet, and it carries
the round's §4.5 closure.
*L36_pfa*: 0.18% behind at B=512 with a 150× smaller run spread, and its 1.94× is the round's
cleanest demonstration of pin-don't-race.
*L45_mixedradix*: 1.46× better than in r3, now within 1.22× at B=256 and 3–4% at the other
cells, beats every library at every cell, and its `fr=18` is the round's only positive
page-spread reading. It was correctly held back in r3; it has earned promotion now.
*L64_radix8*: structurally distinct from `blocked`'s two-stage 8×8 split, and see rule 4.

**Instructive failures (rule 3)** — kept inside entries already promoted, which is where
their records live: L17_winograd's `ntb` inversion (20–26% on SPR, rejected by the node's own
arena), L13_direct's `st:2245` vs `tf:615` staging refutation, L8_radix8's `@56` = 0.06%
alias null, L36_mixedradix's `sntx` tie, and L23's doubly-measured `si` rejection (+9.9%,
+8.7%). No additional entry needs promoting to preserve a negative this round.

**Beat a library (rule 4)** — every promoted entry beats at least one library baseline at
every cell it ran. L64_radix8 qualifies on this ground independently of rule 2 (B=128:
146.753 µs/vol against mkl_dfti's 282.294, 1.92×) despite regressing 12.3% at B=1.

**Not promoted, and why:**

- **L8_batchsimd** — SIGSEGV at both batched cells, 3/3 runs each, root cause identified at
  `impl_4/L8_batchsimd.c:1588` (§3); and no mt_r4 strategy record exists for the 168 diff
  lines that broke it. Two separate CURATION disqualifiers. `impl_4/` preserves the source
  and the crash is documented here; nothing is lost by leaving it off the reading list.
- **L17_rader** — fourth consecutive round third at all three L=17 cells, a 1.56×
  regression at B=4096 with 44.5% spread, and by its own record ~1300 lines copied verbatim
  from L17_winograd's engine via L17_matrixsimd's port. That is the near-duplicate case
  CURATION names explicitly. Its instructive number (`eng/pl=1.039/1.112` in-arena against
  2.012 scored under `dsp=pool`) is preserved in `strategies/L17_rader.md`, which is tracked
  regardless of promotion, and is quoted in §5.
- **L36_pencilfused** — regressed 7.5% at B=512 to outside both of its own pre-registered
  bands, third at all three cells, and a third member of the same PFA 4×9 family as two
  entries already being promoted. Its negative (borrowing the winner's `ncw` read-flow
  verbatim did not reproduce the winner's number, so the residual delta is in the engine,
  not the schedule) is worth keeping and is kept in its tracked strategy record and in
  `exemplars/mt_r3/`.

PROMOTE: L6_pfa L6_unrolled L8_fusedaxes L8_radix8 L13_direct L13_rader L17_matrixsimd L17_winograd L23_rader L23_matrixsimd L36_mixedradix L36_pfa L45_pfa L45_mixedradix L64_blocked L64_radix8
