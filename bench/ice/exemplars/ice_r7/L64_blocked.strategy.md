# L64_blocked — strategy record (Ice Lake panel)

Lineage: this entry carries the full CLX-panel history in
`bench/geom/strategies/L64_blocked.md` (rounds panel_r6–panel_r11: the 8x8
two-stage radix-8 structure, odd-line-padded hugepage scratch, the st=3
split-complex fused 2-sweep adopted from L64_radix8, the one-bit-class
candidate pool). This file records the Ice Lake panel rounds only.

## Round ice_r1 (reconstructed — the implementer agent crashed)

The ice_r1 agent for this entry died at launch (Bun segfault, see
`results/ice_r1/agents/L64_blocked.log`), so the file ran on the node
exactly as inherited from CLX panel_r11 and no record was written. Node
result: **1205.7 us/transform at the graded B=2 m=134 chain, vs MKL 1016.4
(1.19x) and L64_radix8 1184.0** — the panel's only geometry behind the best
library. The node tuner's pick string on the board reads `mode=nt pf=0 st=3
pro=0`: the cold-streaming arena chose NT stores.

## Round ice_r2

### The diagnosis: the tuner was answering the wrong question

The graded workload is `--chain 134 --unitary` (cases.txt). Reading
driver.c: each chain step's output is immediately re-read by a **cached**
driver-side scaling pass (`*= 1/sqrt(V)`), then consumed as the next step's
input, ping-pong between two 8-MiB buffers — everything stays L3-resident
(16 MiB hot vs the node's 24 MB L3, plus my 4.5 MB scratch). The corpus
already said it plainly (§10.3: "iterate each volume through all m steps
while cache-resident"; "NT stores only for write-once outputs"). My
create-time arena instead timed cold streaming volumes — a workload on
which NT genuinely wins — so the node shipped NT stores, putting a 4-MiB
DRAM round trip into every chain step's scaling pass. The rival
L64_radix8's ice_r1 pick string (`fused-nt+...`) shows the same trap.

### What changed (no arithmetic change; the pool stays one bit class)

1. **Chain-tuned arena.** New `chain_step()` (execute + the driver's exact
   unitary scaling loop) and `chain_round()` (two self-warming steps, then
   4 timed ping-pong steps; rounds interleaved round-robin across
   candidates, min taken). All candidate timing, the sb A/B, and the pro
   A/B now run on this. The NT-vs-cached 2% bar and the 1% simplest-wins
   hysteresis are unchanged — under the chain arena NT loses on merit.
2. **sb axis (my own idea this round):** pass-2's x-FFT can write its
   ky-slab to a 68-KB buffer SB (rows SBKS=136 doubles = 17 lines, odd
   stride, placed in the OB region of the hugepage mapping) instead of back
   in place into SC. SC then stays clean in pass 2, deleting ~4.5 MB/vol of
   dirty-writeback traffic whose data is dead after the z-pass. A/B-decided
   at create time like pro; env `FFT64B_SB=0|1` forces it.
3. **Prefetch hints parameterized:** the hardcoded T1 hints on the pass-1
   next-plane cursor and the ky-slab prefetch became `-D`-sweepable
   (`FFT64B_H1`, `FFT64B_HS`; defaults unchanged at T1).

### Measured on the node (a80n0 via tryout.sh; all PASS rel_l2=4.46e-16,
### chain-check 2.9e-14, repeatable bit-identical across runs)

Same-window candidate table at B=2 (quiet window, FFT64B_VERBOSE):

| candidate (all st=3 pw4) | us/chain-step-vol |
|---|---|
| cached pf0 | 1263.0 |
| cached pf1 (p1pf) | 1212.5 |
| cached pf5 (pfw) | 1167.1 |
| cached pf9 (slabpf+pfw+p1pf) | **1155.4 (pick)** |
| nt pf0 | 1514.9 |
| nt pf6 | 1639.3 |

NT is 20–25% off the cached rows under chain semantics — the whole ice_r1
deficit and then some. Headline same-window pairs (mine vs MKL, min
us/transform):

| B | L64_blocked | mkl_dfti | ratio |
|---|---|---|---|
| 2 (graded) | **903.1** | 1006.8 | 0.90 |
| 1 | **827.7** | 978.3 | 0.85 |
| 8 | **1188.6** | 1547.4 | 0.77 |

The board's ice_r1 numbers (1205.7 vs 1016.4) become ~903 vs ~1007: the
one loss should flip to a win. Operation count is unchanged from panel_r10
(~1.6M FMA-port vector ops/volume; the workload is L3-traffic-bound, which
is exactly why the store-mode decision dominated).

Bit-class verification: forced `FFT64B_SB=1`, `FFT64B_PRO=1`,
`FFT64B_MODE=1 FFT64B_PF=0`, `FFT64B_MODE=0 FFT64B_PF=5` all cmp-identical
to the shipped pick's output **and** end-of-chain state at B=2. A
run-to-run pick flip still cannot produce an unvalidated number.

### What did not work, with the numbers that killed it

* **sb=1 at B=2 on the node: declined**, off 975.9 / on 1023.7 us (and
  off 1178.2 / on 1190.3 in a second window). Deleting the SC dirty
  writeback does not pay when everything is L3-resident — L3 writebacks
  overlap; the extra 68-KB L2 footprint costs more. Kept as an A/B (it may
  pay in a DRAM-bound regime, e.g. large B).
* **T0 hint twins (`-DFFT64B_HS=3`, `-DFFT64B_H1=3 -DFFT64B_HS=3`):**
  MKL-ratio 0.912 / 0.888 vs defaults' 0.897 across different windows —
  inside window noise, defaults kept. Cross-window absolute comparisons at
  this geometry are worthless: my own identical build read 903–1288 us
  across four windows while MKL read 1007–1101. Only same-window pairs and
  same-window tuner tables are usable (the CLX record's standing rule,
  reconfirmed here; contention is other implementers' leases sharing L3).
* **Noisy-window tuner inversions:** in a contended window the candidate
  table inverted (pf0 980 beat pf9 1023) and pro flipped ON. Harmless for
  provenance (one bit class) and the monitor's scoring window drains all
  leases, so the shipped tuner decides under quiet conditions.

### Attribution

* The chain-residency and NT-only-for-write-once facts that drove the
  arena rewrite: corpus §10 (`docs/literature/10-icelake-under-glass.md`
  §3), plus PANEL_BRIEF's "the workload IS the graded chain".
* Round-robin interleaved best-of-N timing protocol: corpus §10 §0.
* sb, and the chain_step/chain_round arena implementation: this entry.
* Everything structural (st=3 split-complex 2-sweep, DFT8S codelet, padded
  scratch, one-bit-class pool, pro): carried from geom panel_r6–r11 with
  the attributions recorded there (L64_radix8, L36_pfa/pencilfused,
  L8_batchsimd lineage).

### Next

1. Read the ice_r2 board. If the rival L64_radix8 also drops NT (their
   agent reading this: the chain re-reads your output; your `fused-nt`
   pick is the same trap), the gap between us returns to the ~2% structural
   band from CLX and the shared enemy is MKL's remaining ~0 — we should
   both be ahead of it now.
2. The remaining ~900-us budget is L3 traffic: in-read 4M + SC round trips
   ~13M + out-write 8M (RFO) + the driver's own 8.4M scaling pass, at
   ~30 GB/s single-core L3. The two candidate structural cuts, in order:
   (a) the corpus §10 z-split octant layout (7x working-set compression,
   "the single biggest structural win of the best run" — read
   `ext/reference/fft_v4_solutions/1000f989_score1.00/implementation.c`);
   (b) an out-write reorder that fixes the z-pass's 64 store streams at
   exactly 512-KB stride (same 4K page offset every kx — the one
   power-of-2 pathology padding cannot reach because `out` is the ABI's
   buffer).
3. B=8 setup is 3.4 s (chain arena over nv=8). Excluded from scoring, but
   trim rounds at nv>=4 if the monitor ever complains.

## Round ice_r4

### The task changed: the map is the battleground

The graded step is now `state <- (z+c)/(1+|z+c|)`, z = unnormalized
FFT(state), and the driver detects an optional `fft3d_chain` weak symbol.
Without it you are timed through the driver's unfused execute+map fallback
(the 2.24 s suite configuration). Rivals' L=64 number to beat: 0.226 s =
843 us/step-vol at B=2 m=134.

### What changed (in order of impact)

1. **`fft3d_chain` exported, map fused as the LAZY MAP** (adopted from
   corpus 10 2, the rival pipelines' 4/7-convergent shape, verified in
   `ext/reference/fft_v4_solutions/1000f989_score1.00/implementation.c`
   `mapc()`): ping-pong buffers hold RAW FFT output between steps; the map
   is applied inside the NEXT step's pass 1, where raw z and c both stream
   sequentially plane-by-plane, and the y-FFT consumes the mapped state
   directly in split form. One streaming epilogue maps the final step
   (1/134 of the chain). The map costs no extra memory pass, only c's
   4.19 MB/vol of reads. Second ping-pong buffer is plan-owned,
   hugepage-backed (pass 2+3 stores rows into it at 512-KB plane stride).
2. **All-FMA map ladder** (`FFT64B_MAPDIV=0`, the shipped default): m2 with
   a 1e-300 bias (kills the m2=0 NaN and denormal-assist traps, corpus
   10 2), `vrsqrt14pd` seed + one CUBIC Newton for sqrt, `vrcp14pd` seed +
   one cubic Newton for the reciprocal; 15 FMA-port ops per 8 points, no
   divider use. Node-raced best-of-6 alternating (same window, same core):
   **MAPDIV=0 1107.7 / MAPDIV=1 1161.5 / MAPDIV=2 1159.3 us/step-vol** —
   pass 1 was `vdivpd`-throughput-bound (32768 divides x ~16 cyc/vec all
   in one pass), so the corpus's CONTESTED question lands on 1760b1bf's
   side on this exact silicon: burn FMA, not the divider, when all the
   divides sit in one pass. (MAPDIV=2 = rsqrt14+2 quadratic Newtons + one
   exact vdivpd, ~1e-16/app, kept as the -D control and for any future
   long-chain reuse; MAPDIV=1 = cubic + vdivpd.) Map precision arithmetic
   for this point, per the brief's instruction: ~4e-13/application; the
   map is a contraction, measured chain end drift at m=134 is
   **1.683e-12 vs tol 1.34e-11 (8x margin)** at B=2, 1.42e-12 at B=1,
   1.95e-12 at B=8. The exact-map control measures 4.08e-14. This tier is
   legal HERE because m=134; at L=6/8/13 chain lengths ship MAPDIV=2.
3. **Split-complex intermediate layout between chain steps** (my own idea
   this round): step s's pass 2+3 used to ILV-interleave (Wr,Wi) into out
   rows and step s+1's pass 1 immediately DEIN-deinterleaved them back.
   Chain-internal buffers are mine, so pass 2+3 now stores the split
   (re,im) vector pairs at the same addresses (`splo`), pass 1's mapped
   loads skip the deinterleave, and the epilogue interleaves the final
   state. Kills ~131K port-5 shuffles/step-vol — and port 5 hosts an FMA
   pipe on ICX, so every shuffle removed is FMA headroom. Node race,
   same window/core, 5 alternations: **split 1089.4-1093.0 (4/5 reps) vs
   interleaved 1106.1-1108.1** (-1.4%, matches the estimate). Driver-visible
   layouts (x0, final_out, execute) unchanged.
4. **Tuner arena re-targeted at the new graded semantics** (the ice_r2
   lesson, third application): candidates, sb and pro A/Bs are timed on
   fused chain steps with a 0.1-scaled c field. **New NT candidates
   nt+p1pf and nt+slabpf+p1pf**: the c field pushes the B=2 chain working
   set past L3 (ping 8.4 + pong 8.4 + c 8.4 + SC 4.5 = ~30 MB > 24 MB),
   so the ice_r2 "NT always loses under the chain" verdict no longer
   binds. The node agreed emphatically: B=2 arena picked **nt pf1 1153.8
   vs best cached pf9 1378.8**, and the timed run posted the round's best
   number. Write-once NT out stores drop the 4.19 MB/vol RFO read;
   p1pf covers the now-DRAM-resident next-step in-read.
5. **Create-time fused-chain interlock**: 3-step fused chain vs an exact
   chain built from the INDEPENDENT st=0 interleaved kernel + scalar
   sqrt/divide map (gate: rel L2 < 1e-13 for MAPDIV=2, < 2e-12 for the
   cubic tiers). chain_ok=0 (or env FFT64B_CHAIN=0) falls back to
   execute + exact scalar map — verified correct (drift 4.06e-14) and
   slow (1744 us/step-vol): a wrong fused chain can never ship.

### Operation count (per step-volume, AVX-512 vector ops)

FFT unchanged (~1.6M FMA-port ops + 328K shuffles, minus the 131K shuffles
item 3 removes = ~197K). Map adds 15 x 32768 = 492K FMA-port ops + 32768
rsqrt14 + 32768 rcp14, zero divider. Traffic per step-vol is the wall:
~30 MB of L2/L3-level movement (z 4.19 in + c 4.19 in + SC 13.4 round
trips + out 4.19 + RFO-or-not), which is why the store mode and prefetch
schedule decided the round.

### Measured on the node (a80n0 via tryout leases; windows contended,
### spread across windows up to 14% — only same-window pairs quoted)

| config | us/step-vol |
|---|---|
| graded B=2, final build (nt pf1 pick window) | **1038.6** (median 1039.3) |
| graded B=2, same window MKL through driver fallback | 1992.2 |
| B=1 | **1043.5** (7.6%-spread window, median 1224) |
| B=8 | **1169.5** (vs MKL 2073.1) |
| fallback FFT64B_CHAIN=0 (exact map, unfused) | 1744.2 |
| FFT-only unitary chain, same build (for the delta) | 994.4 |

All correctness: single-transform rel_l2 4.46e-16; map-chain PASS at
B=1/2/8 (above); two full runs bit-identical in out AND chain end state.
Setup: 1.0 s at B=2, 4.2 s at B=8 (arena nv=8; trim if the monitor asks).

### What did not work / was measured and declined, with numbers

* **Hardware vdivpd in the map** (both MAPDIV=2 and =1): 1159.3/1161.5 vs
  1107.7 all-FMA, best-of-6 same-window — the lazy map concentrates all
  32768 divides into pass 1 and the divider (~16 cyc/vec zmm) becomes the
  pass's binding resource. The corpus's "one hardware divide per point"
  consensus assumes the divides spread across the whole step; under
  input-side fusion they don't.
* **sb=1 declined again** under the r4 arena: off 1150.0 / on 1167.1
  (consistent with ice_r2; SC's dead writeback is still not the wall).
* **pro stays ON** (off 1147.5 / on 1139.8), now also covering c plane 0.
* Not attempted this round, estimated ~1%: alternating div/rcp chunks
  (0f45aeae's trick) — with all-FMA already winning, the hybrid's ceiling
  is ~11 us/vol; noise floor here is larger.
* The tryout.sh chain-check plumbing is broken for everyone (`$W` unset at
  line 36 under `set -u`, and the checker's `--cin` expands remotely to
  `/c.bin`): worked around by pre-seeding `W=` in the env and running
  check.py manually on the node. Monitor may want to know.

### Attribution

* Lazy map (raw buffers + map in next step's first pass), the rsqrt-Newton
  ladder shape, the 1e-300 bias: corpus 10 2 and the rivals' 1000f989
  `mapc()` / 1760b1bf tiering notes. The all-FMA (no-divider) resolution
  of their contested rsqrt/div question: measured here, this entry.
* Split-complex chain-internal layout, the enriched NT pool under the
  over-L3 working set, the fused-chain interlock against the independent
  st=0 kernel: this entry.
* Everything structural carried from earlier rounds with attributions
  recorded there (st=3 split-sc from L64_radix8, padded hugepage scratch,
  one-bit-class pool, chain-shaped arena).

### Next

1. The remaining gap to the rivals' 843 us is FFT-structural, not map:
   the corpus 10 z-split octant layout (7x working-set compression, "the
   single biggest structural win of the best run") is now clearly the next
   move — the whole chain working set would drop back under L3 even WITH
   the c field, and the store-mode dilemma dissolves. Read
   `1000f989_score1.00/implementation.c` before attempting.
2. If the scoring window (quiet, drained leases) flips the pick back to
   cached, fine — one bit class, and the arena decides under scoring
   conditions. If nt pf1 holds, consider pairing NT with an L2-targeted
   next-step prefetch instead of T1 (FFT64B_H1 sweep).
3. The map ladder can drop ~2 ops (skip the rcp cubic, divide by
   (1+mag)^2 trick or 1 quadratic) only by eating ~1e-11-class drift —
   do NOT: the m=134 budget is 1.34e-11 and 8x margin is already the
   floor I am comfortable signing.

## Round ice_r5

### What changed: the z-split CUSTODY chain (the structural move both L=64
### records had queued for two rounds, finally executed)

`fft3d_chain` no longer ping-pongs driver-shaped volumes through the r4
lazy-map pipeline.  The state now lives, for the WHOLE m-step chain, in a
split-complex padded custody layout -- slot (x, y, g) at
`x*SCXS + y*SCKS + g*16` doubles (the SC strides reused: row 17 lines,
plane 1089 lines, both odd), re vector at +0, im at +8, **lanes = 8 ADJACENT
indices** (z = 8g + lane) -- and **each volume iterates through all m=134
steps while cache-resident** (corpus 10 3's directive, verbatim), one volume
at a time.  One chain step is TWO in-place sweeps of one 4.46-MB buffer:

* sweep A, per 68-KB (L2-hot) x-plane: the lazy map (raw z + custody c,
  MAP8V, all-FMA ladder unchanged from r4) fused into the loads of each of
  the 64 z-rows, the mapped values feeding the 64-point z-line **in
  registers**; then the 8 y-lines (FFT64S at SCKS stride) in place.
* sweep B, per ky-slab: the 8 x-lines (FFT64S at SCXS stride, T0 next-slot
  prefetch) in place.

The natural interleaved layout is touched only at the chain ends: step 1
loads x0 rows with one DEIN pair per slot (custody <-> natural is a pure
interleave -- slot g holds 8 consecutive indices -- ZERO transposes), the
epilogue maps step m's raw z and ILV-stores final_out, and c is repermuted
once per volume per chain into custody form (cost 1/134 of the c traffic;
the steady-state map is now completely shuffle-free, deleting even r4's
per-step c deinterleave).

**The key adaptation vs the rivals' 1000f989 `run64_zsplit`** (read in full
before writing code): their lanes are z-OCTANTS (z = zh + 8l), which forces
a natural/bit-reversed form alternation across steps and a cross-lane masked
DFT8 (`xl_dif8`, ~256 arith + 80 shuffle-class ops per 64-pt z-line).  With
lane = index mod 8 and slot = index div 8, my existing transpose-based
z-line -- DFT8S over slots (g -> k2), CTWV lane twiddle W64^{l*k2}, ONE TR8
pair, DFT8S (l -> k1); the sc_pass23 tail verbatim -- maps custody form to
custody form EXACTLY (slot k1 = k div 8, lane k2 = k mod 8): stable across
steps, no bit reversal, no alternation, at 132 arith + 48 shuffles per
z-line, about half their vector-op count for the z pass.

Working set per volume: state 4.46 + custody c 4.46 = **8.9 MB, fully
L3-resident** (vs r4's ~30 MB that forced NT stores and DRAM round trips).
Per-step traffic: sweep A R+R+W 13.4 MB + sweep B R+W 8.9 MB = **~22.3 MB,
all L3** (in-place writes hit lines the read just brought in; no RFO
question, no NT question, no 512-KB-stride out-store pathology -- both of
r2's "next" items dissolve rather than get solved).

Also this round: the r2-r4 chain-arena tournament is RETIRED (it tuned the
pipeline custody replaces; every knob it decided -- store mode, pf level,
pro, sb -- affects only the unscored execute path now).  create() is one
correctness gate (shipped execute config vs the independent st=0 kernel) +
two chain interlocks (custody 3-step vs exact st=0 chain + scalar map, gate
rel L2 < 2e-12; the r4 pipeline verified only if custody is unavailable or
env-vetoed).  Setup: **0.03-0.07 s** (was 1.0-4.2 s).  Dispatch: custody if
zs_ok, else r4-fused if chain_ok, else exact fallback; FFT64B_ZS=0 /
FFT64B_CHAIN=0 force the controls at create time.  No run-time tuner
decisions remain, so runs are bit-identical by construction.

### Operation count (per step-volume, AVX-512 vector ops)

Sweep A: 4096 z-rows x (map 136 + z-line 132) = 1.10M FMA-port + 197K
shuffles (TR8 pair); 512 y-lines x 1028 = 526K.  Sweep B: 512 x-lines x
1028 = 526K.  Total ~2.15M FMA-port + ~197K shuffles -- essentially r4's
count (the z-line moved passes but kept its shape; the map is unchanged at
15 ops/8 pts).  The round's entire win is traffic and residency, exactly as
the r2 record's traffic audit predicted.  At the measured 667 us/step-vol,
22.3 MB/step is ~33 GB/s of L3 movement -- this decomposition now sits ON
its traffic roofline; the arithmetic (~2.15M ops at ~2/cyc = ~370 us) still
hides under it.

### Measured (a80n0 leased cores via tryout.sh, graded m=134 chain with map;
### same-window MKL pairs; final shipped build)

| config | us/step-vol (min) | same-window MKL | ratio |
|---|---|---|---|
| graded B=2 | **667.5** (best window 660.1; medians 660-670 in quiet windows) | 1767.4 | 0.378 |
| B=1 | **665.2** | 1761.0 | 0.378 |
| B=8 | **675.2** (sd 0.03%) | 2104.7 | 0.321 |
| FFT64B_ZS=0 (r4 pipeline, same build) | 1226.0 | -- | -- |
| FFT64B_CHAIN=0 (exact fallback) | 1731.5 | -- | -- |

vs the r4 board (1048.6, rival L64_radix8 1043.0, MKL 1723.2): **-36%**.
vs the rival pipelines' L=64 target of 843 us/step: **-21%** -- the number
to beat is beaten with the correctness gates intact.  The custody-vs-r4
structural delta measured on one build, adjacent windows: 667 vs 1226.

Correctness, all on the final build: single-transform rel_l2 4.46e-16 at
B=1/2/8; **map-chain end state 1.42e-12 (B=1) / 1.69e-12 (B=2) / 1.95e-12
(B=8) vs tol 1.34e-11** (~8x margin, the same MAPDIV=0 ladder as r4 -- the
FFT reassociation change moved the drift by <1%); out AND chain end state
bit-identical across independent processes at all three batches; both
control paths PASS (r4 1.683e-12, exact 4.06e-14).

### What did not work, with the numbers that killed it

* **Next-ky-slab prefetch in sweep B** (the slabpf lineage, shipped ON at
  first): OFF 667.3/667.6 vs ON 709.2/710.3 us/step-vol, two alternating
  same-window pairs at sd 0.05% -- a clean 6% LOSS.  128 extra prefetches
  per FFT64S are pure issue overhead when the whole volume is already
  L3-resident.  Corpus 10 3 ("software prefetch mostly loses; restructuring
  wins") confirmed on bare metal; default now ZSLAB=0.
* **Dropping the T0 next-slot-pair prefetch in sweep B** (ZT0=0): 684.4 /
  689.0 / 691.0 vs ON 660.1-668.1 across six runs -- OFF loses ~3%.  (One
  644.8-min outlier in a sd-8.6% window taught the usual lesson: never read
  a single noisy-window min as signal.)  Short-range T0 on strided loads
  earns its keep; long-range slab warming does not.
* The first B=1 run read 759 us in an otherwise-quiet-looking window
  (sd 0.01%); re-runs gave 660-665.  Cross-window absolutes at this
  geometry remain worthless even when a single window looks stable --
  the standing same-window-pairs-only rule, reconfirmed again.

### Borrowed / attribution

* The z-split custody idea, per-volume chain residency, and the
  materialize-only-at-the-ends scheme: rival pipeline **1000f989**
  (`run64_zsplit`, read in full) via corpus 10 3/10 6; the brief's
  "iterate a volume through steps while cache-resident".
* The custody-layout enabling observation ("intermediate states need not be
  ABI-interleaved, so pass-1 deinterleave and pass-23 interleave both die"):
  **L64_radix8 ice_r4 next-round note (a)** -- executed here a round later.
* The stable-form transpose-based z-line that replaces their cross-lane
  DFT8 + form alternation: this entry (the sc_pass23 tail from panel_r10,
  reused unchanged).
* Lazy map, all-FMA ladder, 1e-300 bias: carried from my ice_r4 (corpus
  10 2 lineage).  c-in-consumption-form: L64_radix8 r4's `build_csplit`
  idea, custody-shaped.
* tryout.sh chain-check plumbing is STILL broken (r4 note): `$W` unset at
  line 36 under `set -u`, and line 49's `--cin '$W/c.bin'` single-quotes
  $W inside the command substitution so the REMOTE shell expands it (empty)
  -- exporting W locally fixes the driver invocation but not check.py; I
  ran check.py manually on the node for every number above.  Monitor: two
  rounds running now.

### Next

1. This decomposition is traffic-bound at ~33 GB/s of L3 movement; the
   arithmetic hides under it with ~45% slack.  The only remaining traffic
   cut I see is fusing sweep B of step s with sweep A of step s+1, which
   the orthogonal blocking (ky-slabs vs x-planes) forbids; a 4-sweep
   two-step scheme (x of step s + z+y+map of s+1 per plane) would need the
   map BETWEEN them and does not commute with the x-line.  I do not see a
   legal fusion; treat ~640 us as this structure's floor on this node.
2. PMU pass-level attribution (perf_event_open works; still never done at
   L=64): price sweep A vs sweep B in cycles before believing item 1's
   slack estimate.
3. If a future round changes m or the tolerance budget: the map ladder and
   its 8x margin transfer unchanged; at m beyond ~1000 switch MAPDIV=2
   (one vdivpd per 8 points) and re-race -- under custody the divides are
   spread across sweep A, so the r4 "all-FMA wins" verdict may invert.
4. The retired tuner means execute (unscored) runs fixed pf=8/pro=1; if the
   monitor ever scores FFT-only chains again, resurrect the r4 arena from
   git history rather than re-deriving it.

## Round ice_r6

### Standing at round start, and what this round was

ice_r5 board: L64_blocked 639.5 us/step-vol at the graded case (B=2, m=134,
map chain), the geometry lead (L64_radix8 1011.1, MKL 1721.3, rivals' target
843 already beaten).  The r5 record called ~640 "this structure's floor" and
promised (for the third round running) PMU/cycle attribution before believing
that.  This round was a SCHEDULE round: measure first, then attack what the
measurement says -- no arithmetic change, no layout change, one bit class
throughout (every change is prefetch-only or a bit-identical re-blocking, so
end states cmp-identical across all configs raced).

### The attribution (finally done -- and it inverted the r5 guess)

New `FFT64B_PROF=1` (env, create-time): rdtsc around each sweep, printed per
chain call; shipped path untouched (separate instrumented loop).  Graded
chain, quiet window, r5-equivalent build:

| phase | cyc/step-vol | share | traffic | effective BW |
|---|---|---|---|---|
| sweep A (map+z rows + y-lines) | 1,468,900 | 73% | 13.4 MB | ~26 GB/s |
| sweep B (x-lines)              |   528,900 | 26% |  8.9 MB | ~49 GB/s |
| build_c + epilogue             | ~2.6M /chain-vol | ~1% | | |

The r5 record's "sweep B is the latency-exposed one" was BACKWARDS: sweep B's
next-body T0 prefetch already runs it at ~49 GB/s, near this core's practical
L3 ceiling; sweep A -- whose rows I assumed the L2 streamer handled because
they look sequential -- was the entire remaining slack.  Mechanism: the map+z
row bodies (~350 uops) barely fill the ROB, so demand-miss MLP is ~one row's
32 lines, and the streamer retrains at every 4-KB boundary (~3.7 rows).
Lesson for every entry: "looks sequential" does not mean "prefetched"; the
attribution costs one afternoon and answers what three rounds of guessing
did not.

### What changed (shipped)

1. **ZAPF: sweep-A map-row software prefetch** -- T0, THREE rows ahead,
   state row AND c row (32 lines per row body).  This is the rivals'
   1000f989 `iter_z64` schedule verbatim (they prefetch `x+3` at T0 for both
   buffers; I re-read their source this round and it was sitting there all
   along).  Node, same-window pairs: **652.4/660.0 (on) vs 678.6 (off)**,
   ~-3.5%.  Defaults now ZAPF=1, ZAD=3, ZAH=3 (T0).
2. **ZTC: sweep-B seam coverage fix** -- the next-body prefetch aimed at
   `cb+16`, which at g=7 is the row's PAD line, so the first body of every
   ky-slab (1/8 of the sweep) was never covered; now the offset is
   `SCKS-112` across the seam.  Node: wash (679.2/678.9 vs 679.7/679.2).
   Kept: free and principled.  (The fact that it IS a wash confirms sweep B
   simply has no latency problem left.)
3. Final shipped build, node (quiet windows, same-window MKL):
   **B=2 641.1 / 637.4 / 642.7 us/step-vol** across three windows (MKL
   1927.8/1715.9/1722.8), **B=1 713.5** (MKL 1705.0), **B=8 644.5** (MKL
   2024.5).  Same-day, same-window delta vs the r5 code: 667.7/679 -> 637-653,
   about **-4%**.  Correctness: single rel_l2 4.464e-16; map-chain end state
   **1.688e-12 (B=2) / 1.423e-12 (B=1) / 1.951e-12 (B=8) vs tol 1.34e-11**
   (map ladder untouched from r4/r5); B=2 chain end state bit-identical
   across independent processes; both interlocks (custody vs exact st=0
   chain) still gate at create.

### Operation count

Unchanged from r5 (~2.15M FMA-port vector ops + ~197K shuffles per step-vol;
map 15 ops/8pt).  The round added only prefetch instructions: +2 per sweep-A
map row x 4096 = +8K prefetch uops/step-vol at 32 lines/row covered, +0 to
every other phase.  Traffic unchanged at ~22.3 MB/step all-L3; effective
sweep-A bandwidth is what moved (~26 -> ~30 GB/s implied by the -4%).

### What did not work, with the numbers that killed it

* **ZS2 g-slab rebalance** (y-lines moved from sweep A to a per-g sweep B2:
  512-KB slab, y-pass pulls it into L2, x-pass runs on L2 hits;
  bit-identical by construction, same total traffic, motivated by the 73/26
  imbalance): **DECLINED, ~+3%** -- build-level pairs 668.9/699.3 (zs2) vs
  649.6/656.6 (zs0); same-window env-alternated medians 2/3 pairs to zs0.
  Rebalancing compute INTO the efficient sweep does not help when the slow
  sweep's problem is prefetch coverage, not compute occupancy -- and sweep A
  loses the y-line phase that gave its stores/prefetches time to drain.
  Kept env/-D-forcible (`FFT64B_ZS2`), default 0.
* **T1 hints, anywhere**: sweep-B next-body at T1 699.4/699.2 vs T0 679.2/
  678.9; sweep-A ZAPF at T1+distance-2 754.2 vs T0+distance-3 652.4.  On
  this bare-metal node T0-into-L1 wins every raced prefetch site; T1 has
  lost every A/B this entry has ever run here (H-twins r2, these two).
* **ZT2 (sweep-B 2-bodies-ahead)**: 689.7 vs 679.2 -- extra issue slots, no
  latency left to hide.
* **ZAW (ZAPF wrap into plane x+1's first rows)**: 652.8/655.3 vs
  **637.4/642.7 off** -- the wrap bursts 192 T0 lines into L1 during rows
  61-63, displacing the just-written plane that the y-line phase re-reads
  next.  Coverage is not free when the target cache is the one you are
  about to reuse.
* Cross-window absolutes remain worthless (own identical builds read
  637-751 across this round's windows; MKL 1705-1933): every decision above
  is same-window pairs or same-lease env alternation only.

### Borrowed / attribution

* ZAPF's exact shape (T0, distance 3, prefetch BOTH state and c rows in the
  map pass): rival pipeline **1000f989** `iter_z64`, re-read this round.
  Their same loop also software-pipelines the map one row ahead of the
  z-pass -- not adopted yet, noted below.
* The measure-before-restructuring discipline: the corpus 10 0 protocol
  notes and three rounds of my own record nagging me to do it.
* Everything structural carried from r5 (custody chain, 1000f989 via corpus
  10 3/10 6; L64_radix8's split-state observation; map ladder from r4).

### Next

1. Sweep A is STILL 73% of the step after ZAPF (~1.42M cyc in the final
   profile) and now has row-level coverage; the remaining gap to its ~800K-
   cyc compute floor is plane-grain: the y-line phase (8 FFT64S, ~2.8K cyc,
   zero prefetches issued) is dead prefetch time for the NEXT plane's 136 KB
   -- but ZAW showed naive T0 there backfires; try T2/L3->L2 hints or a
   partial (16-line) burst instead.
2. The rivals' map-one-row-ahead software pipeline (map row y+1 to memory,
   z-line row y from memory) decouples the rsqrt ladder from the z-line's
   critical path; my zs_row_map fuses them in registers.  Worth one race.
3. Vol-pair interleaving at B=2 (alternate two volumes' row bodies to double
   demand-miss MLP; working set 17.8 MB, still < 24 MB L3) -- the biggest
   untried lever if prefetch plateaus; risky near the L3 capacity edge.
4. buildc+epilogue are ~2.6M cyc/chain-vol (~1%); the epilogue writes the
   driver's 4-KB-paged final_out -- prefetchw could shave half of one
   percent.  Only worth it if the board gap ever narrows to that scale.
5. tryout.sh chain-check plumbing STILL broken (third round: `$W` unset at
   line 36 under `set -u`; line 49's single-quoted `'$W/c.bin'` expands
   empty on the remote).  Workaround unchanged: pre-seed W=, run check.py
   manually on the node.  Monitor: please fix or bless the workaround.

## Round ice_r7

### Standing at round start, and what this round was

ice_r6 board: 638.8 us/step-vol at the graded case, the geometry lead
(L64_radix8 660.3, MKL 1720.6).  New this round: the rivals' codes
re-benchmarked on THIS node (`results/rivals_icelake/`) put the honest L=64
target at **v6_3f30d81f, 0.175 s = 653 us/step-vol, gate-passing** -- a 2%
margin, not the comfortable 24% the old 843-us number suggested.  The round's
plan was the two structural moves my r6 record queued (the rivals' x-pass map
fusion + cross-step z pipelining, and vol-pair interleaving).  Both were
built, raced, and DECLINED by the node; the round's actual win came out of
the fusion post-mortem: a one-attribute compiler fix and one prefetch
constant.  Net: **607 us/step-vol, -5% vs the r6 board, ratio 0.353 vs
same-window MKL** -- and the r6 "sweep A is mysteriously inefficient"
finding is now fully explained.

### What changed (shipped)

1. **`zs_ztail` force-inlined** (`__attribute__((always_inline))`).  gcc 11.4
   had been declining the plain `inline` at all three z-row call sites since
   the r5 custody rewrite: every one of the 4096 z-rows per step spilled all
   16 Vr/Vi zmm to the stack, re-loaded them in the callee, and paid a
   stack-protector canary (Ubuntu gcc defaults to
   `-fstack-protector-strong`; the arrays forced a frame).  Found by
   counting `%zmm -> (%rsp)` moves in the asm after the ZMS fusion (below)
   ran at the same ~310 cyc/row on L1-hot data as on cold -- residency
   could not matter because the body was call/spill-bound.  Node,
   same-window build pairs vs the r6 exemplar: **615.2/622.0/623.4 (r7) vs
   645.9/642.2/645.3 (r6)**, sd 0.05%, bit-identical output -- **-3.5%**.
   Lesson for every entry: read the asm of your hottest body once per
   toolchain; "static inline" is a request, not a fact.
2. **ZAPF distance 3 -> 2**: the inlined row body is ~25% shorter in uops,
   so the 3-row lead overshoots.  Node: **709.6/711.1 (d2) vs 718.8/718.8
   (d3) vs 727.3/727.5 (d4)** same-window, sd 0.05% -- another **-1.3%**.
   (r6's "distance 2 loses" datum was distance-2-at-T1; at T0 it wins
   post-inline.)  ZAPF itself re-verified: OFF costs ~2-3% (634.9/642.0/
   634.2 vs 615.2/622.0/623.4 same window).
3. Everything else in the shipped path is unchanged from r6 (custody chain,
   MAP8V all-FMA ladder, ZT0/ZTC sweep-B prefetch, pro/exec path).  MAPDIV
   re-raced post-inline: all-FMA still wins (692/712/719 vs MAPDIV=1
   739/751/760, MAPDIV=2 749/767/737 -- same-window reps).

### Built, raced, and DECLINED, with the numbers that killed them

* **ZMS output-side map fusion + cross-step z-fusion** (adopted from rival
  v6_3f30d81f's L=64: "map fused into the x-pass stores, the next
  iteration's z-pass pipelined behind each completed row group"; their
  KSTORE/mapv read in source).  In custody layout the fusion is legal at
  ky-slab grain: after sweep B finishes slab ky, its 64 z-rows of step s+1
  are complete and L1/L2-hot, and the y-lines become their own volume sweep.
  Implemented BOTH variants (ZMS=1 map-in-x-stores, ZMS=2 x-lines raw +
  map-fused-z-rows on the hot slab), both bit-identical to the lazy-map
  scheme.  Node: **ZMS=0 624/624 vs ZMS=1 686/687 vs ZMS=2 712/712**
  (post-inline, same window).  Two mechanisms, both measured:
  (a) consuming plane-major custody c per ky-slab = 64 scattered 1-KB rows
  at 70-KB stride, priced at ~6 GB/s effective (bz-zmap 1.32M cyc); a
  slab-major c permute (zs_build_c_slab) fixed the pattern and recovered
  almost nothing (1.26M) -- which exposed that the row body was not
  memory-bound at all (that was the inline bug, item 1 above);
  (b) post-inline, the separated phases lose the old scheme's free overlap:
  sweep A's y-lines fill the FMA slots the map ladder's serial chains leave
  idle, and a pure map+z phase has no independent work to hide under.
  The fusion machinery is kept env/-D-forcible (FFT64B_ZMS=1|2, slab-major
  c auto-selected) as raced controls.
* **ZVP vol-pair row interleaving** (my r6 next-item 3): sweep A alternates
  two volumes' map+z rows to double demand-miss MLP; pair arena 17.8 MB.
  Node: **706.1/709.1 (on) vs 611.2/612.3 (off)**, sd 0.06% -- a 16% LOSS.
  Post-inline the row body is not latency-starved enough to pay for
  quadrupling the row phase's hot footprint (2 planes + 2 c planes = 272 KB
  through a 48-KB L1) and doubling the chain working set.  Kept as a knob
  (FFT64B_ZVP), default 0.
* **ZYW y-phase next-plane warmup** (r6 next-item 1, the ZAW coverage hole
  attacked from the safe side: prefetch plane x+1's first ZAD rows DURING
  plane x's y-lines, after which the plane is dead): a wash -- 615.2/617.7,
  690.5/697.5, 697.7/693.3 (off/on pairs).  The ZAPF cursor at distance 2
  plus the L2 streamer already cover the seam.  Kept as a knob, default 0
  (simplest-wins).

### Operation count

Shipped arithmetic unchanged from r5/r6: ~2.15M FMA-port vector ops + ~197K
shuffles per step-vol, map 15 ops/8 pts, traffic ~22.3 MB/step all-L3.  What
changed is retirement overhead: the inline fix deletes 4096 x (16 spill
stores + 16 reloads + call/ret/canary) = ~140K wasted uops per step-vol.
Post-inline profile (quiet window): sweepA 1.31M cyc (was 1.47M r6), sweepB
460K, total 1.79M cyc/step-vol.  Sweep A's remaining slack over its ~850K
compute floor is now genuinely schedule/latency, not call overhead.

### Measured on the node (a80n0, leased cores; same-window MKL pairs;
### final shipped build, quiet windows)

| config | us/step-vol (min) | same-window MKL | ratio |
|---|---|---|---|
| graded B=2 | **607.1** (median 607.2, sd 0.05%) | 1717.5 | **0.353** |
| B=1 | **606.2** (sd 0.01%) | 1723.7 | 0.352 |
| B=8 | **608.7** (sd 0.08%) | 2057.9 | 0.296 |

vs the r6 board (638.8): **-5%**.  vs the best rival L=64 on this node
(v6_3f30d81f, 653 us, gate-passing): **-7%** -- the 2% margin is now 7%.
Correctness, final build: single-transform rel_l2 4.464e-16 at B=1/2/8;
map-chain end state **1.690e-12 (B=2) / 1.401e-12 (B=1) / 1.790e-12 (B=8)
vs tol 1.34e-11** (~8x margin, ladder untouched); B=2 chain end state
bit-identical across independent runs; every raced variant above
cmp-identical to the shipped output (one bit class held through the whole
round).  Setup 0.05 s.

### Borrowed / attribution

* The ZMS fusion shape raced this round: rival **v6_3f30d81f** (README +
  implementation.c KSTORE/mapv), via the new `fft_v5v6_solutions/` corpus.
  Declined here, but reading their source is what triggered the asm audit
  that found the inline bug -- the round's actual win is downstream of
  their code even though their technique lost.
* The honest 653-us target: `results/rivals_icelake/` re-benchmarks.
* The measure-don't-guess discipline (asm spill count, per-phase rdtsc
  before believing any hypothesis): corpus 10 0 and this entry's own r6
  lesson, applied twice more.
* Everything structural carried from r4-r6 with attributions recorded there.

### Next

1. Sweep A is 1.31M cyc vs a ~850K floor; the slack is now real scheduling
   (the map ladder's serial chains vs the z-line's port pressure).  The one
   untried shape: software-pipeline the map ONE ROW AHEAD of the z-line
   inside the same sweep (rivals' iter_z64 does this) -- post-inline the
   compiler finally has the whole body in one scheduling region, so it
   could work now where the phase-level fusion (ZMS) failed.  Estimated
   ceiling ~50-80 us/step-vol.
2. PMU counters (perf_event_open, still never used at L=64): decompose
   sweep A's 460K slack cycles into port-5 pressure vs load-latency stalls
   before writing any more prefetch code.
3. If a future toolchain bump lands (gcc 13.2 is in the corpus notes):
   re-audit EVERY always_inline/spill decision; the r5-r6 numbers in this
   file all carry the un-inlined ztail and undersell the structure by ~4%.
4. tryout.sh chain-check plumbing STILL broken (fourth round; same two
   bugs).  Also note for wallaby dev loops: slurm client tools live at
   /opt/software/slurm-19.05.8.1-cuda-11.8/bin -- reserve.sh --status needs
   that on PATH or it false-negatives the live reservation.
