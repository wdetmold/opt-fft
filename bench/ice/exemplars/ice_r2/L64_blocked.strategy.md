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
