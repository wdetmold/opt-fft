# L64_radix8 — strategy record (ice panel)

No earlier ice-panel rounds are recorded here: the ice_r1 L=64 workers were among the
15/19 agents killed by the launch crash-storm (see `results/ice_r1/VERDICT.md` §3), so
the code scored in ice_r1 was untouched multicore-panel `panel_r11` work. The pre-ice
history (rounds panel_r6..r11: the split-complex layout, odd-line padding, the tiled
structure, slabpf/propf/scst/xb/fout twins) lives in the header comment of
`impl/L64_radix8.c` and in the multicore panel's records.

## Round ice_r2

**Standing at round start.** The board's ONE loss: 1184.0 µs/xform vs mkl_dfti 1016.4
(MKL 1.16× ahead) at the graded case (B=2, chain m=134, unitary). Same-day dev
baseline on the node: 1246.7 vs MKL 1081.4 on a leased core.

**What changed (in order of measured value):**

1. **Chain-shaped, clock-settled, per-candidate-warmed tuner.** Adopted from
   `L17_matrixsimd`'s ice_r1 round (the r1 verdict names it the round's one validated
   change and says port it everywhere). The graded workload ping-pongs dst→src through
   m=134 steps with a driver-side unitary sweep (full read+write of dst) after every
   execute; the old tuner timed `exec(ti→to)` in place, so dst was never re-read and
   the store-mode ranking came out INVERTED. The tuner now times units of
   `{exec(a→b); sweep b; exec(b→a); sweep a}` (sweep scale = true 1/√V so values stay
   O(1) over hundreds of units), after a ~40 ms 512-bit settle spin.
   - This exposed the round's decisive fact: **NT final stores — the r1 pick — lose
     ~15% under the chain** (warmed chain units, same window: fused-plain+slabpf1
     1021.9 µs/xf vs fused-nt+slabpf1 1260.9). Plain stores keep the output L3-hot for
     the driver's sweep and the next step's pass 1; NT forces an 8 MB DRAM round trip
     per step that MKL does not pay.
   - **Per-candidate warm-up is load-bearing, not hygiene:** the chain-shaped tuner
     WITHOUT it still misranked, because each candidate was timed against the previous
     candidate's cache/NT residue — zfirst-plain read 1598 µs/xf in-tuner vs 1149
     (1049 + ~100 sweep) in the real chain, a 30% error that flipped the tournament.
     One warm unit before each timed unit fixed the tables to ±0.2%.
   - Acceptance bar lowered 2% → 0.5%: the warmed protocol resolves ~0.2%, and the old
     bar was measured discarding a real 1.2% slabpf win (1021.9 rejected for 1034.0).
2. **New `zfirst` structure (sstruct=2)** — the z-split L=64 layout of corpus §10,
   taken from `ext/reference/fft_v4_solutions/1000f989`'s `run64_zsplit` and adapted to
   our ABI (they could keep the volume in split layout between chain steps; our
   contract materializes interleaved output every call). z-axis FFT FIRST, straight
   out of the contiguous input pencils (deinterleave fused into loads, the existing
   register-resident z-line codelet, raw split vectors stored sequentially to SC);
   pass X = `xline_sc` verbatim; pass Y per kx-plane through a 66-KB lb2 buffer
   (129-line odd v-stride) emitting straight to the interleaved output — SC is
   read-only in pass Y (no second dirty/writeback sweep) and the output is one
   near-sequential stream per plane instead of 64 concurrent 1-KB-per-64-KB row
   streams. Correct (rel_l2 4.2e-16 forced via env). **It loses to retuned fused by
   ~10%** (in-tuner same window: zfirst-plain 1134.7 vs fused-plain+slabpf1 1021.9;
   forced real chain 1048.9 vs 967.9): the extra full SC read sweep from L3 costs more
   than the sequential in/out streams save. Kept in the tournament — the margin is
   regime-dependent and it beat MKL outright (1049 vs 1081) before the fused retune
   landed.
3. **`zx` variant (sstruct=3) — refuted, kept env-only.** Fuses the z-pencils y-major
   with the x-FFT per y-row (row's 64 KB still L2-hot) to kill zfirst's extra L3
   round trip. Measured 1336.8 (plain) / 1617.8 (nt) / 1382.3 (pfw) vs fused 1120.2
   in the same window: reading input pencils at 64-KB stride (64 concurrent 16-line
   read streams) costs far more than the SC sweep it saves. Number that killed it:
   **+19% vs fused, +18% vs plain zfirst.**
4. Tuner batch clamp 8 → 4 (chain units are 2 execs + 2 sweeps; bt=8 setup was 3.4 s,
   bt=4 is 1.8 s, and a 36-MB bt=4 ping-pong set is already past L3 so the streaming
   regime is still what gets tuned). Graded B=2 unaffected.

**Op count.** Unchanged: the 52-instr/56-flop radix-8 codelet everywhere, 90 nominal
flops/point over three axes (23.6 Mflop/volume as the driver counts it). This round
was entirely schedule/protocol; zfirst adds no arithmetic (same codelets, the ILO/IHI
interleave absorbs the TR8 SW residue at output exactly as pass 3 always did).

**Measured (a80n0 leased core via tryout.sh, graded chain m=134, unitary; MKL from the
same window each time).** Quiet windows: **B=2: 967.9–1020.7 µs/xform vs mkl_dfti
1018.3–1086.0 — we lead 1.05–1.06×** (r1: 1.16× behind; this was the board's one
loss). B=1: 986.1 vs 1043.7 (1.058×). B=8: 1208.2 vs 1627.9 (1.35×). rel_l2 4.46e-16,
chain check 2.89e-14, repeatable. Noisy windows (other leases active) degrade both
entries but MKL worse (1183 vs 1415 in the worst one). Setup 0.7–0.9 s at B=2.

**Tried and rejected, with the killing number:** NT final stores under chain (1260.9
vs 1021.9); zfirst as pick (1134.7 vs 1021.9); zfirst-NT (1443.0); zx (1336.8 vs
1120.2); passZ/pass-1 SC store pfw / NT (1153.2 / 1302.7 vs 1131.6 plain); xb compact
x-buffer (1181.7 vs 1143.0, r6/r11 verdict again); fout FMA-fed stores (tie, window-
dependent, 1279.9 vs 1283.5 then 1181.0 vs 1170.2 — the tuner decides per run);
the unwarmed tuner protocol itself (30% candidate misranking, see above).

**Borrowed:** chain-shaped tuner stage + clock-settle spin from `L17_matrixsimd`
(ice_r1); z-split z-first layout from rival pipeline `1000f989` (corpus §10, via
`ext/reference/fft_v4_solutions/`); the per-candidate warm-up is the corpus §10
"interleaved best-of-N" timing lesson applied to an in-process tuner.

**Next round:** (a) the remaining fused-structure residual is the pass-23 output
pattern (64 row streams at 64-KB stride) — zfirst proved sequential output + plain
stores is the right direction but paid an extra sweep for it; a passB-style kx-major
z-line emission from 8 compact slab buffers could get sequential output without the
extra SC sweep. (b) The graded ratio moved 1.16×-behind → 1.06×-ahead entirely on
protocol; nobody has yet profiled the fused chain with the PMU (`perf_event_open`
works on this node) — pass-level cycle attribution would say whether pass 1's RFOs or
pass 23's stores bind. (c) If the scoring window shows plan instability across runs,
consider majority-vote tuning (3 picks, take the mode) — the pick flips between
plain/pfw+slabpf0/1 at the ~1% level from window noise.
