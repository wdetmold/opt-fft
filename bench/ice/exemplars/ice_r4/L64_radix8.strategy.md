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

## Round ice_r4

**Task change, and standing at round start.** The graded step is now the rivals' full
step, `state <- (FFT(state)+c)/(1+|FFT(state)+c|)`, and the driver detects an exported
`fft3d_chain` (weak symbol) that owns the whole m-step chain; entries without it pay a
driver-side unfused map pass (the brief's "2.24 s configuration" — the map is the
battleground). ice_r3 board (FFT-only chain): L64_blocked 902.9, me 949.5, MKL 1020.0.
Rivals' full-task time to beat at L=64: 0.226 s = **843 µs/step** at the graded case
(B=2, m=134).

**What changed (in order of measured value):**

1. **`fft3d_chain`, map fused AT THE STORE SIDE of pass 2+3.** The z-line codelet's
   final radix-8 outputs ARE the finished z values, still in split (re,im) vector form,
   one instruction before the interleave+store — so the map twins of pass 23
   (`pass23m_*`) add c there, apply the map, and store the mapped STATE instead of raw
   z. Consequences: every chain step is identical (no cold un-mapped first step, no
   epilogue sweep — step m writes `final_out` directly and only that one step per chain
   touches the driver's 4-KB-paged buffer; steps 1..m-1 ping-pong two hugepage state
   buffers), pass 1 is completely stock, and the buffers hold bounded O(1) states.
   I deliberately did NOT copy the rivals' lazy load-side map: L17_matrixsimd's ice_r4
   record measured it LOSING at their cell (v4 16.48 vs 13.26 µs/step) because the
   map's ~60-cycle chain lands in front of the next pass's critical path. On the store
   side it hides behind store retirement.
2. **c repermuted once per chain into pass-23's exact consumption form**
   (`build_csplit`): split vectors in (ky,kx) row order — ONE sequential read stream —
   with the z-line's TR8 SW lane residue folded into the deinterleave tables
   (CEV[m] = 2·SW(m)), so the `+c` costs zero shuffles inside the hot loop. Built in
   the untimed warmup (driver reuses one c pointer; rebuilt only if it changes).
3. **Map kernel: rsqrt14 seed + 2 Newton for sqrt(m2), then ALL-FMA reciprocal
   (rcp14 + 2 Newton) — no divider.** The rival-consensus/L64_blocked shape uses one
   vdivpd per 8 points; measured HERE the FMA form won **5 of 6 same-window pairs**
   (quiet pair 1048.5 vs 1061.7 µs/step; settled pairs ~1102 vs ~1123, +1.9%). At this
   fusion site the z-line binds shuffles/adds and the FMA pipes have headroom, while
   16–18 cyc/vector of divider throughput (L17's mapbench number) does not fully hide
   once the map joins pass 23. vdivpd kept as `-DFFT64R_MAPDIV` (one bit class per
   build). Split layout makes the pair-shared denominator free — the 8 denominators
   are already one vector, no deinterleave (L17's s6 trick, by construction).
   1e-300 bias for m2=0 safety; ~2 ulp per application, exact tier.
4. **Tuner retimed on the NEW graded unit**: fused rows time real
   `chain_step_fused` ping-pong pairs (map at stores, stand-in 0.1-scale c + its
   csplit image); tiled/zfirst rows time what fft3d_chain would actually give them
   (execute + one vectorized map sweep). Same candidate grid, same warmed best-of-3
   protocol. **The ice_r2 NT verdict INVERTED under the fused map chain**: the tuner
   now picks fused-**nt**+slabpf1 (1090.3 vs plain 1130.3 in-tuner) — pass 23m writes
   each state exactly once, the B=2 working set (2 states 8 MB + csplit 8 MB + SC
   4.5 MB ≈ 28 MB) is past the 22-MB L3 anyway, so skipping the RFO wins. The r2
   lesson generalizes: NT is not "bad under chains", it is bad when something re-reads
   the lines while they could still be cached — retune per regime, never carry the
   verdict.
5. **Create-time chain verification with safe fallback** (`chain_ok`, borrowed from
   L64_blocked's interlock): 3 steps at nvol=1, the REAL fft3d_chain code path vs an
   exact reference (execute + scalar libm map), bar rel_l2 < 1e-12; measured 1.96e-14.
   Failure or `FFT64R_NOCHAIN=1` degrades to execute + exact scalar map per step —
   slow, never wrong (a fast entry that drifts past the budget is a rejected entry).

**Op count.** FFT unchanged (52-instr radix-8 codelet everywhere, 90 nominal
flops/point over three axes). Map adds, per 8 complex points: 2 loads + 2 adds (+c),
2 FMA (m2), 1 vrsqrt14pd + 6 FMA-class (2 sqrt-Newtons), 1 FMA (den), 1 vrcp14pd +
4 FMA (2 recip-Newtons), 2 muls out ≈ 21 vector uops, zero shuffles, zero divider ops
= ~0.69M extra vector uops/volume on top of the FFT's ~1.6M.

**Measured (a80n0 leased core via tryout, graded chain m=134, --map; MKL = the
driver-fallback configuration, same window).** Shipped build: **B=2: 1154.2 µs/step vs
MKL 1769.7 (1.53×)** — same code read 1048.5–1160 across windows, quiet-window floor
~1050; B=1: 971.4 vs 1739.8 (1.79×); B=8: 1176.5 vs 2019.1 (1.72×). Correctness:
single rel_l2 4.46e-16; **map-chain end state 3.91e-14 at B=2 (budget 1.34e-11, 340×
margin)**, 3.98e-14 at B=1, 5.37e-14 at B=8; two independent processes bit-identical
on both single and .chain outputs. Setup 0.45–1.6 s. Versus the round's target: rivals
843 µs/step — not caught yet, but the unfused configuration this round replaced would
have sat near MKL's ~1770.

**Tried / observed, with numbers:** vdivpd map as default (lost 5/6 same-window pairs,
see above; kept as -D twin); NT final stores REJECTED in r2, now WIN (1090.3 vs 1130.3
— regime change, not noise); tiled and zfirst under the new unit are not close
(1395–1426 / 1494–1555 vs fused 1090 — an unfused map sweep costs ~300 µs/step at this
size, so only the fused structure is competitive until zfirst gets its own pass23m);
xb (1067.6 vs 1151.2, rejected again), fout (1064.3 vs 1068.8, off), scst (plain
1086.6 / pfw 1109.3 / nt 1340.6 — plain again), propf on (1074.7 vs 1081.2).
Cross-window absolutes at this geometry remain worthless (own code 1048–1339 in one
afternoon); only same-window pairs were used for decisions (L64_blocked's standing
rule, reconfirmed).

**Borrowed:** map kernel shape (rsqrt14+2N, 1e-300 bias, split-form MAP8V) from
**L64_blocked ice_r4**, which took it from corpus §10 §2 / the 1.00-scorer's mapc;
the store-side (vs load-side lazy) placement decision, the divider-throughput and
vrsqrt14pd≈2.3cyc facts, and the tryout.sh `W=` workaround + manual check.py
invocation from **L17_matrixsimd ice_r4**; the create-time chain_ok interlock from
**L64_blocked ice_r4**; chain-shaped warmed tuner protocol carried from my ice_r2
(itself from L17_matrixsimd ice_r1).

**Next round:** (a) the gap to the rivals' 843 is structural, and we now OWN the
chain: intermediate states need not be ABI-interleaved, so a split-complex inter-step
layout would delete pass 1's deinterleave and pass 23's interleave shuffles AND free
the inter-step buffer layout so pass-23 emission can be sequential (the corpus §10
z-split custody idea, finally applicable because the contract no longer forces
interleaved materialization between steps — only final_out, one step per chain, must
interleave). (b) PMU pass-level cycle attribution has STILL never been run
(perf_event_open works on the node); before any relayout, price pass 1 vs pass 23m in
cycles. (c) The plain/NT pick sits ~3.5% apart and flips with window noise —
majority-vote tuning if the monitor reports plan instability. (d) If anyone ports
pass23m to zfirst: build csplit in (kx, k2, k1) order and map on the rb bounce rows.
