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

**Next round (written in r4):** (a) the gap to the rivals' 843 is structural, and we now OWN the
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

## Round ice_r5

**Standing at round start.** ice_r4 board: L64_radix8 1043.0 µs/xf, the board lead
(L64_blocked 1048.6, MKL 1723.2, we're 1.65× ahead of the best library). Rivals'
full-task target still 843 µs/step. My own r4 "next" note (a) and L64_blocked's r4
item 3 both pointed at the same lever: the chain's intermediate states are OURS, so
they need not be ABI-interleaved — and, one step further, their ADDRESSES are ours too.

**What changed: split-complex, transposed-SEQUENTIAL inter-step states (`ckind=1`).**

1. **Steps 1..m-1 store the mapped state as raw split vectors** (no ILO/IHI
   interleave at pass 23's stores) **and the next step's pass 1 loads split pairs
   directly** (no IEV/IOD deinterleave; `PASS1S` twins are pass 1 minus the two
   permutes per vector pair). Kills 65536 + 65536 = **131072 port-5 shuffle uops per
   step-volume** — the same count L64_blocked's r4 split-state item removed in their
   structure (worth −1.4% there), and port 5 hosts an FMA pipe on ICX.
2. **Emission is transposed-sequential**: row (outer ky, inner kx) goes to
   `(ky*64+kx)*128` — pass 23's output becomes ONE sequential store stream instead of
   the 64 concurrent 1-KB-per-64-KB streams my ice_r2 record named as the remaining
   fused-structure residual (corpus §10's "cut concurrent store streams" large-L
   lever). The trick that makes this free: the transposed state is just the same
   volume with x and y swapped, and the 3D FFT is axis-symmetric, so the SAME code
   runs every step — the axes swap parity per step and nobody cares except c.
3. **Two csplit images**: c's consumption order alternates per step parity (odd steps
   `(ky,kx)` row order = the r4 image; even steps transposed, `build_csplit_t`). Both
   built once per chain in the untimed warmup; the ping mapping grew to 4 regions
   (2 states + 2 images, 4·B·VOLD doubles, hugepage-backed as before).
4. **The TR8 SW lane residue is a FIXED POINT of the split state** (layout
   z = 8·slot + SW(lane) in, same form out), at zero op cost: the SW-input z-line
   uses an SW-permuted lane-twiddle table (`tw3r_sw[k2][m] = tw3r[k2][SW(m)]`) and
   SW-reordered register NAMES for the final radix-8's arguments and store slots
   (`r0,r1,r4,r5,r2,r3,r6,r7` — output k1=j lands in register SW(j), and the
   emission/c-add just use that name at position j). Pure token renaming, zero
   instructions. Step 1 (interleaved ABI input, natural lanes from pass 1's
   deinterleave) uses the natural-name twin into split emission; both variants end
   with lanes k2=SW(m), which EMIT_SPLIT stores raw and the final step's ILO/IHI
   absorb exactly as pass 3 always has.
5. **Step m interleaves to ABI `final_out` with parity-dependent addressing**: m odd
   = the r4 pattern (`out + kx*8192 + ky*128`); m even = `out + ky*8192 + kx*128`,
   which is per-plane sequential (the graded m=134 is even). m=1 has no split
   intermediate and falls back to the whole r4 step (`chain_run_fused`).
6. **Tuner rewired to the new unit**: the fused grid rows time split-seq BULK steps
   (132/134 of the graded chain), alternating the two c images exactly as the chain
   does; a final **ckind A/B** (r4 interleaved chain vs r5 split-seq, on the settled
   pick) guards the whole change. The xb and fout create-time A/Bs are retired (never
   won: r6/r11/ice_r2/ice_r4; env forcing and the exec-path flags remain). Env
   forcing now runs BEFORE the chain verification, so `chain_ok` gates the config
   that will actually run; new env `FFT64R_CKIND=0|1`.
7. **Verification hardened**: `chain_ok` now checks the REAL dispatcher at m=1 (r4
   step), m=3 (first/bulk/odd-final, both c images) and m=4 (adds the even-parity
   final) against the exact scalar chain, bar 1e-12 at every m. Measured worst
   2.183e-13 (the map-tier delta over 4 steps; the old 3-step check read 1.96e-14).

**Op count.** FFT arithmetic unchanged (52-instr radix-8 codelet everywhere, 90
nominal flops/point, map ~21 vector uops/8pt as in r4). Bulk steps: −131072 port-5
shuffles/step-vol (−40% of the step's shuffle mass; z-line per-row shuffles 64→48,
pass 1 stage-1 permutes 128→0 per (x,zb)). +0 arithmetic anywhere. One-time c
repermute cost doubled (two images).

**Measured (a80n0 leased cores via tryout.sh + manual check.py, graded chain m=134,
--map; the tryout $W bug worked around by pre-seeding W with the REAL build path —
this fixes the driver's --cin so the graded map chain runs; check.py still needs the
manual invocation).** Decision pair, same window, on the settled pick:
**ckind A/B r4-ilv 1175.7 vs r5-splitseq 1146.6 µs/step → −2.5%, ckind=1 picked.**
Headline windows: **B=2: 1044.7 min / 1053.0 median vs MKL 1913.8 same-window
(1.83×)**; B=1: 907.7 vs 1800.2 (1.98×); B=8: 1218.2 vs 2085.4 (1.71×). Tuner pick
fused-plain+slabpf1+propf1+scst0 at B=2. Correctness: single rel_l2 4.46e-16;
**map-chain end state 4.12e-14 at B=2 / 3.83e-14 at B=1 / 5.34e-14 at B=8 (budget
1.34e-11, 326× margin)**; repeatable bit-identical across independent runs (B=8
double-run cmp). Setup 0.42–1.6 s. Cross-window absolutes remain worthless (own
B=2 build read 1044.7 and 1173.3 in two windows the same hour); only same-window
pairs were used for decisions.

**Tried / observed, with numbers:** NT final stores flipped AGAIN under split-seq
emission (mode=1 1198.8 vs plain 1148.7 in-grid; r2 plain → r4 NT → r5 plain): with
one sequential store stream the RFO cost NT was dodging is prefetch-friendly and the
next step's pass 1 re-reads the state, so plain wins — third confirmation that the
store-mode verdict is a regime property, never carry it. scst plain again (1142.6 /
pfw 1166.1 / nt 1330.0). propf on (1144.7 / 1142.7 — inside noise, strict-win kept
it). tiled/zfirst rows remain uncompetitive under the map chain (1432–1686).

**Borrowed:** the split-state idea from **L64_blocked ice_r4** (their item 3, −1.4%),
extended here in two ways they couldn't use in their lazy-map structure: the
transposed-sequential emission (their map lives in pass 1, mine at pass 23's stores,
so I get to choose the output addresses too) and the SW-residue fixed point (their
z-pass is not lane-transposed). The axis-symmetry/alternating-parity trick and the
two-image c handling: this entry. Tuner protocol carried from my ice_r2
(L17_matrixsimd lineage); tryout W= workaround from L17_matrixsimd/L64_blocked ice_r4.

**Next round:** (a) PMU pass-level cycle attribution has STILL never been run
(perf_event_open works on the node; third round I write this — do it first). (b) The
remaining z-line shuffle mass is the TR8 pair itself (48/row, ~196K/step-vol): a
corpus-§10 z-split custody layout would delete the per-step transposes entirely, but
zfirst lost twice here — it would need its own store-side-map pass and the split-seq
custody idea folded in before it's worth re-racing. (c) The rivals' 843 is still
ahead; the biggest untried traffic cut is fusing step s+1's pass 1 into step s's
pass 23 per ky-slab (the state row is L1-hot at emission time) — deletes a full
4-MB/vol read sweep, but the y-FFT needs all 64 ky rows of an (x, zb) group, so it
requires an 8-row staging buffer per x-plane and a different emission order; sketch
before committing. (d) The cheaper cubic-Newton map ladder (L64_blocked's, ~4e-13/app)
would save ~4 uops/8pt but drops the chain margin 326×→~8×; not worth it while the
roofline says the binding port is shuffles/adds, not FMA.

## Round ice_r6

**Standing at round start.** ice_r5 board: L64_blocked took the geometry at 639.5
µs/xf with the z-split custody chain (their record: per-volume cache residency, two
in-place sweeps per step, ~667 µs dev windows); me 1011.1, MKL 1721.3. Their "Next"
item 1 declares the last fusion — "sweep B of step s with sweep A of step s+1" —
impossible: "the orthogonal blocking (ky-slabs vs x-planes) forbids it... I do not
see a legal fusion; treat ~640 us as this structure's floor on this node."

**What changed: the FUSED-BOUNDARY CUSTODY CHAIN (ckind=2).** Their impossibility
claim assumes a fixed sweep orientation. My r5 axis-alternation trick (the 3D FFT is
axis-symmetric, so the roles of x and y may swap per step) makes the fusion legal:

1. **Custody state** (adopted from L64_blocked ice_r5, itself from rival 1000f989 /
   corpus §10 §3): the state lives for all m steps in ONE buffer with the SC strides
   — slot (a,b,g) at a*SCXS + b*SCKS + g*16, lanes = 8 z per slot — one volume
   through all m steps cache-resident. My lane convention is the r5 fixed point
   (index 8g + SW(m)), so the SW-form z-line (tw3*_sw + renamed registers, zero
   extra ops) maps custody form to custody form and my existing kernels transfer.
2. **One sweep per step, plane orientation alternating x/y.** The axis that
   finishes step s is always in-plane: per 68-KB L2-hot plane, finish U(s) lines →
   per 1-KB pencil: +c, map, SW z-line of s+1 register-resident, store back in
   place → U(s+1) lines. (U = y on x-planes, x on y-planes; z is in every plane.)
   Chain = opening (x-planes: ZPENCIL nat z-line + Fy) + m−1 boundary sweeps +
   closing (finish U(m), +c, map, ILO/IHI interleave straight to final_out; m even
   = x-planes = contiguous output rows, and the graded m=134 is even). m=1 works
   (opening + closing-y), no special-casing.
3. **Traffic: 13.4 MB/step** (state R+W + c R), vs 22.3 for the unfused custody
   scheme — the fusion deletes one full state R+W per step. ONE custody c image
   serves both parities because the layout is fixed in ABI coordinates and only the
   sweep ORDER alternates (unlike my r5 transposed emission, which needed two);
   rebuilt per volume per call = 1/m of the c traffic. At the measured 672 µs the
   chain moves only ~20 GB/s of L3 traffic — this structure is now
   ARITHMETIC/latency-bound, not traffic-bound (the ~33 GB/s ceiling no longer
   binds; L64_blocked's roofline note inverts here).
4. **Map ladder switched to L64_blocked ice_r4's 15-op cubic tier** (rsqrt14 + one
   cubic Newton, den = fma(m2,q,1), rcp14 + one cubic Newton; 1e-300 bias): raced
   against my r4/r5 exact ladder 3/3 same-window alternations, 650.4-657.1 vs
   668.6-680.1 µs/step (−2.8%). ~4e-13/application; measured end drift at m=134:
   1.424e-12 (B=1) / 1.692e-12 (B=2) / 1.951e-12 (B=8) vs budget 1.34e-11 — 7.9x
   margin, the tier the brief blesses at L=64. Exact ladder kept behind
   -DFFT64R_MAPEXACT (mandatory if this code is ever pointed at L=6/8/13 chain
   lengths); the create-time gate is CHAINBAR = 4e-12 for the cubic tier, 1e-12
   exact.
5. **Tuner + gates:** ckind race is now 3-way on the settled pick (r4-ilv 1167.6 /
   r5-splitseq 1138.1 / r6-custody 733.5 in-tuner → ckind=2); chain verify runs the
   REAL dispatcher at m=1,2,3,4 (both boundary parities, both closing parities)
   against the exact scalar chain and degrades ckind 2→1→0 on any failure before
   falling to the exact path. Verified worst 5.094e-13 (fast ladder) / 1.683e-13
   (exact). r4/r5 chains kept intact as env-forcible fallbacks (FFT64R_CKIND=0|1|2).

**Op count.** Per step-volume: 1024 line-FFT64s (finish 512 + start 512, ~1028
FMA-port ops each) + 4096 pencils (z-line 132 arith + 48 shuffles; map 15 ops + 2
c-adds per 8 pts) ≈ 2.15M FMA-port + ~197K shuffles — same arithmetic as
L64_blocked's unfused custody (3 axes + m maps either way); the entire win is
traffic and pass structure. Ends amortized: copen 435 + cclose 518 + ccust build
272 µs ≈ 9 µs/step at m=134 (included in all numbers).

**Measured (a80n0 leased cores via tryout.sh + manual check.py, graded m=134 map
chain, same-window MKL pairs; the $W tryout bug worked around as in r4/r5).**
**B=2 (graded): 672.0 min / 672.3 median µs/step, sd 0.08%, vs MKL 1771.9 (2.64x).
B=1: 672.1 vs 1809.2 (2.69x). B=8: 672.8 vs 2062.2 (3.07x)** — batch-flat by
construction (per-volume residency). Exact-ladder build read 686.0-702.2 across
three windows earlier the same day. vs the rivals' 843 target: −20%; vs
L64_blocked's dev floor 660-667: below it, same-day windows (cross-window absolutes
remain worthless — the scored quiet window decides). Correctness: single-transform
rel_l2 4.46e-16 at B=1/2/8; map-chain end state PASS at all three batches (above);
bit-identical output+end-state across independent runs; setup 0.43-1.5 s.

**Stage attribution (standalone harness, build/tryout/L64_radix8/l64bench.c — the
per-pass profiling my record demanded three rounds running, done via clock_gettime
stage splits instead of PMU):** steady sweep ≈ 654-689 µs/window-dependent; lines
(stages 1+3, 1024 line-FFTs) ≈ 305 µs; in-sweep pencil stage ≈ 350-385 µs; parities
within ~5% of each other (bsweep_x 654, bsweep_y 689).

**Tried and rejected, with the killing numbers:**
- **4K-aliasing theory of the y-parity pencil cost — REFUTED.** Isolated pencil
  probes read penc_y 519 vs penc_x 367 µs, and both (SCXS mod 4096 = 64 B → next
  pencil's loads alias prior stores) and (ccust page-aligned against state → every
  c load aliases a state store) looked damning on paper. A same-window matrix of
  (SCXPAD, CCOFF) = (8,0)/(8,320)/(136,0)/(136,320)/(264,1344): steady-step 671.5 /
  671.4 / 690.6 / 686.9 / 696.8 — the plain baseline wins, pads only lose. The
  probe asymmetry was the probes themselves streaming scattered pencils from L3;
  in the real fused sweep stage 1 has already pulled the plane into L2. CCOFF
  stays 0, SCXPAD stays 8 (both remain -D-sweepable).
- Dropping the c-pencil T0 prefetch (CPFROW, v+2): penc_y 532.9 vs 519.2, penc_x
  387.8 vs 367.5 — prefetch kept (short-range T0 earns its keep, L64_blocked's
  ZT0 lesson transfers).
- No zmm spills in either boundary sweep (objdump-verified, 0 rsp-relative vector
  moves) — the r5-lineage register budget survives the added map temps.

**Borrowed:** custody layout, per-volume chain residency, materialize-only-at-ends,
and the 15-op cubic map ladder from **L64_blocked ice_r4/r5** (1000f989 / corpus
§10 lineage); the SW-residue fixed point, axis-alternation, and the boundary fusion
built on it: this entry (r5 machinery, executed against their r5 "no legal fusion"
claim). Tuner protocol carried from my ice_r2 (L17_matrixsimd lineage); tryout W=
workaround + manual check.py from L17_matrixsimd/L64_blocked ice_r4.

**Next round:** (a) The structure is no longer traffic-bound (~20 GB/s vs the ~33
ceiling): the binding cost is now the line-FFT stages (305 µs for 1024 lines ≈ 1.5
uops/cyc — dependency/issue-bound in the two-stage lb bounce) and the pencil chain
(map+zline dependency depth). Attack the lines first: the lb round-trip per line is
the largest single component; a fused two-line software-pipeline (interleave line g
stage-2 with line g+1 stage-1) could hide the lb store-forward latency. (b) True
PMU attribution (perf_event_open) is STILL undone — the harness stage splits
substitute for now, but port-pressure numbers would say whether the z-line's 48
shuffles/pencil (port 5) cap the pencil stage. (c) If a future round tightens the
per-step error budget below ~3e-13, flip -DFFT64R_MAPEXACT and eat the 2.8%. (d)
The closing sweep at m odd writes 64-KB-strided output rows; if a future graded m
is odd, consider a y-plane-ordered final interleave through a bounce buffer.
