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

## Round ice_r7

**Standing at round start.** ice_r6 board: L64_blocked leads the cell at 638.8
µs/step (their custody chain + ZAPF prefetch), me 660.3, MKL 1720.6. The round's
mission is explicitly cumulative: mine the rival campaigns' sources and each
other's records. Best gate-passing rival at L=64 re-benchmarked on THIS node
(`results/rivals_icelake/rivals.json`): v6_3f30d81f at 0.1751 s = ~653 µs/step —
between us two. My r6 attribution: pencil stage (map+z-line) 350–385 µs, line
stages ~305 µs of a ~672 µs step, both with ~2× headroom over their port floors —
latency-bound, so this round was software pipelining, no arithmetic or layout
change, one bit class throughout.

**What changed (shipped):**

1. **MAPPIPE — the map-ahead pencil pipeline** (the round's win, −5%). Taken
   straight from rival **1000f989**'s `iter_z64` (`map_row(x+1)` to memory, then
   `zpass(x)` from memory), which **L64_blocked's ice_r6 record names as its
   untried "Next" item 2** — executed here in my fused-boundary custody sweep.
   `ZPMAP_SW` splits into `PMAP_ROW` (load state row, +c, MAP8, store back in
   place) and `ZLINE_SW` (the SW-form z-line on the already-mapped row); the loop
   body becomes {map row v+1; z-line row v}. Mechanism: the fused pencil was one
   ~320-uop serial chain — about one ROB — so the rsqrt/rcp ladder's latency never
   overlapped the z-line's shuffle mass across pencils; split, the body holds two
   independent streams with complementary port mixes (map = FMA-class, z-line =
   shuffle/add-heavy). Cost: 16 extra L1-hot stores + 16 loads per pencil (the
   row round-trips memory; doubles round-trip exactly, so the split is
   **bit-identical**). Node, same-window triples: mp1 634.8–641.5 vs mp0
   664.2–675.8 µs/step.
2. **Pencil-loop prefetch reshaped for the pipeline**: c prefetch moved into the
   loop at cpfd=2 (raced 1/2/3/4: 622.8 / vs cpfd=3 638.5 / cpfd=1 637.0 /
   cpfd=4 648.4 — the old v+2 lead was right, the rivals' distance-3 is wrong
   HERE because my plane is already L2-hot from the line phase, unlike their
   cold-plane iter); c rows 0..2 covered by a plane prologue issued before the
   line phase (~8K cycles of lead; the stock loop always demand-missed rows 0–1).
   **SPF** — a second T0 16-line prefetch of state row v+2 feeding the map
   (spfd raced: 2 wins, 1 loses 643.7, 3 loses 650.5): +1–1.5% more (spf1
   622.8–632.2 vs spf0 633.5–656.0). State rows sit in L2 after the line phase;
   the T0 lift matters once the map is the loop's load-latency-critical stream.
3. **Create-time races, all on custody bulk units**: mappipe/linepipe 4-way
   (in-tuner: mp0lp0 771.6 / mp1lp0 738.8 / mp0lp1 811.0 / mp1lp1 779.9 →
   mp=1 lp=0), then spf A/B on the winner. Strict min — every candidate is
   bit-identical, so no output-risk hysteresis is needed. Chain verify
   (m=1..4, real dispatcher, both parities) unchanged at worst 5.099e-13.

**Op count.** FFT and map arithmetic unchanged from r6 (~2.15M FMA-port +
~197K shuffles per step-volume, map 15 ops/8pt). MAPPIPE adds 16 stores + 16
loads per pencil (+131K L1 memory ops/step-vol, zero arithmetic); prefetch
uops: c-prologue 48/plane, spf 16/pencil, cpfd unchanged in count.

**Measured (a80n0 leased cores via tryout.sh + manual check.py; graded chain
m=134, --map; same-window MKL pairs; the $W tryout bug worked around as in
r4–r6 — and tryout's repeatability step silently dies with the check.py bug,
ran manually).** Final shipped build, quiet window: **B=2 (graded): 620.2 min /
620.3 median µs/step, sd 0.06%, vs MKL 1725.4 (2.78×)** — from 670.6/671.2 for
the r6 code in the same afternoon's windows (−7.5%), vs L64_blocked's scored
638.8 and the best rival's 653. B=8: 625.1 (sd 0.12%). B=1: **bimodal across
runs** — 627.9 (batch-flat) in some runs, 699–732 in others at sd<0.1% within
each run; L64_blocked's r6 record shows the same B=1 split (713.5 vs 641 at
B=2); looks like page-placement luck of the B=1 mappings, pre-existing, not
chased this round (the graded case is B=2). Correctness: single rel_l2
4.464e-16 (B=1/2/8); **map-chain end state 1.691e-12 (B=2) / 1.423e-12 (B=1) /
1.953e-12 (B=8) vs budget 1.34e-11** — identical to r6 to the last digit
(bit-identity of the restructure confirmed end-to-end); output bit-identical
across independent runs; zero zmm spills in either bsweep (objdump); setup
0.88 s at B=2.

**Tried and rejected, with the killing numbers:**
- **LINEPIPE — my own r6 "next" idea, refuted.** Two-lb-buffer line phase
  interleaving line g's stage-1 iterations 1:1 with line g−1's stage-2: **+5%
  (811.0 vs 771.6 in-tuner; mp1lp1 779.9 vs mp1lp0 738.8)**, no spills (1 rsp
  ref in 587 instrs). The per-line lb "barrier" I theorized in r6 was never
  real: stage-2 loads of line g and stage-1 stores of line g+1 to the same lb
  are WAR through the store buffer — no OOO hazard — so the stock code already
  overlaps lines, and the interleave only added loop overhead. Kept
  env-forcible (FFT64R_LINEPIPE); the create-time race auto-rejects it.
- **NPF (T1 next-plane state prefetch spread over the pencil stage,
  L64_blocked's r6 "next" item 1 adapted): +3.3%** (654.4 vs 633.4 same-window
  mins) — slabpf's r5 loss repeated: extra issue slots when the volume is
  L3-resident and demand+short-range prefetch already run near the L3 ceiling.
  Kept env-forcible.
- **MAPDIV re-race under the new schedule** (L64_blocked's r5 note 3 predicted
  the all-FMA verdict might invert once the map overlaps other work): it does
  NOT invert — vdivpd build 645.1–648.0 vs all-FMA 623.4–625.4 (+3.5%). The
  all-FMA reciprocal stays the default; third confirmation on this node.
- Cross-window absolutes remain worthless (own identical builds read 620–656
  across this round's windows); every decision above is same-window pairs or
  same-lease env alternation, per the standing rule.

**Borrowed:** the map-ahead row pipeline from rival **1000f989** `iter_z64`
(via L64_blocked ice_r6's "Next" item 2 naming it); the T1-not-T0 choice for
the (rejected) next-plane prefetch and the "T0 bursts displace reused L1"
caution from **L64_blocked ice_r6** (ZAW); the strict-min race protocol for
bit-identical twins and the chain-shaped tuner lineage carried from my
ice_r2/r5/r6 (L17_matrixsimd ice_r1 origin); tryout W= workaround + manual
check.py from L17_matrixsimd/L64_blocked ice_r4 (fourth round running —
monitor, the tryout.sh chain-check AND repeatability steps are both dead
under the $W bug).

**Next round:** (a) The line stages are now clearly the largest component
(~305 µs at r6 attribution, untouched this round) and the two-line interleave
is REFUTED — the remaining line-phase lever is memory-op count, not
scheduling: the lb bounce is 256 loads + 256 stores per line; a single-stage
line FFT would need 128 live vectors (impossible), but a 4+4 split with two
32-vector half-bounces or lb-in-registers for the last stage-2 column might
shave the store-port floor. Price it against the ~520-cyc FP floor before
writing code. (b) PMU port attribution: STILL undone after four rounds of
promising it; the pencil loop is now two overlapped streams and only counters
can say whether port 5 or the load ports bind it. (c) B=1 bimodality
(627 vs 699–732 across runs) is page-placement luck worth one afternoon:
try MAP_POPULATE + explicit 2M-offset staggering of sc/ping/ccust, or simply
re-mmap until the offsets differ — L64_blocked has the same disease, so the
fix generalizes. (d) If anyone ports this map-ahead pipeline: it only pays
where the map and transform are FUSED in one loop (my custody boundary);
under a lazy map in a separate pass there is nothing to overlap.

## Round ice_r8

**Standing at round start.** ice_r7 board: me 614.6 µs/step, L64_blocked
608.2, MKL 1722.8 — but the URGENT addendum voids both: **both L64 entries
FAIL the new two-step precision gate** (m=2 chain vs numpy, rel L2 < 3e-14,
contract ~1.5e-14/step). The failure is the r6 cubic map ladder (~4e-13 per
application), carried by my ckind=2 custody chain since r6 and by
L64_blocked since r5. Under matched gates our cell falls back to r5's clean
0.271 s. Warm cohort target: 0.1748 s = 652 µs/step at the graded case
(B=2, m=134). This round was the gate fix plus whatever speed could be kept.

**What changed (shipped):**

1. **Map ladder default flipped to the r4/r5 EXACT tier** (rsqrt14 + 2
   quadratic Newtons for sqrt(m2), rcp14 + 2 quadratic Newtons for the
   reciprocal, 1e-300 bias, ~2 ulp/application — my own r4 code restored,
   it never left the file). Reproduced the addendum's failure first:
   cubic build **FAIL map-2-step 1.315e-13 vs 3e-14** (their 1.38e-13,
   different seed). Exact build: **PASS 1.929e-15 (15× margin)** at B=2,
   2.264e-15 at B=1, 2.059e-15 at B=8; one-step arithmetic is the contract
   tier by construction (seed 2^-14 → 2^-56 after two quadratic steps on
   both ladders). m=134 end state vs a numpy reference chain: **4.200e-14
   (B=2) / 3.976e-14 (B=1) / 4.577e-14 (B=8)** vs the 300×-anchor tolerance
   floored at 1e-10 — 2400× margin, and identical to the r5 exact-ladder
   drift, confirming the restore is clean. The cubic tier is demoted to
   `-DFFT64R_MAPFAST` (race twin only; it cannot ship under this gate).
   **Cost, same-window alternations ×3: exact 634.7/644.3/636.8 vs cubic
   625.7/614.4/615.1 µs/step ≈ +3.2%** — the op count says why (map is
   18 vs 15 vector uops per 8 points, +98K uops/step-vol, and the ladder's
   dependency chain grows ~16 cyc).
2. **NULL-final_out guard in fft3d_chain — a live segfault, pre-existing
   since r4, in EVERY entry.** The driver at `--map --chain 1` passes the
   unallocated pong buffer (driver.c:141 allocates it only for chain>1) as
   final_out and never reads it back; my entry (and the driver's own
   fallback memcpy) segfaulted the moment anyone ran the ONE-step form of
   the new gate. Guard: final_out==NULL && m==1 → do the real step into
   p->ping (keeps a timed m=1 unit honest), no buffer → return.
   **Monitor / other implementers: check your m=1 map path — check.py
   supports --map-check 1 and the driver crashes everyone there today.**
3. **XPIPE — cross-phase pipeline, a WASH, kept as a raced twin.** The
   finish-line phase of plane u+1 depends only on step-s data of plane u+1,
   so its 8 line-FFTs (decomposed into r7's LP_S1/LP_S2 units, 128 per
   plane) embed 2-per-pencil-iteration into plane u's pencil loop; plane
   0's line phase hoists to a prologue; phase B stays put and reuses lb
   after the embedded units are done with it. Bit-identical (verified: out
   AND .chain byte-identical vs xp0). Same-window pairs: xp0
   633.7/646.5/635.6 vs xp1 635.1/646.2/633.4 — noise; in-tuner 832.3 off
   / 842.7 on → race picks off. Zero zmm spills in either variant
   (objdump). Interpretation: the two streams don't starve ports, they
   contend for L2 (line loads pull the next 68-KB plane while the pencil
   loop round-trips the current one) — the port-mix argument that made
   MAPPIPE win does not transfer to cross-phase distance. Kept
   env-forcible (FFT64R_XPIPE) with a create-time A/B after the spf race;
   costs nothing when off and may differ on other machines.

**Op count.** FFT unchanged (~2.15M FMA-port + ~197K shuffles per
step-volume). Map per 8 complex points: 2 c-loads + 2 adds + 2 FMA (m2) +
rsqrt14 + 7 (h + 2 quadratic sqrt-Newtons) + 1 (den) + rcp14 + 4 (2
quadratic recip-Newtons) + 2 muls out = 18 vector uops (cubic tier was 15).
XPIPE adds zero arithmetic (pure reorder).

**Measured (a80n0 core 4 via manual ssh + slot_lease — see the tooling note
below; graded chain m=134 --map, same-window MKL pairs).** Final shipped
build: **B=2 (graded): 633.8–635.7 min / 634.7–636.0 median µs/step across
three windows, vs MKL 1722.1 same window (2.72×)**; B=1: 632.2; B=8: 633.2
(batch-flat). vs the warm cohort's 652: −2.8%. vs my r7 scored 614.6: +3.2%
= exactly the price of the honest map, nothing else regressed. Correctness:
single rel_l2 4.464e-16 (B=1/2/8); two-step gate PASS at all three batches
(above); m=134 end state PASS at all three (above); chain outputs
bit-identical across independent runs; m=1/2/3/4 create-time verify worst
1.660e-13 vs the exact scalar chain (CHAINBAR 1e-12, chain_ok=1); setup
0.87 s. Tuner picks unchanged: ckind=2, mp=1 lp=0, spf=1, xp=0, plain
stores.

**Tried and rejected, with the killing numbers (all same-window):**
- **MAPDIV under the exact ladder** (one vdivpd replacing rcp14+2N):
  642.9/643.7 vs 635.6/637.0 → **+1.2%**, fourth confirmation on this node
  that the all-FMA reciprocal wins at this fusion site.
- **Prefetch retune under the heavier map**: cpfd=3 650.1 / cpfd=1 640.3 /
  spfd=3 643.5 / spf=0 644.3 vs base 635.6–637.0 — the r7 shape (cpfd=2,
  spfd=2, spf=1) survives the +3 uops/8pt.
- **XPIPE as a win** (see above — shipped as a raced twin, not a win).
- **A cheaper exact ladder**: priced on paper, not built — the gate needs
  BOTH the sqrt and the reciprocal at ~1e-15, one cubic step lands at
  2^-42 ≈ 2.3e-13 on either half (that IS the failing tier), and
  cubic+quadratic costs the same ops as quadratic+quadratic with a longer
  chain. 18 uops/8pt is minimal for this map at this precision.

**Findings for everyone (tooling):**
- **perf_event_paranoid = 4 on a80n0** — perf_event_open returns EACCES
  even for user-space-only counters. The brief's "PMU exposed" is FALSE as
  of today; five rounds of "do PMU attribution next" end here. Stage-split
  timing (l64bench.c) remains the only attribution tool.
- **check.py's m>2 map-check crashes** (`math.floor` without `import
  math`, check.py:94) — the tryout chain-check and repeatability steps die
  for every graded L. I verified m=134 with my own replica of the grader
  rule (build/tryout/L64_radix8/chain134_check.py).
- **tryout.sh is dead on this login host**: the $W-before-definition bug
  (line 36, fifth round running) AND reserve.sh --status false-negatives
  because squeue does not exist locally, which trips tryout's reservation
  check. Workaround: replicate tryout's exact remote steps over ssh
  (build, gen_input, run, check) while holding a slot_lease — same
  pinning, same protocol.

**Borrowed:** the two-step-gate diagnosis and the "fix the custody map,
keep ~0.163 s" framing from the round brief's urgent addendum; the exact
ladder is my own r4/r5 kernel un-demoted (originally corpus §10 §2 / the
1.00-scorer's mapc lineage); XPIPE composes my r6 custody boundary with
r7's LP_S1/LP_S2 decomposition (built for LINEPIPE, reused here).

**Next round:** (a) The +3.2% honest-map cost is structural until someone
finds a cheaper exact reciprocal-of-(1+sqrt) — I priced the obvious
algebra dead; a genuinely new identity would be worth ~20 µs/step to
whoever finds it. (b) The line phase (~297 µs, untouched two rounds) is
now 47% of the step; the 4+4 half-bounce idea from my r7 note is still the
only unpriced lever — but XPIPE's lesson says check L2 traffic, not ports,
first. (c) PMU is DEAD (paranoid=4) — stop planning on it, or get the
monitor to lower the knob. (d) B=1 bimodality did not show this round
(632.2, batch-flat) — leave it unless it returns on the scored window.
(e) If L64_blocked ports the exact ladder (they must — same gate), our
cells re-converge at ~635; the differentiator becomes whoever cracks the
line phase.
