# gen_pfa_small — strategy record

Class: PFA of coprime pairs, small. Owned acceptance sizes: 10 (=2x5), 12 (=4x3),
15 (=3x5), 20 (=4x5). Graded cases: 10:64:1000, 12:64:600, 15:32:600, 20:32:256.

## Round gen_r1

### Where this entry started
The round-0 stub was the dense O(L^4) row-column floor. Everything below is new this
round; there is no earlier leaderboard.

### What was built
One C file, three cooperating structures:

1. **PFA (Good–Thomas) line codelets, no twiddles anywhere.** Each length-L transform
   is DFT_n1 (x) DFT_n2 over the coprime split, with the Ruritanian input map
   `(n2*j1 + n1*j2) mod L` and the CRT output map baked into the codelet's load/store
   index tables (verified against numpy: single-call rel L2 = 2.8–3.2e-16 at every
   size). Modules are exact-constant DFT2/3/4/5 in split-complex form, so every
   multiply-by-(±i) is a component swap folded into the following add, and every
   module constant is a broadcast FMA operand. Per-line op shape: 10 = 5xDFT2+2xDFT5,
   12 = 3xDFT4+4xDFT3, 15 = 5xDFT3+3xDFT5, 20 = 5xDFT4+4xDFT5.

2. **SoA-8 batch-lane engine for the graded path** (BORROWED: ice L45_pfa ice_r7's
   quad-volume "q4" arena, transitively rival v6_5a869e40's run4_ "SIMD lanes =
   volumes, no transposes, no tail lanes" — widened to 8 lanes/zmm for the two Ice
   Lake 512-bit FMA pipes; also the shape of the warm 3907 AoSoA seed). State is
   split re/im, point-major, 8 volumes in the lanes. All three axis passes are the
   SAME pencil codelet at compile-time strides (z: 8, y: 8L, x: 8L^2 doubles); every
   access is a full 64-byte vector; there are ZERO shuffle-class instructions inside
   the transform. Passes run in place (the codelet buffers a whole pencil in v8
   temps, so read-before-write is structural). Pack/unpack are 8x8 in-register
   transposes (24 shuffles per 4 points), once per execute or once per chain.

3. **fft3d_chain owns the whole graded m-step chain** (BORROWED: the ice campaign's
   universal endgame — the chain owns the inter-step state format, licence
   L17_winograd ice_r5). Per group of 8 volumes: pack x0 and c once, run all m steps
   inside the SoA arena, unpack once. The map is fused onto the z-pass stores of each
   step while the pencil is L1-hot (ice L45_pfa ice_r6: "map at the stores of the
   last axis"; a standalone staged map starves the ROB — their r5 lesson, not
   re-tested here). Groups are processed one at a time through all m steps, so the
   state stays L2-resident for the whole chain of that group.

4. **Fast map** — the single biggest win of the round. The graded map
   `s = 1/(1+sqrt(re^2+im^2))` written with vsqrtpd+vdivpd costs ~30 serial divider
   cycles per zmm and was ~half the whole chain step. Replaced with rsqrt14 + 2
   Newton steps (sqrt) and rcp14 + 2 Newton steps (reciprocal): ~1e-15 rel/point
   against the 1.5e-14/step contract, measured two-step gate 0.9–1.3e-15 (tol
   3e-14), chain ends 1.3–2.0x the honest anchor (tol 300x). `max(m, DBL_MIN)`
   guards rsqrt(0) making 0*inf = NaN at z = 0. Worth −24..−32% of the whole
   chain step at every size.

5. **4K-alias stagger**: the SoA component stride at L=12 and L=20 is an exact
   multiple of 4096, so re/im accesses would collide in the low 12 address bits at
   every point. +8 doubles (one line) of stagger between all arena components
   (ducc0's odd-stride rule via LITERATURE.md §08 §5.5). Cheap insurance; not
   A/B-isolated this round.

6. **B%8 remainders and B=1**: per-volume split-complex path reusing the same pencil
   codelets. Lanes are 8 consecutive inner points (x pass: flat (y,z) index — near
   zero waste; y pass: z-chunks with overlapped tails; z pass: y lanes via in-register
   8x8 transposes). Out-of-place ping-pong passes make overlapped tail chunks
   idempotent. Correct at every size, but SLOW (see "known gaps").

### Measured on the node (a80n0 Ice Lake, leased core 4, min over samples)

| case | this entry (chain us/xform) | MKL 2022 + driver map | ratio |
|---|---|---|---|
| L=10 B=64 m=1000 | **1.43–1.48** | 4.56 | **3.1–3.2x** |
| L=12 B=64 m=600  | **2.50–2.72** | 7.73 | **2.8–3.1x** |
| L=15 B=32 m=600  | **6.44–6.56** | 16.45 | **2.5x** |
| L=20 B=32 m=256  | **17.05–17.40** | 58.53 | **3.4x** |

(Ranges are run-to-run wobble with other implementers on neighbouring cores; sd
within a run is <0.1% except when a neighbour lands.) Before the fast map the same
structure measured 2.08 / 3.70 / 8.91 / 22.6 us — i.e. the FFT structure alone was
already ~2.1x MKL, the map cut another quarter to third.

Gates, all sizes: single call 2.8–3.2e-16; two-step 0.9–1.3e-15 (tol 3e-14);
chain-end 4.2e-14–1.8e-13 at 1.3–2.0x the honest anchor (tol 1e-10); bit-repeatable
across processes; B=12 mixed group+remainder and B=1 chains verified.

Plain execute (no map): L=10 B=64: 1.92 us vs MKL 1.51 (we LOSE — pack/unpack every
call is ~0.9 us of the 1.92); L=20 B=32: 25.9 vs 31.5 (we win). Not a graded number
this campaign (all our cases are map chains), but see next-round list.

B=1 plain: 3.75 / 5.30 / 15.06 / 30.94 us vs MKL 1.48 / 2.13 / 6.39 / 31.08 — we
lose B=1 badly at 10/12/15. Known, not scored in cases.txt, highest-priority gap.

### What did NOT work (with the number that killed it)
* **Software prefetch of the c field two pencils ahead in the fused-map z pass**
  (aimed at L=20, whose Q+CQ = 2 MB working set exceeds the 1.25 MB L2 so c streams
  from L3 every step): 19.5–20.1 us vs 17.05–17.5 control at L=20, i.e. **+14%**.
  The pass is issue-bound and the prefetch uops are pure tax — the same mechanism as
  ice L45_pfa ice_r8's "qc-pf next-plane pokes +20%" record, which I should have
  trusted without re-measuring. Reverted; do not rediscover.
* `--chain 1 --map` segfaults in the DRIVER for every implementation (pong buffer is
  only allocated when chain > 1, and both the fft3d_chain call and the fallback
  memcpy write it). Not our bug; the m=1 precision witness is covered by m=2.
* `tryout.sh` builds its `--cin` path from `$W` before `$W` is assigned, so its
  chain/map leg cannot work as shipped; I ran the identical build/run/check commands
  manually over ssh with a slot lease. (Also: `reserve.sh --status` needs `squeue`,
  which is not in PATH on wallaby, so tryout's reservation gate false-fails while
  the reservation heartbeat is in fact alive.)

### Operation count (per volume, per axis pass: L^2 pencils / 8 lanes)
Vector instructions per pencil (loads+stores+FP, all zmm): L=10 ~130, L=12 ~170,
L=15 ~210, L=20 ~330. No shuffles inside passes; pack/unpack ~6 shuffles/point/dir.
Map: 15 FMA-class ops per 8 points (no divider).

### What I would do next (ranked)
1. **B=1**: the split path loses 2.5x to MKL at 10/12/15. The waste is overlapped
   tail chunks (y pass at L=10 does 2 chunks where 1.25 are needed) plus ~3840
   port-5 shuffles/volume in the z-pass transpose sandwich. The known-good shape is
   ice L6_pfa's: interleaved complex, 2–4 complex per vector along adjacent-z lanes
   with a provably-minimal in-register lane turn for the z pass, plus half-width
   (ymm/xmm) tail codelets instead of overlap. Estimate 2.5–3x available.
2. **Fuse pack into the x pass and unpack into the z pass** for plain execute (saves
   one full round trip of the SoA arena per call; worth ~0.5 us/vol at L=10 B=64,
   would flip the plain-execute loss vs MKL). Chain already amortizes this.
3. **L=20 residency**: Q+CQ = 2 MB > L2. Prefetch failed (above). Candidates:
   interleave c INTO the state arena point-major (one stream instead of two), or
   accept L3 streaming. Measure `ld_blocks` off-node if PMU ever appears.
4. **Instruction diet in DFT5**: the e1/e2 pairs share `x0 + (a,b)` subexpressions;
   a 3-FMA lifting form (§08 §6.3) could cut ~10% of pencil FP at 15/20. Ice lesson:
   budget an hour, not a round (r2's 11.9% op cut bought 0.8% there).
5. **Round-3 generality**: the engine is already table-driven per (n1,n2); adding
   any coprime pair whose factors are in {2,3,4,5,7,8,9} is a table + one pencil
   function. Coordinate with gen_planner/gen_race on the create()-race when the
   driver starts asking for unlisted sizes.

### For other entries reading this
* The SoA-8 + PFA pencil structure here is generic: any line codelet that reads
  8-double vectors at a compile-time stride drops in. gen_powp/gen_pow2 could reuse
  the arena, pack/unpack, and fused-map chain scaffolding wholesale (twiddle stages
  slot in as extra broadcast-FMA operands between the two PFA-style stages).
* The fast map (rsqrt14/rcp14 + 2 Newton each) is worth −24..−32% of the whole
  graded step to ANYONE who owns their chain, and passes the two-step gate with 20x
  margin. Take it.
* Do not put prefetch uops in an issue-bound pass; two campaigns have now measured
  it at +14% and +20%.
