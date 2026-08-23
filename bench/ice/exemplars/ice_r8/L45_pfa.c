/* L45_pfa.c -- forward complex 3D DFT of a 45^3 cube, batched, out-of-place.
 *
 * ROUND ice_r8, WHAT SHIPS: the CUSTODY chain family ("qc", FFT45_QC=1
 * default -- BORROWED: L45_mixedradix ice_r7's vg4-custody schedule,
 * transitively L64_radix8 ice_r6 ckind=2 -- rebuilt on the r7 q4
 * point-major group layout).  m+1 alternating-orientation sweeps instead
 * of 2m volume sweeps: sweep s finishes step s's carried axis WITH THE MAP
 * AT ITS STORES, then runs step s+1's z pass and the sweep-local axis --
 * three passes per 129.6 KB plane/slab while it is L2-resident, so the
 * group volume crosses L3 once per step instead of twice.  The Q sweeps'
 * mapped x pass reads c from a pencil-order image CQ (built once per group
 * per call, 354 us; point-major c at x-store sites is this entry's own r6
 * +139 us/vol trap).  Custody is a NEW BIT CLASS (step axis order
 * alternates (z,y,x)/(z,x,y) with parity, deterministic per (m,B)), so
 * FFT45_QC selects it at compile time exactly like FFT45_XF selects
 * families; the gate now checks every arm at BOTH m and m+1 (all five
 * sweep types + both final parities).  Same-window forced A/B, 3/3:
 * qc 214.6/215.6/217.7 vs q4 231.2/231.4/231.5 us/xform (-6.0..-7.2%);
 * B=8 214.7 (two groups); B=1 falls back to xf unchanged.  Gates: single
 * 4.002e-16, two-step 2.045e-15 (tol 3e-14), chain m=177 4.701e-14 at
 * 1.05x the honest-library anchor (tol 1e-10), bit-repeatable.  Priced
 * OUT this round: qc2 pencil-order c in the P sweeps too (+0.8-0.9%, 3/3
 * -- plane-local point-major c was already OoO-covered), qc-pf next-plane
 * T2 pokes in the L2-hot z pass (+20%: the pass is issue-bound, the poke
 * tax is port pressure, not cache pollution).
 *
 * ROUND ice_r7 (now the QC=0 control): the QUAD-VOLUME chain family ("q4"/"q4-pf" --
 * BORROWED: rival v6_5a869e40's run4_ structural idea, "SIMD lanes =
 * volumes, no transposes, no tail lanes", rebuilt on this entry's DAGs so
 * it is BIT-IDENTICAL to the xf family and races in the same gated pool).
 * State for each group of 4 volumes lives point-major in a plan-owned,
 * hugepage-madvised arena Q (point (x,y,z) of volumes 0..3 = one zmm);
 * c is permuted into C4 (+2048 B page phase) once per group per call.
 * Every codelet access in all three passes is one full-width 64B-aligned
 * load/store; the corner-turn broadcasts, the slot scratch, and all tail
 * machinery vanish from the steady loop.  Same x-first step, same map at
 * the y-store sites, final map fused into the deinterleave.  Same-window
 * driver A/B: q4 232.5-233.7 vs xf-pf 263.7 us/xform (-11.8%); B=1 falls
 * back to xf per-volume (bit-identical), B%4 remainders too.  Priced OUT
 * this round: c-poke in the z-pass (+16%), MS=1 rcp ladder (+7%),
 * store-direct z-DAG (probe zsd 268 vs 245/group).  See the strategy
 * record, round ice_r7.
 *
 * ROUND ice_r6, WHAT SHIPPED: the X-FIRST chain family ("xf"/"xf-pf",
 * FFT45_XF=1 default -- BORROWED: L45_mixedradix ice_r4's CXF step shape +
 * their ice_r5 MPAIR store pairing, on my tr machinery).  Per step: x pass
 * FIRST (in place on the MAPPED state), then per plane zsubt (plain) ->
 * ysubtm, whose stores apply map(z+c) as the step's LAST write -- 22
 * stash-paired ladders + 1 single per call (one divide per 8 points), c
 * loads plane-local.  Step m keeps a raw ysubt; one span writes final_out.
 * Same-window A/B on the node, 2/2: xf-pf 270.23/270.22 vs vt-pf (the r5
 * shipped shape) 275.25/274.93 us/xform at B=4 (-1.7/-1.8%); B=1 259.8.
 * WHY: this round's probes showed the r5 STAGED map costs p1tm-p1t = +90
 * us/vol (the ~430-cyc span cannot overlap the codelet -- the ROB cannot
 * span both; L36_pencilfused's "standalone staged map starves" trap,
 * now quantified), while the store-site placement hangs the ladders off
 * the END of the dependency graph (L36_mixedradix r6's placement lesson).
 * The axis order (x,z,y vs z,y,x) is a DIFFERENT BIT CLASS, so the two
 * families are compile-time-selected (FFT45_XF), never raced adaptively:
 * arm[0] of the pool sets the gate's bit reference and the other family
 * auto-disqualifies, keeping cross-process repeatability.  FFT45_CH
 * indices (XF=1): 0=xf-pf 1=xf 2=vt-pf 3=vt 4=vm-zs 5=uf 6=vs 7=vs2.
 *
 * ROUND ice_r6, ALSO BUILT (kept as priced controls): SPLIT-COMPLEX (SoA)
 * CHAIN ARMS ("vs" lazy-map, "vs2" map-at-x-stores).
 *   The one structural lever both L=45 records left on the table: the chain
 *   OWNS the inter-step state format (licence: L17_winograd ice_r5; working
 *   precedent: L64_radix8's split-complex pipeline), so the vs arms keep raw
 *   z SPLIT -- per plane a 45x48 re array then a 45x48 im array (y-major,
 *   rows padded to 48 doubles = 6 whole lines; pad columns stay zero).
 *   DELETED per volume-step vs vt: all 116,766 codelet SWAPs (the entire
 *   FFT p5-only term, ~40 us of two-pipe floor at 2.9 GHz -- every
 *   SWAP+VPAIR crossing becomes one FMA on the other component with the
 *   sign folded into the constant) and the map's pack/unpack permutes.
 *   PAID: 8 B scalar-broadcast gathers (vbroadcastsd (mem),zmm{k} = one
 *   load-port uop) instead of 16 B ones at the two corner turns, BOTH now
 *   confined to L1 buffers (map ping-pong, plane scratch); every state
 *   access is a full-width 64B-ALIGNED load/store; +6.7% codelet-lane
 *   waste (45 rides 6 blocks of 8).  Two-pipe port floor 108.9 -> ~96
 *   us/vol; codelets audited on the .o before the node: ZERO shuffle-class
 *   instructions, all 720 gathers/call memory-folded, spills 65-68 (vs
 *   48-94 interleaved).  BIT-IDENTITY with vt/vm/uf preserved by
 *   per-component transcription of the SAME DAGs (PFA45RS = PFA45R at the
 *   z-site incl. rows 40-43, scalar PFA45S line for row 44 = dft45_line1's
 *   PFA45 order, PFA45S at y/x sites; map = same op sequence, one ladder +
 *   one divide per 8 points, same barrier), so the cross-arm memcmp gate
 *   and cross-process repeatability survive with vs in the pool.  x0/c
 *   convert to split once per volume-chain (2/177 amortized); final map
 *   emits interleaved final_out in one pass.  Split c copy sits at +2048 B
 *   page phase from the (4096B-strided) state volumes.
 *
 * ROUND ice_r5 (cumulative round: techniques adopted across entries).
 *   1. TRANSPOSE-FREE PHASE 1 ("tr", BORROWED: L36_pfa ice_r2's mechanism,
 *      transfer to L=45 proven by L45_mixedradix ice_r2 at -9.6..-12%): on
 *      bare-metal ICX a masked vbroadcastf64x2 from memory is ONE load-port
 *      uop, so zsubt gathers lanes=y as 4x16B broadcasts per codelet input
 *      and stores codelet output UNtransposed to a slot-major scratch
 *      (slot = ygroup*45 + kz, 64 B full-width aligned stores); ysubt
 *      re-gathers lanes=kz from the slots (element (y,kz) = slot
 *      (y>>2)*45+kz, granule y&3 -- one formula covers the odd 45th row, so
 *      no y-side tail case).  BOTH TRNC passes, GCOL/SCOL, and the Wv spill
 *      array die: ~90k port-5 shuffles/volume deleted, two-pipe floor
 *      124.5 -> ~109 us/vol; the corner turn is repaid as ~2x180 16 B load
 *      uops/group against 2x64B/cyc load ports that were near idle.
 *      Bit-identity is preserved BY CONSTRUCTION (gathers deliver exactly
 *      the values TRNC delivered; z-site keeps PFA45R order, y/tail sites
 *      keep PFA45), so tr and classic arms race in one pool and the
 *      cross-process repeatability contract survives an adaptive pick.
 *   2. ROW-GROUP-PIPELINED LAZY MAP (BORROWED: L36_pencilfused ice_r5's
 *      maprows): the mapped zsubt stages map(z+c) of row group yg+1
 *      (4 contiguous rows, 2.88 KB) into an L1 ping-pong immediately BEFORE
 *      the FFT of group yg -- the map's rsqrt/divide chains issue under a
 *      full codelet of independent work, and the map stores sit ~500 uops
 *      ahead of their own broadcast reloads.  Their priced traps respected:
 *      map fused INTO broadcast sites spills (+57% for them); a standalone
 *      staged map starves the divider.  Map arithmetic itself is UNCHANGED
 *      from ice_r4 (pair ladder, 1 vdivpd/8pt) => per-point bits unchanged.
 *   3. STATE-ARENA PAGE-PHASE CONTROL (BORROWED: L45_mixedradix ice_r4,
 *      transitively L23_rader): the chain now runs in a plan-owned
 *      4096-aligned arena placed at page phase (phase(c)+2048)&4095 -- the
 *      state-vs-c 4K false-dependence lottery (their measurement: 294 vs
 *      333 us across runs of one binary) is the leading suspect for this
 *      entry's 4.2% r4 leaderboard spread.  final_out takes ONE write (the
 *      last step's map span); x0 is read once.
 *   4. Chain arms rebuilt: vt (tr, rank 0), vt-pf (tr + PF45 one-line-ahead
 *      T0 pokes on phase 2's 45 RMW streams -- L45_mixedradix ice_r4
 *      measured the same-shape poke worth ~11 us under the chain, my ice_r2
 *      "pf is a tax" number was the unfused exec), vm-zs (the ice_r4
 *      incumbent, priced fallback), uf (unfused control).  vp/bm/pp DELETED
 *      (r4 drained window: 315.9/317.3/310.3 vs vm 301.8).  Exec tournament
 *      gains pw4-tr0 (rank 0) and il0's rank-0 bet is reversed (r4 drained
 *      window: ip0 259.8 < il0 262.2).  Spin now also estimates the core
 *      clock for the description (L36_pfa ice_r5's 3.3-vs-2.9 GHz window
 *      decode).
 *
 * ROUND ice_r4: THE GRADED STEP CHANGED to the rivals' full task,
 *     state <- (z + c) / (1 + |z + c|),  z = raw (unnormalized) FFT(state),
 * and the driver times an exported fft3d_chain(plan, x0, c, final_out, m)
 * weak symbol as the whole m=177-step unit (entries without it pay
 * fft3d_execute + a driver-side vectorized map pass).  What ships:
 *   1. fft3d_chain, with final_out as the in-place state arena: phase 1 is
 *      in-place-safe by construction (zsub drains plane x through the plane
 *      scratch before ysub writes it back), phase 2 always was, so raw z
 *      ping-pongs NOWHERE -- the chain runs in final_out end to end, x0 is
 *      read once, and one in-place map45_span pass finishes step m.
 *   2. The LAZY MAP (borrowed: the rival pipelines' winning fusion, corpus
 *      10 s2, via L13_rader ice_r4): between steps the buffer stays RAW z;
 *      the map is applied inside the next step's zsub loads, where both z
 *      and c stream sequentially (mp flag on zsub; the odd z-column maps on
 *      its GCOL gather, the odd 45th row pre-maps into a 720 B stack line
 *      before dft45_line1).  The map's ~60-cycle dep chains sit exactly in
 *      the pass with the most latency slack (zsub runs at ~1.9x its port
 *      floor, ice_r2 probes).
 *   3. Map arithmetic (borrowed: the L13_rader/L17_matrixsimd/L23_matrixsimd
 *      ice_r4 consensus, transitively the rivals' mapF): s = re^2+im^2 with
 *      a 1e-300 guard folded into the fma (rsqrt14(0)=inf would NaN);
 *      vrsqrt14pd seed + 2 Newton (2^-14 -> 5.6e-9 -> 4.7e-17, below double
 *      rounding); d = fma(s, y, 1); ONE exact vdivpd per 8 points -- two
 *      interleaved vectors' |w|^2 pack into one zmm (even lanes), one
 *      divide, unpack, two mul-outs (L17's s6 shape: half the divider of
 *      per-vector divides, ~19 pipe uops + 1 div per 8 pts).  Every op is
 *      PER-POINT IDENTICAL under any vector grouping/width, so all chain
 *      arms below are bit-identical -- racing them at plan time can never
 *      flip output bits (L13_rader's repeatability design).  ~2-3 ulp per
 *      application; budget 1e-13/step (tol 1.77e-11 at m=177) untouched.
 *      -DFFT45_MS=1 swaps the divide for an rcp14+2-Newton ladder (bit
 *      class differs -> compile-time only, per L23_matrixsimd).
 *   4. Chain STRUCTURE arms, gated + raced in create() (bit-identical, see
 *      3): vm-zs volume-major (corpus 10 s3: iterate one volume through all
 *      m steps; at L=45 state+c = 2.8 MiB, so this is an L2/L3-locality
 *      play, not full residency); vp-zs volume PAIRS with the ice_r2 il
 *      interleave running ACROSS STEPS (p2 of one volume's step hides under
 *      the partner's p1); bm-zs batch-major il (ice_r2's shape); vm-pp
 *      plane-premap (map45_span into the P image, zsub untouched -- hedges
 *      the zsub register-pressure/DSB risk); uf unfused control (exec +
 *      streaming map pass, prices fusion itself; L13_rader measured uf WIN
 *      at their B=1).  FFT45_CH forces an arm.
 * FFT kernels, tuner, and fft3d_execute are unchanged from ice_r2.
 *
 * ROUND ice_r2 (first round with a live agent on the ICE panel; the ice_r1
 * "entry" was this file byte-identical to geom/impl_11 -- the r1 agent died
 * in the launch crash-storm, VERDICT ice_r1 §3).  Everything below the r11
 * header is CLX-era history; the machine is now a bare-metal Ice Lake-SP
 * Gold 6326 (TWO 512-bit FMA pipes on p0/p5, 1.25 MB L2, 48 KB L1d, no
 * AVX-512 licence downclock) and the workload is the graded chain (m=177
 * unitary steps, B=4, 11.1 MiB working set, L3-resident end to end).
 * Budget on that machine: 1497 zmm calls x (344 FMA + 78 shuf) + ~87k
 * transpose shuffles ~= 360k two-pipe port cycles = 124 us/vol; ice_r1
 * measured 295 us with phase 1 ~73% of it (the rival's node probe:
 * p1=208 of fu=274 us) -- i.e. phase 1 is L3-TRAFFIC-bound (in read 1.46 MB
 * + out write 1.46 MB + out RFO read 1.46 MB per volume), not port-bound.
 * ice_r2's create()-time decomposition probes (LOUD/verbose only) pin the
 * budget down: p1=213-227, p2=65-68 us/vol, and with BOTH memory sides
 * forced hot phase 1 still costs p1c=164-168 us -- the kernel runs at
 * ~1.9x its two-pipe port floor with memory adding only ~45 us (30 store-
 * side + 14 read-side).  Every uop added to the loops (prefetch pokes,
 * ERMS copies) made it slower; every mechanism that DELETES stall overlap
 * paid.  What ice_r2 ships:
 *   1. IL exec variants (cross-volume software pipelining, mine -- untried
 *      in either L=45 lineage): between phase-1 planes of volume b, run a
 *      proportional band of phase-2 tiles of volume b-1, so phase 2's
 *      45-stream read latency (the L2 streamer tracks ~16 streams) hides
 *      under phase 1's compute.  Won EVERY dev window it appeared in:
 *      -1.5% vs ip0 drained (288.3 vs 291-293), -5..9% contended.  Ranked
 *      first in the hysteresis tie-break for exactly that reason.
 *   2. HZ hybrid candidates (mine): pw2 (ymm) z-subpass + pw4 (zmm) y-sub-
 *      pass through the width-agnostic pl layout -- ymm FMAs issue on
 *      p0/p1 leaving p5 wholly to the transpose shuffles (pw2 measures
 *      EQUAL to pw4 whole-volume despite a 63% worse slot floor, so the
 *      z-site is where its scheduling headroom lives).  Dev windows: within
 *      noise of ip0, never beat il0; left tuner-gated for the drained
 *      window to price.  phase1_plane split into zsub/ysub to compose this.
 *   3. CPY mode (BORROWED from L45_mixedradix r8): y-pass into an L1-hot
 *      plane image, one ERMS rep movsb per plane -> out, to delete the
 *      1.46 MB/vol RFO.  MEASURED LOSS on the node, +11% (cp0 314.8-320.6
 *      vs ip0 282.5-286.0 us/vol, quiet windows): at 32 KB the copy neither
 *      skips the RFO reliably nor runs faster than the stall it removes.
 *      Kept in the pool for the record; expect it never picked.
 *   4. Clock-settle spin (BORROWED from L17_matrixsimd ice_r1, transitively
 *      L17_winograd): ~150 ms of dense FMA before the tournament; the
 *      schedutil governor hands short create()s an unramped core and the
 *      ramp edge mis-ranks candidates (ice_r1's 12% spread = plan re-raced
 *      per run).  nv=4 timing rounds 6 -> 8, min-of-rounds kept.
 *   5. Pool pruned for this node: pw2-ip-pf3 / pw2-sp-* DELETED (every
 *      256-bit variant lost every ice_r1 tournament); pf3/pf3a kept but
 *      ranked last (L13_rader: prefetchw +7.4% when the store target is
 *      L3-resident, which the drained graded chain guarantees -- though
 *      CONTENDED dev windows do pick pf3, consistent with that rule).
 *   6. Negative controls with numbers: FFT45_OVERLAP_TAIL (r7-r9 recompute
 *      tails) re-measured on ICX: p1c 173.4 vs 167.7 -- the r10 xmm tail
 *      lines survive on a two-pipe machine too.  The outer group loops are
 *      NOT unrolled by gcc 11 (byte-identical with forced unroll-1), so the
 *      DSB-vs-MITE hypothesis for the p1c gap is dead: each loop body
 *      (~1k uops) replays from the DSB; the gap is latency/scheduling, not
 *      the front end.  perf_event_open is DENIED on the node for
 *      implementers (perf_event_paranoid=4; the brief's "PMU exposed" does
 *      not hold at our privilege) -- the pmu probe degrades to a message.
 *
 * ROUND panel_r11 (sixth round).  Both changes attack phase 1's data
 * movement, per the r10 VERDICT (phase 1 = 76% of B=1, port 0 non-binding,
 * front end closed; §4.1 priced the spill difference between the two L=45
 * entries' takes of the same DFT9 DAG):
 *   1. Per-site codelet stage order.  The memory-store sites (y-subloop,
 *      phase 2, PW=1 tail lines) now run the codelet DFT5-FIRST into
 *      T_[9*k2+n1], then DFT9s that read 9 contiguous hot slots and hand
 *      every output straight to the store macro (L45_mixedradix's
 *      ST1G/ST2G shape; VERDICT §4.1: their take of the identical DAG
 *      added +3 stack moves where mine added +23, worth ~3 points of B=1).
 *      The z-site, whose ST writes a vec array that is transposed
 *      afterwards, keeps the old DFT9-first order (PFA45R): a per-function
 *      audit under node flags shows the store-direct order is WORSE there
 *      (phase1_plane spills 83 -> 91) and better at the memory sites
 *      (phase2 spills 53/55 -> 46/51).  Same DAG, same 344 ops, same maps.
 *   2. NEW zal exec variants (pf0a/pf3a): the z-pass in-loads become
 *      aligned rolling loads recombined with valignq -- see the ZLA comment
 *      in the template for the phase arithmetic (row phase is
 *      16*(b+x+j) mod 64 within every y-group because yb == 0 mod PW, so
 *      ONE per-plane value zc selects among 4 compile-time shift patterns).
 *      Deletes ~15k of the ~16k split z-loads per volume.  Node-only by
 *      construction (wallaby's SPR core hides splits; its arena prices
 *      pf0a at +3.2% over pf0, as pre-registered); shipped tuner-gated so
 *      the node's tournament prices the split-load class directly -- the
 *      r10 record's named next step.  pf codes 4/5 are REUSED from the r10
 *      pfp candidates, which took zero node picks (3/3 pf0 at B=1/B=2,
 *      3/3 pf3 at B=16) and are deleted per their pre-registered branch
 *      ("closed for good").
 *
 * ROUND panel_r10.  r10 changes:
 *   1. The phase-1 overlap-recompute tails are replaced by true PW=1
 *      (128-bit) tail lines.  45 = 11*4 + 1, so the z and y subpasses each
 *      ran 12 full-width groups where 11.25 are needed: 90 full zmm codelet
 *      calls per volume (30,960 port-0 ops, plus their 44-load transpose
 *      blocks / 45-store rows) were recompute.  Now 11 full groups + ONE
 *      PW=1 line per subpass per plane: the 45-point codelet on a single
 *      complex per xmm vector (VEX-coded, dual-issues on ports 0 AND 1 on
 *      CLX, no transposes, 16 B loads/stores that never split a line).
 *      zmm calls/volume 1587 -> 1497 (-5.7% port-0); the 90 xmm lines add
 *      ~15.5k port-0-equivalent cycles back.  This REVERSES r7's change 3/4
 *      (overlap tails, chosen then for instruction-count leanness): the
 *      front-end premise died in r9 (L36's triple null), and the r7 form
 *      recomputed 8.3% of phase 1's full-width group work.
 *      -DFFT45_OVERLAP_TAIL restores the r7-r9 overlap form for a control
 *      (and compiles the r11 zal candidates out -- their phase arithmetic
 *      needs yb == 0 mod PW, which the clamped overlap group breaks).
 *   2. pl-column prefetch (pfp) candidates -- took zero node picks in r10,
 *      DELETED in r11 per their pre-registered branch.
 *
 * ROUND panel_r9 (fourth round).  r9 changes:
 *   1. DFT9 module replaced: the hand CT 3x3 form (44 FMA-port vector ops +
 *      10 swaps) is replaced by a pairwise transcription of genfft's FMA
 *      n1_9 DAG (fftw-3.3.10/dft/scalar/codelets/n1_9.c: 24 add + 56 fma =
 *      80 scalar FMA-port ops) onto interleaved-complex vectors: 40 FMA-port
 *      ops + 12 swaps.  Every scalar re/im line pair maps to ONE vector op;
 *      the points where the scalar DAG crosses re and im (multiplies by i and
 *      the (1 + c*i) spiral factors) each cost one SWAP, with all signs
 *      folded into VPAIR constants.  Per line: 364 -> 344 port-0 ops (-5.5%,
 *      the node's bottleneck port), 68 -> 78 port-5 shuffles (still <= half
 *      of port 0).  L36_pfa r1 burned three attempts deriving this by hand;
 *      the transcription of the actual generated DAG is mechanical and the
 *      create()-time reference gate proves it.
 *   2. The file-level '#pragma GCC optimize("unroll-loops")' is REMOVED
 *      (r8 VERDICT 3c: the scored build has carried -funroll-loops in its
 *      Makefile all along, so the pragma's premise was false, and L17_rader
 *      measured the pragma FORM as a ~2% tax because optimize() rebuilds the
 *      whole per-function option set).  -DFFT45_UNROLL_PRAGMA restores it
 *      for A/B.
 *
 * ROUND panel_r8 (r6, r7 before).  r8 changes:
 *   1. Compile-time specialization of the prefetch ladder and the mid-buffer
 *      layout (borrowed from L45_mixedradix's exec-variant structure): phase 1
 *      and phase 2 are always_inline bodies taking const flags and const
 *      strides, instantiated into per-(pw, mode, pf) exec functions.  The
 *      node's r7 pick at B=1/B=2 was pw4-inplace-pf0, and in r7 that path
 *      still carried runtime pf branches, two cursor pointers per plane, and
 *      ~180 never-executed prefetch instructions inside the hot loops (my
 *      3535 hot instructions vs the rival's 3134 -- and they beat me 328 vs
 *      343 us at equal op count).  Now pf0 compiles to zero prefetch code.
 *   2. NEW padded-scratch mode (scratchp): phase 1 writes a plan-owned S with
 *      row pitch 52 complex (832 B = 13 lines, odd, 64B-aligned rows) and
 *      plane pitch 45*52 = 2340 complex (37440 B = 585 lines, odd).  The
 *      y-pass stores and the x-pass loads become 64-byte ALIGNED, deleting
 *      two of the three split-access classes (~75% of the 64 B accesses at
 *      the natural 720 B row stride split a cache line; only the x-pass
 *      stores to `out` keep paying the odd-L toll).  The odd-line pitches
 *      also break the same-stride read/write mod-4096 correlation that
 *      L23_rader measured at -25..30% (their self-inflicted-aliasing rule):
 *      r6's unpadded scratch read S and wrote out at the SAME 32400 B plane
 *      stride and lost ~66% -- padding recovers most of that (wallaby r8:
 *      sp is within 11-13% of inplace at B=1, was ~66% behind) but does not
 *      win there; it is a candidate for the node's different cache/split
 *      physics, not a wallaby winner.  scratchp's x pass is out-of-place
 *      (S -> out), so its odd
 *      tails overlap-recompute per y-row (12 tiles/row at PW=4, +2.2% volume
 *      ops vs flat tiling) instead of needing masked calls.
 *
 * TECHNIQUE (unchanged from r7 otherwise)
 *   Row-column 3D DFT; every 45-point line is a Good-Thomas / prime-factor
 *   9 x 5 codelet, on INTERLEAVED complex vectors whose lanes are a spectator
 *   axis.  gcd(9,5) = 1, so with
 *
 *       input  (Ruritanian): n = (5*n1 + 9*n2) mod 45     n1 in [0,9), n2 in [0,5)
 *       output (CRT):        k = (10*k1 + 36*k2) mod 45   (10 = 5*[5^-1]_9,
 *                                                          36 = 9*[9^-1]_5)
 *
 *   W45^{nk} = W9^{n1 k1} * W5^{n2 k2} exactly: 5 DFT9s then 9 DFT5s with NO
 *   twiddles in between, both maps folded into compile-time addressing.
 *   DFT9 = genfft n1_9 FMA DAG transcribed to interleaved vectors (40
 *   FMA-port ops + 12 swaps, r9), DFT5 = FFTW n1_5's FMA form (16 ops).
 *   Per 45-point line: 344 FMA-port vector ops + 78 shuffles.
 *
 *   Two sweeps (the structure that won L=36 and L=45 on the node):
 *   phase 1, per x-plane:
 *       z transform: lanes = PW y-rows, PWxPW complex-granule register
 *                    transposes on load and store, into plane scratch
 *                    pl[y][kz] (row pitch PPITCH = 52 complex, 13 lines,
 *                    coprime with the 64 L1 sets)
 *       y transform: lanes = PW kz (contiguous in pl), store to mid[x][ky][kz]
 *                    where mid = out (INPLACE) or the padded S (SCRATCHP)
 *   phase 2:
 *       x transform: lanes = PW kz, 45 streams at the plane stride.
 *       INPLACE:  mid = out, in place; tiles the FLAT (y,z) index
 *                 (2025 = 506*4 + 1: ONE masked tail call per volume).
 *       SCRATCHP: S -> out, out of place; tiles per y-row with an
 *                 overlapping last tile (rows are padded so flat tiling
 *                 does not map affinely to out).
 *
 *   Tail policy (r10): phase 1's z and y subpasses run NFULL (11) full-width
 *   groups plus ONE PW=1 xmm line each (dft45_line1) for the odd 45th
 *   row/column -- no overlap recompute; the z-line's odd 45th column within
 *   a group is a 16 B column gather/scatter (GCOL/SCOL); the in-place x
 *   pass keeps the single masked flat-tail call; scratchp's out-of-place x
 *   pass keeps its per-row overlapping tile (sp takes no node picks; not
 *   worth a third tail scheme).
 *
 *   A transpose-count note for the record: a lanes=x phase-1 variant (x is a
 *   spectator of BOTH the z and y transforms) still needs one transpose-class
 *   gather on the z load and one more entering phase 2 -- every arrangement
 *   of this pass structure pays exactly 2 granule transposes per element,
 *   which is what both L=45 entries already pay.  Not a lever; documented so
 *   nobody chases it.
 *
 *   PREFETCH LADDER (tournament-gated, all compile-time now):
 *     ip-pf0: nothing.
 *     ip-pf1: phase 2 pokes its 45 x-read-streams one line ahead (PF45).
 *     ip-pf2: + paced T1 read prefetch of phase 1's in-stream (PFIN) and
 *             per-tile pre-coverage of the next volume's input (PFNX).
 *     ip-pf3: + write-intent prefetchw of phase 1's cold mid-plane stores.
 *     ip-pf0a / ip-pf3a (codes 4/5): pf0 / pf3 plus the r11 zal aligned
 *             z-load path (not a prefetch; the codes slot into the same
 *             forcing knob).
 *     cp-pf0 / cp-pf0a / cp-pf2 (codes 6/7/8, ice_r2): the ERMS plane-image
 *             copy mode, bare / plus zal / plus the pf2 read set.
 *     sp-pf0: nothing (S is cache-hot; out is only touched in phase 2).
 *     sp-pfs: PFIN on in + prefetchw poke of phase 2's 45 cold out-streams
 *             + PFNX (the streaming set for scratchp).
 *   fft3d_create() gates every candidate against a scalar O(n^2)-per-line
 *   reference at 1e-13 (volume 0 AND the last arena volume, so S-reuse bugs
 *   across a batch cannot hide), times them interleaved, and installs the
 *   fastest with a 3% simplest-first hysteresis.  FFT45_PW / FFT45_MODE /
 *   FFT45_PF force the choice at plan time for the monitor's control runs.
 *
 * ATTRIBUTION
 *   - Const-propagated exec-variant instantiation (pf compiled out of the
 *     pf0 path): L45_mixedradix (their body()/exec_v_c structure, all
 *     rounds).  Their leaner pf0 loop is why they won r7 on the node.
 *   - Padded-scratch odd-line pitches: L23_rader r6/r7 (the 1058->1064
 *     stride pad, -25..30% at B=1, and the "pad so every stride is an odd
 *     number of cache lines" corpus rule, LITERATURE 04/08).
 *   - Two-sweep plane-fused structure, spectator lanes, 6-op DFT3 / 2-op
 *     CMUL, TRNC transpose, PFIN/PFNX, prefetchw, tuner + hysteresis + env
 *     forcing: L36_pfa (r2-r5), transitively L36_mixedradix r1, L6_unrolled
 *     r3.  Flat phase-2 tiling: L45_mixedradix r7.  Masked z-column tail:
 *     L45_mixedradix r6 (as 128-bit inserts/extracts).  16-op DFT5: FFTW
 *     n1_5 via the corpus.
 *
 * OPERATION COUNT (PW=4, r10 tails)
 *   INPLACE: 45*(11+11) + 506 + 1 = 1497 zmm codelet calls x 344 = 514,968
 *   zmm FMA-port ops/volume (r9: 545,928; the PW=1 tails cut 5.7%) plus
 *   90 xmm lines x 344 ops that dual-issue on ports 0+1 (~15.5k
 *   port-0-equivalent cycles).  Port-0-equivalent floor ~ 530k cycles
 *   ~ 183 us at 2.9 GHz (r9 floor: 188 us; r9 node B=1: 315.9 us = 1.68x,
 *   port 0 measured NON-binding, so the tails' real value is the deleted
 *   movement of 90 full-width groups, not the op count).
 *   SCRATCHP: 45*22 + 540 = 1530 calls = 526,320 + the same 90 xmm lines.
 *
 * ACCURACY: ~4e-16 relative L2 vs numpy (unchanged module family).
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <sys/mman.h>

/* ice_r7: 2 MB-hugepage madvise on the q4 group arena (see Q4OFF). */
#ifndef FFT45_THP
# define FFT45_THP 1
#endif

#include "fft3d_api.h"

#ifndef L45_PFA_ONCE            /* ============ COMMON, first pass ============ */
#define L45_PFA_ONCE

/* r7 shipped '#pragma GCC optimize("unroll-loops")' here on the belief the
 * scored build lacked -funroll-loops.  r8's VERDICT 3c proved the premise
 * false (Makefile:15 has carried the flag for >= 3 rounds), and L17_rader
 * measured the pragma FORM as a ~2% tax in its own file (optimize() rebuilds
 * the entire per-function option set, not just the named flag).  Default is
 * now OFF; -DFFT45_UNROLL_PRAGMA restores it for a control build. */
#ifdef FFT45_UNROLL_PRAGMA
# pragma GCC optimize("unroll-loops")
#endif

#define L      45
#define NPLANE 2025              /* 45*45 complex per x-plane                  */
#define PLND   4050              /* doubles per x-plane = x-stream stride      */
#define VDBL   ((size_t)2 * L * NPLANE)  /* doubles per volume = 182250        */
/* plane-scratch row pitch in complex.  52 complex = 832 B = 13 cache lines:
 * coprime with 64 sets, so the y-pass column walk spreads over all L1 sets
 * (pitch 48 = 12 lines measured 15-20% slower in r6, both entries). */
#ifndef PPITCH
# define PPITCH 52
#endif
/* padded mid-volume S for SCRATCHP: row pitch 52 complex (13 lines, odd),
 * plane pitch 45*52 = 2340 complex (585 lines, odd).  Odd-line pitches so no
 * fixed mod-4096 relation between phase 2's S reads (37440 B stride) and its
 * out writes (32400 B stride) can lock in -- the L23_rader aliasing rule. */
#ifndef SPITCH
# define SPITCH 52
#endif
#define SROWD  (2 * SPITCH)              /* S row stride, doubles = 104        */
#define SPLND  (2 * L * SPITCH)          /* S plane stride, doubles = 4680     */
#define SVDBL  ((size_t)L * SPLND)       /* doubles per padded S volume        */
/* CPY-mode plane image: lives in the P allocation right after the 45 padded
 * plane-scratch rows (offset 4680 doubles = 37440 B, still 64B-aligned).
 * Laid out exactly like one out plane (2*L-double rows, PLND doubles total)
 * so ONE rep movsb moves it. */
#define IMGOFF ((size_t)L * PPITCH * 2)

#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)

/* phase-1 input prefetch distance, doubles (32 KB) */
#ifndef FFT45_PFD
# define FFT45_PFD 4096
#endif
/* prefetch hint: 3=T0, 2=T1 (the measured winner at L=36/L=6), 0=NTA */
#ifndef FFT45_PFH
# define FFT45_PFH 2
#endif
/* cache lines of the NEXT volume's input prefetched per phase-2 tile */
#ifndef FFT45_PFN
# define FFT45_PFN 2
#endif
/* ip-pf3: write-intent cursor lead over the mid-plane stores, doubles */
#ifndef FFT45_PFWD
# define FFT45_PFWD 4050
#endif

/* sqrt(3)/2 (the DFT3 rotation, and the radix-3 spine of the 9-point DAG) */
#define KS3  0.86602540378443864676372317075294
/* genfft n1_9 (FMA form) DAG constants, fftw-3.3.10 n1_9.c */
#define K176 0.17632698070846497347109038686862  /* tan(pi/18)                */
#define K839 0.83909963117728001176312729812318
#define K777 0.77786191343020616002817797731863
#define K984 0.98480775301220805936674302458952
#define K492 0.49240387650610402968337151229476
#define K852 0.85286853195244320962825096394007
#define K363 0.36397023426620236135104788277683  /* tan(pi/9)                 */
#define K954 0.95418889413867113349926836418725
/* 5-point module constants (FFTW n1_5 form) */
#define K59  0.55901699437494742410229341718282   /* sqrt(5)/4                 */
#define KIG  0.61803398874989484820458683436564   /* sin(4pi/5)/sin(2pi/5)     */
#define KS5  0.95105651629515357211665325776975   /* sin(2pi/5)                */

/* exec variant signature: whole batch, so dispatch happens once per execute */
typedef void (*exec45_fn)(const double *in, double *out, long nvol,
                          double *S, double *P);

/* The 45-point Good-Thomas 9x5 codelet over PW interleaved-complex lanes.
 * LD(n) must yield input element n as a vec rvalue; ST(k, v) consumes output
 * element k.  Both index maps fold to compile-time constants once the loops
 * unroll.  ALL LDs (stage 1) complete before any ST (stage 2), so LD/ST may
 * alias freely.
 *
 * r11 STAGE ORDER SWAP (the r10 VERDICT 4.1 spill mechanism, adopted from
 * L45_mixedradix's ST1G/ST2G shape): stage 1 is now 9 x DFT5 (FFTW n1_5 FMA
 * form, 16 FMA-port ops + 2 swaps, SHORT live ranges) writing T_[9*k2 + n1];
 * stage 2 is 5 x DFT9 (genfft's FMA n1_9 DAG, 40 ops + 12 swaps, LONG live
 * ranges) reading 9 CONTIGUOUS hot slots and handing every output to ST the
 * moment it exists -- no DFT9 output ever parks in a named register, and the
 * T_ reads can fold into FMA memory operands.  r9-r10 ran the stages the
 * other way (5 x DFT9 into a stride-5-scattered array, then 9 x DFT5): same
 * 344 ops, but the DAG's q1/q2/a0/i0/S* held across three output blocks ON
 * TOP of pending array writes cost +23 stack moves where the rival's shape
 * pays +3, and the r10 VERDICT prices that difference at ~3 points of B=1.
 *
 * DFT9 DAG key (transcribed r9 from fftw-3.3.10 n1_9.c, 24 add + 56 fma =
 * 80 scalar FMA-port ops; each scalar re/im line pair is one vector op, each
 * re/im crossing one SWAP with signs folded into VPAIR constants).  Per
 * radix-3 column {n, n+3, n+6}: sJ_ = column sum, SJ_ = full sum,
 * aJ_ = xJ - sJ/2, iJ_ = SWAP(x(J+3) - x(J+6)); p/q = aJ -+ 866*i*e (the two
 * rotated DFT3 outputs).  k={0,3,6} is a DFT3 on the sums; k={1,4,7} and
 * k={2,5,8} build the (1 +- c*i) spiral factors w from p/q (one SWAP+FMA
 * each), cross them (u, z), and fan out through the 984 / 492+-852 split.
 * Total: 9*16 + 5*40 = 344 FMA-port ops + 78 shuffles per call, unchanged
 * since r9.  Output map k = (10*k1 + 36*k2) mod 45, input n = (5*n1 + 9*n2)
 * mod 45, both folded at compile time. */
#define PFA45(LD, ST) do {                                                   \
    vec T_[45];                            /* T_[9*k2 + n1] */               \
    _Pragma("GCC unroll 9")                                                  \
    for (int n1_ = 0; n1_ < 9; ++n1_) {                                      \
        vec x0_ = LD((5 * n1_     ) % 45);                                   \
        vec x1_ = LD((5 * n1_ +  9) % 45);                                   \
        vec x2_ = LD((5 * n1_ + 18) % 45);                                   \
        vec x3_ = LD((5 * n1_ + 27) % 45);                                   \
        vec x4_ = LD((5 * n1_ + 36) % 45);                                   \
        vec t1_ = x1_ + x4_, t4_ = x1_ - x4_;                                \
        vec t2_ = x2_ + x3_, t7_ = x2_ - x3_;                                \
        vec te_ = t1_ + t2_, ta_ = t1_ - t2_;                                \
        T_[n1_] = x0_ + te_;                                                 \
        vec tm_ = VFNMA(te_, VSPLAT(0.25), x0_);                             \
        vec tp_ = VFMA (ta_, VSPLAT(K59), tm_);                              \
        vec tq_ = VFNMA(ta_, VSPLAT(K59), tm_);                              \
        vec tv_ = VFMA (t7_, VSPLAT(KIG), t4_);                              \
        vec tw_ = VFNMA(t4_, VSPLAT(KIG), t7_);                              \
        vec sv_ = SWAP(tv_), sw_ = SWAP(tw_);                                \
        T_[ 9 + n1_] = VFMA (sv_, VPAIR(KS5, -KS5), tp_);                    \
        T_[18 + n1_] = VFNMA(sw_, VPAIR(KS5, -KS5), tq_);                    \
        T_[27 + n1_] = VFMA (sw_, VPAIR(KS5, -KS5), tq_);                    \
        T_[36 + n1_] = VFNMA(sv_, VPAIR(KS5, -KS5), tp_);                    \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k2_ = 0; k2_ < 5; ++k2_) {                                      \
        const vec *f_ = T_ + 9 * k2_;                                        \
        vec s0_ = f_[3] + f_[6], e0_ = f_[3] - f_[6];                        \
        vec S0_ = f_[0] + s0_,  a0_ = VFNMA(s0_, VSPLAT(0.5), f_[0]);        \
        vec i0_ = SWAP(e0_);                                                 \
        vec s1_ = f_[4] + f_[7], e1_ = f_[4] - f_[7];                        \
        vec S1_ = f_[1] + s1_,  a1_ = VFNMA(s1_, VSPLAT(0.5), f_[1]);        \
        vec i1_ = SWAP(e1_);                                                 \
        vec p1_ = VFMA (i1_, VPAIR(KS3, -KS3), a1_);                         \
        vec q1_ = VFNMA(i1_, VPAIR(KS3, -KS3), a1_);                         \
        vec s2_ = f_[5] + f_[8], e2_ = f_[5] - f_[8];                        \
        vec S2_ = f_[2] + s2_,  a2_ = VFNMA(s2_, VSPLAT(0.5), f_[2]);        \
        vec i2_ = SWAP(e2_);                                                 \
        vec p2_ = VFMA (i2_, VPAIR(KS3, -KS3), a2_);                         \
        vec q2_ = VFNMA(i2_, VPAIR(KS3, -KS3), a2_);                         \
        /* k1 = 0, 3, 6: DFT3 on the column sums */                          \
        vec sg_ = S1_ + S2_, d3_ = S2_ - S1_, id_ = SWAP(d3_);               \
        vec b0_ = VFNMA(sg_, VSPLAT(0.5), S0_);                              \
        ST((36 * k2_      ) % 45, S0_ + sg_);                                \
        ST((36 * k2_ + 30) % 45, VFNMA(id_, VPAIR(KS3, -KS3), b0_));         \
        ST((36 * k2_ + 60) % 45, VFMA (id_, VPAIR(KS3, -KS3), b0_));         \
        /* k1 = 1, 4, 7 */                                                   \
        {                                                                    \
        vec v1_ = VFMA (i0_, VPAIR(KS3, -KS3), a0_);                         \
        vec w2_ = VFMA (SWAP(p2_), VPAIR(-K176, K176), p2_);                 \
        vec w1_ = VFMA (SWAP(p1_), VPAIR(K839, -K839), p1_);                 \
        vec u1_ = VFMA (w1_, VPAIR(K777, -K777), SWAP(w2_));                 \
        vec z1_ = VFMA (SWAP(w1_), VPAIR(K777, -K777), w2_);                 \
        ST((36 * k2_ + 10) % 45, VFMA (u1_, VPAIR(K984, -K984), v1_));       \
        vec r1_ = VFNMA(u1_, VPAIR(K492, -K492), v1_);                       \
        ST((36 * k2_ + 40) % 45, VFMA (z1_, VSPLAT(K852), r1_));             \
        ST((36 * k2_ + 70) % 45, VFNMA(z1_, VSPLAT(K852), r1_));             \
        }                                                                    \
        /* k1 = 2, 5, 8 */                                                   \
        {                                                                    \
        vec v2_ = VFNMA(i0_, VPAIR(KS3, -KS3), a0_);                         \
        vec wA_ = VFMA (q1_, VPAIR(K176, -K176), SWAP(q1_));                 \
        vec wB_ = VFNMA(SWAP(q2_), VPAIR(K363, -K363), q2_);                 \
        vec uB_ = VFMA (wB_, VPAIR(-K954, K954), wA_);                       \
        vec zB_ = VFMA (SWAP(wB_), VPAIR(-K954, K954), SWAP(wA_));           \
        ST((36 * k2_ + 20) % 45, VFMA (uB_, VPAIR(K984, -K984), v2_));       \
        vec rB_ = VFNMA(uB_, VPAIR(K492, -K492), v2_);                       \
        ST((36 * k2_ + 50) % 45, VFNMA(zB_, VSPLAT(K852), rB_));             \
        ST((36 * k2_ + 80) % 45, VFMA (zB_, VSPLAT(K852), rB_));             \
        }                                                                    \
    }                                                                        \
} while (0)

/* Register-array variant of the same codelet: the r9-r10 stage order
 * (5 x DFT9 into A_[5*k1 + n2], then 9 x DFT5).  Kept ONLY for the z-site,
 * where ST writes a vec array (Wv) that is transposed afterwards, not
 * memory: there the store-direct order above measures WORSE (per-function
 * audit under node flags, phase1_plane_pw4 spills 83 -> 91) because the
 * DFT9's nine scattered array writes pile onto its long live ranges, while
 * at the memory-ST sites the same order deletes spills (phase2_pw4
 * 53/55 -> 46/51).  Same DAG, same 344 ops + 78 swaps, same maps. */
#define DFT9F(x0,x1,x2,x3,x4,x5,x6,x7,x8, o0,o1,o2,o3,o4,o5,o6,o7,o8) do {   \
    vec s0_ = (x3) + (x6), e0_ = (x3) - (x6);                                \
    vec S0_ = (x0) + s0_,  a0_ = VFNMA(s0_, VSPLAT(0.5), (x0));              \
    vec i0_ = SWAP(e0_);                                                     \
    vec s1_ = (x4) + (x7), e1_ = (x4) - (x7);                                \
    vec S1_ = (x1) + s1_,  a1_ = VFNMA(s1_, VSPLAT(0.5), (x1));              \
    vec i1_ = SWAP(e1_);                                                     \
    vec p1_ = VFMA (i1_, VPAIR(KS3, -KS3), a1_);                             \
    vec q1_ = VFNMA(i1_, VPAIR(KS3, -KS3), a1_);                             \
    vec s2_ = (x5) + (x8), e2_ = (x5) - (x8);                                \
    vec S2_ = (x2) + s2_,  a2_ = VFNMA(s2_, VSPLAT(0.5), (x2));              \
    vec i2_ = SWAP(e2_);                                                     \
    vec p2_ = VFMA (i2_, VPAIR(KS3, -KS3), a2_);                             \
    vec q2_ = VFNMA(i2_, VPAIR(KS3, -KS3), a2_);                             \
    vec sg_ = S1_ + S2_, d3_ = S2_ - S1_, id_ = SWAP(d3_);                   \
    vec b0_ = VFNMA(sg_, VSPLAT(0.5), S0_);                                  \
    (o0) = S0_ + sg_;                                                        \
    (o3) = VFNMA(id_, VPAIR(KS3, -KS3), b0_);                                \
    (o6) = VFMA (id_, VPAIR(KS3, -KS3), b0_);                                \
    {                                                                        \
    vec v1_ = VFMA (i0_, VPAIR(KS3, -KS3), a0_);                             \
    vec w2_ = VFMA (SWAP(p2_), VPAIR(-K176, K176), p2_);                     \
    vec w1_ = VFMA (SWAP(p1_), VPAIR(K839, -K839), p1_);                     \
    vec u1_ = VFMA (w1_, VPAIR(K777, -K777), SWAP(w2_));                     \
    vec z1_ = VFMA (SWAP(w1_), VPAIR(K777, -K777), w2_);                     \
    (o1) = VFMA (u1_, VPAIR(K984, -K984), v1_);                              \
    vec r1_ = VFNMA(u1_, VPAIR(K492, -K492), v1_);                           \
    (o4) = VFMA (z1_, VSPLAT(K852), r1_);                                    \
    (o7) = VFNMA(z1_, VSPLAT(K852), r1_);                                    \
    }                                                                        \
    {                                                                        \
    vec v2_ = VFNMA(i0_, VPAIR(KS3, -KS3), a0_);                             \
    vec wA_ = VFMA (q1_, VPAIR(K176, -K176), SWAP(q1_));                     \
    vec wB_ = VFNMA(SWAP(q2_), VPAIR(K363, -K363), q2_);                     \
    vec uB_ = VFMA (wB_, VPAIR(-K954, K954), wA_);                           \
    vec zB_ = VFMA (SWAP(wB_), VPAIR(-K954, K954), SWAP(wA_));               \
    (o2) = VFMA (uB_, VPAIR(K984, -K984), v2_);                              \
    vec rB_ = VFNMA(uB_, VPAIR(K492, -K492), v2_);                           \
    (o5) = VFNMA(zB_, VSPLAT(K852), rB_);                                    \
    (o8) = VFMA (zB_, VSPLAT(K852), rB_);                                    \
    }                                                                        \
} while (0)

#define PFA45R(LD, ST) do {                                                  \
    vec A_[45];                            /* A_[5*k1 + n2] */               \
    _Pragma("GCC unroll 5")                                                  \
    for (int n2_ = 0; n2_ < 5; ++n2_) {                                      \
        vec g_[9];                                                           \
        _Pragma("GCC unroll 9")                                              \
        for (int n1_ = 0; n1_ < 9; ++n1_)                                    \
            g_[n1_] = LD((5 * n1_ + 9 * n2_) % 45);                          \
        DFT9F(g_[0], g_[1], g_[2], g_[3], g_[4], g_[5], g_[6], g_[7], g_[8],\
              A_[      n2_], A_[ 5 + n2_], A_[10 + n2_],                     \
              A_[15 + n2_], A_[20 + n2_], A_[25 + n2_],                      \
              A_[30 + n2_], A_[35 + n2_], A_[40 + n2_]);                     \
    }                                                                        \
    _Pragma("GCC unroll 9")                                                  \
    for (int k1_ = 0; k1_ < 9; ++k1_) {                                      \
        const vec *f_ = A_ + 5 * k1_;                                        \
        vec t1_ = f_[1] + f_[4], t4_ = f_[1] - f_[4];                        \
        vec t2_ = f_[2] + f_[3], t7_ = f_[2] - f_[3];                        \
        vec te_ = t1_ + t2_,     ta_ = t1_ - t2_;                            \
        ST((10 * k1_      ) % 45, f_[0] + te_);                              \
        vec tm_ = VFNMA(te_, VSPLAT(0.25), f_[0]);                           \
        vec tp_ = VFMA (ta_, VSPLAT(K59), tm_);                              \
        vec tq_ = VFNMA(ta_, VSPLAT(K59), tm_);                              \
        vec tv_ = VFMA (t7_, VSPLAT(KIG), t4_);                              \
        vec tw_ = VFNMA(t4_, VSPLAT(KIG), t7_);                              \
        vec sv_ = SWAP(tv_), sw_ = SWAP(tw_);                                \
        ST((10 * k1_ +  36) % 45, VFMA (sv_, VPAIR(KS5, -KS5), tp_));        \
        ST((10 * k1_ +  72) % 45, VFNMA(sw_, VPAIR(KS5, -KS5), tq_));        \
        ST((10 * k1_ + 108) % 45, VFMA (sw_, VPAIR(KS5, -KS5), tq_));        \
        ST((10 * k1_ + 144) % 45, VFNMA(sv_, VPAIR(KS5, -KS5), tp_));        \
    }                                                                        \
} while (0)

/* PW=1 instantiation of the same codelet: ONE complex per 128-bit vector,
 * for the odd 45th row/column of phase 1's subpasses (45 = 11*4 + 1).
 * VEX-coded xmm FMAs dual-issue on ports 0 AND 1 on the node (CLX runs two
 * 256-bit-or-narrower FMA pipes; only 512-bit is single-ported), and every
 * access is a 16 B complex that never splits a cache line.  Strides are in
 * DOUBLES.  Called ~90x per volume; gcc may inline it into the exec
 * variants or not -- either is fine (front end measured non-binding, r9). */
typedef double    v1d __attribute__((vector_size(16), aligned(16)));
typedef long long v1i __attribute__((vector_size(16)));

static void dft45_line1(const double *restrict s, const long sstr,
                        double *restrict d, const long dstr)
{
#define vec        v1d
#define VSPLAT(a)  ((vec){(a),(a)})
#define VPAIR(a,b) ((vec){(a),(b)})
#ifdef __clang__
# define SWAP(v)   __builtin_shufflevector((v),(v), 1,0)
#else
# define SWAP(v)   __builtin_shuffle((v),(v),(v1i){1,0})
#endif
#ifdef __FMA__
# define VFMA(a,b,c)  ((vec)_mm_fmadd_pd((__m128d)(a),(__m128d)(b),(__m128d)(c)))
# define VFNMA(a,b,c) ((vec)_mm_fnmadd_pd((__m128d)(a),(__m128d)(b),(__m128d)(c)))
#else
# define VFMA(a,b,c)  ((a)*(b) + (c))
# define VFNMA(a,b,c) ((c) - (a)*(b))
#endif
#define LDL1(n)    (*(const vec *)(s + (size_t)(n) * sstr))
#define STL1(k, v) (*(vec *)(d + (size_t)(k) * dstr) = (v))
    PFA45(LDL1, STL1);
#undef LDL1
#undef STL1
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef vec
}

/* ice_r5 tr twin of dft45_line1 for the odd 45th kz column: loads address
 * the slot-major z-scratch (element y sits at slot (y>>2)*45 + 44, 16 B
 * granule y&3 -- every offset folds at compile time), stores to the mid
 * column kz=44 at dstr doubles.  Same PFA45 order as dft45_line1, so the
 * output bits match the classic pl-column tail exactly.  (AVX512-gated only
 * because its sole caller, ysubt, is: keeps -Wall clean elsewhere.) */
#ifdef __AVX512F__
static void dft45_line1t(const double *restrict s, double *restrict d,
                         const long dstr)
{
#define vec        v1d
#define VSPLAT(a)  ((vec){(a),(a)})
#define VPAIR(a,b) ((vec){(a),(b)})
#ifdef __clang__
# define SWAP(v)   __builtin_shufflevector((v),(v), 1,0)
#else
# define SWAP(v)   __builtin_shuffle((v),(v),(v1i){1,0})
#endif
#ifdef __FMA__
# define VFMA(a,b,c)  ((vec)_mm_fmadd_pd((__m128d)(a),(__m128d)(b),(__m128d)(c)))
# define VFNMA(a,b,c) ((vec)_mm_fnmadd_pd((__m128d)(a),(__m128d)(b),(__m128d)(c)))
#else
# define VFMA(a,b,c)  ((a)*(b) + (c))
# define VFNMA(a,b,c) ((c) - (a)*(b))
#endif
#define LDL1T(n)    (*(const vec *)(s + (size_t)((n) >> 2) * (8 * 45)         \
                                      + (8 * 44) + 2 * ((n) & 3)))
#define STL1T(k, v) (*(vec *)(d + (size_t)(k) * dstr) = (v))
    PFA45(LDL1T, STL1T);
#undef LDL1T
#undef STL1T
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef vec
}
#endif /* __AVX512F__ */

/* one ERMS rep-movsb per x-plane (CPY mode; BORROWED from L45_mixedradix r8,
 * their plane_copy verbatim): ERMS full-cache-line writes do not
 * read-for-ownership the destination, so the 1.46 MB/volume RFO read of
 * L3-resident `out` disappears from phase 1's traffic, while the lines still
 * land in the cache hierarchy for phase 2 to re-read (unlike NT stores). */
static inline void plane_copy(void *dst, const void *src, size_t n)
{
#if defined(__x86_64__)
    void *d = dst;
    const void *s = src;
    size_t c = n;
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(c) : : "memory");
#else
    memcpy(dst, src, n);
#endif
}

/* ---- ice_r4: the graded-step map, state = (z+c)/(1+|z+c|) ---------------
 * Canonical arithmetic, chosen so every vector width and grouping computes
 * BIT-IDENTICAL per-point results (all chain arms must agree exactly):
 *   t  = fma(w, w, 1e-300)            per lane: re^2+g / im^2+g
 *   s  = t + swap(t)                  re^2+im^2+2g, same bits in both lanes
 *   y  = rsqrt14(s); 2x Newton        y = 1/sqrt(s) to < 1 ulp
 *   d  = fma(s, y, 1.0)               1 + |w|
 *   q  = 1.0 / d                      ONE correctly-rounded divide
 *   o  = w * q
 * The guard makes s >= 2e-300, so rsqrt14 never sees 0 (inf would NaN the
 * ladder); for any representable nonzero w the guard is beyond the 53rd bit
 * of s, and for w = 0 the result is exactly 0 (q rounds to 1).  The pair
 * form packs two interleaved vectors' s (even lanes) into one zmm: one
 * ladder + one divide per 8 points, then unpack + two mul-outs.  Packing
 * changes no per-point values, so pair/one/xmm forms agree bitwise. */
#ifndef FFT45_MS
# define FFT45_MS 0     /* 0: one vdivpd per 8 pts;  1: rcp14 + 2-Newton    */
#endif
#define MG45 1.0e-300

/* scalar map, full sqrt+div: the create()-time reference, and the no-AVX512
 * fallback chain.  n is in complex elements; o may alias z. */
static void map45_ref(const double *z, const double *c, double *o, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        double re = z[2 * i] + c[2 * i], im = z[2 * i + 1] + c[2 * i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        o[2 * i] = re * sc;
        o[2 * i + 1] = im * sc;
    }
}

#ifdef __AVX512F__
static inline __attribute__((always_inline))
__m512d map45_q8(__m512d S)             /* S: 8 values -> 1/(1+sqrt(S))      */
{
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d k32 = _mm512_set1_pd(1.5);
    __m512d y = _mm512_rsqrt14_pd(S);
    __m512d h = _mm512_mul_pd(S, _mm512_set1_pd(0.5));
    __m512d t = _mm512_mul_pd(y, y);
    y = _mm512_mul_pd(y, _mm512_fnmadd_pd(h, t, k32));
    t = _mm512_mul_pd(y, y);
    y = _mm512_mul_pd(y, _mm512_fnmadd_pd(h, t, k32));
    __m512d d = _mm512_fmadd_pd(S, y, one);
#if FFT45_MS
    __m512d x = _mm512_rcp14_pd(d);
    x = _mm512_fmadd_pd(x, _mm512_fnmadd_pd(d, x, one), x);
    x = _mm512_fmadd_pd(x, _mm512_fnmadd_pd(d, x, one), x);
    return x;
#else
    return _mm512_div_pd(one, d);
#endif
}

static inline __attribute__((always_inline))
__m512d map45_s(__m512d w)              /* |w|^2 + 2e-300, dup in both lanes */
{
    __m512d t = _mm512_fmadd_pd(w, w, _mm512_set1_pd(MG45));
    return _mm512_add_pd(t, _mm512_permute_pd(t, 0x55));
}

/* The trailing asm barriers pin the map's FINAL MULTIPLY as a rounded mul:
 * gcc's default -ffp-contract=fast otherwise CONTRACTS o = w*q into the
 * consuming codelet's first adds (x1_+x4_ -> vfmadd) when the map inlines
 * into zsub, silently changing the mapped state by 1 ulp vs the stored
 * form -- which broke the cross-arm bit-identity gate (uf/vm-pp BIT FAIL,
 * first diagnosed ice_r4).  A barrier costs zero uops. */
static inline __attribute__((always_inline))
void map45_pair(__m512d *w0, __m512d *w1)   /* 8 complex, ONE divide        */
{
    __m512d S = _mm512_permutex2var_pd(map45_s(*w0),
                    _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14),
                    map45_s(*w1));
    __m512d q = map45_q8(S);
    __m512d a0 = _mm512_mul_pd(*w0,
            _mm512_permutexvar_pd(_mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3), q));
    __m512d a1 = _mm512_mul_pd(*w1,
            _mm512_permutexvar_pd(_mm512_setr_epi64(4, 4, 5, 5, 6, 6, 7, 7), q));
    __asm__("" : "+v"(a0), "+v"(a1));
    *w0 = a0;
    *w1 = a1;
}

static inline __attribute__((always_inline))
__m512d map45_one(__m512d w)            /* 4 complex (tails, GCOL column)    */
{
    __m512d r = _mm512_mul_pd(w, map45_q8(map45_s(w)));
    __asm__("" : "+v"(r));
    return r;
}

#ifdef __AVX512VL__
static inline __attribute__((always_inline))
__m128d map45_1x(__m128d w)             /* 1 complex, same per-point bits    */
{
    const __m128d one = _mm_set1_pd(1.0), k32 = _mm_set1_pd(1.5);
    __m128d t = _mm_fmadd_pd(w, w, _mm_set1_pd(MG45));
    __m128d S = _mm_add_pd(t, _mm_permute_pd(t, 1));
    __m128d y = _mm_rsqrt14_pd(S);
    __m128d h = _mm_mul_pd(S, _mm_set1_pd(0.5));
    __m128d u = _mm_mul_pd(y, y);
    y = _mm_mul_pd(y, _mm_fnmadd_pd(h, u, k32));
    u = _mm_mul_pd(y, y);
    y = _mm_mul_pd(y, _mm_fnmadd_pd(h, u, k32));
    __m128d d = _mm_fmadd_pd(S, y, one);
#if FFT45_MS
    __m128d x = _mm_rcp14_pd(d);
    x = _mm_fmadd_pd(x, _mm_fnmadd_pd(d, x, one), x);
    x = _mm_fmadd_pd(x, _mm_fnmadd_pd(d, x, one), x);
    __m128d r = _mm_mul_pd(w, x);
#else
    __m128d r = _mm_mul_pd(w, _mm_div_pd(one, d));
#endif
    __asm__("" : "+v"(r));
    return r;
}
#else
static inline __m128d map45_1x(__m128d w)   /* no VL: zero-extended 512 form */
{
    return _mm512_castpd512_pd128(map45_one(_mm512_zextpd128_pd512(w)));
}
#endif

/* map a contiguous span of n complex: o = map(z + c); o may alias z.
 * noinline: called from inside zsub/zsubt -- keep the big bodies lean.
 * noclone (ice_r5): zsubt's constant-n row-group calls otherwise make gcc
 * emit a .constprop clone whose object-size loop bound trips a spurious
 * -Waggressive-loop-optimizations warning; one shared copy is also what the
 * store-to-load-distance argument in zsubt assumes. */
static __attribute__((noinline, noclone))
void map45_span(const double *z, const double *c, double *o, long n)
{
    long i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d w0 = _mm512_add_pd(_mm512_loadu_pd(z + 2 * i),
                                   _mm512_loadu_pd(c + 2 * i));
        __m512d w1 = _mm512_add_pd(_mm512_loadu_pd(z + 2 * i + 8),
                                   _mm512_loadu_pd(c + 2 * i + 8));
        map45_pair(&w0, &w1);
        _mm512_storeu_pd(o + 2 * i, w0);
        _mm512_storeu_pd(o + 2 * i + 8, w1);
    }
    if (i + 4 <= n) {
        __m512d w = _mm512_add_pd(_mm512_loadu_pd(z + 2 * i),
                                  _mm512_loadu_pd(c + 2 * i));
        _mm512_storeu_pd(o + 2 * i, map45_one(w));
        i += 4;
    }
    for (; i < n; ++i) {
        __m128d w = _mm_add_pd(_mm_loadu_pd(z + 2 * i), _mm_loadu_pd(c + 2 * i));
        _mm_storeu_pd(o + 2 * i, map45_1x(w));
    }
}
#endif /* __AVX512F__ */

/* instantiate the kernel template: PW=2, and PW=4 on AVX-512 */
#define PW 2
#include __FILE__
#undef PW
#ifdef __AVX512F__
# define PW 4
# include __FILE__
# undef PW
# define HAVE_PW4 1
#endif

/* ---- ice_r4: the fused chain (fft3d_chain) -------------------------------
 * final_out is the state arena: raw z lives there between steps (LAZY map,
 * applied inside the next step's zsub loads), one map45_span finishes.
 * Every arm below computes BIT-IDENTICAL results (the map is per-point
 * deterministic under any grouping; the FFT arithmetic per line never
 * changes), so create() may race them without risking the repeatability
 * check.  Workhorses are noinline so five arms share one copy of the big
 * inlined phase bodies. */
#ifdef HAVE_PW4
/* ice_r5: the chain runs in a plan-owned state arena W whose mod-4096 phase
 * is DERIVED FROM c ((phase(c)+2048)&4095: maximally far from c's page
 * phase for every volume, since both stride VDBL).  BORROWED from
 * L45_mixedradix ice_r4 (their fix for the state-vs-c 4K false-dependence
 * lottery: 294.1 vs 333.0 us/xform across runs of ONE binary, in-run sd
 * 0.05%), transitively L23_rader ice_r4.  final_out is written ONCE, by the
 * last step's map span (its only RFO); x0 is read once.  W==final_out is
 * the alloc-failure fallback (= the ice_r4 behavior). */
typedef void (*chain45_fn)(const double *x0, const double *C, double *W,
                           double *FO, long m, long B, double *P);

static __attribute__((noinline))
void ch45_p1_plane(const double *src, double *F, double *P, int x)
{
    phase1_plane_pw4(src, F, P, x, 0, 0, 0, 2 * L, PLND, 0, 0);
}

static __attribute__((noinline))
void ch45_p1m_plane(double *F, const double *cv, double *P, int x)
{
    /* in place: reads F's plane x (raw z + map), writes it back transformed */
    phase1_plane_pw4(F, F, P, x, 0, 0, 0, 2 * L, PLND, 1, cv);
}

/* ice_r5 tr twins: transpose-free phase 1 (broadcast gathers + slot-major
 * scratch), lazy map staged per row group inside zsubt.  In-place safe by
 * the same argument as the classic pair: zsubt drains plane x into the slot
 * scratch before ysubt's first store back to plane x. */
static __attribute__((noinline))
void ch45_p1t_plane(const double *src, double *F, double *P, int x)
{
    zsubt_pw4(src + (size_t)x * PLND, P, 0, 0);
    ysubt_pw4(P, F + (size_t)x * PLND, 2 * L);
}

static __attribute__((noinline))
void ch45_p1tm_plane(double *F, const double *cv, double *P, int x)
{
    zsubt_pw4(F + (size_t)x * PLND, P, 1, cv + (size_t)x * PLND);
    ysubt_pw4(P, F + (size_t)x * PLND, 2 * L);
}

static __attribute__((noinline))
void ch45_p2_chunk(double *F, int t0, int t1)
{
    phase2_chunk_pw4(F, F, 0, 0, 0, t0, t1);
}

#define CH45_NT (NPLANE / 4)              /* 506 flat tiles + masked tail    */

static void ch45_p2(double *F) { ch45_p2_chunk(F, 0, CH45_NT); }

/* vt-pf's phase 2: the 45-stream one-line-ahead T0 poke (PF45).  BORROWED:
 * L45_mixedradix ice_r4 measured the poke on the same-shape RMW pass worth
 * ~11 us/xform under the chain (287.1 vs 298.3), and L36_pencilfused ice_r5
 * re-proved the 36-stream analog load-bearing (113.4 vs 115.5) -- the one
 * prefetch flavor that survives on cache-resident chains.  My ice_r2 "pf1
 * is a tax" number was the UNFUSED exec at B=4; the race decides. */
static __attribute__((noinline))
void ch45_p2f(double *F)
{
    phase2_chunk_pw4(F, F, 0, 1, 0, 0, CH45_NT);
}

/* vt: volume-major (corpus 10 s3), tr phase 1, lazy row-group map, chain in
 * W, final map span -> FO.  pfx selects the phase-2 poke variant. */
static inline __attribute__((always_inline))
void ch45_vt_body(const double *x0, const double *C, double *W, double *FO,
                  long m, long B, double *P, const int pfx)
{
    for (long b = 0; b < B; ++b) {
        const double *xb = x0 + (size_t)b * VDBL;
        const double *cb = C  + (size_t)b * VDBL;
        double       *Wb = W  + (size_t)b * VDBL;
        for (int x = 0; x < L; ++x) ch45_p1t_plane(xb, Wb, P, x);
        if (pfx) ch45_p2f(Wb); else ch45_p2(Wb);
        for (long s = 1; s < m; ++s) {
            for (int x = 0; x < L; ++x) ch45_p1tm_plane(Wb, cb, P, x);
            if (pfx) ch45_p2f(Wb); else ch45_p2(Wb);
        }
        map45_span(Wb, cb, FO + (size_t)b * VDBL, (long)L * NPLANE);
    }
}

static void ch45_vt_zs(const double *x0, const double *C, double *W,
                       double *FO, long m, long B, double *P)
{
    ch45_vt_body(x0, C, W, FO, m, B, P, 0);
}

static void ch45_vt_pf(const double *x0, const double *C, double *W,
                       double *FO, long m, long B, double *P)
{
    ch45_vt_body(x0, C, W, FO, m, B, P, 1);
}

/* vm-zs: the ice_r4 incumbent (classic zsub with the register-fused lazy
 * map), kept as the priced fallback should tr misbehave on the node.
 * (The ice_r4 vp/bm/pp arms are DELETED: the r4 drained window priced them
 * at vp=315.9 bm=317.3 pp=310.3 vs vm=301.8 us/step -- recorded, closed.) */
static void ch45_vm_zs(const double *x0, const double *C, double *W,
                       double *FO, long m, long B, double *P)
{
    for (long b = 0; b < B; ++b) {
        const double *xb = x0 + (size_t)b * VDBL;
        const double *cb = C  + (size_t)b * VDBL;
        double       *Wb = W  + (size_t)b * VDBL;
        for (int x = 0; x < L; ++x) ch45_p1_plane(xb, Wb, P, x);
        ch45_p2(Wb);
        for (long s = 1; s < m; ++s) {
            for (int x = 0; x < L; ++x) ch45_p1m_plane(Wb, cb, P, x);
            ch45_p2(Wb);
        }
        map45_span(Wb, cb, FO + (size_t)b * VDBL, (long)L * NPLANE);
    }
}

/* uf: unfused control -- plain exec + streaming map pass per step.
 * Prices the fusion itself (L13_rader ice_r4: this WON at their B=1). */
static void ch45_uf(const double *x0, const double *C, double *W,
                    double *FO, long m, long B, double *P)
{
    for (long s = 0; s < m; ++s) {
        for (long b = 0; b < B; ++b) {
            const double *src = s ? W + (size_t)b * VDBL
                                  : x0 + (size_t)b * VDBL;
            double *Wb = W + (size_t)b * VDBL;
            for (int x = 0; x < L; ++x) ch45_p1_plane(src, Wb, P, x);
            ch45_p2(Wb);
        }
        map45_span(W, C, (s + 1 < m) ? W : FO, (long)B * L * NPLANE);
    }
}

/* ================= ice_r6: SPLIT-COMPLEX (SoA) CHAIN ARMS ==================
 * The chain OWNS the state format between steps (L17_winograd ice_r5's
 * licence; L64_radix8 runs whole-pipeline split-complex): only the x0/c
 * converts and the final map touch the driver's interleaved layout.  The vs
 * arms keep the RAW z state SPLIT: per plane, a 45x48 re array then a 45x48
 * im array (y-major rows, kz minor, rows padded 45 -> 48 doubles = 6 whole
 * cache lines so every y-block base is 64B-aligned; pad columns kz=45..47
 * stay 0 forever).  What this deletes vs the vt arms, per volume-step:
 * ALL 116,766 codelet SWAPs (the entire FFT p5-only term, ~40 us of p5
 * floor -- every SWAP+VPAIR crossing becomes a plain FMA on the other
 * component with the sign folded), and the map's pack/unpack permutes
 * (split s = fma(re,re,g)+fma(im,im,g) needs no lane shuffles at all).
 * What it pays: scalar (8 B) broadcast gathers instead of 16 B ones at the
 * two corner turns -- BOTH confined to L1 buffers (the map ping-pong and
 * the plane scratch); every access to the L2/L3-resident state is a
 * full-width 64B-ALIGNED vector load/store -- plus a 6.7% codelet-lane
 * waste (45 rows/cols ride 6 blocks of 8; z side: 5 full + masked-4 + a
 * scalar 45th line so the bit classes below hold).
 *
 * BIT-IDENTITY with the vt/vm/uf arms, BY CONSTRUCTION: the split codelets
 * are per-component transcriptions of the SAME DAGs the interleaved
 * vectors execute per lane -- PFA45RS transcribes PFA45R (the z-site
 * keeps the DFT9-first factor order, incl. rows 40-43 through it exactly
 * as vt's group 10), PFA45S transcribes PFA45 (y/x sites and the scalar
 * z-line for row 44, which vt runs through dft45_line1 = PFA45 order).
 * Every VFMA(SWAP(a), VPAIR(k,-k), c) becomes re=fma(a_im,k,c_re),
 * im=fnma(a_re,k,c_im) (IEEE: identical per-lane rounding); the map is
 * the same op sequence (s = (re^2+g)+(im^2+g), one q8 ladder + divide per
 * 8 points, o = w*q behind the same asm barrier).  So the cross-arm
 * memcmp gate and the cross-process repeatability contract survive with
 * vs in the pool -- a transcription bug shows up as a gate DQ, not as a
 * shipped wrong answer. */
#define SPROW 48                       /* split row pitch, doubles          */
#define SPCO  (L * SPROW)              /* 2160: im offset within a plane    */
#define SPPL  (2 * SPCO)               /* 4320 doubles per split plane      */
#define SPVOL ((size_t)L * SPPL)       /* 194400 doubles per split volume   */
#define SPVST 194560                   /* volume stride: 380*512, 4096B mult */

/* masked scalar-broadcast gathers: vbroadcastsd (mem), zmm{k} folds to ONE
 * load-port uop on this node (same bare-metal proviso as the 16 B G4B form,
 * L36_pfa ice_r2 / corpus 10).  8-deep merge chains are independent across
 * the 45 codelet inputs, so OoO overlaps them like G4B's 4-deep ones. */
static inline __attribute__((always_inline))
__m512d g8b8(const double *p, const long str)
{
    __m512d v = _mm512_broadcastsd_pd(_mm_load_sd(p));
    v = _mm512_mask_broadcastsd_pd(v, 0x02, _mm_load_sd(p +     str));
    v = _mm512_mask_broadcastsd_pd(v, 0x04, _mm_load_sd(p + 2 * str));
    v = _mm512_mask_broadcastsd_pd(v, 0x08, _mm_load_sd(p + 3 * str));
    v = _mm512_mask_broadcastsd_pd(v, 0x10, _mm_load_sd(p + 4 * str));
    v = _mm512_mask_broadcastsd_pd(v, 0x20, _mm_load_sd(p + 5 * str));
    v = _mm512_mask_broadcastsd_pd(v, 0x40, _mm_load_sd(p + 6 * str));
    v = _mm512_mask_broadcastsd_pd(v, 0x80, _mm_load_sd(p + 7 * str));
    return v;
}
static inline __attribute__((always_inline))
__m512d g8b4(const double *p, const long str)   /* lanes 0-3, rest ZERO */
{
    __m512d v = _mm512_maskz_broadcastsd_pd(0x01, _mm_load_sd(p));
    v = _mm512_mask_broadcastsd_pd(v, 0x02, _mm_load_sd(p +     str));
    v = _mm512_mask_broadcastsd_pd(v, 0x04, _mm_load_sd(p + 2 * str));
    v = _mm512_mask_broadcastsd_pd(v, 0x08, _mm_load_sd(p + 3 * str));
    return v;
}
static inline __attribute__((always_inline))
__m512d g8b5(const double *p, const long str)   /* lanes 0-4, rest ZERO */
{
    __m512d v = _mm512_maskz_broadcastsd_pd(0x01, _mm_load_sd(p));
    v = _mm512_mask_broadcastsd_pd(v, 0x02, _mm_load_sd(p +     str));
    v = _mm512_mask_broadcastsd_pd(v, 0x04, _mm_load_sd(p + 2 * str));
    v = _mm512_mask_broadcastsd_pd(v, 0x08, _mm_load_sd(p + 3 * str));
    v = _mm512_mask_broadcastsd_pd(v, 0x10, _mm_load_sd(p + 4 * str));
    return v;
}

/* PFA45S: split-complex transcription of PFA45 (DFT5-first, store-direct).
 * PFA45RS: transcription of PFA45R (DFT9-first; the z-site bit class).
 * GVEC/GADD/GSUB/GFMA/GFNMA are bound before each instantiation. */
#define PFA45S(LDR, LDI, STR, STI) do {                                      \
    GVEC Tr_[45], Ti_[45];                     /* T[9*k2 + n1] */            \
    _Pragma("GCC unroll 9")                                                  \
    for (int n1_ = 0; n1_ < 9; ++n1_) {                                      \
        GVEC x0r_=LDR((5*n1_   )%45), x0i_=LDI((5*n1_   )%45);               \
        GVEC x1r_=LDR((5*n1_+ 9)%45), x1i_=LDI((5*n1_+ 9)%45);               \
        GVEC x2r_=LDR((5*n1_+18)%45), x2i_=LDI((5*n1_+18)%45);               \
        GVEC x3r_=LDR((5*n1_+27)%45), x3i_=LDI((5*n1_+27)%45);               \
        GVEC x4r_=LDR((5*n1_+36)%45), x4i_=LDI((5*n1_+36)%45);               \
        GVEC t1r_=GADD(x1r_,x4r_), t1i_=GADD(x1i_,x4i_);                     \
        GVEC t4r_=GSUB(x1r_,x4r_), t4i_=GSUB(x1i_,x4i_);                     \
        GVEC t2r_=GADD(x2r_,x3r_), t2i_=GADD(x2i_,x3i_);                     \
        GVEC t7r_=GSUB(x2r_,x3r_), t7i_=GSUB(x2i_,x3i_);                     \
        GVEC ter_=GADD(t1r_,t2r_), tei_=GADD(t1i_,t2i_);                     \
        GVEC tar_=GSUB(t1r_,t2r_), tai_=GSUB(t1i_,t2i_);                     \
        Tr_[n1_]=GADD(x0r_,ter_);  Ti_[n1_]=GADD(x0i_,tei_);                 \
        GVEC tmr_=GFNMA(ter_,0.25,x0r_), tmi_=GFNMA(tei_,0.25,x0i_);         \
        GVEC tpr_=GFMA (tar_,K59,tmr_),  tpi_=GFMA (tai_,K59,tmi_);          \
        GVEC tqr_=GFNMA(tar_,K59,tmr_),  tqi_=GFNMA(tai_,K59,tmi_);          \
        GVEC tvr_=GFMA (t7r_,KIG,t4r_),  tvi_=GFMA (t7i_,KIG,t4i_);          \
        GVEC twr_=GFNMA(t4r_,KIG,t7r_),  twi_=GFNMA(t4i_,KIG,t7i_);          \
        Tr_[ 9+n1_]=GFMA (tvi_,KS5,tpr_); Ti_[ 9+n1_]=GFNMA(tvr_,KS5,tpi_);  \
        Tr_[18+n1_]=GFNMA(twi_,KS5,tqr_); Ti_[18+n1_]=GFMA (twr_,KS5,tqi_);  \
        Tr_[27+n1_]=GFMA (twi_,KS5,tqr_); Ti_[27+n1_]=GFNMA(twr_,KS5,tqi_);  \
        Tr_[36+n1_]=GFNMA(tvi_,KS5,tpr_); Ti_[36+n1_]=GFMA (tvr_,KS5,tpi_);  \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k2_ = 0; k2_ < 5; ++k2_) {                                      \
        const GVEC *fr_ = Tr_ + 9*k2_, *fi_ = Ti_ + 9*k2_;                   \
        GVEC s0r_=GADD(fr_[3],fr_[6]), s0i_=GADD(fi_[3],fi_[6]);             \
        GVEC e0r_=GSUB(fr_[3],fr_[6]), e0i_=GSUB(fi_[3],fi_[6]);             \
        GVEC S0r_=GADD(fr_[0],s0r_),   S0i_=GADD(fi_[0],s0i_);               \
        GVEC a0r_=GFNMA(s0r_,0.5,fr_[0]), a0i_=GFNMA(s0i_,0.5,fi_[0]);       \
        GVEC s1r_=GADD(fr_[4],fr_[7]), s1i_=GADD(fi_[4],fi_[7]);             \
        GVEC e1r_=GSUB(fr_[4],fr_[7]), e1i_=GSUB(fi_[4],fi_[7]);             \
        GVEC S1r_=GADD(fr_[1],s1r_),   S1i_=GADD(fi_[1],s1i_);               \
        GVEC a1r_=GFNMA(s1r_,0.5,fr_[1]), a1i_=GFNMA(s1i_,0.5,fi_[1]);       \
        GVEC p1r_=GFMA (e1i_,KS3,a1r_), p1i_=GFNMA(e1r_,KS3,a1i_);           \
        GVEC q1r_=GFNMA(e1i_,KS3,a1r_), q1i_=GFMA (e1r_,KS3,a1i_);           \
        GVEC s2r_=GADD(fr_[5],fr_[8]), s2i_=GADD(fi_[5],fi_[8]);             \
        GVEC e2r_=GSUB(fr_[5],fr_[8]), e2i_=GSUB(fi_[5],fi_[8]);             \
        GVEC S2r_=GADD(fr_[2],s2r_),   S2i_=GADD(fi_[2],s2i_);               \
        GVEC a2r_=GFNMA(s2r_,0.5,fr_[2]), a2i_=GFNMA(s2i_,0.5,fi_[2]);       \
        GVEC p2r_=GFMA (e2i_,KS3,a2r_), p2i_=GFNMA(e2r_,KS3,a2i_);           \
        GVEC q2r_=GFNMA(e2i_,KS3,a2r_), q2i_=GFMA (e2r_,KS3,a2i_);           \
        GVEC sgr_=GADD(S1r_,S2r_), sgi_=GADD(S1i_,S2i_);                     \
        GVEC d3r_=GSUB(S2r_,S1r_), d3i_=GSUB(S2i_,S1i_);                     \
        GVEC b0r_=GFNMA(sgr_,0.5,S0r_), b0i_=GFNMA(sgi_,0.5,S0i_);           \
        STR((36*k2_   )%45, GADD(S0r_,sgr_));                                \
        STI((36*k2_   )%45, GADD(S0i_,sgi_));                                \
        STR((36*k2_+30)%45, GFNMA(d3i_,KS3,b0r_));                           \
        STI((36*k2_+30)%45, GFMA (d3r_,KS3,b0i_));                           \
        STR((36*k2_+60)%45, GFMA (d3i_,KS3,b0r_));                           \
        STI((36*k2_+60)%45, GFNMA(d3r_,KS3,b0i_));                           \
        {                                                                    \
        GVEC v1r_=GFMA (e0i_,KS3,a0r_), v1i_=GFNMA(e0r_,KS3,a0i_);           \
        GVEC w2r_=GFNMA(p2i_,K176,p2r_), w2i_=GFMA (p2r_,K176,p2i_);         \
        GVEC w1r_=GFMA (p1i_,K839,p1r_), w1i_=GFNMA(p1r_,K839,p1i_);         \
        GVEC u1r_=GFMA (w1r_,K777,w2i_), u1i_=GFNMA(w1i_,K777,w2r_);         \
        GVEC z1r_=GFMA (w1i_,K777,w2r_), z1i_=GFNMA(w1r_,K777,w2i_);         \
        STR((36*k2_+10)%45, GFMA (u1r_,K984,v1r_));                          \
        STI((36*k2_+10)%45, GFNMA(u1i_,K984,v1i_));                          \
        GVEC r1r_=GFNMA(u1r_,K492,v1r_), r1i_=GFMA(u1i_,K492,v1i_);          \
        STR((36*k2_+40)%45, GFMA (z1r_,K852,r1r_));                          \
        STI((36*k2_+40)%45, GFMA (z1i_,K852,r1i_));                          \
        STR((36*k2_+70)%45, GFNMA(z1r_,K852,r1r_));                          \
        STI((36*k2_+70)%45, GFNMA(z1i_,K852,r1i_));                          \
        }                                                                    \
        {                                                                    \
        GVEC v2r_=GFNMA(e0i_,KS3,a0r_), v2i_=GFMA (e0r_,KS3,a0i_);           \
        GVEC wAr_=GFMA (q1r_,K176,q1i_), wAi_=GFNMA(q1i_,K176,q1r_);         \
        GVEC wBr_=GFNMA(q2i_,K363,q2r_), wBi_=GFMA (q2r_,K363,q2i_);         \
        GVEC uBr_=GFNMA(wBr_,K954,wAr_), uBi_=GFMA (wBi_,K954,wAi_);         \
        GVEC zBr_=GFNMA(wBi_,K954,wAi_), zBi_=GFMA (wBr_,K954,wAr_);         \
        STR((36*k2_+20)%45, GFMA (uBr_,K984,v2r_));                          \
        STI((36*k2_+20)%45, GFNMA(uBi_,K984,v2i_));                          \
        GVEC rBr_=GFNMA(uBr_,K492,v2r_), rBi_=GFMA(uBi_,K492,v2i_);          \
        STR((36*k2_+50)%45, GFNMA(zBr_,K852,rBr_));                          \
        STI((36*k2_+50)%45, GFNMA(zBi_,K852,rBi_));                          \
        STR((36*k2_+80)%45, GFMA (zBr_,K852,rBr_));                          \
        STI((36*k2_+80)%45, GFMA (zBi_,K852,rBi_));                          \
        }                                                                    \
    }                                                                        \
} while (0)

#define PFA45RS(LDR, LDI, STR, STI) do {                                     \
    GVEC Ar_[45], Ai_[45];                     /* A[5*k1 + n2] */            \
    _Pragma("GCC unroll 5")                                                  \
    for (int n2_ = 0; n2_ < 5; ++n2_) {                                      \
        GVEC g0r_=LDR((   9*n2_)%45), g0i_=LDI((   9*n2_)%45);               \
        GVEC g1r_=LDR(( 5+9*n2_)%45), g1i_=LDI(( 5+9*n2_)%45);               \
        GVEC g2r_=LDR((10+9*n2_)%45), g2i_=LDI((10+9*n2_)%45);               \
        GVEC g3r_=LDR((15+9*n2_)%45), g3i_=LDI((15+9*n2_)%45);               \
        GVEC g4r_=LDR((20+9*n2_)%45), g4i_=LDI((20+9*n2_)%45);               \
        GVEC g5r_=LDR((25+9*n2_)%45), g5i_=LDI((25+9*n2_)%45);               \
        GVEC g6r_=LDR((30+9*n2_)%45), g6i_=LDI((30+9*n2_)%45);               \
        GVEC g7r_=LDR((35+9*n2_)%45), g7i_=LDI((35+9*n2_)%45);               \
        GVEC g8r_=LDR((40+9*n2_)%45), g8i_=LDI((40+9*n2_)%45);               \
        GVEC s0r_=GADD(g3r_,g6r_), s0i_=GADD(g3i_,g6i_);                     \
        GVEC e0r_=GSUB(g3r_,g6r_), e0i_=GSUB(g3i_,g6i_);                     \
        GVEC S0r_=GADD(g0r_,s0r_), S0i_=GADD(g0i_,s0i_);                     \
        GVEC a0r_=GFNMA(s0r_,0.5,g0r_), a0i_=GFNMA(s0i_,0.5,g0i_);           \
        GVEC s1r_=GADD(g4r_,g7r_), s1i_=GADD(g4i_,g7i_);                     \
        GVEC e1r_=GSUB(g4r_,g7r_), e1i_=GSUB(g4i_,g7i_);                     \
        GVEC S1r_=GADD(g1r_,s1r_), S1i_=GADD(g1i_,s1i_);                     \
        GVEC a1r_=GFNMA(s1r_,0.5,g1r_), a1i_=GFNMA(s1i_,0.5,g1i_);           \
        GVEC p1r_=GFMA (e1i_,KS3,a1r_), p1i_=GFNMA(e1r_,KS3,a1i_);           \
        GVEC q1r_=GFNMA(e1i_,KS3,a1r_), q1i_=GFMA (e1r_,KS3,a1i_);           \
        GVEC s2r_=GADD(g5r_,g8r_), s2i_=GADD(g5i_,g8i_);                     \
        GVEC e2r_=GSUB(g5r_,g8r_), e2i_=GSUB(g5i_,g8i_);                     \
        GVEC S2r_=GADD(g2r_,s2r_), S2i_=GADD(g2i_,s2i_);                     \
        GVEC a2r_=GFNMA(s2r_,0.5,g2r_), a2i_=GFNMA(s2i_,0.5,g2i_);           \
        GVEC p2r_=GFMA (e2i_,KS3,a2r_), p2i_=GFNMA(e2r_,KS3,a2i_);           \
        GVEC q2r_=GFNMA(e2i_,KS3,a2r_), q2i_=GFMA (e2r_,KS3,a2i_);           \
        GVEC sgr_=GADD(S1r_,S2r_), sgi_=GADD(S1i_,S2i_);                     \
        GVEC d3r_=GSUB(S2r_,S1r_), d3i_=GSUB(S2i_,S1i_);                     \
        GVEC b0r_=GFNMA(sgr_,0.5,S0r_), b0i_=GFNMA(sgi_,0.5,S0i_);           \
        Ar_[     n2_]=GADD(S0r_,sgr_);      Ai_[     n2_]=GADD(S0i_,sgi_);   \
        Ar_[15 + n2_]=GFNMA(d3i_,KS3,b0r_); Ai_[15 + n2_]=GFMA (d3r_,KS3,b0i_);\
        Ar_[30 + n2_]=GFMA (d3i_,KS3,b0r_); Ai_[30 + n2_]=GFNMA(d3r_,KS3,b0i_);\
        {                                                                    \
        GVEC v1r_=GFMA (e0i_,KS3,a0r_), v1i_=GFNMA(e0r_,KS3,a0i_);           \
        GVEC w2r_=GFNMA(p2i_,K176,p2r_), w2i_=GFMA (p2r_,K176,p2i_);         \
        GVEC w1r_=GFMA (p1i_,K839,p1r_), w1i_=GFNMA(p1r_,K839,p1i_);         \
        GVEC u1r_=GFMA (w1r_,K777,w2i_), u1i_=GFNMA(w1i_,K777,w2r_);         \
        GVEC z1r_=GFMA (w1i_,K777,w2r_), z1i_=GFNMA(w1r_,K777,w2i_);         \
        Ar_[ 5 + n2_]=GFMA (u1r_,K984,v1r_); Ai_[ 5 + n2_]=GFNMA(u1i_,K984,v1i_);\
        GVEC r1r_=GFNMA(u1r_,K492,v1r_), r1i_=GFMA(u1i_,K492,v1i_);          \
        Ar_[20 + n2_]=GFMA (z1r_,K852,r1r_); Ai_[20 + n2_]=GFMA (z1i_,K852,r1i_);\
        Ar_[35 + n2_]=GFNMA(z1r_,K852,r1r_); Ai_[35 + n2_]=GFNMA(z1i_,K852,r1i_);\
        }                                                                    \
        {                                                                    \
        GVEC v2r_=GFNMA(e0i_,KS3,a0r_), v2i_=GFMA (e0r_,KS3,a0i_);           \
        GVEC wAr_=GFMA (q1r_,K176,q1i_), wAi_=GFNMA(q1i_,K176,q1r_);         \
        GVEC wBr_=GFNMA(q2i_,K363,q2r_), wBi_=GFMA (q2r_,K363,q2i_);         \
        GVEC uBr_=GFNMA(wBr_,K954,wAr_), uBi_=GFMA (wBi_,K954,wAi_);         \
        GVEC zBr_=GFNMA(wBi_,K954,wAi_), zBi_=GFMA (wBr_,K954,wAr_);         \
        Ar_[10 + n2_]=GFMA (uBr_,K984,v2r_); Ai_[10 + n2_]=GFNMA(uBi_,K984,v2i_);\
        GVEC rBr_=GFNMA(uBr_,K492,v2r_), rBi_=GFMA(uBi_,K492,v2i_);          \
        Ar_[25 + n2_]=GFNMA(zBr_,K852,rBr_); Ai_[25 + n2_]=GFNMA(zBi_,K852,rBi_);\
        Ar_[40 + n2_]=GFMA (zBr_,K852,rBr_); Ai_[40 + n2_]=GFMA (zBi_,K852,rBi_);\
        }                                                                    \
    }                                                                        \
    _Pragma("GCC unroll 9")                                                  \
    for (int k1_ = 0; k1_ < 9; ++k1_) {                                      \
        const GVEC *fr_ = Ar_ + 5*k1_, *fi_ = Ai_ + 5*k1_;                   \
        GVEC t1r_=GADD(fr_[1],fr_[4]), t1i_=GADD(fi_[1],fi_[4]);             \
        GVEC t4r_=GSUB(fr_[1],fr_[4]), t4i_=GSUB(fi_[1],fi_[4]);             \
        GVEC t2r_=GADD(fr_[2],fr_[3]), t2i_=GADD(fi_[2],fi_[3]);             \
        GVEC t7r_=GSUB(fr_[2],fr_[3]), t7i_=GSUB(fi_[2],fi_[3]);             \
        GVEC ter_=GADD(t1r_,t2r_), tei_=GADD(t1i_,t2i_);                     \
        GVEC tar_=GSUB(t1r_,t2r_), tai_=GSUB(t1i_,t2i_);                     \
        STR((10*k1_)%45, GADD(fr_[0],ter_));                                 \
        STI((10*k1_)%45, GADD(fi_[0],tei_));                                 \
        GVEC tmr_=GFNMA(ter_,0.25,fr_[0]), tmi_=GFNMA(tei_,0.25,fi_[0]);     \
        GVEC tpr_=GFMA (tar_,K59,tmr_), tpi_=GFMA (tai_,K59,tmi_);           \
        GVEC tqr_=GFNMA(tar_,K59,tmr_), tqi_=GFNMA(tai_,K59,tmi_);           \
        GVEC tvr_=GFMA (t7r_,KIG,t4r_), tvi_=GFMA (t7i_,KIG,t4i_);           \
        GVEC twr_=GFNMA(t4r_,KIG,t7r_), twi_=GFNMA(t4i_,KIG,t7i_);           \
        STR((10*k1_+ 36)%45, GFMA (tvi_,KS5,tpr_));                          \
        STI((10*k1_+ 36)%45, GFNMA(tvr_,KS5,tpi_));                          \
        STR((10*k1_+ 72)%45, GFNMA(twi_,KS5,tqr_));                          \
        STI((10*k1_+ 72)%45, GFMA (twr_,KS5,tqi_));                          \
        STR((10*k1_+108)%45, GFMA (twi_,KS5,tqr_));                          \
        STI((10*k1_+108)%45, GFNMA(twr_,KS5,tqi_));                          \
        STR((10*k1_+144)%45, GFNMA(tvi_,KS5,tpr_));                          \
        STI((10*k1_+144)%45, GFMA (tvr_,KS5,tpi_));                          \
    }                                                                        \
} while (0)

/* ---- zmm instantiations ---- */
#define GVEC __m512d
#define GADD(a,b)   _mm512_add_pd((a),(b))
#define GSUB(a,b)   _mm512_sub_pd((a),(b))
#define GFMA(a,k,c)  _mm512_fmadd_pd((a), _mm512_set1_pd(k), (c))
#define GFNMA(a,k,c) _mm512_fnmadd_pd((a), _mm512_set1_pd(k), (c))

/* z subpass, one full y-block (rows y0..y0+7, lanes = y): broadcasts from
 * the L1 row buffer (or the state on the unmapped step-1 path), full-width
 * ALIGNED stores to the split scratch column y0. */
static __attribute__((noinline))
void z45s_blk8(const double *restrict rr, const double *restrict ri,
               double *restrict sr, double *restrict si)
{
#define LDRZ(n)    g8b8(rr + (n), SPROW)
#define LDIZ(n)    g8b8(ri + (n), SPROW)
#define STRZ(k, v) _mm512_store_pd(sr + (size_t)(k) * SPROW, (v))
#define STIZ(k, v) _mm512_store_pd(si + (size_t)(k) * SPROW, (v))
    PFA45RS(LDRZ, LDIZ, STRZ, STIZ);
#undef LDRZ
#undef LDIZ
#undef STRZ
#undef STIZ
}

/* z tail block: rows 40..43 (lanes 0-3 live, dead lanes ZERO -- no denormal
 * risk), masked stores.  Row 44 goes through the scalar line below, so the
 * bit classes match vt exactly (rows 40-43 = PFA45R group, row 44 = PFA45). */
static __attribute__((noinline))
void z45s_blk4(const double *restrict rr, const double *restrict ri,
               double *restrict sr, double *restrict si)
{
#define LDRZ(n)    g8b4(rr + (n), SPROW)
#define LDIZ(n)    g8b4(ri + (n), SPROW)
#define STRZ(k, v) _mm512_mask_store_pd(sr + (size_t)(k) * SPROW, 0x0F, (v))
#define STIZ(k, v) _mm512_mask_store_pd(si + (size_t)(k) * SPROW, 0x0F, (v))
    PFA45RS(LDRZ, LDIZ, STRZ, STIZ);
#undef LDRZ
#undef LDIZ
#undef STRZ
#undef STIZ
}

/* y subpass, one kz-block (lanes = kz): broadcasts from the L1 scratch,
 * full-width aligned stores back onto the state plane (in place: the whole
 * plane was drained into the scratch before the first y store). */
static __attribute__((noinline))
void y45s_blk8(const double *restrict sr, const double *restrict si,
               double *restrict mr, double *restrict mi)
{
#define LDRY(n)    g8b8(sr + (n), SPROW)
#define LDIY(n)    g8b8(si + (n), SPROW)
#define STRY(k, v) _mm512_store_pd(mr + (size_t)(k) * SPROW, (v))
#define STIY(k, v) _mm512_store_pd(mi + (size_t)(k) * SPROW, (v))
    PFA45S(LDRY, LDIY, STRY, STIY);
#undef LDRY
#undef LDIY
#undef STRY
#undef STIY
}

static __attribute__((noinline))
void y45s_blk5(const double *restrict sr, const double *restrict si,
               double *restrict mr, double *restrict mi)
{
#define LDRY(n)    g8b5(sr + (n), SPROW)
#define LDIY(n)    g8b5(si + (n), SPROW)
#define STRY(k, v) _mm512_mask_store_pd(mr + (size_t)(k) * SPROW, 0x1F, (v))
#define STIY(k, v) _mm512_mask_store_pd(mi + (size_t)(k) * SPROW, 0x1F, (v))
    PFA45S(LDRY, LDIY, STRY, STIY);
#undef LDRY
#undef LDIY
#undef STRY
#undef STIY
}

/* x pass, one (ky, kz-block) tile, in place across the 45 planes: every
 * access is a full-width aligned load/store (pad lanes are zeros). */
static __attribute__((noinline))
void p2s_blk8(double *br, double *bi)
{
#define LDRX(n)    _mm512_load_pd(br + (size_t)(n) * SPPL)
#define LDIX(n)    _mm512_load_pd(bi + (size_t)(n) * SPPL)
#define STRX(k, v) _mm512_store_pd(br + (size_t)(k) * SPPL, (v))
#define STIX(k, v) _mm512_store_pd(bi + (size_t)(k) * SPPL, (v))
    PFA45S(LDRX, LDIX, STRX, STIX);
#undef LDRX
#undef LDIX
#undef STRX
#undef STIX
}

static __attribute__((noinline))
void p2s_blk5(double *br, double *bi)
{
#define LDRX(n)    _mm512_load_pd(br + (size_t)(n) * SPPL)
#define LDIX(n)    _mm512_load_pd(bi + (size_t)(n) * SPPL)
#define STRX(k, v) _mm512_mask_store_pd(br + (size_t)(k) * SPPL, 0x1F, (v))
#define STIX(k, v) _mm512_mask_store_pd(bi + (size_t)(k) * SPPL, 0x1F, (v))
    PFA45S(LDRX, LDIX, STRX, STIX);
#undef LDRX
#undef LDIX
#undef STRX
#undef STIX
}

#undef GVEC
#undef GADD
#undef GSUB
#undef GFMA
#undef GFNMA

/* ---- scalar instantiation: the odd 45th z row (bit-matches dft45_line1's
 * PFA45 order; __builtin_fma(-(a),k,c) is exactly the vfnmadd lane op) ---- */
#define GVEC double
#define GADD(a,b)   ((a) + (b))
#define GSUB(a,b)   ((a) - (b))
#define GFMA(a,k,c)  __builtin_fma((a), (k), (c))
#define GFNMA(a,k,c) __builtin_fma(-(a), (k), (c))
static __attribute__((noinline))
void z45s_line(const double *restrict rr, const double *restrict ri,
               double *restrict sr, double *restrict si)
{
#define LDR1(n)    rr[(n)]
#define LDI1(n)    ri[(n)]
#define STR1(k, v) (sr[(size_t)(k) * SPROW] = (v))
#define STI1(k, v) (si[(size_t)(k) * SPROW] = (v))
    PFA45S(LDR1, LDI1, STR1, STI1);
#undef LDR1
#undef LDI1
#undef STR1
#undef STI1
}
#undef GVEC
#undef GADD
#undef GSUB
#undef GFMA
#undef GFNMA

/* split lazy map: o = map(z + c) over nv full-width vectors per component.
 * Same per-point op sequence as map45_pair/map45_one (s = (re^2+g)+(im^2+g),
 * one q8 ladder + one divide per 8 points, o = w*q behind the contraction
 * barrier), zero lane shuffles.  noinline+noclone per the r5 constprop
 * lesson. */
static __attribute__((noinline, noclone))
void map45s_span(const double *zr, const double *zi,
                 const double *cr, const double *ci,
                 double *outr, double *outi, long nv)
{
    const __m512d g = _mm512_set1_pd(MG45);
    for (long j = 0; j < nv; ++j) {
        __m512d rv = _mm512_add_pd(_mm512_load_pd(zr + 8 * j),
                                   _mm512_load_pd(cr + 8 * j));
        __m512d iv = _mm512_add_pd(_mm512_load_pd(zi + 8 * j),
                                   _mm512_load_pd(ci + 8 * j));
        __m512d s  = _mm512_add_pd(_mm512_fmadd_pd(rv, rv, g),
                                   _mm512_fmadd_pd(iv, iv, g));
        __m512d q  = map45_q8(s);
        __m512d ar = _mm512_mul_pd(rv, q), ai = _mm512_mul_pd(iv, q);
        __asm__("" : "+v"(ar), "+v"(ai));
        _mm512_store_pd(outr + 8 * j, ar);
        _mm512_store_pd(outi + 8 * j, ai);
    }
}

/* interleaved (driver) -> split volume convert: pure data movement, exact.
 * Also writes the kz=45..47 pad columns with ZEROS (they stay 0 forever:
 * every later store to the state is masked at kz block 5). */
static __attribute__((noinline))
void conv45_in(const double *restrict x, double *restrict W)
{
    const __m512i iE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i iO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    for (int p = 0; p < L; ++p)
        for (int y = 0; y < L; ++y) {
            const double *s = x + (size_t)p * PLND + (size_t)y * (2 * L);
            double *dr = W + (size_t)p * SPPL + (size_t)y * SPROW;
            double *di = dr + SPCO;
            _Pragma("GCC unroll 5")
            for (int k = 0; k < 5; ++k) {
                __m512d v0 = _mm512_loadu_pd(s + 16 * k);
                __m512d v1 = _mm512_loadu_pd(s + 16 * k + 8);
                _mm512_store_pd(dr + 8 * k, _mm512_permutex2var_pd(v0, iE, v1));
                _mm512_store_pd(di + 8 * k, _mm512_permutex2var_pd(v0, iO, v1));
            }
            {   /* z 40..44 + zero pads (v1's dead lanes are maskz zeros) */
                __m512d v0 = _mm512_loadu_pd(s + 80);
                __m512d v1 = _mm512_maskz_loadu_pd(0x03, s + 88);
                _mm512_store_pd(dr + 40, _mm512_permutex2var_pd(v0, iE, v1));
                _mm512_store_pd(di + 40, _mm512_permutex2var_pd(v0, iO, v1));
            }
        }
}

/* final step: map(split z + split c) -> INTERLEAVED final_out (one pass,
 * the chain's only write to the driver's buffer).  Interleaving happens
 * AFTER the barrier'd multiply, so bits equal map45_span's exactly. */
static __attribute__((noinline))
void map45s_final(const double *restrict W, const double *restrict CS,
                  double *restrict o)
{
    const __m512i iL = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i iH = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const __m512d g  = _mm512_set1_pd(MG45);
    for (int p = 0; p < L; ++p)
        for (int y = 0; y < L; ++y) {
            const double *zr = W  + (size_t)p * SPPL + (size_t)y * SPROW;
            const double *cr = CS + (size_t)p * SPPL + (size_t)y * SPROW;
            double *d = o + (size_t)p * PLND + (size_t)y * (2 * L);
            _Pragma("GCC unroll 6")
            for (int k = 0; k < 6; ++k) {
                __m512d rv = _mm512_add_pd(_mm512_load_pd(zr + 8 * k),
                                           _mm512_load_pd(cr + 8 * k));
                __m512d iv = _mm512_add_pd(_mm512_load_pd(zr + SPCO + 8 * k),
                                           _mm512_load_pd(cr + SPCO + 8 * k));
                __m512d s  = _mm512_add_pd(_mm512_fmadd_pd(rv, rv, g),
                                           _mm512_fmadd_pd(iv, iv, g));
                __m512d q  = map45_q8(s);
                __m512d ar = _mm512_mul_pd(rv, q), ai = _mm512_mul_pd(iv, q);
                __asm__("" : "+v"(ar), "+v"(ai));
                __m512d lo = _mm512_permutex2var_pd(ar, iL, ai);
                __m512d hi = _mm512_permutex2var_pd(ar, iH, ai);
                if (k < 5) {
                    _mm512_storeu_pd(d + 16 * k,     lo);
                    _mm512_storeu_pd(d + 16 * k + 8, hi);
                } else {
                    _mm512_storeu_pd(d + 80, lo);
                    _mm512_mask_storeu_pd(d + 88, 0x03, hi);
                }
            }
        }
}

/* split phase 1, one plane.  mp: stage map(z+c) of y-row group g+1 into the
 * L1 ping-pong before the FFT of group g (the L36_pencilfused maprows shape,
 * carried over from the vt arm); the ping-pong lives in the P allocation
 * (deterministic page phase -- a hot 12 KB stack block would re-enter the
 * stack-phase lottery this plan's P comment documents). */
static __attribute__((noinline))
void ch45s_p1_plane(double *Wp, const double *cp, double *P, const int mp)
{
    double *pong = P + 4352;                 /* 2 bufs x [re 8x48 | im 8x48] */
    if (mp) {
        map45s_span(Wp, Wp + SPCO, cp, cp + SPCO,
                    pong, pong + 8 * SPROW, 48);
        _Pragma("GCC unroll 1")
        for (int gI = 0; gI < 5; ++gI) {
            double *nb = pong + ((gI + 1) & 1) * (16 * SPROW);
            if (gI < 4)
                map45s_span(Wp + (size_t)(gI + 1) * (8 * SPROW),
                            Wp + SPCO + (size_t)(gI + 1) * (8 * SPROW),
                            cp + (size_t)(gI + 1) * (8 * SPROW),
                            cp + SPCO + (size_t)(gI + 1) * (8 * SPROW),
                            nb, nb + 8 * SPROW, 48);
            else                             /* rows 40..44 (5 rows) */
                map45s_span(Wp + (size_t)40 * SPROW,
                            Wp + SPCO + (size_t)40 * SPROW,
                            cp + (size_t)40 * SPROW,
                            cp + SPCO + (size_t)40 * SPROW,
                            nb, nb + 8 * SPROW, 30);
            const double *rr = pong + (gI & 1) * (16 * SPROW);
            const double *ri = rr + 8 * SPROW;
            double *sr = P + 8 * gI, *si = P + SPCO + 8 * gI;
            __asm__("" : "+r"(rr), "+r"(ri), "+r"(sr), "+r"(si));
            z45s_blk8(rr, ri, sr, si);
        }
        {   /* tail: rows 40-43 via the block, row 44 via the scalar line */
            const double *rr = pong + 16 * SPROW, *ri = rr + 8 * SPROW;
            z45s_blk4(rr, ri, P + 40, P + SPCO + 40);
            z45s_line(rr + 4 * SPROW, ri + 4 * SPROW, P + 44, P + SPCO + 44);
        }
    } else {
        _Pragma("GCC unroll 1")
        for (int gI = 0; gI < 5; ++gI) {
            const double *rr = Wp + (size_t)(8 * gI) * SPROW;
            const double *ri = rr + SPCO;
            double *sr = P + 8 * gI, *si = P + SPCO + 8 * gI;
            __asm__("" : "+r"(rr), "+r"(ri), "+r"(sr), "+r"(si));
            z45s_blk8(rr, ri, sr, si);
        }
        z45s_blk4(Wp + (size_t)40 * SPROW, Wp + SPCO + (size_t)40 * SPROW,
                  P + 40, P + SPCO + 40);
        z45s_line(Wp + (size_t)44 * SPROW, Wp + SPCO + (size_t)44 * SPROW,
                  P + 44, P + SPCO + 44);
    }
    /* y subpass: scratch -> back onto the plane (in place safe) */
    _Pragma("GCC unroll 1")
    for (int kb = 0; kb < 5; ++kb) {
        const double *sr = P + (size_t)(8 * kb) * SPROW;
        const double *si = sr + SPCO;
        double *mr = Wp + 8 * kb, *mi = Wp + SPCO + 8 * kb;
        __asm__("" : "+r"(sr), "+r"(si), "+r"(mr), "+r"(mi));
        y45s_blk8(sr, si, mr, mi);
    }
    y45s_blk5(P + (size_t)40 * SPROW, P + SPCO + (size_t)40 * SPROW,
              Wp + 40, Wp + SPCO + 40);
}

/* ice_r6 "vs2": the map fused into the x pass's STORES (the L36_mixedradix
 * r6 placement -- at the store site the ladder hangs off the END of the
 * dependency graph under 688 FMAs of independent work, where the standalone
 * staged span starves on its own chains: measured here p1sm-p1s = +63
 * us/vol, L36_pencilfused's trap reproduced in split form).  The state
 * between steps then holds MAPPED values, so p1 needs no map path at all;
 * bits are unchanged (same per-point op sequence, applied one pass earlier
 * in the SAME value chain).  PFA45S emits STR(k,·) immediately before
 * STI(k,·) at every site, so STR stashes re and STI runs one ladder + one
 * divide per 8 points and stores both components. */
#define GVEC __m512d
#define GADD(a,b)   _mm512_add_pd((a),(b))
#define GSUB(a,b)   _mm512_sub_pd((a),(b))
#define GFMA(a,k,c)  _mm512_fmadd_pd((a), _mm512_set1_pd(k), (c))
#define GFNMA(a,k,c) _mm512_fnmadd_pd((a), _mm512_set1_pd(k), (c))
static __attribute__((noinline))
void p2sm_blk8(double *br, double *bi, const double *cr, const double *ci)
{
    const __m512d gG_ = _mm512_set1_pd(MG45);
    __m512d stR_ = _mm512_setzero_pd();
#define LDRX(n) _mm512_load_pd(br + (size_t)(n) * SPPL)
#define LDIX(n) _mm512_load_pd(bi + (size_t)(n) * SPPL)
#define STRX(k, v) (stR_ = (v))
#define STIX(k, v) do {                                                      \
    __m512d rv_ = _mm512_add_pd(stR_,                                        \
                                _mm512_load_pd(cr + (size_t)(k) * SPPL));    \
    __m512d iv_ = _mm512_add_pd((v),                                         \
                                _mm512_load_pd(ci + (size_t)(k) * SPPL));    \
    __m512d s_  = _mm512_add_pd(_mm512_fmadd_pd(rv_, rv_, gG_),              \
                                _mm512_fmadd_pd(iv_, iv_, gG_));             \
    __m512d q_  = map45_q8(s_);                                              \
    __m512d ar_ = _mm512_mul_pd(rv_, q_), ai_ = _mm512_mul_pd(iv_, q_);      \
    __asm__("" : "+v"(ar_), "+v"(ai_));                                      \
    _mm512_store_pd(br + (size_t)(k) * SPPL, ar_);                           \
    _mm512_store_pd(bi + (size_t)(k) * SPPL, ai_);                           \
} while (0)
    PFA45S(LDRX, LDIX, STRX, STIX);
#undef LDRX
#undef LDIX
#undef STRX
#undef STIX
}

static __attribute__((noinline))
void p2sm_blk5(double *br, double *bi, const double *cr, const double *ci)
{
    const __m512d gG_ = _mm512_set1_pd(MG45);
    __m512d stR_ = _mm512_setzero_pd();
#define LDRX(n) _mm512_load_pd(br + (size_t)(n) * SPPL)
#define LDIX(n) _mm512_load_pd(bi + (size_t)(n) * SPPL)
#define STRX(k, v) (stR_ = (v))
#define STIX(k, v) do {                                                      \
    __m512d rv_ = _mm512_add_pd(stR_,                                        \
                                _mm512_load_pd(cr + (size_t)(k) * SPPL));    \
    __m512d iv_ = _mm512_add_pd((v),                                         \
                                _mm512_load_pd(ci + (size_t)(k) * SPPL));    \
    __m512d s_  = _mm512_add_pd(_mm512_fmadd_pd(rv_, rv_, gG_),              \
                                _mm512_fmadd_pd(iv_, iv_, gG_));             \
    __m512d q_  = map45_q8(s_);                                              \
    __m512d ar_ = _mm512_mul_pd(rv_, q_), ai_ = _mm512_mul_pd(iv_, q_);      \
    __asm__("" : "+v"(ar_), "+v"(ai_));                                      \
    _mm512_mask_store_pd(br + (size_t)(k) * SPPL, 0x1F, ar_);                \
    _mm512_mask_store_pd(bi + (size_t)(k) * SPPL, 0x1F, ai_);                \
} while (0)
    PFA45S(LDRX, LDIX, STRX, STIX);
#undef LDRX
#undef LDIX
#undef STRX
#undef STIX
}
#undef GVEC
#undef GADD
#undef GSUB
#undef GFMA
#undef GFNMA

/* split x pass over the volume; pfx pokes the 90 read streams one line
 * ahead (the vt-pf mechanism on the split layout). */
static __attribute__((noinline))
void ch45s_p2(double *W, const int pfx)
{
    _Pragma("GCC unroll 1")
    for (int ky = 0; ky < L; ++ky) {
        _Pragma("GCC unroll 1")
        for (int kb = 0; kb < 6; ++kb) {
            double *br = W + (size_t)ky * SPROW + 8 * kb;
            double *bi = br + SPCO;
            __asm__("" : "+r"(br), "+r"(bi));
            if (pfx) {
                _Pragma("GCC unroll 45")
                for (int n = 0; n < L; ++n) {
                    __builtin_prefetch(br + (size_t)n * SPPL + 8, 0, 3);
                    __builtin_prefetch(bi + (size_t)n * SPPL + 8, 0, 3);
                }
            }
            if (kb < 5) p2s_blk8(br, bi); else p2s_blk5(br, bi);
        }
    }
}

/* mapped split x pass (vs2): every store tile maps in place */
static __attribute__((noinline))
void ch45s_p2m(double *W, const double *CS, const int pfx)
{
    _Pragma("GCC unroll 1")
    for (int ky = 0; ky < L; ++ky) {
        _Pragma("GCC unroll 1")
        for (int kb = 0; kb < 6; ++kb) {
            double *br = W + (size_t)ky * SPROW + 8 * kb;
            double *bi = br + SPCO;
            const double *cr = CS + (size_t)ky * SPROW + 8 * kb;
            const double *ci = cr + SPCO;
            __asm__("" : "+r"(br), "+r"(bi), "+r"(cr), "+r"(ci));
            if (pfx) {
                _Pragma("GCC unroll 45")
                for (int n = 0; n < L; ++n) {
                    __builtin_prefetch(br + (size_t)n * SPPL + 8, 0, 3);
                    __builtin_prefetch(bi + (size_t)n * SPPL + 8, 0, 3);
                }
            }
            if (kb < 5) p2sm_blk8(br, bi, cr, ci);
            else        p2sm_blk5(br, bi, cr, ci);
        }
    }
}

/* vs2 chain: state holds MAPPED values between steps (map fused into the
 * x pass's stores; step m's x pass stays raw and the final span maps to
 * final_out).  p1 is always the plain path -- no pong, no spans. */
static inline __attribute__((always_inline))
void ch45_vs2_body(const double *x0, const double *C, double *W, double *FO,
                   long m, long B, double *P, const int pfx)
{
    double *CS = W + (size_t)B * SPVST + 256;
    for (long b = 0; b < B; ++b) {
        const double *xb = x0 + (size_t)b * VDBL;
        const double *cb = C  + (size_t)b * VDBL;
        double       *Wb = W  + (size_t)b * SPVST;
        conv45_in(xb, Wb);
        conv45_in(cb, CS);
        for (long s = 1; s <= m; ++s) {
            for (int x = 0; x < L; ++x)
                ch45s_p1_plane(Wb + (size_t)x * SPPL, CS, P, 0);
            if (s < m) ch45s_p2m(Wb, CS, pfx);
            else       ch45s_p2(Wb, pfx);
        }
        map45s_final(Wb, CS, FO + (size_t)b * VDBL);
    }
}

static void ch45_vs2_zs(const double *x0, const double *C, double *W,
                        double *FO, long m, long B, double *P)
{
    ch45_vs2_body(x0, C, W, FO, m, B, P, 0);
}
/* (vs2-pf raced once: 510.0 vs vs2's 497.9 in-arena -- the poke is a tax
 * on an arm whose store-site c loads already saturate the memory side.) */

/* vs chain: volume-major split-state chain.  The split c copy CS sits after
 * the batch's state volumes at a +2048 B page-phase skew (volume stride is
 * a 4096 B multiple, so state planes are phase-0 and CS planes phase-2048:
 * the map's two read streams never share a page phase). */
static inline __attribute__((always_inline))
void ch45_vs_body(const double *x0, const double *C, double *W, double *FO,
                  long m, long B, double *P, const int pfx)
{
    double *CS = W + (size_t)B * SPVST + 256;
    for (long b = 0; b < B; ++b) {
        const double *xb = x0 + (size_t)b * VDBL;
        const double *cb = C  + (size_t)b * VDBL;
        double       *Wb = W  + (size_t)b * SPVST;
        conv45_in(xb, Wb);
        conv45_in(cb, CS);
        for (int x = 0; x < L; ++x)
            ch45s_p1_plane(Wb + (size_t)x * SPPL, CS + (size_t)x * SPPL, P, 0);
        ch45s_p2(Wb, pfx);
        for (long s = 1; s < m; ++s) {
            for (int x = 0; x < L; ++x)
                ch45s_p1_plane(Wb + (size_t)x * SPPL, CS + (size_t)x * SPPL,
                               P, 1);
            ch45s_p2(Wb, pfx);
        }
        map45s_final(Wb, CS, FO + (size_t)b * VDBL);
    }
}

static void ch45_vs_zs(const double *x0, const double *C, double *W,
                       double *FO, long m, long B, double *P)
{
    ch45_vs_body(x0, C, W, FO, m, B, P, 0);
}
/* (vs-pf was raced once and deleted: the poke on top of the lazy split arm
 * read +2.7% -- 465.0 vs 452.7 in-arena -- its 180 poke uops/call are pure
 * tax when the arm is already issue-starved by its standalone map spans.) */

/* ice_r6 "xf": the X-FIRST chain family (BORROWED wholesale from
 * L45_mixedradix ice_r4's winning step shape, on my tr machinery): per
 * step, the x pass runs FIRST (in place on the mapped state), then per
 * plane zsubt (plain -- the state is already mapped) -> ysubtm, whose
 * stores apply map(z+c) as the step's last write.  Step m keeps a raw
 * ysubt and the final span writes final_out.
 *
 * BIT-CLASS NOTE: the axis ORDER (x,z,y vs z,y,x) changes intermediate
 * rounding, so the xf family is a DIFFERENT bit class from vt/vs -- the
 * two families must not be raced adaptively in one pool (a timing pick
 * would flip output bits across processes).  The pool below puts ONE
 * family first at compile time (-DFFT45_XF=0|1); the gate's arm[0]-bits
 * rule then auto-disqualifies the other family, so whichever family ships
 * is process-deterministic.  A/B across the two builds decides FFT45_XF's
 * default. */
static __attribute__((noinline))
void ch45_p1x_plane(double *F, const double *cv, double *P, int x)
{
    zsubt_pw4(F + (size_t)x * PLND, P, 0, 0);
    ysubtm_pw4(P, F + (size_t)x * PLND, 2 * L, cv + (size_t)x * PLND);
}

static inline __attribute__((always_inline))
void ch45_xf_body(const double *x0, const double *C, double *W, double *FO,
                  long m, long B, double *P, const int pfx)
{
    for (long b = 0; b < B; ++b) {
        const double *cb = C + (size_t)b * VDBL;
        double       *Wb = W + (size_t)b * VDBL;
        memcpy(Wb, x0 + (size_t)b * VDBL, VDBL * sizeof(double));
        for (long s = 1; s <= m; ++s) {
            if (pfx) ch45_p2f(Wb); else ch45_p2(Wb);
            if (s < m)
                for (int x = 0; x < L; ++x) ch45_p1x_plane(Wb, cb, P, x);
            else
                for (int x = 0; x < L; ++x) ch45_p1t_plane(Wb, Wb, P, x);
        }
        map45_span(Wb, cb, FO + (size_t)b * VDBL, (long)L * NPLANE);
    }
}

static void ch45_xf_zs(const double *x0, const double *C, double *W,
                       double *FO, long m, long B, double *P)
{
    ch45_xf_body(x0, C, W, FO, m, B, P, 0);
}

static void ch45_xf_pf(const double *x0, const double *C, double *W,
                       double *FO, long m, long B, double *P)
{
    ch45_xf_body(x0, C, W, FO, m, B, P, 1);
}

/* ================= ice_r7: QUAD-VOLUME (lanes = volumes) CHAIN ARMS ========
 * BORROWED: the structural idea is rival v6_5a869e40's run4_ dispatch for
 * L <= 45 ("4-volume interleaved layout, SIMD lanes = volumes, no tail
 * lanes, no transposes"), the fastest L=45 code measured on this node
 * (0.167 s, albeit env-gated); the graded case IS B=4, so the lane count is
 * exact.  The chain owns the inter-step state format (L17_winograd ice_r5's
 * licence): state lives in a plan-owned arena Q as POINT-MAJOR, 4 volumes
 * interleaved -- point (x,y,z) of volumes 0..3 is ONE zmm at
 * Q + ((x*45+y)*45+z)*8.  Consequences per volume-step vs the xf arms:
 *   - EVERY codelet load/store in all three passes is one full-width
 *     64B-ALIGNED access (z-pencils are contiguous 2880 B, y stride 2880 B,
 *     x stride 129600 B): the ~178k 16 B corner-turn broadcasts, the
 *     slot-major scratch round-trip (~32 KB/plane written+regathered), and
 *     ALL tail machinery (90 xmm lines, GCOL/SCOL, odd-column map paths)
 *     are gone.  Load-class uops/vol: ~178k -> ~68k.
 *   - The FFT op count is unchanged per point (6075 zmm calls per group of
 *     4 volumes x (344 FMA + 78 swaps) = the same 344/line DAGs).
 *   - PAID: the working set per group is 4 volumes + permuted c = 11.7 MB,
 *     all L3-resident (the xf arms' per-volume 2.9 MB partial-L2 locality is
 *     gone), and one pack/unpack/c-permute per group per chain call
 *     (3 streaming passes over ~5.8 MB each, amortized over m steps).
 * BIT-IDENTITY with the xf family BY CONSTRUCTION: x-first step order, the
 * z site runs PFA45R for y-rows 0..43 and PFA45 for row 44 (exactly
 * zsubt's group/tail split), y and x sites run PFA45 for every pencil
 * (ysubt/ysubtm and phase2 use PFA45 for lanes AND tails), the map is the
 * same ladder fused at the y-store sites with the same pair-stash shape
 * (pair grouping is value-transparent), and step m stores raw z with one
 * map span at the end.  So the q4 arms sit in the same racing pool as
 * xf/vt and the cross-arm memcmp gate proves the transcription -- a wrong
 * site shows up as a DQ, not a shipped wrong answer.  Under FFT45_XF=0 the
 * q4 arms are the other bit class and auto-DQ, as designed. */
#define Q4NPT ((size_t)L * NPLANE)      /* complex points per volume         */
#define Q4VD  (Q4NPT * 8)               /* doubles per 4-volume group        */
#define Q4PLD ((size_t)NPLANE * 8)      /* doubles per x-plane of Q = 16200  */
#define Q4YS  (L * 8)                   /* y stride in doubles = 360         */
#define Q4XS  Q4PLD                     /* x stride in doubles = 16200       */
/* C4 sits behind Q: pad 344 doubles = 2752 B, chosen so C4 is 64B-aligned
 * AND lands at page phase (phase(Q)+2048)&4095 (Q4VD*8 = 5832000 B = 3392
 * mod 4096; 3392+2752 = 2048 mod 4096) -- the L45_mixedradix/L23_rader 4K
 * anti-correlation rule, applied to the arena-internal state-vs-c pair. */
#define Q4PAD 344
/* ice_r8: the custody arms' pencil-order c image CQ sits behind C4 at a pad
 * chosen so phase(CQ) == phase(C4) == phase(Q)+2048 (Q4VD*8 = 3392 mod
 * 4096; 88*8 = 704; 3392+704 = 0 mod 4096).  C4 and CQ are never hot in
 * the same pass (C4: P sweeps + unpackm; CQ: Q sweeps), so they may share
 * the anti-correlated phase. */
#define QCPAD 88
/* Q region offset inside the W arena, doubles from the phase-shifted base:
 * behind the split arms' (B+1) SPVST volumes + slack, rounded to 4096 B so
 * Q keeps W's page phase. */
#define Q4OFF(B) ((((size_t)(B) + 1) * SPVST + 512 + 511) & ~(size_t)511)
/* region order behind Q: C4 (+Q4PAD), CQ (+QCPAD), CP (+QCPAD) -- every c
 * image lands at phase(Q)+2048.  C4/CQ/CP are never hot in the same pass. */
#define Q4END(B) (Q4OFF(B) + 4 * Q4VD + Q4PAD + 2 * QCPAD)

static int g_q4ok;                      /* create(): W arena covers Q4END    */

/* local PW=4 vector bindings (the template's were #undef'd after
 * instantiation; these are the same operations bit-for-bit) */
#define vec        vec_pw4
#define VSPLAT(a)  ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
#define VPAIR(a,b) ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
#ifdef __clang__
# define SWAP(v)   __builtin_shufflevector((v),(v), 1,0,3,2,5,4,7,6)
#else
# define SWAP(v)   __builtin_shuffle((v),(v),(veci_pw4){1,0,3,2,5,4,7,6})
#endif
#define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define LDU4(p)    ((vec)*(const uvec_pw4 *)(p))
#define STU4(p, v) (*(uvec_pw4 *)(p) = (uvec_pw4)(v))

/* z pass, one x-plane of Q, in place: 45 contiguous 2880 B pencils.
 * Rows 0..43 keep the z-site PFA45R bit order (zsubt's 11 groups), row 44
 * runs PFA45 (dft45_line1's order) -- the exact per-row split zsubt ships.
 * cpf (the q4-cz arm): poke the C4 plane one y-row (45 lines) per call, so
 * the mapped y-pass that follows on this SAME plane finds its c loads in
 * L2 -- the ice_r7 probe read ym = +88 us/vol over plain y, and the z-pass
 * is the pass with both uop headroom and a ~70 us lead over the consumer. */
static inline __attribute__((always_inline))
void q4_zplane_body(double *Wp, const double *Cp, const int cpf)
{
    _Pragma("GCC unroll 1")
    for (int y = 0; y < L - 1; ++y) {
        double       *b_ = Wp + (size_t)y * Q4YS;
        const double *c_ = Cp + (size_t)y * Q4YS;
        __asm__("" : "+r"(b_));
        if (cpf) {
            _Pragma("GCC unroll 45")
            for (int n_ = 0; n_ < 45; ++n_)
                __builtin_prefetch(c_ + 8 * n_, 0, 3);
        }
#define LDQZ(n)    LDU4(b_ + (size_t)(n) * 8)
#define STQZ(k, v) STU4(b_ + (size_t)(k) * 8, (v))
        PFA45R(LDQZ, STQZ);
#undef LDQZ
#undef STQZ
    }
    {   /* odd 45th y-row: PFA45 order, same pencil shape */
        double       *b_ = Wp + (size_t)(L - 1) * Q4YS;
        const double *c_ = Cp + (size_t)(L - 1) * Q4YS;
        __asm__("" : "+r"(b_));
        if (cpf) {
            _Pragma("GCC unroll 45")
            for (int n_ = 0; n_ < 45; ++n_)
                __builtin_prefetch(c_ + 8 * n_, 0, 3);
        }
#define LDQZ(n)    LDU4(b_ + (size_t)(n) * 8)
#define STQZ(k, v) STU4(b_ + (size_t)(k) * 8, (v))
        PFA45(LDQZ, STQZ);
#undef LDQZ
#undef STQZ
    }
}

static __attribute__((noinline))
void q4_zplane(double *Wp)
{ q4_zplane_body(Wp, Wp, 0); }

static __attribute__((noinline))
void q4_zplanec(double *Wp, const double *Cp)
{ q4_zplane_body(Wp, Cp, 1); }

/* PROBE-ONLY (LOUD): the z pass with the store-direct PFA45 order on rows
 * 0..43 -- a DIFFERENT BIT CLASS (would need the whole family switched),
 * timed here to price whether the z-site DAG swap is worth a family knob.
 * Never enters the arm pool. */
static __attribute__((noinline, unused))
void q4_zplane_sd(double *Wp)
{
    _Pragma("GCC unroll 1")
    for (int y = 0; y < L; ++y) {
        double *b_ = Wp + (size_t)y * Q4YS;
        __asm__("" : "+r"(b_));
#define LDQZ(n)    LDU4(b_ + (size_t)(n) * 8)
#define STQZ(k, v) STU4(b_ + (size_t)(k) * 8, (v))
        PFA45(LDQZ, STQZ);
#undef LDQZ
#undef STQZ
    }
}

/* y pass, one x-plane, in place, RAW stores (step m): PFA45 for every kz
 * (ysubt runs PFA45 for lanes and the kz=44 tail alike). */
static __attribute__((noinline))
void q4_yplane(double *Wp)
{
    _Pragma("GCC unroll 1")
    for (int z = 0; z < L; ++z) {
        double *b_ = Wp + (size_t)z * 8;
        __asm__("" : "+r"(b_));
#define LDQY(n)    LDU4(b_ + (size_t)(n) * Q4YS)
#define STQY(k, v) STU4(b_ + (size_t)(k) * Q4YS, (v))
        PFA45(LDQY, STQY);
#undef LDQY
#undef STQY
    }
}

/* y pass with the map fused at the store sites (steps 1..m-1): the xf
 * family's placement (L45_mixedradix CXF + MPAIR), same one-register stash,
 * one ladder + ONE vdivpd per 8 points, odd 45th ST through map45_one.
 * c loads hit C4 at the same offsets as the stores -- plane-local, stream-
 * shaped, and at a controlled +2048 B page phase from Q. */
static __attribute__((noinline))
void q4_yplanem(double *Wp, const double *Cp)
{
    _Pragma("GCC unroll 1")
    for (int z = 0; z < L; ++z) {
        double       *b_ = Wp + (size_t)z * 8;
        const double *c_ = Cp + (size_t)z * 8;
        __asm__("" : "+r"(b_), "+r"(c_));
        vec  stv_ = VSPLAT(0.0);
        long stk_ = -1;
#define LDQY(n)    LDU4(b_ + (size_t)(n) * Q4YS)
#define STQYM(k, v) do {                                                     \
    if (stk_ < 0) { stv_ = (v); stk_ = (k); }                                \
    else {                                                                   \
        __m512d w0_ = (__m512d)(stv_ + LDU4(c_ + (size_t)stk_ * Q4YS));      \
        __m512d w1_ = (__m512d)((v)  + LDU4(c_ + (size_t)(k)  * Q4YS));      \
        map45_pair(&w0_, &w1_);                                              \
        STU4(b_ + (size_t)stk_ * Q4YS, (vec)w0_);                           \
        STU4(b_ + (size_t)(k)  * Q4YS, (vec)w1_);                           \
        stk_ = -1;                                                           \
    }                                                                        \
} while (0)
        PFA45(LDQY, STQYM);
        if (stk_ >= 0) {                     /* 45 is odd: flush the single */
            __m512d w_ = (__m512d)(stv_ + LDU4(c_ + (size_t)stk_ * Q4YS));
            STU4(b_ + (size_t)stk_ * Q4YS, (vec)map45_one(w_));
        }
#undef LDQY
#undef STQYM
    }
}

/* x pass, whole group, in place: 2025 flat (y,z) pencils at 129600 B
 * stride (2624 mod 4096: no fixed page-offset relation between streams).
 * pfx pokes the 45 read streams one line ahead, the vt-pf/PF45 shape. */
static inline __attribute__((always_inline))
void q4_xpass_body(double *Q, const int pfx)
{
    _Pragma("GCC unroll 1")
    for (long t = 0; t < NPLANE; ++t) {
        double *b_ = Q + (size_t)t * 8;
        __asm__("" : "+r"(b_));
        if (pfx) {
            _Pragma("GCC unroll 45")
            for (int n_ = 0; n_ < 45; ++n_)
                __builtin_prefetch(b_ + (size_t)n_ * Q4XS + 8, 0, 3);
        }
#define LDQX(n)    LDU4(b_ + (size_t)(n) * Q4XS)
#define STQX(k, v) STU4(b_ + (size_t)(k) * Q4XS, (v))
        PFA45(LDQX, STQX);
#undef LDQX
#undef STQX
    }
}

static __attribute__((noinline)) void q4_xp0(double *Q)
{ q4_xpass_body(Q, 0); }
static __attribute__((noinline)) void q4_xp1(double *Q)
{ q4_xpass_body(Q, 1); }

/* ============== ice_r8: CUSTODY SWEEP PASSES (qc family) ===================
 * BORROWED: the custody schedule is L45_mixedradix ice_r7's vg4-custody
 * (transitively L64_radix8 ice_r6 ckind=2): m+1 alternating-orientation
 * sweeps for an m-step chain instead of 2m full-volume sweeps.  Sweep s
 * finishes step s's carried axis WITH THE MAP FUSED AT ITS STORES, then
 * runs step s+1's z pass and the sweep-local axis -- three passes per
 * plane/slab while it is L2-resident (129.6 KB state + 129.6 KB c image),
 * so the group volume crosses L3 once per STEP instead of twice.
 *   sweep 0      (P, fixed-x planes): z(1), y(1)               [carries x(1)]
 *   sweep s odd  (Q, fixed-y slabs):  x(s)+map, z(s+1), x(s+1) [carries y]
 *   sweep s even (P, fixed-x planes): y(s)+map, z(s+1), y(s+1) [carries x]
 *   sweep m: carried axis of step m RAW; q4_unpackm applies map(m).
 * The map always sits at the LAST axis of its step (the carried axis),
 * same MPAIR stash + one vdivpd per 8 points.  Step axis order alternates
 * (z,y,x)/(z,x,y) with parity -- a NEW BIT CLASS, deterministic per (m,B)
 * (L23_matrixsimd's m-parity discipline), compile-time selected by
 * FFT45_QC the way FFT45_XF already selects families: arm[0] sets the
 * gate's bit reference and the other families auto-DQ, so the timing race
 * can never flip output bits across processes.
 * A fixed-y slab of Q is 45 chunks of 2880 B at Q4XS stride (129600 B
 * total, 2624 mod 4096 -- no fixed page-offset relation). */

/* z pass, one fixed-y slab, in place: 45 contiguous 2880 B pencils at
 * (x, y fixed), x-stride Q4XS.  Same DAG-by-y rule as q4_zplane so every
 * z pencil in the volume gets one DAG regardless of sweep parity:
 * PFA45R for y < 44 (qc_zslab), PFA45 for the y = 44 slab (qc_zslab44). */
static __attribute__((noinline))
void qc_zslab(double *Sy)
{
    _Pragma("GCC unroll 1")
    for (int x = 0; x < L; ++x) {
        double *b_ = Sy + (size_t)x * Q4XS;
        __asm__("" : "+r"(b_));
#define LDQZ(n)    LDU4(b_ + (size_t)(n) * 8)
#define STQZ(k, v) STU4(b_ + (size_t)(k) * 8, (v))
        PFA45R(LDQZ, STQZ);
#undef LDQZ
#undef STQZ
    }
}

static __attribute__((noinline))
void qc_zslab44(double *Sy)
{
    _Pragma("GCC unroll 1")
    for (int x = 0; x < L; ++x) {
        double *b_ = Sy + (size_t)x * Q4XS;
        __asm__("" : "+r"(b_));
#define LDQZ(n)    LDU4(b_ + (size_t)(n) * 8)
#define STQZ(k, v) STU4(b_ + (size_t)(k) * 8, (v))
        PFA45(LDQZ, STQZ);
#undef LDQZ
#undef STQZ
    }
}

/* x pass on one fixed-y slab, RAW stores (the sweep-local x(s+1), and the
 * final sweep of an odd-m chain): 45 pencils (z = 0..44) at Q4XS stride --
 * one slab-slice of q4_xpass_body, L2-fed under custody. */
static __attribute__((noinline))
void qc_xslab(double *Sy)
{
    _Pragma("GCC unroll 1")
    for (int z = 0; z < L; ++z) {
        double *b_ = Sy + (size_t)z * 8;
        __asm__("" : "+r"(b_));
#define LDQX(n)    LDU4(b_ + (size_t)(n) * Q4XS)
#define STQX(k, v) STU4(b_ + (size_t)(k) * Q4XS, (v))
        PFA45(LDQX, STQX);
#undef LDQX
#undef STQX
    }
}

/* carried x pass with the map fused at the store sites: q4_yplanem's exact
 * stash/pair shape.  c comes CONTIGUOUS per pencil from the pencil-order
 * CQ image -- this entry's own r6 vs2 negative (point-major c at x-store
 * sites = loads scattered over 45 pages, +139 us/vol) is why CQ exists. */
static __attribute__((noinline))
void qc_xslabm(double *Sy, const double *cSy)
{
    _Pragma("GCC unroll 1")
    for (int z = 0; z < L; ++z) {
        double       *b_ = Sy  + (size_t)z * 8;
        const double *c_ = cSy + (size_t)z * Q4YS;
        __asm__("" : "+r"(b_), "+r"(c_));
        vec  stv_ = VSPLAT(0.0);
        long stk_ = -1;
#define LDQX(n)    LDU4(b_ + (size_t)(n) * Q4XS)
#define STQXM(k, v) do {                                                     \
    if (stk_ < 0) { stv_ = (v); stk_ = (k); }                                \
    else {                                                                   \
        __m512d w0_ = (__m512d)(stv_ + LDU4(c_ + (size_t)stk_ * 8));         \
        __m512d w1_ = (__m512d)((v)  + LDU4(c_ + (size_t)(k)  * 8));         \
        map45_pair(&w0_, &w1_);                                              \
        STU4(b_ + (size_t)stk_ * Q4XS, (vec)w0_);                           \
        STU4(b_ + (size_t)(k)  * Q4XS, (vec)w1_);                           \
        stk_ = -1;                                                           \
    }                                                                        \
} while (0)
        PFA45(LDQX, STQXM);
        if (stk_ >= 0) {                     /* 45 is odd: flush the single */
            __m512d w_ = (__m512d)(stv_ + LDU4(c_ + (size_t)stk_ * 8));
            STU4(b_ + (size_t)stk_ * Q4XS, (vec)map45_one(w_));
        }
#undef LDQX
#undef STQXM
    }
}

/* pencil-order c for the Q sweeps' mapped x pass:
 *   CQ[((y*45+z)*45+x)*8 ..] = C4[((x*45+y)*45+z)*8 ..]
 * -- per fixed-y slab a 45x45 zmm transpose (x <-> z).  Reads are
 * slab-local (the same 45-chunk pattern the Q sweeps use), stores stream
 * sequentially.  Pure data movement, built once per group per chain call.
 * BORROWED: L45_mixedradix ice_r7's VGC pencil-order c copies (their -6%),
 * applied only to the orientation that needs it -- the P sweeps'
 * q4_yplanem keeps reading plane-local point-major C4. */
static __attribute__((noinline))
void qc_packQ(const double *C4, double *CQ)
{
    for (int y = 0; y < L; ++y) {
        const double *s0 = C4 + (size_t)y * Q4YS;
        double       *d0 = CQ + (size_t)y * Q4PLD;
        for (int z = 0; z < L; ++z)
            _Pragma("GCC unroll 5")
            for (int x = 0; x < L; ++x)
                _mm512_store_pd(
                    d0 + (size_t)z * Q4YS + (size_t)x * 8,
                    _mm512_load_pd(s0 + (size_t)x * Q4XS + (size_t)z * 8));
    }
}

/* qc-pf arm: during a sweep's z pass (pass 2, whose OWN loads are L2-hot
 * under custody -- the plane was pulled by pass 1), poke the NEXT plane/
 * slab's state and c lines T2 so pass 1 of the next plane finds them
 * closer than L3.  This is NOT the q4-cz trap re-tried: q4-cz poked into
 * a z pass whose own loads were COLD (they starved, +16%); here the z
 * pass has load-port headroom by construction.  L45_mixedradix ice_r7
 * flagged exactly this re-test ("the custody shape has different slack").
 * Pokes move no data: bit-identical, raced in the same pool. */
static __attribute__((noinline))
void qc_zplane_pf(double *Wp, const double *nxs, const double *nxc)
{
    _Pragma("GCC unroll 1")
    for (int y = 0; y < L - 1; ++y) {
        double *b_ = Wp + (size_t)y * Q4YS;
        __asm__("" : "+r"(b_));
        if (nxs) {
            const double *s_ = nxs + (size_t)y * Q4YS;
            const double *c_ = nxc + (size_t)y * Q4YS;
            _Pragma("GCC unroll 45")
            for (int n_ = 0; n_ < 45; ++n_) {
                __builtin_prefetch(s_ + 8 * n_, 0, 2);
                __builtin_prefetch(c_ + 8 * n_, 0, 2);
            }
        }
#define LDQZ(n)    LDU4(b_ + (size_t)(n) * 8)
#define STQZ(k, v) STU4(b_ + (size_t)(k) * 8, (v))
        PFA45R(LDQZ, STQZ);
#undef LDQZ
#undef STQZ
    }
    {
        double *b_ = Wp + (size_t)(L - 1) * Q4YS;
        __asm__("" : "+r"(b_));
        if (nxs) {
            const double *s_ = nxs + (size_t)(L - 1) * Q4YS;
            const double *c_ = nxc + (size_t)(L - 1) * Q4YS;
            _Pragma("GCC unroll 45")
            for (int n_ = 0; n_ < 45; ++n_) {
                __builtin_prefetch(s_ + 8 * n_, 0, 2);
                __builtin_prefetch(c_ + 8 * n_, 0, 2);
            }
        }
#define LDQZ(n)    LDU4(b_ + (size_t)(n) * 8)
#define STQZ(k, v) STU4(b_ + (size_t)(k) * 8, (v))
        PFA45(LDQZ, STQZ);
#undef LDQZ
#undef STQZ
    }
}

/* Q-sweep twin: pencil x pokes chunk x of the next slab (+ the next CQ
 * slab's pencil x), 45+45 lines per call. */
static __attribute__((noinline))
void qc_zslab_pf(double *Sy, const double *nxs, const double *nxc, int y44)
{
    _Pragma("GCC unroll 1")
    for (int x = 0; x < L; ++x) {
        double *b_ = Sy + (size_t)x * Q4XS;
        __asm__("" : "+r"(b_));
        if (nxs) {
            const double *s_ = nxs + (size_t)x * Q4XS;
            const double *c_ = nxc + (size_t)x * Q4YS;
            _Pragma("GCC unroll 45")
            for (int n_ = 0; n_ < 45; ++n_) {
                __builtin_prefetch(s_ + 8 * n_, 0, 2);
                __builtin_prefetch(c_ + 8 * n_, 0, 2);
            }
        }
        if (y44) {
#define LDQZ(n)    LDU4(b_ + (size_t)(n) * 8)
#define STQZ(k, v) STU4(b_ + (size_t)(k) * 8, (v))
            PFA45(LDQZ, STQZ);
#undef LDQZ
#undef STQZ
        } else {
#define LDQZ(n)    LDU4(b_ + (size_t)(n) * 8)
#define STQZ(k, v) STU4(b_ + (size_t)(k) * 8, (v))
            PFA45R(LDQZ, STQZ);
#undef LDQZ
#undef STQZ
        }
    }
}

/* qc2 arm: pencil-order c for the P sweeps as well (the other half of
 * L45_mixedradix's VGC): CP[x][z][y] so q4_yplanem's 2880 B-strided C4
 * reads become contiguous-per-pencil.  Same values at new addresses --
 * BIT-IDENTICAL to qc by construction, so both arms sit in one racing
 * pool.  CP is a per-x-plane 45x45 zmm transpose of C4 (fully
 * plane-local). */
static __attribute__((noinline))
void qc_packP(const double *C4, double *CP)
{
    for (int x = 0; x < L; ++x) {
        const double *s0 = C4 + (size_t)x * Q4PLD;
        double       *d0 = CP + (size_t)x * Q4PLD;
        for (int z = 0; z < L; ++z)
            _Pragma("GCC unroll 5")
            for (int y = 0; y < L; ++y)
                _mm512_store_pd(
                    d0 + (size_t)z * Q4YS + (size_t)y * 8,
                    _mm512_load_pd(s0 + (size_t)y * Q4YS + (size_t)z * 8));
    }
}

/* carried y pass with the map at stores, c CONTIGUOUS from CP (the qc2
 * twin of q4_yplanem; per-point arithmetic identical). */
static __attribute__((noinline))
void qc_yplanem2(double *Wp, const double *cp)
{
    _Pragma("GCC unroll 1")
    for (int z = 0; z < L; ++z) {
        double       *b_ = Wp + (size_t)z * 8;
        const double *c_ = cp + (size_t)z * Q4YS;
        __asm__("" : "+r"(b_), "+r"(c_));
        vec  stv_ = VSPLAT(0.0);
        long stk_ = -1;
#define LDQY(n)    LDU4(b_ + (size_t)(n) * Q4YS)
#define STQYM2(k, v) do {                                                    \
    if (stk_ < 0) { stv_ = (v); stk_ = (k); }                                \
    else {                                                                   \
        __m512d w0_ = (__m512d)(stv_ + LDU4(c_ + (size_t)stk_ * 8));         \
        __m512d w1_ = (__m512d)((v)  + LDU4(c_ + (size_t)(k)  * 8));         \
        map45_pair(&w0_, &w1_);                                              \
        STU4(b_ + (size_t)stk_ * Q4YS, (vec)w0_);                           \
        STU4(b_ + (size_t)(k)  * Q4YS, (vec)w1_);                           \
        stk_ = -1;                                                           \
    }                                                                        \
} while (0)
        PFA45(LDQY, STQYM2);
        if (stk_ >= 0) {                     /* 45 is odd: flush the single */
            __m512d w_ = (__m512d)(stv_ + LDU4(c_ + (size_t)stk_ * 8));
            STU4(b_ + (size_t)stk_ * Q4YS, (vec)map45_one(w_));
        }
#undef LDQY
#undef STQYM2
    }
}

/* volume interleave/deinterleave: 4 x 16 B <-> one zmm per point.  Pure
 * data movement (no arithmetic), so it cannot change bits. */
static inline __attribute__((always_inline))
__m512d q4_g4(const double *p, const size_t str)
{
#ifdef __AVX512DQ__
    return _mm512_mask_broadcast_f64x2(
        _mm512_mask_broadcast_f64x2(
            _mm512_mask_broadcast_f64x2(
                _mm512_castpd128_pd512(_mm_loadu_pd(p)),
                (__mmask8)0x0C, _mm_loadu_pd(p + str)),
            (__mmask8)0x30, _mm_loadu_pd(p + 2 * str)),
        (__mmask8)0xC0, _mm_loadu_pd(p + 3 * str));
#else
    __m256d lo = _mm256_insertf128_pd(
        _mm256_castpd128_pd256(_mm_loadu_pd(p)), _mm_loadu_pd(p + str), 1);
    __m256d hi = _mm256_insertf128_pd(
        _mm256_castpd128_pd256(_mm_loadu_pd(p + 2 * str)),
        _mm_loadu_pd(p + 3 * str), 1);
    return _mm512_insertf64x4(_mm512_castpd256_pd512(lo), hi, 1);
#endif
}

static __attribute__((noinline))
void q4_pack(const double *src, double *Q)      /* 4 volumes -> point-major */
{
    _Pragma("GCC unroll 4")
    for (size_t i = 0; i < Q4NPT; ++i)
        _mm512_store_pd(Q + 8 * i, q4_g4(src + 2 * i, VDBL));
}

static inline __attribute__((always_inline))
void q4_s4(__m512d v, double *p)                /* one point -> 4 volumes   */
{
    __m256d h = _mm512_extractf64x4_pd(v, 1);
    _mm_storeu_pd(p,            _mm512_castpd512_pd128(v));
    _mm_storeu_pd(p + VDBL,
                  _mm256_extractf128_pd(_mm512_castpd512_pd256(v), 1));
    _mm_storeu_pd(p + 2 * VDBL, _mm256_castpd256_pd128(h));
    _mm_storeu_pd(p + 3 * VDBL, _mm256_extractf128_pd(h, 1));
}

/* final map FUSED into the deinterleave: o = map(z+c) per point, scattered
 * straight to the 4 volumes of final_out -- deletes the separate in-place
 * map span's 11.7 MB round-trip.  Same pair ladder (one vdivpd per 8
 * points; pair grouping is value-transparent), so per-point bits match the
 * span+unpack form exactly.  Q4NPT is odd (91125): one map45_one tail. */
static __attribute__((noinline))
void q4_unpackm(const double *Q, const double *C4, double *dst)
{
    size_t i = 0;
    _Pragma("GCC unroll 2")
    for (; i + 2 <= Q4NPT; i += 2) {
        __m512d w0 = _mm512_add_pd(_mm512_load_pd(Q  + 8 * i),
                                   _mm512_load_pd(C4 + 8 * i));
        __m512d w1 = _mm512_add_pd(_mm512_load_pd(Q  + 8 * i + 8),
                                   _mm512_load_pd(C4 + 8 * i + 8));
        map45_pair(&w0, &w1);
        q4_s4(w0, dst + 2 * i);
        q4_s4(w1, dst + 2 * i + 2);
    }
    {   /* odd last point */
        __m512d w = _mm512_add_pd(_mm512_load_pd(Q  + 8 * i),
                                  _mm512_load_pd(C4 + 8 * i));
        q4_s4(map45_one(w), dst + 2 * i);
    }
}

#undef LDU4
#undef STU4
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef vec

#ifndef FFT45_XF
# define FFT45_XF 1
#endif

/* q4 chain: volume-major over GROUPS of 4 -- pack x0 + permute c once per
 * group per call (amortized over m steps), run the x-first step entirely in
 * Q, final map span in place, deinterleave once into final_out.  Remainder
 * volumes (B mod 4, or the whole batch when the arena is missing) run the
 * xf body -- bit-identical per point, so the arm stays gate-safe at any B. */
static inline __attribute__((always_inline))
void ch45_q4_body(const double *x0, const double *C, double *W, double *FO,
                  long m, long B, double *P, const int pfx, const int czf)
{
    long g = 0;
    if (g_q4ok) {
        double *Q  = W + Q4OFF(B);
        double *C4 = Q + Q4VD + Q4PAD;
        for (; g + 4 <= B; g += 4) {
            q4_pack(C  + (size_t)g * VDBL, C4);
            q4_pack(x0 + (size_t)g * VDBL, Q);
            for (long s = 1; s <= m; ++s) {
                if (pfx) q4_xp1(Q); else q4_xp0(Q);
                if (s < m)
                    for (int x = 0; x < L; ++x) {
                        if (czf) q4_zplanec(Q + (size_t)x * Q4PLD,
                                            C4 + (size_t)x * Q4PLD);
                        else     q4_zplane (Q + (size_t)x * Q4PLD);
                        q4_yplanem(Q + (size_t)x * Q4PLD,
                                   C4 + (size_t)x * Q4PLD);
                    }
                else
                    for (int x = 0; x < L; ++x) {
                        q4_zplane(Q + (size_t)x * Q4PLD);
                        q4_yplane(Q + (size_t)x * Q4PLD);
                    }
            }
            q4_unpackm(Q, C4, FO + (size_t)g * VDBL);
        }
    }
    if (g < B)
        ch45_xf_body(x0 + (size_t)g * VDBL, C + (size_t)g * VDBL,
                     W + (size_t)g * VDBL, FO + (size_t)g * VDBL,
                     m, B - g, P, pfx);
}

static void ch45_q4_zs(const double *x0, const double *C, double *W,
                       double *FO, long m, long B, double *P)
{
    ch45_q4_body(x0, C, W, FO, m, B, P, 0, 0);
}

static void ch45_q4_pf(const double *x0, const double *C, double *W,
                       double *FO, long m, long B, double *P)
{
    ch45_q4_body(x0, C, W, FO, m, B, P, 1, 0);
}

static __attribute__((unused))
void ch45_q4_cz(const double *x0, const double *C, double *W,
                double *FO, long m, long B, double *P)
{
    ch45_q4_body(x0, C, W, FO, m, B, P, 0, 1);
}

#ifndef FFT45_QC
# define FFT45_QC 1
#endif

/* ice_r8 custody chain (see the qc pass block above for the schedule).
 * Entry/exit identical to q4: pack x0 + permute c once per group per call
 * (plus the CQ slab transpose), remainder volumes run the xf body.
 * cp2 (the qc2 arm) also builds CP and reads the P-sweep map's c from it
 * contiguously -- bit-identical, so qc/qc2 race in one pool. */
static inline __attribute__((always_inline))
void ch45_qc_body(const double *x0, const double *C, double *W,
                  double *FO, long m, long B, double *P, const int cp2,
                  const int pf)
{
    long g = 0;
    if (g_q4ok) {
        double *Q  = W + Q4OFF(B);
        double *C4 = Q + Q4VD + Q4PAD;
        double *CQ = C4 + Q4VD + QCPAD;
        double *CP = CQ + Q4VD + QCPAD;
        for (; g + 4 <= B; g += 4) {
            q4_pack(C  + (size_t)g * VDBL, C4);
            qc_packQ(C4, CQ);
            if (cp2) qc_packP(C4, CP);
            q4_pack(x0 + (size_t)g * VDBL, Q);
            for (int x = 0; x < L; ++x) {      /* sweep 0 (P): z(1), y(1)   */
                q4_zplane(Q + (size_t)x * Q4PLD);
                q4_yplane(Q + (size_t)x * Q4PLD);
            }
            for (long s = 1; s < m; ++s) {
                if (s & 1)              /* Q sweep: x(s)+map, z(s+1), x(s+1) */
                    for (int y = 0; y < L; ++y) {
                        double *Sy = Q + (size_t)y * Q4YS;
                        qc_xslabm(Sy, CQ + (size_t)y * Q4PLD);
                        if (pf) {
                            const int nx = y + 1 < L;
                            qc_zslab_pf(Sy,
                                nx ? Q  + (size_t)(y + 1) * Q4YS  : NULL,
                                nx ? CQ + (size_t)(y + 1) * Q4PLD : NULL,
                                y == L - 1);
                        } else if (y == L - 1) qc_zslab44(Sy);
                        else                   qc_zslab(Sy);
                        qc_xslab(Sy);
                    }
                else                    /* P sweep: y(s)+map, z(s+1), y(s+1) */
                    for (int x = 0; x < L; ++x) {
                        double *Px = Q + (size_t)x * Q4PLD;
                        if (cp2) qc_yplanem2(Px, CP + (size_t)x * Q4PLD);
                        else     q4_yplanem (Px, C4 + (size_t)x * Q4PLD);
                        if (pf) {
                            const int nx = x + 1 < L;
                            qc_zplane_pf(Px,
                                nx ? Q  + (size_t)(x + 1) * Q4PLD : NULL,
                                nx ? C4 + (size_t)(x + 1) * Q4PLD : NULL);
                        } else q4_zplane(Px);
                        q4_yplane(Px);
                    }
            }
            if (m & 1)                  /* final sweep: carried axis RAW    */
                for (int y = 0; y < L; ++y)
                    qc_xslab(Q + (size_t)y * Q4YS);
            else
                for (int x = 0; x < L; ++x)
                    q4_yplane(Q + (size_t)x * Q4PLD);
            q4_unpackm(Q, C4, FO + (size_t)g * VDBL);
        }
    }
    if (g < B)
        ch45_xf_body(x0 + (size_t)g * VDBL, C + (size_t)g * VDBL,
                     W + (size_t)g * VDBL, FO + (size_t)g * VDBL,
                     m, B - g, P, 0);
}

static void ch45_qc_zs(const double *x0, const double *C, double *W,
                       double *FO, long m, long B, double *P)
{
    ch45_qc_body(x0, C, W, FO, m, B, P, 0, 0);
}

static void ch45_qc2_zs(const double *x0, const double *C, double *W,
                        double *FO, long m, long B, double *P)
{
    ch45_qc_body(x0, C, W, FO, m, B, P, 1, 0);
}

static __attribute__((unused))
void ch45_qc_pf(const double *x0, const double *C, double *W,
                double *FO, long m, long B, double *P)
{
    ch45_qc_body(x0, C, W, FO, m, B, P, 0, 1);
}

/* rank: the family named by FFT45_XF leads the pool (its arm[0] sets the
 * bit reference; the other family auto-DQs at the gate, so the adaptive
 * pick can never flip bits across processes).  Within the z-first family,
 * vt-pf keeps its r5 rank-0 (it beat vt in all three r5 dev windows);
 * the vs split arms sit at the tail as priced controls (r6: the lazy
 * split staged-span and store-site-map placements both measured out --
 * see the strategy record). */
static const struct { chain45_fn fn; int rank; const char *nm; } g_ch45[] = {
#if FFT45_XF
# if FFT45_QC
    /* ice_r8: the custody family is a NEW bit class (axis order alternates
     * with sweep parity), so it is compile-time selected exactly like
     * FFT45_XF selects families: qc leads the TABLE (arm[0] sets the
     * gate's bit reference) and the x-first arms below rel-pass but
     * bit-DQ, printing as OUT by design.  If qc ever fails the rel gate,
     * the first rel-passing arm (q4-pf) sets the bits instead and the r7
     * behavior returns wholesale. */
    { ch45_qc_zs,  0, "qc"     },   /* ice_r8: custody sweeps              */
    { ch45_qc2_zs, 1, "qc2"    },   /* + pencil-order c in the P sweeps:
                                     * priced +0.8-0.9%, 3/3 forced A/Bs
                                     * (215.2-215.5 vs 217.0-217.3) -- kept
                                     * as the raced control */
    /* qc-pf (next-plane T2 pokes in the custody z pass) PRICED OUT and
     * removed: forced A/B read 258.2/258.8 vs qc 215.9/216.9 (+20%, 2/2
     * clean pairs).  The poke tax holds even when the poked-into pass's
     * own loads are L2-HIT: the z pass is ISSUE-bound at ~2x floor, and
     * 90 poke uops/call is +18% port pressure.  Fourth poke-family
     * negative in this lineage (pf ladder r2, q4-cz r7, cpf, this). */
    { ch45_q4_pf,  3, "q4-pf"  },
    { ch45_q4_zs,  4, "q4"     },
    { ch45_xf_pf,  5, "xf-pf"  },
    { ch45_xf_zs,  6, "xf"     },
# else
    /* ice_r8 rank swap inside the x-first family: the r7 SCORED drained
     * window read q4-pf 253.5 < q4 255.5 us/step (my own dev windows had
     * them a dead tie) -- per the r5 rule the rank moves the same round
     * the drained window contradicts the bet. */
    { ch45_xf_pf,  3, "xf-pf"  },
    { ch45_xf_zs,  4, "xf"     },
    { ch45_q4_zs,  1, "q4"     },   /* ice_r7: quad-volume lanes           */
    { ch45_q4_pf,  0, "q4-pf"  },
# endif
    /* q4-cz (c-poke in the z-pass) PRICED OUT and removed from the pool:
     * same-window driver A/B read 287.6 vs q4's 248.6/247.1 us/xform
     * (+16%) -- the 2025 poke uops/plane starve the z-pass's own loads and
     * the y-map's c loads were already OoO-covered (the CPFY rule, third
     * confirmation in this lineage).  ch45_q4_cz stays compiled (unused)
     * as the priced negative's reference implementation. */
#endif
    { ch45_vt_pf,  4, "vt-pf"  },
    { ch45_vt_zs,  5, "vt"     },
    { ch45_vm_zs,  6, "vm-zs"  },
    { ch45_uf,     9, "uf"     },
    { ch45_vs_zs,  8, "vs"     },   /* lazy-map split: staged-span trap,   */
                                    /* kept as the priced control          */
    { ch45_vs2_zs, 7, "vs2"    },
#if !FFT45_XF
    { ch45_xf_pf,  9, "xf-pf"  },
    { ch45_xf_zs, 10, "xf"     },
    { ch45_q4_pf, 13, "q4-pf"  },   /* other bit class: auto-DQ, priced 0  */
    { ch45_q4_zs, 14, "q4"     },
    { ch45_qc_zs, 15, "qc"     },
    { ch45_qc2_zs, 16, "qc2"   },
    { ch45_qc_pf, 17, "qc-pf"  },
#endif
};
#define NCH45 ((int)(sizeof g_ch45 / sizeof g_ch45[0]))
#endif /* HAVE_PW4 */

/* ---- plan, tuner, API ---------------------------------------------------- */

enum { M_INPLACE = 0, M_SCRATCHP = 1, M_CPY = 2 };

/* candidate table, rank = simplest-first tie-break order (3% hysteresis).
 * pw4 ranks ahead of pw2 (the V1-first hardening, L36_mixedradix r6): a
 * narrower kernel must now beat pw4's minimum by >3%, not win a coin flip.
 * ice_r2 pool: cp candidates NEW (pf codes 6/7/8); pf3/pf3a ranked LAST
 * (prefetchw is a measured +7.4% when the store target is L3-resident --
 * L13_rader's gate rule -- and the graded chain is L3-resident); pw2 pool
 * cut to the single pf0 fallback (256-bit lost every ice_r1 tournament). */
struct cand45 {
    exec45_fn   fn;
    int         pw, mode, pf, rank;
    const char *nm;
};
static const struct cand45 g_cands[] = {
#ifdef HAVE_PW4
    /* rank note (ice_r2): zal ranks AHEAD of plain ip0.  Its win on this
     * node is real but ~2.5% -- inside the 3% hysteresis -- so with ip0 at
     * rank 0 the tie-break would suppress the faster plan forever. */
    /* rank = tie-break preference inside the 3% hysteresis.  ice_r5: tr0
     * first (the transpose-free phase 1: -9.6..-12% for L45_mixedradix on
     * this node, and it carries the chain's structure -- give it the tie).
     * ip0 next: the r4 DRAINED window read ip0=259.8 < il0=262.2, so il0's
     * ice_r2 rank-0 bet is reversed here.  pf3/pf3a last: prefetchw is a
     * measured +7.4% when the store target is L3-resident (L13_rader),
     * which the graded chain guarantees. */
    { x_tr0_pw4,  4, M_INPLACE, 15, 0,  "pw4-tr0"     },   /* ice_r5: tr   */
    { x_ip0_pw4,  4, M_INPLACE,  0, 1,  "pw4-ip-pf0"  },
    { x_ip1_pw4,  4, M_INPLACE,  1, 4,  "pw4-ip-pf1"  },
    { x_ip2_pw4,  4, M_INPLACE,  2, 5,  "pw4-ip-pf2"  },
    { x_ip3_pw4,  4, M_INPLACE,  3, 11, "pw4-ip-pf3"  },
#ifndef FFT45_OVERLAP_TAIL
    { x_ip0a_pw4, 4, M_INPLACE,  4, 3,  "pw4-ip-pf0a" },   /* r11: zal     */
    { x_ip3a_pw4, 4, M_INPLACE,  5, 12, "pw4-ip-pf3a" },   /* r11: zal     */
    { x_cp0a_pw4, 4, M_CPY,      7, 7,  "pw4-cp-pf0a" },   /* ice_r2       */
    { x_il0a_pw4, 4, M_INPLACE, 10, 6,  "pw4-il-pf0a" },   /* ice_r2: il   */
#endif
    { x_cp0_pw4,  4, M_CPY,      6, 8,  "pw4-cp-pf0"  },   /* ice_r2       */
    { x_cp2_pw4,  4, M_CPY,      8, 9,  "pw4-cp-pf2"  },   /* ice_r2       */
    { x_il0_pw4,  4, M_INPLACE,  9, 2,  "pw4-il-pf0"  },   /* ice_r2: il   */
    { x_hz0_pw4,  4, M_INPLACE, 11, 14, "pw4-hz-pf0"  },   /* ice_r2: hz   */
    { x_hzil_pw4, 4, M_INPLACE, 12, 15, "pw4-hzil"    },   /* ice_r2       */
    { x_sp0_pw4,  4, M_SCRATCHP, 0, 9,  "pw4-sp-pf0"  },
    { x_sps_pw4,  4, M_SCRATCHP, 2, 10, "pw4-sp-pfs"  },
#endif
    { x_ip0_pw2,  2, M_INPLACE,  0, 13, "pw2-ip-pf0"  },
};
#define NCAND ((int)(sizeof g_cands / sizeof g_cands[0]))

struct fft3d_plan {
    int       batch;
    exec45_fn fn;
    int       charm;             /* ice_r4: picked chain arm (g_ch45 index) */
    double   *S;                 /* padded scratch volume (SCRATCHP mid)    */
    double   *P;                 /* plane scratch: page-aligned heap, NOT the
                                    stack -- a stack plane 4K-aliases the
                                    in/out streams in unlucky runs (measured
                                    bimodal 204 vs 377 us at B=1, r6) */
    double   *W;                 /* ice_r5: chain state arena (4096-aligned,
                                    batch volumes + a page of phase slack;
                                    the chain places the state at page phase
                                    (phase(c)+2048)&4095).  On no-AVX512
                                    builds it doubles as the fallback z
                                    buffer. */
    void     *rawS, *rawP, *rawW;
};

#ifdef HAVE_PW4
/* chain state base: the plan arena at page phase (phase(c)+2048)&4095 --
 * maximally far from c's phase (only state-vs-c matters in the steady loop:
 * x0/final_out are touched once).  Falls back to FO when the arena alloc
 * failed (= the ice_r4 behavior). */
static double *ch45_state(fft3d_plan *p, const double *C, double *FO)
{
    if (!p->W) return FO;
    size_t off = ((((uintptr_t)C & 4095) + 2048) & 4095) & ~(size_t)63;
    return (double *)(void *)((char *)p->W + off);
}
#endif

const char *fft3d_name(void) { return "L45_pfa"; }

static char g_desc[512];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "Good-Thomas PFA 9x5, two sweeps + fft3d_chain (tr "
                       "bcast-gather p1, row-group lazy map, rsqrt14+2NR, "
                       "1 vdivpd/8pt); arms {vt,vt-pf,vm,uf} raced bit-"
                       "identically";
}
int fft3d_supports(int Lq) { return Lq == L; }

/* scalar O(L^2)-per-line reference: independent ground truth for the gate */
static void ref3d(const double _Complex *in, double _Complex *out)
{
    double _Complex Wt[L], buf[L];
    for (int k = 0; k < L; ++k)
        Wt[k] = cexp(-2.0 * M_PI * I * (double)k / (double)L);
    for (int x = 0; x < L; ++x)                       /* z axis: in -> out */
        for (int y = 0; y < L; ++y) {
            const double _Complex *r = in  + ((size_t)x * L + y) * L;
            double _Complex       *w = out + ((size_t)x * L + y) * L;
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += r[j] * Wt[(j * k) % L];
                w[k] = s;
            }
        }
    for (int x = 0; x < L; ++x)                       /* y axis, in place  */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)x * NPLANE + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * L];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * L] = s;
            }
        }
    for (int y = 0; y < L; ++y)                       /* x axis, in place  */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)y * L + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * NPLANE];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * NPLANE] = s;
            }
        }
}

static volatile double g_spin_sink;     /* clock-settle spin escape hatch */
static double g_clk;                    /* ice_r5: spin-derived clock, GHz */

/* ice_r2, LOUD-only: minimal perf_event probe (cycles + instructions + two
 * raw ICX events), for the phase decomposition.  The brief says the node's
 * PMU is exposed; degrade to -1 readings anywhere it is not. */
#ifdef FFT45_LOUD
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
static long p45_open(unsigned type, unsigned long long conf, int group)
{
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = type; a.config = conf;
    a.disabled = (group < 0); a.exclude_kernel = 1; a.exclude_hv = 1;
    return syscall(SYS_perf_event_open, &a, 0, -1, group, 0);
}
#endif

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int rel_ok(const double *got, const double *ref, size_t n)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = got[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
    }
    return num <= den * 1e-26;                        /* rel L2 < 1e-13 */
}

fft3d_plan *fft3d_create(int Lq, int batch)
{
    if (Lq != L || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;
    if (posix_memalign(&p->rawS, 4096, SVDBL * sizeof(double)) != 0) {
        free(p); return NULL;
    }
    p->S = (double *)p->rawS;
    memset(p->S, 0, SVDBL * sizeof(double));  /* pad columns stay 0 forever */
    /* plane scratch + (ice_r2) the CPY plane image right behind it */
    if (posix_memalign(&p->rawP, 4096, (IMGOFF + PLND) * sizeof(double)) != 0) {
        free(p->rawS); free(p); return NULL;
    }
    p->P = (double *)p->rawP;
    memset(p->P, 0, (IMGOFF + PLND) * sizeof(double));
    p->fn = g_cands[0].fn;                            /* safe default        */

    int    live[NCAND];
    double tc[NCAND];
    for (int c = 0; c < NCAND; ++c) { live[c] = 1; tc[c] = 1e300; }

    /* run-time forcing for the monitor's control jobs */
    { const char *e;
      if ((e = getenv("FFT45_PW"))) {
          int v = atoi(e);
          for (int c = 0; c < NCAND; ++c) if (g_cands[c].pw != v) live[c] = 0;
      }
      if ((e = getenv("FFT45_MODE"))) {
          int v = (e[0] >= '0' && e[0] <= '9') ? atoi(e)
                : (e[0] == 's' ? M_SCRATCHP : M_INPLACE);
          for (int c = 0; c < NCAND; ++c) if (g_cands[c].mode != v) live[c] = 0;
      }
      if ((e = getenv("FFT45_PF"))) {
          int v = atoi(e);
          for (int c = 0; c < NCAND; ++c) if (g_cands[c].pf != v) live[c] = 0;
      }
      { int any = 0;
        for (int c = 0; c < NCAND; ++c) any |= live[c];
        if (!any) for (int c = 0; c < NCAND; ++c) live[c] = 1; } }

    /* tuning arena: must actually stream at large batch (L36_pfa r2 lesson);
     * 32 volumes = 2 x 44.5 MB in+out, past both machines' L3 on the walk */
    const int nv = batch < 32 ? batch : 32;
    void *ri = NULL, *ro = NULL, *r0 = NULL, *r1 = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&r0, 64, VDBL * sizeof(double)) ||
        (nv > 1 && posix_memalign(&r1, 64, VDBL * sizeof(double)))) {
        free(ri); free(ro); free(r0);
        snprintf(g_desc, sizeof g_desc,
                 "GT-PFA 9x5 two-sweep; tuner SKIPPED (arena alloc failed): %s",
                 g_cands[0].nm);
        return p;
    }
    double *tin = ri, *tout = ro, *ref0 = r0, *refN = r1;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }
    ref3d((const double _Complex *)tin, (double _Complex *)ref0);
    if (nv > 1)          /* gate the LAST volume too: catches S-reuse bugs */
        ref3d((const double _Complex *)(tin + (size_t)(nv - 1) * VDBL),
              (double _Complex *)refN);

    for (int c = 0; c < NCAND; ++c) {
        if (!live[c]) continue;
        memset(tout, 0, (size_t)nv * VDBL * sizeof(double));
        g_cands[c].fn(tin, tout, nv, p->S, p->P);
        if (!rel_ok(tout, ref0, VDBL)) live[c] = 0;
        if (nv > 1 && live[c] &&
            !rel_ok(tout + (size_t)(nv - 1) * VDBL, refN, VDBL)) live[c] = 0;
    }
    /* Clock-settle spin (ice_r2; BORROWED from L17_matrixsimd ice_r1,
     * transitively L17_winograd): ~150 ms of dense FMA before anything is
     * timed.  The node's schedutil governor hands a short create() an
     * unramped core (2.90 GHz base vs 3.30 ramped) and a ramp edge inside
     * the tournament mis-ranks candidates -- the mechanism behind this
     * entry's 12% ice_r1 run spread (the plan re-raced on every run).
     * Costs unscored plan time only.  ice_r5: the spin now also yields a
     * clock ESTIMATE for the description (the loop is latency-bound at 4
     * cycles/iteration -- scalar FMA latency on ICX -- whether or not gcc
     * SLPs the four chains into one vector): L36_pfa's ice_r5 decode showed
     * a 3.3-vs-2.9 GHz window class explains every >10% swing, so print
     * which clock the plan raced under. */
    {
        double t0 = now_s(), t1;
        long   it = 0;
        double sa = 1.0, sb = 1.0, sc = 1.0, sd = 1.0;
        const double sm = 1e-15;
        do {
            for (long i = 0; i < 200000; ++i) {
                sa = sa * sm + sa; sb = sb * sm + sb;
                sc = sc * sm + sc; sd = sd * sm + sd;
            }
            it += 200000;
            t1 = now_s();
        } while (t1 - t0 < 0.15);
        g_spin_sink = sa + sb + sc + sd;
        g_clk = 4.0e-9 * (double)it / (t1 - t0);
    }

    /* small arenas get more interleaved rounds: wallaby toggles a fast/slow
     * machine state on a seconds scale, and a toggle edge crossing the
     * tournament mis-ranks kernels (observed here in r8: a slow-window race
     * installed sp-pfs at B=1, 205 vs 175 us).  Min over more rounds gives
     * every candidate a shot at the fast state.  (L36/L45_mixedradix's
     * tuner-hardening lesson.)  ice_r2: nv=4 rounds 6 -> 8 (the graded case
     * IS nv=4; buy the extra samples there). */
    const int R = (nv >= 8) ? 1 : (nv >= 2 ? 3 : 8);
    const int NR = (nv >= 16) ? 4 : (nv >= 4 ? 8 : 10);
    for (int round = 0; round < NR; ++round)
        for (int c = 0; c < NCAND; ++c) {
            if (!live[c]) continue;
            /* self-warm so each candidate is timed from its own steady state */
            g_cands[c].fn(tin, tout, nv, p->S, p->P);
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                g_cands[c].fn(tin, tout, nv, p->S, p->P);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = -1;
    for (int c = 0; c < NCAND; ++c)
        if (live[c] && (best < 0 || tc[c] < tc[best])) best = c;
    int pick = best < 0 ? 0 : best;
    if (best >= 0)
        for (int c = 0; c < NCAND; ++c)
            if (live[c] && tc[c] <= tc[best] * 1.03 &&
                g_cands[c].rank < g_cands[pick].rank) pick = c;
    p->fn = g_cands[pick].fn;
    /* carry the head-to-head that matters (inplace vs cpy) onto the
     * leaderboard description, L45_mixedradix-style, for the next round */
    { double t_ip0 = -1, t_il0 = -1, t_tr0 = -1;
      for (int c = 0; c < NCAND; ++c) {
          if (live[c] && tc[c] < 1e299) {
              if (!strcmp(g_cands[c].nm, "pw4-ip-pf0")) t_ip0 = tc[c] * 1e6 / nv;
              if (!strcmp(g_cands[c].nm, "pw4-il-pf0")) t_il0 = tc[c] * 1e6 / nv;
              if (!strcmp(g_cands[c].nm, "pw4-tr0"))    t_tr0 = tc[c] * 1e6 / nv;
          }
      }
      snprintf(g_desc, sizeof g_desc,
               "GT-PFA 9x5 2-sweep +tr(bcast-gather p1); pick %s %.1f us/vol "
               "(tr0=%.1f ip0=%.1f il0=%.1f) B=%d nv=%d clk=%.2f",
               g_cands[pick].nm,
               tc[pick] < 1e299 ? tc[pick] * 1e6 / nv : -1.0,
               t_tr0, t_ip0, t_il0, batch, nv, g_clk);
    }

#ifdef FFT45_LOUD
    if (1) {
#else
    if (getenv("FFT45_VERBOSE")) {
#endif
        for (int c = 0; c < NCAND; ++c)
            fprintf(stderr, "L45_pfa tuner: %-12s  %s  %.1f us/vol\n",
                    g_cands[c].nm, live[c] ? "ok " : "OUT",
                    live[c] ? tc[c] * 1e6 / nv : 0.0);
        fprintf(stderr, "L45_pfa tuner: chose %s (nv=%d)\n",
                g_cands[pick].nm, nv);
#ifdef HAVE_PW4
        {   /* ice_r2 phase decomposition on THIS machine */
            double t1 = 1e300, t2 = 1e300;
            for (int round = 0; round < 3; ++round) {
                x_p1o_pw4(tin, tout, nv, p->S, p->P);
                double t0 = now_s();
                x_p1o_pw4(tin, tout, nv, p->S, p->P);
                double t = now_s() - t0; if (t < t1) t1 = t;
                x_p2o_pw4(tin, tout, nv, p->S, p->P);
                t0 = now_s();
                x_p2o_pw4(tin, tout, nv, p->S, p->P);
                t = now_s() - t0; if (t < t2) t2 = t;
            }
            fprintf(stderr, "L45_pfa probe: p1=%.1f p2=%.1f us/vol\n",
                    t1 * 1e6 / nv, t2 * 1e6 / nv);
        }
        {   /* phase-1 stall decomposition */
            struct { exec45_fn f; const char *nm; } pr[] = {
                { x_p1i_pw4, "p1i(hot-store)" },
                { x_p1h_pw4, "p1h(hot-read)"  },
                { x_p1c_pw4, "p1c(both-hot)"  },
            };
            for (int k = 0; k < 3; ++k) {
                double tb = 1e300;
                for (int round = 0; round < 3; ++round) {
                    pr[k].f(tin, tout, nv, p->S, p->P);
                    double t0 = now_s();
                    pr[k].f(tin, tout, nv, p->S, p->P);
                    double t = now_s() - t0; if (t < tb) tb = t;
                }
                fprintf(stderr, "L45_pfa probe: %s=%.1f us/vol\n",
                        pr[k].nm, tb * 1e6 / nv);
            }
        }
#ifdef FFT45_LOUD
        {   /* PMU: IPC + memory-stall + store-bound shares per phase probe.
             * Raw events (ICX): CYCLE_ACTIVITY.STALLS_MEM_ANY = A3/14/c20,
             * EXE_ACTIVITY.BOUND_ON_STORES = A6/40. */
            long fd_cyc = p45_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1);
            long fd_ins = fd_cyc < 0 ? -1 :
                p45_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, (int)fd_cyc);
            long fd_mem = fd_cyc < 0 ? -1 :
                p45_open(PERF_TYPE_RAW, 0x140014A3ull, (int)fd_cyc);
            long fd_st  = fd_cyc < 0 ? -1 :
                p45_open(PERF_TYPE_RAW, 0x40A6ull, (int)fd_cyc);
            if (fd_cyc >= 0) {
                struct { exec45_fn f; const char *nm; } pp[] = {
                    { x_p1c_pw4, "p1c" }, { x_p1o_pw4, "p1" }, { x_p2o_pw4, "p2" },
                };
                for (int k = 0; k < 3; ++k) {
                    pp[k].f(tin, tout, nv, p->S, p->P);      /* warm */
                    ioctl(fd_cyc, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
                    ioctl(fd_cyc, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
                    pp[k].f(tin, tout, nv, p->S, p->P);
                    ioctl(fd_cyc, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
                    long long c = 0, ii = 0, m = 0, st = 0;
                    if (read(fd_cyc, &c, 8) != 8) c = 0;
                    if (fd_ins >= 0 && read(fd_ins, &ii, 8) != 8) ii = 0;
                    if (fd_mem >= 0 && read(fd_mem, &m, 8) != 8) m = 0;
                    if (fd_st  >= 0 && read(fd_st, &st, 8) != 8) st = 0;
                    fprintf(stderr,
                            "L45_pfa pmu %-3s: %.0f cyc/vol  %.0f ins/vol  "
                            "ipc=%.2f  stall_mem=%.0f%%  bound_st=%.0f%%\n",
                            pp[k].nm, (double)c / nv, (double)ii / nv,
                            c ? (double)ii / (double)c : 0.0,
                            c ? 100.0 * (double)m / (double)c : 0.0,
                            c ? 100.0 * (double)st / (double)c : 0.0);
                }
            } else fprintf(stderr, "L45_pfa pmu: unavailable\n");
            if (fd_st >= 0) close(fd_st);
            if (fd_mem >= 0) close(fd_mem);
            if (fd_ins >= 0) close(fd_ins);
            if (fd_cyc >= 0) close(fd_cyc);
        }
#endif
#endif
    }

#ifdef HAVE_PW4
    /* ice_r5: the chain state arena (see ch45_state) -- batch volumes plus
     * a page of phase slack, 4096-aligned so the phase written by
     * ch45_state is deterministic; memset so first-touch faults land here,
     * not in the first timed chain.  ice_r6: sized for the vs split arms
     * (batch SPVST-strided volumes + the split-c copy at +2048 B phase
     * skew); SPVST > VDBL so the vt arms fit in the same block. */
    {
        /* ice_r7: grown to also hold the q4 arms' point-major group arena
         * (Q at Q4OFF(batch), permuted-c copy behind it at a controlled
         * 64B-aligned +2048 B page phase).  Layout of the first
         * (batch+1)*SPVST doubles is unchanged for the older arms. */
        const size_t wbytes =
            (Q4END(batch) + 512) * sizeof(double) + 8192;
        if (posix_memalign(&p->rawW, 4096, wbytes) == 0) {
            p->W = (double *)p->rawW;
#if FFT45_THP && defined(MADV_HUGEPAGE)
            /* BORROWED from rival v6_5a869e40: 2 MB hugepages on the hot
             * arena kill dTLB thrash in the 45-page x-pass walk.  madvise
             * only the interior 2M-aligned span of the q4 region; first
             * touch (the memset below) then faults hugepages in. */
            {
                char *qb = (char *)(p->W + Q4OFF(batch));
                size_t qs = (4 * Q4VD + Q4PAD + 2 * QCPAD) * sizeof(double);
                uintptr_t a0 = ((uintptr_t)qb + ((1u << 21) - 1))
                               & ~(uintptr_t)((1u << 21) - 1);
                uintptr_t a1 = ((uintptr_t)qb + qs)
                               & ~(uintptr_t)((1u << 21) - 1);
                if (a1 > a0) (void)madvise((void *)a0, a1 - a0, MADV_HUGEPAGE);
            }
#endif
            memset(p->rawW, 0, wbytes);
            g_q4ok = 1;
        }
    }
    /* ---- ice_r4: gate + race the chain arms (all bit-identical) ---------
     * Reference = the gated exec pick + the exact scalar map, chained a few
     * steps over synthetic 0.1-scaled c.  Arms must (a) match it in rel L2
     * and (b) match EACH OTHER bitwise -- an arm that breaks either is
     * dropped, so the per-process adaptive pick can never change bits. */
    {
        void *rc = NULL, *rrf = NULL, *rbt = NULL, *rrf2 = NULL, *rbt2 = NULL;
        const size_t adbl = (size_t)nv * VDBL;
        if (posix_memalign(&rc, 64, adbl * sizeof(double)) == 0 &&
            posix_memalign(&rrf, 64, adbl * sizeof(double)) == 0 &&
            posix_memalign(&rbt, 64, adbl * sizeof(double)) == 0 &&
            posix_memalign(&rrf2, 64, adbl * sizeof(double)) == 0 &&
            posix_memalign(&rbt2, 64, adbl * sizeof(double)) == 0) {
            double *cN = rc, *rr = rrf, *rb = rbt, *rr2 = rrf2, *rb2 = rbt2;
            unsigned long long q = 0xD1B54A32D192ED03ull;
            for (size_t i = 0; i < adbl; ++i) {
                q = q * 6364136223846793005ull + 1442695040888963407ull;
                cN[i] = 0.1 * (double)(long long)(q >> 11) * 0x1p-53;
            }
            int msg = getenv("FFT45_MSG") ? atoi(getenv("FFT45_MSG")) : 3;
            if (msg < 1) msg = 1;
            /* ice_r8: the reference chain is extended one step so every arm
             * is gated at BOTH m = msg and m = msg+1 -- the custody family's
             * sweep types and final-sweep parity all get exercised (the
             * L45_mixedradix ice_r7 m=2/m=3 gate discipline). */
            memcpy(rr, tin, adbl * sizeof(double));
            for (int s2 = 0; s2 < msg; ++s2) {
                p->fn(rr, tout, nv, p->S, p->P);
                map45_ref(tout, cN, rr, adbl / 2);
            }
            memcpy(rr2, rr, adbl * sizeof(double));
            p->fn(rr2, tout, nv, p->S, p->P);
            map45_ref(tout, cN, rr2, adbl / 2);
            int    chlive[NCH45];
            double tch[NCH45];
            int    have_bits = 0, have_bits2 = 0;
            const int chv =
#ifdef FFT45_LOUD
                1;
#else
                getenv("FFT45_VERBOSE") != 0;
#endif
            for (int c2 = 0; c2 < NCH45; ++c2) {
                chlive[c2] = 1; tch[c2] = 1e300;
                for (int mt = 0; mt < 2 && chlive[c2]; ++mt) {
                    const double *rf = mt ? rr2 : rr;
                    double       *bf = mt ? rb2 : rb;
                    int          *hb = mt ? &have_bits2 : &have_bits;
                    memset(tout, 0, adbl * sizeof(double));
                    g_ch45[c2].fn(tin, cN, ch45_state(p, cN, tout), tout,
                                  msg + mt, nv, p->P);
                    if (!rel_ok(tout, rf, adbl)) {
                        chlive[c2] = 0;
                        if (chv) {
                            double nu = 0, de = 0;
                            for (size_t i = 0; i < adbl; ++i) {
                                double dd = tout[i] - rf[i];
                                nu += dd * dd; de += rf[i] * rf[i];
                            }
                            fprintf(stderr, "L45_pfa chain gate: %s REL FAIL "
                                    "m=%d rel_l2=%.3e\n", g_ch45[c2].nm,
                                    msg + mt,
                                    de > 0 ? sqrt(nu / de) : -1.0);
                        }
                        continue;
                    }
                    if (!*hb) {
                        memcpy(bf, tout, adbl * sizeof(double));
                        *hb = 1;
                    } else if (memcmp(bf, tout, adbl * sizeof(double))) {
                        chlive[c2] = 0;     /* bit divergence = disqualified */
                        if (chv) {
                            size_t fi = 0;
                            while (fi < adbl && bf[fi] == tout[fi]) ++fi;
                            fprintf(stderr, "L45_pfa chain gate: %s BIT FAIL "
                                    "m=%d first at [%zu] %.17g vs %.17g\n",
                                    g_ch45[c2].nm, msg + mt, fi,
                                    fi < adbl ? tout[fi] : 0.0,
                                    fi < adbl ? bf[fi] : 0.0);
                        }
                    }
                }
            }
            { const char *e = getenv("FFT45_CH");   /* monitor forcing      */
              if (e) { int v = atoi(e);
                       for (int c2 = 0; c2 < NCH45; ++c2)
                           if (c2 != v) chlive[c2] = 0; }
              int any = 0;
              for (int c2 = 0; c2 < NCH45; ++c2) any |= chlive[c2];
              if (!any) chlive[0] = 1; }
            /* ice_r7: msT 5 -> 12.  The q4 arms pay their pack/permute/
             * unpack (~0.7 ms/group) once per CALL, amortized over msT
             * steps here but over m=177 in production -- at msT=5 that is
             * a ~13% false penalty on q4, enough to flip a pick in a noisy
             * window.  msT=12 cuts it to ~5% while q4's measured true
             * margin is ~12-24%; setup grows ~0.15 s (unscored). */
            const int msT = 12, NRC = 5;
            for (int round = 0; round < NRC; ++round)
                for (int c2 = 0; c2 < NCH45; ++c2) {
                    if (!chlive[c2]) continue;
                    g_ch45[c2].fn(tin, cN, ch45_state(p, cN, tout), tout,
                                  2, nv, p->P);                     /* warm */
                    double t0c = now_s();
                    g_ch45[c2].fn(tin, cN, ch45_state(p, cN, tout), tout,
                                  msT, nv, p->P);
                    double tt = (now_s() - t0c) / ((double)msT * nv);
                    if (tt < tch[c2]) tch[c2] = tt;
                }
            int bc = -1;
            for (int c2 = 0; c2 < NCH45; ++c2)
                if (chlive[c2] && (bc < 0 || tch[c2] < tch[bc])) bc = c2;
            int pc = bc < 0 ? 0 : bc;
            if (bc >= 0)
                for (int c2 = 0; c2 < NCH45; ++c2)
                    if (chlive[c2] && tch[c2] <= tch[bc] * 1.03 &&
                        g_ch45[c2].rank < g_ch45[pc].rank) pc = c2;
            p->charm = pc;
            { size_t dl = strlen(g_desc);
              dl += (size_t)snprintf(g_desc + dl, sizeof g_desc - dl,
                       " | ch=%s %.1fus/step ms%d xf%d qc%d (",
                       g_ch45[pc].nm,
                       tch[pc] < 1e299 ? tch[pc] * 1e6 : -1.0,
                       FFT45_MS, FFT45_XF, FFT45_QC);
              for (int c2 = 0; c2 < NCH45 && dl < sizeof g_desc; ++c2)
                  dl += (size_t)snprintf(g_desc + dl, sizeof g_desc - dl,
                       "%s%s=%.1f", c2 ? " " : "", g_ch45[c2].nm,
                       chlive[c2] && tch[c2] < 1e299 ? tch[c2] * 1e6 : -1.0);
              if (dl < sizeof g_desc)
                  snprintf(g_desc + dl, sizeof g_desc - dl, ")"); }
#ifdef FFT45_LOUD
            if (1) {
#else
            if (getenv("FFT45_VERBOSE")) {
#endif
                for (int c2 = 0; c2 < NCH45; ++c2)
                    fprintf(stderr, "L45_pfa chain: %-6s %s %.1f us/step/vol\n",
                            g_ch45[c2].nm, chlive[c2] ? "ok " : "OUT",
                            chlive[c2] && tch[c2] < 1e299 ? tch[c2] * 1e6 : 0.0);
                fprintf(stderr, "L45_pfa chain: chose %s (nv=%d, MS=%d)\n",
                        g_ch45[pc].nm, nv, FFT45_MS);
                if (p->W) {  /* ice_r6: split-arm phase decomposition */
                    double *Wb  = ch45_state(p, cN, tout);
                    double *CSp = Wb + (size_t)nv * SPVST + 256;
                    double t1s = 1e300, t1m = 1e300, t2s = 1e300,
                           t2m = 1e300, tcv = 1e300, t1t = 1e300,
                           t1tm = 1e300;
                    conv45_in(cN, CSp);
                    conv45_in(tin, Wb);
                    for (int r2 = 0; r2 < 3; ++r2) {
                        double t0p = now_s(), tt;
                        for (int x = 0; x < L; ++x)
                            ch45s_p1_plane(Wb + (size_t)x * SPPL,
                                           CSp + (size_t)x * SPPL, p->P, 0);
                        tt = now_s() - t0p; if (tt < t1s) t1s = tt;
                        t0p = now_s();
                        for (int x = 0; x < L; ++x)
                            ch45s_p1_plane(Wb + (size_t)x * SPPL,
                                           CSp + (size_t)x * SPPL, p->P, 1);
                        tt = now_s() - t0p; if (tt < t1m) t1m = tt;
                        t0p = now_s();
                        ch45s_p2(Wb, 0);
                        tt = now_s() - t0p; if (tt < t2s) t2s = tt;
                        t0p = now_s();
                        ch45s_p2m(Wb, CSp, 0);
                        tt = now_s() - t0p; if (tt < t2m) t2m = tt;
                        t0p = now_s();
                        conv45_in(tin, Wb);
                        tt = now_s() - t0p; if (tt < tcv) tcv = tt;
                        /* vt (interleaved tr) twins in the same window */
                        t0p = now_s();
                        for (int x = 0; x < L; ++x)
                            ch45_p1t_plane(tout, tout, p->P, x);
                        tt = now_s() - t0p; if (tt < t1t) t1t = tt;
                        t0p = now_s();
                        for (int x = 0; x < L; ++x)
                            ch45_p1tm_plane(tout, cN, p->P, x);
                        tt = now_s() - t0p; if (tt < t1tm) t1tm = tt;
                    }
                    fprintf(stderr, "L45_pfa probe: p1s=%.1f p1sm=%.1f "
                            "p2s=%.1f p2sm=%.1f conv=%.1f | p1t=%.1f "
                            "p1tm=%.1f us/vol\n",
                            t1s * 1e6, t1m * 1e6, t2s * 1e6, t2m * 1e6,
                            tcv * 1e6, t1t * 1e6, t1tm * 1e6);
                }
                if (p->W && g_q4ok && nv >= 4) {
                    /* ice_r7: q4 pass decomposition, us per GROUP of 4
                     * volumes (divide by 4 for us/vol) */
                    double *Wb = ch45_state(p, cN, tout);
                    double *Q  = Wb + Q4OFF((long)nv);
                    double *C4 = Q + Q4VD + Q4PAD;
                    double tpk = 1e300, txp = 1e300, tzp = 1e300,
                           typ = 1e300, tym = 1e300, tum = 1e300;
                    for (int r2 = 0; r2 < 3; ++r2) {
                        double t0p, tt;
                        t0p = now_s(); q4_pack(cN, C4); q4_pack(tin, Q);
                        tt = now_s() - t0p; if (tt < tpk) tpk = tt;
                        t0p = now_s(); q4_xp0(Q);
                        tt = now_s() - t0p; if (tt < txp) txp = tt;
                        t0p = now_s();
                        for (int x = 0; x < L; ++x)
                            q4_zplane(Q + (size_t)x * Q4PLD);
                        tt = now_s() - t0p; if (tt < tzp) tzp = tt;
                        t0p = now_s();
                        for (int x = 0; x < L; ++x)
                            q4_yplane(Q + (size_t)x * Q4PLD);
                        tt = now_s() - t0p; if (tt < typ) typ = tt;
                        t0p = now_s();
                        for (int x = 0; x < L; ++x)
                            q4_yplanem(Q + (size_t)x * Q4PLD,
                                       C4 + (size_t)x * Q4PLD);
                        tt = now_s() - t0p; if (tt < tym) tym = tt;
                        t0p = now_s(); q4_unpackm(Q, C4, tout);
                        tt = now_s() - t0p; if (tt < tum) tum = tt;
                    }
                    {   /* store-direct z-DAG probe (bit-class candidate) */
                        double tzs = 1e300;
                        for (int r2 = 0; r2 < 3; ++r2) {
                            double t0p = now_s();
                            for (int x = 0; x < L; ++x)
                                q4_zplane_sd(Q + (size_t)x * Q4PLD);
                            double tt = now_s() - t0p; if (tt < tzs) tzs = tt;
                        }
                        fprintf(stderr, "L45_pfa probe q4 (us/group): "
                                "pack2=%.1f x=%.1f z=%.1f zsd=%.1f y=%.1f "
                                "ym=%.1f unpkm=%.1f\n",
                                tpk * 1e6, txp * 1e6, tzp * 1e6, tzs * 1e6,
                                typ * 1e6, tym * 1e6, tum * 1e6);
                    }
                    {   /* ice_r8: custody sweep decomposition, us/GROUP:
                         * qkQ = CQ slab transpose (once per call),
                         * qsP = one P sweep (ym+z+y per plane),
                         * qsQ = one Q sweep (xm+z+x per slab) */
                        double *CQ = C4 + Q4VD + QCPAD;
                        double tqk = 1e300, tsp = 1e300, tsq = 1e300;
                        for (int r2 = 0; r2 < 3; ++r2) {
                            double t0p, tt;
                            t0p = now_s(); qc_packQ(C4, CQ);
                            tt = now_s() - t0p; if (tt < tqk) tqk = tt;
                            t0p = now_s();
                            for (int x = 0; x < L; ++x) {
                                double *Px = Q + (size_t)x * Q4PLD;
                                q4_yplanem(Px, C4 + (size_t)x * Q4PLD);
                                q4_zplane(Px);
                                q4_yplane(Px);
                            }
                            tt = now_s() - t0p; if (tt < tsp) tsp = tt;
                            t0p = now_s();
                            for (int y = 0; y < L; ++y) {
                                double *Sy = Q + (size_t)y * Q4YS;
                                qc_xslabm(Sy, CQ + (size_t)y * Q4PLD);
                                if (y == L - 1) qc_zslab44(Sy);
                                else            qc_zslab(Sy);
                                qc_xslab(Sy);
                            }
                            tt = now_s() - t0p; if (tt < tsq) tsq = tt;
                        }
                        fprintf(stderr, "L45_pfa probe qc (us/group): "
                                "packQ=%.1f sweepP=%.1f sweepQ=%.1f\n",
                                tqk * 1e6, tsp * 1e6, tsq * 1e6);
                    }
                }
            }
        }
        free(rc); free(rrf); free(rbt); free(rrf2); free(rbt2);
    }
#else
    /* no AVX-512: the chain falls back to exec + scalar map; it needs one
     * batch-sized z buffer */
    if (posix_memalign(&p->rawW, 64,
                       (size_t)batch * VDBL * sizeof(double)) == 0)
        p->W = (double *)p->rawW;
#endif
    free(ri); free(ro); free(r0); free(r1);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->rawS); free(p->rawP); free(p->rawW); free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->fn((const double *)in, (double *)out, plan->batch, plan->S, plan->P);
}

/* ice_r4: the graded chain.  state <- (FFT(state)+c)/(1+|FFT(state)+c|),
 * m steps, final state in final_out.  Detected by the driver as a weak
 * symbol; entries lacking it pay a driver-side map pass per step. */
void fft3d_chain(fft3d_plan *plan, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    if (m < 1) {
        if ((const void *)x0 != (void *)final_out)
            memcpy(final_out, x0,
                   (size_t)plan->batch * VDBL * sizeof(double));
        return;
    }
#ifdef HAVE_PW4
    g_ch45[plan->charm].fn((const double *)x0, (const double *)c,
                           ch45_state(plan, (const double *)c,
                                      (double *)final_out),
                           (double *)final_out, m, plan->batch, plan->P);
#else
    {   /* correctness-only fallback (no AVX-512): exec + exact scalar map */
        const double *src = (const double *)x0;
        double *W = plan->W, *fo = (double *)final_out;
        const size_t npt = (size_t)plan->batch * (size_t)L * NPLANE;
        if (!W) {   /* alloc failed in create(); last-resort lazy alloc */
            if (posix_memalign(&plan->rawW, 64,
                               (size_t)plan->batch * VDBL * sizeof(double)))
                return;
            W = plan->W = (double *)plan->rawW;
        }
        for (int s = 0; s < m; ++s) {
            plan->fn(src, W, plan->batch, plan->S, plan->P);
            map45_ref(W, (const double *)c, fo, npt);
            src = fo;
        }
    }
#endif
}

#else /* ================ KERNEL TEMPLATE, PW = 2 or 4 ====================== */

#define vec   CAT(vec_pw,  PW)
#define uvec  CAT(uvec_pw, PW)
#define veci  CAT(veci_pw, PW)
#define FN(n) CAT(n, CAT(_pw, PW))

typedef double    vec  __attribute__((vector_size(PW * 16)));
typedef double    uvec __attribute__((vector_size(PW * 16), aligned(8)));
typedef long long veci __attribute__((vector_size(PW * 16)));

#ifdef __clang__
# define VSH(a,b,...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
# define VSH(a,b,...) __builtin_shuffle(a, b, (veci){__VA_ARGS__})
#endif

/* in/out accesses are potentially 16-byte aligned only (720 B rows); the
 * plane scratch and padded S are always 64B-aligned but share the macro
 * (an aligned address through an unaligned op costs nothing on these cores) */
#define LDU(p)    ((vec)*(const uvec *)(p))
#define STU(p, v) (*(uvec *)(p) = (uvec)(v))

#define NGRP  ((L + PW - 1) / PW)     /* lane groups incl. overlap tail: 23/12 */
#define NFULL (L / PW)                /* full groups: 22/11                    */
/* r10: phase 1's subpass group count.  Default = NFULL full groups + one
 * PW=1 tail line (dft45_line1); -DFFT45_OVERLAP_TAIL restores the r7-r9
 * clamped overlap group (NGRP groups, last one recomputing PW-1 lanes). */
#ifdef FFT45_OVERLAP_TAIL
# define NG1 NGRP
#else
# define NG1 NFULL
#endif

#if PW == 4
# define VSPLAT(a)  ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2,5,4,7,6)
# define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
/* one-complex masked load/store (lane 0), dead lanes ZERO (no denormal risk) */
# define LDT(p)    ((vec)_mm512_maskz_loadu_pd((__mmask8)0x03, (p)))
# define STT(p, v) _mm512_mask_storeu_pd((p), (__mmask8)0x03, (__m512d)(v))
/* gather/scatter one complex COLUMN (PW rows at `st` doubles apart) into/from
 * the lanes of one vector: the z-pass odd-column tail.  4 x 16 B accesses:
 * complex elements are 16B-aligned, never split a line. */
# define GCOL(dst, p, st) do {                                               \
    __m256d lo_ = _mm256_insertf128_pd(                                      \
        _mm256_castpd128_pd256(_mm_loadu_pd((p))),                           \
        _mm_loadu_pd((p) + (st)), 1);                                        \
    __m256d hi_ = _mm256_insertf128_pd(                                      \
        _mm256_castpd128_pd256(_mm_loadu_pd((p) + 2 * (st))),                \
        _mm_loadu_pd((p) + 3 * (st)), 1);                                    \
    (dst) = (vec)_mm512_insertf64x4(_mm512_castpd256_pd512(lo_), hi_, 1);    \
} while (0)
# define SCOL(src, p, st) do {                                               \
    __m512d v_ = (__m512d)(src);                                             \
    __m256d h_ = _mm512_extractf64x4_pd(v_, 1);                              \
    _mm_storeu_pd((p),            _mm512_castpd512_pd128(v_));               \
    _mm_storeu_pd((p) +     (st), _mm256_extractf128_pd(                     \
                                      _mm512_castpd512_pd256(v_), 1));       \
    _mm_storeu_pd((p) + 2 * (st), _mm256_castpd256_pd128(h_));               \
    _mm_storeu_pd((p) + 3 * (st), _mm256_extractf128_pd(h_, 1));             \
} while (0)
#elif PW == 2
# define VSPLAT(a)  ((vec){(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2)
# ifdef __FMA__
#  define VFMA(a,b,c)  ((vec)_mm256_fmadd_pd((__m256d)(a),(__m256d)(b),(__m256d)(c)))
#  define VFNMA(a,b,c) ((vec)_mm256_fnmadd_pd((__m256d)(a),(__m256d)(b),(__m256d)(c)))
# else
#  define VFMA(a,b,c)  ((a)*(b) + (c))
#  define VFNMA(a,b,c) ((c) - (a)*(b))
# endif
# ifdef __AVX__
#  define LDT(p)    ((vec)_mm256_maskload_pd((p),                            \
                         (__m256i){-1LL, -1LL, 0LL, 0LL}))
#  define STT(p, v) _mm256_maskstore_pd((p),                                 \
                         (__m256i){-1LL, -1LL, 0LL, 0LL}, (__m256d)(v))
#  define GCOL(dst, p, st)                                                   \
    ((dst) = (vec)_mm256_insertf128_pd(                                      \
        _mm256_castpd128_pd256(_mm_loadu_pd((p))),                           \
        _mm_loadu_pd((p) + (st)), 1))
#  define SCOL(src, p, st) do {                                              \
    _mm_storeu_pd((p),        _mm256_castpd256_pd128((__m256d)(src)));       \
    _mm_storeu_pd((p) + (st), _mm256_extractf128_pd((__m256d)(src), 1));     \
} while (0)
# else  /* portable fallbacks so a pre-AVX build still compiles */
#  define LDT(p)    ((vec){(p)[0], (p)[1], 0.0, 0.0})
#  define STT(p, v) do { (p)[0] = (v)[0]; (p)[1] = (v)[1]; } while (0)
#  define GCOL(dst, p, st)                                                   \
    ((dst) = (vec){(p)[0], (p)[1], (p)[(st)], (p)[(st) + 1]})
#  define SCOL(src, p, st) do {                                              \
    (p)[0] = (src)[0]; (p)[1] = (src)[1];                                    \
    (p)[(st)] = (src)[2]; (p)[(st) + 1] = (src)[3];                          \
} while (0)
# endif
#endif

/* r11: aligned z-load recombination (the "zal" exec variants).  The z-pass
 * in-loads at the natural 720 B row stride are 64 B accesses of which ~75%
 * split a cache line (the odd-L toll, r6).  Byte phase of row (yb+j) of
 * plane x of volume b is 16*(b + x + yb + j) mod 64, and yb = PW*yg with
 * PW = 4 makes yb vanish mod 4 -- so within EVERY y-group the four rows
 * carry the fixed shift pattern s_j = 2*((zc + j) & 3) doubles, where
 * zc = ((uintptr)plane >> 4) & 3 is one per-plane value.  Each shifted row
 * is read as a rolling stream of ALIGNED lines recombined with one valignq
 * (port 5, which has ~2.5x headroom over port 0 here); granule 0 of each
 * row keeps one plain unaligned load so the aligned container never reads
 * before the row (no underflow), and the last rolling line reads at most
 * 32 B into the NEXT row of the same plane (rows 0..43 only -- row 44 is
 * the PW=1 tail line -- so no access ever leaves the buffer).  Split
 * z-loads per volume: ~16k -> ~1.1k.  Node-only mechanism by construction
 * (wallaby hides splits); shipped as tuner candidates pf0a/pf3a so the
 * node's tournament prices the split-load class directly, per this file's
 * r10 "Next" item.  Not valid under FFT45_OVERLAP_TAIL (its clamped last
 * group has yb = 41, which breaks yb == 0 mod 4), so the candidates are
 * compiled out there. */
#if PW == 4
# define VALIGNQ(nx, cu, s) \
    ((vec)_mm512_alignr_epi64((__m512i)(nx), (__m512i)(cu), (s)))
# define ZLA(C) do {                                                         \
    const double *ar_[PW]; vec cu_[PW];                                      \
    _Pragma("GCC unroll 4")                                                  \
    for (int j = 0; j < PW; ++j) {                                           \
        const int s_ = 2 * (((C) + j) & 3);                                  \
        ar_[j] = rows + (size_t)j * (2 * L) - s_;                            \
        if (s_) cu_[j] = LDU(ar_[j] + 8);         /* line 1: aligned */      \
    }                                                                        \
    {   /* granule 0: plain loads, so the aligned stream never underflows */ \
        vec r_[PW];                                                          \
        _Pragma("GCC unroll 4")                                              \
        for (int j = 0; j < PW; ++j)                                         \
            r_[j] = LDU(rows + (size_t)j * (2 * L));                         \
        TRNC(r_, &Zv[0]);                                                    \
    }                                                                        \
    _Pragma("GCC unroll 22")                                                 \
    for (int zg_ = 1; zg_ < NFULL; ++zg_) {                                  \
        vec r_[PW];                                                          \
        _Pragma("GCC unroll 4")                                              \
        for (int j = 0; j < PW; ++j) {                                       \
            const int s_ = 2 * (((C) + j) & 3);                              \
            if (s_ == 0) {                                                   \
                r_[j] = LDU(rows + (size_t)j * (2 * L) + 8 * zg_);           \
            } else {                                                         \
                vec nx_ = LDU(ar_[j] + 8 * (zg_ + 1));                       \
                switch (s_) {                     /* folds after unroll */   \
                case 2:  r_[j] = VALIGNQ(nx_, cu_[j], 2); break;             \
                case 4:  r_[j] = VALIGNQ(nx_, cu_[j], 4); break;             \
                default: r_[j] = VALIGNQ(nx_, cu_[j], 6); break;             \
                }                                                            \
                cu_[j] = nx_;                                                \
            }                                                                \
        }                                                                    \
        TRNC(r_, &Zv[zg_ * PW]);                                             \
    }                                                                        \
} while (0)
#endif

/* PW x PW transpose of 128-bit complex granules (involution) */
#if PW == 4
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,8,9,4,5,12,13);                        \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,10,11,6,7,14,15);                      \
    vec u2_ = VSH((r)[2], (r)[3], 0,1,8,9,4,5,12,13);                        \
    vec u3_ = VSH((r)[2], (r)[3], 2,3,10,11,6,7,14,15);                      \
    (c)[0] = VSH(u0_, u2_, 0,1,2,3,8,9,10,11);                               \
    (c)[2] = VSH(u0_, u2_, 4,5,6,7,12,13,14,15);                             \
    (c)[1] = VSH(u1_, u3_, 0,1,2,3,8,9,10,11);                               \
    (c)[3] = VSH(u1_, u3_, 4,5,6,7,12,13,14,15);                             \
} while (0)
#elif PW == 2
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,4,5);                                  \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,6,7);                                  \
    (c)[0] = u0_; (c)[1] = u1_;                                              \
} while (0)
#endif

/* ---- ice_r5: transpose-free (tr) phase-1 subpasses -----------------------
 * BORROWED: the tr mechanism is L36_pfa ice_r2's (on bare-metal ICX a masked
 * vbroadcastf64x2 with a memory operand is ONE load-port uop, zero port 5,
 * so build every lane-transposed vector as 4 masked 128-bit broadcasts and
 * delete BOTH TRNC passes), proven to transfer to L=45 by L45_mixedradix
 * ice_r2 (-9.6..-12%).  z-subpass: gather lanes=y straight from the 4 y-rows
 * of the group; codelet output goes UNtransposed to a slot-major scratch as
 * full-width 64B-aligned stores (slot s = ygroup*45 + kz holds one lanes=y
 * vector at slt + s*8 doubles; the odd 45th row's xmm tail line writes lane
 * 0 of slots 495..539 at a 64 B stride).  y-subpass: re-gather lanes=kz --
 * element (y, kz) sits at slot (y>>2)*45 + kz, 16 B granule (y&3), ONE
 * formula valid for ALL 45 y including the tail row, so the y-side load
 * path has no special case at all.  Deleted per y-group: 176 TRNC shuffles
 * + GCOL/SCOL inserts (~90k port-5 uops/volume); added: ~360 16 B load uops
 * per group against load ports that were near idle (2x64B/cyc).
 * Bit-identity: gathers deliver exactly the values the TRNC path delivered,
 * the z-site keeps PFA45R and the y/tail sites keep PFA45, so tr twins are
 * bitwise interchangeable with the classic subpasses -- the cross-arm
 * memcmp gate and the cross-process repeatability check stay safe with
 * both families in one racing pool.
 *
 * The mapped (mp) z-subpass stages the ice_r4 lazy map at ROW-GROUP
 * granularity with a ONE-GROUP software pipeline (BORROWED: L36_pencilfused
 * ice_r5's maprows shape): group yg+1's 4 rows (2.88 KB) are mapped into an
 * L1-resident ping-pong buffer immediately BEFORE the FFT of group yg, so
 * the map's rsqrt/divide dep chains issue under a full codelet (~500 uops)
 * of independent port work and the map stores sit a whole codelet away from
 * their own 16 B broadcast reloads.  Their measured traps are respected:
 * fusing the map INTO the broadcast sites spilled hard (+57% for them, the
 * bcstm trap), and staging it as a standalone tight loop starves the
 * divider -- the one-group pipeline is the shape that avoids both. */
#if PW == 4
# ifdef __AVX512DQ__
#  define G4B(p, str) ((vec)_mm512_mask_broadcast_f64x2(                     \
    _mm512_mask_broadcast_f64x2(                                             \
        _mm512_mask_broadcast_f64x2(                                         \
            _mm512_castpd128_pd512(_mm_loadu_pd((p))),                       \
            (__mmask8)0x0C, _mm_loadu_pd((p) + (str))),                      \
        (__mmask8)0x30, _mm_loadu_pd((p) + 2 * (str))),                      \
    (__mmask8)0xC0, _mm_loadu_pd((p) + 3 * (str))))
# else   /* no DQ: same values through the AVX insert gather (correctness) */
#  define G4B(p, str) ({ vec g4_; GCOL(g4_, (p), (str)); g4_; })
# endif

static inline __attribute__((always_inline))
void FN(zsubt)(const double *restrict px, double *restrict slt,
               const int mp, const double *restrict cx)
{
    double mb[2][368] __attribute__((aligned(64)));  /* row-group ping-pong */
    double tb[92]     __attribute__((aligned(64)));  /* mapped odd 45th row */
    if (mp) map45_span(px, cx, mb[0], 4 * L);        /* prologue: group 0   */
    _Pragma("GCC unroll 1")
    for (int yg = 0; yg < NFULL; ++yg) {
        const double *rows;
        if (mp) {
            if (yg + 1 < NFULL)                      /* pipeline: map yg+1  */
                map45_span(px + (size_t)(yg + 1) * (8 * L),
                           cx + (size_t)(yg + 1) * (8 * L),
                           mb[(yg + 1) & 1], 4 * L);
            else                                     /* last: map tail row  */
                map45_span(px + (size_t)(L - 1) * (2 * L),
                           cx + (size_t)(L - 1) * (2 * L), tb, L);
            rows = mb[yg & 1];
        } else {
            rows = px + (size_t)yg * (8 * L);
        }
        double *srow = slt + (size_t)yg * (8 * L);   /* 45 slots x 8 dbl    */
        __asm__("" : "+r"(rows), "+r"(srow));
#define LDZT(n)    G4B(rows + 2 * (n), 2 * L)
#define STZT(k, v) STU(srow + (size_t)(k) * 8, (v))
        PFA45R(LDZT, STZT);     /* z-site keeps the DFT9-first bit order    */
#undef LDZT
#undef STZT
    }
    if (mp)
        dft45_line1(tb, 2, slt + (size_t)NFULL * (8 * L), 8);
    else
        dft45_line1(px + (size_t)(L - 1) * (2 * L), 2,
                    slt + (size_t)NFULL * (8 * L), 8);
}

static inline __attribute__((always_inline))
void FN(ysubt)(const double *restrict slt, double *mx, const long mrow)
{
    _Pragma("GCC unroll 1")
    for (int zg = 0; zg < NFULL; ++zg) {
        const double *pz   = slt + (size_t)zg * (PW * 8);
        double       *mcol = mx  + (size_t)zg * (PW * 2);
        __asm__("" : "+r"(pz), "+r"(mcol));
#define LDYT(n)    G4B(pz + (size_t)((n) >> 2) * (8 * L) + 2 * ((n) & 3), 8)
#define STYT(k, v) STU(mcol + (size_t)(k) * mrow, (v))
        PFA45(LDYT, STYT);
#undef LDYT
#undef STYT
    }
    /* odd 45th kz column straight off the slot layout */
    dft45_line1t(slt, mx + 2 * (L - 1), mrow);
}

/* ice_r6 "xf": mapped y-subpass for the X-FIRST chain family (BORROWED:
 * L45_mixedradix ice_r4's CXF=1 map placement + their ice_r5 MPAIR store
 * pairing).  The y store is the LAST write of an x-first step, so the map
 * fuses here: consecutive PFA45 outputs pair through a one-register stash
 * (pair grouping cannot change per-point bits -- map45_pair's documented
 * property), giving one ladder + ONE divide per 8 points hanging off the
 * END of the store graph with a full codelet of independent work above it,
 * and c loads that stay PLANE-LOCAL (the r6 probe indicted the staged
 * span at +72..90 us/vol for both families, and map-at-p2-stores at +139
 * -- its c loads scatter over 45 pages).  The stash toggle is straight-
 * line constant-folded (k literals under full unroll); the odd 45th ST of
 * each call flushes through map45_one, the odd kz column maps per point
 * through map45_1x -- all the same per-point bit class. */
static inline __attribute__((always_inline))
void FN(ysubtm)(const double *restrict slt, double *mx, const long mrow,
                const double *restrict cx)
{
    _Pragma("GCC unroll 1")
    for (int zg = 0; zg < NFULL; ++zg) {
        const double *pz   = slt + (size_t)zg * (PW * 8);
        double       *mcol = mx  + (size_t)zg * (PW * 2);
        const double *ccol = cx  + (size_t)zg * (PW * 2);
        __asm__("" : "+r"(pz), "+r"(mcol), "+r"(ccol));
        vec  stv_ = VSPLAT(0.0);
        long stk_ = -1;
#define LDYT(n)    G4B(pz + (size_t)((n) >> 2) * (8 * L) + 2 * ((n) & 3), 8)
#define STYM(k, v) do {                                                      \
    if (stk_ < 0) { stv_ = (v); stk_ = (k); }                                \
    else {                                                                   \
        __m512d w0_ = (__m512d)(stv_                                         \
                        + LDU(ccol + (size_t)stk_ * mrow));                  \
        __m512d w1_ = (__m512d)((v)                                          \
                        + LDU(ccol + (size_t)(k) * mrow));                   \
        map45_pair(&w0_, &w1_);                                              \
        STU(mcol + (size_t)stk_ * mrow, (vec)w0_);                          \
        STU(mcol + (size_t)(k)  * mrow, (vec)w1_);                          \
        stk_ = -1;                                                           \
    }                                                                        \
} while (0)
        PFA45(LDYT, STYM);
        if (stk_ >= 0) {                     /* 45 is odd: flush the single */
            __m512d w_ = (__m512d)(stv_ + LDU(ccol + (size_t)stk_ * mrow));
            STU(mcol + (size_t)stk_ * mrow, (vec)map45_one(w_));
        }
#undef LDYT
#undef STYM
    }
    {   /* odd kz column: raw line into a stack row, gather the c column
         * beside it, ONE paired span (6 divides for 45 points instead of
         * 45 scalar ladders; per-point bits unchanged -- pair packing is
         * value-transparent), scatter back to the column. */
        double tb_[2 * L] __attribute__((aligned(64)));
        double cb_[2 * L] __attribute__((aligned(64)));
        double ob_[2 * L] __attribute__((aligned(64)));
        const double *cc = cx + 2 * (L - 1);
        double       *mm = mx + 2 * (L - 1);
        _Pragma("GCC unroll 45")
        for (int k = 0; k < L; ++k)
            _mm_store_pd(cb_ + 2 * k, _mm_loadu_pd(cc + (size_t)k * mrow));
        dft45_line1t(slt, tb_, 2);
        map45_span(tb_, cb_, ob_, L);
        _Pragma("GCC unroll 45")
        for (int k = 0; k < L; ++k)
            _mm_storeu_pd(mm + (size_t)k * mrow, _mm_load_pd(ob_ + 2 * k));
    }
}
#endif /* PW == 4 */

/* paced-prefetch step: cover one x-plane (PLND doubles = 507 lines) in
 * 2*NG1 steps -> 24 lines/step at PW=4, 13 at PW=2 (r10: NG1 groups/pass) */
#define PFL    ((PLND + 16 * NG1 - 1) / (16 * NG1))
#define PFSTEP (8 * PFL)
#define PFIN(p) do {                                                          \
    _Pragma("GCC unroll 24")                                                  \
    for (int q_ = 0; q_ < PFL; ++q_)                                          \
        __builtin_prefetch((p) + 8 * q_, 0, FFT45_PFH);                       \
} while (0)
/* ip-pf3: write-intent (prefetchw) cursor over the mid-plane stores */
#define PFWMID(p) do {                                                        \
    _Pragma("GCC unroll 24")                                                  \
    for (int q_ = 0; q_ < PFL; ++q_)                                          \
        __builtin_prefetch((p) + 8 * q_, 1, 3);                               \
} while (0)
/* phase 1, ONE x-plane: z transform (transposed lanes) into plane scratch,
 * then y transform into mid[x][ky][kz].  pfr/pfw are compile-time constants
 * at every instantiation site (always_inline + const), so the pf0 execs
 * carry NO prefetch code and no cursor arithmetic.  mrow/mpln are the mid
 * row/plane strides in doubles: (90, 4050) inplace, (104, 4680) scratchp --
 * also compile-time, so every mid access folds to base + literal. */
/* ice_r2: phase 1 is split into zsub/ysub so exec variants can mix vector
 * widths per subpass (the hz hybrid: pw2 z-subpass -- whose ymm FMAs issue
 * on p0/p1, leaving p5 wholly to the transpose shuffles -- feeding a pw4
 * y-subpass through the width-agnostic pl layout).  Bodies are unchanged;
 * phase1_plane composes them exactly as before.  The explicit "GCC unroll
 * 1" pins the outer group loops rolled: gcc 11 already keeps them rolled
 * (verified byte-identical with/without), the pragma just makes that a
 * contract rather than an accident -- each loop body (~1k uops, one codelet
 * + transposes) must keep fitting the DSB on this two-pipe machine. */
static inline __attribute__((always_inline))
void FN(zsub)(const double *restrict px, double *restrict pld,
              const double *pfc, double *pwc, const int pfr, const int pfw,
              const int zal, const int mp, const double *restrict cx)
{
#if PW == 4
    /* per-plane alignment phase for the zal path (16-byte units mod 4) */
    const int zc = zal ? (int)(((uintptr_t)px >> 4) & 3) : 0;
#else
    (void)zal; (void)mp; (void)cx;
#endif

    _Pragma("GCC unroll 1")
    for (int yg = 0; yg < NG1; ++yg) {
        const int yb = (yg == NGRP - 1) ? (L - PW) : yg * PW;
        /* ONE runtime base per block; every access below is base + a
         * compile-time constant (the r7 single-base fix: folding yb into
         * each address made gcc spill a 48-entry offset table, -14%). */
        const double *rows = px  + (size_t)yb * (2 * L);
        double       *prow = pld + (size_t)yb * (2 * PPITCH);
#if PW == 4
        const double *crows = mp ? cx + (size_t)yb * (2 * L) : 0;
#endif
        if (pfr) { PFIN(pfc);   pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
        vec Zv[L], Wv[L];
#if PW == 4
        if (mp) {
            /* ice_r4 lazy map: the buffer holds RAW z from the previous
             * step; map(z + c) on load, both streams sequential.  Rows pair
             * (0,1)/(2,3) -> one packed ladder + ONE divide per 8 points. */
            _Pragma("GCC unroll 22")
            for (int zg = 0; zg < NFULL; ++zg) {
                const int zb = zg * PW;
                vec r_[PW];
                _Pragma("GCC unroll 4")
                for (int j = 0; j < PW; ++j)
                    r_[j] = LDU(rows  + (size_t)j * (2 * L) + 2 * zb)
                          + LDU(crows + (size_t)j * (2 * L) + 2 * zb);
                map45_pair((__m512d *)&r_[0], (__m512d *)&r_[1]);
                map45_pair((__m512d *)&r_[2], (__m512d *)&r_[3]);
                TRNC(r_, &Zv[zb]);
            }
        } else if (zal) {
            /* aligned rolling loads + valignq; zc is constant across the
             * plane, so this 4-way branch predicts perfectly */
            switch (zc) {
            case 0:  ZLA(0); break;
            case 1:  ZLA(1); break;
            case 2:  ZLA(2); break;
            default: ZLA(3); break;
            }
        } else
#endif
        {
        _Pragma("GCC unroll 22")
        for (int zg = 0; zg < NFULL; ++zg) {
            const int zb = zg * PW;
            vec r_[PW];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                r_[j] = LDU(rows + (size_t)j * (2 * L) + 2 * zb);
            TRNC(r_, &Zv[zb]);
        }
        }
        /* odd 45th column via 16 B column gather (no overlapped split loads) */
#if PW == 4
        if (mp) {
            vec zg_, cg_;
            GCOL(zg_, rows  + 2 * (L - 1), 2 * L);
            GCOL(cg_, crows + 2 * (L - 1), 2 * L);
            zg_ = zg_ + cg_;
            Zv[L - 1] = (vec)map45_one((__m512d)zg_);
        } else
#endif
        GCOL(Zv[L - 1], rows + 2 * (L - 1), 2 * L);
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
        PFA45R(LD1, ST1);        /* array-ST site: DFT9-first order wins */
#undef LD1
#undef ST1
        _Pragma("GCC unroll 22")
        for (int zg = 0; zg < NFULL; ++zg) {
            const int zb = zg * PW;
            vec r_[PW];
            TRNC(&Wv[zb], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                STU(prow + (size_t)j * (2 * PPITCH) + 2 * zb, r_[j]);
        }
        /* odd column scatter: lane j of Wv[44] -> row yb+j, 16 B stores */
        SCOL(Wv[L - 1], prow + 2 * (L - 1), 2 * PPITCH);
    }
#ifndef FFT45_OVERLAP_TAIL
    /* odd 45th row: one PW=1 line, in row 44 -> pl row 44 (contiguous in,
     * no transposes, 16 B accesses that never split a line) */
#if PW == 4
    if (mp) {
        /* pre-map the contiguous 45-complex row into a 720 B stack line
         * (same per-point bits as the vector path), then the plain tail */
        double tb_[2 * L] __attribute__((aligned(64)));
        map45_span(px + (size_t)(L - 1) * (2 * L),
                   cx + (size_t)(L - 1) * (2 * L), tb_, L);
        dft45_line1(tb_, 2, pld + (size_t)(L - 1) * (2 * PPITCH), 2);
    } else
#endif
    dft45_line1(px + (size_t)(L - 1) * (2 * L), 2,
                pld + (size_t)(L - 1) * (2 * PPITCH), 2);
#endif
}

static inline __attribute__((always_inline))
void FN(ysub)(const double *restrict pld, double *restrict mx,
              const double *pfc, double *pwc, const int pfr, const int pfw,
              const long mrow)
{
    _Pragma("GCC unroll 1")
    for (int zg = 0; zg < NG1; ++zg) {
        const int zb = (zg == NGRP - 1) ? (L - PW) : zg * PW;
        const double *pcol = pld + 2 * zb;      /* single runtime base, again */
        double       *mcol = mx  + 2 * zb;
        /* opaque-base barrier: without it gcc re-associates every load
         * address as pld + (zb*16) + n*832, hoists the 45 loop-invariant
         * (pld + n*832) leas, SPILLS them, and reloads one per vector load
         * -- the r6 offset-table pathology relocated into the y-subloop
         * (measured here: 48 leas + 37 GPR spills per exec, -163 instr and
         * all spill-reload serialization gone with the barrier). */
        __asm__("" : "+r"(pcol), "+r"(mcol));
        if (pfr) { PFIN(pfc);   pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
#define LD2(n)    LDU(pcol + (size_t)(n) * (2 * PPITCH))
#define ST2(k, v) STU(mcol + (size_t)(k) * mrow, (v))
        PFA45(LD2, ST2);
#undef LD2
#undef ST2
    }
#ifndef FFT45_OVERLAP_TAIL
    /* odd 45th column: one PW=1 line, pl column 44 -> mid column 44 */
    dft45_line1(pld + 2 * (L - 1), 2 * PPITCH, mx + 2 * (L - 1), mrow);
#endif
}

/* NOTE (ice_r4): in/mid deliberately NOT restrict -- the chain runs phase 1
 * in place (in == mid), which is safe because zsub drains the whole plane
 * into pld before ysub's first store, but would be UB under restrict. */
static inline __attribute__((always_inline))
void FN(phase1_plane)(const double *in, double *mid,
                      double *restrict pld, int x, const int pfr,
                      const int pfw, const int zal, const long mrow,
                      const long mpln, const int mp, const double *cvol)
{
    const double *pfc = pfr ? in  + FFT45_PFD  + (size_t)x * PLND : 0;
    double       *pwc = pfw ? mid + FFT45_PFWD + (size_t)x * mpln : 0;
    const double *px  = in  + (size_t)x * PLND;
    double       *mx  = mid + (size_t)x * mpln;
    FN(zsub)(px, pld, pfc, pwc, pfr, pfw, zal, mp,
             mp ? cvol + (size_t)x * PLND : 0);
    FN(ysub)(pld, mx, pfc ? pfc + (size_t)NG1 * PFSTEP : 0,
             pwc ? pwc + (size_t)NG1 * PFSTEP : 0, pfr, pfw, mrow);
}

/* cpy == 1: the y-subpass targets the plane image at pld + IMGOFF (an exact
 * out-plane layout, 2*L-double rows) and one ERMS rep movsb per x-plane
 * moves it to mid = out.  The image and the plane-scratch rows are disjoint
 * slices of the same P allocation.  pfw is forced 0 under cpy (cpy replaces
 * the store-side mechanism). */
static inline __attribute__((always_inline))
void FN(phase1)(const double *restrict in, double *restrict mid,
                double *restrict pld, const int pfr, const int pfw,
                const int zal, const int cpy, const long mrow,
                const long mpln)
{
    double *img = pld + IMGOFF;
    for (int x = 0; x < L; ++x) {
        if (cpy) {
            FN(phase1_plane)(in, img, pld, x, pfr, 0, zal, 2 * L, 0, 0, 0);
            plane_copy(mid + (size_t)x * PLND, img,
                       (size_t)PLND * sizeof(double));
        } else {
            FN(phase1_plane)(in, mid, pld, x, pfr, pfw, zal, mrow, mpln, 0, 0);
        }
    }
}

/* phase-2 read prefetch: 45 sequential streams at PLND-double stride is more
 * than the L2 streamer tracks (a runtime pf level in r7; compile-time now) */
#define PF45(s_) do {                                                        \
    _Pragma("GCC unroll 45")                                                 \
    for (int n_ = 0; n_ < 45; ++n_)                                          \
        __builtin_prefetch((s_) + (size_t)n_ * PLND + 8, 0, 3);              \
} while (0)
/* sp-pfs: write-intent prefetch of the 45 cold out-streams one tile ahead */
#define PFW45(d_) do {                                                       \
    _Pragma("GCC unroll 45")                                                 \
    for (int n_ = 0; n_ < 45; ++n_)                                          \
        __builtin_prefetch((d_) + (size_t)n_ * PLND + 8, 1, 3);              \
} while (0)
/* per-tile pre-coverage of the NEXT volume's input (first ~63 KB) */
#define PFNX() do { if (pfn && pn_) {                                         \
    _Pragma("GCC unroll 4")                                                   \
    for (int q_ = 0; q_ < FFT45_PFN; ++q_)                                    \
        __builtin_prefetch(pn_ + 8 * q_, 0, FFT45_PFH);                       \
    pn_ += 8 * FFT45_PFN; } } while (0)

/* phase 2, INPLACE: x transform in place in `out` (the codelet reads all 45
 * inputs before its first store).  Tiled over the FLAT (y,z) index:
 * 2025 = 506*4 + 1 (PW=4) leaves ONE masked tail call per volume.
 * (Flat tiling borrowed from L45_mixedradix r7.)
 * ice_r2: chunked form [t0, t1) so the il exec variants can interleave
 * phase-2 tile bands of volume b-1 between phase-1 planes of volume b; the
 * masked tail call ships with the final chunk. */
static inline __attribute__((always_inline))
void FN(phase2_chunk)(const double *mid, double *out, const double *pnext,
                      const int pfx, const int pfn, const int t0,
                      const int t1)
{
    const double *pn_ = pnext;
    _Pragma("GCC unroll 1")
    for (int t = t0; t < t1; ++t) {
        const size_t o = (size_t)t * (PW * 2);
        const double *s_ = mid + o;
        double       *d_ = out + o;
        if (pfx) PF45(s_);
        PFNX();
#define LD3(n)    LDU(s_ + (size_t)(n) * PLND)
#define ST3(k, v) STU(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD3, ST3);
#undef LD3
#undef ST3
    }
    if (t1 == NPLANE / PW) {
        /* tail: flat line 2024 = (y,z) = (44,44); lane 0 live, dead lanes 0 */
        const double *s_ = mid + (size_t)(NPLANE - 1) * 2;
        double       *d_ = out + (size_t)(NPLANE - 1) * 2;
#define LD4(n)    LDT(s_ + (size_t)(n) * PLND)
#define ST4(k, v) STT(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD4, ST4);
#undef LD4
#undef ST4
    }
}

static inline __attribute__((always_inline))
void FN(phase2)(const double *mid, double *out, const double *pnext,
                const int pfx, const int pfn)
{
    FN(phase2_chunk)(mid, out, pnext, pfx, pfn, 0, NPLANE / PW);
}

/* phase 2, SCRATCHP: x transform S -> out, out of place.  S rows are padded
 * (SPITCH complex), so the flat (y,z) index does not map affinely to out and
 * tiling is per y-row: NFULL full tiles + ONE OVERLAPPING tile (starts at
 * z = 45-PW; recompute is idempotent because the pass is out of place).
 * All S loads are 64B-aligned; only the out stores pay the odd-L split toll,
 * and pfw covers their RFO in the streaming variant. */
static inline __attribute__((always_inline))
void FN(phase2s)(const double *S, double *out, const double *pnext,
                 const int pfw, const int pfn)
{
    const double *pn_ = pnext;
    for (int y = 0; y < L; ++y) {
        const double *sy = S   + (size_t)y * SROWD;
        double       *oy = out + (size_t)y * (2 * L);
        _Pragma("GCC unroll 1")
        for (int tg = 0; tg < NGRP; ++tg) {
            const int zb = (tg == NGRP - 1) ? (L - PW) : tg * PW;
            const double *s_ = sy + 2 * zb;
            double       *d_ = oy + 2 * zb;
            if (pfw) PFW45(d_);
            PFNX();
#define LD5(n)    LDU(s_ + (size_t)(n) * SPLND)
#define ST5(k, v) STU(d_ + (size_t)(k) * PLND, (v))
            PFA45(LD5, ST5);
#undef LD5
#undef ST5
        }
    }
}

/* ---- exec variants: every pf/mode/stride argument below is a literal, so
 * const-propagation through the always_inline bodies specializes each one
 * completely (the L45_mixedradix structure). ------------------------------ */

static void FN(x_ip0)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        FN(phase1)(i, o, P, 0, 0, 0, 0, 2 * L, PLND);
        FN(phase2)(o, o, 0, 0, 0);
    }
}

#if PW == 4      /* the pw2 tournament pool only carries ip0/ip3/sp0/sps */
static void FN(x_ip1)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        FN(phase1)(i, o, P, 0, 0, 0, 0, 2 * L, PLND);
        FN(phase2)(o, o, 0, 1, 0);
    }
}

static void FN(x_ip2)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        const double *nx = (b + 1 < nvol) ? i + VDBL : 0;
        FN(phase1)(i, o, P, 1, 0, 0, 0, 2 * L, PLND);
        FN(phase2)(o, o, nx, 1, 1);
    }
}

#ifndef FFT45_OVERLAP_TAIL
/* r11: pf0 + aligned z-loads (the node's B=1/B=2 incumbent plus zal) */
static void FN(x_ip0a)(const double *in, double *out, long nvol,
                       double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        FN(phase1)(i, o, P, 0, 0, 1, 0, 2 * L, PLND);
        FN(phase2)(o, o, 0, 0, 0);
    }
}

/* r11: pf3 + aligned z-loads (the node's B=16 incumbent plus zal) */
static void FN(x_ip3a)(const double *in, double *out, long nvol,
                       double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        const double *nx = (b + 1 < nvol) ? i + VDBL : 0;
        FN(phase1)(i, o, P, 1, 1, 1, 0, 2 * L, PLND);
        FN(phase2)(o, o, nx, 1, 1);
    }
}

/* ice_r2: cpy + aligned z-loads */
static void FN(x_cp0a)(const double *in, double *out, long nvol,
                       double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        FN(phase1)(i, o, P, 0, 0, 1, 1, 2 * L, PLND);
        FN(phase2)(o, o, 0, 0, 0);
    }
}
#endif /* !FFT45_OVERLAP_TAIL */

/* ice_r5: transpose-free phase 1 (broadcast-gather z + y subpasses through
 * the slot-major scratch), plain phase 2.  Bit-identical to ip0. */
static void FN(x_tr0)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        for (int x = 0; x < L; ++x) {
            FN(zsubt)(i + (size_t)x * PLND, P, 0, 0);
            FN(ysubt)(P, o + (size_t)x * PLND, 2 * L);
        }
        FN(phase2)(o, o, 0, 0, 0);
    }
}

/* ice_r2 phase-isolation probes (verbose/tuner-diagnostic only, never in the
 * candidate table): p1 = phase 1 alone, p2 = phase 2 alone in place. */
static void FN(x_p1o)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b)
        FN(phase1)(in + (size_t)b * VDBL, out + (size_t)b * VDBL, P,
                   0, 0, 0, 0, 2 * L, PLND);
}


static void FN(x_p2o)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)in; (void)S; (void)P;
    for (long b = 0; b < nvol; ++b) {
        double *o = out + (size_t)b * VDBL;
        FN(phase2)(o, o, 0, 0, 0);
    }
}

/* p1i: phase 1 with the y-stores going to the L1-hot image (no out traffic,
 * no copy) -- isolates the store-side stall.  p1h: phase 1 reading x-plane 0
 * every time (L1-hot input; the pointer un-advance is a probe-only hack) --
 * isolates the read-side stall.  p1c: both hot = compute + pl round-trip. */
static void FN(x_p1i)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)out; (void)S;
    double *img = P + IMGOFF;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in + (size_t)b * VDBL;
        for (int x = 0; x < L; ++x)
            FN(phase1_plane)(i, img, P, x, 0, 0, 0, 2 * L, 0, 0, 0);
    }
}

static void FN(x_p1h)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        double *o = out + (size_t)b * VDBL;
        for (int x = 0; x < L; ++x)
            FN(phase1_plane)(in - (size_t)x * PLND, o, P, x,
                             0, 0, 0, 2 * L, PLND, 0, 0);
    }
}

static void FN(x_p1c)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)out; (void)S;
    double *img = P + IMGOFF;
    for (long b = 0; b < nvol; ++b)
        for (int x = 0; x < L; ++x)
            FN(phase1_plane)(in - (size_t)x * PLND, img, P, x,
                             0, 0, 0, 2 * L, 0, 0, 0);
    (void)nvol;
}

/* ice_r2 IL variants (cross-volume software pipelining, mine -- untried in
 * either L=45 lineage): between phase-1 planes of volume b, run a band of
 * phase-2 tiles of volume b-1.  Phase 1 is compute-heavy and store-stalled;
 * phase 2 is latency-bound on 45 read streams the L2 streamer cannot track
 * -- interleaved, the two stall pools overlap instead of serializing.
 * Volume b-1's phase 1 is complete before any of its phase-2 tiles run, and
 * the two volumes touch disjoint buffers, so correctness is order-trivial;
 * the last volume's phase 2 drains after the loop.  nvol==1 degenerates to
 * the sequential form. */
static inline __attribute__((always_inline))
void FN(x_il_body)(const double *in, double *out, long nvol,
                   double *P, const int zal)
{
    double *oprev = 0;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        for (int x = 0; x < L; ++x) {
            FN(phase1_plane)(i, o, P, x, 0, 0, zal, 2 * L, PLND, 0, 0);
            if (oprev) {
                const int t0 = ((NPLANE / PW) * x)       / L;
                const int t1 = ((NPLANE / PW) * (x + 1)) / L;
                FN(phase2_chunk)(oprev, oprev, 0, 0, 0, t0, t1);
            }
        }
        oprev = o;
    }
    FN(phase2)(oprev, oprev, 0, 0, 0);
}

static void FN(x_il0)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    FN(x_il_body)(in, out, nvol, P, 0);
}

#ifndef FFT45_OVERLAP_TAIL
static void FN(x_il0a)(const double *in, double *out, long nvol,
                       double *S, double *P)
{
    (void)S;
    FN(x_il_body)(in, out, nvol, P, 1);
}
#endif

/* ice_r2 HZ hybrid (mine): pw2 (ymm) z-subpass + pw4 (zmm) y-subpass,
 * composed through the width-agnostic pl layout.  Rationale: pw2's whole-
 * volume time measures EQUAL to pw4's despite a 63% worse FMA-slot floor,
 * because ymm FMAs issue on p0/p1 and leave p5 wholly to shuffles; the
 * z-subpass is exactly the shuffle-dense site (2 register transposes per
 * element), so take the ymm form there and the zmm form where FMA slots
 * dominate.  hzil composes it with the il interleave. */
static void FN(x_hz0)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        for (int x = 0; x < L; ++x) {
            zsub_pw2(i + (size_t)x * PLND, P, 0, 0, 0, 0, 0, 0, 0);
            FN(ysub)(P, o + (size_t)x * PLND, 0, 0, 0, 0, 2 * L);
        }
        FN(phase2)(o, o, 0, 0, 0);
    }
}

static void FN(x_hzil)(const double *in, double *out, long nvol,
                       double *S, double *P)
{
    (void)S;
    double *oprev = 0;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        for (int x = 0; x < L; ++x) {
            zsub_pw2(i + (size_t)x * PLND, P, 0, 0, 0, 0, 0, 0, 0);
            FN(ysub)(P, o + (size_t)x * PLND, 0, 0, 0, 0, 2 * L);
            if (oprev) {
                const int t0 = ((NPLANE / PW) * x)       / L;
                const int t1 = ((NPLANE / PW) * (x + 1)) / L;
                FN(phase2_chunk)(oprev, oprev, 0, 0, 0, t0, t1);
            }
        }
        oprev = o;
    }
    FN(phase2)(oprev, oprev, 0, 0, 0);
}

/* ice_r2 CPY variants (mechanism from L45_mixedradix r8): y-pass into the
 * L1-hot plane image, rep movsb -> out, no RFO on phase 1's cold stores. */
static void FN(x_cp0)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        FN(phase1)(i, o, P, 0, 0, 0, 1, 2 * L, PLND);
        FN(phase2)(o, o, 0, 0, 0);
    }
}

/* cpy + the pf2 read set (paced PFIN on in, PF45 stream poke, PFNX) */
static void FN(x_cp2)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        const double *nx = (b + 1 < nvol) ? i + VDBL : 0;
        FN(phase1)(i, o, P, 1, 0, 0, 1, 2 * L, PLND);
        FN(phase2)(o, o, nx, 1, 1);
    }
}
#endif

static void FN(x_ip3)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    (void)S;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        const double *nx = (b + 1 < nvol) ? i + VDBL : 0;
        FN(phase1)(i, o, P, 1, 1, 0, 0, 2 * L, PLND);
        FN(phase2)(o, o, nx, 1, 1);
    }
}

static void FN(x_sp0)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        FN(phase1)(i, S, P, 0, 0, 0, 0, SROWD, SPLND);
        FN(phase2s)(S, o, 0, 0, 0);
    }
}

static void FN(x_sps)(const double *in, double *out, long nvol,
                      double *S, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        const double *nx = (b + 1 < nvol) ? i + VDBL : 0;
        FN(phase1)(i, S, P, 1, 0, 0, 0, SROWD, SPLND);
        FN(phase2s)(S, o, nx, 1, 1);
    }
}

#undef PF45
#undef PFW45
#undef PFNX
#undef PFIN
#undef PFWMID
#undef PFSTEP
#undef PFL

#undef TRNC
#ifdef G4B
# undef G4B
#endif
#undef VALIGNQ
#undef ZLA
#undef LDT
#undef STT
#undef GCOL
#undef SCOL
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef NG1
#undef NFULL
#undef NGRP
#undef LDU
#undef STU
#undef VSH
#undef FN
#undef veci
#undef uvec
#undef vec

#endif /* template */
