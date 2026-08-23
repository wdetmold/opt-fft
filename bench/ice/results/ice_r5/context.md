# What previous generations produced (round ice_r5 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/ice/results/ice_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/ice/results/ice_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/ice/results/ice_r3/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/ice/results/ice_r4/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/ice/results/ice_smoke/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L13_direct.md 352 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L13_rader.md 459 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L17_matrixsimd.md 637 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L17_rader.md 399 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L17_winograd.md 296 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L23_matrixsimd.md 334 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L23_rader.md 162 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L36_mixedradix.md 367 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L36_pencilfused.md 544 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L36_pfa.md 261 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L45_mixedradix.md 339 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L45_pfa.md 281 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L64_blocked.md 277 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L64_radix8.md 200 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L6_pfa.md 276 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L6_unrolled.md 349 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L8_batchsimd.md 385 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L8_fusedaxes.md 441 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L8_radix8.md 237 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  /home/lqcd/wdetmold/fft/bench/ice/exemplars/ice_r1/
      # Round ice_r1 — what it established
      
      Promoted: L6_unrolled L8_batchsimd L13_direct L13_rader L17_matrixsimd L17_winograd L17_rader L23_rader L23_matrixsimd L36_pfa L45_mixedradix L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      
  /home/lqcd/wdetmold/fft/bench/ice/exemplars/ice_r2/
      # Round ice_r2 — what it established
      
      Promoted: L6_unrolled L8_batchsimd L8_fusedaxes L13_direct L13_rader L17_matrixsimd L23_matrixsimd L36_pencilfused L36_pfa L45_mixedradix L64_blocked L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      
  /home/lqcd/wdetmold/fft/bench/ice/exemplars/ice_r3/
      # Round ice_r3 — what it established
      
      Promoted: L6_unrolled L8_fusedaxes L8_batchsimd L13_rader L13_direct L17_matrixsimd L17_winograd L23_matrixsimd L36_pencilfused L36_mixedradix L45_mixedradix L64_blocked L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      
  /home/lqcd/wdetmold/fft/bench/ice/exemplars/ice_r4/
      # Round ice_r4 — what it established
      
      Promoted: L6_unrolled L6_pfa L8_radix8 L8_fusedaxes L13_direct L13_rader L17_matrixsimd L17_winograd L23_rader L36_mixedradix L36_pencilfused L45_pfa L64_radix8 L64_blocked
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
=== round ice_r4 ===
# round ice_r4
host: a80n0.lqcd.mit   date: 2026-08-23T01:16:03-04:00   slurm_job: 438572
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (batched B=64, chain m=4856), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.332 us 103303.641 us    25.20       0.0%    0.338s  ok 2.4e-16       1.00x
   L6_pfa                         0.363 us 112949.601 us    23.04       0.1%    0.533s  ok 2.4e-16       1.09x
   mkl_dfti                       0.941 us 292561.588 us     8.90       0.1%    0.003s  ok 2.4e-16       2.83x
   mkl2026_dfti                   0.955 us 296820.679 us     8.77       0.4%    0.002s  ok 2.5e-16       2.87x
   fftw3_patient                  0.991 us 307839.300 us     8.46       0.1%    0.020s  ok 2.0e-16       2.98x
   fftw3_measure                  0.995 us 309268.558 us     8.42       1.8%    0.012s  ok 2.0e-16       2.99x
   fftw3_estimate                 1.559 us 484545.906 us     5.37       0.8%    0.001s  ok 2.0e-16       4.69x
   ducc0_c2c                      2.455 us 763061.734 us     3.41       0.6%    0.000s  ok 1.8e-16       7.39x
   baseline_matrix                7.335 us 2279645.130 us     1.14       0.0%    0.000s  ok 6.0e-16       22.07x

-- L=8 (batched B=64, chain m=2572), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_radix8                      0.564 us 92916.893 us    40.82       0.2%    0.031s  ok 2.3e-16       1.00x
   L8_fusedaxes                   0.744 us 122426.258 us    30.98       0.4%    0.380s  ok 2.3e-16       1.32x
   L8_batchsimd                   0.779 us 128187.991 us    29.59       3.6%    0.208s  ok 2.3e-16       1.38x
   mkl_dfti                       2.102 us 345957.045 us    10.96       0.7%    0.001s  ok 1.6e-16       3.72x
   mkl2026_dfti                   2.104 us 346341.595 us    10.95       1.9%    0.001s  ok 1.6e-16       3.73x
   fftw3_patient                  2.349 us 386592.086 us     9.81       4.2%    0.020s  ok 1.8e-16       4.16x
   fftw3_measure                  2.426 us 399412.707 us     9.50       0.7%    0.013s  ok 1.8e-16       4.30x
   ducc0_c2c                      4.154 us 683714.442 us     5.55       1.2%    0.000s  ok 1.3e-16       7.36x
   fftw3_estimate                 5.810 us 956382.444 us     3.97       0.1%    0.001s  ok 1.7e-16       10.29x
   baseline_matrix               22.697 us 3736077.120 us     1.02       0.1%    0.000s  ok 3.9e-16       40.21x

-- L=13 (batched B=32, chain m=1278), volume 2197, working set 2.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     5.837 us 238703.051 us    20.89       0.4%    0.152s  ok 2.9e-16       1.00x
   L13_rader                      6.363 us 260212.625 us    19.17       0.6%    0.165s  ok 3.0e-16       1.09x
   mkl2026_dfti                  11.795 us 482351.799 us    10.34       0.5%    0.003s  ok 3.3e-16       2.02x
   mkl_dfti                      11.994 us 490525.897 us    10.17       0.1%    0.002s  ok 3.2e-16       2.05x
   fftw3_patient                 13.437 us 549533.140 us     9.08       1.7%    0.022s  ok 3.2e-16       2.30x
   fftw3_measure                 13.581 us 555401.759 us     8.98       3.6%    0.012s  ok 3.2e-16       2.33x
   fftw3_estimate                14.780 us 604454.104 us     8.25       0.2%    0.001s  ok 3.2e-16       2.53x
   ducc0_c2c                     34.690 us 1418667.500 us     3.52       1.1%    0.000s  ok 2.5e-16       5.94x
   baseline_matrix              153.846 us 6291691.530 us     0.79       0.0%    0.000s  ok 7.9e-16       26.36x

-- L=17 (batched B=32, chain m=98), volume 4913, working set 4.80 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                13.009 us 40797.509 us    23.15       0.1%    1.523s  ok 3.3e-16       1.00x
   L17_winograd                  14.854 us 46581.667 us    20.28       2.2%    0.519s  ok 3.3e-16       1.14x
   L17_rader                     17.521 us 54946.643 us    17.19       2.5%    0.661s  ok 3.2e-16       1.35x
   ducc0_c2c                     86.405 us 270966.091 us     3.49       2.4%    0.000s  ok 2.6e-16       6.64x
   mkl2026_dfti                  88.483 us 277482.842 us     3.40       0.1%    0.053s  ok 3.1e-16       6.80x
   mkl_dfti                      88.821 us 278542.421 us     3.39       0.3%    0.051s  ok 3.1e-16       6.83x
   fftw3_estimate                90.140 us 282679.532 us     3.34       0.2%    0.002s  ok 3.0e-16       6.93x
   fftw3_measure                 90.166 us 282760.824 us     3.34       0.2%    0.009s  ok 3.0e-16       6.93x
   fftw3_patient                 90.230 us 282962.838 us     3.34       0.3%    0.016s  ok 3.0e-16       6.94x
   baseline_matrix              446.782 us 1401109.480 us     0.67       0.0%    0.000s  ok 8.4e-16       34.34x

-- L=23 (batched B=16, chain m=165), volume 12167, working set 5.94 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     38.105 us 100596.480 us    21.67       0.6%    0.487s  ok 3.8e-16       1.00x
   L23_matrixsimd                44.872 us 118462.267 us    18.40       0.8%    0.682s  ok 3.8e-16       1.18x
   ducc0_c2c                    247.547 us 653524.778 us     3.34       0.2%    0.000s  ok 2.8e-16       6.50x
   mkl_dfti                     262.099 us 691940.844 us     3.15       0.1%    0.050s  ok 4.2e-16       6.88x
   fftw3_estimate               269.164 us 710593.111 us     3.07       0.1%    0.002s  ok 3.7e-16       7.06x
   fftw3_measure                269.405 us 711230.216 us     3.06       0.0%    0.010s  ok 3.7e-16       7.07x
   fftw3_patient                269.602 us 711750.328 us     3.06       1.8%    0.032s  ok 3.7e-16       7.08x
   mkl2026_dfti                 283.552 us 748576.402 us     2.91       0.2%    0.048s  ok 4.2e-16       7.44x
   baseline_matrix             1480.089 us 3907434.150 us     0.56       0.0%    0.000s  ok 7.4e-16       38.84x

-- L=36 (batched B=8, chain m=64), volume 46656, working set 11.39 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix               111.425 us 57049.604 us    32.47       1.0%    0.314s  ok 3.6e-16       1.00x
   L36_pencilfused              111.962 us 57324.313 us    32.32       0.5%    0.751s  ok 3.6e-16       1.00x
   L36_pfa                      128.551 us 65818.087 us    28.15       0.4%    0.617s  ok 3.6e-16       1.15x
   mkl_dfti                     283.322 us 145060.786 us    12.77       0.3%    0.049s  ok 3.9e-16       2.54x
   mkl2026_dfti                 285.467 us 146158.981 us    12.67       0.7%    0.054s  ok 4.0e-16       2.56x
   fftw3_patient                372.977 us 190964.279 us     9.70       5.1%    0.943s  ok 3.9e-16       3.35x
   fftw3_measure                414.749 us 212351.678 us     8.72       0.7%    0.120s  ok 3.8e-16       3.72x
   ducc0_c2c                    443.480 us 227061.847 us     8.16       0.5%    0.000s  ok 3.0e-16       3.98x
   fftw3_estimate               489.327 us 250535.578 us     7.39       1.4%    0.005s  ok 3.5e-16       4.39x
   baseline_matrix             8802.888 us 4507078.530 us     0.41       0.1%    0.000s  ok 8.0e-16       79.00x

-- L=45 (batched B=4, chain m=177), volume 91125, working set 11.12 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                      283.339 us 200604.335 us    26.49       4.2%    1.175s  ok 4.0e-16       1.00x
   L45_mixedradix               283.800 us 200930.674 us    26.45       1.2%    0.396s  ok 4.1e-16       1.00x
   mkl_dfti                     756.802 us 535816.058 us     9.92       0.2%    0.053s  ok 4.4e-16       2.67x
   mkl2026_dfti                 778.250 us 551000.774 us     9.65       0.7%    0.051s  ok 4.5e-16       2.75x
   ducc0_c2c                    903.235 us 639490.422 us     8.31       2.3%    0.000s  ok 3.7e-16       3.19x
   fftw3_patient                933.881 us 661187.873 us     8.04       1.4%    0.609s  ok 4.3e-16       3.30x
   fftw3_measure                989.597 us 700634.952 us     7.59       1.4%    0.063s  ok 4.3e-16       3.49x
   fftw3_estimate              1183.630 us 838009.783 us     6.34       1.2%    0.002s  ok 4.2e-16       4.18x
   baseline_matrix            21678.063 us 15348068.800 us     0.35       0.7%    0.000s  ok 8.0e-16       76.51x

-- L=64 (batched B=2, chain m=134), volume 262144, working set 16.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                  1042.956 us 279512.293 us    22.62       2.0%    0.782s  ok 4.5e-16       1.00x
   L64_blocked                 1048.605 us 281026.112 us    22.50       5.1%    0.842s  ok 4.5e-16       1.01x
   mkl_dfti                    1723.169 us 461809.295 us    13.69       0.4%    0.048s  ok 3.4e-16       1.65x
   mkl2026_dfti                1855.447 us 497259.672 us    12.72       0.6%    0.049s  ok 3.4e-16       1.78x
   fftw3_patient               2023.096 us 542189.792 us    11.66       3.2%    3.626s  ok 3.5e-16       1.94x
   fftw3_measure               2063.531 us 553026.267 us    11.43       7.2%    0.240s  ok 3.5e-16       1.98x
   ducc0_c2c                   2568.259 us 688293.399 us     9.19       1.8%    0.000s  ok 3.0e-16       2.46x
   fftw3_estimate              3403.647 us 912177.272 us     6.93       0.4%    0.001s  ok 3.5e-16       3.26x
   baseline_matrix            94597.629 us 25352164.500 us     0.25       0.7%    0.000s  ok 7.8e-16       90.70x

backends:
   L13_direct               conj-folded dense 13x13 per axis, lanes=lines, pinned sines; 512b all-pinned zsolidY+xmm-tail X-first+ov; fchain=lazy-pairmap(v2)+xstep-ov; chain-ab[B32]=zs4076,ov4013,oz4261,op4605,os4084,pf4419 ns/vol pick=ov(inc); fch-ab=cz6088,cv6087 pick=cv(inc)
   L13_rader                Rader-13 CRT (93 FP/chunk) lanes=lines + FUSED MAP CHAIN (lazy X-pass map, 1 vsqrtpd/pt + rcp14 2-Newton), 512-bit; exec=ov chain=fo chain-ab[B32]=fo:6369,fz:6479,fs:7118,uf:6469
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, merged-reordered, 512-bit+ymm tail, pinned, X-first, addr-safe t1, pf=0, pw=0, pt=0, b1dec[yz/kyz/x/kx]=7.45/7.02/3.51/3.23, clk512/256=3.30/3.49 GHz, d256=3.50, chain=v2 map=s6 volmajor-inplace
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; tuned: xl 512t sp dy pin, pf=0, pfw=0, clk256=3.50 clk512=3.30, xrace xl/xfs=16.74/18.09, probe ph/xp/fu=12.33/4.98/17.73 us/vol, chain pv fused map ch=xk pin 18.67 nm=16.37 us/step
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, sched=0, var=h8, pf=0, pfw=0, cw=0, clk256=3.50GHz, clk512=3.30GHz, p1=5.36 f23=7.57 fu=12.82 fu4=13.52, chain=mx8, mch[nv=4,m=16]: mh8=16.24 mx8=15.15 mc8=16.10 mh4=18.13 mi4=18.14 mxc8=15.19
   L23_matrixsimd           dense 23x23/axis conj-folded, 512-bit, pinned, X-first, za, pf=0, pw=0, chain[vol-res fused-map mv=0], tune[ch pick=39.18 inc=39.18 us/t nv=16], clk512/256=3.30/3.50 GHz
   L23_rader                rader23 folded pair, 512-bit, pinned two-sweep, X-first, pf=0 pw=1, zmap=nr, tuner pick=37.11 inc=37.97 us/t nv=8
   L36_mixedradix           PFA 4x9 2-sweep + lazy fused map chain (in-place per-vol, own arena hp=2); pick=v1+mB-pf1 pinD=2112 B=8 mA-pf1=114.0 mA-pf0=115.4 mB-pf1=110.4 mB-pf0=111.8 mS-pf1=112.3 mS-pf0=113.5
   L36_pencilfused          L=36 PFA4x9; fchain pw=4 volres inplace lazymap2 rsqrt14+2NR+vdiv; exec tuner pw=4 mode=bcst+vr (B=8); chain probe us ip4=121.0 cs4=120.6 is04=127.3 bc4=112.2 xv4=124.2 xr4=122.5 vr4=110.4 nt4=121.0 nv4=118.8
   L36_pfa                  GT-PFA 4x9 (n1_9 DAG) two-sweep +fused lazy-map chain (ch=1 chm=0 mix=0 sp=1 pcc=152.2); tuner pick: pw=4 mode=inplace pf=0 tr=1 (B=8, nv=8, nc=21); probe us p1=71.4 p1t=68.5 p1z=32.1 p1y=22.7 p2w=23.6 fu=91.7
   L45_mixedradix           PFA 9x5 2-sweep; pick=v1-tr-pf1-pfw (B=4, arena=4, stream=0, 10 cand); chain=xfirst-ym ms0 cpf1/1 d1 w1 n0 k0 v1; nv1 us fu=248.2 p1=209.8 p1z=99.7 p1y=96.4 p1t=194.8 p1zt=75.2 p1yt=106.6 p2w=68.0; fe=na(open:13)
   L45_pfa                  GT-PFA 9x5 2-sweep +il(xvol-pipeline)+hz; pick pw4-il-pf0 262.2 us/vol (ip0=259.8 il0=262.2 hz=261.6) B=4 nv=4 | ch=vm-zs 301.8us/step ms0 (vm=301.8 vp=315.9 bm=317.3 pp=310.3 uf=320.6)
   L64_blocked              L64 8x8 two-stage, hugepage scratch, fused-chain (lazy map: rsqrt-Newton+1 vdivpd in pass1) ck=1; pick: pw=4 mode=nt pf=1 st=3(split-sc) pro=1 sb=0 (B=2, nv=2)
   L64_radix8               radix-8^2/axis split-cplx AVX-512; ice_r4 fused chain (map@z-store, rsqrt14+2N, all-FMA recip) ck=1; pick[B=2]=fused-plain+slabpf0+pro0+p10+sc0+xb0+fo0
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, zmm/ymm mixed, chain-pingpong-raced; variant=z512x ch=sep cst=0.414us clkS256=2.90 clkD256=2.90 clkS512=2.90 kclk=2.90GHz bf=195.1 bsp=195.0 bx=56.3 byz=136.9ns
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), zmm-x/ymm-yz zxf incumbent, fused fft3d_chain (vol-major in-place, lazy map, pair-shared exact ladder); variant=zxf cmap=bdiv kclk=3.30GHz ab1=y166.7,z155.1ns zwd=-3.4% xod=+2.2%
   L8_batchsimd             radix-8 split + fused-map fft3d_chain (rsqrt14+2N ladder, 1 div-op/8pt); pick[B=64]: mode=FUSEDAA3 nt=0 pf=s0 chain{fam=lnat,pf=s0} alloc=r8(a64,si512) chmap{nat/s0=0.890,nat/sc=0.982,aa2/s0=0.926,aa3/s0=0.862,lnat/s0=0.847,laa3/s0=0.851}
   L8_fusedaxes             8^3 fused/AA/AA2 c52 + fused-map chain[slot+pfs]; B=64 pick=fusedAA2+pfs (chain-tuned) chain-arena{fused+pfs=0.422,fusedAA2=0.424,fusedAA2+pfs=0.416,fusedAA2b=0.425,fusedAA2b+pfs=0.420} map-arena{div=0.781,div+pfs=0.771,fma=0.831,fma+pfs=0.819,div-pp=0.838,div+pfs-pp=0.832,slot=0.758,slot+pfs=0.756,lz=0.783,lz+pfs=0.770,rr46=0.753} pmc=na
   L8_radix8                radix-8 52-instr codelet; 2p/1f/3p; fused chain(vol-major L1, map=rsqrt14+2NR+div, ch=2); pick[B=64]=avx512-1faa-pfs (default) arena{1faa-pfs=0.451 1f-pfs=0.455 1faa=0.463 3p-pfs*=0.460}
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
