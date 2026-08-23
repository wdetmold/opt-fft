# What previous generations produced (round ice_r3 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/ice/results/ice_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/ice/results/ice_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/ice/results/ice_smoke/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L13_direct.md 175 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L13_rader.md 294 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L17_matrixsimd.md 339 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L17_rader.md 205 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L23_matrixsimd.md 174 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L36_mixedradix.md 199 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L36_pencilfused.md 224 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L36_pfa.md 119 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L45_mixedradix.md 181 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L45_pfa.md 133 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L64_blocked.md 136 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L64_radix8.md 97 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L6_pfa.md 114 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L6_unrolled.md 181 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L8_batchsimd.md 126 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L8_fusedaxes.md 150 lines
  /home/lqcd/wdetmold/fft/bench/ice/strategies/L8_radix8.md 117 lines

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
      

## Current standings (most recent leaderboard)
=== round ice_r2 ===
# round ice_r2
host: a80n0.lqcd.mit   date: 2026-08-22T19:06:56-04:00   slurm_job: 438572
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (batched B=64, chain m=4856), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.213 us 66169.683 us    39.34       0.1%    0.491s  ok 2.4e-16       1.00x
   L6_pfa                         0.213 us 66281.947 us    39.27       0.3%    0.481s  ok 2.4e-16       1.00x
   mkl_dfti                       0.341 us 105854.671 us    24.59       3.3%    0.002s  ok 2.4e-16       1.60x
   mkl2026_dfti                   0.356 us 110600.409 us    23.53       0.6%    0.003s  ok 2.5e-16       1.67x
   fftw3_measure                  0.445 us 138414.301 us    18.81       5.5%    0.012s  ok 2.0e-16       2.09x
   fftw3_patient                  0.451 us 140027.694 us    18.59       0.5%    0.020s  ok 2.0e-16       2.12x
   fftw3_estimate                 0.958 us 297602.029 us     8.75       0.3%    0.001s  ok 2.0e-16       4.50x
   ducc0_c2c                      1.973 us 613025.466 us     4.25       0.2%    0.000s  ok 1.8e-16       9.26x
   baseline_matrix                7.179 us 2231092.700 us     1.17       0.0%    0.000s  ok 5.9e-16       33.72x

-- L=8 (batched B=64, chain m=2572), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   0.544 us 89471.444 us    42.39       1.5%    0.034s  ok 2.3e-16       1.00x
   L8_fusedaxes                   0.544 us 89575.645 us    42.34       0.5%    0.183s  ok 2.3e-16       1.00x
   L8_radix8                      0.565 us 92978.925 us    40.79       1.6%    0.029s  ok 2.3e-16       1.04x
   mkl_dfti                       0.628 us 103306.237 us    36.71       3.2%    0.001s  ok 1.6e-16       1.15x
   mkl2026_dfti                   0.694 us 114203.005 us    33.21       3.2%    0.001s  ok 1.6e-16       1.28x
   fftw3_measure                  0.998 us 164217.181 us    23.09       2.5%    0.013s  ok 1.8e-16       1.84x
   fftw3_patient                  1.019 us 167688.858 us    22.62       5.0%    0.020s  ok 1.8e-16       1.87x
   ducc0_c2c                      2.940 us 483918.149 us     7.84       2.0%    0.000s  ok 1.3e-16       5.41x
   fftw3_estimate                 4.355 us 716919.608 us     5.29       0.4%    0.001s  ok 1.7e-16       8.01x
   baseline_matrix               21.374 us 3518358.670 us     1.08       1.2%    0.000s  ok 3.9e-16       39.32x

-- L=13 (batched B=32, chain m=1278), volume 2197, working set 2.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     4.662 us 190657.651 us    26.16       1.0%    0.128s  ok 2.9e-16       1.00x
   L13_rader                      4.680 us 191394.625 us    26.06       0.9%    0.009s  ok 3.0e-16       1.00x
   mkl2026_dfti                   6.027 us 246462.231 us    20.24       0.9%    0.003s  ok 3.2e-16       1.29x
   mkl_dfti                       6.245 us 255409.264 us    19.53       0.0%    0.003s  ok 3.2e-16       1.34x
   fftw3_patient                  7.659 us 313205.758 us    15.92       2.1%    0.021s  ok 3.2e-16       1.64x
   fftw3_measure                  7.775 us 317965.148 us    15.68       7.0%    0.012s  ok 3.2e-16       1.67x
   fftw3_estimate                 9.977 us 408030.412 us    12.22       3.2%    0.001s  ok 3.2e-16       2.14x
   ducc0_c2c                     29.476 us 1205449.380 us     4.14       0.7%    0.000s  ok 2.5e-16       6.32x
   baseline_matrix              149.958 us 6132699.080 us     0.81       0.0%    0.000s  ok 7.8e-16       32.17x

-- L=17 (batched B=32, chain m=98), volume 4913, working set 4.80 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                13.562 us 42531.929 us    22.21       1.9%    1.321s  ok 3.3e-16       1.00x
   L17_winograd                  16.082 us 50431.843 us    18.73       1.4%    0.733s  ok 3.3e-16       1.19x
   L17_rader                     18.693 us 58620.336 us    16.11       4.1%    0.581s  ok 3.2e-16       1.38x
   ducc0_c2c                     74.733 us 234362.426 us     4.03       0.6%    0.000s  ok 2.6e-16       5.51x
   mkl2026_dfti                  75.993 us 238315.199 us     3.96       0.2%    0.057s  ok 3.1e-16       5.60x
   mkl_dfti                      76.258 us 239144.701 us     3.95       0.3%    0.050s  ok 3.1e-16       5.62x
   fftw3_measure                 79.978 us 250811.004 us     3.77       0.4%    0.007s  ok 3.0e-16       5.90x
   fftw3_estimate                80.032 us 250980.464 us     3.76       0.2%    0.002s  ok 3.0e-16       5.90x
   fftw3_patient                 80.128 us 251280.452 us     3.76       0.1%    0.017s  ok 3.0e-16       5.91x
   baseline_matrix              436.697 us 1369481.360 us     0.69       0.0%    0.000s  ok 8.4e-16       32.20x

-- L=23 (batched B=16, chain m=165), volume 12167, working set 5.94 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     39.214 us 103524.638 us    21.05       1.3%    0.416s  ok 3.8e-16       1.00x
   L23_matrixsimd                39.223 us 103548.328 us    21.05       2.6%    0.683s  ok 3.8e-16       1.00x
   ducc0_c2c                    218.175 us 575980.830 us     3.78       2.8%    0.000s  ok 2.8e-16       5.56x
   mkl_dfti                     230.786 us 609275.124 us     3.58       0.1%    0.048s  ok 4.2e-16       5.89x
   fftw3_patient                243.119 us 641835.379 us     3.40       0.6%    0.028s  ok 3.7e-16       6.20x
   fftw3_measure                243.148 us 641911.703 us     3.40       0.2%    0.011s  ok 3.7e-16       6.20x
   fftw3_estimate               243.672 us 643294.094 us     3.39       0.1%    0.002s  ok 3.7e-16       6.21x
   mkl2026_dfti                 252.540 us 666705.637 us     3.27       3.1%    0.051s  ok 4.2e-16       6.44x
   baseline_matrix             1454.418 us 3839663.010 us     0.57       0.0%    0.000s  ok 7.4e-16       37.09x

-- L=36 (batched B=8, chain m=64), volume 46656, working set 11.39 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pencilfused              110.477 us 56563.978 us    32.75       0.8%    0.552s  ok 3.6e-16       1.00x
   L36_pfa                      112.727 us 57716.082 us    32.10       1.2%    0.429s  ok 3.6e-16       1.02x
   L36_mixedradix               116.814 us 59808.648 us    30.97       3.7%    0.526s  ok 3.6e-16       1.06x
   mkl_dfti                     161.140 us 82503.669 us    22.45       0.1%    0.056s  ok 3.9e-16       1.46x
   mkl2026_dfti                 163.507 us 83715.459 us    22.13      31.6%    0.052s  ok 4.0e-16       1.48x
   fftw3_patient                247.378 us 126657.412 us    14.63       3.1%    0.933s  ok 3.9e-16       2.24x
   fftw3_measure                294.750 us 150912.153 us    12.28       1.5%    0.122s  ok 3.8e-16       2.67x
   ducc0_c2c                    317.760 us 162693.348 us    11.39       1.3%    0.000s  ok 3.0e-16       2.88x
   fftw3_estimate               365.910 us 187345.765 us     9.89       2.0%    0.004s  ok 3.5e-16       3.31x
   baseline_matrix             8698.571 us 4453668.310 us     0.42       0.0%    0.000s  ok 8.0e-16       78.74x

-- L=45 (batched B=4, chain m=177), volume 91125, working set 11.12 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_mixedradix               263.344 us 186447.903 us    28.51       0.2%    0.339s  ok 4.0e-16       1.00x
   L45_pfa                      297.097 us 210344.849 us    25.27       0.9%    1.098s  ok 4.0e-16       1.13x
   mkl_dfti                     520.811 us 368734.430 us    14.41       0.2%    0.050s  ok 4.4e-16       1.98x
   mkl2026_dfti                 541.555 us 383420.697 us    13.86       0.2%    0.056s  ok 4.5e-16       2.06x
   ducc0_c2c                    656.202 us 464591.250 us    11.44       0.9%    0.000s  ok 3.7e-16       2.49x
   fftw3_patient                696.483 us 493110.022 us    10.78       1.4%    0.547s  ok 4.2e-16       2.64x
   fftw3_measure                773.300 us 547496.592 us     9.71       6.1%    0.060s  ok 4.1e-16       2.94x
   fftw3_estimate               975.719 us 690808.930 us     7.69       0.7%    0.002s  ok 4.2e-16       3.71x
   baseline_matrix            21475.546 us 15204686.300 us     0.35       1.3%    0.000s  ok 8.0e-16       81.55x

-- L=64 (batched B=2, chain m=134), volume 262144, working set 16.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_blocked                  891.973 us 239048.683 us    26.45       4.0%    0.726s  ok 4.5e-16       1.00x
   L64_radix8                   935.296 us 250659.329 us    25.23       0.7%    0.603s  ok 4.5e-16       1.05x
   mkl_dfti                    1021.687 us 273812.183 us    23.09       0.5%    0.052s  ok 3.4e-16       1.15x
   mkl2026_dfti                1148.693 us 307849.626 us    20.54       6.9%    0.053s  ok 3.4e-16       1.29x
   fftw3_patient               1393.232 us 373386.048 us    16.93       0.6%    3.618s  ok 3.5e-16       1.56x
   fftw3_measure               1418.263 us 380094.422 us    16.64       1.0%    0.271s  ok 3.4e-16       1.59x
   ducc0_c2c                   1881.409 us 504217.713 us    12.54       2.0%    0.000s  ok 3.0e-16       2.11x
   fftw3_estimate              2756.295 us 738686.934 us     8.56       0.7%    0.001s  ok 3.5e-16       3.09x
   baseline_matrix            94086.074 us 25215067.800 us     0.25       0.5%    0.000s  ok 7.8e-16       105.48x

backends:
   L13_direct               conj-folded dense 13x13 per axis, lanes=lines, pinned sines; 512b all-pinned zsolidY+xmm-tail X-first+ov; chain-ab[B32]=zs3801,ov3735,os3764,pf4153 ns/vol pick=ov(inc)
   L13_rader                Rader-13 CRT (93 FP/chunk) in L13_direct's lanes=lines pipeline, 512-bit, zsolidY+xmm-tail X-first; pick=zs ab[B32]=zs:4209,t1:4500,y2:4198,p7:4324,xl:5780
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, merged-phase, 512-bit+ymm tail, pinned, X-first, deferred-Z, addr-safe t1, pf=0, pw=0, pt=0, b1dec[yz/kyz/x/kx]=7.72/7.12/3.56/3.39, clk512/256=3.30/3.37 GHz, d256=3.50
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; tuned: xl 512t sp dy pin, pf=0, pfw=0, clk256=3.50 clk512=3.30, xrace xl/xfs=16.27/17.65, probe ph/xp/fu=11.71/4.87/16.89 us/vol
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=h4, pf=0, pfw=0, cw=0, clk256=3.50GHz, clk512=3.30GHz, p1=5.11 f23=7.97 fu=12.81 fu4=13.48
   L23_matrixsimd           dense 23x23/axis conj-folded, 512-bit, pinned, X-first, za, pf=0, pw=0, tune[ch pick=39.54 inc=39.54 us/t nv=16], clk512/256=3.30/3.50 GHz
   L23_rader                rader23 folded pair, 512-bit, pinned two-sweep, X-first, pf=0 pw=0, tuner pick=33.20 inc=33.20 us/t nv=8
   L36_mixedradix           PFA 4x9 2-sweep, lanes=lines, n1_9 DFT9, chain-tuned; pick=v1-pf1 (B=8, arena=8 vol, stream=0, 6 cand, pinD=2112) ex us/vol pf0=114.9 pfw=112.0 ppw=113.7 l1w=110.0
   L36_pencilfused          L=36 plane-fused y+z then strided x, PFA4x9 interleaved lanes, rev order; chain-shaped tuner picked pw=4 mode=bcst0 (B=8); chain probe us ip4=119.2 cs4=119.4 is04=124.2 bc4=110.0
   L36_pfa                  GT-PFA 4x9 (n1_9 DAG) two-sweep; tuner pick: pw=4 mode=inplace pf=0 tr=1 (B=8, nv=8, nc=21); probe us p1=75.1 p1t=70.6 p1z=32.7 p1y=25.2 p2w=23.9 fu=94.5
   L45_mixedradix           PFA 9x5 2-sweep; pick=v1-tr-pf1-pfw (B=4, arena=4, stream=0, 10 cand); nv1 us fu=215.1 p1=177.5 p1z=84.9 p1y=84.1 p1t=166.0 p1zt=63.9 p1yt=92.4 p2w=58.7; fe=na(open:13)
   L45_pfa                  GT-PFA 9x5 2-sweep +il(xvol-pipeline)+hz; pick pw4-ip-pf1 358.0 us/vol (ip0=363.1 il0=385.9 hz=356.0) B=4 nv=4
   L64_blocked              L64 8x8 two-stage, hugepage odd-line-padded scratch, CHAIN-tuned; pick: pw=4 mode=cached pf=1 st=3(split-sc) pro=0 sb=0 (B=2, nv=2)
   L64_radix8               radix-8^2/axis split-cplx AVX-512; ice_r2 chain-shaped tuner; pick[B=2]=fused-pfw+slabpf1+pro0+p10+sc0+xb0+fo0
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, zmm/ymm mixed, chain-pingpong-raced; variant=z512x clkS256=3.50 clkD256=3.50 clkS512=3.30 kclk=3.30GHz bf=172.0 bsp=171.4 bx=49.4 byz=120.4ns
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), radix-2-first VD6, ymm+zmm raced; variant=zxf kclk=3.30GHz ab1=y166.6,z155.1ns zwd=-2.6% xod=-0.1%
   L8_batchsimd             radix-8 split; pick[B=64]: mode=FUSEDAA2 nt=0 pf=s0 alloc=r8(a64,si512) arena{FUSEDAA2/s0=0.473,FUSEDAA2/none=0.491,FUSEDAA/s0=0.475,FUSED/s0=0.478,FUSED/none=0.491}
   L8_fusedaxes             8^3 fused/AA/AA2 c52; B=64 pick=fusedAA2+pfs (chain-tuned) chain-arena{fused=0.425,fused+pfs=0.423,fusedAA2=0.418,fusedAA2+pfs=0.411} pmc=na
   L8_radix8                radix-8 52-instr codelet; 2p/1f/3p shapes; pick[B=64]=avx512-1faa-pfs (default) arena{1faa-pfs=0.426 1f-pfs=0.427 1faa=0.435 3p-pfs*=0.428} scr@0x500
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
