```
=== round ice_r6 ===
# round ice_r6
host: a80n0.lqcd.mit   date: 2026-08-23T08:38:08-04:00   slurm_job: 438633
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (batched B=64, chain m=4856), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.291 us 90435.515 us    28.78       0.0%    0.308s  ok 2.4e-16       1.00x
   L6_pfa                         0.296 us 91850.012 us    28.34       0.0%    0.490s  ok 2.4e-16       1.02x
   mkl_dfti                       0.940 us 292218.384 us     8.91       1.2%    0.002s  ok 2.4e-16       3.23x
   mkl2026_dfti                   0.951 us 295640.176 us     8.80       0.6%    0.002s  ok 2.5e-16       3.27x
   fftw3_measure                  0.992 us 308362.793 us     8.44       2.6%    0.012s  ok 2.0e-16       3.41x
   fftw3_patient                  0.994 us 308921.481 us     8.43       0.4%    0.020s  ok 2.0e-16       3.42x
   fftw3_estimate                 1.556 us 483460.671 us     5.38       0.1%    0.001s  ok 2.0e-16       5.35x
   ducc0_c2c                      2.465 us 766230.330 us     3.40       0.7%    0.000s  ok 1.8e-16       8.47x
   baseline_matrix                7.335 us 2279653.870 us     1.14       0.0%    0.000s  ok 6.0e-16       25.21x

-- L=8 (batched B=64, chain m=2572), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.555 us 91301.214 us    41.54       0.0%    0.209s  ok 2.3e-16       1.00x
   L8_radix8                      0.569 us 93678.671 us    40.48       0.3%    0.037s  ok 2.3e-16       1.03x
   L8_batchsimd                   0.575 us 94645.469 us    40.07       0.0%    0.153s  ok 2.3e-16       1.04x
   mkl2026_dfti                   2.112 us 347580.976 us    10.91       0.6%    0.001s  ok 1.6e-16       3.81x
   mkl_dfti                       2.120 us 349017.698 us    10.87       0.9%    0.002s  ok 1.6e-16       3.82x
   fftw3_patient                  2.308 us 379841.295 us     9.98       5.4%    0.020s  ok 1.8e-16       4.16x
   fftw3_measure                  2.423 us 398779.660 us     9.51       0.4%    0.013s  ok 1.8e-16       4.37x
   ducc0_c2c                      4.165 us 685535.242 us     5.53       0.7%    0.000s  ok 1.3e-16       7.51x
   fftw3_estimate                 5.796 us 954110.443 us     3.97       0.4%    0.001s  ok 1.8e-16       10.45x
   baseline_matrix               22.702 us 3736966.450 us     1.01       0.0%    0.000s  ok 3.9e-16       40.93x

-- L=13 (batched B=32, chain m=1278), volume 2197, working set 2.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     5.287 us 216220.850 us    23.07       0.2%    0.210s  ok 2.9e-16       1.00x
   L13_rader                      5.377 us 219907.269 us    22.68       0.3%    0.197s  ok 3.0e-16       1.02x
   mkl2026_dfti                  11.798 us 482493.898 us    10.34       0.0%    0.003s  ok 3.2e-16       2.23x
   mkl_dfti                      12.016 us 491406.691 us    10.15       0.6%    0.002s  ok 3.2e-16       2.27x
   fftw3_patient                 13.433 us 549352.572 us     9.08       0.9%    0.022s  ok 3.3e-16       2.54x
   fftw3_measure                 13.492 us 551770.248 us     9.04       3.8%    0.013s  ok 3.2e-16       2.55x
   fftw3_estimate                14.787 us 604733.343 us     8.25       1.5%    0.001s  ok 3.2e-16       2.80x
   ducc0_c2c                     34.693 us 1418798.780 us     3.52       1.1%    0.000s  ok 2.5e-16       6.56x
   baseline_matrix              153.799 us 6289754.460 us     0.79       0.0%    0.000s  ok 7.8e-16       29.09x

-- L=17 (batched B=32, chain m=98), volume 4913, working set 4.80 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_winograd                  11.649 us 36530.338 us    25.86       1.7%    0.536s  ok 3.3e-16       1.00x
   L17_matrixsimd                11.935 us 37428.877 us    25.24       0.2%    1.548s  ok 3.3e-16       1.02x
   L17_rader                     12.284 us 38523.234 us    24.52       0.4%    0.791s  ok 3.2e-16       1.05x
   ducc0_c2c                     86.263 us 270520.560 us     3.49       0.7%    0.000s  ok 2.6e-16       7.41x
   mkl2026_dfti                  88.505 us 277552.679 us     3.40       0.0%    0.055s  ok 3.1e-16       7.60x
   mkl_dfti                      88.825 us 278553.849 us     3.39       0.2%    0.050s  ok 3.1e-16       7.63x
   fftw3_estimate                90.146 us 282698.091 us     3.34       0.1%    0.002s  ok 3.0e-16       7.74x
   fftw3_patient                 90.178 us 282796.987 us     3.34       0.4%    0.018s  ok 3.0e-16       7.74x
   fftw3_measure                 90.184 us 282817.191 us     3.34       0.0%    0.008s  ok 3.0e-16       7.74x
   baseline_matrix              446.854 us 1401333.980 us     0.67       0.0%    0.000s  ok 8.4e-16       38.36x

-- L=23 (batched B=16, chain m=165), volume 12167, working set 5.94 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_matrixsimd                35.888 us 94743.665 us    23.00       0.5%    0.720s  ok 3.8e-16       1.00x
   L23_rader                     37.172 us 98134.052 us    22.21       0.8%    0.506s  ok 3.8e-16       1.04x
   ducc0_c2c                    247.482 us 653353.355 us     3.34       2.7%    0.000s  ok 2.8e-16       6.90x
   mkl_dfti                     262.077 us 691883.064 us     3.15       0.1%    0.049s  ok 4.2e-16       7.30x
   fftw3_measure                268.975 us 710094.633 us     3.07       0.2%    0.010s  ok 3.7e-16       7.49x
   fftw3_patient                269.034 us 710248.995 us     3.07       0.6%    0.032s  ok 3.7e-16       7.50x
   fftw3_estimate               269.170 us 710608.240 us     3.07       0.2%    0.002s  ok 3.7e-16       7.50x
   mkl2026_dfti                 283.602 us 748710.084 us     2.91       0.1%    0.053s  ok 4.2e-16       7.90x
   baseline_matrix             1480.238 us 3907827.740 us     0.56       0.0%    0.000s  ok 7.4e-16       41.25x

-- L=36 (batched B=8, chain m=64), volume 46656, working set 11.39 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix               100.801 us 51610.332 us    35.89       0.4%    0.351s  ok 3.6e-16       1.00x
   L36_pfa                      106.249 us 54399.405 us    34.05       0.7%    0.730s  ok 3.6e-16       1.05x
   L36_pencilfused              106.908 us 54737.148 us    33.84       0.0%    0.752s  ok 3.6e-16       1.06x
   mkl_dfti                     284.283 us 145553.069 us    12.73       0.2%    0.051s  ok 3.9e-16       2.82x
   mkl2026_dfti                 287.169 us 147030.332 us    12.60       0.4%    0.048s  ok 4.0e-16       2.85x
   fftw3_patient                372.670 us 190806.869 us     9.71       2.7%    0.933s  ok 3.9e-16       3.70x
   fftw3_measure                415.846 us 212913.380 us     8.70       1.3%    0.121s  ok 3.8e-16       4.13x
   ducc0_c2c                    441.392 us 225992.614 us     8.20       1.8%    0.000s  ok 3.0e-16       4.38x
   fftw3_estimate               486.929 us 249307.457 us     7.43       1.6%    0.004s  ok 3.5e-16       4.83x
   baseline_matrix             8802.685 us 4506974.880 us     0.41       0.1%    0.000s  ok 8.0e-16       87.33x

-- L=45 (batched B=4, chain m=177), volume 91125, working set 11.12 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                      264.111 us 186990.279 us    28.42       3.2%    1.018s  ok 4.0e-16       1.00x
   L45_mixedradix               264.977 us 187603.385 us    28.33       0.8%    0.392s  ok 4.1e-16       1.00x
   mkl_dfti                     756.922 us 535900.524 us     9.92       0.3%    0.050s  ok 4.4e-16       2.87x
   mkl2026_dfti                 777.737 us 550637.617 us     9.65       0.1%    0.033s  ok 4.5e-16       2.94x
   ducc0_c2c                    893.724 us 632756.904 us     8.40       0.9%    0.000s  ok 3.7e-16       3.38x
   fftw3_patient                939.951 us 665485.313 us     7.99       0.6%    0.586s  ok 4.3e-16       3.56x
   fftw3_measure                986.186 us 698219.377 us     7.61       1.4%    0.063s  ok 4.3e-16       3.73x
   fftw3_estimate              1184.376 us 838538.466 us     6.34       0.3%    0.002s  ok 4.2e-16       4.48x
   baseline_matrix            21666.742 us 15340053.500 us     0.35       0.0%    0.000s  ok 8.0e-16       82.04x

-- L=64 (batched B=2, chain m=134), volume 262144, working set 16.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_blocked                  638.757 us 171186.990 us    36.94       0.5%    0.049s  ok 4.5e-16       1.00x
   L64_radix8                   660.301 us 176960.540 us    35.73       1.9%    0.788s  ok 4.5e-16       1.03x
   mkl_dfti                    1720.604 us 461121.771 us    13.71       0.4%    0.052s  ok 3.4e-16       2.69x
   mkl2026_dfti                1852.553 us 496484.319 us    12.74       0.2%    0.048s  ok 3.4e-16       2.90x
   fftw3_measure               2069.810 us 554709.184 us    11.40       2.4%    0.269s  ok 3.5e-16       3.24x
   fftw3_patient               2071.931 us 555277.542 us    11.39       3.7%    3.685s  ok 3.6e-16       3.24x
   ducc0_c2c                   2561.552 us 686495.933 us     9.21       2.2%    0.000s  ok 3.0e-16       4.01x
   fftw3_estimate              3404.486 us 912402.126 us     6.93       0.2%    0.001s  ok 3.5e-16       5.33x
   baseline_matrix            94941.767 us 25444393.500 us     0.25       0.3%    0.000s  ok 7.8e-16       148.64x

backends:
   L13_direct               conj-folded dense 13x13 per axis, lanes=lines, pinned sines; 512b all-pinned zsolidY+xmm-tail X-first+ov; fchain=pairmap(v2)+mz1; chain-ab[B32]=zs4534,ov4477,oz4759,op5136,os4578,pf4972 ns/vol pick=ov(inc); fch-ab=cz7177,cv7096,v16392,v26367,v46356,m16198,m26304 pick=m1(inc)
   L13_rader                Rader-13 CRT (93 FP/chunk) lanes=lines + paired lazy-map chain, ice_r6 VOLUME-GROUP-MAJOR (vm, L13_direct r5 machinery: all m steps per G-vol L2-resident group, xstep-ov inside), 512-bit; exec=ov chain=v2 fch-ab[B32]=cv:7959,cz:7882,v1:6504,v2:6553,v4:6563
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, merged-reordered, 512-bit+ymm tail, pinned, X-first, addr-safe t1, pf=0, pw=1, pt=0, b1dec[yz/kyz/x/kx]=7.31/6.84/3.46/3.23, clk512/256=3.30/3.50 GHz, d256=3.50, chain=v6 map=s6 volmajor-inplace sf1 xs1 p11
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; tuned: xl 512t ov pin, pf=0, pfw=0, clk256=3.50 clk512=3.30, xrace xl/xfs=22.51/25.12, probe ph/xp/fu=17.16/7.35/23.30 us/vol, chain ch=msr 12.28 nm=16.82 msr=12.28 msnm=9.03 msok=1 us/step
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, sched=0, var=h8, pf=0, pfw=0, cw=0, clk256=3.50GHz, clk512=3.30GHz, p1=5.19 f23=7.48 fu=12.80 fu4=13.50, chain=mrot8 rok=1, mch[nv=4,m=16]: mh8=15.97 mx8=15.25 mc8=16.19 mh4=18.05 mi4=18.06 mxc8=15.26 msp8=14.54 mspc8=14.51 mrot8=13.07 mrotc8=13.18
   L23_matrixsimd           dense 23x23/axis conj-folded, 512-bit, pinned, X-first, pipelined, za, pf=0, pw=0, chain[vol-res inplace zst20+cpad map@store pcmp sfold mapv=1], tune[ch pick=39.28 inc=45.80 us/t nv=16], clk512/256=3.30/3.49 GHz
   L23_rader                rader23 folded pair, 512-bit, pinned two-sweep, X-first, pf=0 pw=0, zmap=nr, ch=r, tuner pick=32.94 inc=32.94 us/t nv=8, chAB e=38.61 g=38.59 r=38.13
   L36_mixedradix           PFA 4x9 2-sweep chain (per-vol in-place, arena hp=2); pick=v1+nF3-pf1 prot=p2map-cperm pinD=2112 B=8 nF3-pf1=100.1 nF-pf1=101.5 nF4-pf1=99.4 nFD-pf1=101.2 nE-pf1=102.1 nF-pf2=101.8 nF2-pf1=102.1 nF-pf0=103.0 mQ-pf1=106.5
   L36_pencilfused          L=36 PFA4x9; fchain pw=4 volres inplace lazymap2-rgi rsqrt14+2NR+vdiv; exec tuner pw=4 mode=bcst+vr (B=8); chain probe us ip4=121.0 cs4=120.3 is04=126.8 bc4=112.0 xv4=124.0 xr4=122.1 vr4=110.2 nt4=120.9 nv4=118.6
   L36_pfa                  GT-PFA 4x9 (n1_9 DAG) two-sweep +fused map chain (ch=1 chm=0 mix=6 sp=0 pcc=107.8 clk=3.50); tuner pick: pw=4 mode=inplace pf=0 tr=1 (B=8, nv=8, nc=21); probe us p1=60.4 p1t=58.9 p1z=28.2 p1y=19.8 p2w=20.8 fu=79.1
   L45_mixedradix           PFA 9x5 2-sweep; pick=v1-tr-pf1-pfw (B=4, arena=4, stream=0, 10 cand); chain=xfirst-ymp48 ov0 ms0 mp1 cpf1/1 d1 w1 n0 k0 v1; nv1 us fu=220.1 p1=180.2 p1z=87.6 p1y=86.0 p1t=170.6 p1zt=65.7 p1yt=94.8 p2w=59.8; fe=na(open:13)
   L45_pfa                  GT-PFA 9x5 2-sweep +tr(bcast-gather p1); pick pw4-tr0 226.4 us/vol (tr0=226.4 ip0=257.8 il0=257.3) B=4 nv=4 clk=3.50 | ch=xf-pf 316.6us/step ms0 xf1 (xf-pf=316.6 xf=321.4 vt-pf=-1.0 vt=-1.0 vm-zs=-1.0 uf=-1.0 vs=-1.0 vs2=-1.0)
   L64_blocked              L64 zsplit-custody chain (2 in-place sweeps/step, per-vol residency, lazy map, zapf3@T3 zs2=0) zs=1 ck=0 xk=1; exec pw=4 mode=cached pf=8 st=3(split-sc) pro=1 sb=0 (B=2)
   L64_radix8               radix-8^2/axis split-cplx AVX-512; ice_r6 chain ckind=2 (2=fused-boundary custody: alt x/y-plane sweeps, 1 sweep/step, map+zline in plane) ck=1; pick[B=2]=fused-pfw+slabpf1+pro0+p10+sc0
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, zmm/ymm mixed, chain-pingpong-raced; variant=z512x ch=fip cst=0.295us clkS256=3.50 clkD256=3.50 clkS512=3.30 kclk=3.30GHz bf=171.8 bsp=171.0 bx=49.4 byz=120.3ns
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), zmm-x/ymm-yz zxf incumbent, fused fft3d_chain (pair-interleaved vol-major chains, fused in-place map pass, pair-shared ladder, split-pass pair FFT); variant=zxf cmap=p2x tier=fast kclk=3.30GHz ab1=y166.9,z155.1ns zwd=-2.8% xod=+3.2%
   L8_batchsimd             radix-8 split; VOLUME-MAJOR fused-map fft3d_chain (ice_r6: IN-PLACE single-state step, fixed clay, FTZ, split state 384 shuf/step, rsqrt14+2N+1div map); pick[B=64]: mode=FUSEDAA3 nt=0 pf=s0 chain{vm-ip,sig=0,perm=r0} alloc=r8(a64,si512) chv{s0=0.620,s8=0.626,s16=0.639,s24=0.637,s40=0.627,s48=0.622,r0=0.615,nat=0.614,r0d=0.623,oe=0.614,r0g=0.618,natg=0.620}
   L8_fusedaxes             8^3 fused/AA/AA2 c52 + vm3 rot-chain[sig=0,so=1,go=0,gs=2;fb=slot+pfs]; B=64 pick=fusedAA2+pfs (chain-tuned) chain-arena{fused+pfs=0.425,fusedAA2=0.424,fusedAA2+pfs=0.417,fusedAA2b=0.426,fusedAA2b+pfs=0.420} vm-arena{s0=0.583,s8=0.583,s16=0.583,s24=0.583,s40=0.583,s48=0.583,s56=0.583|nn=0.583,dn=0.583,nr=0.583,dr=0.583,gn=0.583,gd=0.585,hp=0.556,hf=0.646} pmc=na
   L8_radix8                radix-8 52-instr codelet; 2p/1f/3p; fused chain(vol-major L1, map=rsqrt14+2NR+div, ch=2); pick[B=64]=avx512-1faa-pfs (default) arena{1faa-pfs=0.476 1f-pfs=0.480 1faa=0.481 3p-pfs*=0.462}
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
