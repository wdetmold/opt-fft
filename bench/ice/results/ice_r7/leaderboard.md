```
=== round ice_r7 ===
# round ice_r7
host: a80n0.lqcd.mit   date: 2026-08-23T12:59:51-04:00   slurm_job: 438637
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (batched B=64, chain m=4856), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.238 us 73961.842 us    35.19       0.1%    0.348s  ok 2.4e-16       1.00x
   L6_pfa                         0.320 us 99362.639 us    26.20       0.0%    0.552s  ok 2.4e-16       1.34x
   mkl_dfti                       0.939 us 291906.553 us     8.92       0.3%    0.002s  ok 2.4e-16       3.95x
   mkl2026_dfti                   0.952 us 295878.055 us     8.80       0.6%    0.002s  ok 2.5e-16       4.00x
   fftw3_patient                  0.986 us 306418.709 us     8.49       2.5%    0.020s  ok 2.0e-16       4.14x
   fftw3_measure                  0.993 us 308525.787 us     8.44       0.3%    0.013s  ok 2.0e-16       4.17x
   fftw3_estimate                 1.554 us 482970.290 us     5.39       0.2%    0.001s  ok 2.0e-16       6.53x
   ducc0_c2c                      2.466 us 766308.260 us     3.40       2.6%    0.000s  ok 1.9e-16       10.36x
   baseline_matrix                7.335 us 2279633.420 us     1.14       0.0%    0.000s  ok 5.9e-16       30.82x

-- L=8 (batched B=64, chain m=2572), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.511 us 84087.141 us    45.10       0.0%    0.209s  ok 2.3e-16       1.00x
   L8_radix8                      0.551 us 90722.335 us    41.80       0.3%    0.037s  ok 2.3e-16       1.08x
   L8_batchsimd                   0.555 us 91390.419 us    41.50       0.1%    0.148s  ok 2.3e-16       1.09x
   mkl_dfti                       2.112 us 347615.497 us    10.91       0.5%    0.002s  ok 1.6e-16       4.13x
   mkl2026_dfti                   2.120 us 348929.088 us    10.87       1.7%    0.001s  ok 1.6e-16       4.15x
   fftw3_measure                  2.344 us 385782.759 us     9.83       4.0%    0.013s  ok 1.8e-16       4.59x
   fftw3_patient                  2.362 us 388790.397 us     9.75       0.8%    0.022s  ok 1.8e-16       4.62x
   ducc0_c2c                      4.164 us 685489.757 us     5.53       0.6%    0.000s  ok 1.3e-16       8.15x
   fftw3_estimate                 5.808 us 956061.657 us     3.97       0.2%    0.001s  ok 1.7e-16       11.37x
   baseline_matrix               22.722 us 3740211.960 us     1.01       0.0%    0.000s  ok 3.9e-16       44.48x

-- L=13 (batched B=32, chain m=1278), volume 2197, working set 2.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     3.471 us 141947.584 us    35.13       0.3%    0.208s  ok 2.9e-16       1.00x
   L13_rader                      3.688 us 150833.868 us    33.06       0.1%    0.000s  ok 3.0e-16       1.06x
   mkl2026_dfti                  11.798 us 482496.063 us    10.34       2.3%    0.002s  ok 3.2e-16       3.40x
   mkl_dfti                      11.994 us 490516.322 us    10.17       1.4%    0.002s  ok 3.2e-16       3.46x
   fftw3_patient                 13.403 us 548127.915 us     9.10       1.0%    0.021s  ok 3.2e-16       3.86x
   fftw3_measure                 13.575 us 555182.145 us     8.98       0.9%    0.013s  ok 3.3e-16       3.91x
   fftw3_estimate                14.781 us 604469.454 us     8.25       0.8%    0.001s  ok 3.2e-16       4.26x
   ducc0_c2c                     34.732 us 1420391.440 us     3.51       1.6%    0.000s  ok 2.5e-16       10.01x
   baseline_matrix              153.821 us 6290650.050 us     0.79       0.0%    0.000s  ok 7.9e-16       44.32x

-- L=17 (batched B=32, chain m=98), volume 4913, working set 4.80 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                 9.035 us 28334.365 us    33.34       0.0%    1.138s  ok 3.3e-16       1.00x
   L17_winograd                  10.922 us 34252.665 us    27.58      17.1%    0.551s  ok 3.3e-16       1.21x
   L17_rader                     11.919 us 37376.616 us    25.27       0.3%    0.665s  ok 3.2e-16       1.32x
   ducc0_c2c                     86.235 us 270433.356 us     3.49       1.4%    0.000s  ok 2.6e-16       9.54x
   mkl2026_dfti                  88.604 us 277860.601 us     3.40       0.0%    0.050s  ok 3.1e-16       9.81x
   mkl_dfti                      88.832 us 278576.744 us     3.39       4.1%    0.053s  ok 3.1e-16       9.83x
   fftw3_estimate                90.082 us 282498.096 us     3.34       0.4%    0.002s  ok 3.0e-16       9.97x
   fftw3_patient                 90.173 us 282782.047 us     3.34       0.1%    0.018s  ok 3.0e-16       9.98x
   fftw3_measure                 90.211 us 282902.256 us     3.34       2.0%    0.008s  ok 3.0e-16       9.98x
   baseline_matrix              446.826 us 1401246.970 us     0.67       0.0%    0.000s  ok 8.4e-16       49.45x

-- L=23 (batched B=16, chain m=165), volume 12167, working set 5.94 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_matrixsimd                33.876 us 89431.939 us    24.37       0.1%    0.682s  ok 3.8e-16       1.00x
   L23_rader                     35.919 us 94825.237 us    22.98       0.7%    0.525s  ok 3.8e-16       1.06x
   ducc0_c2c                    249.083 us 657580.376 us     3.31       1.4%    0.000s  ok 2.8e-16       7.35x
   mkl_dfti                     262.032 us 691765.212 us     3.15       0.1%    0.051s  ok 4.2e-16       7.74x
   fftw3_estimate               269.147 us 710547.604 us     3.07       0.3%    0.002s  ok 3.7e-16       7.95x
   fftw3_patient                269.394 us 711199.563 us     3.06       0.2%    0.032s  ok 3.7e-16       7.95x
   fftw3_measure                269.492 us 711458.742 us     3.06       0.1%    0.010s  ok 3.7e-16       7.96x
   mkl2026_dfti                 283.460 us 748334.771 us     2.91       0.3%    0.049s  ok 4.2e-16       8.37x
   baseline_matrix             1480.017 us 3907244.880 us     0.56       0.0%    0.000s  ok 7.4e-16       43.69x

-- L=36 (batched B=8, chain m=64), volume 46656, working set 11.39 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix                99.809 us 51102.444 us    36.25       7.2%    0.352s  ok 3.6e-16       1.00x
   L36_pfa                      100.296 us 51351.547 us    36.07       0.2%    0.789s  ok 3.6e-16       1.00x
   L36_pencilfused              107.009 us 54788.640 us    33.81       0.4%    0.861s  ok 3.6e-16       1.07x
   mkl_dfti                     283.287 us 145042.944 us    12.77       0.5%    0.048s  ok 3.9e-16       2.84x
   mkl2026_dfti                 286.931 us 146908.714 us    12.61       0.3%    0.054s  ok 4.0e-16       2.87x
   fftw3_patient                371.831 us 190377.296 us     9.73       2.6%    0.913s  ok 3.9e-16       3.73x
   fftw3_measure                409.162 us 209490.848 us     8.84       3.6%    0.124s  ok 3.8e-16       4.10x
   ducc0_c2c                    443.705 us 227177.194 us     8.15       1.4%    0.000s  ok 3.1e-16       4.45x
   fftw3_estimate               487.700 us 249702.166 us     7.42       1.0%    0.003s  ok 3.5e-16       4.89x
   baseline_matrix             8806.553 us 4508955.160 us     0.41       0.0%    0.000s  ok 8.0e-16       88.23x

-- L=45 (batched B=4, chain m=177), volume 91125, working set 11.12 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_mixedradix               225.571 us 159704.099 us    33.28       3.2%    0.439s  ok 4.1e-16       1.00x
   L45_pfa                      231.220 us 163703.613 us    32.47       3.1%    1.268s  ok 4.0e-16       1.03x
   mkl_dfti                     757.588 us 536372.038 us     9.91       0.1%    0.056s  ok 4.4e-16       3.36x
   mkl2026_dfti                 778.597 us 551246.695 us     9.64       0.1%    0.052s  ok 4.5e-16       3.45x
   ducc0_c2c                    901.304 us 638123.072 us     8.33       0.3%    0.000s  ok 3.7e-16       4.00x
   fftw3_patient                937.978 us 664088.740 us     8.00       2.3%    0.594s  ok 4.3e-16       4.16x
   fftw3_measure                988.021 us 699519.099 us     7.60       0.4%    0.062s  ok 4.1e-16       4.38x
   fftw3_estimate              1183.316 us 837787.392 us     6.34       0.9%    0.002s  ok 4.2e-16       5.25x
   baseline_matrix            21677.947 us 15347986.500 us     0.35       0.9%    0.000s  ok 8.0e-16       96.10x

-- L=64 (batched B=2, chain m=134), volume 262144, working set 16.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_blocked                  608.213 us 163001.140 us    38.79       0.6%    0.051s  ok 4.5e-16       1.00x
   L64_radix8                   614.640 us 164723.629 us    38.38       0.4%    0.832s  ok 4.5e-16       1.01x
   mkl_dfti                    1722.819 us 461715.456 us    13.69       0.2%    0.046s  ok 3.4e-16       2.83x
   mkl2026_dfti                1851.935 us 496318.638 us    12.74       0.3%    0.048s  ok 3.4e-16       3.04x
   fftw3_patient               2040.737 us 546917.539 us    11.56       3.6%    3.790s  ok 3.5e-16       3.36x
   fftw3_measure               2076.793 us 556580.638 us    11.36       0.4%    0.275s  ok 3.6e-16       3.41x
   ducc0_c2c                   2558.460 us 685667.303 us     9.22       2.3%    0.000s  ok 3.0e-16       4.21x
   fftw3_estimate              3398.998 us 910931.333 us     6.94       0.4%    0.001s  ok 3.5e-16       5.59x
   baseline_matrix            94945.726 us 25445454.700 us     0.25       1.3%    0.000s  ok 7.8e-16       156.11x

backends:
   L13_direct               conj-folded dense 13x13 per axis, lanes=lines, pinned sines; 512b all-pinned zsolidY+xmm-tail X-first+ov; chain=S8[B>=8] soa8-2sweep-pinned map@Ystore(v0) else pairmap+mz1; chain-ab[B32]=zs4442,ov4368,oz4656,op5026,os4453,pf4904 ns/vol pick=ov(inc); fch-ab=cz7127,cv7173,v15922,v25917,v45910,m15558,m25584 pick=m1(inc)
   L13_rader                Rader-13; ice_r7 SoA BATCH-LANE chain (8 vols/zmm split re/im, zero-shuffle vertical pencils, 2 sweeps/step: map+z+y per L1 x-plane, then x-pencils; kernel=rader-split186 mf=2 sqr=1; convert 1/m; mined from 1000f989/v5_3907583b/v6_f40c5e25), tails+B<8 = r6 vm classic, 512-bit; exec=ov chain=soa8
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, merged-reordered, 512-bit+ymm tail, pinned, X-first, deferred-Z, addr-safe t1, pf=0, pw=0, pt=0, b1dec[yz/kyz/x/kx]=7.24/6.77/3.41/3.13, clk512/256=3.30/3.50 GHz, d256=3.50, chain=v7 map=s6 volsoa4-inplace xs1 p11 v7p1=0 sf1
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; tuned: xl 512t sp dy pin, pf=0, pfw=0, clk256=3.50 clk512=3.30, xrace xl/xfs=16.35/18.02, probe ph/xp/fu=11.73/4.85/17.08 us/vol, chain ch=ms6 12.58 nm=15.97 msr=12.27 msnm=9.03 msok=1 ms6=12.58 ms6nm=8.65 ok6=1 us/step
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, sched=0, var=h8, pf=0, pfw=0, cw=0, clk256=3.50GHz, clk512=3.30GHz, p1=5.50 f23=7.51 fu=12.99 fu4=13.50, chain=mrotc8 rok=1 rok2=1, mch[nv=4,m=16]: mh8=16.27 mx8=14.98 mc8=15.65 mh4=18.02 mi4=18.02 mxc8=14.82 msp8=14.73 mspc8=14.46 mrot8=12.65 mrotc8=12.53 mrp8=13.44 mrpc8=13.30
   L23_matrixsimd           dense 23x23/axis conj-folded, 512-bit, pinned, X-first, za, pf=0, pw=0, chain[vol-res inplace ZPLAIN-rot+cab yt20 map@store pcmp sfold mapv=1], tune[ch pick=39.42 inc=39.42 us/t nv=16], clk512/256=3.30/3.50 GHz
   L23_rader                rader23 folded pair, 512-bit, pinned two-sweep, X-first, pf=0 pw=0, zmap=nr, ch=h, tuner pick=32.88 inc=32.88 us/t nv=8, chAB e=38.79 g=37.98 r=37.84 h=36.72
   L36_mixedradix           PFA 4x9 2-sweep chain (per-vol in-place, arena hp=2); pick=v1+nF3-pf1 prot=p2map-cperm pinD=2112 B=8 nF3-pf1=122.5 nF4-pf1=127.3 nF-pf1=129.7 nE-pf1=133.8 nF-pf0=141.1 mQ-pf1=135.7
   L36_pencilfused          L=36 PFA4x9; fchain pw=4 volres inplace EAGER map@passB cperm hyb12B:6D r7[cu111.5 or109.1 tpp112.0 vs 108.1]; exec tuner pw=4 mode=bcst+vr (B=8); chain probe us ip4=139.1 cs4=138.3 is04=144.9 bc4=127.9 xv4=141.6 xr4=139.6 vr4=126.0 nt4=138.7 nv4=136.0
   L36_pfa                  GT-PFA 4x9 (n1_9 DAG) two-sweep +fused map chain (ch=1 chm=0 mix=7 sp=0 pcc=132.3 clk=2.90); tuner pick: pw=4 mode=inplace pf=0 tr=1 (B=8, nv=8, nc=21); probe us p1=81.8 p1t=74.7 p1z=32.2 p1y=23.3 p2w=23.5 fu=98.2
   L45_mixedradix           PFA 9x5 2-sweep; pick=v1-tr-pf1-pfw (B=4, arena=4, stream=0, 10 cand); chain=vg4-custody ov0 ms0 mp1 cpf1/1 d1 w1 n0 k0 v1; nv1 us fu=231.1 p1=201.9 p1z=89.8 p1y=95.4 p1t=185.8 p1zt=67.9 p1yt=101.5 p2w=61.8; fe=na(open:13)
   L45_pfa                  GT-PFA 9x5 2-sweep +tr(bcast-gather p1); pick pw4-tr0 221.1 us/vol (tr0=221.1 ip0=250.3 il0=250.5) B=4 nv=4 clk=2.90 | ch=q4 255.5us/step ms0 xf1 (xf-pf=281.8 xf=283.2 q4=255.5 q4-pf=253.5 vt-pf=-1.0 vt=-1.0 vm-zs=-1.0 uf=-1.0 vs=-1.0 vs2=-1.0)
   L64_blocked              L64 zsplit-custody chain (ice_r7: ztail inlined, zvp=0 vol-pair rows, zapf2@T3, zms=0 zs2=0) zs=1 ck=0 xk=1; exec pw=4 mode=cached pf=8 st=3(split-sc) pro=1 sb=0 (B=2)
   L64_radix8               radix-8^2/axis split-cplx AVX-512; r6 custody chain ckind=2 + ice_r7 pipes mp=1 lp=0 (map-ahead pencils, two-lb line phase) ck=1; pick[B=2]=fused-plain+slabpf1+pro1+p10+sc0
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, zmm/ymm mixed, chain-pingpong-raced; variant=z512x tier=h ch=hp2 cst=0.364us clkS256=2.90 clkD256=2.90 clkS512=2.90 kclk=2.90GHz bf=195.5 bsp=194.8 bx=56.3 byz=137.6ns
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), zmm-x/ymm-yz zxf incumbent, fused fft3d_chain (ice_r7: SoA 8-volumes-per-zmm group chain, zero-shuffle split-complex steady state, exact ladder+vdivpd map; pair/ipdiv tail); variant=zxf_pf cmap=g8 tier=exact g8rcp=0 kclk=3.30GHz ab1=y166.8,z154.6ns zwd=-2.7% xod=+3.8%
   L8_batchsimd             radix-8 split; VOLUME-MAJOR fused-map fft3d_chain (ice_r7: hp half-pass split-B via scr2, grid comb pass A, all-page-aligned frame; in-place, fixed clay, FTZ, rsqrt14+2N+1div map); pick[B=64]: mode=FUSEDAA3 nt=0 pf=s0 chain{vm-ip,sig=0,perm=hp} alloc=r8(a64,si512) chv{s0=0.710,s8=0.710,s16=0.711,s24=0.713,s40=0.714,s48=0.715,hp=0.697,hpo=0.698,hpd=0.694,hpr=0.701,hpc=0.696,r0=0.707,nat=0.701}
   L8_fusedaxes             8^3 fused/AA/AA2 c52 + bl8 batch-lane chain[pf=0;B%8==0] fb=vm3[sig=0,so=1,go=0,gs=2;fb2=slot+pfs]; B=64 pick=fusedAA2+pfs (chain-tuned) chain-arena{fused+pfs=0.428,fusedAA2=0.428,fusedAA2+pfs=0.422,fusedAA2b=0.429,fusedAA2b+pfs=0.426} vm-arena{s0=0.585,s8=0.585,s16=0.586,s24=0.585,s40=0.584,s48=0.584,s56=0.585|nn=0.584,dn=0.585,nr=0.586,dr=0.585,gn=0.586,gd=0.583,hp=0.554,hf=0.642} pmc=na
   L8_radix8                radix-8 52-instr codelet; 2p/1f/3p; fused chain(vol-major L1, map=rsqrt14+2NR+div, ch=6); pick[B=64]=avx512-1faa-pfs (default) arena{1faa-pfs=0.494 1f-pfs=0.494 1faa=0.505 3p-pfs*=0.493}
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
