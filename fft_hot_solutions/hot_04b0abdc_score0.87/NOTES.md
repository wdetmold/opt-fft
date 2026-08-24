# Solution notes

Single C file (`implementation.c`), compiled to `implementation.so` (prebuilt;
solution.py rebuilds from source only if the .so is missing).

## Structure
- L = 6, 8, 13, 17, 23: prior-work engine (warm attempt 00291a90 final artifact,
  regenerated + verified): batch-lane SoA (8 volumes/zmm) with hand-scheduled asm
  prime codelets and per-volume tail paths for batch remainders.
- L = 36: routed: multiples of 8 volumes -> prior batch-lane PFA(4x9) engine;
  remainder -> new within-volume PFA engine (below).
- L = 45: new within-volume PFA(5x9) engine.
- L = 64: new within-volume CT(8x8) engine.

L=64 fused stage additionally runs as a 1-deep software pipeline
(map stage of group k overlapped with the second DFT of group k-1 through a
small staging buffer) -- worth ~9% on that loop on this VM.

New engines (this round): two-stage column codelets as small rolled loops
(L1i-resident), digit-swapped / PFA-input-order per-axis storage so stage-1
reads and stage-2 writes are contiguous slot groups; 8x8 register-tile
transposes fused into the z-axis stages; elementwise map z/(1+|z|)
(rsqrt14+NR+exact-FMA-Heron sqrt; rcp14+3rd-order NR reciprocal, ~1 ulp)
fused between the completing stage-2 and the next step's stage-1; c
pre-permuted into consumption order (slab + pencil copies); one full-volume
sweep per step by alternating slab visits (y,z axes) and pencil visits (x),
each completing step t and pre-transforming step t+1; first/final-step
variants; per-volume chain processing keeps the active set ~13 MB (L=64).

Measured on this VM (key numbers): rsqrt14/rcp14 are fast (1.3 tsc tput, not
microcoded in small-loop context); 2 FMA/cyc sustains with up to ~2 loads +
1 store per 8 FMAs; 1024-byte strides cause heavy 4K-alias stalls (rows padded
to 136/144 doubles); sw prefetch loses consistently; hw sqrt/div map loses
(divider tput).

## Self-benchmark (final, this machine, fresh processes, pinned core 0)
Workload proxy W3 = seed 7, B=(4,..,4), m=(25000,10000,3000,1200,800,500,400,300)
(local MKL DFTI reference on W3: ~13.3 s best-of-2, matching the graded
C_sota scale of prior rounds):
  solution.py fresh-process walls (5 runs x best-of-3, pinned core 0):
    [2.280 2.275 2.244] [2.296 2.296 2.271] [2.225 2.202 2.216]
    [2.199 2.231 2.263] [2.301 2.342 2.332]  -> best 2.199 s, median-of-bests ~2.24 s
  (base.py/pocketfft on the same workload: ~22.5-23.6 s; local MKL DFTI: ~13.3 s)
Correctness: all 16 per-size gates pass vs base.py on W3-scale chain lengths
(one-step rel ~8e-16, worst m-step margin >12x under its gate), plus a matrix
of edge shapes (B=0/1/odd/16, m=1..9), bitwise-deterministic across calls.
