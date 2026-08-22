# 09 — Small batched complex-double 3D FFTs on one A100 (sm_80): capacity, roofline, baselines, DMMA

**Audience and purpose.** You are about to write a CUDA kernel for the GPU phase of this
project: forward, unnormalized, **complex-double** 3D DFT of a fixed cube `L^3` over a batch
of `B` volumes, on **one NVIDIA A100-PCIE-40GB (sm_80)**, with no FFT library allowed inside
the kernel. The geometries are `L = 6, 8, 17, 36` and the second wave `L = 13, 23, 45, 64`.
The correctness gate is a relative L2 error below `1e-12` against a reference — which, as §5
shows, is the single constraint that rules out most of the published tensor-core FFT
literature.

Sections 01–06 of this corpus are CPU algorithm literature. §07 is the *only* prior
accelerator section and it was written to mine the GPU world for CPU lessons — it is
deliberately upside-down for our purpose now, and several of its facts are stale or were
taken from a README that its own source code contradicts (§10 lists every correction). §08
is the CPU hardware section for the Xeon Gold 5218. **This section is the GPU hardware and
GPU baseline section.** It assumes you have read `bench/geom/PANEL_BRIEF.md` for the shape of
the contract and §07 §1 for the fusion argument, and it does not repeat either.

**The one-paragraph version.** Every one of the eight geometries is **memory-bandwidth-bound
by a factor of 2.2× to 5.2×** on the A100's *ordinary* FP64 pipe (§2.2), so the figure of
merit is bytes moved, not butterflies, and the target is a single global read plus a single
global write per volume. For `L ≤ 17` that target is *achievable in one kernel*: an entire
`17^3` complex-double volume is 76.8 KiB, which is 47 % of the 163 KiB a single thread block
may address in shared memory and 30 % of an SM's whole register file, so all three axes can
be fused with no round trip (§1). For `L = 23, 36, 45, 64` it is not, and the design question
becomes how few passes you can get away with and whether the intermediate can be kept in the
**40 MiB L2**, which reads at 7.2 TB/s — 4.6× HBM (§2.4). The baseline to beat is not cuFFT:
VkFFT's own committed FP64 A100 benchmark shows cuFFT sitting at bands of ≈630 and ≈420 GB/s
where VkFFT holds ≈1250–1290 GB/s, and those bands are almost exactly 1/2 and 1/3 of the top
band — i.e. **cuFFT is spending two or three global passes where one is possible** (§2.3,
§3.3). And the most interesting result: FP64 tensor cores are **IEEE-compliant** (NVIDIA's own
words) and exactly 2× the vanilla FP64 rate, which moves the size at which a *dense DFT matrix
per axis* becomes free — hidden entirely under the bandwidth floor — from `L ≈ 8` up to
`L ≈ 17` (§5.4). The CPU panel's surprise winner at `L = 17` was a dense conjugate-symmetric
matvec. On an A100 that formulation has a dedicated datapath, and 17 is precisely where it
breaks even.

**How to read the citations.** Same convention as §07 and §08:

* `[F]` — I fetched this URL in this session and the quoted text came out of that fetch.
* `[F-agent]` — fetched in this session by a research subagent I dispatched; the URL and the
  quote came back from its fetch, and where the claim is load-bearing I re-fetched or
  re-derived it myself and say so.
* `[F-src]` — read directly out of source code in this repository at `ext/src/VkFFT`
  (VkFFT v1.3.4, commit `066a17c`). File and line given.
* `[M]` — **measured on this machine** by a read-only device query, not a benchmark.
* `[UNVERIFIED — could not fetch]` — I could not retrieve it. Do not rely on it.

Anything with no tag lives in §11 (unsourced engineering notes) and is my own arithmetic.

---

## 0. The target hardware, as the machine itself reports it

This is not a citation. It is `nvidia-smi` and the CUDA 11.7 `deviceQuery` sample run on the
login node `wombat`, which has one of these parts attached. `[M]`

```
$ nvidia-smi -L
GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-8fb35d76-...)
   driver 560.35.03,  CUDA 12.6,  ECC enabled,  250 W cap,  MIG disabled

$ deviceQuery
  CUDA Capability Major/Minor version number:    8.0
  (108) Multiprocessors, ( 64) CUDA Cores/MP:    6912 CUDA Cores
  GPU Max Clock rate:                            1410 MHz
  Memory Clock rate:                             1215 MHz
  Memory Bus Width:                              5120-bit
  L2 Cache Size:                                 41943040 bytes        (40 MiB)
  Total amount of shared memory per block:       49152 bytes           (default carveout)
  Total number of registers available per block: 65536                 (32-bit registers)
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Total amount of constant memory:               65536 bytes
```

Two things in that dump are traps. **`49152` is the *default* shared-memory limit per block,
not the maximum** — the maximum on sm_80 is 163 KiB and reaching it requires an explicit
opt-in (§1.1). And the **250 W** cap is the PCIe SKU's, against 400 W for the SXM part that
NVIDIA's 9.7 / 19.5 TFLOP/s figures are quoted for; §7.4 explains why your harness must read
the clock-throttle reasons rather than assume 1410 MHz.

The cluster also has 8-GPU A100 nodes in the `a100l` / `a100r` partitions (`gpu:a100:8`), so
an exclusive single-GPU measurement is arrangeable. `[M]` (`sinfo`)

### 0.1 The documented budgets, from NVIDIA

| resource | value on sm_80 | source |
|---|---|---|
| SMs | **108** (of 128 on a full GA100) | whitepaper `[F]` |
| FP64 cores per SM | **32** | CUDA Programming Guide `[F-agent]` |
| Peak FP64 (vanilla) | **9.7 TFLOP/s** | whitepaper Table 1 `[F]` |
| Peak FP64 **Tensor Core** | **19.5 TFLOP/s** | whitepaper Table 1 `[F]` |
| HBM2 bandwidth (40 GB) | **1555 GB/s** | whitepaper Table 4 `[F]` |
| L2 | **40 MiB** (40960 KB), read **5120 B/clk** | whitepaper Table 4 + L2 section `[F]` |
| Register file | **256 KB per SM** = 64 K 32-bit registers; **255** per thread; **64 K per block** | whitepaper Table 4 `[F]`, Ampere Tuning Guide `[F]`, Programming Guide `[F-agent]` |
| Unified L1+shared | **192 KB per SM**; shared carveout ∈ {0, 8, 16, 32, 64, 100, 132, **164**} KB | Programming Guide 8.x `[F-agent]`, Ampere Tuning Guide `[F]` |
| Max shared **per block** | **163 KB** (1 KB of the 164 is reserved for system use) | Programming Guide 8.x `[F-agent]` |
| Shared memory banks | **32**, 32-bit each, 32 bits/clk each | Programming Guide `[F-agent]`, Best Practices `[F]` |
| Global coalescing granule | **32-byte transactions** | Best Practices Guide `[F]` |
| Occupancy ceilings | 2048 threads, **64 warps**, **32 blocks** per SM; 1024 threads per block | Programming Guide / Ampere Tuning Guide `[F]` `[F-agent]` |
| `cp.async` copy sizes | **4, 8 or 16 bytes**; requires sm_80 | PTX ISA `[F-agent]`, Programming Guide `[F-agent]` |
| Warp shuffle payload | 4 **or 8** bytes — `double` is supported | Programming Guide `[F-agent]` |

Verbatim, the two that constrain the design most:

> "Devices of compute capability 8.0 and 8.7 allow a single thread block to address up to
> 163 KB of shared memory… **Kernels relying on shared memory allocations over 48 KB per block
> are architecture-specific, and must use dynamic shared memory rather than statically sized
> shared memory arrays.** These kernels require an explicit opt-in by using
> `cudaFuncSetAttribute()` to set the `cudaFuncAttributeMaxDynamicSharedMemorySize`…"
> — CUDA C++ Programming Guide, Compute Capability 8.x → Shared Memory `[F-agent]`
> <https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html>

> "Note that the maximum amount of shared memory per thread block is smaller than the maximum
> shared memory partition available per SM. **The 1 KB of shared memory not made available to a
> thread block is reserved for system use.**" (same page) `[F-agent]`

> "**Shared memory has 32 banks that are organized such that successive 32-bit words map to
> successive banks. Each bank has a bandwidth of 32 bits per clock cycle.**" (same page,
> §16.4.3) `[F-agent]`

Derived numbers used throughout (my arithmetic from the above):

* **FP64 peak check.** 108 SM × 32 FP64 cores × 2 flop/FMA × 1.41 GHz = **9.75 TFLOP/s**.
  Matches NVIDIA's 9.7. So the "32 FP64 cores per SM" and the headline TFLOP/s are the same
  fact and the clock is the boost clock.
* **L2 read bandwidth.** 5120 B/clk × 1.41 GHz = **7219 GB/s ≈ 7.2 TB/s**, i.e. **4.64×** HBM.
  This is the single most useful derived number in the section.
* **Machine balance.** 9.7 TFLOP/s ÷ 1555 GB/s = **6.24 flop/byte** on the vanilla FP64 pipe,
  and **12.54 flop/byte** on the DMMA pipe. Everything in §2 is measured against these two.
* **A warp's register file** is 32 × 255 × 4 B = 32,640 B = **2040 complex doubles**.
* **A block's register file** is 65,536 × 4 B = 262,144 B = **16,384 complex doubles** — but
  claiming it means one block per SM.
* **A block's shared memory** is 163 KiB = 166,912 B = **10,432 complex doubles**.

---

## 1. Target 1 — how a small batched 3D FFT should be organised on an A100

### 1.1 The five containers, and which volumes fit in each

Complex double = 16 bytes. The capacities from §0.1, expressed in complex doubles:

| container | capacity | in complex doubles |
|---|---|---|
| one thread's registers | 255 × 4 B = 1020 B | **63** |
| one warp's registers | 32,640 B | **2040** |
| one block's registers (⇒ 1 block/SM) | 262,144 B | **16,384** |
| one block's shared memory (max carveout, opt-in) | 166,912 B | **10,432** |
| one SM's shared memory | 167,936 B | 10,496 |
| L2 (whole device) | 41,943,040 B | **2,621,440** |
| HBM2 | 40 GiB | 2.68 × 10⁹ |

And the volumes:

| L | points `L^3` | bytes | KiB | ÷ warp regs | ÷ block regs | ÷ block shared | volumes that fit in L2 |
|---|---|---|---|---|---|---|---|
| **6** | 216 | 3,456 | 3.38 | 0.11 | 0.013 | 0.021 | 12,136 |
| **8** | 512 | 8,192 | 8.00 | 0.25 | 0.031 | 0.049 | 5,120 |
| **13** | 2,197 | 35,152 | 34.3 | **1.08** | 0.134 | 0.211 | 1,193 |
| **17** | 4,913 | 78,608 | 76.8 | 2.41 | **0.300** | **0.471** | 534 |
| **23** | 12,167 | 194,672 | 190.1 | 5.96 | **0.743** | **1.166** | 216 |
| **36** | 46,656 | 746,496 | 729.0 | 22.9 | 2.85 | 4.47 | 56 |
| **45** | 91,125 | 1,458,000 | 1424 | 44.7 | 5.56 | 8.74 | 28.8 |
| **64** | 262,144 | 4,194,304 | 4096 | 128.5 | 16.0 | 25.1 | 10.0 |

**Read this table as four structural regimes, and they are the whole of Target 1:**

* **Regime A — `L = 6`, `L = 8`: one volume per *warp*, in registers.** 216 and 512 complex
  doubles are 11 % and 25 % of a warp's register file. All three axes fuse trivially with
  nothing in shared memory at all except (optionally) as a transpose buffer. At `L = 8` each
  lane holds 16 complex doubles (32 registers) of data plus temporaries — comfortable against
  the 255 limit. A block of 8 warps does 8 volumes; a block of 32 warps (1024 threads) does 32.
* **Regime B — `L = 13`: one volume per warp *just* misses (108 % of the warp file), one
  volume per two warps fits.** 2197 complex doubles over 64 lanes is 34.3 complex = 69
  registers of data per lane. Fine. Or keep it in shared memory (21 % of a block's max) and
  use registers only for the codelet.
* **Regime C — `L = 17`: one volume per *block*, all three axes fused, and it fits in shared
  memory with room to spare.** 76.8 KiB is 47 % of the 163 KiB per-block maximum, or 77 % of
  the 100 KiB carveout. This is the sweet spot of the whole set: a single kernel reads the
  volume once, does x, y and z out of shared memory, writes it once. In registers it is 30 %
  of the SM register file — also viable (a block of 256 threads holds 19.2 complex doubles =
  39 registers of data per lane), and that route reaches the register file's much higher
  bandwidth. `L = 23` is the boundary: 190.1 KiB is **1.17× the per-block shared maximum**, so
  a 23³ volume does *not* fit in one block's shared memory, but it does fit in one block's
  *registers* (74 % of the SM file, 1024 threads × 11.9 complex = 24 registers of data each).
* **Regime D — `L = 36, 45, 64`: no single-block option exists.** 729 KiB, 1.39 MiB and 4 MiB
  are 2.85×, 5.6× and 16× a block's entire register file. These must be multi-block and
  multi-pass, and the design question moves to *how few passes* and *where the intermediate
  lives* (§2.4). The natural unit is a **plane**: a `36×36` plane is 20.25 KiB, a `45×45` plane
  31.6 KiB, a `64×64` plane 64 KiB — all comfortably inside one block's shared memory, so the
  two contiguous-ish axes can be fused per plane and only the third axis needs a second pass.

**Nothing in this set fits in one *thread's* registers** (63 complex doubles; the smallest
volume is 216). So the cuFFTDx "thread-level FFT" model (§4) applies to a *line*, never to a
volume: a 6-, 8-, 13-, 17- or 23-point line is 6…23 complex doubles = 12…46 registers, all
comfortably per-thread; a 36-, 45- or 64-point line is 72, 90 or 128 registers of pure data,
which is where the 255 ceiling starts to bite (§6.5).

### 1.2 The parallelism is the batch, and small `B` cannot fill the GPU

The A100 has 108 SMs and admits at most 32 blocks and 64 warps per SM `[F]`. So:

* **Full block-level occupancy needs 108 × 32 = 3456 concurrent blocks** (and enough
  registers/shared memory to actually host them — a 17³-volume-in-shared block using the 100 KB
  carveout hosts **one** block per SM, so 108 blocks is the whole machine's worth of them).
* **One volume per block** ⇒ blocks = `B`. So *at `L = 17` with `B = 8` you occupy 8 of 108
  SMs.* At `L = 36` with `B = 4` you occupy 4.

This is the single biggest difference from the CPU phase, and it changes what "batched" means.
On the Xeon the panel scored `B = 1` and `B = 64…32768` separately and both were meaningful.
On an A100:

| L | `B` at which one-volume-per-block first fills all 108 SMs | `B` for 3456 blocks |
|---|---|---|
| 6 | 108 | 3456 |
| 8 | 108 | 3456 |
| 13 | 108 | 3456 |
| 17 | 108 | 3456 (won't fit — shared-limited to ~108–216) |
| 23 | 108 | register-limited to 108 |
| 36 | must split the volume across blocks | — |
| 45 | must split | — |
| 64 | must split | — |

For `L ≥ 36` you split the volume: one **plane** per block gives `B·L` blocks (`B = 4` at
`L = 36` is 144 blocks — enough to cover the machine once), and one **pencil-slab** per block
gives more still. **So the large geometries have a structural advantage at small batch and a
disadvantage at large batch (more passes), and the small geometries the reverse.** Expect the
per-geometry winners to differ in kind, not just in tuning.

### 1.3 The contract's layout is already coalesced for `L ≤ 17`, and that is a gift

The CPU contract's layout (`PANEL_BRIEF.md`) is

```
element (b, x, y, z) at index ((b*L + x)*L + y)*L + z
```

so **`z` is contiguous, `y` has stride `L`, `x` has stride `L²`, and each volume is one
contiguous `L³·16`-byte block.** Global coalescing on an A100 is measured in 32-byte
transactions:

> "the concurrent accesses of the threads of a warp will coalesce into a number of transactions
> equal to the number of 32-byte transactions necessary to service all of the threads of the
> warp." — CUDA C++ Best Practices Guide `[F]`
> <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>

A 16-byte complex double means **two threads per 32-byte transaction**, so a warp reading 32
consecutive elements issues 16 transactions and wastes nothing. Therefore:

* **For `L ≤ 17`, where a whole volume lives in one block, there is no strided-access problem
  in global memory at all.** The block streams its own contiguous `L³·16` bytes linearly —
  3456 B at `L = 6`, 78,608 B at `L = 17` — with a flat `tid`-to-offset map. The 3D transpose
  problem is thereby *moved entirely into shared memory*, where it becomes a bank-conflict
  problem (§6.2) rather than a DRAM-efficiency problem. This is a genuinely easier problem than
  the one the GPU FFT literature is written about, and it is a direct consequence of the volumes
  being small.
* **For `L = 36, 45, 64` the problem returns**, because a pass over the `x` axis touches
  elements `L²` apart. The remedy is the standard one and is the same one VkFFT uses (§3.4):
  never index a warp along the strided axis. Read a `(x, z)` or `(y, z)` slab *coalesced along
  `z`*, transpose it in shared memory, transform along the strided axis out of shared memory,
  transpose back, write coalesced along `z`.
* A useful corollary for the big sizes: at `L = 36` a single `z`-line is 576 B and a single
  `y`-line has stride 576 B. A warp assigned one lane per `y` therefore issues 32 separate
  32-byte transactions to use 512 bytes — a 2× read amplification. At `L = 64` a `y`-line has
  stride 1024 B and the amplification is the same 2×; at `L = 45`, stride 720 B, likewise. The
  amplification is 2× and not worse only because a complex double is exactly half a
  transaction. **This is the single number that justifies the shared-memory transpose for
  `L ≥ 36` and says it is worth at most 2×, not more.**

### 1.4 What the two best libraries actually choose, read from the source

VkFFT's block-shaping code is the most concrete published statement of "how many small FFTs
per block" that exists, and it is in this repository. From
`ext/src/VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h`
`[F-src]`:

```c
pfUINT maxBatchCoalesced = app->configuration.coalescedMemory / axis->specializationConstants.complexSize;
axis->groupedBatch = maxBatchCoalesced;
...
pfUINT maxSingleSizeStrided = (app->configuration.coalescedMemory > complexSize)
        ? allowedSharedMemory / (app->configuration.coalescedMemory)
        : allowedSharedMemory / complexSize;
```

and, for NVIDIA specifically, from `vkFFT_AppManagement/vkFFT_InitializeApp.h:496-504` `[F-src]`:

```c
case 0x10DE://NVIDIA
        app->configuration.coalescedMemory = (app->configuration.halfPrecision) ? 64 : 32;
                //the coalesced memory is equal to 32 bytes between L2 and VRAM.
        app->configuration.useLUT = (app->configuration.doublePrecision || ...) ? 1 : -1;
        app->configuration.warpSize = 32;
        app->configuration.registerBoostNonPow2 = 0;
        app->configuration.registerBoost = 4;
```
plus `app->configuration.aimThreads = 128;` and `app->configuration.numSharedBanks = 32;`
(same file, lines 1230 and 1231).

Decoded for our case — FP64, so `complexSize = 16`, `coalescedMemory = 32`:

1. **`maxBatchCoalesced = 32/16 = 2`.** VkFFT's *floor* on how many adjacent sequences a block
   processes together is 2 for complex double: the minimum needed for one 32-byte transaction.
   For complex float it is 4. **Batching adjacent sequences is not an optimization in VkFFT, it
   is the coalescing mechanism**, exactly as §07 §1.2 quoted from the poster — but the source
   gives the number, and for FP64 the number is only 2.
2. **The strided axes get half the shared-memory budget of the contiguous one.** Because
   `coalescedMemory (32) > complexSize (16)`, the strided-axis maximum sequence length is
   `sharedMem/32`, not `sharedMem/16`. VkFFT deliberately reserves 32 bytes per column of its
   shared tile so that the global access stays coalesced. **A hand-written kernel for a small
   cube does not need to pay this**, because §1.3 says the volume is contiguous and there is no
   strided global access to protect.
3. **`registerBoost = 4` but `registerBoostNonPow2 = 0`.** VkFFT's "use the register file
   instead of shared memory, it is 4× bigger" trick — the asymmetry §07 §4.1 made its headline
   — is **disabled for non-power-of-two lengths**. So of our eight geometries it applies only
   to `L = 8` and `L = 64`. For `L = 6, 13, 17, 23, 36, 45` VkFFT itself does not take the
   register route. That is a caution, not a prohibition: VkFFT's constraint is that its
   *generated* code must handle every length, and ours does not.
4. **`aimThreads = 128`**, and on NVIDIA there is an explicit clamp
   (`if (app->configuration.vendorID == 0x10DE)`) that halves the block's `x` extent while
   `axisBlock[0]*axisBlock[1] >= 2*aimThreads` — i.e. **VkFFT targets 128 and tolerates up to
   256 threads per block on NVIDIA**, not 1024. `[F-src]`
5. **Shared memory caps the per-block batch, explicitly:**
   `while ((axisBlock[1] * (fftDim/registerBoost)) > maxSequenceLengthSharedMemory) axisBlock[1] /= 2;`
6. **`axisSwapped`**: when the sequence is even or the thread count small, VkFFT swaps the
   block's x and y extents so that the *batch* index becomes the fast thread dimension. That is
   the "make the batch the vector lane" move from §04/§08, expressed as a block-shape swap.

The other library, cuFFTDx, states the same model in its docs and is covered in §4.

### 1.5 The synthesis for Target 1

| L | recommended primary structure | volume lives in | global passes | blocks |
|---|---|---|---|---|
| **6** | one volume per **warp**, 3 axes fused in registers; 8–32 volumes per block | registers (11 % of warp file) | **1 read + 1 write** | `B/warps_per_block` |
| **8** | one volume per **warp**, 3 axes fused in registers | registers (25 %) | **1 + 1** | as above |
| **13** | one volume per **2 warps** in registers, or per block in shared (21 %) | either | **1 + 1** | `B` or `B/k` |
| **17** | one volume per **block**: shared (47 % of 163 KiB) or registers (30 % of SM file) | shared or registers | **1 + 1** | `B` |
| **23** | one volume per **block in registers** (74 % of SM file); shared does **not** fit (117 %) | registers | **1 + 1** | `B` |
| **36** | plane-per-block: fuse `y,z` in a 20.25 KiB plane, second kernel for `x` | shared per plane | **2 + 2** | `B·36` then `B·36` |
| **45** | as 36 (plane = 31.6 KiB) | shared per plane | **2 + 2** | `B·45` ×2 |
| **64** | as 36 (plane = 64 KiB), or 3 passes with L2 blocking across the batch | shared per plane | **2 + 2** | `B·64` ×2 |

The "1 read + 1 write" column is the whole point: §2 shows that column *is* the performance.

---

## 2. Target 2 — the roofline, and it is unambiguous

### 2.1 The two machine balances

From §0.1: **6.24 flop/byte** on the vanilla FP64 pipe (9.7 TFLOP/s ÷ 1555 GB/s) and
**12.54 flop/byte** on the DMMA pipe (19.5 ÷ 1.555). A kernel whose arithmetic intensity is
below the balance is bandwidth-bound and its runtime is set by bytes, full stop.

Minimum traffic for our problem, out-of-place, one pass: read `16 B` and write `16 B` per point
= **32 bytes per point**, independent of `L`. At 1555 GB/s that is a hard floor of

> **20.58 picoseconds per complex-double point.**

Everything below is expressed against that number, because it is the number to beat.

### 2.2 Arithmetic intensity per geometry — the butterfly formulation

Flop convention: the standard `5·N·log2 N` for a length-`N` complex FFT (benchFFT's
convention, used by every paper in §7), so a 3D `L³` transform is
`3·L²·(5·L·log2 L) = 15·L³·log2 L` flops, i.e. **`15·log2 L` flops per point**.

| L | flop/point | AI (flop/B) | × below the FP64 balance | × below the DMMA balance |
|---|---|---|---|---|
| 6 | 38.8 | **1.21** | 5.15× | 10.3× |
| 8 | 45.0 | **1.41** | 4.44× | 8.9× |
| 13 | 55.5 | **1.74** | 3.60× | 7.2× |
| 17 | 61.3 | **1.92** | 3.26× | 6.5× |
| 23 | 67.9 | **2.12** | 2.94× | 5.9× |
| 36 | 77.5 | **2.42** | 2.57× | 5.2× |
| 45 | 82.4 | **2.57** | 2.42× | 4.8× |
| 64 | 90.0 | **2.81** | 2.22× | 4.4× |

**Every geometry is bandwidth-bound by 2.2× to 5.2× on the ordinary FP64 pipe**, and the
margin *shrinks* with `L` — the small cubes are the most bandwidth-bound of all. Two direct
consequences:

1. **Arithmetic optimisation has no headroom whatsoever in the batched regime.** You could
   halve the flop count of a `6³` kernel and change nothing. Everything §01–§04 of this corpus
   is about — Winograd modules, PFA twiddle elimination, 3-multiply complex products — buys
   nothing here *unless* it changes the number of memory passes or the register/shared
   footprint (which it does: fewer live values ⇒ more volumes resident ⇒ possibly fewer passes).
2. **Conversely you can afford an enormous amount of extra arithmetic for free.** At `L = 6`
   the budget before you hit the FP64 pipe is `6.24 × 32 = 200` flops per point against a
   butterfly's 38.8 — a **5.15× arithmetic allowance**. That allowance is what §5 spends.

The corresponding time floors, per transform, at the 1555 GB/s HBM roof:

| L | points | bytes moved (min) | **floor per transform** | floor at the 7.2 TB/s L2 roof |
|---|---|---|---|---|
| 6 | 216 | 6.9 kB | **4.4 ns** | 0.96 ns |
| 8 | 512 | 16.4 kB | **10.5 ns** | 2.3 ns |
| 13 | 2,197 | 70.3 kB | **45.2 ns** | 9.7 ns |
| 17 | 4,913 | 157 kB | **101 ns** | 21.8 ns |
| 23 | 12,167 | 389 kB | **250 ns** | 53.9 ns |
| 36 | 46,656 | 1.49 MB | **960 ns** | 207 ns |
| 45 | 91,125 | 2.92 MB | **1.88 µs** | 404 ns |
| 64 | 262,144 | 8.39 MB | **5.39 µs** | 1.16 µs |

Compare against the CPU panel's *winning* single-core numbers on the Xeon Gold 5218
(`bench/geom/results/panel_r5/leaderboard.txt`, in-tree measurement, not a citation): `L = 6`
0.219 µs, `L = 8` 0.570 µs, `L = 17` 15.2 µs, `L = 36` 120 µs. **The A100's bandwidth floor is
50×, 54×, 150× and 125× below those.** That is the size of the prize and also the size of the
trap: at these floors, *anything* the harness does badly — a launch, a synchronise, an event —
dominates the measurement (§7).

### 2.3 What real FFT kernels actually achieve on an A100, in FP64

**The primary source is in this repository.** VkFFT ships its own committed A100 FP64
benchmark plot at `ext/src/VkFFT/benchmark_plot/fp64_cuda_a100.png` (VkFFT v1.3.4) `[F-src]`,
titled *"FP64 batched 1D C2C"*, plotting **achieved bandwidth in GB/s** against FFT length
2…4096 for VkFFT and cuFFT, split by algorithm: `radix(2-13)`, `Rader(17-4096)` / `Rader(17-127)`,
and Bluestein.

I read the marker positions out of that PNG programmatically (axis calibration from the tick
marks: `value = (7919 − y_px)·1400/7212` GB/s, `length = (x_px − 1439)·250/967`). The numbers
below are therefore **my measurement of VkFFT's published plot**, not printed figures — but the
plot is the author's own and the calibration is exact:

| series | lengths ≤ 67 | lengths ≤ 130 | whole plot (≤ 4096) |
|---|---|---|---|
| **VkFFT radix(2–13)** | 1253–1296, median **1277** | 1253–1297, median 1274 | median 1249 |
| **VkFFT Rader(17–4096)** | 704–1277, median **1226** | 562–1278, median 1154 | median 744 |
| **cuFFT Rader(17–127)** | 459–1275, median **629** | 314–1275, median 564 | median 368 |
| cuFFT radix(2–13) | top band coincides with VkFFT's at small length; drops to bands at ≈630 and ≈420 for longer lengths | | |

**Three conclusions, and they set the whole GPU phase's expectations.**

1. **≈1280 GB/s is what a good FP64 FFT kernel achieves on an A100 — 82 % of the 1555 GB/s
   peak.** That is the realistic roof. Multiply every floor in §2.2 by 1/0.82 = 1.22 to get a
   *credible* target: `L = 6` ≈ 5.4 ns, `L = 17` ≈ 123 ns, `L = 36` ≈ 1.17 µs per transform.
2. **At radix-friendly small lengths cuFFT is not behind.** In the left-hand region of the plot
   (lengths below ≈100) cuFFT's `radix(2-13)` markers sit in the same top band as VkFFT's. So
   do not expect the 1D story to hand you an easy 2× over cuFFT at `L = 6, 8, 13`. Where cuFFT
   collapses is at **primes**: its `Rader(17-127)` median is 629 GB/s against VkFFT's 1226 for
   the same lengths — **a factor of 1.95 at exactly our `L = 17` and `L = 23`.**
3. **cuFFT's lower bands are almost exactly 1/2 and 1/3 of the top band** (1277/2 = 639 vs the
   observed ≈630; 1277/3 = 426 vs ≈420). *(This next step is my inference, §11.)* The two
   benchmark drivers compute bandwidth differently: VkFFT's multiplies the buffer size by
   `4 × Σ numAxisUploads`, i.e. its *actual* device traffic, while cuFFT's multiplies by a fixed
   4 — the *minimum* traffic
   (`ext/src/VkFFT/benchmark_scripts/{vkFFT,cuFFT}_scripts/src/sample_1001_*` `[F-src]`).
   So a cuFFT point at 1/2 or 1/3 of the top band is a cuFFT plan doing **2 or 3 global memory
   passes where VkFFT does 1**. The whole visible gap is pass count. This is the concrete,
   quantitative version of §07 §1.6's fusion rule, on our exact part, in our exact precision.

**Caveat that matters.** The plot is **1D batched**, not 3D. VkFFT's *3D* benchmark
(`sample_1003_benchmark_VkFFT_single_3d_2_512.cpp` `[F-src]`) uses a single volume with no
batch dimension at all (`bufferSize = 16·size[0]·size[1]·size[2]`), so **there is no published
batched-small-3D FP64 measurement anywhere I could find, from anyone.** That is the gap this
project's GPU phase fills, and it is worth saying in the eventual write-up.

### 2.4 L2 is a second roof, and for the small geometries it is the *operative* one

40 MiB of L2 reading at 7.2 TB/s (§0.1) is 4.64× HBM. The whole in+out working set is
`B·L³·32` bytes, so it fits in L2 when `B·L³ < 1.31 × 10⁶` points:

| L | `B` at which in+out **exceeds** L2 | `B` needed for 108 blocks (1 vol/block) | is the GPU-filling batch L2-resident? |
|---|---|---|---|
| 6 | 6,069 | 108 | **yes, by 56×** |
| 8 | 2,560 | 108 | **yes, by 24×** |
| 13 | 597 | 108 | **yes, by 5.5×** |
| 17 | 267 | 108 | **yes, by 2.5×** |
| 23 | 108 | 108 | **exactly at the boundary** |
| 36 | 29 | 3 (plane-per-block: `B·36 ≥ 108`) | yes at the filling batch |
| 45 | 15 | 3 | yes |
| 64 | 5 | 2 | yes |

**This is a benchmark-design hazard before it is an optimisation.** If the harness runs the
same two buffers in a tight repeat loop at `L = 6, B = 512`, the working set is 1.7 MiB, every
iteration after the first hits L2, and the measured "bandwidth" will be an L2 number that no
real application would see. A kernel tuned on that measurement is tuned for the wrong roof.
§8 fixes this by mandating a batch large enough to overflow L2 by a wide margin, and by
*also* reporting the small-batch (L2-resident) point separately and labelling it as such.

**And it is a real optimisation for `L ≥ 36`.** Those geometries need 2 or more passes, and a
pass whose intermediate is in L2 costs 1/4.6 of a pass that goes to HBM. At `L = 64`, two
volumes (in + out, 8 MiB) fit in L2 five times over, so a batch processed in **chunks of ~4
volumes** can do all three axis passes with only the first read and last write touching HBM.
That is the GPU form of §08 §1.9's "cache-block across the batch into L2", which three
independent sources endorsed for the CPU. The A100 even gives you a lever for it:

> "**A100 allows L2 cache to be set-aside for persistent accesses in 1/16th increments
> (2.5 MB)**. Persistent accesses have prioritized use of this set-aside portion of L2 cache.
> Normal or streaming accesses to global memory can only utilize this portion of L2 when it is
> unused by persistent accesses." — A100 whitepaper `[F]`

with the API being `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, …)` plus a per-stream
`accessPolicyWindow` `[F-agent]`. Note the whitepaper's own restriction: "when the GPU is
configured in Multi-Instance GPU (MIG) mode, the L2 cache set-aside functionality is disabled"
`[F]` — our part reports `MIG M.: Disabled` `[M]`, so it is available.

### 2.5 What "bandwidth-bound" does **not** mean

It does not mean the kernel will reach 1280 GB/s by itself. Three things stop small-transform
kernels from reaching the bandwidth roof, and they are what §6 is about:

* **Not enough concurrency in flight.** A bandwidth-bound kernel needs enough outstanding loads
  to cover HBM latency. With one volume per block and `B = 108` you have 108 blocks — a single
  wave — and no second wave to hide anything behind. This is the GPU analogue of §08 §1.1's
  "the single-core ceiling is a concurrency limit, not a DRAM limit".
* **Launch overhead**, which at these floors is enormous (§6.6): 4.4 ns of work per `L = 6`
  transform against a documented **1–5 µs** per-launch hardware/driver cost `[F-agent]`.
* **Shared-memory bank conflicts**, which convert the 3D transpose into a serialised access and
  can cost a factor equal to the conflict degree (§6.2).

---

## 3. Target 3 — cuFFT and VkFFT, the two baselines we have to beat

### 3.1 The call: `cufftPlanMany` for a batched 3D c2c

Verbatim from the cuFFT documentation (self-identifying as cuFFT 13.3) `[F-agent]`
<https://docs.nvidia.com/cuda/cufft/index.html>:

> `cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride,
> int idist, int *onembed, int ostride, int odist, cufftType type, int batch);`
>
> "Creates a FFT plan configuration of dimension `rank`, with sizes specified in the array `n`.
> The `batch` input parameter tells cuFFT how many transforms to configure. With this function,
> batched plans of 1, 2, or 3 dimensions may be created."

and the addressing formula the baseline harness must match:

> "An element of coordinates [z][y][x] in signal number b in the batch will be associated with
> the following addresses in the memory:"
> `input[ b*idist + ((x*inembed[1] + y)*inembed[2] + z)*istride ]` `[F-agent]`

For our contract (`rank = 3`, `n = {L,L,L}`, contiguous, batch-slowest) the correct call is
`cufftPlanMany(&p, 3, n, NULL, 1, L*L*L, NULL, 1, L*L*L, CUFFT_Z2Z, B)` — with the documented
warning:

> "Please note that behavior of cufftPlanMany function when inembed and onembed is NULL is
> different than corresponding function in FFTW library fftw_plan_many_dft." `[F-agent]`

Two more documented facts the baseline needs:

> "In the worst case, the cuFFT Library allocates space for `8*batch*n[0]*..*n[rank-1]`
> cufftComplex or cufftDoubleComplex elements" `[F-agent]`

so at `L = 17, B = 27000` a worst-case cuFFT workspace is 8× a 2 GiB buffer = 16 GiB. Check
`cufftGetSize` before allocating, and be prepared for cuFFT to fail on batch sizes our own
kernel handles fine. And:

> "Results produced by the cuFFT library are deterministic (ie, bitwise reproducible) as long
> as the following are kept constant between runs: plan input parameters, cuFFT version, and
> GPU model." `[F-agent]`

which means the repeatability clause of the CPU brief (§8) is satisfiable by the baseline too.

### 3.2 What cuFFT documents about its own weak spots — and one correction to §07

The key paragraph, verbatim `[F-agent]`:

> "The cuFFT Library implements the following building blocks: radix-2, radix-3, radix-5, and
> radix-7. Hence the performance of any transform size that can be factored as
> 2^a × 3^b × 5^c × 7^d … is optimized in the cuFFT library. **There are also radix-m building
> blocks for other primes, m, whose value is < 128.** When the length cannot be decomposed as
> multiples of powers of primes from 2 to 127, Bluestein's algorithm is used. Since the
> Bluestein implementation requires more computations per output point than the Cooley-Tukey
> implementation, the accuracy of the Cooley-Tukey algorithm is better."

**This corrects §07 §1.2.** §07 quotes VkFFT's SC22 poster saying "cuFFT does not use Rader's
algorithm in FP32 and switches to Bluestein's algorithm for primes after 17", and left the
impression that cuFFT falls off a cliff at 17. NVIDIA's own documentation says the opposite:
**cuFFT has a dedicated radix-m building block for every prime below 128, so 13, 17, 23, 41, 43
and 89 are all handled by a native radix kernel, not by Bluestein.** Bluestein starts at 131.
Both statements can be true — VkFFT's poster also says cuFFT "implements it as a **direct
matrix multiplication**" `[F-agent]` (SC22 poster) — i.e. cuFFT's radix-17 *is* a dense 17×17
matvec, which is not Rader, which is why the poster's author describes cuFFT as "not using
Rader". Cite both; the operative fact for us is the third one:

> **cuFFT's radix-17 and radix-23 are dense matrix multiplications, and so was the CPU panel's
> winning `L = 17` kernel.** We are not proposing something exotic; we are proposing to do
> better at something a vendor library already does this way.

The other documented cuFFT weak spots, all verbatim `[F-agent]`:

* Prime sizes cost workspace: "In particular powers of 2 are very efficient in terms of
  temporary storage. **Large prime numbers, however, use different algorithms and may need up to
  the eight times that of a similarly sized power of 2.**"
* Mixed factorizations are penalised: "A transform of size 2^n or 3^n will usually be faster
  than one of size 2^i × 3^j even if the latter is slightly smaller, due to the composition of
  specialized paths." — **directly relevant to `L = 6 = 2·3`, `L = 36 = 4·9`, `L = 45 = 9·5`.**
* Precision: "**Single precision transforms require less bandwidth per computation than double
  precision transforms.**" and "Transforms of lower precision have higher performance." No
  quantified FP64:FP32 ratio is published anywhere in the doc.
* Callbacks — cuFFT's own fusion mechanism — are size-restricted and shared-memory-starved:
  "**Callback functions are not supported on transforms with a dimension size that does not
  factor into primes smaller than 127.** Callback functions on plans whose dimensions' prime
  factors are limited to 2, 3, 5, and 7 can safely call `__syncthreads()`. On other plans,
  results are not defined." and "`cufftXtSetCallbackSharedSize()` … **The maximum allowable
  amount of shared memory is 16K bytes.**" Legacy callbacks are deprecated since CUDA 11.4 and
  replaced by LTO callbacks (CUDA 12.6 Update 2), with "an increase in planning time (which
  could be in the order of hundreds of milliseconds)".
* The only documented size ceilings are in the minimal-workspace mode: "Transforms of type C2C
  are supported with sizes up to 4096 in any dimension." / "Transforms of type **Z2Z** are
  supported with sizes up to **2048** in any dimension."
* **cuFFT does not document how much shared memory its own kernels use, and NVIDIA publishes no
  single-GPU A100 FP64 batched-FFT performance numbers at all** — the cuFFT product page's two
  charts are FP32 multi-GPU, and the only NVIDIA A100 FFT figures found anywhere were cuFFTMp
  multi-node ("cuFFTMp reaching over 1.8 PFlop/s, more than 70% of the peak machine bandwidth"
  `[F-agent]` <https://developer.nvidia.com/blog/multinode-multi-gpu-using-nvidia-cufftmp-ffts-at-scale/>).
  So the baseline numbers for our sizes do not exist and we will be generating them.

### 3.3 VkFFT's published FP64 A100 comparison — the numbers, and the methodology

The peer-reviewed record, which neither §07 nor §08 has:

**D. Tolmachev, "VkFFT — A Performant, Cross-Platform and Open-Source GPU FFT Library",
*IEEE Access* **11**, 12039–12058 (2023), DOI `10.1109/ACCESS.2023.3242240`.** `[F-agent]`
Bibliographic record verified against Crossref
(<https://api.crossref.org/works/10.1109/ACCESS.2023.3242240>) and OpenAlex.
**Full text `[UNVERIFIED — could not fetch]`** — every IEEE Xplore route timed out. Do not
quote from inside the article. A companion exists: Tolmachev, "VkFFT and beyond — a platform
for runtime GPU code generation", IWOCL 2023, DOI `10.1145/3585341.3585357` `[F-agent]`.

The published *numbers* we can use come from the SC22 poster and the repository. Methodology
first, verbatim from the README `[F-agent]`
<https://raw.githubusercontent.com/DTolm/VkFFT/master/README.md>:

> "The test configuration below takes multiple 1D FFTs of all lengths from the range of 2 to
> 4096, **batch them together so the full system takes from 500MB to 1GB of data** and perform
> multiple consecutive FFTs/iFFTs (-vkfft 1001 key). After that time per a single FFT is
> obtained by averaging the result. Total system size will be divided by the time taken by a
> single transform upload+download, resulting in the estimation of an achieved global
> bandwidth. The GPUs used in this comparison are Nvidia A100 and AMD MI250. The performance was
> compared against Nvidia cuFFT (CUDA 11.7 version) and AMD rocFFT (ROCm 5.2 version) libraries
> **in double precision**"

**500 MB–1 GB of data is 12–24× the 40 MiB L2, so this is a genuine HBM measurement**, which is
exactly what §2.4 warns you to check. And the crucial calibration sentence, from the SC22
poster `[F-agent]`
<https://sc22.supercomputing.org/proceedings/tech_poster/poster_files/rpost143s3-file2.pdf>:

> "Total system size is divided by the time taken by a single transform, resulting in the
> achieved bandwidth. **The peak global memory bandwidth of A100 is ~1.3TB/s.**"
>
> "We compare the VkFFT performance against Nvidia's cuFFT on Nvidia A100 HPC GPU
> (**40GB, 250W, P0 profile, CUDA 11.7**)"

Two things follow. First, **the author of the fastest GPU FFT library treats the A100's usable
peak as ~1.3 TB/s, not the datasheet's 1.555** — an 84 % achievable fraction, which is the
number our roofline should actually be drawn at. Second, his test part is the *same SKU as
ours*: 40 GB, 250 W PCIe.

Against that ~1.3 TB/s, the plot measurements of §2.3 read as: **VkFFT's radix path ≈98 % of
achievable peak; cuFFT's radix path ≈48 %; cuFFT's prime path ≈48 % of VkFFT's.** And the
poster's own conclusion is appropriately modest:

> "VkFFT matches in performance with Nvidia's cuFFT library on Nvidia GPUs **for small
> sequences** and outperforms it on big ones." `[F-agent]`

**A dissenting measurement, and it must be reported.** TurboFFT (§5.7) benchmarks the same
three libraries on the same A100-PCIE-40GB and reports the opposite ordering in FP64:

> "For double precision, **VkFFT has an average overhead of 11% compared to cuFFT**, with
> performance varying between 3% faster and 51% slower." — Wu et al., arXiv:2405.02520
> `[F-agent]` <https://arxiv.org/html/2405.02520v1>

The two are not reconcilable from the published information: VkFFT's sweep is all lengths
2–4096 at 500 MB–1 GB, TurboFFT's is a fixed problem size at selected lengths, and the CUDA
versions differ. **Treat "which library is faster in FP64 on A100" as genuinely open, measure
both yourself, and do not build a strategy on either claim.** What both agree on is the next
subsection.

### 3.4 The one thing every source agrees cuFFT does badly: small problems

> "**The state-of-the-art closed library fails to fully utilize the SMs on A100 for problem
> sizes less than 2 MB** … The inefficient kernel parameter cannot fully utilize all 108
> streaming multiprocessors in a A100 GPU."
> — Wu et al., TurboFFT, arXiv:2412.05824 `[F-agent]` <https://arxiv.org/html/2412.05824v1>
>
> "For both FP32 and FP64, TurboFFT is faster than cuFFT for problem sizes within 2 MB.
> TurboFFT outperforms cuFFT by **40% to 200%** on signal length N = 2^14." (same) `[F-agent]`

**2 MB is exactly our regime.** A `36³` volume is 746 KB; an `L = 17` batch of 25 volumes is
2 MB. Every one of our geometries at a modest batch is inside the window where a published,
peer-reviewed measurement says cuFFT leaves the machine idle. That is the opening.

### 3.5 VkFFT's design, read from the source — what to actually copy

§07 §1.2 and §08 §5.7 already cover VkFFT's *policy* (single upload while it fits in shared
memory; four-step beyond; never transpose, coalesce neighbouring sequences instead; LUT
twiddles precomputed in higher precision). Here is what reading the code adds, and one place
where the code contradicts the README.

**(a) The single-upload capacity rule, from VkFFT's own API guide** `[F-agent]`
<https://raw.githubusercontent.com/DTolm/VkFFT/master/documentation/VkFFT_API_guide.pdf> §2.5.2:

> "To estimate if your sequence size is single upload or not, divide the amount of available
> shared memory (48KB - Nvidia GPUs with Vulkan/OpenCL API, 64KB - AMD GPUs, **100KB - Nvidia
> GPUs in CUDA API**) by the complex size used for calculations (4 byte - half precision, 8 byte
> - single precision, **16 byte - double precision**, 32 byte - double-double). For 64KB of
> shared memory, we get 8192 as max single upload single-precision non-strided FFT, **4096 for
> double precision. For strided axes (H and D parts of the layout) these numbers have to be
> divided by 4 and 2 respectively to achieve coalescing**, resulting in 2048 length for single
> upload in both precisions."

Note: **VkFFT asks for a 100 KB carveout on NVIDIA, not the 164 KB maximum.** So the strongest
GPU FFT library in existence leaves 92 KB of the unified cache as L1 rather than claiming it
all for shared memory. That is a data point about the right carveout to request (§6.7), and it
is the opposite of what a naive reading of "163 KB is available" suggests. It also means
VkFFT's own FP64 single-upload ceiling on A100 is `100·1024/16 = 6400` complex doubles
non-strided and 3200 strided — comfortably above every `L` in our set, so **VkFFT is doing all
eight of our per-axis transforms as single uploads.** Its 3 passes for a 3D transform are one
per axis, not one per split.

**(b) The coalescing constants, and one undocumented hazard that hits `L ≥ 36`.** From the
same guide `[F-agent]`:

> "For Nvidia and AMD we have to coalesce 32-byte memory requests… **There exist undocumented
> issues for Nvidia and AMD, when bandwidth drops when code tries to do distant, but coalesced
> memory accesses, i.e. data are still grouped in 32-byte transactions but target addresses
> with > 2^18 bytes between them**… It is possible to regain full bandwidth by switching to
> 128-byte memory coalescing on Nvidia."

`2^18 = 262,144 bytes`. **Our volumes cross that threshold between `L = 23` (194,672 B) and
`L = 36` (746,496 B).** So any kernel for `L = 36, 45, 64` whose warps span *different volumes
of the batch* is in the regime VkFFT's author warns about, and the documented remedy is to make
each thread move 128 bytes (8 complex doubles) instead of 32. For `L ≤ 23` the whole batch
stride is under 2^18 and the hazard does not arise. This is a specific, actionable, per-geometry
warning that appears nowhere else in this corpus.

**(c) Block shaping.** Already given in §1.4: `maxBatchCoalesced = 2` for FP64,
`aimThreads = 128`, an NVIDIA-specific clamp at ~256 threads, shared memory capping the
per-block batch, and `axisSwapped` putting the batch index in the fast thread dimension.

**(d) Shared-memory padding, from the code.** `vkFFT_SharedMemory.h:49-51` `[F-src]`:

```c
sharedStrideBankConflictFirstStages = ((fftDim > numSharedBanks/2) && isPow2(fftDim))
     ? (fftDim/registerBoost) * (numSharedBanks/2 + 1) / (numSharedBanks/2)
     : fftDim/registerBoost;
sharedStrideReadWriteConflict = (numSharedBanks/2 <= localSize[1])
     ? fftDim/registerBoost + 1
     : fftDim/registerBoost + (numSharedBanks/2)/localSize[1];
```

and for the strided/grouped kernels (`case 1: case 2:`):

```c
pfUINT shift = (fftDim < numSharedBanks/2) ? (numSharedBanks/2)/fftDim : 1;
sharedStrideReadWriteConflict = (axisSwapped && (localSize[0] % 4 == 0))
     ? localSize[0] + shift : localSize[0];
```

With `numSharedBanks = 32`, the rules are: **pad a power-of-two row length by 17/16**; pad every
row length by **+1** when the block's second dimension is ≥ 16; and for short sequences pad by
`16/fftDim`. §6.2 works out why `numSharedBanks/2 = 16` is the right divisor for complex
*float* and the wrong one for complex *double*, and what the correct FP64 rule is.

**(e) The radix set — the README is wrong, or at least badly out of date, and both §07 and §08
repeat it.** `ext/src/VkFFT/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixKernels.h`
is a single `switch (radix)` whose cases are `[F-src]`:

```
case 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32
```

The README says "Radix-2/3/4/5/7/8/11/13 FFT" `[F-agent]`, which §07 §1.2 and §08 §5.7 both
quote. **The shipped code has hand-written straight-line kernels for every radix from 2 to 16
and for 32.** For our geometries that means VkFFT executes:

| L | VkFFT's likely decomposition | native kernels used |
|---|---|---|
| 6 | one radix-6 stage | `case 6` |
| 8 | one radix-8 stage | `case 8` |
| 13 | one radix-13 stage | `case 13` |
| 17 | Rader (prime, no kernel) | — |
| 23 | Rader (prime, no kernel) | — |
| 36 | 6×6, or 12×3, or 4×9 | `case 6`/`12`/`3`/`4`/`9` |
| 45 | 15×3 or 9×5 | `case 15`/`9`/`5`/`3` |
| 64 | 8×8, 16×4, or 32×2 | `case 8`/`16`/`32` |

**So six of our eight geometries are a single native VkFFT radix stage or a product of two.**
That is a strong prior that `L = 6, 8, 13, 64` will be hard to beat and `L = 17, 23` will be
where the win is — the same shape of result as the CPU phase.

### 3.6 The `L = 17` opening, restated in VkFFT's own terms

VkFFT's Rader has two implementations, and the struct comment names them `[F-src]`
(`vkFFT_Structs.h:689`):

```c
int type; //0 - FFT, 1 - Direct multiplication
```

and the thresholds are set per vendor in `vkFFT_InitializeApp.h:1257-1291` `[F-src]`:

```c
fixMinRaderPrimeMult = 17;                 // direct-multiplication Rader from 17 up
case 0x10DE://NVIDIA
        fixMaxRaderPrimeMult = 89;
...
default:  fixMinRaderPrimeFFT = 17;        // FFT-convolution Rader from 17 up
fixMaxRaderPrimeFFT = 16384;
```

**Note that the code says `fixMaxRaderPrimeMult = 89` on NVIDIA, while the API guide says
"Default is vendor-specific (currently ~40)"** `[F-agent]`. §08 §5.7 quotes the ~40. The code
is the authority: on NVIDIA in v1.3.4 it is **89**, so both `L = 17` and `L = 23` are inside the
direct-multiplication window as well as the FFT-convolution window, and which one runs depends
on the plan.

And the direct-multiplication kernel is, structurally, *our CPU winner*. From
`vkFFT_RaderKernels.h`, `appendMultRaderStage` `[F-src]`:

```c
pfINT num_logical_subgroups = ... localSize[0] / ((stageRadix + 1) / 2);
...
temp_int.data.i = (stageRadix->data.i - 1) / 2;      // loop bound over conjugate pairs
```

`(P+1)/2` threads per transform and a loop over `(P-1)/2` pairs is **exactly a dense
conjugate-symmetric matrix-vector product**: for `P = 17`, nine threads, eight symmetric pairs
plus the DC term. The `L17_matrixsimd` entry that leads this project's CPU leaderboard by
5.37× over the best library (`bench/geom/results/panel_r5/leaderboard.txt`) is the same
algorithm, and VkFFT's own guide says of the alternative:

> "`fixMinRaderPrimeFFT` — start FFT convolution version of Rader … Better than direct
> multiplication version for almost all primes (**except small ones, like 17-23 on some
> GPUs**)." `[F-agent]`

**Two independent implementations — VkFFT's `type = 1` and cuFFT's "radix-m as a direct matrix
multiplication" — both reach for a dense matrix at exactly `P = 17…23`, and VkFFT's own
documentation says that is the right choice there.** The GPU phase does not have to discover
this. It has to do it better, and §5 says how.

---

## 4. Target 4 — cuFFTDx and the device-function model

**cuFFTDx is banned inside the competition kernels.** It is a library, and the rule that
forbade FFTW and MKL on the CPU forbids it here. Read this section for two things only: NVIDIA's
own published statement of *what size of transform fits in what container*, and the design of
the thread/block API, which is the design a hand-written kernel should imitate.

### 4.1 The size limits, updated (and §07's table is now stale)

§07 §1.1 quotes the cuFFTDx 1.2.1 requirements table. The current documentation `[F]`
<https://docs.nvidia.com/cuda/cufftdx/requirements_func.html> gives:

| execution scope | half / float | **double** |
|---|---|---|
| **thread** (one thread's registers) | `[2; 64]` | **`[2; 32]`** |
| **block**, SM 75 | `[2; 16384]` | `[2; 8192]` |
| **block**, SM 86/89/120/121 | `[2; 24389]` | `[2; 12167]` |
| **block**, **SM 80**/87/90/100/103 | `[2; 32768]` | **`[2; 16384]`** |

plus a new row §07 does not have: FFTs that require a workspace top out at **8191** on
SM 80 `[F]`.

The sm_80 double-precision numbers are unchanged from §07 (32 per thread, 16384 per block); the
SM 86/89 row has changed from `[2; 8192]` to `[2; 12167]` (note `12167 = 23³`, and
`24389 = 29³` — NVIDIA is expressing the ceiling as a cube of a prime, which is a hint about
their internal decomposition). **§07's table should be read as "cuFFTDx 1.2.1", not as current.**

For us the operative numbers are:

* **32 complex doubles is the largest 1D transform NVIDIA will put in one thread's registers.**
  Every one of `L = 6, 8, 13, 17, 23` is inside that. `L = 36, 45, 64` are outside it — so
  NVIDIA's own judgement is that a 36-, 45- or 64-point double-precision line should *not* be
  held by a single thread. §6.5 gives the register arithmetic that explains why (a 64-point line
  is 128 registers of data before any temporary).
* **16384 complex doubles per block** is 256 KiB — larger than the 163 KiB a block can address in
  shared memory, so cuFFTDx's block ceiling at sm_80 assumes the register file participates.
  Consistent with VkFFT's `registerBoost`.

### 4.2 The model worth copying

Verbatim from the cuFFTDx docs, quoted in §07 §1.1 from the same pages and still current:
thread-level execution keeps "the input array … in registers" sized by `FFT::storage_size`;
block-level execution distributes elements across threads so that thread *n* holds elements
`n + FFT::stride*i` and uses `FFT::shared_memory_size` bytes as the inter-thread exchange
buffer; `ElementsPerThread` "determines the number of registers required per thread and the
exact implementation". Three things to lift from that:

1. **The strided element-to-thread map (`n + stride*i`), not a blocked one.** A thread owns
   elements `n, n+S, n+2S, …`, so a warp's *i*-th load is 32 consecutive elements — coalesced by
   construction, both from global memory and from shared memory. A blocked map (thread *n* owns
   elements `n*e … n*e+e-1`) makes every load a 16-byte stride-`e` gather. **This is the single
   most important layout decision in the kernel and cuFFTDx's answer is unambiguous.**
2. **Shared memory as a *communication* buffer, not as the working set.** The transform lives in
   registers; shared memory exists only to move data between threads at a stage boundary. That
   is also VkFFT's stated model ("use shared memory only as a communication buffer", quoted in
   §08 §5.7). For a fused 3D kernel the analogue is: hold a *pencil* in registers, and use
   shared memory only for the axis-to-axis transpose.
3. **Elements-per-thread as the primary tuning knob**, chosen by a size- and
   architecture-dependent heuristic rather than fixed. Our version of that knob is
   "points per thread", and it should be a compile-time constant we sweep (§8).

### 4.3 What its 3D examples do and do not demonstrate

cuFFTDx ships, and the current examples page still lists `[F]`
<https://docs.nvidia.com/cuda/cufftdx/examples.html>:

* `fft_3d_cube_single_block` — "Small 3D (equal dimensions) **FP32** FFT that fits into a single
  block"
* `fft_3d_box_single_block` — "Small 3D **FP32** FFT that fits into a single block, each
  dimension is different"
* `fft_3d` — "Example showing how to perform 3D **FP32** C2C FFT with cuFFTDx"
* `convolution_3d`, `convolution_3d_padded`, … — "cuFFTDx fused 3D convolution with
  preprocessing, filtering and postprocessing"

**Every 3D example is FP32.** That was true when §07 was written and is still true. NVIDIA
demonstrates the single-block fused 3D cube — which is precisely our Regime A/B/C — and
demonstrates it only at half the element size we need. Our `L ≤ 17` regime is therefore *the
FP64 version of a structure NVIDIA ships as an example and never published in double*. Encouraging
and unproven in equal measure.

The published performance claims, for calibration (all `[F]`, same examples page and the
convolution example):

* "Depending on the device, the precision and the size of a given FFT the improvements from
  using cuFFTDx range from **45% to up to 3x** speed-ups" — but that is a *forward FFT +
  pointwise + inverse FFT* pipeline against a 3-kernel cuFFT path, i.e. two of three global
  round trips removed. §07 §1.1 already flags this and the flag stands.
* 3D convolution: "speedups of between **3.9x and 1.3x** depending on the size".
* Mixed-precision 1D: "**1.7x**" on H100.
* And the honest one, from the "Achieving High Performance" page `[F-agent]`
  <https://docs.nvidia.com/cuda/cufftdx/performance.html>: "Taking the regular cuFFT library as
  baseline, **the performance may be up to one order of magnitude better or worse**." Also
  "**Best parameters for compute bound and memory bound kernels might not be identical**" —
  which for us means: tune for the batched, bandwidth-bound case, and expect a different
  configuration to win in the small-batch, latency-bound case (the mechanism §08 §3 established
  on CPU).

**None of these is a bare-forward-transform number.** There is still no published cuFFTDx
figure for a bare batched FP64 3D transform, which is the thing we are building.

---

## 5. Target 5 — FP64 tensor cores (DMMA) for the DFT: the answer is yes, up to `L ≈ 23`

This is the open question the brief asked about, and it has a clean answer. **The idea is
supported by the arithmetic and by one 2026 paper that does exactly this for a different
operator; it is refuted for `L ≥ 36`; and every published tensor-core FFT paper is irrelevant to
us because they are all half- or single-precision and miss our accuracy gate by ten orders of
magnitude.**

### 5.1 The instruction, and the fact that settles the accuracy question first

> "To meet the rapidly growing compute needs of HPC computing, A100 Tensor Cores support
> acceleration of **IEEE-compliant FP64 computations** … The new Double Precision Matrix
> Multiply Add instruction on A100 replaces 8 DFMA instructions on V100, reducing instruction
> fetches, scheduling overhead, register reads, datapath power, **and shared memory read
> bandwidth**. Using Tensor Cores, each SM in A100 computes a total of **64 FP64 FMA
> operations/clock (or 128 FP64 operations/clock)**, which is twice the throughput of Tesla
> V100. The A100 Tensor Core GPU with 108 SMs delivers a peak FP64 throughput of **19.5
> TFLOPS**, which is 2.5x that of Tesla V100."
> — NVIDIA A100 Tensor Core GPU Architecture whitepaper, "A100 Tensor Cores Accelerate HPC"
> `[F]` <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf>

and from the CUDA C++ Programming Guide, §7.24.3 "Double Precision" `[F-agent]`
<https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html>:

> "Tensor Cores support double-precision floating point operations on devices with compute
> capability 8.0 and higher. To use this new functionality, a fragment with the `double` type
> must be used. **The `mma_sync` operation will be performed with the `.rn` (rounds to nearest
> even) rounding modifier.**"

**So DMMA is round-to-nearest-even IEEE double in and out.** A DMMA-based dense DFT has exactly
the same error behaviour as the same dot products on CUDA cores — the `1e-12` gate is a
non-issue. That single sentence is what separates this idea from the whole published
tensor-core FFT literature (§5.7).

Note the whitepaper's own list of what DMMA saves: instruction fetches, scheduling, register
reads, **and shared-memory read bandwidth**. Hold on to the last one; §5.6 says it is the real
reason to use DMMA here.

### 5.2 The shape, and it is the binding constraint on A100

From the PTX ISA `[F-agent]` <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
(PTX ISA 9.3; the shape table and §9.7.15.5.2/§9.7.15.5.14):

> "`.f64` floating point type `mma` operation with `.m8n8k4` shape requires **sm_80** or higher."
>
> "`.f64` floating point type `mma` operation with `.m16n8k4`, `.m16n8k8`, and `.m16n8k16` shapes
> require **sm_90** or higher."

**On an A100 there is exactly one FP64 tensor-core shape: `m8n8k4`.** The wider Hopper shapes do
not exist for us. The fragment layout, verbatim `[F-agent]`:

> Multiplicand A: "A vector expression containing a single `.f64` register, containing single
> `.f64` element from the matrix A" — `row = %laneid >> 2`, `col = %laneid % 4`
>
> Multiplicand B: "A vector expression containing a single `.f64` register" — `row = %laneid % 4`,
> `col = %laneid >> 2`
>
> Accumulator C/D: "A vector expression containing of two `.f64` registers" —
> `groupID = %laneid >> 2`, `row = groupID`,
> `col = (threadID_in_group * 2) + (i & 0x1) for ci where i = {0, 1}`

Per warp per instruction: A is 8×4 (32 doubles, one per lane), B is 4×8 (32 doubles, one per
lane), D is 8×8 (64 doubles, two per lane). **256 FP64 FMAs per instruction per warp, from 4
registers per lane.** CUTLASS's sm_80 atom confirms the encoding `[F-agent]`
<https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/arch/mma_sm80.h>:

```
// Matrix Multiply 884 - F64,  GemmShape<8,8,4>
//   FragmentA = Array<double,1>, FragmentB = Array<double,1>, FragmentC = Array<double,2>
asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
```

— note **A row-major × B column-major is fixed** in that atom. The C++ `wmma` API also exposes
double, and only at `8x8x4` `[F-agent]` (Programming Guide "Element Types and Matrix Sizes" →
"Double Precision Support: `double | double | double | 8x8x4`").

**A trap that will otherwise cost someone a day.** There is a widely-cited warning that
`mma.m8n8k4` is a trap on Ampere:

> "there is one special instruction `mma.m8n8k4`. On Turing Tensor Cores, it will be compiled
> into a couple of HMMA.884 instructions … However, **on Ampere Tensor Cores, it will be
> compiled into a set of FPU instructions which will eventually run on the CUDA cores and lead
> to inferior performance** than expected for the Tensor Cores."
> — Sun, Li, Geng, Stuijk & Corporaal, *Dissecting Tensor Cores via Microbenchmarks: Latency,
> Throughput and Numeric Behaviors*, IEEE TPDS, arXiv:2206.02874 `[F-agent]`
> <https://arxiv.org/pdf/2206.02874>

**That statement is about `.f16`, not `.f64`.** Their figure labels it `HMMA.884`, their numeric
study covers "TF32, BF16, FP16" only, and the PTX doc's matching note ("`mma.sync.m8n8k4` is
optimized for target architecture sm_70 and may have substantially reduced performance on other
target architectures") sits under the `.f16` entry. The `.f64` variant compiles to a real
`DMMA.884` — see §5.3.

### 5.3 The measured throughput: ≈97 % of the 19.5 TFLOP/s figure

The one microbenchmark paper that actually measures `m8n8k4.f64` on an A100:

**Abdelkhalik, Arafa, Santhi & Badawy, *Demystifying the Nvidia Ampere Architecture through
Microbenchmarking and Instruction-level Analysis*, arXiv:2208.11174 (2022)** `[F-agent]`
<https://arxiv.org/pdf/2208.11174>. Their Table III row for f64, verbatim:

```
Supported shapes: m8n8k4 | Inputs: .f64 | Accumulators: .f64 | Cycles: 16 | measured-theoretical: 19-19.5
PTX:  wmma.mma.sync.aligned.row.row.m8n8k4.f64.f64.f64.f64.rn
SASS: 1*DMMA.884  -  each inst. is 16 cycles
```

Two facts to take: **the PTX maps 1:1 to a single `DMMA.884` SASS instruction with a 16-cycle
issue interval**, and the measured throughput is **19–19.5 TFLOP/s against a 19.5 theoretical
peak — about 97 %**. (Their column header says "GB/s"; the sibling rows read 311–312 and
594–624 against A100's 312 TFLOPS and 624 TOPS peaks, so the unit is plainly TFLOP/s and the
header is a typo. Flag that if you cite it.)

Sanity check against the whitepaper: 64 FP64 FMA/clk/SM × 108 SM × 2 flop × 1.41 GHz =
19.49 TFLOP/s, and one `DMMA.884` is 256 FMAs, so 16 cycles per instruction per warp × 4
warp schedulers = 64 FMA/clk/SM. **The whitepaper's rate, the measured 16-cycle latency and the
19.5 TFLOP/s headline are all the same number.** Nothing is hidden.

For contrast, **nobody publishes an achieved A100 DGEMM TFLOP/s in text.** The CUTLASS README's
A100 figure is an image; its current README has no A100 chart at all; the one peer-reviewed
CUTLASS-on-A100 study (Huang et al., *Appl. Sci.* **13**, 13022 (2023) `[F-agent]`
<https://xianweiz.github.io/doc/papers/cutlass_applsci23.pdf>) reports "the achieved fp32 GEMM
performance via CUTLASS is about 95% of the peak fp32 performance" and puts its FP64 result only
in a plot. **Do not accept an A100 DGEMM efficiency number from anyone without a text source.**

### 5.4 The DFT as a GEMM: the exact arithmetic, and where the crossover is

**Formulation.** One axis of the 3D transform is `Y = F · X` where `F` is the `L×L` DFT matrix
and `X` is `L × (L²·B)` — every line along that axis, from every plane, from every volume, as
one enormous matrix. `M = L`, `K = L`, `N = L²·B`. `N` is in the millions, which is exactly the
regime tensor cores want. Three such GEMMs (one per axis, with an index permutation between)
give the whole transform. §08 §2.3 already identified the "3D DFT as three batched GEMMs"
formulation on CPU; on an A100 it has a dedicated datapath.

**Flop count.** A complex MAC is 8 real flops (4 real multiplies, 4 real adds — the 4M form;
see §5.6 for 3M). An `L×L` complex matvec is `8L²` flops; a whole volume is
`3·L²·8L² = 24L⁴` flops, i.e. **`24L` flops per point** — against the butterfly's
`15·log2 L`. Arithmetic intensity `24L/32 = 0.75L` flop/byte.

**The crossover.** The dense matvec's arithmetic disappears under the 32-byte-per-point
bandwidth floor when `24L ≤ 32 × (machine balance)`:

* on the **vanilla FP64** pipe (6.24 flop/B): `L ≤ 8.32`
* on the **DMMA** pipe (12.54 flop/B): `L ≤ 16.72`

| L | dense flop/point | AI (flop/B) | time ÷ bandwidth floor, **vanilla FP64** | ÷ floor, **DMMA at 100 %** | ÷ floor, **DMMA with `m8n8k4` padding** | ÷ floor, **conj-symmetric + padded DMMA** |
|---|---|---|---|---|---|---|
| **6** | 144 | 4.5 | **0.72** ✅ | 0.36 | 0.64 ✅ | **0.32** ✅ |
| **8** | 192 | 6.0 | **0.96** ✅ | 0.48 | 0.48 ✅ | **0.48** ✅ |
| **13** | 312 | 9.75 | 1.56 | 0.78 ✅ | 1.18 | **0.59** ✅ |
| **17** | 408 | 12.75 | 2.04 | **1.02** | 1.69 | **0.84** ✅ |
| **23** | 552 | 17.25 | 2.77 | 1.38 | 1.50 | **0.75** ✅ |
| **36** | 864 | 27.0 | 4.33 | 2.15 | 2.39 | 1.20 |
| **45** | 1080 | 33.75 | 5.41 | 2.69 | 3.06 | 1.53 |
| **64** | 1536 | 48.0 | 7.69 | 3.83 | 3.83 | 2.15 |

(✅ = the arithmetic is entirely hidden under the memory floor and therefore *free*. All four
right-hand columns are my arithmetic, §11; the inputs are the two published peaks and the
`m8n8k4` shape.)

**The padding column.** `m8n8k4` forces `M` to a multiple of 8 and `K` to a multiple of 4 (`N`
is the huge batch dimension and pads for free). Waste factor `= ⌈L/8⌉·8/L × ⌈L/4⌉·4/L`:

| L | M pads to | K pads to | waste | effective DMMA rate |
|---|---|---|---|---|
| 6 | 8 | 8 | **1.78×** | 11.0 TFLOP/s |
| 8 | 8 | 8 | **1.00×** | 19.5 |
| 13 | 16 | 16 | 1.52× | 12.9 |
| 17 | 24 | 20 | **1.66×** | 11.7 |
| 23 | 24 | 24 | **1.09×** | 17.9 |
| 36 | 40 | 36 | 1.11× | 17.6 |
| 45 | 48 | 48 | 1.14× | 17.1 |
| 64 | 64 | 64 | **1.00×** | 19.5 |

**`L = 8` and `L = 64` are exact fits. `L = 23` and `L = 36` are nearly free. `L = 17` is the
worst case in the set** — 24×20 of instruction for a 17×17 product, 1.66× waste, which eats
five sixths of DMMA's 2× advantage. This is precisely the mechanism the one comparable published
work measured (§5.5).

**The conjugate-symmetry column, which is where `L = 17` is rescued.** The CPU winner is not a
plain complex matvec; it is a *conjugate-symmetric* one, and the symmetry is a property of the
matrix, not of the data. Write `x = a + i·b` with `a, b` real. Then `F·x = F·a + i·(F·b)`, and
because `a` and `b` are real, `(F·a)_{L-j} = conj((F·a)_j)`. So only `⌊L/2⌋+1` output rows of
each real transform need computing; the rest are conjugates. Two real matrices (`cos` and `sin`
parts, `(⌊L/2⌋+1) × L` each) applied to two real vectors, then combined:

| L | rows needed `⌊L/2⌋+1` | conj-sym flop/point | ÷ naive dense |
|---|---|---|---|
| 6 | 4 | 96 | 0.667 |
| 8 | 5 | 120 | 0.625 |
| 13 | 7 | 168 | 0.538 |
| **17** | **9** | **216** | **0.529** |
| 23 | 12 | 288 | 0.522 |
| 36 | 19 | 456 | 0.528 |
| 64 | 33 | 792 | 0.516 |

**`≈1.9×` fewer flops at our sizes, and it makes everything real** — which matters twice over,
because a real GEMM maps to `m8n8k4` directly with no 4M/3M decomposition and no complex
register layout. Putting the two together: **a conjugate-symmetric real-matrix formulation on
DMMA is under the HBM bandwidth floor for `L = 6, 8, 13, 17, 23`, and only clearly above it for
`L = 36, 45, 64`.**

**That is the answer to the brief's question.** The FP64-tensor-core DFT idea is **supported**,
and it is supported exactly at the geometry where the CPU panel found the dense matvec winning.
It is **refuted for `L ≥ 36`**, where even a symmetric dense matvec on DMMA is 1.2–2.2× over the
memory floor and a butterfly (2.4–2.8 flop/byte) is 2.2–2.6× *under* it.

### 5.5 The one comparable published work, and it says the mechanism is shared memory

**Tu, Karlin, Camier, Dobrev, Kolev, Henneking & Ghattas, *Accelerating High-Order Finite Element
Simulations at Extreme Scale with FP64 Tensor Cores*, arXiv:2603.09038v2 (9 Apr 2026)**
`[F-agent]` <https://arxiv.org/html/2603.09038v2>. They hand-program
`mma.sync.aligned.m8n8k4…f64` via inline PTX to accelerate tensor contractions decomposed into
**small dense GEMMs of order O(10)** — e.g. `m=25, n=5, k=4`. Structurally this is our problem.
Verbatim:

> "To the best of our knowledge, **this is the first time that FP64 tensor cores have been
> directly programmed** to accelerate large-scale finite element scientific computing
> applications."
>
> "many applications … have not been able to leverage tensor cores as successfully because they
> either do not perform matrix-matrix multiplies or their operations do not map well to tensor
> units."
>
> **On padding waste:** "The DMMA PA kernel has a 54% utilization of the DMMA pipe on the GH200,
> while the original PA kernel has a 14% utilization of the FP64 pipe … One might ask why the
> 54 / 14 = 3.8× increase in compute utilization only leads to **1.5× speedup** for the entire
> kernel: this is primarily because when using the m8n8k4 DMMA instruction … the shapes of these
> (m=25, n=5, k=4) matrix-matrix multiplications **do not exactly fit the (m=8, n=8, k=4)
> instruction shape, thus wasting a large percentage of the computation** due to the mismatch."
>
> **On what the real bottleneck is:** "We show that **shared memory data motion, not FLOPs, is
> the performance bottleneck**, and demonstrate strong correlation between tensor core data
> motion reduction and improved application performance." / "shared memory bandwidth ('L1: Data
> Pipe Lsu Wavefronts') is still the highest-used resource … now at 84%."
>
> **On why DMMA helped anyway:** "**By using tensor cores instead of CUDA cores for FP64
> matrix-matrix multiplications, with the same shapes, fewer bytes are loaded from the shared
> memory.**"
>
> **On why they rejected the libraries — read this before planning a CUTLASS route:** "the CUDA
> PTX DMMA instruction is chosen over other CUDA libraries like CUTLASS or CUBLAS, because our
> small O(10) matrices—such as 25×5×4—require precise control for custom thread-to-fragment
> mappings and bank conflict elimination. **CUTLASS is optimized for larger GEMMs and lacks the
> needed granularity**; its primary method to avoid shared memory bank conflicts is swizzling,
> which requires much larger alignments (e.g., 128 bytes) and shapes (e.g., m64n64k32). While
> padding could enable the use of CUTLASS kernel-level interfaces, with the small matrix sizes we
> are targeting, **the majority of the shared memory bandwidth and FLOP/s would be wasted on the
> padding**… CUBLAS, on the other hand, does not support such unconventional scenarios with
> small, irregular GEMMs and kernel fusions."

Their results are on GH200/GB200, **not A100** — "up to 2× performance gains and up to 83%
energy efficiency gains", 1.5× for the DMMA change alone. And a sentence that is almost a
mission statement for this project: "larger performance gains than the ones presented here could
likely be achieved **if the operators were discretized in such a way to best fit the tensor-core
architecture**."

**Three lessons for us, and they are the practical core of §5:**

1. **Hand-write the PTX; do not reach for CUTLASS.** Their reasoning applies verbatim to an
   `L×L` DFT matrix with `L ∈ [6, 64]`, and CUTLASS is a library and therefore banned anyway.
2. **The reason to use DMMA is bytes, not flops.** `mma.m8n8k4` reads each A element **once per
   warp** (32 elements for an 8×4 tile) where a CUDA-core dot product reads it once per thread.
   For a dense DFT, `F` is reused across every column of `X`, so the shared-memory read traffic
   for `F` is the thing DMMA deletes. The whitepaper says the same thing in its own list
   ("…and shared memory read bandwidth"). §6.2 is why that matters here.
3. **Expect the padding waste to show up as a real fraction, not as a rounding error.** They
   measured 3.8× compute-utilisation buying 1.5×. Our worst case, `L = 17`, is 1.66× of waste
   before any implementation loss.

### 5.6 Complex-on-real: 4M, 3M, and what to do

DMMA is real-valued. A complex GEMM is either **4M** (four real GEMMs: `AR·BR − AI·BI`,
`AR·BI + AI·BR`) or **3M / "Gaussian"** (three real GEMMs, Karatsuba-style, 25 % fewer flops).
CUTLASS ships both for `cf64` on sm_80 `[F-agent]`
<https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/functionality.md>:

> `| **TensorOp** | 80+ | 11.4+ | cf64 * cf64 + cf64 => cf64 | {N,T} x {N,T} => {N,T} |`

with unit tests `gemm_cf64n_cf64t_cf64t_tensor_op_f64_sm80.cu` and
`gemm_cf64n_cf64t_cf64t_tensor_op_f64_gaussian_sm80.cu`, and the accumulator arrangement
documented in `mma_gaussian_complex_tensor_op.h` `[F-agent]`:

> "this storage arrangement is to be considered 'gaussian complex' in the sense that the
> accumulation is done in three parts namely part1, part2, and part3 … This matches the structure
> of Tensor Cores which are always real-valued matrix multiplies."

**So complex-double GEMM on A100 FP64 tensor cores is a supported, unit-tested configuration —
including the 3M variant.** That establishes feasibility. But two cautions:

* 3M needs **three** real accumulators per complex output against 4M's two (re, im) — a 50 %
  larger accumulator register footprint for a 25 % flop saving, and §6.5 says registers are the
  scarce resource. At `L ≤ 23` we are under the bandwidth floor anyway (§5.4), so *the 25 % is
  worth nothing and the 50 % costs*. **Use 4M** — or, better, avoid the question entirely by
  using the conjugate-symmetric real formulation of §5.4, which never forms a complex matrix.
* **I found no published error analysis of 3M on tensor cores in any precision**, and CUTLASS
  ships the kernel with no error bound. §07 §6 established that the FMA complex multiply is
  provably optimal at 2u and that the 3-multiply form loses on CPU; nothing here contradicts
  that. Do not spend the accuracy budget on 3M for a 25 % flop saving we do not need.

### 5.7 The half-precision tensor-core FFT literature, and why it is inadmissible here

The brief asked for these to be covered and clearly distinguished. They are all
**half/single/FP8** and none of them is relevant to a `1e-12` gate.

**tcFFT** — Binrui Li, Shenggan Cheng, James Lin, *tcFFT: Accelerating Half-Precision FFT
through Tensor Cores*, arXiv:2104.11471 (2021); conference version *tcFFT: A Fast Half-Precision
FFT Library for NVIDIA Tensor Cores*, IEEE `[F-agent]` <https://arxiv.org/abs/2104.11471>,
<https://arxiv.org/pdf/2104.11471>.

> "We evaluated our tcFFT and the NVIDIA cuFFT in various sizes and dimensions on NVIDIA V100
> and A100 GPUs. The results show that our tcFFT can **outperform cuFFT 1.29x-3.24x and
> 1.10x-3.03x** on the two GPUs, respectively."

**The decisive number is the accuracy table**, Table 4, "Average relative error of 1D and 2D
FFT": `cuFFT-1D 1.78±0.5 %`, `tcFFT-1D 1.76±0.5 %`, `cuFFT-2D 1.65±0.1 %`, `tcFFT-2D
1.65±0.1 %` `[F-agent]`. **≈1.7 × 10⁻² relative error against our gate of 1 × 10⁻¹².** Ten orders
of magnitude. And their own diagnosis of why:

> "Although in theory, tcFFT uses matrix multiplication as the basic operator, which has better
> error control. But the complete progress of FFT consists of multiple mergings … **The storage
> of intermediate results is the main source of error.**"

Two of their observations *do* transfer, and both support §2:

> "compared to V100, A100 has 2.5x half-precision computing power but **only a 1.7x global memory
> bandwidth**. As a result, optimized FFT algorithm that resorts to more computing power can only
> bring less performance gain."
>
> "in bandwidth-bound cases, tcFFT can use up almost all the bandwidth."

**Sorna, Cheng, D'Azevedo, Wong & Tomov, *Optimizing the Fast Fourier Transform using Mixed
Precision on Tensor Core Hardware*, 2018 IEEE 25th Intl. Conf. on HPC Workshops (HiPCW)**
`[F-agent]` <https://www.osti.gov/servlets/purl/1559731>. Splits FP32 input into two FP16 sets,
multiplies on Volta tensor cores via cuBLAS batched GEMM, recombines. Its result is the useful
part:

> "**it is seen that our implementation is currently inferior to the highly optimized cuFFT
> library.**" and, on accuracy: "The relative error exceeds 0.1 or the program throws an error
> when: input data range greater than 3 × 1E10; or input data range less than 5 × 1E-11."

**TurboFFT is not a tensor-core paper and must not be cited as one.** Wu, Zhai, Liu, Huang,
Jian, Dai, Di, Chen & Cappello, *TurboFFT*, arXiv:2405.02520 (2024) and *TurboFFT: Co-Designed
High-Performance and Fault-Tolerant Fast Fourier Transform on GPUs*, PPoPP '25, DOI
`10.1145/3710848.3710853` / arXiv:2412.05824 `[F-agent]`. It is FP32/FP64 on ordinary CUDA cores
with an ABFT fault-tolerance scheme; a grep of the full PPoPP text for `tensor core|mma|wmma`
returns nothing in the technical body. Its useful contributions to *us* are the small-problem
findings already quoted in §3.4, plus:

> "**TurboFFT achieves 90% peak memory bandwidth.** The memory bandwidth is measured by
> 2× problem size divided by execution time." `[F-agent]` (arXiv:2412.05824)
>
> "For double precision, the overhead of sin(x) and cos(x) can occupy **up to 20% of the total
> execution time**, hence the twiddle factors are prepared in global memory to be fetched."
> `[F-agent]` (arXiv:2405.02520)

That second quote is a GPU restatement of §07 §5's conclusion: **never compute twiddles with
`sincos` at runtime in FP64.** On this hardware it costs 20 % of the kernel.

**Durrani et al.** — two real records, both paywalled, metadata from dblp `[F-agent]`
<https://dblp.org/pid/141/0119.html>: "FFT blitz: the tensor cores strike back", PPoPP 2021,
pp. 488–489 (a 2-page brief), DOI `10.1145/3437801.3441623`; and "Accelerating Fourier and Number
Theoretic Transforms using Tensor Cores and Warp Shuffles", PACT 2021, pp. 345–355, DOI
`10.1109/PACT52795.2021.00032`. **`[UNVERIFIED — could not fetch]`** — ACM returned 403 and no
preprint was reachable. GPU, precision and accuracy are unknown to me. A search snippet claimed
"up to 4× for NTT and up to 1.5× for FFT" but I could not verify it; do not cite that figure.
Note that the NTT half of that work is *integer* arithmetic and therefore exact — a different
accuracy story from FP16 FFT, and possibly the more interesting half.

**Matsuoka, *FP8 is All You Need (Part 2): Efficient Ozaki–Bailey Style FFT Through Tensor-core
Garner Reformulation and Kulisch Escape Route*, arXiv:2606.23698v2 (26 Jun 2026)** `[F-agent]`
<https://arxiv.org/pdf/2606.23698v2>. This is the closest thing in the literature to our idea — a
3D FFT expressed as GEMMs against the DFT matrix — but on **FP8** tensor cores with Ozaki-scheme
emulation, on **Blackwell B300**, and it is **analysis, not measurement** ("The figures are model
upper bounds, not measurements"). Three quotes that matter to us anyway:

> "Steps 2 and 5 are **dense GEMMs with the DFT matrix when p and q are small** (typically
> zero-padded to powers of two); the GEMMs are tensor-core-amenable."
>
> "This route is structurally a tall-skinny GEMM: the contraction depth is only S = 11, well
> below the native MMA k-dimension of 16, so each band runs at **∼11/16 ≈ 69% tile utilisation
> with a 31% padding waste**."
>
> "At any η_FP64 ≥ 1.56·B_HBM **the native FFT path is already memory-bound, the emulated path is
> not required for FFT**."

The last one is the same conclusion as §2.2, reached independently: on any part where the FP64
rate exceeds ~1.56× the HBM bandwidth, an FFT is memory-bound and no amount of extra arithmetic
throughput helps. Our part's ratio is 6.24, four times that threshold.

**And one negative result on the general "emulate FP64 with tensor cores" programme** —
Ootomo & Yokota, *Recovering single precision accuracy from Tensor Cores while surpassing the
FP32 theoretical peak performance*, arXiv:2203.03341 `[F-agent]`
<https://arxiv.org/pdf/2203.03341>, on Mukunoki's Ozaki-scheme DGEMM: "This method is also **not
competitive for FP64 matrix multiplication when compared to cuBLAS DGEMM** on NVIDIA Tesla series
GPUs."

**Summary of §5.7: there is no prior work on FP64-tensor-core FFT at all.** Every tensor-core FFT
paper trades accuracy we are not allowed to trade; the one paper that uses genuine FP64 tensor
cores for small dense GEMMs (§5.5) is a finite-element paper from 2026 that says it is the first
to program DMMA directly. **That is a real, checkable gap in the literature, and it is the gap
this project's `L = 17` geometry sits in.**

### 5.8 Verdict, and what an implementer should actually do

**Supported**, with these boundaries:

* `L = 6, 8`: a dense per-axis matvec is free **even on the ordinary FP64 pipe** (0.72 and 0.96
  of the memory floor). **Do not bother with DMMA at these sizes** — the arithmetic is already
  invisible and DMMA adds fragment-shuffling complexity for nothing. Note `L = 8` is nevertheless
  an *exact* `m8n8k4` fit, so it is the right place to *learn* the instruction.
* `L = 13, 17, 23`: **this is where DMMA earns its keep.** A conjugate-symmetric real
  formulation on DMMA lands at 0.59, 0.84 and 0.75 of the memory floor — under it, with margin —
  where the same formulation on CUDA cores is 1.56, 2.04 and 2.77× over. **This is the single
  most promising untried idea in the GPU phase.**
* `L = 36, 45, 64`: **refuted.** 1.20, 1.53 and 2.15× over the floor even with symmetry and
  DMMA, against a butterfly that is 2.4–2.8× under. Use a butterfly. (A *hybrid* is worth one
  experiment: factor `36 = 6·6`, `45 = 9·5`, `64 = 8·8` and use a DMMA dense matvec for the
  small factors, which is a `L=6/8/9`-sized dense matrix and therefore free. That is Good–Thomas
  or Cooley–Tukey with dense codelets, and the codelets are exactly the sizes DMMA likes.)
* Prefer **hand-written inline PTX `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64`** over
  `wmma` (which fixes `.rn` and hides the fragment map) and over CUTLASS (a library, banned, and
  wrong for `O(10)` matrices for the reasons its own users give in §5.5).
* **Validate the accuracy claim empirically anyway.** IEEE-compliance is documented, but the
  reduction order inside `DMMA.884` is not, and a `K = 20` accumulation with a different
  association order than your reference will differ in the last bits. Our gate is `1e-12` and a
  correct implementation should land near `1e-16`, so there is four orders of margin — but check
  it, and check it at the largest `L` you use DMMA for.
