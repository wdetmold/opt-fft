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
opt-in (§1.1). And the **250 W** cap is the PCIe SKU's, against 400 W for the SXM
part. NVIDIA's datasheet does publish the same 9.7 / 19.5 TFLOP/s for the PCIe 40GB SKU as for
SXM `[F-agent]`, and the whitepaper notes "Peak rates are based on GPU Boost Clock" `[F]` — but
whether a *sustained* FP64 or DMMA kernel holds 1410 MHz inside a 250 W envelope is not something
NVIDIA publishes. §7.2 and §8.8 say to read the clock-throttle counters rather than assume.

The cluster also has 8-GPU A100 nodes in the `a100l` / `a100r` partitions (`gpu:a100:8`), so
an exclusive single-GPU measurement is arrangeable. `[M]` (`sinfo`)

### 0.1 The documented budgets, from NVIDIA

| resource | value on sm_80 | source |
|---|---|---|
| SMs | **108** (of 128 on a full GA100) | whitepaper `[F]` |
| FP64 cores per SM | **32** | CUDA Programming Guide `[F-agent]` |
| Peak FP64 (vanilla) | **9.7 TFLOP/s** | whitepaper Table 1 (SXM) `[F]`; datasheet, all four SKUs incl. PCIe 40GB `[F-agent]` |
| Peak FP64 **Tensor Core** | **19.5 TFLOP/s** | same two sources `[F]` `[F-agent]` |
| HBM2 bandwidth (40 GB) | **1555 GB/s** | whitepaper Table 4 `[F]`; datasheet row "GPU Memory Bandwidth", 40GB PCIe = 1,555GB/s `[F-agent]` |
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

### 0.2 Ten things to do differently from the CPU phase

Each links to the subsection that justifies it with a source and a number.

1. **Stop optimising arithmetic.** Every geometry is bandwidth-bound by 2.2–5.2× on the *ordinary*
   FP64 pipe. Halving the flop count of any of these kernels changes nothing. §2.2.
2. **Count global memory passes, and make it one read plus one write.** For `L ≤ 23` that is
   achievable in a single kernel; for `L ≥ 36` two passes is the target and three is a 3× traffic
   penalty. cuFFT's visible deficit in FP64 on A100 appears to *be* extra passes. §1.5, §2.3.
3. **Drop `B = 1`.** One `6³` transform is 4.4 ns of traffic against 1–4 µs of launch overhead. A
   `B = 1` number on this hardware measures the launch path. §6.6, §8.4.
4. **Size the scored batch so the working set is 25–50× the 40 MiB L2**, or you will be measuring
   L2 and tuning for the wrong roof. §2.4, §8.4.
5. **Load and store 128 bytes per thread — eight complex doubles.** A measured curve puts
   16 B/thread at 208 GB/s and 128 B/thread at 836 GB/s: a 4× difference from the load pattern
   alone. §6.1.
6. **Make every shared-memory row stride an odd number of complex doubles.** `L = 8` and `L = 64`
   unpadded are worst-case 8-way bank conflicts, which by §6.2's budget makes shared memory the
   bottleneck by 2×. `L = 13, 17, 23, 45` need no padding. §6.2.
7. **Try FP64 tensor cores at `L = 13, 17, 23`, and nowhere else.** DMMA is IEEE-compliant FP64
   and exactly 2× the vanilla rate, which moves the size at which a dense per-axis DFT matrix
   becomes *free* from `L ≈ 8` to `L ≈ 17`. This is the CPU phase's `L = 17` winner with a
   dedicated datapath, and there is no prior work on it. §5.
8. **Hand-write the PTX; do not reach for a library.** cuFFTDx, CUTLASS and cuBLAS are all banned
   and all wrong for `O(10)` matrices anyway — the one published FP64-tensor-core HPC paper
   explains why in detail. §5.5, §8.7.
9. **Report effective bandwidth as the primary metric**, with the 1.3 TB/s roofline printed beside
   it, and keep the benchFFT `5N log₂N` GF/s column only for continuity with the CPU leaderboard.
   §7.3, §8.6.
10. **Lock the clocks, read the throttle counters before and after every run, and cap the inner
    repeat count.** One published FFT/DSP paper measured a 20–25 % slowdown from repeating a
    kernel in a tight loop *with clocks already locked*. §7.2, §8.8.

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
  lane holds 16 complex doubles = **64 of its 255 registers** of data, plus temporaries — still
  comfortable. A block of 8 warps does 8 volumes; a block of 32 warps (1024 threads) does 32.
* **Regime B — `L = 13`: one volume per warp *just* misses (108 % of the warp file), one
  volume per two warps fits.** 2197 complex doubles over 64 lanes is 34.3 complex = **137
  registers** of data per lane — under the 255 ceiling but only just, with temporaries to come. Or keep it in shared memory (21 % of a block's max) and
  use registers only for the codelet.
* **Regime C — `L = 17`: one volume per *block*, all three axes fused, and it fits in shared
  memory with room to spare.** 76.8 KiB is 47 % of the 163 KiB per-block maximum, or 77 % of
  the 100 KiB carveout. This is the sweet spot of the whole set: a single kernel reads the
  volume once, does x, y and z out of shared memory, writes it once. In registers it is 30 %
  of the SM register file — also viable (a block of 256 threads holds 19.2 complex doubles =
  **77 registers** of data per lane), and that route reaches the register file's much higher
  bandwidth. `L = 23` is the boundary: 190.1 KiB is **1.17× the per-block shared maximum**, so
  a 23³ volume does *not* fit in one block's shared memory, but it does fit in one block's
  *registers* (74 % of the SM file, 1024 threads × 11.9 complex = **48 registers** of data each).
* **Regime D — `L = 36, 45, 64`: no single-block option exists.** 729 KiB, 1.39 MiB and 4 MiB
  are 2.85×, 5.6× and 16× a block's entire register file. These must be multi-block and
  multi-pass, and the design question moves to *how few passes* and *where the intermediate
  lives* (§2.4). The natural unit is a **plane**: a `36×36` plane is 20.25 KiB, a `45×45` plane
  31.6 KiB, a `64×64` plane 64 KiB — all comfortably inside one block's shared memory, so the
  two contiguous-ish axes can be fused per plane and only the third axis needs a second pass.

**Nothing in this set fits in one *thread's* registers** (63 complex doubles; the smallest
volume is 216). So the cuFFTDx "thread-level FFT" model (§4) applies to a *line*, never to a
volume: a 6-, 8-, 13-, 17- or 23-point line is 6…23 complex doubles = **24…92 registers**, all
comfortably per-thread; a 36-, 45- or 64-point line is **144, 180 or 256 registers of pure
data** — and 256 is *past* the 255-register ceiling, so a 64-point double-precision line cannot
be held by one thread at all (§6.5).

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

| L | `B` for **one block on every SM** (one volume per block) | how many such blocks an SM can actually host, and what limits it |
|---|---|---|
| 6 | 108 | 32 — the hardware block limit; 3456 blocks (`B = 3456`) saturates |
| 8 | 108 | 32 — block limit; 3456 blocks saturates |
| 13 | 108 | 4 at a 164 KB carveout (34.3 KiB each) → 432 blocks saturates |
| 17 | 108 | **2** at a 164 KB carveout, **1** at 100 KB (76.8 KiB each) → 108–216 blocks saturates |
| 23 | 108 | **1** — 74 % of the SM register file per block → 108 blocks *is* saturation |
| 36 | one volume per block cannot fill the GPU below `B = 108`; **split the volume** | plane-per-block gives `B·36` blocks — `B = 3` already covers 108 SMs |
| 45 | split | plane-per-block gives `B·45` — `B = 3` covers the machine |
| 64 | split | plane-per-block gives `B·64` — `B = 2` covers the machine |

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
| cuFFT radix(2–13) | reaches the top band at *some* small lengths — I count **16 distinct markers between lengths 2 and 563 at 1170–1317 GB/s** — but most of its radix points sit in lower bands at **≈630** and **≈420** GB/s | | |

**Three conclusions, and they set the whole GPU phase's expectations.**

1. **≈1280 GB/s is what a good FP64 FFT kernel achieves on an A100 — 82 % of the 1555 GB/s
   peak.** That is the realistic roof. Multiply every floor in §2.2 by 1/0.82 = 1.22 to get a
   *credible* target: `L = 6` ≈ 5.4 ns, `L = 17` ≈ 123 ns, `L = 36` ≈ 1.17 µs per transform.
2. **At *some* radix-friendly small lengths cuFFT reaches the same top band**, so do not assume
   the 1D story hands you an easy 2× over cuFFT at `L = 6, 8, 13`. But it is not uniform: most of
   cuFFT's radix markers below length 563 sit in the ≈630 and ≈420 GB/s bands, and only 16 of
   them reach 1170–1317. Where cuFFT collapses reliably is at **primes**: its `Rader(17-127)`
   median is 629 GB/s against VkFFT's 1226 for the same lengths — **a factor of 1.95 at exactly
   our `L = 17` and `L = 23`.**
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
  is 256 registers of data, which is past the 255-per-thread ceiling).
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

---

## 6. Target 6 — kernel-level technique, with the per-geometry numbers

### 6.1 Coalescing: what the granule is, and the one measured curve

The documented rule is the 32-byte transaction (§1.3). The measured curve is more useful, and
tcFFT publishes it — Table 2, "Achievable Global Memory Bandwidth under Different Continuous
Size", for a radix-256 merging kernel `[F-agent]` <https://arxiv.org/pdf/2104.11471>:

| contiguous run per thread | 4 el / 16 B | 8 / 32 B | 16 / 64 B | **32 / 128 B** | 64 / 256 B |
|---|---|---|---|---|---|
| achieved GB/s | 208.1 | 384.6 | 553.5 | **836.3** | 715.8 |

> "the achievable memory throughput increases as the continuous size increases when it is no
> more than 32. It is reasonable for the largest cache line size on GPU is 128 bytes. **After
> that, the bandwidth drops instead.**" `[F-agent]`

**128 bytes per contiguous run is the optimum and 256 is worse.** For complex double that is
**8 complex doubles per thread, contiguously** — and it is the same conclusion VkFFT's guide
reaches from the other direction ("It is possible to regain full bandwidth by switching to
128-byte memory coalescing on Nvidia", §3.5b). Note the 16-byte-run figure: **a kernel where
each thread moves one complex double gets 208/836 = 25 % of the bandwidth of one where each
thread moves eight.** That single number should shape the load/store code of every kernel in
this competition.

Concretely: for `L ≤ 17` a block loads its whole contiguous volume, so give each thread a
`double4`-aligned run of 8 complex doubles (128 B) and loop. At `L = 17` the volume is 4913
complex doubles = 614.1 runs of 8 — not an integer, so either handle the tail or pad the
in-shared volume to 4920 and mask.

### 6.2 Shared memory banks with 16-byte elements: the exact padding rule, per geometry

This is the part of the GPU port that has no CPU analogue and where the documented rules are
written for 4-byte elements. Derive it properly.

32 banks of 32 bits (§0.1). **A complex double is 16 bytes = 4 consecutive banks.** A warp
issuing 32 × 16-byte shared accesses necessarily spans 512 bytes = 4 × the 128-byte bank width,
so the request is serviced in (at least) 4 phases of 8 threads; within a phase the 8 threads'
four-bank groups must be distinct mod 32.

For a shared array of complex doubles with **row stride `S` complex elements**, reading a
*column* (fixed column index, 8 consecutive rows in one phase) puts thread `r` at bank group
`4·(S·r mod 8)`. Those are distinct for `r = 0…7` **iff `gcd(S, 8) = 1`, i.e. iff `S` is odd.**
More precisely the conflict degree is exactly `gcd(S, 8)`:

| L | natural stride | `gcd(L,8)` | conflict on a column read | **pad the stride to** |
|---|---|---|---|---|
| **6** | 6 | 2 | **2-way** | **7** |
| **8** | 8 | 8 | **8-way — worst case** | **9** |
| 13 | 13 | 1 | none | 13 (already fine) |
| 17 | 17 | 1 | none | 17 |
| 23 | 23 | 1 | none | 23 |
| **36** | 36 | 4 | **4-way** | **37** |
| 45 | 45 | 1 | none | 45 |
| **64** | 64 | 8 | **8-way — worst case** | **65** |

**So `L = 8, 36, 64` (and mildly `L = 6`) need shared-memory padding and `L = 13, 17, 23, 45`
do not.** The odd `L` get this for free, which is a pleasant inversion of the CPU story where
odd sizes were the awkward ones.

**How much does it cost to get this wrong?** Two published measurements of exactly this fix:

* CUDA C++ Best Practices Guide, Table 3, `C = AAᵀ` on a **Tesla V100** `[F]`: "No
  optimization **12.8 GB/s**" → "Using shared memory to coalesce global reads **140.2 GB/s**"
  → "Removing bank conflicts **199.4 GB/s**". The padding step alone is **1.42×**. The
  guide's own words: "These many-way bank conflicts are very expensive. The simple remedy is to
  pad the shared memory array so that it has an extra column … **This padding eliminates the
  conflicts entirely, because now the stride between threads is w+1 banks (i.e., 33 for current
  devices), which, due to modulo arithmetic used to compute bank indices, is equivalent to a
  unit stride.**"
* Mark Harris, *An Efficient Matrix Transpose in CUDA C/C++*, NVIDIA developer blog (2013)
  `[F-agent]` <https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/>:
  `transposeCoalesced` → `transposeNoBankConflicts` is 51.3 → 99.5 GB/s (M2050) and
  97.6 → 144.3 GB/s (K20c) — **1.94× and 1.48×**. "**Removing the bank conflicts in this way
  brings us to about 95% of our fastest copy throughput.**" And the framing sentence, which
  matches §1.3: "**coalescing global memory accesses is by far the most critical aspect of
  achieving good performance.**" (Caveat: Fermi and Kepler parts, 2013.)

**And here is why it matters *quantitatively* for a fused 3-axis kernel** (my arithmetic, §11).
Shared memory delivers `32 banks × 4 B × 1.41 GHz × 108 SM = 19.5 TB/s` aggregate —
**15× the ~1.3 TB/s achievable HBM bandwidth.** A fused 3-axis kernel touches shared memory
about 8 times per point (one write on load, a read+write per axis, one read on store) = 128 B
of shared traffic per 32 B of HBM traffic, a ratio of 4. So:

* **conflict-free: 4 against 15 — shared memory has 3.75× of headroom and HBM is the limiter.**
* **4-way conflict (`L = 36` unpadded): 16 against 15 — the headroom is gone and shared memory
  becomes co-limiting.**
* **8-way conflict (`L = 8`, `L = 64` unpadded): 32 against 15 — shared memory is the
  bottleneck by 2×, and the kernel runs at half the bandwidth-bound speed for a reason that
  has nothing to do with bandwidth.**

That is the whole argument for padding, in numbers, on our part.

**A caution about copying VkFFT's padding rule.** §3.5d quotes VkFFT's two rules. The
`(numSharedBanks/2 + 1)/(numSharedBanks/2) = 17/16` rule is derived for **complex float**
(8 bytes = 2 banks, so 16 complex spans the 32 banks and the stride must be coprime to 16). For
**complex double** the relevant modulus is 8, not 16. Applying the 17/16 rule at `fftDim = 64`
gives a stride of **68**, and `gcd(68, 8) = 4` — still a 4-way conflict for 16-byte elements —
whereas VkFFT's other rule, `fftDim + 1 = 65`, is conflict-free. The code takes
`max(68, 65) = 68`. *(This is my inference from reading the code, not a measurement; I have not
profiled VkFFT. §11.)* **The lesson for us is just: derive the padding for 16-byte elements
yourself, and the answer is "make the stride odd".**

Two more documented facts worth having:

> "Shared memory can also be used to avoid uncoalesced memory accesses by loading and storing
> data in a coalesced pattern from global memory and then reordering it in shared memory.
> **Aside from memory bank conflicts, there is no penalty for non-sequential or unaligned
> accesses by a warp in shared memory.**" — Best Practices Guide `[F-agent]`

> "when multiple threads in a warp address the same shared memory location, resulting in a
> broadcast. In this case, multiple broadcasts from different banks are coalesced into a single
> multicast." — Best Practices Guide `[F]`

The second one matters for the **twiddle table and the DFT matrix**: every thread reading the
same `F[j][k]` is a broadcast, not a conflict. So a dense-matrix kernel's matrix reads are free
even at full warp width — which is the CUDA-core version of §5.5's DMMA argument.

### 6.3 `cp.async` and double buffering

> "**On devices with compute capability 8.0, the `cp.async` family of instructions allows copying
> data from global to shared memory asynchronously. These instructions support copying 4, 8, and
> 16 bytes at a time.** If the size provided to `memcpy_async` is a multiple of 4, 8, or 16, and
> both pointers passed to `memcpy_async` are aligned to a 4, 8, or 16 alignment boundary, then
> `memcpy_async` can be implemented using exclusively asynchronous memory operations.
> **Additionally for achieving best performance when using `memcpy_async` API, an alignment of
> 128 Bytes for both shared memory and global memory is required.**"
> — CUDA C++ Programming Guide, §7.27 `[F-agent]`

> "The A100 GPU includes a new asynchronous copy instruction that loads data directly from global
> memory into SM shared memory, **eliminating the need for intermediate register file (RF)
> usage**. Async-copy reduces register file bandwidth, uses memory bandwidth more efficiently,
> and reduces power consumption." … "**Two variants of the async copy instruction are available
> … BYPASS, which bypasses L1 cache and the register file … and ACCESS which saves data to L1
> for subsequent accesses and reuse.**" — A100 whitepaper `[F]`

**16 bytes is exactly one complex double**, so `cp.async` maps to our element size perfectly,
and the 128-byte alignment recommendation coincides with §6.1's 128-byte coalescing optimum. The
`.cg` (BYPASS) variant is the right choice for a streaming batched transform: the volume is read
once and never re-read, so caching it in L1 is pure pollution — and L1 is what you are giving up
to get a large shared carveout (§6.7).

**Where it pays and where it does not:**

* **`L ≥ 36` (Regime D): yes.** These kernels have multiple passes and a plane-per-block
  structure, so a block can `cp.async` plane `p+1` while transforming plane `p`. Classic double
  buffering, and the shared budget allows it: two `36×36` planes is 40.5 KiB of the 163 KiB.
* **`L ≤ 23` (Regimes A–C): probably not, and this is worth stating.** These kernels load the
  volume *once* and there is nothing to overlap it with. The only overlap available is *across
  volumes* — `cp.async` the next volume while transforming the current one, which doubles the
  shared/register footprint and halves the number of resident blocks. At `L = 17` in shared
  memory that takes you from 2 blocks per SM (2 × 76.8 KiB = 153.6 of 164) to 1, which is
  exactly the wrong trade because occupancy is already the scarce thing (§1.2). **Prefer more
  concurrent blocks over software pipelining within a block, at the small geometries.**
* The A100 also adds "a new shared-memory-based barrier unit (asynchronous barriers)"
  `[F]` for the split arrive/wait pattern that `cp.async` pipelines want.

### 6.4 Warp shuffles for the in-register transpose

> "`T` can be int, unsigned int, long, unsigned long, long long, unsigned long long, float or
> **`double`**." … "The exchange occurs simultaneously for all active threads within the warp
> (and named in `mask`), **moving 4 or 8 bytes of data per thread depending on the type**."
> — CUDA C++ Programming Guide, §7.22 Warp Shuffle Functions `[F-agent]`

So a complex double needs **two** `__shfl_sync` calls (real, imaginary) at 8 bytes each. That is
the register-resident alternative to a shared-memory transpose, and for **Regime A (`L = 6, 8`,
one volume per warp)** it is the natural one: the whole volume is in the warp's registers, so an
axis transpose is a lane permutation, and lane permutations are shuffles. No shared memory, no
barriers, no bank conflicts.

Volkov's FFT case study is the canonical statement of the trade and it is worth reading before
choosing (§6.5). His slide sequence "Fewer threads – lower shared memory traffic" enumerates it
exactly `[F-agent]`
<https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf>:

> "2 outputs/thread — 8 threads, 3 shuffles / 4 outputs — 4 threads, 1 shuffle / 8 outputs — 2
> threads, 1 shuffle / **16 outputs — 1 thread, no shuffles**"

**More points per thread ⇒ fewer shuffles ⇒ less data motion, at the cost of registers.** That
is the same knob as cuFFTDx's `ElementsPerThread` (§4.2) and the same knob §08 §3.4 found an
inversion point for on CPU.

### 6.5 Occupancy versus register pressure — and the published FFT example says go low

The register arithmetic, from NVIDIA `[F-agent]` (Best Practices Guide §10.3):

> "on devices of CUDA Compute Capability 7.0 each multiprocessor has 65,536 32-bit registers and
> can have a maximum of 2048 simultaneous threads resident (64 warps x 32 threads per warp).
> **This means that in one of these devices, for a multiprocessor to have 100% occupancy, each
> thread can use at most 32 registers.**" … "**register allocations are rounded up to the nearest
> 256 registers per warp.**"

sm_80 has the same 65,536 registers and 2048 threads `[F-agent]`, so **32 registers per thread is
the 100 %-occupancy budget and 255 is the hard ceiling.** **A complex double is 4 registers** (two doubles, two 32-bit registers each), so:

| complex points per thread | registers of *data* | occupancy ceiling from registers alone |
|---|---|---|
| 4 | 16 | 100 % |
| **8** | **32** | **100 % — exactly the budget, nothing left for temporaries** |
| 16 | 64 | 50 % |
| 23 | 92 | 25 % |
| 32 | 128 | 25 % |
| 36 | 144 | ~12.5 % |
| 63 | 252 | ~12.5 %, at the 255 ceiling |
| **64** | **256** | **impossible — exceeds the 255-register limit** |

A **64-point line held entirely in one thread needs 256 registers of data and therefore does not
fit at all**; a 36-point line is 144 registers and caps occupancy near 12.5 % before a single
temporary — which is precisely why cuFFTDx's thread-level double-precision
ceiling is 32 points (§4.1).

**And NVIDIA's own guidance is that this is fine:**

> "**Higher occupancy does not always equate to higher performance—there is a point above which
> additional occupancy does not improve performance. However, low occupancy always interferes
> with the ability to hide memory latency, resulting in performance degradation.**"
>
> "improving occupancy from 66 percent to 100 percent generally does not translate to a similar
> increase in performance. **A lower occupancy kernel will have more registers available per
> thread than a higher occupancy kernel, which may result in less register spilling to local
> memory; in particular, with a high degree of exposed instruction-level parallelism (ILP) it
> is, in some cases, possible to fully cover latency with a low occupancy.**"
> — Best Practices Guide §10.3 `[F-agent]`

**The canonical case study is an FFT, and it is ours.** Volkov, *Better Performance at Lower
Occupancy*, GTC 2010 (a conference presentation, not a peer-reviewed paper) `[F-agent]`:

> "**Batch of 1024-point complex-to-complex FFTs, single precision:** CUFFT 2.2 → CUFFT 2.3;
> Threads per block **256 → 64** (4x smaller thread blocks); Occupancy (G80) **33% → 17%** (2x
> lower occupancy); Performance (G80) **45 Gflop/s → 93 Gflop/s** (2x higher performance)"
>
> "**Two common fallacies: multithreading is the only way to hide latency on GPU; shared memory
> is as fast as registers.**"
>
> "Small FFT are done in registers; **Shuffles are done using shared memory**"
>
> "**Do more parallel work per thread to hide latency with fewer threads. Use more registers per
> thread to access slower shared memory less. Both may be accomplished by computing multiple
> outputs per thread.**"

His slide 73 is titled "**GFLOPS go up, occupancy goes down**" as points-per-thread goes 2 → 4
→ 8 → 16. (The plotted values are chart graphics and were not recoverable; only the axis ranges.)

**A corroborating measurement of the other direction** — Merry, *Efficient channelization on a
Graphics Processing Unit*, arXiv:2303.09886 `[F-agent]` <https://arxiv.org/pdf/2303.09886>:

> "as the number of taps goes up, the number of registers needed increases, the number of threads
> that can be run concurrently decreases and the GPU's ability to hide memory latency is reduced.
> With 32 taps, the theoretical occupancy … is 41.67%."

**The synthesis for us:** with §1.2 saying we are *block-starved* at small batch and §2 saying we
are bandwidth-bound at large batch, the two regimes want opposite things. At large batch, keep
points-per-thread high (8–16) and accept 50–100 % occupancy; at the small-batch end there are
not enough blocks to fill the machine anyway, so occupancy per SM is not the constraint and
points-per-thread should go higher still. **Make points-per-thread a compile-time constant and
sweep it — that is the single most valuable tuning parameter in the kernel** (and it is the one
both cuFFTDx and VkFFT expose).

### 6.6 Launch overhead, CUDA graphs, and persistent kernels: at our sizes this is the story

§2.2's floors are 4.4 ns to 5.4 µs *per transform*. Launch overhead:

* Zhang, Wahib, Zhang & Matsuoka, IPDPS 2020, DOI `10.1109/IPDPS47924.2020.00057`,
  arXiv:2004.05371 `[F-agent]` <https://arxiv.org/pdf/2004.05371> — Table I, V100/P100, CUDA 10:
  **traditional launch overhead 1081 ns; null-kernel total latency 8888 ns.** With the crucial
  methodology caveat, verbatim: "We found that directly using a null kernel would not give a
  correct result here. Because at this point the stream pipeline is not saturated enough: **the
  overhead tested would be larger than usual.** The kernel execution latency needs to be larger
  than a certain number. This value is around **5us for a single GPU**".
* NVIDIA developer blog, *Getting Started with CUDA Graphs* (Alan Gray, 2019) `[F-agent]`
  <https://developer.nvidia.com/blog/cuda-graphs/> — Tesla V100, a 2.9 µs kernel: **9.6 µs per
  kernel when synchronising every launch, 3.8 µs when launching asynchronously into a stream,
  3.4 µs with CUDA graphs**; graph instantiation costs "around 400μs" once, and "the first graph
  launch is around 33% slower that all subsequent launches".
* NVIDIA's CUDA-graph docs give the component breakdown as "Driver Operations (5-15 μs) …
  **Hardware Submission (1-5 μs)**" and "you can assume **~1-5 μs per kernel** for driver and
  hardware overhead", with the page's own caveat that these are estimates `[F-agent]`
  <https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/quantitative-benefits.html>.

**Read that against §2.2.** A *single* `6³` transform is 4.4 ns of memory traffic against
~1–4 µs of launch. **At `L = 6`, one launch is worth ~250–900 transforms.** The consequences are
structural, not cosmetic:

1. **`B = 1` is not a measurable configuration on this hardware.** The CPU brief scores `B = 1`
   and batched separately; on an A100 a `B = 1` "measurement" at `L = 6, 8, 13, 17` measures the
   launch path and nothing else. §8 replaces it with a defined *small-batch* point.
2. **The number of kernels is a first-order design variable.** A 3-pass structure at `L = 36`
   costs 2 extra launches ≈ 2–8 µs against a 960 ns floor — i.e. **the launches can cost more
   than the transform.** At `B = 4` (`L = 36`, 3.8 µs of floor) a 3-kernel implementation is
   dominated by launches. This is a much sharper version of §07 §1.6's fusion argument: on a GPU
   at these sizes you fuse to save *launches* as much as to save bandwidth.
3. **Use CUDA graphs, or a persistent kernel, or both** — but know what each buys.

**Persistent kernels.** Zhang, Wahib, Chen, Meng, Wang, Endo & Matsuoka, *PERKS: PERsistent
KernelS*, ICS '23, DOI `10.1145/3577193.3593705`, arXiv:2204.02064 `[F-agent]`
<https://arxiv.org/pdf/2204.02064>:

> "We propose an execution model for running memory-bound iterative GPU kernels: PERsistent
> KernelS (PERKS). In this model, the time loop is moved inside persistent kernel, and
> device-wide barriers are used for synchronization. We then reduce the traffic to device memory
> by **caching subset of the output in each time step in the unused registers and shared
> memory** … geomean speedup of **2.12x for 2D stencils and 1.24x for 3D stencils** over
> state-of-art libraries"

on A100 and V100, "All experimental results reported are done in double precision." And the
counterpoint, which they cite themselves — a device-wide `grid.sync()` is **not** cheaper than a
kernel launch (Zhang et al. IPDPS 2020 `[F-agent]`):

> "**No matter how small the grid is, it seems that it is still slower than the overhead of
> kernel launch we measured in Section IV. Single GPU grid synchronization might not bring about
> any benefit in performance, in comparison to implicit barrier methods.** Yet we argue that this
> performance difference is negligible (**at most 2.5us with two blocks/SM**) in real
> applications. In addition, using the implicit barrier instead would eliminate the possibility
> of data reuse in shared memory and registers."

**So for us: a persistent kernel is worth it only for `L ≥ 36`**, where the value is not the
barrier (it is not cheaper) but the *data reuse across passes in shared memory and registers* —
i.e. it is a way to implement §2.4's "keep the intermediate on-chip" without a second launch.
For `L ≤ 23` there is only one pass and nothing to be persistent about; there, the right answer
is one kernel and a CUDA graph if the batch is small.

### 6.7 The shared-memory carveout, and L1

The carveout is a per-kernel choice from `{0, 8, 16, 32, 64, 100, 132, 164}` KB, requested with
`cudaFuncSetAttribute(…, cudaFuncAttributePreferredSharedMemoryCarveout, …)`, and:

> "**For example, for devices of compute capability 8.0, 50% will map to a 100 KB carveout
> instead of an 82 KB one.** Setting the `cudaFuncAttributePreferredSharedMemoryCarveout` is
> considered a hint by the driver; the driver may choose a different configuration, if needed."
> — Programming Guide 8.x `[F-agent]`

and going above 48 KB per block additionally requires setting
`cudaFuncAttributeMaxDynamicSharedMemorySize` and using **dynamic** shared memory (§0.1).

Per-geometry (my arithmetic):

| L | volume in shared | smallest carveout that holds 1 | that holds 2 | blocks/SM at that carveout |
|---|---|---|---|---|
| 6 | 3.38 KiB | 8 KB | 8 KB | 32 (block-limited, not shared-limited) |
| 8 | 8.00 KiB | 8 KB | 16 KB | 20 at 8 KB (164/8) → block-limited at 32 |
| 13 | 34.3 KiB | 64 KB | 100 KB | 4 at 32 KB… 2 at 64 KB |
| 17 | 76.8 KiB | 100 KB | **164 KB (153.6 fits)** | 1 at 100 KB, **2 at 164 KB** |
| 23 | 190.1 KiB | **does not fit** | — | use registers |
| 36 | plane 20.25 KiB | 32 KB | 64 KB (2 planes, for `cp.async`) | 5 at 32 KB |
| 45 | plane 31.6 KiB | 32 KB | 64 KB | 5 at 32 KB |
| 64 | plane 64 KiB | 64 KB | 132 KB | 2 at 64 KB |

**`L = 17` is the one case where asking for the full 164 KB carveout changes the occupancy
class** — 2 volumes per SM instead of 1 — and it is worth the experiment. Everywhere else, ask
for the *smallest* carveout that fits, because the rest of the 192 KB is L1 and §6.1's coalescing
and §6.3's `cp.async ACCESS` variant both use it. Recall from §3.5a that **VkFFT asks for 100 KB
on NVIDIA and no more**, which is a real-world vote for leaving L1 alone.

The L2 side has its own lever (§2.4): `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, …)`
plus a per-stream `accessPolicyWindow`, in 2.5 MB increments `[F]` `[F-agent]`. Worth one
experiment at `L = 36, 45, 64`, where a chunk of the batch plus its intermediate is a few MB.

---

## 7. Target 7 — what to measure, and how the field measures it

### 7.1 Timing: `cudaEvent`, and never a bare CPU timer

NVIDIA's rule, verbatim from the CUDA C++ Best Practices Guide §9.1 `[F]` `[F-agent]`
<https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>:

> "**When using CPU timers, it is critical to remember that many CUDA API functions are
> asynchronous; that is, they return control back to the calling CPU thread prior to completing
> their work. All kernel launches are asynchronous** … Therefore, to accurately measure the
> elapsed time for a particular call or sequence of CUDA calls, **it is necessary to synchronize
> the CPU thread with the GPU by calling `cudaDeviceSynchronize()` immediately before starting
> and stopping the CPU timer.**"
>
> "Although it is also possible to synchronize the CPU thread with a particular stream or event
> on the GPU, these synchronization functions are **not suitable for timing code in streams other
> than the default stream**. … Because the driver may interleave execution of CUDA calls from
> other non-default streams, calls in other streams may be included in the timing."
>
> "The `cudaEventElapsedTime()` function returns the time elapsed between the recording of the
> start and stop events. This value is expressed in milliseconds and has a **resolution of
> approximately half a microsecond** … the timings are **measured on the GPU clock, so the timing
> resolution is operating-system-independent**."
>
> "Be aware that CPU-to-GPU synchronization points … imply a stall in the GPU's processing
> pipeline and should thus be used sparingly."

**Half a microsecond of event resolution against §2.2's 4.4 ns floor at `L = 6` settles the
argument by itself**: a single small transform cannot be timed. The scored quantity must be a
region containing enough work, and §8 sizes it so that resolution contributes < 0.1 %.

Note what the *libraries* actually do, which is weaker than NVIDIA's own advice: both VkFFT's and
cuFFT's benchmark drivers in this repository use `std::chrono::steady_clock` around a loop of
`num_iter` launches with **one** `cudaDeviceSynchronize()` at the end, and report the mean of 3
runs with a standard error (`ext/src/VkFFT/benchmark_scripts/*/src/sample_1001_*` `[F-src]`).
That is defensible for a millisecond-scale region and indefensible for a microsecond-scale one.
**We will do better than the reference implementations here, which is cheap and worth it.**

### 7.2 Clock locking, and the throttle check that must be part of every run

From the `nvidia-smi` documentation `[F-agent]` <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
and the tool installed on this machine (driver 560.35.03) `[M]`:

> "**`-lgc, --lock-gpu-clocks=MIN_GPU_CLOCK,MAX_GPU_CLOCK`** — Specifies `<minGpuClock,maxGpuClock>`
> clocks as a pair (e.g. 1500,1500) that defines closest desired locked GPU clock speed in MHz …
> **Supported on Volta+. Requires root.**" and locally: "Setting this will supersede application
> clocks and take effect regardless if an app is running." `[M]`
>
> "**`-lmc, --lock-memory-clocks`**" — same form for the memory clock. `[F-agent]` `[M]`
>
> "`-ac, --applications-clocks` — **This option is deprecated** and will be removed in a future
> CUDA release. Please use `-lmc` for locking memory clocks and `-lgc` for locking graphics
> clocks." `[F-agent]`
>
> "`-pm, --persistence-mode`" — "**When persistence mode is enabled the NVIDIA driver remains
> loaded even when no active clients … exist. This minimizes the driver load latency associated
> with running dependent apps, such as CUDA programs.**" and "it does not persist across
> reboots." `[F-agent]`
>
> **"Clocks Event Reasons** — Retrieves information about factors that are reducing the frequency
> of clocks. **If all event reasons are returned as 'Not Active' it means that clocks are running
> as high as possible.**" with `SW Power Cap` = "SW Power Scaling algorithm is reducing the clocks
> below requested clocks because the GPU is consuming too much power", `HW Slowdown` = "reducing
> the core clocks by a factor of 2 or more", etc. There is also a "Clock Event Reasons Counters"
> section giving "Counters, in microseconds, for the amount of time factors have been reducing the
> frequency of clocks." `[F-agent]`

**This is not optional on our part.** Our GPU is the **250 W** PCIe SKU (§0), against the 400 W
SXM part NVIDIA's 9.7 / 19.5 TFLOP/s figures are quoted from; `Max Clocks: Graphics 1410 MHz` but
`Default Applications Clocks: Graphics 765 MHz` `[M]`. A sustained DMMA kernel is a plausible
power-cap trigger, and a bandwidth-bound kernel is a plausible *nothing* — but you cannot know
which without reading the counters. On this machine, idle, `nvidia-smi -q -d PERFORMANCE` reports
every event reason "Not Active" and `Performance State: P0` `[M]`.

Nsight Compute's guidance says the same thing in stronger terms `[F-agent]`
<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>:

> "**For many metrics, their value is directly influenced by the current GPU SM and memory clock
> frequencies. For example, if a kernel instance is profiled that has prior kernel executions in
> the application, the GPU might already be in a higher clocked state and the measured kernel
> duration, along with other metrics, will be affected. Likewise, if a kernel instance is the
> first kernel to be launched in the application, GPU clocks will regularly be lower.**"
>
> "**Thermal throttling directed by the hardware or driver cannot be controlled by the tool and
> always overrides any selected options.**"
>
> "**When comparing results, we recommend to lock clocks with `nvidia-smi` externally before
> profiling and use `--clock-control none` for ncu.**"
>
> and, relevant to §2.4's L2 hazard: "Cache control: Nsight Compute **flushes all GPU caches by
> default between replay passes.** … If the workload is highly sensitive to cache state, it's
> recommended to use `--replay-mode application --cache-control none`".

**A published example of a benchmark that did this, and found something anyway** — Merry,
arXiv:2303.09886 `[F-agent]`:

> "**To improve reproducibility, we have locked the graphics and memory clocks** to 1575 MHz and
> 9251 MHz respectively … Despite this, **we have found that the performance of the
> post-processing kernel drops by 20–25% if it is repeated thousands of times in a tight loop**,
> so the results for that kernel are measured on 1000 iterations at a time. This does not seem to
> occur when mixed with the other kernels in real-world usage."

**A 20–25 % drop from repeating a kernel in a tight loop, with clocks locked.** That is exactly
the harness structure our CPU driver uses (auto-calibrated inner repeat count). Cap the inner
repeat count and interleave, or the leaderboard measures thermal/power state rather than code.

### 7.3 The performance metric: bandwidth first, GFLOP/s second

NVIDIA's own priority `[F-agent]` (Best Practices Guide §9.2):

> "**High Priority: Use the effective bandwidth of your computation as a metric when measuring
> performance and optimization benefits.**"
>
> "Effective bandwidth = ((B_r + B_w) ÷ 10⁹) ÷ time … B_r is the number of bytes read per kernel,
> B_w is the number of bytes written per kernel, and time is given in seconds."
>
> "Some calculations use 1024³ instead of 10⁹ … **It is important to use the same divisor when
> calculating theoretical and effective bandwidth so that the comparison is valid.**"
>
> "**The actual memory throughput shows how close the code is to the hardware limit**, and a
> comparison of the effective or requested bandwidth to the actual bandwidth presents a good
> estimate of how much bandwidth is wasted by suboptimal coalescing."

For a bandwidth-bound problem this is the right primary metric, and it makes the leaderboard
directly comparable to VkFFT's and TurboFFT's published numbers. `B_r + B_w = 32·B·L³` bytes for
an out-of-place forward transform.

The secondary metric, for continuity with the CPU phase and with the FFT literature, is the
**benchFFT convention**, and the definition has a proper source `[F-agent]`
<https://www.fftw.org/speed/method.html>:

> "**mflops = 5 N log₂(N) / (time for one FFT in microseconds)** for complex transforms … **This
> is not an actual flop count; it is simply a convenient scaling, based on the fact that the
> radix-2 Cooley-Tukey algorithm asymptotically requires 5 N log₂(N) floating-point
> operations.**"

with `N = L³`. That is the convention our CPU leaderboard's GF/s column already uses, so the two
phases stay comparable. **Report both, and label the GF/s column as a scaling and not a flop
count** — especially important here, because a dense-matrix kernel (§5) does far more real flops
than `5N log₂N` and a `5N log₂N`-scaled "GF/s" would understate it while a true-flop rate would
flatter it. Bandwidth is the honest number for both.

Note also tcFFT's different convention (`6 × 2 × log₂N × N × batch / time`) `[F-agent]` and
Verma et al.'s `10 N³log₂N³` for an FFT *pair* `[F-agent]` <https://arxiv.org/pdf/2202.12756> —
**three mutually incompatible "FFT GFLOP/s" conventions in the literature.** Always state which.

### 7.4 What the field excludes, and what it repeats

**Host↔device transfers are excluded, explicitly and universally.** The cleanest statement is
tcFFT's `[F-agent]`:

> "In the performance tests, **the data are first transferred from CPU to GPU and a plan is
> created. Then, the execute function are executed thousands of times and the average performance
> is reported. The time spent on the data transferring and plan creating are not counted, for a
> plan can be reused during the whole life of real applications.**"

VkFFT does the same (device buffers allocated and filled before timing) `[F-src]`, heFFTe does
the same, and NVIDIA's cuFFTMp post reports device-resident timings with "GPU application clocks
were set to the maximum" `[F-agent]`. **So: device-resident timing is the field's standard, and
our contract should match it — with an end-to-end number reported separately, since for a
consumer of this kernel the H2D/D2H cost is real.**

Repetition and statistics — three positions, and they disagree:

* **benchFFT / FFTW** `[F-agent]`: "**we compute enough repeated FFTs so that the total time is
  sufficient for accurate timing, and divide by the number of iterations to obtain the average
  time. Second, we repeat this averaging process eight times, and report the minimum average
  time (to avoid fluctuations due to system interrupts, cache priming, etcetera).**" — i.e.
  **min of 8 means**. This is what our CPU harness already does (minimum across 3 processes).
* **heFFTe / Ayala et al., IPDPS 2022** `[F-agent]`
  <https://www.netlib.org/utk/people/JackDongarra/PAPERS/Performance_Analysis-fft-ipdps22.pdf>:
  "we report the **average runtime of 8 FFTs** (4 forward and 4 backward), which are preceded by
  **2 FFTs to warm up the accelerators**."
* **Hoefler & Belli, *Scientific Benchmarking of Parallel Computing Systems*, SC '15** `[F-agent]`
  <https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf> — argues
  against exactly the benchFFT rule: "sometimes a simple minimum or maximum across the processes
  seems sufficient. **We recommend avoiding such non-robust measures.**" Their Rule 5: "Report if
  the measurement values are deterministic. For nondeterministic data, report confidence
  intervals of the measurement", with the model sentence "We collected measurements until the 99%
  confidence interval was within 5% of our reported means." Rule 3: "Use the arithmetic mean only
  for summarizing costs." Rule 11: "**If possible, show upper performance bounds to facilitate
  interpretability**" — i.e. print the roofline next to the result, which §8 does.

**Resolution:** report **both** the minimum (for continuity with the CPU leaderboard and with
benchFFT) **and** a confidence interval or spread (for Hoefler & Belli's Rule 5), and always
print the §2.2 floor beside them. That costs nothing and satisfies both conventions.

For calibration of how much variance to expect: a 2025 study of production GPU systems reports
"**up to 1.4× variability for nanoGPT [on Perlmutter], and on Frontier, DeepCAM exhibited up to
3.2× variability**", attributing it mainly to the network `[F-agent]`
<https://sc25.supercomputing.org/proceedings/posters/poster_files/post256s2-file3.pdf>. On a
single exclusive GPU with locked clocks we should see far less — our CPU harness sees 0.04 % on an
idle node — but the number to beat is "spread small enough that a 5 % change is visible".

### 7.5 The achievable bandwidth ceiling, from three independent sources

| source | figure | context |
|---|---|---|
| NVIDIA datasheet | **1,555 GB/s** | nominal peak, A100 40GB PCIe and SXM `[F]` `[F-agent]` |
| BabelStream via Tsai, Cojean & Anzt, arXiv:2008.08478 `[F-agent]` | **1.33–1.40 TB/s** | "For an array of size 8.2 GB … the A100 reaches a bandwidth between 1.33 and 1.4 TB/s" |
| VkFFT SC22 poster `[F-agent]` | **~1.3 TB/s** | "The peak global memory bandwidth of A100 is ~1.3TB/s" — the author's working figure |
| VkFFT's own FP64 A100 plot, my measurement (§2.3) | **1253–1296 GB/s** | achieved by a real FP64 batched FFT |

**These agree: the usable HBM ceiling is ≈1.3 TB/s, or 84–90 % of the datasheet number, and a
good FP64 FFT reaches essentially all of it.** Draw the roofline at 1.3 TB/s and treat 1.555 as
marketing. Note also Tsai et al.'s small-array warning `[F-agent]`: "we show the performance
ratio for increasing array sizes and observe **the A100 providing a lower bandwidth than the V100
GPU for arrays of small size**" — another reason §8 sizes the scored point large.

**Published %-of-peak achieved by FFT kernels**, for context:

* TurboFFT on A100-PCIE-40GB, FP32 and FP64: "**TurboFFT achieves 90% peak memory bandwidth.**
  The memory bandwidth is measured by 2× problem size divided by execution time." `[F-agent]`
* cuFFT C2C, single precision, batched 1D, on an RTX 3070 Ti with clocks locked: "the
  complex-to-complex FFT is memory bound: **arithmetic intensity is at most 4.5 flops per byte,
  and at least 87% of the memory bandwidth is used**" — Merry, arXiv:2303.09886 `[F-agent]`.
  Note this is the same conclusion as §2.2 reached by a different route, on a different part.
* tcFFT, V100 and A100, half precision: "In the bandwidth-bound cases … **the memory throughput
  of cuFFT is close to the theoretical bandwidth peak**, and our tcFFT can reach 96.4% to 97.8%
  performance of cuFFT." `[F-agent]`

**One widely-cited number that must NOT be used here.** heFFTe's "about 90 % of peak" is a
**network** roofline, not an HBM one: their model is built from "a node interconnection of
B = 25 GB/s" and the figure caption reads "Roofline performance from Eq. 3 and heFFTe performance
on a 3D FFT of size 1024³; using 40 MPI processes … per node" — Ayala, Tomov, Haidar & Dongarra,
*heFFTe: Highly Efficient FFT for Exascale*, ICCS 2020, DOI `10.1007/978-3-030-50371-0_19`
`[F-agent]` <https://www.iccs-meeting.org/archive/iccs2020/papers/121370260.pdf>. The same
applies to cuFFTMp's "more than 70% of the peak machine bandwidth" `[F-agent]`, which is
NVLink+InfiniBand. **Distributed-FFT papers are about the network and say nothing about a
single-GPU kernel.**

---

## 8. Deliverable (a) — the recommended benchmark contract for the GPU phase

This is a concrete proposal, meant to be implemented as written. It is the CPU contract in
`bench/geom/PANEL_BRIEF.md` adapted where the hardware forces a change, and every change is
justified by a section above.

### 8.1 The task and the layout — unchanged

Forward, unnormalized, complex-double 3D DFT of `L^3` over a batch of `B` volumes,
**out-of-place**, in **exactly the CPU layout**:

```
out[b][k0][k1][k2] = sum_j in[b][j0][j1][j2] * exp(-2*pi*i*(k0*j0+k1*j1+k2*j2)/L)
element (b,x,y,z) at index ((b*L + x)*L + y)*L + z
```

Keeping the layout identical is not a formality: §1.3 shows this layout is already perfectly
coalescible for `L ≤ 17`, and keeping it makes GPU and CPU results directly comparable and keeps
the same `check.py` usable. Both buffers are **device** allocations from `cudaMalloc` (which is
256-byte aligned, so the 128-byte alignment `cp.async` wants (§6.3) is guaranteed). `in` must not
be modified.

### 8.2 The API

```c
/* fft3d_gpu_api.h -- one .cu file per implementation, mirroring the CPU contract */
typedef struct fft3d_plan fft3d_plan;

const char *fft3d_name(void);
const char *fft3d_description(void);
int         fft3d_supports(int L);

/* ALL setup: device allocations, twiddle/matrix tables built on the host in
   long double and copied down, module load, carveout and L2-policy requests,
   CUDA-graph capture, autotuning sweeps.  Excluded from the score.            */
fft3d_plan *fft3d_create(int L, int batch, cudaStream_t stream);

/* THE MEASURED OPERATION.  Enqueues all work on `stream` and returns
   immediately; it MUST NOT synchronize, allocate, or memcpy H<->D.            */
void        fft3d_execute(fft3d_plan *p, const cuDoubleComplex *d_in,
                                          cuDoubleComplex *d_out);

void        fft3d_destroy(fft3d_plan *p);
```

Three deliberate differences from the CPU header:

1. **`fft3d_execute` takes device pointers and is asynchronous on a caller-supplied stream.**
   The driver owns synchronisation. An implementation that calls `cudaDeviceSynchronize()`
   internally is disqualified — it would serialise the pipeline and (§6.6) that alone is worth
   6 µs per call.
2. **`cudaStream_t` is passed at plan time** so an implementation may capture a CUDA graph in
   `fft3d_create` and replay it in `fft3d_execute`. Graph capture is setup; graph launch is
   measured. This is legitimate and (§6.6) it is the correct answer for the small-batch point.
3. **No allocation in `execute`.** All scratch is allocated in `create`, as on CPU.

### 8.3 What is timed

**Timed:** the wall-clock/event-clock span of `N_inner` back-to-back `fft3d_execute` calls on one
stream, bracketed by `cudaEventRecord` on the same stream, with `cudaEventSynchronize` after the
stop event only. Reported per transform: `elapsed / (N_inner · B)`.

**Excluded:** compilation, `fft3d_create` (reported separately, as on CPU), host↔device transfers,
warm-up. **Also reported separately:** an end-to-end number that includes a `cudaMemcpy` of `in`
down and `out` back, because for a real consumer that cost is real and because it is the number
that makes a GPU-versus-CPU comparison honest.

**Why `cudaEvent` and not `std::chrono`:** §7.1. **Why a loop and not a single call:** the event
resolution is ~0.5 µs (§7.1) and the `L = 6` floor is 4.4 ns (§2.2).

`N_inner` is auto-calibrated as on CPU, but with a **cap** — §7.2's measured 20–25 % drop from
tight-loop repetition means `N_inner` must not run for more than a few milliseconds. Recommend
`N_inner` chosen so the timed span is 20–50 ms, and the *sample* repeated 20 times with the
samples interleaved across backends rather than all of one backend then all of the next.

### 8.4 The batch points — and `B = 1` is dropped

**`B = 1` is not a scored configuration.** §6.6: at `L = 6` one transform is 4.4 ns of work
against 1–4 µs of launch overhead, so a `B = 1` number measures the launch path. Replace it with
two defined points, both specified by *working set* rather than by `B`, so that all eight
geometries sit in the same place on the memory hierarchy:

| L | **`B_L2`** (in+out ≈ 32 MiB, **L2-resident**) | **`B_HBM`** (one buffer ≈ 1 GiB, **HBM-bound**) | HBM floor at `B_HBM` |
|---|---|---|---|
| 6 | 4,855 | **310,608** | 1.65 ms |
| 8 | 2,048 | **131,004** | 1.65 ms |
| 13 | 477 | **30,456** | 1.65 ms |
| 17 | 213 | **13,608** | 1.65 ms |
| 23 | 86 | **5,508** | 1.65 ms |
| 36 | 22 | **1,404** | 1.61 ms |
| 45 | 12 | **648** | 1.45 ms |
| 64 | 4 | **216** | 1.39 ms |

* **`B_HBM` is the primary score.** One buffer ≈ 1 GiB matches VkFFT's published methodology
  ("500MB to 1GB of data", §3.3) so our numbers are directly comparable to theirs; the working set
  is 25–50× L2 so the measurement is unambiguously HBM-bound (§2.4); and the 1.4–1.65 ms floor
  makes launch overhead and event resolution contribute under 0.1 % (§6.6, §7.1). `B_HBM` is
  chosen as the largest multiple of 108 with one buffer ≤ 1 GiB, so blocks divide evenly across
  the 108 SMs.
* **`B_L2` is a labelled secondary score**, reported as "L2-resident" and never compared against
  `B_HBM`. It is where a fused single-kernel design and an L2-blocked multi-pass design can
  actually differ (§2.4), and where CUDA graphs matter. Note `B_L2 < 108` for `L = 45, 64`, so a
  one-volume-per-block kernel cannot fill the machine there; that is a real property of the
  regime, not a defect of the contract, and it is exactly the case §1.2 says must split the
  volume across blocks.
* Optionally a third point at in+out ≈ 8 GiB for `L ≥ 36`, to check that nothing degrades as the
  batch grows past what fits comfortably in the 40 GiB of HBM.

### 8.5 Correctness

* **Gate: relative L2 error < `1e-12`** against the same numpy/`slow_dft` reference the CPU phase
  uses, on the host, after a D2H copy. A correct implementation should land near `1e-16`; anything
  above `1e-14` means a real problem. Unchanged from the CPU contract, and §5.1 confirms that even
  a DMMA-based kernel has no excuse.
* **Repeatability: bit-identical output on a second call with the same plan and input.** Same
  clause as the CPU brief, and it is if anything more important here — cuFFT's own documentation
  promises determinism only for a fixed plan/version/GPU (§3.1), and a kernel whose block
  scheduling changes its reduction order would fail this.
* **Check at both batch points**, and additionally at a small odd `B` (e.g. `B = 109`) to catch
  tail-handling bugs in kernels that assume `B` is a multiple of the block count.
* **Check the throttle counters after every scored run** (§7.2). A run in which any "Clocks Event
  Reason" was Active is void.

### 8.6 The baselines

| baseline | what it is | why |
|---|---|---|
| **cuFFT `cufftPlanMany` Z2Z, rank 3** | the vendor library, CUDA 12.2 | the number everyone will ask about. Call it exactly as §3.1 says, check `cufftGetSize` first (§3.1: worst case is 8× the buffer) |
| **cuFFT, 3 separate 1D batched plans** | one `cufftPlanMany` per axis with `istride`/`idist` set from §3.1's addressing formula | isolates how much of cuFFT's 3D performance is its own axis fusion, and gives the "3 passes" reference point that §2.3 infers cuFFT is paying |
| **VkFFT** (`ext/src/VkFFT`, CUDA backend, FP64, `FFTdim = 3`, `numberBatches = B`) | the strongest existing GPU FFT | §3.3 says which library is faster in FP64 on A100 is genuinely open; measure it, do not assume |
| **`baseline_matrix`** — dense DFT matrix per axis, naive CUDA cores, no tricks | the library-free floor | as on CPU: beating it means nothing, but it is the reference implementation the correctness gate is written against, and (§5) it is also the *starting point* of the most promising strategy, so its number is genuinely informative here in a way it was not on CPU |
| **the roofline** | `32·B·L³ / 1.3 TB/s` printed beside every row | Hoefler & Belli Rule 11 (§7.4). Without it a reader cannot tell 60 % of peak from 95 % |

### 8.7 Rules

1. **No FFT or GEMM library inside `fft3d_execute`.** Not cuFFT, not cuFFTDx, not VkFFT, not
   cuBLAS, not CUTLASS, not cuTENSOR, not cuFFTDx-via-NVRTC. The arithmetic must be the
   implementation's own. `libm`/host `long double` during `fft3d_create` is fine and encouraged
   (§5.8, and §07 §5's twiddle-precision conclusion).
2. **Tensor cores are allowed and encouraged.** `mma.sync…f64` via inline PTX, or the `wmma` C++
   API, are *instructions*, not libraries. §5 is an argument for using them at `L = 13, 17, 23`.
3. **Single GPU, single stream, no MPS, no MIG.** MIG would disable the L2 set-aside (§2.4).
4. **Precision:** all arithmetic in FP64. No FP32/TF32/FP16 path, no split-precision emulation.
   §5.7 is why: every published shortcut misses the gate by orders of magnitude, and a `1e-12`
   gate is the whole reason this project's results mean anything.
5. **Everything else is fair game:** hand-written PTX, `cp.async`, warp shuffles, persistent
   kernels, CUDA graphs, custom carveouts, L2 residency policy, generated code selected at plan
   time, autotuning inside `fft3d_create`.
6. **Report which point you optimised for** (`B_L2` or `B_HBM`), as the CPU brief requires for
   batched vs non-batched.

### 8.8 Environment discipline

```bash
# before any scored run, as root on the exclusive node:
nvidia-smi -pm 1                        # persistence mode (does not survive reboot)
nvidia-smi -lgc 1410,1410               # lock the SM clock
nvidia-smi -lmc 1215,1215               # lock the memory clock
nvidia-smi -q -d PERFORMANCE            # record: every event reason must be "Not Active"
nvidia-smi -q -d CLOCK                  # record the actual clocks
# after the run:
nvidia-smi -q -d PERFORMANCE            # re-check; any Active reason voids the run
nvidia-smi -rgc ; nvidia-smi -rmc       # restore
```

`-lgc` and `-lmc` require root (§7.2); if root is unavailable, record the clocks and the event
counters before and after every sample and report them, and treat any run with a nonzero
throttle counter as void. Request the whole node (`--exclusive`) on `a100l`/`a100r` and one GPU,
so no other process shares the L2 or the power budget.

Record in every results file: driver version, CUDA version, `nvcc -V`, the compile line, the GPU
UUID, ECC state, persistence state, the locked clocks, and the throttle counters. §07 and §08 both
show how much a single unstated hardware fact can cost a corpus.

---

## 9. Deliverable (b) — per-geometry opening strategy

Read the columns as: **the two or three kernel structures to build first**, in order, with the
section that justifies each. Every geometry's structures share the same three invariants, which
come out of §2 and are worth stating once:

* **One global read and one global write of the volume, or as close as the capacity allows.**
  That is the entire performance story (§2.2, §2.3).
* **Load and store in 128-byte contiguous runs per thread** — 8 complex doubles (§6.1: 208 GB/s
  at 16 B/thread against 836 GB/s at 128 B/thread).
* **Make every shared-memory row stride an odd number of complex doubles** (§6.2). For
  `L = 6, 8, 36, 64` this means padding; for `L = 13, 17, 23, 45` the natural stride is already
  odd.

Reference numbers for every row: the HBM floor per transform from §2.2 and the CPU panel's best
single-core time from `bench/geom/results/panel_r5/leaderboard.txt` (in-tree measurement).

| L | HBM floor/transform | best CPU single-core (r5) | ratio |
|---|---|---|---|
| 6 | 4.4 ns | 0.219 µs | 50× |
| 8 | 10.5 ns | 0.570 µs | 54× |
| 17 | 101 ns | 15.22 µs | 151× |
| 36 | 960 ns | 120.4 µs | 125× |

### 9.1 `L = 6` — 216 complex doubles, 3.38 KiB

| # | structure | why | §|
|---|---|---|---|
| **1** | **One volume per warp, all three axes in registers, transposes by `__shfl_sync`.** 216 complex = 11 % of a warp's register file. Load 27 lanes × 8 complex (128 B each), transform, shuffle, store. Block = 8–32 warps = 8–32 volumes. | Regime A. Zero shared memory, zero barriers, zero bank conflicts, one read and one write. This is the structural optimum for the smallest geometry and there is nothing above it. | §1.1, §6.4 |
| **2** | **One volume per warp, all three axes in *shared* memory, stride padded 6 → 7.** Simpler to write than the shuffle version; 8 volumes per block is 27 KiB with padding. | The fallback if register pressure or shuffle count bites. `gcd(6,8) = 2`, so unpadded costs a 2-way conflict — mild, but free to fix. | §1.1, §6.2 |
| **3** | **Dense 6×6 DFT matrix per axis, on CUDA cores, PFA-free and twiddle-free.** 144 flop/point is **0.72× the memory floor — the arithmetic is invisible**. | §5.4: at `L = 6` a dense matvec is *free even without tensor cores*. It is also the simplest possible kernel: three matvecs, no butterflies, no permutations, no twiddle table beyond the 6×6 matrix in constant memory (and a broadcast read of the matrix costs nothing, §6.2). **Do not use DMMA here** — 6 pads to 8×8, a 1.78× waste, for arithmetic that is already free. | §5.4, §5.8 |

**Watch for:** at `B_HBM = 310,608` this is 310,608 volumes and the grid is enormous — make sure
the index arithmetic is 32-bit where possible (`useStrict32BitAddress` is a real VkFFT option for
this reason) but note the buffer is 1 GiB so element indices exceed 2³² only above 4 GiB. And
`L = 6`'s tiny volume means the batch dimension is the only parallelism; get the load pattern
right first and the rest follows.

### 9.2 `L = 8` — 512 complex doubles, 8.00 KiB

| # | structure | why | § |
|---|---|---|---|
| **1** | **Thread ↔ line: 64 threads (2 warps) per volume, each thread owning one full 8-point line = 8 complex = 32 registers = exactly 128 bytes.** Radix-8 straight-line codelet in registers, three times, with a shared-memory transpose between axes at **stride 9**. Block = 512 threads = 8 volumes = 73.7 KiB shared. | Every constant lines up: 8 complex is simultaneously the 128-byte coalescing optimum (§6.1), the 100 %-occupancy register budget (§6.5), a complete 1D transform (no inter-thread communication during the codelet), and a `cp.async` multiple. **This is the cleanest geometry in the whole set and the one to write first, in any language.** | §1.1, §6.1, §6.5 |
| **2** | **One volume per warp, 16 complex per lane (64 registers), fully in registers with shuffle transposes.** | Halves the block count and eliminates shared memory entirely; 50 % occupancy ceiling from registers, which §6.5's Volkov evidence says is fine or better. Directly comparable to structure 1, and the comparison is *the* experiment that tells the panel where the register/shared crossover is on this part. | §6.4, §6.5 |
| **3** | **Dense 8×8 DFT matrix per axis — and this is the place to *learn* `mma.m8n8k4`.** `L = 8` is an **exact** fit for the instruction shape (M = 8, K = 8 = 2 mma steps, N = the batch), waste factor **1.00**. On CUDA cores the dense matvec is 0.96× the memory floor (break-even); on DMMA it is 0.48×. | Even though structure 1 should win, `L = 8` is the only geometry where the DMMA path has zero padding waste, so it is where an implementer should debug the fragment mapping before applying it at `L = 13, 17, 23` where it actually matters. | §5.2, §5.4, §5.8 |

**Watch for:** `gcd(8,8) = 8` — an unpadded `8×8×8` shared array is the **worst-case 8-way bank
conflict** and §6.2's arithmetic says that alone makes shared memory the bottleneck by 2×. Pad to
9. This is the single most likely silent performance bug at this geometry.

### 9.3 `L = 13` — 2197 complex doubles, 34.3 KiB

| # | structure | why | § |
|---|---|---|---|
| **1** | **Volume in shared memory (21 % of a block's max, natural stride 13 is already odd), thread ↔ line: 169 threads per volume rounded up to 192 (6 warps, 88 % lane utilisation), 13 complex = 52 registers each.** Radix-13 straight-line codelet, three axes, one read, one write. | Regime B/C. `13` is a native VkFFT radix (§3.5e) so a hand-written 13-point codelet has a known-good competitor; the shared stride needs no padding; and 34.3 KiB lets 4 volumes share a 164 KB carveout. | §1.1, §3.5e, §6.2 |
| **2** | **One volume per two warps in registers** (2197 complex / 64 lanes = 34.3 complex = 137 registers per lane) with shuffle transposes. | The register file is 4× shared memory (VkFFT's `registerBoost` premise) but 137 registers caps occupancy near 12.5 %; this is exactly the point where §6.5's trade must be measured rather than argued. Note VkFFT itself sets `registerBoostNonPow2 = 0` and would *not* take this route (§1.4). | §1.4, §6.5 |
| **3** | **Conjugate-symmetric dense real matrix per axis on DMMA** — a `7×13` real matrix (cos and sin parts) applied to the real and imaginary halves. 168 flop/point at a 1.52× padding waste is **0.59× the memory floor**. | §5.4. `L = 13` is the first geometry where DMMA is *required* to make the dense route free (on CUDA cores it is 1.56× over). A cheaper opening experiment than `L = 17` because 13 → 16 padding is simpler than 17 → 24. | §5.4, §5.8 |

### 9.4 `L = 17` — 4913 complex doubles, 76.8 KiB. **The geometry with the most headroom.**

The CPU phase's `L = 17` result (dense conjugate-symmetric matvec, **5.37×** the best library) and
§5.4's DMMA crossover point (`L ≤ 16.7` at 100 % efficiency, `L = 17` at 1.02× of the floor, and
**0.84×** with conjugate symmetry) land on the same size. Both cuFFT and VkFFT reach for a dense
matrix here too (§3.6). This is where the GPU phase should spend its effort.

| # | structure | why | § |
|---|---|---|---|
| **1** | **One volume per block, all three axes fused in shared memory. 76.8 KiB of a 100 KB or 164 KB carveout; natural stride 17 is odd so no padding needed. Thread ↔ line: 289 threads (17 complex = 68 registers), or 17×17 as a 2D block.** One global read, one global write. | Regime C, and this is the *`fft_3d_cube_single_block`* structure that cuFFTDx ships — in FP32 only (§4.3). We would be doing the FP64 version of a structure NVIDIA demonstrates but never published in double. Ask for the **164 KB** carveout and check whether 2 volumes per SM beats 1 (§6.7 — `L = 17` is the only geometry where the carveout changes the occupancy class). | §1.1, §4.3, §6.7 |
| **2** | **Structure 1, with each axis evaluated as a conjugate-symmetric dense real matrix product on `mma.sync…m8n8k4.f64`.** Per axis: a `9×17` real cos-matrix and a `9×17` real sin-matrix (or the stacked `18×17`), against `X` reshaped to `17 × 289` per volume — `N` grows with the volumes per block, so batch 8–16 volumes into one GEMM if shared memory allows. Fill the rest of the output rows by conjugation. | **The single most promising untried idea in the GPU phase.** §5.4: 216 flop/point on an effective ~9.3 TFLOP/s (after 17 → 24 / 17 → 20 padding) is 0.84× the memory floor, i.e. the arithmetic disappears. §5.1: DMMA is IEEE FP64, so the `1e-12` gate is safe. §5.5: hand-write the PTX; CUTLASS is both banned and structurally wrong for a 17×17 matrix. §5.5 also says the real win is fewer shared-memory reads of the matrix, not the flop rate. | §5.1–§5.5 |
| **3** | **Structure 1 with a Rader kernel** (`P − 1 = 16 = 2⁴`, the friendliest possible Rader) **or a hand-derived 17-point Winograd module**, one line per thread. | The baseline the dense route must beat, and the thing both libraries do. VkFFT's own guide says the FFT-convolution Rader is "**Better than direct multiplication version for almost all primes (except small ones, like 17-23 on some GPUs)**" (§3.6) — so this is genuinely a coin flip and must be measured, not assumed. On CPU the dense matvec beat Rader by 1.12× and Winograd by 1.19× at r5. | §3.6 |

**Watch for:** 289 threads is 9.03 warps, so a 289-thread block wastes 9.7 % of its lanes; a
320-thread block with 289 active is the same thing stated differently. Consider instead 256
threads each handling `⌈289/256⌉` lines with a masked tail, or 512 threads handling two volumes'
worth of lines. Also: 4913 is not a multiple of 8, so the 128-byte load runs (§6.1) leave a tail
of 1 complex double — pad the shared allocation to 4920 and mask.

### 9.5 `L = 23` — 12,167 complex doubles, 190.1 KiB. **The one that does not fit.**

190.1 KiB is **1.17× the 163 KiB per-block shared maximum** (§1.1). This is the only geometry in
the set where the obvious structure is unavailable, which makes it the most interesting after 17.

| # | structure | why | § |
|---|---|---|---|
| **1** | **One volume per block, fused, held in *registers*: 529 threads (23 complex = 92 registers each) = 74 % of the SM's 64 K registers, using shared memory only as the axis-transpose exchange buffer.** One block per SM, one global read, one global write. | The only single-pass option. It is exactly VkFFT's `registerBoost` premise taken to its limit, and the register file (256 KiB) is the only container on the SM big enough. 92 registers of data leaves ~160 for temporaries — enough for a 23-point codelet. Natural stride 23 is odd, so the exchange buffer needs no padding. | §1.1, §1.4, §6.2 |
| **2** | **Two-pass: fuse the two contiguous-ish axes over a `23×23` plane (8.48 KiB, many planes per block), then a second kernel for the third axis.** 2 reads + 2 writes instead of 1 + 1. | The safe structure, and the honest comparison point: §2.3's inference is that cuFFT's visible deficit *is* extra passes, so a 2-pass kernel here should land near 1/2 of structure 1's bandwidth and that is the number to check the model against. Plane-per-block also fixes the occupancy problem structure 1 has (1 block per SM). | §1.1, §2.3 |
| **3** | **Conjugate-symmetric dense real matrix per axis on DMMA** — a `12×23` real matrix. Padding waste is only **1.09×** (23 → 24 in both M and K), the *second best* in the whole set, giving 288 flop/point at ~17.9 TFLOP/s = **0.75× the memory floor**. | §5.4. `L = 23` is where DMMA is most efficient *and* most needed: on CUDA cores the dense route is 2.77× over the floor, the worst ratio of the small geometries. And 23 is a prime with no radix kernel in either library (§3.5e), so the competition is Rader or Bluestein and cuFFT's own docs warn that Bluestein loses accuracy (§3.2). | §3.2, §5.4 |

### 9.6 `L = 36` — 46,656 complex doubles, 729 KiB

| # | structure | why | § |
|---|---|---|---|
| **1** | **Two-pass, plane-per-block. Kernel 1: load a `(y,z)` plane (1296 complex = 20.25 KiB, **pad the stride 36 → 37**), transform `z` then `y` entirely in shared memory, write back. Kernel 2: transform `x` with a `(x,z)` slab read coalesced along `z`.** Grid = `B·36` blocks in each kernel — 50,544 blocks at `B_HBM = 1404`, plenty. | Regime D. Two passes instead of three, both fully coalesced (§1.3), both with a padded stride (`gcd(36,8) = 4`, so unpadded is a 4-way conflict that §6.2 says *exactly* consumes the shared-memory headroom). `36 = 4·9` is coprime, so Good–Thomas removes the inter-factor twiddles inside each axis, which frees registers rather than flops. | §1.3, §1.5, §6.2 |
| **2** | **Structure 1 plus `cp.async` double buffering of the next plane, plus an L2 residency policy over a chunk of the batch.** Two planes is 40.5 KiB of a 64 KB carveout. | §6.3: `L ≥ 36` is where `cp.async` has something to overlap. §2.4: at `L = 36` a chunk of ~13 volumes plus its intermediate fits the 40 MiB L2, whose 7.2 TB/s is 4.6× HBM, so the *intermediate* pass can be made free. This is the GPU form of §08 §1.9's L2-blocking-across-the-batch, the technique three independent CPU sources endorsed. | §2.4, §6.3 |
| **3** | **Single-pass persistent kernel over a batch chunk**, keeping the intermediate volume in shared memory + registers across all three axes with `grid.sync()` between axes. | PERKS reports "**1.24x for 3D stencils**" on A100/V100 in double precision from exactly this pattern (§6.6), and its value here is data reuse, not the barrier (which is *not* cheaper than a launch — §6.6). Worth one experiment; expect it to be hard and to win modestly if at all. | §6.6 |

**Watch for:** a `36³` volume is 746,496 B, which is **above the 2¹⁸-byte threshold** VkFFT's
author warns about for "distant, but coalesced" accesses (§3.5b). Any warp that spans two volumes
of the batch is in that regime; the documented remedy is 128-byte-per-thread accesses, which
§6.1's measured curve wants anyway. Also: a butterfly at `L = 36` is 2.42 flop/byte against a
6.24 machine balance — 2.6× under. **Do not try to be clever with arithmetic here; every cycle
spent on flops at `L = 36` is spent in the shadow of the memory system.**

### 9.7 `L = 45` — 91,125 complex doubles, 1.39 MiB

| # | structure | why | § |
|---|---|---|---|
| **1** | **Two-pass, plane-per-block, exactly as `L = 36`.** A `45×45` plane is 2025 complex = 31.6 KiB, fits a 32 KB carveout; **natural stride 45 is odd, so no padding is needed** (unlike 36 and 64). Grid = `B·45`. | §1.5, §6.2. This is the same kernel as `L = 36` with different constants and one fewer thing to get wrong. | §1.5, §6.2 |
| **2** | **`45 = 9 · 5`, coprime — Good–Thomas per axis with dense 9-point and 5-point codelets.** A dense 9×9 or 5×5 matvec is trivially free (§5.4 extrapolated: `24·9 = 216` flop/point-of-a-9-line, far under the floor), and PFA removes the inter-factor twiddles entirely. | The hybrid §5.8 recommends for the large geometries: butterfly *structure*, dense *codelets*, and the codelets are exactly the sizes the arithmetic budget makes free. No DMMA needed — the CUDA-core dense route already fits at these factor sizes. | §5.4, §5.8 |
| **3** | As `L = 36` structure 2: `cp.async` double buffering + L2 blocking across a chunk of ~14 volumes. | §2.4, §6.3. | §2.4, §6.3 |

### 9.8 `L = 64` — 262,144 complex doubles, 4.00 MiB

| # | structure | why | § |
|---|---|---|---|
| **1** | **Two-pass, plane-per-block. A `64×64` plane is 4096 complex = 64 KiB exactly — pad the stride 64 → 65** (`gcd(64,8) = 8`, the worst-case 8-way conflict), which makes it 65 KiB and needs a 100 KB carveout. **Two threads per 64-point line** (32 complex = 128 registers each) or four (16 complex = 64 registers) — a 64-point line **cannot** live in one thread (256 registers > the 255 ceiling, §6.5). | Regime D at its largest. The 8-way conflict is the dominant risk (§6.2: unpadded, shared memory becomes the bottleneck by 2× and the kernel runs at half speed for a reason unrelated to bandwidth). | §1.5, §6.2, §6.5 |
| **2** | **`64 = 8·8`: radix-8 twice per axis, with the radix-8 codelet as a dense 8×8 matvec** (free on CUDA cores per §5.4, and an exact `m8n8k4` fit if you want DMMA). Each thread owns 8 complex = 32 registers = 128 bytes — the same sweet spot as `L = 8` structure 1. | The `L = 8` kernel *is* the `L = 64` codelet. Whatever wins at `L = 8` should be lifted here verbatim, which is the strongest cross-geometry transfer in the set. | §5.4, §9.2 |
| **3** | **L2-blocked three-pass over batch chunks of 4–5 volumes.** In + out for 5 volumes is 40 MiB = exactly L2, so a three-pass structure whose middle pass hits L2 costs `1 + 1/4.6 + 1/4.6` HBM passes instead of 3. | §2.4. `L = 64` is the geometry where L2 blocking has the most to give, because it is the one where a single volume (4 MiB) is a meaningful fraction of the 40 MiB L2 and where the alternative (a 3× traffic multiplier) is most expensive. `B_L2 = 4` in §8.4 is exactly this chunk size. | §2.4 |

### 9.9 The cross-geometry summary

| L | primary structure | volume lives in | passes | DMMA verdict | shared padding |
|---|---|---|---|---|---|
| **6** | warp-resident fused, shuffles | registers | 1 | **no** (dense already free on CUDA cores; 1.78× pad waste) | 6 → 7 (2-way) |
| **8** | 64 threads/volume, line-per-thread | registers + shared | 1 | **optional** (exact fit; use it to learn) | **8 → 9 (8-way!)** |
| **13** | block-resident fused | shared (21 %) | 1 | **yes** (0.59× floor) | none needed |
| **17** | block-resident fused | shared (47 %) or registers (30 %) | 1 | **yes — the headline idea** (0.84× floor) | none needed |
| **23** | block-resident fused **in registers** | registers (74 %) | 1 | **yes** (0.75× floor, only 1.09× pad waste) | none needed |
| **36** | plane-per-block, 2 kernels | shared per plane | 2 | **no** (1.20× floor; butterfly is 2.6× under) | **36 → 37 (4-way)** |
| **45** | plane-per-block, 2 kernels, PFA 9×5 | shared per plane | 2 | **no** (1.53× floor) | none needed |
| **64** | plane-per-block, 2 kernels, radix-8² | shared per plane | 2 | **no** for the whole axis; **yes/free** for the 8-point codelet | **64 → 65 (8-way!)** |

---

## 10. Corrections to §07 (and two to §08)

§07 is the corpus's only prior accelerator section and it was written to mine the GPU world for
CPU lessons. Read against an actual A100 and against VkFFT's shipped source, five of its claims
need amending. Both sides are cited in each case.

### 10.1 VkFFT's radix set is 2–16 and 32, not "2/3/4/5/7/8/11/13"

§07 §1.2 and §08 §5.7 both quote VkFFT's README: "Radix-2/3/4/5/7/8/11/13 FFT" `[F-agent]`
<https://raw.githubusercontent.com/DTolm/VkFFT/master/README.md>. The shipped code
(`vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixKernels.h`, v1.3.4) is a single
`switch (radix)` with cases `2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32` `[F-src]`.

**Why it matters:** §07 and §08 both reasoned that `L = 6` and `L = 36` would be built from
2·3 and 4·9 factor kernels. In fact VkFFT has a **native radix-6** kernel and native 9, 12, 15 and
16 kernels, so `L = 6` is one stage and `L = 36`, `L = 45`, `L = 64` are each a product of two
native stages. **Six of our eight geometries are one or two native VkFFT radix stages**, which
raises the bar at `L = 6, 8, 13, 36, 45, 64` and leaves `L = 17, 23` as the opening (§3.5e).

### 10.2 cuFFT does not fall off a cliff at 17 — it has a native radix for every prime below 128

§07 §1.2 quotes the SC22 poster: "cuFFT does not use Rader's algorithm in FP32 and switches to
Bluestein's algorithm for primes after 17" `[F-agent]`, and concludes that both libraries handle
17 badly. NVIDIA's own documentation says otherwise `[F-agent]`
<https://docs.nvidia.com/cuda/cufft/index.html>:

> "**There are also radix-m building blocks for other primes, m, whose value is < 128.** When the
> length cannot be decomposed as multiples of powers of primes from 2 to 127, Bluestein's
> algorithm is used."

Both can be true: the poster's next sentence is that cuFFT "implements it as a **direct matrix
multiplication**" `[F-agent]`, i.e. cuFFT's radix-17 is a dense matvec rather than Rader, which is
why the poster's author describes cuFFT as not using Rader. **The operative correction is that
cuFFT's `L = 17` and `L = 23` are dense matrix multiplications, not Bluestein**, and therefore
that the CPU panel's winning `L = 17` formulation is what the vendor library already does. That
changes the framing of the `L = 17` opportunity from "the libraries have no algorithm" to "the
libraries have *our* algorithm and we have to implement it better" (§3.2, §3.6).

### 10.3 The register-file-vs-shared-memory ratio on an A100 is 1.56×, not 4×

§07 §4.1 makes this its headline: "On an NVIDIA SM the register file is **4× larger** than the
usable shared memory, so the optimizer's move is to push data *out of* shared memory *into*
registers… **This is the single most important asymmetry in this whole section.**" It is sourced
to VkFFT's own comment, which §07 quotes accurately and which is still in the tree `[F-src]`
(`vkFFT_Structs.h:277`):

```c
pfUINT registerBoost; //specify if register file size is bigger than shared memory and can be
                      //used to extend it X times (on Nvidia 256KB register file can be used
                      //instead of 32KB of shared memory, set this constant to 4 to emulate
                      //128KB of shared memory). Default 1
```

**But that comment's "32KB of shared memory" is not the A100-in-CUDA number.** VkFFT's own API
guide gives the per-API figures: "48KB - Nvidia GPUs with Vulkan/OpenCL API, 64KB - AMD GPUs,
**100KB - Nvidia GPUs in CUDA API**" `[F-agent]`, and NVIDIA documents the hard maximum as
**164 KB per SM / 163 KB per block** with a 256 KB register file `[F-agent]` `[F]`. So the true
ratios on our part are:

| comparison | ratio |
|---|---|
| 256 KB register file per SM ÷ **164 KB** max shared per SM | **1.56×** |
| 256 KB ÷ **163 KB** max shared per block | **1.57×** |
| 256 KB ÷ **100 KB** (what VkFFT actually asks for) | 2.56× |
| 256 KB ÷ 48 KB (default static limit, and the Vulkan/OpenCL limit) | 5.3× |
| 256 KB ÷ 32 KB (the comment's figure — a pre-Volta/Vulkan-era number) | 8× |

**The direction of §07's asymmetry is right and its magnitude is wrong for this hardware.** On an
A100 in CUDA the register file is only about 1.6× the shared memory, not 4×, and §07's "single
most important asymmetry" is a smaller effect than it says. Two consequences for us: (i) the
choice between register-resident and shared-resident at `L = 13, 17, 23` (§9.3–§9.5) is genuinely
close rather than obviously favouring registers; and (ii) **VkFFT sets
`registerBoostNonPow2 = 0`** `[F-src]`, so it does not take the register route for six of our
eight sizes at all (§1.4).

### 10.4 "192 KB of shared memory per block" is wrong, and it is a common slip

§07 §1.2 quotes the SC22 poster's "64KB vs 192KB" for MI250 vs A100 `[F-agent]`, and TurboFFT's
paper makes the slip explicitly: "the FFT size cannot fit into a threadblock's shared memory, e.g.
64KB for T4, and **192 KB for A100**" `[F-agent]` <https://arxiv.org/html/2405.02520v1>. **192 KB
is the *unified* L1 + shared + texture capacity per SM; the shared carveout maxes at 164 KB per SM
and a block can address 163 KB** `[F-agent]` (Programming Guide 8.x; §0.1). The 29 KB difference
matters at exactly one of our geometries: **`L = 23` is 190.1 KiB, which fits under a mistaken
192 KB budget and does not fit under the real 163 KiB one** (§9.5). Anyone who takes the 192 KB
figure at face value will design the wrong `L = 23` kernel.

### 10.5 §07's cuFFTDx limits table is cuFFTDx 1.2.1 and one row has changed

§07 §1.1's table gives "block, SM 70/72/86/89 → double `[2; 8192]`". The current requirements page
gives **SM 86/89/120/121 → double `[2; 12167]`** and adds a workspace ceiling of **8191** on
SM 80 `[F]` <https://docs.nvidia.com/cuda/cufftdx/requirements_func.html>. **The sm_80 rows §07
quotes (thread double `[2; 32]`, block double `[2; 16384]`) are unchanged and remain correct** —
which is what matters for us — but the table should be labelled with its version (§4.1).

Worth adding rather than correcting: §07 observes that "32 complex doubles is the largest transform
NVIDIA will put in a single thread's register file" without saying why. **It is the register
ceiling exactly**: 32 complex doubles = 128 of a thread's 255 registers, and 64 complex doubles
would be 256 — one past the limit (§6.5). NVIDIA's number is not a heuristic.

### 10.6 Two corrections to §08

* **§08 §5.7 quotes `fixMaxRaderPrimeMult` as "Default is vendor-specific (currently ~40)"** from
  the API guide `[F-agent]`. The shipped code sets **89** on both NVIDIA and AMD and 17 elsewhere
  (`vkFFT_InitializeApp.h:1262-1271`) `[F-src]`. The documentation is stale relative to the code,
  and the practical effect is that **both `L = 17` and `L = 23` are inside VkFFT's
  direct-multiplication Rader window on NVIDIA**, not just 17 (§3.6).
* **§08 §5.7 repeats the README radix list** — see §10.1.

Neither is a reasoning error; both are cases where the documentation and the source disagree and
the corpus quoted the documentation. **When VkFFT's docs and VkFFT's code disagree, the code is
in this repository — read it.**

---

## 11. Unsourced engineering notes (my own analysis, attributed to nobody)

Everything in this section is arithmetic I did from the published constants in §0.1. It is not
cited because it is not citable; it is checkable, and the inputs are all `[F]`.

1. **Machine balance.** 9.7 TFLOP/s ÷ 1555 GB/s = 6.24 flop/byte (vanilla FP64); 19.5 ÷ 1.555 =
   12.54 (DMMA). Used throughout §2 and §5.
2. **L2 read bandwidth = 7.2 TB/s** = the whitepaper's "5120 Bytes/clk" × the 1410 MHz boost
   clock, and **4.64× HBM**. The whitepaper gives the two factors and does not multiply them.
3. **Aggregate shared-memory bandwidth = 19.5 TB/s** = 32 banks × 4 B × 1.41 GHz × 108 SM, hence
   **15× the ~1.3 TB/s achievable HBM**. Used for §6.2's conflict-degree budget.
4. **The whole of §1.1's capacity table**, and the four-regime classification. The container
   capacities are documented; dividing them by 16-byte volumes is mine.
5. **§2.2's arithmetic-intensity table and time floors.** The `15·L³·log₂L` flop count follows
   from the benchFFT `5N log₂N` convention `[F-agent]` applied per axis; the 32-bytes-per-point
   traffic is the out-of-place minimum.
6. **§5.4's entire crossover analysis**: the `24L` flop/point dense count (8 real flops per
   complex MAC), the `L ≤ 8.32` and `L ≤ 16.72` crossovers, the `⌈L/8⌉·8/L × ⌈L/4⌉·4/L` padding
   waste factors, and the conjugate-symmetric `⌊L/2⌋+1` row count and its ≈1.9× saving. The
   inputs (the two peaks, the `m8n8k4` shape) are `[F]`/`[F-agent]`; the conclusions are mine.
   **Note where I differ from a subagent's reading:** if you also pad `N` to a multiple of 8 the
   `L = 17` waste becomes ≈2.3×, but `N` is the batch dimension (millions of columns) and pads for
   free, so 1.66× is the right figure.
7. **§6.2's bank-conflict rule for 16-byte elements**: that the conflict degree of a column access
   on a shared array of complex doubles with row stride `S` is `gcd(S, 8)`, hence "make the stride
   odd", hence the per-geometry padding table. Derived from the documented 32-bank/32-bit
   structure plus the fact that a 16-byte access spans 4 banks. **Not measured.**
8. **The inference that VkFFT's `17/16` padding rule is FP32-derived and can leave a 4-way
   conflict for FP64** (stride 68 at `fftDim = 64`; `gcd(68,8) = 4`). Read from the code, not
   profiled. §6.2.
9. **The inference that cuFFT's ≈630 and ≈420 GB/s bands in VkFFT's FP64 A100 plot are 1/2 and
   1/3 of the top band because cuFFT is doing 2 and 3 global passes.** Supported by the two
   benchmark drivers' differing bandwidth formulas (`4 × Σ numAxisUploads` for VkFFT, a fixed 4
   for cuFFT) `[F-src]` and by the numerical coincidence, but it is an inference. §2.3.
10. **The measurement of VkFFT's committed A100 FP64 plot** in §2.3. The plot is theirs `[F-src]`;
    the axis calibration and the marker-position statistics are mine, done programmatically. The
    numbers are good to a few percent, not better.
11. **§8.4's batch table** (`B_L2`, `B_HBM`) and the reasoning that `B = 1` is unmeasurable.
12. **The observation that the A100-PCIE-40GB's 250 W cap makes a sustained DMMA kernel a
    plausible power-throttle candidate.** The datasheet publishes the same 9.7 / 19.5 TFLOP/s for
    the 250 W PCIe SKU as for the 400 W SXM one, and the whitepaper says "Peak rates are based on
    GPU Boost Clock" — neither says anything about sustaining the boost clock in a 250 W envelope.
    I have not measured it; §7.2 and §8.8 say to read the counters, which is the honest response.
13. **§9's thread/block mappings** (64 threads/volume at `L = 8`, 289 at `L = 17`, 529 at `L = 23`,
    plane-per-block above) and every register count in them.

---

## 12. Citation ledger

### 12.1 Verified — fetched in this session (by me directly, or by a research subagent whose URL and quote I recorded)

**NVIDIA primary documentation (18)**

1. A100 Tensor Core GPU Architecture whitepaper — <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf> `[F]` (I extracted the text locally and read the DMMA, L2 and Table 4 passages myself)
2. NVIDIA Ampere GPU Architecture Tuning Guide — <https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html> `[F]`
3. CUDA C++ Programming Guide 12.6 (single page) — <https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html> `[F-agent]`
4. CUDA Programming Guide (current), Compute Capabilities appendix — <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html> `[F-agent]`
5. CUDA C++ Programming Guide 12.4 (for the `wmma` double-precision section) — <https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html> `[F-agent]`
6. CUDA C++ Best Practices Guide — <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html> `[F]` `[F-agent]`
7. PTX ISA (9.3) — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html> `[F-agent]`
8. PTX ISA 8.2 (CUDA 12.2 archive) — <https://docs.nvidia.com/cuda/archive/12.2.0/parallel-thread-execution/index.html> `[F]`
9. cuFFT documentation (13.3) — <https://docs.nvidia.com/cuda/cufft/index.html> `[F-agent]`
10. cuFFT documentation (CUDA 11.8 archive) — <https://docs.nvidia.com/cuda/archive/11.8.0/cufft/index.html> `[F-agent]`
11. cuFFTDx requirements — <https://docs.nvidia.com/cuda/cufftdx/requirements_func.html> `[F]`
12. cuFFTDx examples — <https://docs.nvidia.com/cuda/cufftdx/examples.html> `[F]`
13. cuFFTDx "Achieving High Performance" — <https://docs.nvidia.com/cuda/cufftdx/performance.html> `[F-agent]`
14. Nsight Compute Profiling Guide — <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html> `[F-agent]`
15. `nvidia-smi` documentation — <https://docs.nvidia.com/deploy/nvidia-smi/index.html> `[F-agent]`
16. A100 datasheet (JUN21, 4-SKU table) — <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf> `[F-agent]`
17. CUDA Graph docs (launch-overhead breakdown) — <https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/quantitative-benefits.html> and `.../cuda-graph.html` `[F-agent]`
18. NVIDIA developer blog, *Getting Started with CUDA Graphs* — <https://developer.nvidia.com/blog/cuda-graphs/> `[F-agent]` (V100, flagged as a blog)

**NVIDIA secondary (4)**

19. *Double-Precision Tensor Cores Speed High-Performance Computing* — <https://blogs.nvidia.com/blog/double-precision-tensor-cores/> `[F-agent]`
20. *NVIDIA Ampere Architecture In-Depth* — <https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/> `[F-agent]`
21. *Multinode Multi-GPU: Using NVIDIA cuFFTMp FFTs at Scale* — <https://developer.nvidia.com/blog/multinode-multi-gpu-using-nvidia-cufftmp-ffts-at-scale/> `[F-agent]`
22. Mark Harris, *An Efficient Matrix Transpose in CUDA C/C++* — <https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/> `[F-agent]` (Fermi/Kepler, 2013)

**VkFFT (5, plus the in-tree source)**

23. VkFFT README — <https://raw.githubusercontent.com/DTolm/VkFFT/master/README.md> `[F-agent]`
24. VkFFT API guide PDF — <https://raw.githubusercontent.com/DTolm/VkFFT/master/documentation/VkFFT_API_guide.pdf> `[F-agent]`
25. Tolmachev, SC22 technical poster — <https://sc22.supercomputing.org/proceedings/tech_poster/poster_files/rpost143s3-file2.pdf> `[F-agent]`
26. Bibliographic record for Tolmachev, *VkFFT — A Performant, Cross-Platform and Open-Source GPU FFT Library*, **IEEE Access 11, 12039–12058 (2023), DOI 10.1109/ACCESS.2023.3242240** — <https://api.crossref.org/works/10.1109/ACCESS.2023.3242240> `[F-agent]`. **Full text not fetched** (§12.2).
27. Bibliographic record for Tolmachev, *VkFFT and beyond — a platform for runtime GPU code generation*, IWOCL 2023, DOI 10.1145/3585341.3585357 — <https://api.openalex.org/works?filter=title.search:VkFFT> `[F-agent]`
28. `ext/src/VkFFT` at commit `066a17c` (v1.3.4) — `[F-src]`, files and lines cited inline
29. `ext/src/VkFFT/benchmark_plot/fp64_cuda_a100.png` — `[F-src]`, measured programmatically (§2.3)

**Papers (16)**

30. Wu, Zhai, Liu, Huang, Jian, Dai, Di, Chen & Cappello, *TurboFFT*, arXiv:2405.02520 — <https://arxiv.org/html/2405.02520v1> `[F-agent]`
31. Wu et al., *TurboFFT: Co-Designed High-Performance and Fault-Tolerant FFT on GPUs*, PPoPP '25, DOI 10.1145/3710848.3710853, arXiv:2412.05824 — <https://arxiv.org/html/2412.05824v1> `[F-agent]`
32. Li, Cheng & Lin, *tcFFT: Accelerating Half-Precision FFT through Tensor Cores*, arXiv:2104.11471 — <https://arxiv.org/pdf/2104.11471> `[F-agent]`
33. Sorna, Cheng, D'Azevedo, Wong & Tomov, *Optimizing the FFT using Mixed Precision on Tensor Core Hardware*, HiPCW 2018 — <https://www.osti.gov/servlets/purl/1559731> `[F-agent]`
34. Tu, Karlin, Camier, Dobrev, Kolev, Henneking & Ghattas, *Accelerating High-Order Finite Element Simulations at Extreme Scale with FP64 Tensor Cores*, arXiv:2603.09038v2 — <https://arxiv.org/html/2603.09038v2> `[F-agent]`
35. Abdelkhalik, Arafa, Santhi & Badawy, *Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis*, arXiv:2208.11174 — <https://arxiv.org/pdf/2208.11174> `[F-agent]`
36. Sun, Li, Geng, Stuijk & Corporaal, *Dissecting Tensor Cores via Microbenchmarks*, IEEE TPDS, arXiv:2206.02874 — <https://arxiv.org/pdf/2206.02874> `[F-agent]`
37. Luo, Fan, Li, Du, Wang & Chu, *Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*, IPDPS 2024, arXiv:2402.13499 — <https://arxiv.org/pdf/2402.13499> `[F-agent]`
38. Matsuoka, *FP8 is All You Need (Part 2): … Ozaki–Bailey Style FFT …*, arXiv:2606.23698v2 — <https://arxiv.org/pdf/2606.23698v2> `[F-agent]`
39. Ootomo & Yokota, *Recovering single precision accuracy from Tensor Cores…*, arXiv:2203.03341 — <https://arxiv.org/pdf/2203.03341> `[F-agent]`
40. Huang, Zhang, Yang & Xiao, *Benchmarking GPU Tensor Cores on General Matrix Multiplication Kernels through CUTLASS*, Appl. Sci. **13**, 13022 (2023) — <https://xianweiz.github.io/doc/papers/cutlass_applsci23.pdf> `[F-agent]`
41. Merry, *Efficient channelization on a Graphics Processing Unit*, arXiv:2303.09886 — <https://arxiv.org/pdf/2303.09886> `[F-agent]`
42. Tsai, Cojean & Anzt, *Evaluating the Performance of NVIDIA's A100 Ampere GPU for Sparse Linear Algebra Computations*, arXiv:2008.08478 — <https://arxiv.org/pdf/2008.08478> `[F-agent]` (BabelStream A100 numbers)
43. Ayala, Tomov, Haidar & Dongarra, *heFFTe: Highly Efficient FFT for Exascale*, ICCS 2020, DOI 10.1007/978-3-030-50371-0_19 — <https://www.iccs-meeting.org/archive/iccs2020/papers/121370260.pdf> `[F-agent]`
44. Ayala, Tomov, Stoyanov, Haidar & Dongarra, *Performance Analysis of Parallel FFT on Large Multi-GPU Systems*, IPDPS 2022 — <https://www.netlib.org/utk/people/JackDongarra/PAPERS/Performance_Analysis-fft-ipdps22.pdf> `[F-agent]`
45. Verma, Chatterjee, Garg, Sharma, Arya, Kumar, Saxena & Verma, *Scalable Multi-node Fast Fourier Transform on GPUs*, arXiv:2202.12756 — <https://arxiv.org/pdf/2202.12756> `[F-agent]`

**Benchmarking methodology (5)**

46. benchFFT / FFTW, *FFT Benchmark Methodology* — <https://www.fftw.org/speed/method.html> `[F-agent]` (the `5N log₂N` convention and the min-of-8 rule)
47. Hoefler & Belli, *Scientific Benchmarking of Parallel Computing Systems*, SC '15 — <https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf> `[F-agent]`
48. Zhang, Wahib, Zhang & Matsuoka, *[persistent-kernel / launch-overhead microbenchmarks]*, IPDPS 2020, DOI 10.1109/IPDPS47924.2020.00057, arXiv:2004.05371 — <https://arxiv.org/pdf/2004.05371> `[F-agent]`
49. Zhang, Wahib, Chen, Meng, Wang, Endo & Matsuoka, *PERKS: PERsistent KernelS*, ICS '23, DOI 10.1145/3577193.3593705, arXiv:2204.02064 — <https://arxiv.org/pdf/2204.02064> `[F-agent]`
50. Wei, Pradeep & Bhatele, *Unmasking Performance Variability in GPU Codes on Production Supercomputers*, SC25 poster — <https://sc25.supercomputing.org/proceedings/posters/poster_files/post256s2-file3.pdf> `[F-agent]`
51. Volkov, *Better Performance at Lower Occupancy*, GTC 2010 (presentation, not peer-reviewed) — <https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf> `[F-agent]`

**CUTLASS source and docs (5)**

52. `media/docs/cpp/functionality.md` — <https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/functionality.md> `[F-agent]`
53. `include/cutlass/arch/mma_sm80.h` — <https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/arch/mma_sm80.h> `[F-agent]`
54. `include/cutlass/arch/mma.h` — <https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/arch/mma.h> `[F-agent]`
55. `include/cutlass/gemm/warp/mma_gaussian_complex_tensor_op.h` — <https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/warp/mma_gaussian_complex_tensor_op.h> `[F-agent]`
56. CUTLASS README (current, and v2.11.0) — <https://raw.githubusercontent.com/NVIDIA/cutlass/main/README.md> `[F-agent]`

**Measured on this machine `[M]`** — `nvidia-smi -L`, `nvidia-smi -q`, `nvidia-smi -q -d MEMORY,CLOCK`, `nvidia-smi -q -d PERFORMANCE`, `nvidia-smi --help`, `/usr/local/cuda-11.7/extras/demo_suite/deviceQuery`, `sinfo`. All read-only; no benchmark was run.

**In-tree, not citations** — `bench/geom/PANEL_BRIEF.md`, `bench/geom/results/panel_r4/leaderboard.txt`, `bench/geom/results/panel_r5/leaderboard.txt`.

**Count: 56 external sources fetched and quoted, plus in-tree VkFFT source and on-machine device queries.**

### 12.2 Could NOT fetch — and the claims each leaves unverified

1. **Tolmachev, *VkFFT…*, IEEE Access 11, 12039–12058 (2023), DOI 10.1109/ACCESS.2023.3242240** — every IEEE Xplore route timed out; scispace returned 403; the Semantic Scholar API was unreachable. *The bibliographic record is verified via Crossref; nothing in this section quotes the article's body.* **Unverified:** whatever quantitative comparisons the peer-reviewed paper makes that the SC22 poster and README do not. This is the single most valuable missing source for §3.
2. **Durrani, Chughtai, Dakkak, Hwu & Rauchwerger, "FFT blitz: the tensor cores strike back", PPoPP 2021, pp. 488–489, DOI 10.1145/3437801.3441623** — ACM returned 403, no preprint reachable. **Unverified:** the GPU, the precision, the accuracy, and a search-snippet claim of "up to 1.5× for FFT" which §5.7 explicitly declines to use.
3. **Durrani, Chughtai, Hidayetoglu, Tahir, Dakkak, Rauchwerger, Zaffar & Hwu, "Accelerating Fourier and Number Theoretic Transforms using Tensor Cores and Warp Shuffles", PACT 2021, pp. 345–355, DOI 10.1109/PACT52795.2021.00032** — same. **Unverified:** everything except the bibliographic record (from dblp). Note the NTT half of this work is integer and therefore exact, which makes it the one piece of tensor-core transform literature that might survive an accuracy gate; **someone with library access should read it.**
4. **Ayala et al., "FFT Benchmark Performance Experiments on Systems Targeting Exascale", ICL-UT-22-02** (<https://icl.utk.edu/files/publications/2022/icl-utk-1548-2022.pdf>) — downloaded, but every digit renders as U+FFFD in two independent PDF text extractors. It contains §3.1 "Bandwidth Analysis", §3.2.2 "FFT Performance Analysis for a single GPU" and Figure 3.4 "Percentage of theoretical peak achieved on a single GPU of Summit and Spock". **This is probably the single most on-target missing source for §2.3** (single-GPU %-of-peak for cuFFT and rocFFT) and it should be retrieved by another route.
5. **An A100 CUTLASS/cuBLAS DGEMM efficiency figure in text form** — does not appear to exist. The CUTLASS 2.11 README's A100 chart is an image; the current README has no A100 chart; Huang et al.'s FP64 result is figure-only. **Unverified:** what fraction of 19.5 TFLOP/s a production DGEMM actually reaches. §5.3's ≈97 % is a *microbenchmark* of the instruction, not of a GEMM; treat the gap between them as unknown.
6. **NVIDIA's cuFFT product-page performance charts** (`1-d-single-precision-fft-850x480.svg`, `multi-node-support-850x480.svg`) — fetch succeeded, but the SVGs contain glyph outlines with no text elements. **Unverified:** any NVIDIA-published cuFFT number. As §3.2 says, NVIDIA appears to publish no single-GPU A100 FP64 batched-FFT figure at all.
7. **Kawakami & Takahashi, "Computing FFTs at target precision using lower-precision FFTs", arXiv:2603.29129 (March 2026)** — known only from Matsuoka's citation of it; not fetched. **Unverified:** the "107–1315×" figure Matsuoka quotes, whose direction is ambiguous in his phrasing. A CPU result, so low priority for us, but it is the nearest thing to a *high-accuracy* low-precision FFT and worth reading if the DMMA route disappoints.
8. **Cheng, Sorna, D'Azevedo, Wong & Tomov, "Accelerating 2D FFT: Exploit GPU Tensor Cores through Mixed-Precision", SC'18 poster** — referenced by tcFFT; not fetched. **Unverified:** everything. Superseded for our purposes by tcFFT and by Sorna et al., both of which are fetched.
9. **A "2260.5 ns nullKernel launch overhead on A100" figure** widely attributed to arXiv:2504.11750 — the paper was fetched and the string does not appear in it. **Do not use that number.** §6.6 uses the IPDPS 2020 and NVIDIA-blog figures instead.
10. `https://developer.nvidia.com/blog/double-precision-tensor-cores/` → **404**; the article is at `blogs.nvidia.com` (source 19). Anyone citing the `developer.nvidia.com` path is citing a dead URL.
11. `https://docs.nvidia.com/cuda/cuda-c-programming-guide/05-appendices/compute-capabilities.html` → **404**. The working path drops the `-c-`: `cuda-programming-guide`. Relevant because §07-era citations to `cuda-c-programming-guide` anchors may now be dead.

**Also worth recording as a tooling limit:** this session exhausted its WebSearch budget (200/200
calls) partway through, so items 2, 3 and 5 above could not be pursued by further searching. A
future round should start with those three.

### 12.3 The three biggest gaps this section could not close

1. **There is no published batched-small-3D FP64 FFT measurement on an A100, from anyone.**
   VkFFT's 3D benchmark uses a single volume `[F-src]`; its batched benchmark is 1D `[F-agent]`;
   TurboFFT is 1D `[F-agent]`; cuFFT publishes nothing `[F-agent]`. **Our leaderboard will be the
   first data of its kind**, which is worth knowing both for confidence and for humility.
2. **There is no prior work on FP64-tensor-core FFT.** §5.7 establishes this by exhaustion: an
   arXiv full-text search for `"tensor cores" AND FFT` returns four papers, all FP16/FP8, and the
   one paper that programs DMMA directly (§5.5) says it is the first to do so for any HPC
   application. So §5.4's crossover analysis has no empirical check anywhere in the literature.
   **Treat it as a hypothesis with good arithmetic behind it, not as a known result.**
3. **Which of cuFFT and VkFFT is faster in FP64 on an A100 is genuinely disputed** (§3.3): VkFFT's
   own plot says VkFFT by ~2× on the radix path, TurboFFT's peer-reviewed measurement says cuFFT
   by 11 % on average. Both were fetched; they cannot both be right for the same sizes. **Measure
   both before designing against either.**
