# 08 — Recent work (2015–2026): the bandwidth wall, AVX-512 as it actually behaves on Cascade Lake, and what is left after the plateau

**Audience and purpose.** You are about to edit a C kernel for one of L = 6, 8, 17, 36 under the
contract in `bench/geom/PANEL_BRIEF.md`: forward complex-double 3D DFT, out-of-place, both
buffers 64-byte aligned, contiguous C order with the batch index slowest-varying, single
thread, no library calls, scored on an isolated **Intel Xeon Gold 5218** (Cascade Lake-SP,
2.30 GHz base, 3.9 GHz max turbo, 16 cores/socket, 32 KiB L1d, **1 MiB L2 per core**,
22 MiB L3, 6 channels DDR4-2666), developed on a **Xeon Gold 6448Y** (Sapphire Rapids).

Sections 01–07 of this corpus are strong on the classical algorithm literature and weak on two
things that now dominate the panel's remaining headroom: **what one core of this exact
microarchitecture can actually move**, and **what 512-bit code actually costs on this exact
part**. §04 §8.2 and `LITERATURE.md` §4.8 gap 6 both say in writing that there is no AVX-512
measurement anywhere in the corpus and no primary source for the memory system of a
Skylake-SP-class server part. This section closes both gaps with published measurements on
Cascade Lake-SP silicon, and then works through the four remaining targets.

**The one-paragraph version.** Three rounds of implementers converged on row–column with
unrolled batch-vectorised codelets and stopped, and the measurements say they stopped for a
good reason at large batch and a bad one at B = 1.

At large batch every geometry is within 1.16–1.83× of the single-core streaming bandwidth this
microarchitecture is *published* as delivering (§1.1–§1.2), so the levers there are traffic and
concurrency, not butterflies. **At B = 1 the picture is different from what the panel believes.**
The r3 verdict computes each leader as sitting 4–13 % above its FP-port floor and concludes the
arithmetic is finished — but that floor was computed at the 2.30 GHz *non-AVX base clock*, and
Intel's own turbo table for the Xeon Gold 5218 says a single active core runs 512-bit code at
**2.9 GHz**. At the right clock every geometry is **~30–43 % above its floor** (§4.1). There is
more headroom at B = 1 than three rounds of records assume.

The four things actually left: (i) the **1/3 of DRAM traffic** that write-allocate is spending and
that non-temporal stores would delete — declined by the panel's tuners for three rounds, for
reasons §1.5 can now name from Intel's own manual; (ii) **cache-blocking across the batch into the
1 MiB L2**, because a single core reads L2 at 87.3 GB/s, L3 at only 18.2 GB/s and DRAM at 11.5 —
so L3 residency is worth 1.6× and L2 residency 7.6× (§1.2, §1.4, §1.9); (iii) **4 KiB store-to-load
aliasing**, which an out-of-place transform between two page-aligned buffers hits *by construction*,
which every geometry hits at every element and which L=8 and L=36 hit most degenerately, and
which nobody has looked for (§1.8); and
(iv) **the schedule rather than the kernel** — the one published measurement of our exact regime
had 1D kernels *slower* than MKL's and a 3D kernel 2–3× faster, and attributed the win entirely to
layout and loop merging (§2.2).

And one correction that removes a self-imposed constraint: **on this SKU at one active core,
AVX-512 costs no frequency at all relative to AVX2** (§4.1). The downclocking figure the brief and
the corpus quote is a 9+-core number. Entries that declined to write a 512-bit path on that basis
declined for a reason that does not apply.

---

## 0. Ten things to do differently, in priority order

Each links to the subsection that justifies it with a source and a number.

1. **Stop treating L3 residency as a goal; block for the 1 MiB L2.** Single-core read
   bandwidth on Cascade Lake-SP is 18.2 GB/s from L3 against 11.5 GB/s from DRAM — a 1.6×
   prize — while L2 delivers 87.3 GB/s, a 7.6× prize. Alappat et al. recommend exactly this
   switch in print. §1.2, §1.4.
2. **Measure the clock, then re-open every B=1 conclusion.** Intel's turbo table for the Gold
   5218 gives 3.9 GHz non-AVX / 2.9 GHz AVX2 / **2.9 GHz AVX-512 at one active core** — AVX2 and
   AVX-512 identical up to 8 cores. So (a) there is **no AVX-512 frequency penalty** for a
   single-threaded kernel on this part, and (b) the FP-port floors in the r3 verdict are 26 % too
   high, which turns "4–13 % of headroom at B=1" into **30–43 %**. One `perf stat -e cycles,ref-cycles`
   settles it. §4.1.
3. **Block across the batch into L2 and run all three axes inside the tile.** Intel recommends
   exactly this for Skylake Server in print ("Consider blocking to L2… if L2 can sustain the
   application's bandwidth requirements"); sustained L2 is 52 B/cy against L3's 15 and DRAM's ~7.
   A tile of ~24 volumes at L=8 is 384 KiB, ~37 % of L2. This one change caps DRAM traffic at the
   compulsory minimum regardless of pass count, makes the output stream long enough for NT stores
   and for the hardware streamer, and is the *only* regime in which the corpus's pass-fusion
   argument has not already been tested and found wanting. §1.9.
4. **Re-test non-temporal stores with the harness's buffer reuse in mind, and instrument them.**
   NT stores delete a full **1/3** of DRAM traffic on a copy-shaped kernel (3:2, not 4:3). The
   panel's tuners reject them anyway — and Intel's own manual explains why on *this*
   microarchitecture: on Skylake Server "the resources within each core remain busy for a longer
   duration… the processor may run out of resources and stall, thus limiting the memory write
   bandwidth from each core." §1.5.
5. **Check for 4 KiB aliasing.** `in` and `out` are separate large page-aligned allocations, so
   their relative offset is almost certainly a multiple of 4096, which makes every load from
   `in` falsely alias a recent store to `out` in the memory order buffer — **at every element, in
   all four geometries.** The volume stride then decides how degenerate the pattern is, and **L=8
   (period 1 — every volume starts at the same page offset) and L=36 (period 4) are the two worst,
   which are also the two with the tightest margins.** One perf counter settles it. §1.8.
6. **Check the huge-page and NUMA-balancing state of the node.** On Cascade Lake a serial
   load-only stream measures a **2× difference** between `THP=always, NUMA balancing off` and
   the common default `THP=madvise, NUMA balancing on`. §1.7.
7. **Prefetch with `prefetcht1`, at a computed distance, along the batch axis — and stop
   prefetching whole volumes ahead.** Cascade Lake has **exactly 10 line fill buffers** (measured
   on a Gold 6226); `prefetcht0`/`prefetchnta` consume them and `prefetcht1` does not; software
   prefetching is worth **up to 1.29× single-core on Skylake-SP** and the win comes from
   prefetching the *regular* streams; and `D ≥ ⌈l/s⌉` (Mowry) puts our distance at ≈**16 cache
   lines ≈ 1 KiB**, not one volume. Our axis passes are 2–5-cache-line streams that the hardware
   streamer can never train on — the long stream is the batch. §1.6, §1.6b.
8. **Reorder the writes, and check your strides are compile-time constants.** Store *order*, not
   traffic volume, was worth 10.8 % in round 3; three independent sources (Popovici et al., ducc0,
   the panel itself) arrive at "choose the axis order for the layout"; and a codelet with a
   fixed stride measured **1.9× faster than the same codelet with a runtime stride** at N=60.
   §2.2, §2.4, §5.5.
9. **At L=17, scan B = 1, 2, 4, 8, 16, 32 to find where the ranking actually inverts.** If the
   crossover is at B ≈ 8 the cause is vectorisability and the fix is a batch-blocked schedule for
   the dense kernel, not a choice of kernel. The panel's batch grid cannot see the difference. §3.1.
10. **Do not spend another round on op counts, vector-radix, or L1↔L2 pass fusion.** §2 confirms
    the corpus verdict with newer evidence, and the panel's own r3 experiment settled fusion at
    "single-digit percent, sometimes negative" — *in the L1/L2 regime*. The untested regime is
    L2↔DRAM (item 3). For the arithmetic itself, §6: keep the 2-FMA complex multiply (provably
    optimal at 2u), try the 3-FMA lifting rotation at L=17, and leave everything else alone.

---

## 1. Target 1 — batched small transforms that are memory-bandwidth-bound

This is the L=8 B=2048 problem (32 MiB working set, the panel's tightest cell), the L=8
B=16384 problem, the L=36 B≥32 problem (the largest unclaimed prize on the board at 1.83×),
and the L=6 B=32768 problem (which is finished, and §1.1 explains why).

### 1.1 The single-core bandwidth ceiling is a *concurrency* limit, not a DRAM limit

**Sources.**

* Daniel Lemire, *Memory-level parallelism: Intel Skylake versus Intel Cannonlake*, 13 Jan
  2019 — https://lemire.me/blog/2019/01/01/memory-level-parallelism-intel-skylake-versus-intel-cannonlake/
  [VERIFIED — fetched]. Measured: a Skylake core is "limited to about ten concurrent memory
  requests"; at 70 ns per query that benchmark reaches **9 GB/s**. The Cannonlake part
  sustains roughly twice as many concurrent requests and reaches **12 GB/s at 110 ns**, i.e.
  *more* bandwidth at *worse* latency — the definition of a concurrency-limited regime. In the
  comments Travis Downs identifies the core-private structures involved as the miss-handling
  registers and a superqueue of "approximately 16 entries".
* Daniel Lemire, *Estimating your memory bandwidth*, 13 Jan 2024 —
  https://lemire.me/blog/2024/01/13/estimating-your-memory-bandwidth/ [VERIFIED — fetched].
  On a dual-socket 64-core Ice Lake system: "I start out at 15 GB/s" with one thread and "I go
  up to over 130 GB/s"; "once I reach about 20 threads, it is no longer possible to get more
  bandwidth out of the system". A comment citing McCalpin gives the Sandy Bridge arithmetic:
  "a single core only supports 10 concurrent L1 Data Cache misses", limiting one thread to
  about **8.1 GB/s** against a **51.2 GB/s** socket peak.
* Markus Velten, Robert Schöne, Thomas Ilsche, Daniel Hackenberg, *Memory Performance of AMD
  EPYC Rome and Intel Cascade Lake SP Server Processors*, ICPE '22 (Beijing, 9–13 April 2022),
  arXiv:2204.03290 — https://arxiv.org/pdf/2204.03290 [VERIFIED — fetched, full text].
  Their CLX part is a **Xeon Gold 6248** (same microarchitecture and same 1 MiB L2/core as our
  Gold 5218). Measured local access latencies, Fig. 7: **L1d 4 cycles (1.6 ns), L2 14 cycles
  (5.6 ns), L3 54 cycles (21.6 ns), RAM 200 cycles (80.0 ns)**.

**The arithmetic, and why it matters here.** Little's Law in the form
`bandwidth = concurrency ÷ latency` with the numbers above:

```
10 lines × 64 B ÷ 80 ns  =  8.0 GB/s
12 lines × 64 B ÷ 80 ns  =  9.6 GB/s
16 lines × 64 B ÷ 80 ns  = 12.8 GB/s      <- superqueue-sized concurrency
to reach 20 GB/s you need 20e9 × 80e-9 / 64 = 25 lines in flight
```

The monitor's round-3 verdict reports the panel's best demonstrated single-core stream as
**12.3 GB/s** of compulsory read+write (L=6 at B=32768), and computes ceilings for every other
cell from the same 12 GB/s figure. That number is not a property of the panel's code. It is
the concurrency ceiling of the core, and independent published microbenchmarks on Cascade
Lake-SP put the single-core figure in the same place (§1.2). **A single core on this part
cannot be made to stream much faster by scheduling; it can only be given less to move, or
more requests in flight.** That reframes the whole large-batch problem:

* **Cutting traffic** is the big lever, and there is exactly one traffic cut available in an
  out-of-place transform: delete the write-allocate read of the output. That is 1/3 of the
  compulsory DRAM traffic. §1.5.
* **Raising concurrency** is the second lever, and it is *not* the same thing as prefetching
  deeper into one stream. Ten to sixteen in-flight lines is what one sequential stream plus
  the L2 streamer already generates. Multiple *independent* streams — several volumes, or
  several pencils on different 4 KiB pages, advancing together — is what adds concurrency.
  The panel's "cross-volume prefetch" experiments in r3 attacked this with droppable
  `prefetch` hints rather than with genuinely interleaved demand streams, and the node's
  tuners rejected them in three of three entries. §1.6.

### 1.2 What the memory hierarchy actually delivers to ONE core on this microarchitecture

Velten et al. (above) Fig. 10, single core of a Xeon Gold 6248, read-only streaming kernel,
core clock **pinned at 1.6 GHz** (the paper states 1.6 GHz is "the nominal core frequency for
AVX-512 for the Intel Xeon Gold 6248 processor"), hardware prefetchers enabled, THP set to
`always`:

| instruction | L1 | L2 | L3 | RAM |
|---|---|---|---|---|
| `add_pd` (SSE, 128-bit) | 46.5 GB/s | 35.3 GB/s | 17.5 GB/s | 10.7 GB/s |
| `avx_add_pd` (256-bit) | 93.0 GB/s | 50.6 GB/s | 18.2 GB/s | 11.4 GB/s |
| `avx512_add_pd` (512-bit) | **186 GB/s** | **87.3 GB/s** | 18.2 GB/s | 11.5 GB/s |

In bytes per cycle at 1.6 GHz: L1 116.25 B/cy measured for AVX-512 (against a theoretical
128 B/cy — the paper says this is "lower than the sustained bandwidths listed in [the Intel
optimization manual] Table 2-6" but in line with other published measurements); L2 ≈ 54.6 B/cy
for 512-bit and ≈ 31.6 B/cy for 256-bit; L3 **11.3 B/cy** for every SIMD width; RAM ≈ 7.2 B/cy.
The paper's own summary sentence: **"The RAM bandwidth is barely affected by SIMD width."**

Four consequences, all directly actionable:

1. **L3 is worth almost nothing to a single core on this part.** 18.2 GB/s from L3 against
   11.5 GB/s from DRAM is a **1.6×** ratio. Every cache-blocking scheme in the corpus that
   aims at L3 residency is chasing 1.6×. L2 residency is worth **7.6×** (87.3 / 11.5) with
   512-bit loads. **Blocking decisions should be made against 1 MiB, not 22 MiB.**
2. **AVX-512's bandwidth advantage is entirely inside L1 and L2** — 2.0× in L1 and 1.7× in L2
   over 256-bit — and is **exactly zero** from L3 and DRAM. So the correct SIMD width is
   size-and-regime dependent, not global: wide where the data is L1/L2-resident, and
   arbitrary (hence: whichever clocks higher) where you are streaming. §4.4.
3. **L=17 is an L2-resident problem and should be optimised as one.** One 17³ volume is
   78.6 KiB; input+output is 157 KiB, 15 % of the 1 MiB L2. The relevant bandwidth for the
   interior of the transform is 87.3 GB/s (512-bit) or 50.6 GB/s (256-bit), not 11.5.
4. **L=36 is a DRAM problem at every batch size ≥ 2** (1.49 MB of compulsory traffic per
   volume against a 1 MiB L2), and no SIMD width will change that. Its optimisation is
   traffic and concurrency, full stop.

Cross-check from a second group on the same microarchitecture: Christie L. Alappat, Johannes
Hofmann, Georg Hager, Holger Fehske, Alan R. Bishop, Gerhard Wellein, *Understanding HPC
Benchmark Performance on Intel Broadwell and Cascade Lake Processors*, ISC High Performance
2020, arXiv:2002.03344 — https://arxiv.org/abs/2002.03344 and
https://arxiv.org/pdf/2002.03344v2 [VERIFIED — fetched, full text]. Their CLX is also a **Xeon
Gold 6248**. Fig. 3(d) annotates single-core main-memory bandwidths of **11.47, 11.7, 12.16 and
9.32 GB/s** across the load-only and copy kernels for CLX (at 1.6 GHz) and BDW (at 2 GHz) — the
same ~11–12 GB/s neighbourhood, from a different tool (`likwid-bench`) and a different group.
Their Table 1 also gives the on-chip datapath widths: **L1↔L2 64 B/cy** and **L2↔L3
16 B/cy + 16 B/cy** on CLX, against a unified 32 B/cy on Broadwell — i.e. Cascade Lake
*halved* the single-direction L2↔L3 load bandwidth relative to the previous generation. That
is the mechanism behind the 11.3 B/cy L3 figure above.

They also warn, usefully, that popular tools get this wrong: with `lmbench` at `-O2` they
measure 25.5 GB/s where the theoretical L1 figure is 204.8 GB/s — "a huge factor of eight" —
and note that raising the optimisation level from `-O2` to `-O3` *decreased* the measured
bandwidth. If you write your own bandwidth probe, check it against a known limit first.

### 1.3 Cascade Lake's L3 is a non-inclusive *victim* cache with a streaming-aware replacement policy

Alappat et al. §4.1, on CLX (Gold 6248, 27.5 MiB L3) [VERIFIED — fetched]:

* "Empirical analysis suggests that SKX's L3 cache uses an adaptive replacement policy… the
  replacement policy selected by the processor for streaming-access patterns involves placing
  new cache lines **only in one of the eleven ways of each cache set** — the same strategy that
  is used when prefetching data using the `prefetchnta` instruction… Consequently, data in the
  remaining ten ways of the sets will not be preempted and can later be reused."
* Measured (Fig. 4a, hardware prefetchers *disabled* for the hit-rate measurement): older
  generations "offer no data reuse for data sets two times the cache capacity, whereas CLX's
  L3 delivers hit rates of **20 % even for data sets almost four times its capacity**. Even
  for data more than **ten times** the L3 cache's size can reuse be detected on CLX."
* Fig. 4b, prefetchers enabled: "the L3-cache hit-rate improvements directly translate into
  higher-than-memory bandwidths for data sets well exceeding the L3 cache's capacity."
* Footnote 9: because the L3 is an exclusive victim cache, "the applicable cache size for
  applications using all cores is thus 47.5 MB, the aggregate L2/L3 cache size."

**Why this matters to us, concretely.** The panel's mental model — "working set > L3, therefore
it streams from DRAM" — is wrong on this part. At L=8 B=2048 the working set is 32 MiB
(1.45× the 22 MiB L3); at L=6 B=4096 it is 27 MiB (1.2×); at L=36 B=32 it is 46 MiB (2.1×).
Every one of those is inside the range where Alappat et al. measure *substantial* residual L3
hit rate on Cascade Lake. Two implications:

* A **batch tile** whose footprint is a modest multiple of L3 is not automatically wasted; the
  hardware is deliberately preserving ten of eleven ways against your stream. This is the
  strongest available argument for cache-blocking *across the batch dimension* on this
  specific part — you are not fighting an LRU that will throw your tile away.
* It is also a strong argument against blanket non-temporal stores, which explicitly place
  into that one streaming way and forfeit the reuse. §1.5.

### 1.4 Block for L2, not L3 — the authors say so in print

Alappat et al. §4.2 measure L3 *scalability* on both parts with a 2 MiB-per-core working set:
Broadwell "scales almost linearly and attains an efficiency within 90 %" (labelled ε = 0.94–0.98
in Fig. 5), while on CLX "this behavior has changed drastically and the L3 cache saturates at
higher core counts both with and without SNC enabled, yielding an efficiency of about **70 %**"
(ε = 0.67–0.73). Their recommendation, verbatim:

> "Due to this for the applications that employ L3 cache blocking it might be worthwhile to
> investigate the impact of switching to pure L2 blocking on SKX and CLX architectures."

That is a multi-core statement, but §1.2's single-core numbers point the same way for us: the
single-core L3 read path on CLX is half of Broadwell's, and only 1.6× DRAM. Combined with
`PANEL_BRIEF.md`'s warning that wallaby has **2 MiB** of L2 per core against the node's
**1 MiB**, the rule for this panel is:

> **Every tile parameter should be chosen so the tile's live footprint is comfortably under
> 1 MiB, and should be a compile-time constant that the node tuner can halve.** A tile tuned to
> wallaby's 2 MiB L2 is exactly twice too big, and the r3 verdict already documents four
> memory-system changes that inverted across that boundary.

Note the one place the corpus's advice needs correcting outright: §05 §10.6 says "do not tune
to this node's 256 KiB L2… block to L1 by planes, which is safe everywhere." L1 blocking is
indeed safe, but it now leaves 87.3 GB/s of L2 bandwidth unexploited on a part whose L2 is
1 MiB — 4× the Haswell L2 that advice was written for. At L=17 in particular, the whole
input+output pair (157 KiB) fits L2 with 6× room to spare, and L2 is 7.6× DRAM.

### 1.5 Non-temporal stores: worth 1.5×, and why this node keeps refusing them

**The arithmetic, sourced.** Alappat et al. §5 [VERIFIED — fetched]: on a write-allocate
machine the STREAM triad moves 32 B per iteration rather than the 24 B the benchmark assumes;
non-temporal stores "bypass the normal cache hierarchy and store into separate write-combine
buffers. If a full cache line is to be written, the write-allocate transfer can thus be
avoided", and measured on CLX with `-qopt-streaming-stores always` versus `never`:

> "The reported saturated bandwidth of the 'NT' variant is a factor of **4/3** higher because
> the memory interface delivers the same bandwidth but the code balance is only 24 byte/it."

Their footnote 8 is the historical caveat and it cuts the other way: "earlier Intel processors
like Ivy Bridge and Sandy Bridge could not attain the same memory bandwidth with NT stores as
without. The difference was small enough, however, to still warrant the use of NT stores in
performance optimization."

**Applied to our contract.** Out-of-place, the compulsory DRAM traffic per volume with normal
stores is `read(in) + RFO(out) + writeback(out)` = 3 × volume bytes; with full-line NT stores
it is 2 × volume bytes. So NT stores are worth up to **33 %** in any cell that is genuinely
DRAM-bound — which is the largest single number available anywhere on the L=36 and L=8
large-batch board. The monitor's r3 verdict records that L=6's tuners have now rejected NT
stores three rounds running, and that at L=36 the tuners *did* pick NT (`mode=scratch+nt`,
`v1-nt-pf1`). So the picture is not "NT stores don't work here", it is "NT stores work at
L=36 and lose at L=6", and nobody has explained why.

**Two Cascade-Lake-specific mechanisms that would explain it, with the counters to tell them
apart.** Stated as hypotheses; the evidence for each premise is cited, the conclusion is mine.

1. **The victim-L3 reuse hypothesis.** §1.3 is measured: on CLX a streaming access pattern
   still gets 20 % L3 hit rate at 4× L3 and detectable reuse at 10× L3, because the streaming
   replacement policy confines newcomers to one of eleven ways. The driver calls
   `fft3d_execute()` thousands of times on the *same* `in` and `out` buffers, so the RFO reads
   of `out` in iteration k+1 can hit lines left in L3 by iteration k. At L=6 the largest cell
   is 216 MiB (≈10× L3) but the *per-volume* footprint is 6.9 KiB and the traversal is
   perfectly sequential, which is the pattern the CLX policy is tuned to preserve. NT stores
   forfeit that hit rate to save the RFO — and if the hit rate exceeds ~1/3, forfeiting it is
   a net loss. At L=36 the per-volume footprint is 1.49 MB, larger than L2 and a 15th of L3,
   so there is nothing to preserve and NT wins. **This hypothesis predicts the observed L=6 /
   L=36 split exactly.**
   *Counters:* `MEM_LOAD_RETIRED.L3_HIT` / `L3_MISS`, and
   `OFFCORE_REQUESTS.ALL_DATA_RD` vs `L2_RQSTS.ALL_RFO` with and without NT.
2. **The store-buffer occupancy hypothesis — and Intel documents it for exactly this
   microarchitecture.** *Intel 64 and IA-32 Architectures Optimization Reference Manual*,
   Volume 1, Ref# 248966-048, §2.5.2 "Non-Temporal Stores on Skylake Server Microarchitecture"
   — https://cdrdv2-public.intel.com/671488/248966-Software-Optimization-Manual-V1-048.pdf
   [VERIFIED — fetched], verbatim:

   > "because of a change in the handling of accesses resulting from non-temporal stores by
   > Skylake Server microarchitecture, the resources within each core remain busy for a longer
   > duration compared to similar accesses on the previous Intel Xeon processor family. As a
   > result, if a series of such instructions are executed, there is a potential that the
   > processor may run out of resources and stall, thus limiting the memory write bandwidth
   > from each core. The increase in cache misses due to overuse of non-temporal stores and
   > the limit on the memory write bandwidth per core for non-temporal stores may result in
   > reduced performance for some applications."

   Skylake Server *is* our microarchitecture (Cascade Lake-SP is the Skylake-SP core with a
   process and stepping refresh). **Intel is telling us, in its own optimisation manual, that
   per-core NT-store write bandwidth on this part may be capped by core resource occupancy.**
   That is a sourced mechanism for three rounds of tuner rejections, and it removes the need
   for speculation.

   The same conclusion from the other direction, John D. McCalpin, *Notes on "non-temporal"
   (aka "streaming") stores*, 1 January 2018 — original at
   `sites.utexas.edu/jdm4372/2018/01/01/notes-on-non-temporal-aka-streaming-stores/`, now
   retired, retrieved via `web.archive.org` [VERIFIED — fetched from the archive]:

   > "L2 hardware prefetchers on this system are also able to perform prefetches for store
   > miss streams, thus reducing the occupancy for store misses and increasing the store miss
   > bandwidth. For non-temporal stores there is no concept corresponding to 'prefetch', so
   > you are stuck with whatever buffer occupancy the hardware gives you… on the Xeon E5-2680
   > (Sandy Bridge EP), non-temporal stores have *higher* occupancy than store misses that
   > activate the L2 hardware prefetcher, so using non-temporal stores slows down the
   > performance of each of the four STREAM kernels… IIRC, STREAM Triad runs at about
   > **10 GB/s** with one thread on a Xeon E5-2680 when using non-temporal stores and at
   > between **12-14 GB/s** when *not* using non-temporal stores."

   He also gives the concurrency arithmetic that §1.1 derives independently: with a 79 ns idle
   latency and 51.2 GB/s socket peak you would need "4045 Bytes 'in flight' at all times…
   This rounds up to 64 cache lines in flight, while a single core only supports 10 concurrent
   L1 Data Cache misses… this limits a single thread to a read bandwidth of
   **10 lines × 64 Bytes/line / 79 ns = 8.1 GB/s**", and the best he achieved on that system
   was "about **17.8 GB/s**, which corresponds to an 'effective concurrency' of about 22 cache
   lines". Note his own dating caveat: the numbers are from 2015 and "newer systems require
   more concurrency."

   *Counter to confirm on our node:* `L1D_PEND_MISS.PENDING / L1D_PEND_MISS.PENDING_CYCLES`
   (average outstanding L1 misses) and `L1D_PEND_MISS.FB_FULL`, with and without NT. If the NT
   variant shows *lower* average outstanding misses or higher `FB_FULL`, this is the mechanism.

**Generational summary, all from verified sources.** Sandy Bridge / Ivy Bridge: single-core NT
stores are a **loss** (McCalpin above; and Johannes Hofmann & Dietmar Fey, *An ECM-based
energy-efficiency optimization approach for bandwidth-limited streaming kernels on recent Intel
Xeon processors*, arXiv:1609.03347 — https://arxiv.org/pdf/1609.03347 [VERIFIED — fetched]:
"there are shortcomings in SNB and IVB that make single-core implementations using NT stores
slower than their regular stores counterpart"). Haswell-EP: a clean **1.5× per core**
(Hofmann & Fey: "On HSW, per-core performance as expected is exactly 1.5× faster with NT
stores"). Skylake Server / Cascade Lake: Intel's own warning above, and **no published
single-core measurement exists.** Our node's three rounds of tuner rejections at L=6 may be the
first data point on it.

**One more sourced reason NT stores may be less attractive here than the textbook says:**
automatic hardware write-allocate evasion (`SpecI2M`) arrived with **Ice Lake-SP and does not
exist on Cascade Lake** — Jan Laukemann, Thomas Gruber, Georg Hager, Dossay Oryspayev, Gerhard
Wellein, *CloverLeaf on Intel Multi-Core CPUs: A Case Study in Write-Allocate Evasion*,
arXiv:2311.04797 — https://arxiv.org/pdf/2311.04797 [VERIFIED — fetched]: "With the Ice Lake SP
microarchitecture, Intel introduced a new data transaction type called 'SpecI2M'… this WA
evasion mechanism is ineffective with serial code but kicks in when approaching bandwidth
saturation within a memory domain." They also confirm the baseline: "The standard stores… all
have a store ratio of 2.0 with a single core, i.e., each store requires a WA." So on our node
the RFO is unavoidable for ordinary stores, and it would *still* be unavoidable on a newer part
because we are single-threaded. Explicit NT stores are the only way to delete it — which is why
this is worth getting right rather than abandoning.

**The exact ratio for our kernel is 3:2, not 4:3.** McCalpin (verified above): "for the STREAM
Copy and Scale kernels, using cached stores results in two memory reads and one memory write,
while using non-temporal stores requires only one memory read and one memory write — a 3:2
ratio. Similarly, the STREAM Add and Triad kernels transfer 4:3 as much data." Georg Hager,
*The McCalpin STREAM benchmark: How to do it right and interpret the results*, 9 March 2019 —
https://blogs.fau.de/hager/archives/8263 [VERIFIED — fetched] gives the same as correction
factors "1.5× (Copy/Scale) and 1.33× (Add/Triad)". **An out-of-place FFT output pass is
Copy-shaped: one load stream, one store stream. So our NT ceiling is 1.5×, not the 4/3 that
Alappat et al.'s Triad measurement gives** — 24 KiB/transform down to 16 KiB at L=8.

**Two practical rules regardless of which mechanism it is.**

* NT stores only pay on **full 64-byte-line, sequential** stores. A partial-line or strided NT
  store forces a read-modify-write of the line and is worse than a normal store — §05 §8.2
  already carries Drepper's 25 %-slower measurement for the strided case, and it is consistent
  with the mechanism above. So NT belongs only on a final, unit-stride, batch-contiguous
  write-out.
* Because the choice is *measurably* size-dependent, both variants must ship as tuner
  candidates in every entry. The r3 verdict's process lesson — "on this hardware pair, add
  candidates; do not replace structures" — is the same conclusion arrived at from the other
  direction.

### 1.6 Software prefetch: compute the distance, and prefer concurrency to depth

**Source.** Jaekyu Lee, Hyesoon Kim, Richard Vuduc, *When Prefetching Works, When It Doesn't,
and Why*, ACM Transactions on Architecture and Code Optimization 9(1), Article 2, March 2012 —
https://vuduc.org/pubs/lee2012-taco.pdf [VERIFIED — fetched, full text]. Older than this
section's 2015 target but it is the paper that states the distance rule and quantifies the
failure modes, and nothing newer supersedes it for our purposes.

* **The distance formula**, attributed to Mowry et al. 1992: `D ≥ ⌈l / s⌉`, "where l is the
  prefetch latency and s is the length of the shortest path through the loop body", with D
  measured in cache blocks. Their caution: "if it is too large, prefetched data could evict
  useful cache blocks, and the elements in the beginning of the array may not be prefetched,
  leading to less coverage and more cache misses."
* **Distance sensitivity is usually low.** They categorise SPEC benchmarks into five groups by
  the shape of the optimal-distance zone (Table VII); seven of thirteen are simply
  "insensitive". And on transferring a static distance across machine configurations: "we
  conclude that the static prefetch distance variance does not impact performance
  significantly even if the machine configuration is changed at runtime."
* **Bandwidth is rarely the thing prefetching breaks in a single-threaded code**: their SW+B
  decomposition (bandwidth effect) is "mostly small… show[ing] that current machines provide
  enough bandwidth for single-thread applications", while the SW+L component shows "software
  prefetching is not completely hiding memory latency". In our terms: for one thread, the
  problem is latency/concurrency, not bandwidth contention — the same conclusion as §1.1.
* The negative results are worth knowing because the panel has hit them: software prefetches
  can **negatively train** the hardware prefetcher ("Software prefetch requests can slow down
  the hardware prefetcher training… which results in early requests"), and they can also
  *positively* train it, worth "up to 3–5 %".

**The numbers for our kernels** *(my arithmetic from the panel's own measured times and
Velten et al.'s 80 ns / 200-cycle DRAM latency; label it as derived, not published)*:

```
l  = 80 ns = 184 cycles at 2.30 GHz
L=36, B=256:  1.49 MB/volume = 23 328 lines, measured 227.5 us = 523 000 cycles
              s ~ 22.4 cycles per line  ->  D >= ceil(184/22.4) = 9 lines  (~0.6 KiB)
              at the 124 us roofline:  s ~ 12.2 cy/line -> D >= 16 lines (~1 KiB)
L=8,  B=2048: 16 KiB/volume = 256 lines, measured 1.243 us = 2 859 cycles
              s ~ 11.2 cycles per line  ->  D >= ceil(184/11.2) = 17 lines (~1 KiB)
L=6,  B=32768: 6912 B/volume = 108 lines, measured 0.563 us = 1 295 cycles
              s ~ 12.0 cycles per line  ->  D >= 16 lines (~1 KiB)
```

Three of the four land on **≈1 KiB ahead**, i.e. **16 cache lines**, or a quarter of a 4 KiB
page. If an entry is prefetching one volume ahead at L=6 (6.9 KiB) or one whole volume ahead at
L=36 (1.49 MB), it is 7× to 1500× beyond the computed distance — deep into the regime Lee et al.
describe as eviction and coverage loss, and beyond the point where the hint can survive in the
L1/L2 queues at all. **That, not the machine difference, is the most likely reason the
cross-volume prefetch schemes were rejected by the node's tuners in three of three L=36
entries.** The cheap experiment is a distance sweep centred on 16 lines, not a
prefetch-on/prefetch-off A/B.

### 1.6b Which prefetch instruction, how many fill buffers there are, and the 1.29× on the table

This is the strongest cross-source agreement anywhere in this section, and it contradicts the
panel's working assumption that the hardware streamer already handles unit-stride streams.

**There are exactly 10 line fill buffers, measured on Cascade Lake.** Roland Kühn, Jan Mühlig,
Jens Teubner, *How to Be Fast and Not Furious: Looking Under the Hood of CPU Cache
Prefetching*, DaMoN '24 (20th International Workshop on Data Management on New Hardware) —
https://dbis.cs.tu-dortmund.de/storages/dbis-cs/r/papers/2024/sw-prefetching-survey/sw-prefetching.pdf
[VERIFIED — fetched]. Their Intel part is a **Xeon Gold 6226 — Cascade Lake**, i.e. our exact
microarchitecture and a near-sibling SKU of the Gold 5218:

> "there is a marked latency escalation starting from the eleventh cache line, where the
> average latency values leap from 7 to a range of 200 – 340 cycles. This pattern suggests that
> the underlying Cascade Lake architecture has **exactly 10 LFB slots**, which is in line with
> (unofficially published) numbers for the very similar Skylake architecture."

corroborated with `L1D_PEND_MISS.FB_FULL`, which "increases from almost zero (when prefetching
a single cache line) to 1.2 per cache line when prefetching a block of 16 cache lines". AMD
Zen 4 needed 32 lines to show the same knee, "reflecting the MAB's larger capacity of 24 slots".
Cost per prefetch instruction: "ranging around **35 additional cycles per prefetch**" with 4 KiB
pages, dropping to "approx. **20 stalled cycles per cache line**" with 1 GiB huge pages — a
second, independent reason to care about §1.7.

**Use `prefetcht1` (into L2), not `prefetcht0`/`prefetchnta`.** Three verified sources agree:

* Kühn et al.: "Our analysis of different instructions reveals that prefetching into the L2
  cache (via `prefetcht1`) tends to minimize latency penalties, especially when prefetching
  **fewer than 12 cache lines**." Mechanism: "prefetching directly into the L1d cache (using
  `prefetchnta` and `prefetcht0`) circumvents requests to the L2 cache—preventing it from
  'learning' from these access patterns."
* Ioan Hadade, Timothy M. Jones, Feng Wang, Luca di Mare, *Software Prefetching for
  Unstructured Mesh Applications*, ACM Transactions on Parallel Computing 7(1), Article 3,
  March 2020 — https://api.repository.cam.ac.uk/server/api/core/bitstreams/9dd57510-1512-4ef5-9936-0bfdd47a9f38/content
  [VERIFIED — fetched]: "best results are obtained when prefetches are only executed for the
  larger L2 cache (`prefetch1`)", because "the L1 prefetches hold critical hardware resources
  such as Line Fill Buffers until the cache line fill completes whereas L2 prefetchers do not."
* McCalpin (verified above): "L1 hardware prefetchers don't help performance here because they
  share the same 10 L1 cache miss buffers. L2 hardware prefetchers do help because bringing the
  data closer reduces the occupancy required by each cache miss transaction."

**Software prefetching is worth 1.29× single-core on this generation, and the win comes from
prefetching the *regular* streams.** Hadade et al.'s SKX platform is a **Xeon Gold 6152**
(22 cores, 2.1 GHz, 32 KiB L1 / 1024 KiB L2 / 30.25 MiB L3, DDR4, STREAM 98 GB/s, icpc 18.3):

> "Compared to the Sandy Bridge and Broadwell systems, software prefetching results in up to
> **1.29× full application speed-up on the Skylake system in a single core** and 1.09× at full
> socket concurrency. The speed-ups obtained in the face-based loops range between **1.18× and
> 1.61×**… We attribute these to the larger L2 cache in the Skylake system, which is four times
> greater than the L2 caches on the Sandy Bridge and Broadwell systems, as well as the fact
> that the L3 cache on Skylake is configured as a victim cache (non-inclusive). This is further
> evidenced by the fact that best results are obtained when prefetches are only executed for
> the larger L2 cache (prefetch1)."

> "In contrast to Sandy Bridge and Broadwell, prefetching only the indirect accesses in
> face-based loops does lead to significant improvements in performance although **inserting
> prefetches for the regular accesses as well is by far the best approach**."

Their tuned distances, for a vectorised loop: start at one vector iteration ahead and step in
powers of two — L1 distance ∈ {8, 16, 32} elements, L2 distance ∈ {2×, 4×, 8×} the L1 distance,
plus L1-only and L2-only variants. On Sandy Bridge "best results are obtained when we only
target the prefetches for either the L1 or the L2 cache rather than both" because of "cache
pollution that results from overlapping prefetches at different distances". And the caveat that
does *not* apply to us: "a large proportion of benefits derived from software prefetching
disappears once we run on the full socket (22 cores) rather than on a single core." **We are
single-threaded, i.e. in the regime where their measurement is 1.29×, not 1.09×.**

**Why the hardware streamer cannot help our axis passes: our streams are too short.** Intel's
optimization manual (verified above) on the automatic hardware prefetcher: "There is a start-up
penalty before the prefetcher triggers… For short arrays, overhead can reduce effectiveness. —
The hardware prefetcher requires **a couple misses before it starts operating**. — Hardware
prefetching generates a request for data beyond the end of an array, which is not be utilized.
This behavior wastes bus bandwidth… It will not prefetch across a **4-KByte page boundary**."
And: "DPL and L2 Streamer are triggered only by writeback memory type… DPL can also be
triggered by read for ownership (RFO) operations" — note that last clause, because it is
precisely the help that NT stores forfeit (§1.5).

Now count our streams. One L=8 line of complex doubles is **128 bytes = 2 cache lines**; an
L=6 line is 96 bytes; an L=17 line is 272 bytes. **Every axis pass in every one of our kernels
is a stream of two to five cache lines.** The streamer needs "a couple misses" to train and
cannot cross a page. It will essentially never train on our inner streams.

The strongest corroborating measurement is Lee, Kim & Vuduc's (verified in §1.6) analysis of
`433.milc`, whose kernels "operate on 3×3 matrices… these matrix accesses correspond to short
streams, which are **too short to train the hardware prefetchers**. By contrast, software
prefetching can effectively prefetch these arrays, provided the prefetch requests are inserted
appropriately near the call site. When doing so, we observe a **2.3× speedup** compared with the
best hardware-only prefetching scheme." A 3×3 complex matrix kernel batched over a long
outer loop is a very close structural analogue of our problem.

**The rule this produces for our contract.** The long stream in our problem is not any axis;
it is the **batch axis**. Issue `_mm_prefetch(..., _MM_HINT_T1)` along the batch direction,
one to four volumes ahead — at L=8 that is 8–32 KiB ahead, which brackets the ≈1 KiB the
distance formula gives for the innermost stream and the ≈16 KiB a whole-volume lookahead gives
— sweep it in powers of two the way Hadade et al. did, and **keep it under 10 outstanding lines
per hint burst** (Kühn et al.'s knee). Do not use `_MM_HINT_T0` or `_MM_HINT_NTA`. This is a
concrete, sourced, unexplored change: the panel's r3 records show entries selecting `pf=0`
versus `pf=1` as a binary, with `prefetchw` and T0 hints, at whole-volume distances — none of
the three parameters this literature says matter.

**Finally, on concurrency.** If a prefetch scheme's purpose is to *raise the number of
in-flight lines* rather than to move a single stream's requests earlier, express it as several concurrent demand
streams (e.g. process 4 volumes' z-pencils in an interleaved inner loop, each on its own 4 KiB
page) rather than as `_mm_prefetch` hints on one stream. Hints are droppable; demand loads are
not.

### 1.7 Huge pages and NUMA balancing: a documented 2× on this microarchitecture

Alappat et al., "Influence of machine and environment settings" [VERIFIED — fetched]:

> "Figure 1(a) shows the influence of different operating system (OS) settings on a **serial**
> load-only benchmark running at 1.6 GHz on CLX for different data-set sizes in L3 and main
> memory. With the default OS setting (NUMA balancing on and transparent huge pages (THP) set
> to 'madvise'), we can see a **2× hit in performance for big data sets**. This behavior also
> strongly depends on the OS version."

They set `NUMA balancing off` and `THP always` for everything afterwards. Velten et al.
independently state they "set `/sys/kernel/mm/transparent_hugepage/enabled` to `always` to
clearly distinguish memory levels". Alappat et al. also measure that **disabling sub-NUMA
clustering reduces single-core main-memory performance by 4 %**.

**Action, and it is the monitor's, not an implementer's:** read
`/sys/kernel/mm/transparent_hugepage/enabled` and
`/proc/sys/kernel/numa_balancing` on the benchmark node and record them in the round's
provenance block, exactly as the gcc version and governor already are. If THP is `madvise`, a
factor of up to 2 on every large-batch cell is sitting in an OS setting, and — importantly —
implementers *can* reach part of it themselves: `madvise(MADV_HUGEPAGE)` on any scratch buffer
allocated in `fft3d_create()` is inside the rules and is precisely what `madvise` mode
requires. The driver's `in`/`out` buffers are outside our control, so this is one of the few
places where a monitor-side setting caps what the panel can achieve. §05 §7's advice to use
huge pages was right; what is new is that on this microarchitecture the penalty for not having
them has been *measured at 2× for a serial stream*, and that the default Ubuntu setting is the
bad one.

### 1.8 4 KiB store-to-load aliasing: the hazard an out-of-place transform hits by construction

**Mechanism, sourced.** Dean Sullivan, Orlando Arias, Travis Meade, Yier Jin,
*Microarchitectural Minefields: 4K-Aliasing Covert Channel and Multi-Tenant Detection in IaaS
Clouds*, NDSS 2018 —
https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_06A-3_Sullivan_paper.pdf
[VERIFIED — fetched, full text]. The memory order buffer compares only the low 12 bits of load
and store addresses when checking for write-after-read hazards, so "a WAR hazard may be
detected falsely on loads and stores whose addresses are separated by a multiple of 4096" and
"the load is re-issued with an associated performance penalty". They evaluate it on Sandy
Bridge, Ivy Bridge, Haswell and **Skylake** (their Skylake part is an i7-6820HQ) with a memory
copy whose source and destination are separated by 4 KiB: "the copy bandwidth drops every time
an address is aligned on 4 KB boundary. Subsequent copies to addresses that do not align on a
4 KB boundary rapidly recover." (Figure 1 is a plot; I cannot read exact values off it.) They
also note Intel's VTune documentation describes 4K-aliasing as exactly this MOB side-effect.

**A measured magnitude.** Jakub Beránek, `hardware-effects/4k-aliasing` —
https://github.com/Kobzol/hardware-effects/blob/master/4k-aliasing/README.md
[VERIFIED — fetched]. Same loop, stride 4000 B (non-aliasing) versus 4092 B (aliasing):
**310 cycles / 3 378 `ld_blocks.store_forward`** against **379 cycles / 2 100 223
`ld_blocks.store_forward`** — a 22 % slowdown and a ~620× increase in blocked loads. The CPU
model is not stated in that README, so treat the 22 % as indicative of scale, not as a Cascade
Lake number.

**Why we are exposed, and it has two levels.** `driver.c` hands us two separate 64-byte-aligned
buffers.

*Level 1, and this is the dominant one: the offset between the two buffers.* Allocations of 27, 32
or 46 MiB come from `mmap` and are page-aligned, so `out − in` is a multiple of 4096 with near
certainty. Our kernels then load `in[i]` and store `out[j]` where i and j run over the same index
space — so **for every element, a load and a recent store share the same low 12 address bits.** This
affects **all four geometries at every element**, and it is invisible in any reasoning about cache
sets or strides, because it is a property of the *pair* of buffers, not of either one.

*Level 2: the volume stride, which decides how degenerate the pattern is.* Whether volume k and
volume k+1 also start at the same page offset:

| L | bytes/volume | mod 4096 | volumes between repeats of a page offset |
|---|---|---|---|
| 6 | 3 456 | 3 456 | 32 |
| **8** | **8 192** | **0** | **1 — every volume starts at the same page offset** |
| 17 | 78 608 | 784 | 256 |
| **36** | **746 496** | **1 024** | **4** |

**L=8 is maximally degenerate (period 1) and L=36 is next (period 4) — and those are the two
geometries with the flattest curves and the tightest margins on the board.** L=6 (32) and L=17 (256)
spread their page offsets far more, which is not protection against Level 1 but does mean the
aliasing is not reinforced volume-to-volume.

**How to test it, cheaply.** `perf stat -e ld_blocks_partial.address_alias` (Skylake/Cascade
Lake name; `ld_blocks.store_forward` on some kernels) on an L=8 B=2048 and an L=36 B=32 run. If
the count is on the order of the number of loads, it is happening.

**How to fix it inside the contract.** You cannot move `in` or `out`, and you cannot change their
relative offset. You can:
* **Stage through a scratch buffer with a deliberately odd page offset.** Allocate scratch in
  `fft3d_create()` and offset the pointer you actually use by an odd multiple of 64 B
  (e.g. `+ 64` or `+ 192`) so that scratch↔in and scratch↔out relative offsets are never
  0 mod 4096. This is free — it changes one addition in `create()`.
* **Separate the load and store phases in time within an unrolled block.** The hazard is a
  *recent* store aliasing a load; a block that does all its loads, then all its arithmetic,
  then all its stores, exposes far fewer alias windows than one that interleaves them per
  element. Note that `L17_matrixsimd`'s "X-first pass reordering" — which changed no arithmetic
  and no traffic volume, only *when* the writes happen, and bought **10.8 %** at B=256 — is
  precisely a change of this shape. **A plausible re-reading of the largest single win of round
  3 is that it removed store-to-load alias stalls.** That is a testable claim and the counter
  above tests it.
* **Skew the output traversal.** If pass k writes `out` in an order offset by one cache line
  relative to the order it reads `in`, the low-12-bit collisions stop being systematic.

This is the highest expected-value unexamined item in this section: the mechanism is
documented, the exposure is structural, our two worst geometries are the two exactly aligned
ones, and the test is one counter.

### 1.9 Cache-blocking ACROSS the batch: the one technique three independent sources endorse for this part

This is, in my reading, the largest untapped structural win on the board, and it is the answer
to "the volume should be read once" that the panel has been circling for three rounds without
naming.

**Intel's own recommendation for Skylake Server**, from the *Optimization Reference Manual*
Vol. 1 §2.5.1.3 [VERIFIED — fetched], verbatim:

> "Recommendation: Rebalance application shared and private data sizes to match the smaller,
> non-inclusive L3 cache, and larger L2 cache."
> "Having four times the L2 cache size and twice the L2 cache bandwidth compared to the previous
> generation Broadwell microarchitecture enables some applications to **block to L2 instead of
> L1** and thereby improves performance."
> "Recommendation: Consider blocking to L2 on Skylake Server microarchitecture if L2 can sustain
> the application's bandwidth requirements."
> "Recommendation: In case of no data sharing, applications should consider cache capacity per
> core as **L2 and L3 cache sizes** and not only L3 cache size."

And its Table 2-9, Broadwell → Skylake Server, which is the quantitative case:

| level | metric | Broadwell | **Skylake Server** |
|---|---|---|---|
| L1 DCU | max / sustained bandwidth | 96 / 93 B/cy | **192 / 133 B/cy** |
| L2 MLC | size | 256 KiB | **1024 KiB** |
| L2 MLC | latency | 12 cy | 14 cy |
| L2 MLC | max / sustained bandwidth | 32 / 25 B/cy | **64 / 52 B/cy** |
| L3 LLC | size per core | up to 2.5 MiB | up to 1.375 MiB |
| L3 LLC | latency | 50–60 cy | 50–70 cy |
| L3 LLC | max / sustained bandwidth | 16 / 14 B/cy | 32 / **15 B/cy** |

**Sustained L2 = 52 B/cy; sustained L3 = 15 B/cy; single-core DRAM ≈ 7 B/cy (§1.2).** L2 is
**3.5×** L3 and **7–8×** DRAM. That is the entire argument, from the vendor, in one row.

**The construction, applied to our contract.** Take L=8. One volume is 8 KiB in and 8 KiB out.
A **tile of NB volumes** occupies `16 × NB` KiB of live data. Choose NB so the tile plus its
scratch is comfortably under half the 1 MiB L2:

```
L=8 :  NB = 24  ->  384 KiB live (in+out tile)  ->  ~37% of L2
L=6 :  NB = 32  ->  216 KiB                     ->  ~21% of L2
L=17:  NB = 2   ->  307 KiB                     ->  ~30% of L2
L=36:  NB = 1 does not fit (1.46 MiB) -> tile WITHIN the volume (planes/pencils), not across it
```

Then: stream the tile in once, run **all three axis passes entirely inside L2**, stream it out
once. DRAM traffic becomes the compulsory `2 ×` volume bytes *regardless of how many passes the
algorithm makes*, every intermediate pass runs at 52 B/cy instead of 7 B/cy, and the only
DRAM-visible store is a single long fully-line-aligned output stream per tile — which is also
the only shape in which an NT store can pay (§1.5) and the only shape long enough to train the
L2 streamer (§1.6b). **One structural change addresses four of this section's findings at
once.**

**Precedent, with numbers.** Rati Gelashvili, Nir Shavit, Aleksandar Zlateski, *L3 Fusion: Fast
Transformed Convolutions on CPUs*, arXiv:1912.02165 — https://arxiv.org/pdf/1912.02165
[VERIFIED — fetched]. Machine: "an 18 core Intel 7980xe with 2.6ghz 4 memory channels each
21.3gb/sec, 20mb shared L3 cache and **1mb L2 per-core cache**" — the same 1 MiB L2 as our node.
They size the fused working buffer to *half* the L2: "Analogous derivation for SkylakeX gives
the following requirement `R max(C,C′)·(T² + 1) ≤ 128kb`". Result: "L3-fused algorithm reliably
and significantly outperforms the best of all 3 other implementations on all layers with 64 and
128 channels. On 64 channel layer of ResNet (VGG), L3 fusion takes 3.16 (46.27) ms as opposed to
the second [best]…" — against ZNN, DNNL/MKL-DNN and their own non-fused three-stage baseline.
They also publish a reusable calibration: "we get that the L3 CMR of the SkylakeX processor was
around 10… For the main memory, the CMR can be easily computed as the ratio of the processors
peak FLOPS and the memory bandwidth… Which was **35** for the SkylakeX". *Caveats: fp32,
multi-threaded, convolution layers, Skylake-X desktop. What transfers is the L2-sizing
derivation and the fuse-so-the-intermediate-never-reaches-DRAM structure, not the numbers.*

**How close this class of code gets to the roofline.** Aleksandar Zlateski, Zhen Jia, Kai Li,
Fredo Durand, *FFT Convolutions are Faster than Winograd on Modern CPUs, Here's Why*,
arXiv:1809.07851 — https://arxiv.org/pdf/1809.07851 [VERIFIED — fetched] — the closest thing in
the literature to "very many small FFTs on a CPU", with a **Xeon Gold 6148** among the machines:

> "While the utilization varied across benchmarked layers and systems, on average, during the
> compute bound stages, **75 % of theoretical peak FLOPS** were attained; in memory bound stages
> slightly more than **85 % of the theoretical memory bandwidth** was achieved."

and their arithmetic-intensity framing, which is ours: "The largest AI of the FFT transforms is
5.55, and for Winograd 2.38, much lower than CMRs of the modern systems." *Caveats: fp32,
multi-threaded, streaming stores in use, "theoretical peak" rather than measured STREAM.*
Nevertheless: **85 % of peak bandwidth in the transform stages is the target number, and it is
achieved by exactly the tile-and-fuse structure described above.**

**Relationship to the corpus's pass-fusion argument, and to the panel's own r3 result.** This is
*not* the same claim as `LITERATURE.md` §4.3, and the distinction matters. §4.3 asked whether
fusing passes over an already-**L1/L2-resident** intermediate pays; the panel measured it at
single-digit percent, sometimes negative, and that measurement stands. What §1.9 proposes is
fusing passes over an intermediate that would otherwise **go to DRAM** — a 7 B/cy versus
52 B/cy gap rather than an L1-versus-L2 gap. Tolmachev's rule from §07 §1.6, which the r3
verdict endorsed, predicts exactly this asymmetry: *payoff = passes avoided × the bandwidth gap
between the two levels involved.* At L1↔L2 the gap is 2.6× and the payoff was ~5 %; at
L2↔DRAM the gap is 7× and the payoff should be large. **The panel has tested pass fusion only
in the regime where the corpus predicts it does not pay.**

A supporting statement on pass count from a practitioner design note: Fabian Giesen, *Notes on
FFTs: for implementers*, 19 March 2023 —
https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/
[VERIFIED — fetched]: "Radix-4 happens to have fewer multiplies, but much more importantly, it
does the work in roughly half the number of passes, and thus **halves the number of coefficient
loads/stores**… Prefer radix-4 (or radix-2²), less so for lower number of multiplies, and more
because having fewer passes and fewer loads/stores is good and memory operation count is not
negligible for this." And a warning that bears on any ping-pong scratch design: Stockham
"'ping-pongs' between two buffers on every pass, which means it has about **twice the active
working set in the cache** as an in-place algorithm (and more traffic in upstream caches or
memory when it doesn't fit anymore)." A batch tile that ping-pongs doubles the NB you can
afford — size it accordingly.

### 1.10 Layout: pick the batch-interleave granule from the cache line, not from the register

§04's split-complex batch-minor recommendation is right, and this section adds the one parameter
it does not fix: **how many volumes go in an interleave granule.**

Zlateski et al. (verified above), on their batched small-FFT convolution:

> "We adopted the data layout proposed in [17, 18, 38] for input images, kernels and output,
> where **16 images are interleaved in memory** for easy vectorization. In [18] 16 was used due
> to the size of the AVX512 vector register, we keep it to 16 regardless of the vector register
> size, as **16 is the cache-line width** (16 32-bit floats), to facilitate efficient
> utilization of the memory subsystem."

They deliberately set the granule from the **cache line**, not the register. For fp32 those
coincide at 16. **For complex double they do not:** a 64-byte line holds 8 doubles, so in split
form the natural granule is **8 volumes per granule** (one zmm of 8 real parts = exactly one
cache line), and on a 256-bit path it is still 8 if you want whole-line traffic — i.e. two
`ymm` loads per granule per component. A granule of 4 gives half-line accesses on every load and
store, which is exactly what the NT-store rule in §1.5 and the streamer-training rule in §1.6b
say to avoid. **Recommendation: NB_granule = 8 for split complex double, on both the 256-bit and
512-bit paths, so every vector access is a whole cache line.** Then tile the batch above that in
multiples of 8 up to the L2 budget of §1.9.

Fabian Giesen (verified above) independently arrives at the same hybrid and describes it
precisely as an internal-only format, which fits our contract exactly (we repack on the way in):

> "Yet another option is to keep the data interleaved, but at the granularity of full vectors
> (or some larger granularity that is a multiple of the vector size). For example, for 8-wide
> SIMD, you might store real parts of elements 0-7, then imaginary parts of elements 0-7, then
> real parts of elements 8-15, and so forth. This is still addressed by a single base pointer
> and has nice data access locality characteristics, but is just as (if not more) convenient as
> fully deinterleaved for SIMD code. The trade-off here is that your vector width… now becomes
> part of your interface; I like this format inside FFT kernels, less so as part of their public
> interface."

He also disposes of the "AoS is fine" argument in the terms this project cares about: with an
interleaved layout in registers, "instead of N 'real-valued' SIMD lanes, we effectively end up
with N/2 'complex-valued' SIMD lanes. The amount useful work per individual float remains
unchanged (at best; in practice, using an interleaved format inside the kernel often picks up
overheads from elsewhere)." *Caveat, in his own words: "I have personally written and used FFT
kernels for small to medium problem sizes (everything still fits comfortably in a 256k L2
cache)… anything that concerns large transforms specifically… is not something I can comment
on." No measured numbers in that post.*

**Two measured data points on layout, for scale:**

* Yifei He, Artur Podobas, Stefano Markidis, *Leveraging MLIR for Loop Vectorization and GPU
  Porting of FFT Libraries*, arXiv:2308.00497 — https://arxiv.org/pdf/2308.00497
  [VERIFIED — fetched]. On a dual-socket **Xeon Gold 6130**: "The green line represents the
  sparsified code with SLP vectorization; it achieves approximately 2x speed-ups for most cases.
  The most significant performance gain comes from the interleaved memory access optimizations:
  **up to 5x speed up** is achieved." *This is a compiler result — the 5× is the cost of getting
  AoS access wrong (gather/scatter replaced by "two consecutive SIMD loads followed by shuffle
  operations"), which a hand-written intrinsics kernel already avoids. Read it as a measure of
  the penalty, not of the SoA-vs-AoS gap in tuned code.* Also directly relevant to §4: "The
  current heuristic in LLVM loop vectorizer to choose the vector length will block AVX512
  instructions for Intel CPUs, due to the probable frequency [reduction]" — they had to override
  `PreferVectorWidth`.
* Vedran Novaković, *Batched computation of the singular value decompositions of order two by
  the AVX-512 vectorization*, arXiv:2005.07403 — https://arxiv.org/pdf/2005.07403
  [VERIFIED — fetched]. Batched 2×2 SVDs on a Xeon Phi 7210: "the vectorized algorithm is nearly
  or more than **three times faster**, in the complex and the real case." His layout rule is ours:
  "A vector should hold S elements from the same matrix sequence, with the same row and column
  indices… When computing with complex numbers, however, it is **more efficient to keep the real
  and the imaginary parts of the elements in separate vectors**, since there are no hardware
  vector instructions for the complex multiplication and division." *Caveats: KNL, not Cascade
  Lake; SVD, not FFT. It transfers as methodology, not as a number.*

**What does not exist.** Neither research pass found a published head-to-head of AoS versus
split versus batch-interleaved for **complex-double batched small FFTs on a Skylake-SP-class
CPU**. §04 §4.4 flagged this gap and it is still open. The panel's own measurement would be the
first.

### 1.11 What the literature on "many small FFTs on a CPU" actually contains

Worth stating explicitly so nobody spends a round searching. Both research passes converged on
the same conclusion: **batched small FFTs on CPUs is a genuine hole in the published
literature.** Almost all batched-FFT work is GPU work. What exists on the CPU side, all
[VERIFIED — fetched]:

* **FFT-based convolution for ML** is the closest analogue and it is CPU-native: Zlateski et al.
  2018 (arXiv:1809.07851) and Gelashvili/Shavit/Zlateski 2019 (arXiv:1912.02165), both used in
  §1.9. Small tiles, batch interleaved at cache-line granularity, streaming stores between
  stages, L2/L3 blocking, roofline analysis. fp32 and multi-threaded.
* **Automatic SIMD vectorisation of small FFTs**: Daniel S. McFarlin, Volodymyr Arbatov, Franz
  Franchetti, Markus Püschel, *Automatic SIMD Vectorization of Fast Fourier Transforms for the
  Larrabee and AVX Instruction Sets*, ICS 2011 —
  https://users.ece.cmu.edu/~franzf/papers/ics2011.pdf. The SPIRAL line; straight-line
  vectorised FFT code with automatic shuffle synthesis over interleaved four-complex vectors.
  Methodologically relevant to our codelets; predates AVX-512 and contains no bandwidth
  analysis. (This is the same group whose 2011 direct-vs-FFT crossover the corpus cites in
  §03 §6.4 / §06 §6.4a.)
* **Plane-wave DFT / ab-initio MD**, i.e. our own application domain: Arjun Ramaswami, Tobias
  Kenter, Thomas D. Kühne, Christian Plessl, *Efficient Ab-Initio Molecular Dynamic Simulations
  by Offloading Fast Fourier Transformations to FPGAs*, arXiv:2006.08435 —
  https://arxiv.org/pdf/2006.08435. Confirms the workload shape — "Accelerating this routine
  involves optimizing execution in ranges of microseconds" — with an FFTW baseline on two
  **Xeon Gold 6148** CPUs: for 64³, **FFTW 0.14 ms vs FPGA 0.74 ms** (single precision, best
  over all four FFTW planners and all thread counts). No batched-CPU optimisation data.
* **PME in molecular dynamics**: *Breaking Down the Parallel Performance of GROMACS*,
  arXiv:2208.13658 — https://arxiv.org/pdf/2208.13658, which reports that "of the PME
  calculation, 50 % of the time is spent on 3D FFT" on Xeon Gold 6130/6132 nodes. Establishes
  that small-grid 3D FFT is a production hotspot; no single-core bandwidth analysis.
* **Lattice QCD specifically**: neither research pass found *any* paper on batched small 3D FFTs
  for lattice QCD on a CPU. The QCD performance literature is about the Dirac operator and about
  GPU frameworks.
* **FFTW's own batched interface** documents `howmany`/`stride`/`dist` — "the input of the k-th
  transform is at location in+k*idist… and its output is at location out+k*odist"
  (https://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html) — and makes exactly one
  performance claim, "Plans obtained in this way can often be faster than calling FFTW multiple
  times for the individual transforms". **It recommends no layout.** So the panel is not
  competing against a documented design; it is competing against an undocumented one.

This is the same gap §03 §7 identified ("you are in genuinely unmeasured territory") and it has
not closed. The practical consequence: **the panel's own numbers are, as far as two independent
literature passes can establish, the best published data in this regime.** They should be
written up as such in the strategy records — including the negative results, which are the part
nobody else has.

### 1.12 Where the roofline actually is, per cell

Combining the compulsory-traffic arithmetic (out-of-place: 2× volume bytes with NT stores,
3× with write-allocate) with §1.1–§1.2's single-core ceiling of ~12 GB/s, and the panel's
r3 measurements:

| cell | compulsory 2× traffic | at 12 GB/s | with write-allocate (3×) | measured r3 | headroom vs 2× bound |
|---|---|---|---|---|---|
| L=6 B=32768 | 6.9 KiB | 0.576 µs | 0.864 µs | 0.563 µs | **already past it** — the volume is not all coming from DRAM |
| L=8 B=2048 | 16 KiB | 1.365 µs | 2.048 µs | 1.243 µs | **already past it** — L3 reuse is real (§1.3) |
| L=8 B=16384 | 16 KiB | 1.365 µs | 2.048 µs | 1.580 µs | 1.16× |
| L=17 B=2048 | 154 KiB | 13.1 µs | 19.7 µs | 22.697 µs | 1.73× (but compute floor is 16.4 µs → 1.39×) |
| L=36 B=32 | 1.46 MiB | 124 µs | 186 µs | 218.4 µs | 1.76× |
| L=36 B=256 | 1.46 MiB | 124 µs | 186 µs | 227.5 µs | 1.83× |

Two readings that change the priorities:

* **L=6 and L=8's large-batch cells are already *faster* than a pure-DRAM 12 GB/s model
  allows.** That is only possible if a significant fraction of the traffic is being served
  from L3 — which is exactly what §1.3 says Cascade Lake does at 1–10× L3 working sets. So
  those cells are not "at the DRAM wall"; they are in a mixed L3/DRAM regime, and the lever
  there is *increasing* the L3 hit fraction (batch tiling, and **not** NT stores), not cutting
  DRAM traffic. This is a genuine correction to the r3 verdict's reading of L=6 as "finished
  at the stream limit": it is at the limit of the *model*, but the model omits CLX's victim L3.
* **L=36 and L=17 remain 1.4–1.8× off, and both gaps are consistent with un-overlapped memory
  latency rather than insufficient bandwidth**: at L=36 B=256 the measured 227.5 µs against a
  186 µs write-allocate bound is only 1.22×, so most of L=36's apparent 1.83× gap is
  write-allocate traffic the panel is already paying. **At L=36, NT stores plus a
  concurrency-raising traversal is the whole remaining prize, and it is bounded by
  186 → 124 µs.**

---

## 2. Target 2 — why the leaders plateaued, and what is actually left

### 2.0 The short answer

Two independent research passes searched 2015–2026 for a *measured* CPU win over separable
row–column on small multidimensional transforms. **Neither found one.** No fetchable source
reports vector-radix, Nussbaumer–Quandalle, or any other genuinely multidimensional butterfly
beating row–column in wall-clock on a CPU. §03's verdict stands, and one 2026 measurement
strengthens it (§2.1).

But that is not the same as saying there is nothing left, and the corpus has been reading the
plateau wrong. **The modern measured wins on small multidimensional transforms all come from
outside the butterfly**: from the layout, from the schedule around the kernels, and from
regime-dependent selection. The decisive piece of evidence is §2.2, a 2015 paper on almost
exactly our problem in which the authors' **own 1D FFT kernels were sometimes slower than
MKL's, and their 3D kernel was still 2–3× faster.** The panel has spent three rounds
perfecting the thing that paper found did not matter.

### 2.1 Vector-radix: confirmed dead, and now with a 2026 flop-count confirmation

The only 2015+ vector-radix paper either pass found with measured runtimes is Keun-Yung Byun,
Chun-Su Park, Jee-Young Sun, Sung-Jea Ko, *Vector Radix 2 × 2 Sliding Fast Fourier Transform*,
**Mathematical Problems in Engineering, 2016** —
https://onlinelibrary.wiley.com/doi/10.1155/2016/2416286 [VERIFIED — fetched]. For a 16×16
sliding window on a "3.3 GHz CPU with 8 GB RAM", 10⁶ iterations: **VR-2×2 SFFT 876.87 ms;
1D DFT×2 23 908.47 ms; 1D gSDFT + 1D FFT 3 812.58 ms; 2D SDFT 532.51 ms.** Read the last
column: the vector-radix method is 27× the naive baseline but **1.65× slower in wall clock than
the plain 2D sliding DFT.** Even the one modern measured vector-radix result does not beat every
non-vector-radix competitor, and it is a sliding-window problem, not a batched cube.

The minimum-multiplication programme is also still alive and still not engaging with hardware:
Ryszard M. Stasiński, *Fast Discrete Fourier Transform algorithms requiring less than O(N log N)
multiplications*, arXiv:2303.02647 (2023) — https://arxiv.org/html/2303.02647
[VERIFIED — fetched] achieves "O(N log^c log N)" multiplicative complexity and contains **no
measured runtime, no benchmark, and no mention of FMA or SIMD anywhere.** That is the cleanest
possible illustration of why §02 §2.7 and §03 §3.3 are right.

And the corpus's consensus #3 ("instructions, not flops") has just been re-measured on the
panel's own development microarchitecture. Nicolas Venkovic, Hartwig Anzt,
*Permutation-Avoiding FFT-Based Convolution*, arXiv:2506.12718v3 (2026, formatted for ACM TOMS)
— https://arxiv.org/html/2506.12718v3 [VERIFIED — fetched], on an Intel Core i5-1334U (Raptor
Lake, GCC 13.3.0) and **dual Xeon Platinum 8480+ (Sapphire Rapids, GCC 11.5.0)**:

> "**radix-4 consistently delivers the best performance across all problem sizes, on both
> systems**"

despite radix-8 requiring "**20 % fewer FLOPs**". Two more findings from the same paper that
bear directly on our pass accounting: index-reversal permutations account for "**48–71 %**" of
total 1D transform time on the i5 and "**55–73 %**" on the 8480+; and — importantly for anyone
tempted to port a 1D trick into 3D — their permutation-avoidance win **evaporates in three
dimensions**: at n₁=n₂=n₃=2⁷, "0.148 s (PA) vs 0.153 s (Std)", because the per-element
permutation cost "decrease[s] dramatically with dimension d at fixed n" while butterfly cost
rises. *Not every 1D optimisation survives the move to 3D.*

Nussbaumer–Quandalle: neither pass found **any** 2010–2026 source measuring NQ polynomial
transforms for multidimensional DFTs on a SIMD/FMA CPU. The one modern measured Nussbaumer
implementation located is a GPU polynomial-multiplication kernel for lattice cryptography and
**could not be fetched** (Springer 303). §03 §3 stands unchallenged and unrefreshed.

### 2.2 The paper that closes `LITERATURE.md` §4.8 gap 1 — and it says the kernel is not the lever

**`LITERATURE.md` §4.8 gap 1 and §03 §7 both assert that "there is no published measurement
telling you what the fastest 6³/8³/17³/36³ complex-double kernel looks like" and that the panel
is "in genuinely unmeasured territory." That is now only half true, and the half that is false
is the important half.**

Doru Thom Popovici, Francis P. Russell, Karl Wilkinson, Chris-Kriton Skylaris, Paul H. J. Kelly,
Franz Franchetti, *Generating Optimized Fourier Interpolation Routines for Density Functional
Theory using SPIRAL*, **IPDPS 2015, pp. 743–752** —
https://users.ece.cmu.edu/~franzf/papers/ipdps15.pdf [VERIFIED — fetched].

**The regime is startlingly close to ours:** a 2×2×2 upsampling of a data cube with **edge
length 7 to 119, odd, possibly rectangular, complex double, single core**, in both split and
interleaved complex layouts, benchmarked against **MKL 11.0.0 and FFTW 3.3.4** on a 3.5 GHz
Haswell 4770K, a 3.3 GHz Ivy Bridge Xeon E3-1230 and a 2.1 GHz Sandy Bridge Xeon E5-2620 with
Intel C++ 14.0.1. That is our problem shape, our data type, our thread count, our baselines, and
our size range — for a spectral-interpolation application rather than a bare transform.

**The result:** the generated kernel "outperforms the FFTW and MKL based implementations by a
large margin (**typically a factor of 2 to 3 except for some large prime numbers**)"; the
abstract gives "speed-ups in isolation averaging **3×**".

**And the diagnostic, which is the reason this paper matters more than its speedup:**

> "for small sizes and numbers with small prime factors, FFTW and MKL are well-optimized and
> there our generated code is **slightly slower than MKL**"

> "while the 1D FFTs of FFTW and MKL occasionally outperform the 1D SPIRAL-generated FFTs, the
> SPIRAL-generated interpolation kernels outperform the handwritten interpolation powered by
> FFTW and MKL. **This shows the impact of the data layout transformations and the loop merging
> we performed to optimize the 3D upsampling.**"

Read that against the panel's three rounds. Every round has been spent on the 1D kernel and its
op count — the thing this paper says it *lost* on and won anyway. The wins came from:

* **Keeping the intermediate representation across stages instead of normalising it between
  them** — in their case zero-padding: "the data can stay zero-padded across interpolation
  stages at very little memory cost, avoiding unpadding and re-padding between x and y pencils
  and y and z pencils". Our analogue: keep the split/batch-interleaved granule form across all
  three axes and repack exactly once on the way in and once on the way out. Any entry that
  repacks between passes is paying this.
* **Using the vector terminal for the axes where it is available and never vectorising the
  transform itself**: "for the y and z pencils one can compute a **vector of DFTs**, `DFT_n⊗I_ν`,
  instead of having to vectorize the actual DFT. This allows stage 2 and 3 to be computed
  without vector overhead." This is §04 §3.1's theorem being cashed in *per axis*, which is
  subtly stronger than what the panel does.
* **Choosing the axis order for the layout, not for the algorithm**: "By picking the x dimension
  as the first dimension to work on we are minimizing the overhead of data copying and ensure
  vector memory access." Note that `L17_matrixsimd`'s **X-first reorder, worth 10.8 % at B=256
  in round 3, is exactly this**, arrived at independently. It is the single most valuable
  technique round 3 produced and it now has a citation and a mechanism.

Their achieved throughput is also a useful yardstick for us: "our code sustains between **0.6
and 1.2 vector flop/cycle (2.4 to 4.8 flop/cycle)**, and there is an overhead of about **0.2 to
0.5 vector shuffles/cycle**." Note the shuffle overhead they consider acceptable — the panel's
split layout has none, so on that axis the panel is already ahead of this paper.

The same result is reproduced as Fig. 20 of Franz Franchetti et al., *SPIRAL: Extreme Performance
Portability*, **Proceedings of the IEEE 106(11):1935–1968 (2018)** —
https://www.spiral.net/doc/papers/SPIRAL_IEEE_2018.pdf [VERIFIED — fetched], captioned "ONETEP
2 × 2 × 2 upsampling kernel with **small odd-sized 3-D batch FFTs** on 3.5-GHz Intel Haswell
4770K: Spiral versus FFTW and Intel MKL", with the text: "the original data cube is small (edge
length between 7 and 119), odd, and may be rectangular. These unusual requirements render
standard FFT libraries (Intel MKL and FFTW) suboptimal and allow SPIRAL-generated **end-to-end**
kernels to be three times faster." The same paper reports 512³–1024³ 3D FFTs on a Kaby Lake
7700K at "49–56 Gflop/s… 80 %–90 % of practical peak… whereas MKL and FFTW achieve at most
47 %."

**What this changes in the corpus.** §03 §7's "genuinely unmeasured territory" should now read:
*the specific cubes 6/8/17/36 are unmeasured, but the regime — small odd complex-double 3D
cubes, batched, single core, against MKL and FFTW — has been measured once, at 2–3×, and the
authors attribute the win to layout and loop merging rather than to the 1D kernel.* The panel's
own margins (1.14× at L=8, 1.37× at L=36, 1.68× at L=6, 4.99× at L=17) sit below that 2–3× at
three of four geometries, which is a reason to believe there is room and a strong hint about
where it is.

### 2.3 The one structurally different formulation with modern support: the 3D DFT as three batched GEMMs

Not vector-radix. **μ-mode products / Tucker operators / tensor-times-matrix.** Applying `DFT_L`
along each axis of an `L³` cube *is exactly* a chain of three μ-mode products, each a GEMM of
shape `(L×L) · (L×L²)`.

* Marco Caliari, Fabio Cassini, Franco Zivcovich, *A μ-mode BLAS approach for multidimensional
  tensor-structured problems*, arXiv:2112.11238 (2022) —
  https://ar5iv.labs.arxiv.org/html/2112.11238 [VERIFIED — fetched]: "The μ-mode product
  𝐓×_μ L might be performed by calling m₁⋯m_{μ−1}m_{μ+1}⋯m_d times level 2 BLAS. However… it is
  possible to perform the same task more efficiently by using a **single level 3 BLAS call**",
  at cost O(n^(d+1)). Measured (MATLAB R2019a, 2D, m=n, Core i7-8750H): nested loops
  1.8e-2 s / for-loop matrix products 7.8e-4 s / **BLAS 2.1e-5 s at n=50**; at n=400,
  80 s / 0.39 s / **1.2e-3 s**. An **860× spread at n=50 between three formulations of identical
  arithmetic** — consensus #4 of `LITERATURE.md` §2, in tensor clothing.
* Marco Caliari, Fabio Cassini, Lukas Einkemmer, Alexander Ostermann, Franco Zivcovich,
  *A μ-mode integrator for solving evolution equations in Kronecker form*, arXiv:2103.01691
  (2021; J. Comput. Phys.) — https://ar5iv.labs.arxiv.org/html/2103.01691 [VERIFIED — fetched].
  Implemented with **MKL `cblas_gemm_batch`** on CPU (dual Xeon Gold 5118) and
  `cublasGemmStridedBatched` on a V100. Abstract, verbatim: "**μ-mode products can be used to
  compute spectral transforms efficiently even if no d-dimensional fast transform is
  available.**" 3D grids n = 20…160, and the Kronecker route wins across all tested accuracies
  against a Time-Splitting Fourier Pseudospectral scheme using FFTW via MATLAB's `fftn`.
  **Caveat, and it is a real one: that comparison is at application level with a different basis
  (Hermite vs Fourier), not a transform-for-transform benchmark of the same DFT.**

**The shape arithmetic for our four sizes** *(mine; the suitability window is libxsmm's, quoted
in §3.3)*:

| L | per-axis GEMM `M×N×K` | `(M·N·K)^{1/3}` | inside libxsmm's ≤64 window? |
|---|---|---|---|
| 6 | 6 × 36 × 6 | 10.9 | yes — smallest bin, where MKL is weakest |
| 8 | 8 × 64 × 8 | 16.0 | yes |
| **17** | **17 × 289 × 17** | **43.7** | **yes — the largest and best-performing bin** |
| 36 | 36 × 1296 × 36 | 119 | **no** — would need blocking |

**Neither research pass found any published measurement of "small 3D DFT as three batched GEMMs
versus row–column FFT" on a CPU.** That is a genuine, paper-sized gap, and it is a cheap
experiment for this panel: the kernel is three straight-line constant-matrix products with
compile-time constants, which is what `L17_matrixsimd` and `baseline_matrix` already are in
part. Note carefully that this is *not* the harness's `baseline_matrix` (an O(L⁴) dense DFT per
axis with no blocking and no batch vectorisation, 22–66× off the pace) — it is the same
mathematics organised as a register-blocked, batch-vectorised, GEMM-shaped kernel. Per the
libxsmm evidence in §3.3, the difference between those two things is a factor of 20+.

**The honest counterweight**, from the tensor-contraction literature: Paul Springer, Paolo
Bientinesi, *Design of a High-Performance GEMM-like Tensor–Tensor Multiplication* (GETT/TCCG),
arXiv:1607.00145, ACM TOMS 2018 — https://arxiv.org/abs/1607.00145 [VERIFIED — fetched,
abstract and numbers] reports GETT outperforming alternatives "by up to 12.4×" and reaching "up
to 1.41× over an equivalent-sized GEMM for bandwidth-bound tensor contractions" and "up to
91.3 % of peak" when compute-bound — and finds that the **LoG (loops-over-GEMM)** formulation,
which is what per-axis DFT is, "experiences a significant performance loss" relative to
GETT/TTGT for general contractions. That is for large contractions, not L=17, but it says the
GEMM route is not automatically a win either.

### 2.4 Two free wins the plateau has been hiding

**(a) Make every stride a compile-time constant. Measured at up to 1.9×.** Stephen Fegan,
`dft_simd` — https://github.com/sfegan/dft_simd and
https://raw.githubusercontent.com/sfegan/dft_simd/master/README.md [VERIFIED — fetched]. This
project benchmarks FFTW's library API against FFTW-`genfft`-generated straight-line codelets
**vectorised across datasets** — i.e. the panel's exact architecture — and it separates the
effect of stride constancy. On an AMD EPYC 9474F (2025 numbers), 16.8 M datasets, N = 60, AVX2:
**FFTW aligned 858 ms; codelet 149 ms; codelet with fixed stride 79 ms.** At N = 1024:
FFTW 12 136 ms; codelet 9 124 ms; fixed-stride 7 270 ms. Earlier runs: Core i5-5287U N=60,
FFTW 1925 ms vs codelet 379 ms; Xeon E5-2650 v4 N=60, FFTW 2194 ms vs codelet 220 ms.

**Fixed stride is worth ≈1.9× at N=60 and ≈1.25× at N=1024 over the same codelet with a runtime
stride.** Our L is a compile-time constant and our batch stride is `L³` complex doubles — also
a compile-time constant. Every entry should verify in the generated assembly that no address
computation multiplies by a runtime value. This costs one `objdump` and, on this evidence, may
be worth double digits.

The same project also carries a caution for §4: in its N=1024 column, FFTW got **slower** under
AVX-512 than under AVX2 (24 657 ms vs 12 136 ms). Treat 512-bit as an empirical question per
size, which is precisely §4.4's rule.

**(b) Stop comparing to the wrong baseline, and check the reordering pass.** Maron Schlemon,
Jamin Naghmouchi, *FFT optimizations and performance assessment targeted towards satellite and
airborne radar processing* (DLR) — https://elib.dlr.de/140530/2/conference_101719.pdf
[VERIFIED — fetched]. A hand-tuned radix-4 DIF SIMD FFT measured with RDTSC, 10 000 measurements
per size: median cycles at N=1024 — hand-tuned **5508**, FFTW scalar **21 524**, FFTW SIMD
**3758**; at N=4096 — 25 442 / 90 437 / **19 428**. Verbatim: "up to a 4096-point FFT, the SIMD
version of FFTW is about **30 % more efficient**" than the hand-tuned kernel — and only after
removing the reordering stage does the hand-tuned version become "about 10 % faster than
SIMD-FFTW". Their reordering cost alone: **8099 cycles at N=4096 (a third of runtime) and
~139 419 at N=16384 (two thirds)**. Two lessons: 4× over a scalar library means nothing, and the
index-permutation pass can dominate — which, with Venkovic & Anzt's 48–73 % figure in §2.1, makes
**two independent 2020s sources saying the permutation, not the butterfly, is where 1D FFT time
goes.** Our PFA index permutations are compile-time and free (§02 §5.5) — but that is a claim
worth *verifying* in the assembly rather than assuming, at L=6 and L=36 where PFA is used.

### 2.5 So what does the plateau mean?

Reading the panel's own r3 numbers against this literature:

* At **B=1**, L=6 (1.04×), L=8 (1.05×) and L=17 (1.05×) are all within 5 % of their own FP-port
  floors. On a **one-FMA-unit part** (§4.1) that floor is real and the arithmetic is finished.
  **§4.1 says that floor was computed at the wrong clock and the real margin is ~30–43 %, so the
  arithmetic may not be finished after all.**
* At **large batch**, every geometry is within 1.16–1.83× of a bandwidth bound, and §1 says the
  levers are traffic (NT, §1.5), concurrency (prefetch and interleaved streams, §1.6b), tiling
  into L2 (§1.9), and possibly a systematic 4 KiB alias stall (§1.8). None of those is a
  butterfly.
* The one *structural* idea with modern support is the three-GEMM formulation (§2.3), and the one
  *scheduling* idea with modern support is Popovici et al.'s "never normalise the intermediate
  between axes, and pick the axis order for the layout" (§2.2).

**The plateau is not evidence that the problem is solved. It is evidence that the panel has
converged on the one part of the problem the literature says is not where the time goes.**

---

## 3. Target 3 — regime-dependent kernel choice, and what governs the crossover

The L=17 observation to explain: the dense conjugate-symmetric kernel wins at B=1, the Rader
kernel wins at B=256, and the ranking **inverts**. (Note that the r3 leaderboard shows
`L17_matrixsimd` sweeping all four cells and `L17_rader` third everywhere, having regressed
7.4 % at B=256 — so on the *current* code the inversion has closed. The r1 numbers are where it
was visible. Either way the mechanism question is live, because the same inversion governs which
structure to invest in.)

The batched-GEMM and small-matrix literature has exactly this problem and has diagnosed it.
Four mechanisms, each with a measurement, in my order of confidence for our case.

### 3.1 Mechanism 1 — at B=1 there is no batch to vectorise over, so vectorisability decides

This is McFarlin et al.'s mechanism for the direct-DFT crossover, and reading it carefully
changes the corpus's interpretation of it. Daniel S. McFarlin, Volodymyr Arbatov, Franz
Franchetti, Markus Püschel, *Automatic SIMD Vectorization of Fast Fourier Transforms for the
Larrabee and AVX Instruction Sets*, **ICS 2011** —
https://users.ece.cmu.edu/~franzf/papers/ics2011.pdf [VERIFIED — fetched], verbatim:

> "indeed up to a size of about **n = 20**, the direct computation is preferable, even though the
> mathematical operations count (counting only additions and multiplications) is inferior. The
> reason is in LRB's dedicated **replicate HW**, which enables efficient **scalar broadcasts and
> FMA instructions** which are well-suited for a direct computation."

Two corrections to how the corpus (§03 §6.4, §06 §6.4a) cites this:

1. **The n ≈ 20 figure is a Larrabee (LRBni) result, not an AVX result.** The paper's AVX numbers
   are a separate comparison against Intel IPP 7.0 on a 3.3 GHz Core i5-2500.
2. **The stated mechanism — dedicated replicate/broadcast hardware plus FMA — is exactly what
   AVX-512 embedded broadcast (`{1toN}`) plus FMA provide, and exactly what AVX2 lacks.** So the
   n≈20 dense-DFT argument transfers *better* to AVX-512 than to AVX2. That is a direct
   explanation for why `L17_matrixsimd` — a dense conjugate-symmetric matrix kernel — is the
   panel's largest margin over the state of the art anywhere on the board, and it says the
   AVX-512 path is the *right* home for that kernel (consistent with §4.4).

**The prediction this generates, and it is the cheapest experiment in this document.** At B=1
there is no batch, so Franchetti's `DFT_n ⊗ I_ν` vector terminal is unavailable and Rader's
permutation/gather structure is exposed; a dense 17×17 applied to 289 columns vectorises
trivially with FMA plus broadcast. At `B ≥ ν` the vector terminal becomes available to *both*
algorithms, the field levels, and Rader's 1.26× flop advantage and smaller live set take over.
**So the crossover should occur at B ≈ ν = 8 for zmm doubles (or 4 for ymm), not at 256.**

> **Scan B = 1, 2, 4, 8, 16, 32 at L=17 with both kernels.** If the crossover is at ~8, the cause
> is vectorisability and the fix is not to pick a kernel but to give the dense kernel a
> batch-blocked schedule. If the crossover is at ~256, the cause is capacity — §3.2 — and the fix
> is tiling. The panel's batch grid (1, 8, 256, 2048) cannot distinguish these, and adding
> B = 2, 4, 16, 32 costs one sweep.

### 3.2 Mechanism 2 — the small-batch regime is latency/ILP-bound; the large-batch regime is throughput-bound

Gianluca Frison, Dimitris Kouzoupis, Tommaso Sartor, Andrea Zanelli, Moritz Diehl, *BLASFEO:
basic linear algebra subroutines for embedded optimization*, **ACM TOMS 44(4), 2018**,
arXiv:1704.02457 — https://arxiv.org/pdf/1704.02457 [VERIFIED — fetched]. On one core of a Core
i7-4800MQ: "for matrices of size up to about one hundred the high-performance implementation of
BLASFEO is about **20-30 % faster than the corresponding level 3 BLAS routines and 2-3 times
faster than the corresponding LAPACK routines**." Mechanism, verbatim: the packing step's
"**quadratic cost can not be well amortized in the case of small matrices**"; "for small
matrices the cost of packing is not negligible with respect to the cost of performing level 3
BLAS operations"; and register blocking exists "to hide latency of instructions", with
accumulators "helping hiding the latency of FP operations."

Quantified on ARMv8 by Jianyu Yao et al., *IAAT: A Input-Aware Adaptive Tuning framework for
Small GEMM*, arXiv:2208.09822 — https://arxiv.org/pdf/2208.09822 [VERIFIED — fetched]: "**pack
step cost can reach 67 %** when input matrices are very [small]… when [matrices are large]
enough, the proportion is near **3 %**." A single overhead curve that runs 67 % → 3 % with size
is by itself enough to invert a ranking.

**Read across to us:** at B=1 the whole transform is one dependency-chain-limited straight-line
block and *latency hiding* decides; at large B it is a throughput problem and *traffic* decides.
Those are different objective functions, so different kernels win, and no amount of op-count
work will unify them. This is a first-principles argument for the r3 verdict's process lesson —
**ship both structures as tuner candidates** — and against the "one best kernel per geometry"
framing the panel has been using.

### 3.3 Mechanism 3 — the layout's interleave block size sets the crossover, and it is measurable

*The Design and Performance of Batched BLAS on Modern High-Performance Computing Systems*,
**ICCS 2017, Procedia Computer Science 108C:495–504** —
https://www.netlib.org/utk/people/JackDongarra/PAPERS/batched-blas-iccs17.pdf
[VERIFIED — fetched]. *(Authorship note: the fetched document's running head reads "Relton,
Valero, Zounon et al." and the text credits "Jack Dongarra et al."; the full author list per the
publisher record is Dongarra, Hammarling, Higham, Relton, Valero-Lara, Zounon, but only the
running head and the "Jack Dongarra et al." credit were verified from the document itself.)*
Platforms: Xeon E5-2650 v3 (icc 16.0.0, MKL 11.3), KNL 7250, Kepler K40c. Batch of 10 000
DGEMMs, sizes 2×2 … 20×20.

Why interleaving across the batch wins at small size, verbatim: "if the matrices are of size
2 × 2 but the length of the vector units is 8 double precision numbers, as in the self-hosted
Intel KNL, then the strided and P2P memory layouts will not fill the vector units. By
comparison, the interleaved memory layout can work on 8 matrices simultaneously to fill the
vector units in each clock cycle."

Why it *stops* winning, verbatim: "for large batch sizes, the machine needs to make large jumps
in memory to access the different elements of the matrices (leading to cache misses)… moving
from the first to the second element of a matrix in the batch requires jumping **batch_count**
memory locations, which can hinder the overall performance."

Hence their block-interleaved layout with a tunable block size k, and the measured crossover:
block-interleaved leads "**until matrices of size 16 × 16 are used; shortly after this point the
P2P approaches become faster**" (peak 115 Gflop/s on KNL); on the GPU the shared-memory
block-interleaved variant "outperforms cuBLAS for very small matrices **up to 13 × 13**." And a
number that governs whether any of this is usable: "excluding the cost of this conversion makes
our routines approximately **twice as fast**" — layout conversion is half the cost, so it must be
free.

**Read across to us, and this is the sharpest single mapping in this section.** The panel's
batch-vectorised split layout **is** a block-interleaved layout with block size k = the SIMD
width. This paper says (i) that is the right choice at small matrix size, (ii) **k is the knob
that governs the crossover**, and (iii) full interleaving across the whole batch fails because
of `batch_count`-sized strides. Combined with §1.10's cache-line argument, the recommendation is
the same from two directions: **k = 8 for split complex double, tiled above that** — not k = the
register width, and definitely not k = B.

### 3.4 Mechanism 4 — register residency, with a measured inversion point

Sameer Deshmukh, Rio Yokota, George Bosilca, *Cache Optimization and Performance Modeling of
Batched, Small, and Rectangular Matrix Multiplication on Intel, AMD, and Fujitsu Processors*,
arXiv:2311.07602 (2023) — https://arxiv.org/html/2311.07602v1 [VERIFIED — fetched]. Machines
include a **Xeon Gold 6148 (AVX-512)**; batch size 20 000; ">2× the throughput of vendor
optimized libraries." The mechanism and the inversion, verbatim:

> "All temporary blocks used in this new algorithm are tiny matrices (size rank×rank) and can
> **fit within the vector registers** when rank is sufficiently small… For the case where
> rank=SVEC, an entire matrix GXY of dimension SVEC×SVEC can be computed **within the registers
> without having to perform expensive reads and writes**"

> "The bandwidth utilization is lesser than for rank 16 for the same block sizes since packing
> smaller skinny matrices individually into the L1 cache does not lead to optimal bandwidth
> utilization"

> "**As the rank increases beyond 96, we can observe SSL showing better performance… This can be
> attributed to the fact that the algorithm becomes more compute bound.**"

**Read across to us:** the dense conjugate-symmetric L=17 kernel needs 34 live vectors against
32 zmm (§02 §2.5, §04 §9) and must spill. At B=1 those spills land in a warm, otherwise-idle L1;
at B=256 (≈38 MiB working set) L1 is simultaneously streaming, so spill traffic competes with
the stream. **Test:** `perf stat` the L1D replacement and `mem_inst_retired` counters per volume
at B=1 and B=256; if spill *traffic* per volume is flat while *stalls* grow, this is the
mechanism, and the fix is output-blocking the dense kernel to fit 32 registers rather than
switching algorithms.

### 3.5 The selection rule the literature actually recommends: compute a utilisation fraction

The cleanest statement of a mechanistic switching rule is a GPU paper, but the *form* of the
argument is portable and is what the panel's tuners should be doing instead of racing candidates
blind. Ahmad Abdelfattah, Stanimire Tomov, Jack Dongarra, *Matrix multiplication on batches of
small matrices in half and half-complex precisions*, **J. Parallel Distrib. Comput. 145 (2020)
188–201** — https://icl.utk.edu/files/publications/2020/icl-utk-1411-2020.pdf
[VERIFIED — fetched]:

> "The generic MAGMA kernel is the best performing kernel for sizes ≈ 10 and up. For sizes
> smaller than 10, the magma-small kernel is the best solution, though no Tensor Cores are used…
> **The threshold of 25 % shown in Fig. 17 is the ratio between the full FP16 compute power in
> both situations. A utilization below this threshold may give the advantage to a kernel that
> does not use the TCs.**"

The portable version for us: **compute what fraction of a vector register a given kernel
actually fills, and compare it against the hardware's width ratio.** A length-17 line fills
17/8 = 2.125 zmm registers, i.e. the last register is 12.5 % utilised; a length-6 line fills
0.75 of one. Batch-vectorising makes utilisation exactly 1 by construction — which is the whole
argument for §04's layout and the reason the panel's kernels beat the libraries. Where a kernel
*cannot* be batch-vectorised (B=1), this ratio is the quantity that decides between structures,
and it is computable at plan time rather than raced.

And the same "predict, then measure" discipline applied to a fused-versus-unfused choice on a
CPU: Rati Gelashvili, Nir Shavit, Aleksandar Zlateski, *L3 Fusion* (verified in §1.9): "We
analyze our L3–cache aware algorithm using the roofline model **in order to predict in which
scenarios it is expected to outperform** currently available implementations… the fused
implementation is sometimes slower and sometimes faster than the state of the art; while
consistently being faster by **50 % on average on layers with lower number channels**", and the
governing law: "**We show that the L3 cache pressure depends on the number of channels, thus, as
the number of channels increases the L3-fused approach becomes inferior to the standard
non-fused one.**" Substitute *batch count* for *channels* and that is our L=8 and L=36 story
exactly — including the prediction that a fused/tiled scheme has a batch range in which it wins
and a batch range in which it loses, and that the boundary is set by cache pressure, not by the
kernel.

### 3.6 libxsmm: the documented policy, and a published ranking inversion

The panel is not allowed to call libxsmm, but its published policy is the best available prior
for "when does a specialised small kernel beat a library", and its numbers quantify the prize.

**Documented policy** — https://libxsmm.readthedocs.io/en/latest/ and
https://github.com/libxsmm/libxsmm/wiki/Q&A [VERIFIED — fetched], verbatim:

> "When characterizing the problem-size by using the M, N, and K parameters, a problem-size
> suitable for LIBXSMM falls approximately within **(M N K)^(1/3) <= 64**"
> "For auto-dispatched problem-sizes above the configurable threshold… LIBXSMM is falling back
> to BLAS."
> "Raising the threshold may not only generate excessive amounts of code (due to unrolling in M
> or K dimension), but also **miss to implement a tiling scheme to effectively utilize the cache
> hierarchy**."

That last clause is the whole of §1.9 in one sentence, from the other side: **unrolled
straight-line specialisation and cache tiling are complementary, and a library that pushes
unrolling past the point where tiling is needed gets slower.** The panel's L=36 entries are
exactly at that boundary.

**Measured, against MKL** — Alexander Heinecke, Hans Pabst, Greg Henry, *LIBXSMM: A High
Performance Library for Small Matrix Multiplications*, **SC15 poster** —
https://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file2.pdf
[VERIFIED — fetched]. Dual-socket Xeon E5-2699v3 (Haswell), 36 cores, 118 GB/s STREAM Triad,
1.1 TFLOPS with large DGEMMs; CP2K/DBCSR, 386 specialisations **binned by arithmetic intensity**
— note that they chose AI, not size, as the axis:

| bin | AI (DP-FLOPS/byte) | libxsmm asm | auto-JIT | **MKL** | inlined C |
|---|---|---|---|---|---|
| MNK ≤ 13³ | 0.4–1.1 | 142 | 137 | **95** | 131 |
| 13³ < MNK ≤ 23³ | 1.1–1.9 | 260 | 249 | **156** | 215 |
| 23³ < MNK | 1.9–4.5 | **427** | 411 | **215** | 285 |

libxsmm-over-MKL is **1.49× / 1.67× / 1.99×** as arithmetic intensity rises. And their SeisSol
panel is the cleanest published picture of a **ranking inversion by size** — % of peak for
libxsmm-asm / MKL / compiler-inlined C at convergence orders 2…7 (M=K = 4, 12, 20, 36, 56, 84;
N=9):

| order | M=K | libxsmm asm | MKL | inlined C |
|---|---|---|---|---|
| 2 | 4 | 39.5 % | **2.3 %** | 12.2 % |
| 4 | 36 | 71.7 % | 34.5 % | 59.5 % |
| 5 | 56 | 78.0 % | 50.7 % | 69.7 % |
| 7 | 84 | 77.6 % | 58.9 % | **17.1 %** |

At M=K=4 MKL delivers **2.3 % of peak** and plain inlined C beats it by 5×; by order 7 MKL has
recovered to 58.9 % while inlined C has collapsed to 17.1 %. **Two implementations of the same
mathematics swap places by a factor of 25 across a size sweep.** That is the phenomenon the
panel is looking at, in a different problem, with the mechanism named: specialised straight-line
code wins while the working set is register-resident and loses once tiling is required.

*(Note: `LIBXSMM_MAX_MNK`'s default numeric value is not printed in the documentation either
pass could fetch, so it is not quoted here. The SC15 poster states the suitability criterion as
`(M·N·K)^(1/3) < 80`; the current docs say ≤ 64.)*

### 3.7 Roofline used to *choose* an algorithm — a worked CPU example where more flops wins

`LITERATURE.md` §2 consensus #3 says instructions, not flops, are the currency. This is the
measured CPU example of *how to decide* when two algorithms differ in both flops and traffic.

Aleksandar Zlateski, Zhen Jia, Kai Li, Fredo Durand, *A Deeper Look at FFT and Winograd
Convolutions*, **SysML/MLSys 2018** — https://mlsys.org/Conferences/doc/2018/28.pdf, and the
extended *FFT Convolutions are Faster than Winograd on Modern CPUs, Here's Why*, arXiv:1809.07851
— https://ar5iv.labs.arxiv.org/html/1809.07851 [both VERIFIED — fetched]. The model, verbatim:

> "T = FPO / min(Peak FLOPS, AI × MB)"
> "The transform stages (①, ②, ④) have very low arithmetic intensity – lower than 2.5 for
> Winograd and lower than 3.5 for FFT, and will thus be **memory bound on all modern systems**.
> In this case Eqn. 1 can be reduced to T = DM/MB"
> "the FFT-based approach will typically have the **AI twice as large** as its Winograd
> counterpart, due to working in the complex domain"
> "In some cases, on machines with large compute-to-memory ratios, the Winograd method will be
> memory bound, while the FFT method is compute bound."

Compute-to-memory ratios, verbatim: "the 4.5 TFLOPS Intel Knights Landing processor has a
compute-to-memory ratio of 11, and **the latest Skylake Xeon processor family has ratios in the
range of 20 to 30**." Result: **FFT beats Winograd by 1.84× averaged over AlexNet layers**
despite doing more flops, because its AI is 2× higher; model fidelity "92.68 % fitness (0.079
relative RMSE)"; compute-bound stages hit "~75 % of theoretical peak FLOPS" and memory-bound
stages ">85 % of theoretical bandwidth".

**A detail worth stealing:** their best FFT tile sizes are **not powers of two** — 27, 25, 31,
15 — chosen to minimise overlap-add padding, and their transforms come from a modified FFTW
`genfft` plus JIT'd matmul for AVX2/AVX-512. Non-power-of-two straight-line codelets in a
production-quality CPU pipeline; the panel is in good company.

**How to use this at L=17 and L=36.** Compute AI for each candidate kernel — total FP
instructions ÷ bytes moved between the two levels that bind — and compare it against the
machine's ratio at that level. Using §1.2's single-core numbers and the arithmetic in §7 item 2,
this node's machine balance is **4.03 flops/byte at the DRAM level, 2.55 at L3 and 0.53 at L2**,
against geometry arithmetic intensities of 1.21 (L=6), 1.41 (L=8), 1.92 (L=17) and 2.42 (L=36). **A kernel whose AI exceeds the balance at
the binding level is compute-bound and should be optimised for instructions; below it, for
traffic. At L=17 batched the two candidates differ in *both*, which is exactly the situation
this model exists to resolve — and nobody on the panel has computed it.**

---

## 4. Target 4 — AVX-512 on Cascade Lake, in practice

This target had no measurement anywhere in the corpus. It now has Intel's own turbo table for
**our exact SKU**, and the answer is the opposite of what the corpus and `PANEL_BRIEF.md` assume.

### 4.1 The headline: on a Xeon Gold 5218 with one active core, AVX-512 costs ZERO frequency relative to AVX2

**Source.** Intel Corporation, *2nd Gen Intel® Xeon® Scalable Processors Specification Update*,
document **338848-028US**, October 2023, Figures 1–6 —
https://cdrdv2-public.intel.com/338848/338848_2nd%20Gen%20Intel%C2%AE%20Xeon%C2%AE%20Scalable%20Processors%20Specification%20Update_Rev028US.pdf
[VERIFIED — fetched, full text extracted]. The Gold 5218 (16 cores, 22 MB LLC, 125 W) appears in
two independent figure pairs with identical values. Frequency in GHz by number of active cores:

| class | base | 1 | 2 | 3 | 4 | 5–8 | 9–12 | 13–16 |
|---|---|---|---|---|---|---|---|---|
| **Non-AVX** (License 0) | 2.3 | **3.9** | 3.9 | 3.7 | 3.7 | 3.6 | 3.1 | 2.8 |
| **AVX 2.0** (License 1) | 1.8 | **2.9** | 2.9 | 2.7 | 2.7 | 2.6 | 2.5 | 2.3 |
| **AVX-512** (License 2) | 1.5 | **2.9** | 2.9 | 2.7 | 2.7 | 2.6 | 2.3 | 2.1 |

**Licence 1 and Licence 2 are identical from 1 through 8 active cores on this SKU.** They diverge
only at 9+ cores. The panel runs **one thread on an exclusive node**, i.e. exactly the 1-core
column, where 256-bit-heavy and 512-bit-heavy both cap at **2.9 GHz**.

**Three things this overturns.**

1. **`PANEL_BRIEF.md`'s warning and `LITERATURE.md` §4.8 gap 6 are, for our workload, wrong.**
   Both cite §04 §8.2's Gold 5120 figures — "2.7 GHz scalar → 2.3 AVX2 → **1.6 AVX-512** with
   several cores active" — as a reason a 512-bit kernel might lose to a 256-bit one on this node.
   Those are **9+-core** numbers. On a Gold 5218 with one core active there is **no AVX-512
   frequency penalty relative to AVX2 at all.** Every implementer who declined to write a
   512-bit path on downclocking grounds declined for a reason that does not apply. This is the
   single largest correction in this section, and possibly in this document.
2. **The real frequency cliff is at 128 bits, not at 512.** Intel's licence definition
   (*Optimization Reference Manual* Vol. 1, doc **248966-049US**, §2.5.3, Table 2-10 —
   https://cdrdv2-public.intel.com/814198/248966-Optimization-Reference-Manual-V1-049.pdf
   [VERIFIED — fetched]) assigns License 0 to "Scalar, AVX128, SSE, Intel AVX2 w/o FP or INT
   MUL/FMA"; License 1 to "AVX2 FP + INT MUL/FMA, AVX-512 without FP or INT MUL/FMA"; License 2
   to "Intel AVX-512 FP + INT MUL/FMA". **So *any* 256-bit floating-point arithmetic — including
   `vaddpd ymm`, not just FMA — is Licence 1.** You cannot get 3.9 GHz out of a 256-bit butterfly.
   The only way to reach License 0 is 128-bit/SSE/scalar FP.
3. **The panel's FP-port floors have been computed at the wrong clock, and there is more headroom
   at B=1 than anyone thinks.** The r3 verdict computes each winner's floor at **2.30 GHz** (the
   non-AVX base) and finds entries 1.04–1.13× above it, concluding the arithmetic is finished. At
   the correct **2.9 GHz** 1-core Licence-2 frequency those floors drop by 1.26×:

| geometry | winner | vector FP ops/volume | floor @2.30 GHz (verdict) | **floor @2.90 GHz** | measured B=1 | **ratio at 2.9 GHz** |
|---|---|---|---|---|---|---|
| L=6 | L6_unrolled | 972 (2×256-bit FMA ports) | 0.211 µs | **0.168 µs** | 0.220 | **1.31×** |
| L=8 | L8_batchsimd | 1248 (1×512-bit FMA unit) | 0.543 µs | **0.431 µs** | 0.570 | **1.32×** |
| L=17 | L17_matrixsimd | 35 964 | 15.6 µs | **12.4 µs** | 16.386 | **1.32×** |
| L=36 | L36_mixedradix | 241 056 | 105 µs | **83.1 µs** | 118.6 | **1.43×** |

*(My arithmetic, from the verdict's own op counts and Intel's turbo table.)* **Every geometry is
~30–43 % above its port floor at B=1, not 4–13 %.** The verdict itself flagged this possibility —
"if the node turbos above 2.30 GHz under AVX-512 the true floors are lower and these margins are
upper bounds — which is exactly the unresolved measurement in §6" — and Intel's table says it
does. **Three rounds of "the arithmetic is finished at B=1" rested on a base-clock assumption
that the vendor's own document contradicts.**

**Verify before betting the round on it.** Two caveats, both real:

* The 5218's 2.9 GHz AVX turbo is **anomalously conservative next to its siblings** in the same
  Intel figures — the Gold 5220 (18 cores, same 125 W) is listed at AVX2 1-core **3.8 GHz** and
  AVX-512 1-core **3.7 GHz**, a mere 100 MHz apart and a full 900 MHz above the 5218. The value
  appears twice in the document, so it is not a transcription error, but Intel adds "All details
  previously shown are subject to change without notice."
* The document is a spec, not a measurement, and the node runs the `powersave` governor.

> **The measurement to take, and it should be the first thing the monitor does next round:**
> `perf stat -e cycles,ref-cycles` on an L=6 B=1 and an L=8 B=1 run gives the actual average
> frequency directly (`cycles/ref-cycles × 2.30 GHz`). Four entries across two geometries have
> now asked for this in writing, and §4.1 says the answer moves every B=1 conclusion on the
> board by up to 26 %. Travis Downs' `avx-turbo` — https://github.com/travisdowns/avx-turbo
> [VERIFIED — fetched] — "measures non-AVX, AVX2 and AVX-512 speeds for various types of CPU
> intensive loops" across active-core counts using APERF/MPERF, and would settle it completely.
> Also read `CORE_POWER.LVL0_TURBO_LICENSE`, `LVL1_TURBO_LICENSE`, `LVL2_TURBO_LICENSE` and
> `CORE_POWER.THROTTLE` (Intel names the first three in §2.5.3) to confirm which licence the
> kernel actually landed in — which, per §4.3, is not always the one the register width implies.

### 4.2 The Gold 5218 has ONE 512-bit FMA unit — so width buys instructions, registers and L1 bandwidth, not FLOPs

**Sources.** Microway, *Detailed Specifications of the "Cascade Lake SP" Intel Xeon Processor
Scalable Family CPUs*, 2 April 2019 —
https://www.microway.com/knowledge-center-articles/detailed-specifications-of-the-cascade-lake-sp-intel-xeon-processor-scalable-family-cpus/
[VERIFIED — fetched]: "the **6200-series and 8200-series** CPUs provide **two** AVX-512 units per
processor core. All remaining CPUs (with the exception of the **Xeon 5222**) include a **single**
AVX-512 unit." Independently, Jeff Hammond, `vpu-count` —
https://github.com/jeffhammond/vpu-count [VERIFIED — fetched] — lists **"Intel® Xeon® Gold 5218
Processor" — 1 FMA unit** (and provides a runtime `vpu_count()` detector usable from
`fft3d_create()`, cost "as low as 2.2 microseconds").

**What one unit means, from Agner Fog**, *The microarchitecture of Intel, AMD and VIA CPUs*,
ch. 11 (Skylake/Skylake-X/Cascade Lake) — https://www.agner.org/optimize/microarchitecture.pdf
[VERIFIED — fetched], verbatim:

> "There are two 256-bit vector execution units at port 0 and 1… These two units can be combined
> into one 512-bit unit when 512-bit vector instructions are executed. This combined 512-bit unit
> is accessed through port 0… There is an additional 512-bit vector unit under port 5… **Permute
> instructions and other instructions that may cross the 128-bit lane boundaries are always
> handled by port 5.** … Early versions of Skylake-X with less than ten cores do not have the
> extra 512-bit floating point unit at port 5. **These versions have a throughput of one 512-bit
> floating point vector operation or two 256-bit floating point vector operations per clock
> cycle.**"

Intel's manual corroborates the mechanism: "an additional Intel AVX-512 FMA unit on port 5
**which is available on some parts**", and "Since port 0 and port 1 are 256-bits wide, Intel
AVX-512 operations that will be dispatched to port 0 will execute on both port 0 and port 1".
On load/store width Agner adds: "There are two identical memory read ports (port 2 and 3) and one
write port (port 4). **These ports all have the full width of 256 or 512 bits.**"

**The per-cycle table for our node** *(derived from the two sources above plus §4.1's turbo
table; the r3 verdict independently assumed the 1-FMA figure for its L=8 floor and landed within
5 % of measurement, which is empirical support)*:

| width | DP FMA lanes/cycle | licence | 1-core clock | effective DP lanes/s |
|---|---|---|---|---|
| 128-bit (SSE/AVX128) | 2 ports × 2 = **4** | L0 | **3.9 GHz** | 15.6 G |
| 256-bit (AVX2) | 2 ports × 4 = **8** | L1 | 2.9 GHz | 23.2 G |
| **512-bit (AVX-512)** | 1 port × 8 = **8** | L2 | **2.9 GHz** | **23.2 G** |

**Read the last two rows together with §4.1: 512-bit and 256-bit have identical peak FP
throughput *and* identical frequency on this part.** So the choice between them is decided
entirely by the secondary effects, and every one of those favours 512-bit:

1. **Half the instructions**, against Skylake's rename/retire limit — Agner: "the throughput of
   the whole design rarely exceeds four instructions per clock". Skylake Server also has its
   **loop stream detector disabled** (Intel §2.5), so the front end matters more here than on a
   client part.
2. **32 zmm registers instead of 16 ymm.** Decisive at L=8 (2L = 16 = the entire ymm file, with
   §01's measured liveness of 19), and the only way to keep L=17's 34-vector working set anywhere
   near the register file.
3. **2× L1 and 1.7× L2 load bandwidth** (§1.2: 186 vs 93 GB/s in L1, 87.3 vs 50.6 in L2;
   Alappat et al. independently measure L1 load bandwidth halving each time the SIMD width halves,
   reaching "88 % of the theoretical L1 bandwidth limit (128 byte/cy, 204.8 Gbyte/s @ 1.6 GHz)").
4. **Embedded broadcast of a memory operand**, which is the natural encoding for our compile-time
   twiddles — and on this microarchitecture it is *not* on the shuffle port. Intel §18.9.1–18.9.2,
   verbatim: "A source from memory can be broadcast… **up to 8 times for a 64-bit data element**,
   without using an additional source register"; "Using embedded broadcast can **reduce the number
   of registers** used in the code"; "the load micro-op is in the same instruction as the operation
   micro-op, and therefore can **benefit from micro fusion**"; and critically — "In Skylake Server
   microarchitecture, a broadcast instruction with a memory operand of 32 bits or above is
   **executed on the load ports; it is not executed on port 5** as other shuffles are."
   **So `vfmadd231pd zmm_a, zmm_b, [rip+tw]{1to8}` costs a load-port slot, no register, and no
   port-5 slot.** That is the single best encoding available for a batch-vectorised codelet with
   compile-time constants, it exists only in AVX-512, and §01 §8 / `LITERATURE.md` §3.5's advice
   to keep constants as broadcast-from-memory FMA operands becomes literally free.
5. **Port 5 is idle on this SKU.** Because the 5218 has no second 512-bit FMA unit, port 5 carries
   only shuffles/permutes. Agner: vector permute is port 5, latency 1, "**3 if lane crossed**".
   A `vpermt2pd` is 1/cycle regardless of width, so a 512-bit permute does 2× the work of a
   256-bit one per port-5 slot. Our split layout deliberately uses none — but on *this* part, a
   kernel that needs a few lane-crossing ops can have them nearly free, which is not true on
   wallaby's 2-FMA part where port 5 is contended. **This is a case where the scoring node is
   more permissive than the development machine, the only one in this document.**

**Where 128-bit becomes a live contender, and it is narrow.** On this SKU 128-bit code runs at
License 0, i.e. **3.9 GHz — a 34 % clock advantage** — at half the FP lanes, so it is 1.49×
behind on peak FP but 1.34× *ahead* on everything that is not FP: loop counters, address
arithmetic, branch throughput, integer index work. For a kernel that is neither FP-throughput-
bound nor bandwidth-bound but *issue*- or *latency*-bound, that is a real trade nobody has tested.
It is unlikely to win at B=1 (§4.1 shows the leaders are within 1.3× of an FP-port floor, so FP
throughput *is* the binding constraint) and it cannot help a DRAM-bound cell (§1.2: bandwidth is
frequency-independent). **Recommendation: one afternoon, one geometry, as a curiosity — not a
round's plan.** [This is a derived argument, not a measurement; no source I fetched tests
128-bit FP against 512-bit on a 1-FMA Cascade Lake part.]

### 4.3 Licence transitions, warm-up, and what they do to a benchmark harness

This subsection is aimed as much at the monitor's methodology as at the kernels. All numbers are
verified.

**Intel's own transition mechanics** (*Optimization Reference Manual* §2.5.3, verified above),
verbatim:

> "When the core requests a higher license level than its current one, it takes the PCU up to
> **500 micro-seconds** to grant the new license. Until then the core operates at a lower peak
> capability… **Cores that execute at other license levels are not affected.**"
> "A timer of approximately **2ms** is applied before going back to a higher frequency level. Any
> condition that would have requested a new license resets the timer."
> "A license transition request may occur when executing instructions **on a mis-speculated
> path**."
> "A large enough mix of Intel AVX-512 light instructions and Intel AVX2 heavy instructions drives
> the core to request License 2, despite the fact that they usually map to License 1… The
> **Intel Xeon Platinum 8180** processor moves from license 1 to license 2 when executing a mix of
> **110 Intel AVX-512 light instructions and 20 256-bit heavy instructions over a window of 65
> cycles**."

**Measured transition costs.** Travis Downs, *Gathering Intel on Intel AVX-512 Transitions*,
17 January 2020 — https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html
[VERIFIED — fetched], on an **Intel Xeon W-2104** (SKX, 4 cores; licence frequencies L0 3.2 /
L1 2.8 / L2 2.4 GHz) and an i7-6700HQ: voltage-only transition **"~8 to 20 µs"**; the halted
period during a frequency transition **"~11 µs"** and "~10 µs"; during voltage transitions the
core executes at **"1/4th the usual instruction dispatch rate"** and "throttling is fine-grained:
it only applies when wide instructions are *in flight*"; decay — "After a period of about
**680 µs** not using the AVX upper bits… the processor enters a mode where using those bits again
requires at least a voltage transition", and "the lower 2.8 GHz frequency period lasts for
**~650 µs**". A **single** wide instruction is enough to trigger it.

**The warm-up penalty, which is the one most likely to be corrupting panel measurements.** Agner
Fog, ch. 11 (Skylake/Cascade Lake), verbatim:

> "Instructions with 256-bit vectors have a throughput that is approximately **4.5 times slower
> than normal** during an initial warm-up period of approximately **56,000 clock cycles or 14 µs**…
> The processor returns to the mode of slow 256-bit execution **2.7 million clock cycles, or
> 675 µs**, after the last 256-bit instruction… **Similar times apply to 512-bit vectors.**"
> "The first YMM or ZMM instruction takes **150 - 250 clock cycles** — probably to start a
> power-up process."
> "The processor simply **turns off power** for the upper parts of the 256-bit and 512-bit units
> when they are not used, rather than just gating the clock. It is using the 128-bit lane twice
> when executing a 256-bit instruction during the warm-up period."
> "Any instruction with YMM or ZMM registers will start the warmup process, **except `vzeroupper`
> and `vzeroall`**."

**Concrete consequences for `driver.c` and for every entry's plan-time tuner:**

* **A 14 µs warm-up at 4.5× slowdown is 63 µs of lost time.** At L=6 B=1 one `execute` is
  0.220 µs; a timed sample must therefore contain thousands of iterations before the warm-up is
  amortised — which the driver's auto-calibration probably achieves, but **it has never been
  checked against this number.** The monitor should confirm that the calibrated inner repeat
  count puts every timed sample well above ~700 µs, and that warm-up iterations are discarded
  *after* the first wide instruction, not before.
* **A plan-time tuner that races 12 candidates back-to-back, some 256-bit and some 512-bit, is
  partly measuring licence transitions.** The r3 verdict reports tuner instability costing
  3.9–6.7 % and picks flipping between runs at L8_radix8 B=64 and L36_mixedradix B=1/32/256.
  Downs' ~11 µs halt plus Intel's 500 µs grant latency plus the 2 ms hysteresis is a fully
  sufficient mechanism for exactly that. **Fix: give every candidate its own warm-up of at least
  56 000 cycles of its own width, run each candidate for > 700 µs, and never interleave widths
  within a tournament.** This is a concrete, sourced explanation for a measured pathology the
  panel has been treating as noise.
* **Never mix widths inside one `execute()`** — a 512-bit main loop with a 256-bit tail, or a
  512-bit pass followed by a 256-bit pass, risks a transition per call. Pick one width per kernel.
* **The "dirty upper" hazard.** Mathias Gottschlag, Peter Brantsch, Frank Bellosa, *Automatic Core
  Specialization for AVX-512 Applications*, **SYSTOR '20** —
  https://os.itec.kit.edu/downloads/Automatic_Core_Specialiszation_for_AVX_512_Application.pdf
  [VERIFIED — fetched]: "executing **any SIMD or floating-point instruction** causes some frequency
  reduction if the upper 256-bit half of the 512-bit registers contains valid contents", and
  "simply having valid 512-bit register state is enough to make many non-AVX-512 instructions
  trigger frequency changes"; the standard mitigation is "**`vzeroupper`** … directly after each
  AVX-512 phase". *For us the implication is inverted:* we want to stay in the high licence for
  the whole timed region, so **do not** emit `vzeroupper` between passes inside `execute()` — but
  a 512-bit `create()` followed by a 256-bit `execute()` will suffer.
* **Out-of-order throttling happens even at low clocks.** Robert Schöne, Thomas Ilsche, Mario
  Bielert, Andreas Gocht, Daniel Hackenberg, *Energy Efficiency Features of the Intel Skylake-SP
  Processor and Their Impact on Performance*, HPCS 2019, arXiv:1905.12468 —
  https://arxiv.org/pdf/1905.12468 [VERIFIED — fetched], on 2× Xeon Gold 6154, measured with
  `CORE_POWER.THROTTLE` and `CORE_POWER.LVL2_TURBO_LICENSE`: with a 2 s High/Low duty cycle,
  throttled out-of-order cycles per thread "ranging between **28 and 34 million**… **62 µs and
  75 µs** for each of the 150 iterations"; in the worst case (200 µs high / 800 µs low) "**more
  than 30 % of the cycles are spent throttled**" and "**more than 85 % of the cycles are spent at
  the reduced AVX-512 frequency range even though no AVX instruction is executed**"; and, decisively,
  "to avoid out-of-order throttling, we reduced the frequency of the system to 1.2 GHz, which is
  lower than any documented AVX frequency. **Even in this situation, the instruction issue is
  throttled** according to the PMC measurements." They also measure Skylake-SP P-state changes on
  a **500 µs** update interval, and a data-dependent power effect on 512-bit registers — "power
  consumption ranges from **362 W to 420 W** for a 3 GHz setup" depending only on bit patterns.

  Note the factor-3 disagreement between Intel's documented 2 ms hysteresis and the ~650–700 µs
  measured independently by Downs, Agner and Schöne et al. Gottschlag et al. remark that Intel's
  documentation "lacks detail and often deviates from the actual behavior observed during
  experiments on the hardware." Use the measured number.
* **Are we in Licence 2 at all?** Yes, and there is a threshold for it. Daniel Lemire, summarising
  Downs — *AVX-512 throttling: heavy instructions are maybe not so dangerous*, 25 August 2018 —
  https://lemire.me/blog/2018/08/25/avx-512-throttling-heavy-instructions-are-maybe-not-so-dangerous/
  [VERIFIED — fetched]: "Even a stream of 1 FMAD every 4 or even 2 cycles doesn't set the frequency
  down lower. **The lowest speed is only reached if FMAs come at a rate of more than 1 every 2
  cycles.**" Our codelets issue ~1 FMA per cycle (§4.1's ratio table). **We are unambiguously in
  Licence 2, and the good news from §4.1 is that on this SKU at one core that costs nothing.**
* **Hyperthreading.** Gottschlag et al.: "If one hyperthread in an SMT system executes AVX-512 or
  AVX2 code, the core has to reduce its frequency. **Other hyperthreads of the same core are
  equally affected.**" The node is exclusive, so this is only a warning against ever co-scheduling
  a second thread on the same physical core during a timed run.

### 4.4 The decision rule for this node

Revised from §4.1–§4.2. The corpus's implicit rule was "AVX-512 might be a catastrophe, measure
it". The evidence says something much more specific:

| your kernel is bound by | on the **Gold 5218** (1 FMA unit, 1 core) prefer | why |
|---|---|---|
| FP port issue (FMA-bound, register-resident) | **512-bit** | same flops/cycle *and* same clock as 256-bit, half the instructions |
| register pressure / spills | **512-bit** | 32 zmm vs 16 ymm, and embedded broadcast frees more |
| L1 or L2 load/store bandwidth | **512-bit** | 2.0× L1, 1.7× L2 (§1.2) |
| lane-crossing shuffles | **512-bit** | port 5 is uncontended on this SKU (no 2nd FMA there) |
| L3 or DRAM bandwidth | **either** | width buys nothing off-chip (§1.2); pick on instruction count |
| pure issue/latency, no FP throughput limit | 128-bit is *worth one test* | License 0 = 3.9 GHz, +34 % clock (§4.2) |

**Per geometry, on this node:**

* **L=6** — 512-bit. 2L = 12 of 32 zmm with room for temporaries (§01's measured liveness of 17
  fits comfortably in 32 but not in 16), 8 volumes per granule (§1.10), embedded-broadcast
  constants, and **no frequency penalty**. `L6_pfa`'s round-1 decision to write no 512-bit path
  ("answered analytically") was reasoned from the downclocking premise §4.1 refutes, and should be
  revisited — it is one of the two entries whose geometry has been flat for three rounds.
* **L=8** — 512-bit, unambiguously. 2L = 16 fits 32 zmm with 16 spare where it fills 16 ymm
  exactly; §01's measured liveness is 19, i.e. **guaranteed spills on AVX2 and none on AVX-512**.
  §07 §4.3's "this is a real reason to prefer the AVX-512 nodes for L=8" is right and the
  frequency objection is now removed.
* **L=17** — 512-bit, for the register file, the 1.7× L2 bandwidth on an L2-resident working set,
  and McFarlin's broadcast+FMA mechanism (§3.1), which is what makes the dense conjugate-symmetric
  kernel competitive in the first place.
* **L=36** — 512-bit for the register-resident tile arithmetic; width is irrelevant for the
  streaming passes (§1.2), so there is no reason to mix widths and pay §4.3's transitions. Use
  512-bit throughout.

**In short: on this SKU, at one active core, there is no case for a 256-bit kernel.** That is
the opposite of the brief's working assumption, and it is the highest-value single correction this
section makes.

### 4.5 AVX-512 features beyond width, and the one real gotcha

* **Mask registers for the batch tail.** Agner Fog, ch. 11, verbatim: "In general, there is no
  extra cost to masking. The latencies and throughputs of masked instructions is the same as
  without masking, except for memory read and write instructions and register-to-register moves…
  Masked memory read instructions have an **extra µop at port 0 or 5 and an extra latency of one
  clock cycle**. Masked memory write instructions have an **extra latency of approximately 10
  clock cycles**… it is recommended to **use the zeroing option** on masked instructions if it is
  acceptable that the disabled elements of the result vector are set to zero." Intel adds the
  fault-behaviour improvement over `vmaskmov`: "**if mask is all-zeros then memory faults will be
  ignored and no assist will be issued**", and "If mask is all 1: data can be forwarded from the
  masked store to the dependent loads."
  **Practical rule for us:** masked *loads* are cheap; masked *stores* cost ~10 cycles of extra
  latency each on this microarchitecture, so **pad the batch tile to a multiple of 8 rather than
  masking the store**, and always use the `{z}` zeroing form to avoid a false dependence on the
  destination register. Intel's §18.2.6 "Peeling and Remainder Masking" is the canonical pattern
  if you must mask.
* **Two-source permutes** `vpermt2pd`/`vpermi2pd`: 1/cycle on port 5, latency 1, "3 if lane
  crossed" (Agner). Note `vpermt2pd` is destructive in its first source. Per §4.2 item 5 these
  are cheaper on this SKU than on wallaby.
* **Mask register file depth**, for completeness: Travis Downs, *AVX-512 Mask Registers, Again*,
  26 May 2020 — https://travisdowns.github.io/blog/2020/05/26/kreg2.html [VERIFIED — fetched]
  measures the speculative mask register file at **128 entries**, shared with the x87/MMX file
  ("We see a clear spike at 128 instructions"), and concludes "it is hard to imagine this having
  an impact in any non-artificial example". Irrelevant unless you mix x87/MMX.

### 4.6 What changed on Ice Lake-SP and Sapphire Rapids — and what this means for wallaby→node transfer

**Intel stopped publishing the licence tables.** This is checkable and it matters: the 2nd Gen
(Cascade Lake) Specification Update 338848-028US contains fifteen figures of per-SKU AVX turbo
frequencies; the 4th Gen (Sapphire Rapids) Specification Update **784461-002** —
https://cdrdv2-public.intel.com/784461/784461_SapphireRapidsEE_SpecificationUpdate_002.pdf
[VERIFIED — fetched, full text extracted] — contains **none**, and the *Optimization Reference
Manual* rev 049 has a licence section for Skylake Server (§2.5.3) and Ice Lake **Client**
(§2.4.1.6) and **none for Ice Lake Server, Sapphire Rapids or Emerald Rapids**.

**What Intel does say, and it names the root cause** (§2.4.1.6, about Ice Lake Client), verbatim:

> "The typical P0n max frequency difference between Intel AVX-512 and Intel AVX2 on Ice Lake
> Client microarchitecture is **much lower** than on Skylake Server microarchitecture."
> "All processors based on Ice Lake Client microarchitecture contain a **single 512-bit FMA unit**,
> whereas some of the processors based on Skylake Server microarchitecture contain two such units.
> Both processors contain two 256-bit FMA units. **The power consumed by Ice Lake Client FMA units
> is the same, whereas on Skylake Server the 512-bit units consume twice as much.**"

That is the entire AVX-512-downclocking story in one sentence — and note that it is about the
*two-unit* Skylake Server parts. Our node is a **one-unit** part, which is consistent with §4.1's
finding that its L1 and L2 licences coincide at low core counts.

**Measured on Ice Lake (client).** Travis Downs, *Ice Lake AVX-512 Downclocking*, 19 August 2020 —
https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html [VERIFIED — fetched], on a
Core i5-1035G4: scalar/128-bit, light and heavy 256-bit, and light and heavy 512-bit all run at
3.7/3.6/3.3/3.3 GHz across 1–4 active cores except a single 100 MHz drop for 512-bit at one core —
"**a paltry only 100 MHz**". Client part; no verified per-core-count table exists for Ice Lake-SP.

**Measured on Sapphire Rapids.** Chester Lam, *Sapphire Rapids: Golden Cove Hits Servers*, Chips
and Cheese, 12 March 2023 — https://chipsandcheese.com/p/a-peek-at-sapphire-rapids
[VERIFIED — fetched], on a Xeon Platinum 8480: "**Unlike early AVX-512 chips, SPR does not appear
to have fixed frequency offsets when dealing with 512-bit vectors**", reaching "clock speeds as
high as **3.8 GHz with 512-bit vectors**" — with the honest caveat that "clock speeds go all over
the place" on a shared cloud machine. Same article: SPR has "**two 512-bit FMA units** — one
created by fusing 256-bit units on ports 0 and 1, with a second added to port 5."

**Emerald Rapids:** the commonly quoted 8592+ figures come from Phoronix, which returned 403 to
both research passes. **[UNVERIFIED — could not fetch.]** Likewise the "Ice Lake 8380 ~175 MHz
average drop" figure, which appears only second-hand inside LLVM issue #102047.

**The wallaby → node transfer verdict, now with the microarchitectural deltas named.** From the
*Optimization Reference Manual* §2.3 (Golden Cove, i.e. Sapphire Rapids cores), verbatim: "Wider
machine: 5→6 wide allocation, 10→12 execution ports, and 4→8 wide retirement"; "4→6 decoders";
"**Maximum load bandwidth increased from 2 loads/cycle to 3 loads/cycle**"; "Mid-level cache size
increased to **2MB on server parts**". Against Skylake Server: 4-wide, 2 loads/cycle, 1 MB L2,
LSD disabled, and one or two 512-bit FMA units by SKU.

| quantity | wallaby (Gold 6448Y, SPR) | node (Gold 5218, CLX) | consequence for tuning transfer |
|---|---|---|---|
| 512-bit FMA units | **2** | **1** | a 512-bit FMA-bound kernel is **2× more port-limited** on the node |
| 512-bit vs 256-bit FP throughput | **2:1** | **1:1** | any 512-vs-256 gain measured on wallaby **overstates the node by ~2×** |
| AVX-512 frequency offset | none observed (SPR) | L1 = L2 at 1–8 cores (this SKU) | wallaby *understates* nothing here — both are benign |
| load ports | 3/cycle | 2/cycle | a schedule tuned to 3 loads/cycle is load-port-bound on the node |
| front end | 6-wide alloc, 6 decoders | 4-wide, LSD disabled | instruction count matters **more** on the node |
| L2 per core | 2 MB | **1 MB** | every tile tuned on wallaby is **2× too large** (§1.4, §1.9) |
| L3 | 60 MB | 22 MB, victim, non-inclusive | a 40 MB L=36 batch is L3-resident on wallaby and streams on the node |

**The rule: on wallaby measure *cycles per volume* and *bytes per volume*, never microseconds;
halve any 512-bit FMA throughput advantage before extrapolating; and halve every tile size.**

**One verified cross-generation data point on how well *code* transfers:** *First Impressions of
the Sapphire Rapids Processor with HBM for Scientific Workloads*, **SN Computer Science (2024)**,
DOI 10.1007/s42979-024-02958-3 — https://link.springer.com/article/10.1007/s42979-024-02958-3
[VERIFIED — fetched]: "**SPR-DDR has a similar per-core performance to the Intel Skylake-X CPU**",
with the authors noting explicitly that "the same dispatch (AVX512_CORE) for jit'd kernels is used
in both cases" — i.e. identical code path, and the generational gain came from hardware rather
than retuning. And for methodology, Ayesha Afzal, Georg Hager, Gerhard Wellein, PMBS23 /
SC'23 Workshops, arXiv:2309.05373 — https://arxiv.org/pdf/2309.05373 [VERIFIED — fetched] compare
Ice Lake Platinum 8360Y against Sapphire Rapids Platinum 8470 with **clocks pinned to base on both
machines** to remove turbo variability. **If the panel wants comparable wallaby and node numbers,
pin both clocks.**

**Gap, stated so nobody hunts for it:** neither research pass found any SPIRAL or autotuning study
quantifying transfer loss across Intel *server* generations, nor any FFTW / OpenBLAS / BLIS / NumPy
published policy statement on 512-bit versus 256-bit. Compiler policy exists and is worth knowing:
LLVM D67259 — https://reviews.llvm.org/D67259 [VERIFIED — fetched] made
`-mprefer-vector-width=256` the default for `skylake-avx512` and `cascadelake` because "AVX512
instructions can cause a frequency drop on these CPUs", and Intel's own manual states
"**`-mprefer-vector-width=256` is the default for `skylake-avx512`**" and recommends building three
ways and picking the fastest. **Note carefully: this default is the compiler acting on the
two-FMA-unit, many-core case. §4.1's table says it is the wrong default for a single-threaded
kernel on a Gold 5218 — so if you write portable C and rely on auto-vectorisation, you are being
capped at 256 bits by default and should pass `-mprefer-vector-width=512` explicitly (or write
intrinsics, which the panel does).** LLVM issue #102047 —
https://github.com/llvm/llvm-project/issues/102047 [VERIFIED — fetched] is the open request to
change that default for newer parts. glibc made the same call for `memcpy`/`memset` (H.J. Lu,
BZ #21396 — https://git.zx2c4.com/glibc/commit/?id=4cb334c4d6249686653137ec273d081371b3672d
[VERIFIED — fetched]: "AVX512 load/store instructions in memcpy/memset may lead to lower CPU turbo
frequency in certain situations").

Finally, two verified cases where 512-bit **did** lose, so the rule above is not read as absolute —
both are *mixed-code, many-core* cases that do not describe us: Cloudflare's OpenSSL
ChaCha20-Poly1305 on a Xeon Silver 4116 served "**10 % fewer requests per second**" than AES-GCM
(https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/ [VERIFIED — fetched]),
and Gottschlag et al. measured nginx+OpenSSL+Brotli "slowed down by **10.7 %**, even though **less
than 1 % of the CPU time was spent in OpenSSL**", with average frequency pulled "down to 2.47 GHz
from 2.77 GHz" on a Xeon Gold 6130. And one case where width bought nothing because the code was
bandwidth-bound — mkFit issue #170 (https://github.com/trackreco/mkFit/issues/170
[VERIFIED — fetched]) on a Xeon Gold 6130: "very little speedup when increasing… from VU=8 to
VU=16", because "performance is limited by memory access speed", plus the observation that "Data
always traverse the levels of memory in units of cache lines, which are 512 bits in any case."
That last sentence is §1.2's "RAM bandwidth is barely affected by SIMD width", from a practitioner.

---

## 5. Target 5 — code generation, autotuning, and library design a panel of agents can actually use

### 5.1 The one number that justifies this entire project: FFTW's AVX-512 path is still experimental

FFTW *release notes* — https://www.fftw.org/release-notes.html [VERIFIED — fetched]. Version 3.3.5
(2016-07-31), verbatim: "**Experimental support for AVX512 and KCVI. (`--enable-avx512`,
`--enable-kcvi`) This code is expected to work but the FFTW maintainers do not have hardware to
test it.**" Version 3.3.9 (2020-12-13), verbatim: "**Note that avx512 support is still
experimental because the FFTW authors have no avx512 hardware available for testing.**" FFTW is
not abandoned — 3.3.11 shipped 2026-04-21 with `fftw_copy_plan()`, SVE support and new cycle
counters — but ten years after AVX-512 shipped, **the world's most-benchmarked FFT library has
never tested its AVX-512 code path on AVX-512 hardware.**

Alongside it, the FFTX authors' assessment of the field — Franz Franchetti, Daniele G. Spampinato,
Anuva Kulkarni, Doru Thom Popovici, Tze Meng Low, Michael Franusich, Andrew Canning, Peter
McCorquodale, Brian Van Straalen, Phillip Colella, *FFTX and SpectralPack: A First Look*,
**HiPC 2018** — https://users.ece.cmu.edu/~franzf/papers/hipc_2018.pdf [VERIFIED — fetched]:

> "Performance becomes an issue for some microarchitectures with **wider SIMD vectors**, large core
> counts, more than 32 MPI ranks, **batch FFTs**, and many important cases supported by the FFTW
> guru interface. The FFTW benchFFT webpage is outdated with a 2004/2006-era Pentium 4s and Xeons."

> "There exist some highly optimized routines (e.g., complex 2-power FFTs) while other kernels use
> some translation routine and can incur (very) high overheads. **Performance can vary by 2x to 10x
> or more. This means a number of routines run at 10% of the efficiency of 2-power FFTs.**"

Their capability table marks MKL's **batched** FFT as "Mixed" — supported but not uniformly
optimised. Between them, these two sources are the strongest published justification for
hand-writing batched AVX-512 kernels at L = 6, 17, 36 that exists.

### 5.2 Write intrinsics, not vectorisable C: the measured multiplier is 5–6×

**Source.** Daisuke Takahashi, Franz Franchetti, *FFTE on SVE: SPIRAL-Generated Kernels*,
**HPCAsia 2020**, DOI 10.1145/3368474.3368488 — https://www.spiral.net/doc/papers/ffte-spiral.pdf
[VERIFIED — fetched]. One core, one thread, double-precision complex, twiddles precomputed,
n = 64…65 536 (L2-resident), 10 runs averaged, on an A64FX (SVE) simulator with Fujitsu C 4.0.0 and
ARM C 19.1. The comparison is **SPIRAL-generated code with explicit SVE intrinsics against the same
code left to the compiler's auto-vectoriser**:

> "When Fujitsu C compiler is used, SPIRAL (SVE intrinsic) is approximately **6.33 times faster**
> than SPIRAL (automatic vectorization) for n = 256."
> "when ARM C compiler is used, the SPIRAL (SVE intrinsic) is approximately **5.62 times faster**
> than the SPIRAL (automatic vectorization) for n = 32,768."
> "for n = 32,768, the SPIRAL (SVE intrinsic) is approximately **3.16 times faster** than FFTE
> (automatic vectorization)."

Nothing else in this section comes close to that multiplier. It is ARM, and it is a simulator — but
§5.6's compiler evidence says the same thing on x86 from three other directions. **The panel already
writes intrinsics; this is the citation that says never to stop.**

The paper also carries a **radix cost table** (Table 2, Stockham, double-precision complex, real
operations in the inner loop) that is directly usable as a selection metric:

| | r2 | r3 | r4 | r5 | r6 | r8 | r10 | r12 | r16 |
|---|---|---|---|---|---|---|---|---|---|
| loads | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 |
| stores | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 |
| mults | 4 | 12 | 12 | 28 | 28 | 32 | 60 | 60 | 84 |
| adds | 6 | 16 | 22 | 40 | 46 | 66 | 102 | 118 | 174 |
| **byte/flop** | 6.400 | 3.429 | 3.765 | 2.353 | **2.595** | **2.612** | 1.975 | 2.157 | 1.984 |

with the note: "Higher radix FFTs require more floating-point registers to hold intermediate
results, and since the A64FX processor has **32 512-bit registers** higher radix kernels are a good
choice." **AVX-512 also has 32 512-bit registers, so that argument transfers exactly** — and it is
the same conclusion §4.2 reaches from the frequency side. Note that these are twiddled Stockham
counts, not our PFA counts, so use the byte/flop column as a *ranking*, not as our op count.

### 5.3 Exo 2: the one generator worth trying, and it wins exactly where we are

**Source.** Yuka Ikarashi, Kevin Qian, Samir Droubi, Alex Reinking, Gilbert Louis Bernstein,
Jonathan Ragan-Kelley, *Exo 2: Growing a Scheduling Language*, **ASPLOS '25**, arXiv:2411.07211 —
https://arxiv.org/pdf/2411.07211 [VERIFIED — fetched]. It compiles a C-like object language plus a
user-extensible *schedule* to **plain C with intrinsics calls** (`exocc file.py` → `file.c` +
`file.h`; `pip install exo-lang` — https://raw.githubusercontent.com/exo-lang/exo/main/README.md
[VERIFIED — fetched]).

All numbers on an AWS `m7i.xlarge` with an **Intel Xeon Platinum 8488C at 3.2 GHz**:

* **AVX-512 sgemm**: runtime ratio Exo / Exo 2 = 0.99–1.00 across M,N ∈ {256, 512, 1024}, K = 512;
  the text states Exo's implementation "is comparable to MKL's".
* **Schedule size** (Fig. 6c/9a): OpenBLAS reference **>1 690 lines**; Exo schedule 180; **Exo 2
  schedule 97 lines for AVX-512**, from an 11-line object program, producing **521 lines of
  generated C** via **756 primitive rewrites**.
* **BLAS-2 against Intel MKL on AVX2** (Fig. 8; ratio MKL / Exo 2, > 1 means Exo 2 faster;
  geometric means per size bucket). At the **10⁰ bucket**: `dgemv_n` **2.28**, `sgemv_n` **2.43**,
  `dgemv_t` **2.08**, `sgemv_t` **2.33**, `dger` **2.13**, `sger` **2.77**. At 10¹: 1.27, 1.29,
  1.32, 1.52, 1.86, 3.00. By 10⁴–10⁸: **0.96–1.16**.
* Scope: 80+ kernels, 24 BLAS-1 and 50 BLAS-2 variants across AVX2 and AVX-512.
* Documented limitation, which does not affect us: "limitations in Exo's object code, which lacks
  support for value-dependent control" blocked `nrm2` and `iamax`. A fixed-size FFT butterfly has no
  value-dependent control.

**Read the shape of that MKL comparison.** Generated code beats MKL by **2.0–2.8× at the smallest
sizes** and converges to parity at large sizes. That is the same phenomenon as libxsmm's bins
(§3.6), the same as the panel's own margins, and the same as Popovici et al.'s small-odd-cube result
(§2.2). **Vendor libraries lose at tiny sizes because they dispatch, and specialised straight-line
code wins.** It is now measured four independent times.

Even without adopting Exo, its `optimize_level_1` / `opt_skinny` pattern is a schedule an agent can
transliterate into C by hand: round the loop bound up to a multiple of the vector width; stage the
reused vector into registers; vectorise; interleave for ILP; **then specialise so register counts
are statically known.** Their AVX-512 micro-kernel generator emits sizes "m × 16n with m = 6, n ≤ 4
or m ≤ 6, n = 4" — i.e. a 6×64 register tile on 32 zmm, which is the same register-budget reasoning
§01 §7.3 applies to codelets.

*(Exo 1 — "Exocompilation for Productive Programming of Hardware Accelerators", Ikarashi, Bernstein,
Reinking, Genc, Ragan-Kelley, PLDI 2022, DOI 10.1145/3519939.3523446 — metadata verified at
https://pldi22.sigplan.org/details/pldi-2022-pldi/21/Exocompilation-for-Productive-Programming-of-Hardware-Accelerators
[VERIFIED — fetched], but the PDF returned 403/405 from every route. Its frequently-quoted
"80–95 % of peak" figures are **[UNVERIFIED — could not fetch]** and are not used here; the Exo 2
paper's own "comparable to MKL's" is the verified version.)*

### 5.4 The layout result, measured three independent times, and it is the panel's layout

This is the most cross-validated finding in this whole section, and it should end §04 §4.4's open
question.

**(i) Popovici, Franchetti, Low, *Mixed Data Layout Kernels for Vectorized Complex Arithmetic*,
IEEE HPEC 2017** — https://aiichironakano.github.io/cs653/Popovici-ComplexSIMD-HPEC17.pdf
[VERIFIED — fetched]. Intel Haswell 4770K and Kabylake 7700K, AVX, **double precision**:

* Mechanism: interleaved layout requires shuffles for the complex multiply; "the number of
  permutations (per cycle) of such instructions are usually limited… **the split data layout
  eliminates permutations**."
* **`DFT_n ⊗ I_ν` kernels — literally our kernel shape** — "are similar to the codelets used by
  FFTW to generate implementations for larger FFT sizes": "our implementations are **1.3 to 2×
  faster** than FFTWs implementations."
* 1D codelets n = 64/128/256/512: mixed format is "**slightly better (5 % to 15 %)** then the FFTW
  and MKL implementations." 2D FFTs 64–512 per side: "between **10 % and 30 %**" over MKL and FFTW.
* `zgemm`: mixed format reaches peak at block size 192 versus 256 interleaved; for k < 100 the
  improvement is **10–33 %** (Kabylake) and **10–37 %** (Haswell), falling to 1–2 % at large k.
* And a **peak-rate model the panel should adopt**, verbatim: "Given that FFTs have predominantly
  more additions than multiplications… we can make the assumption that **FFTs are bounded by
  additions**. Since the Intel Haswell architecture can compute only one vector addition per cycle,
  we can determine that for a vector length ν = 4 the theoretical peak is **4 FLOPs/cycle**. Intel
  Kabylake has twice the throughput… therefore the FFT peak performance for the same ν = 4 is
  **8 FLOPs/cycle**." *(On our Gold 5218 with one 512-bit unit, a 512-bit `vaddpd` is also 1/cycle,
  giving 8 DP FLOPs/cycle — which is exactly the floor the r3 verdict computed. The two models
  agree, which is reassuring.)*
* Their closing recommendation is §1.9 and §2.2 restated: "one such opportunity for improving
  performance is **preserving the block interleaved data format across computation kernels**, thus
  reducing the overall number of packing and unpacking routines."

**(ii) Yifei He, *Domain-Specific Compilation Framework with High-Level Tensor Abstraction for Fast
Fourier Transform and Finite-Difference Time-Domain Methods*, KTH Doctoral Thesis, defended 12 June
2025, TRITA-EECS-AVL-2025:67 —**
https://www.diva-portal.org/smash/get/diva2:1958857/FULLTEXT01.pdf [VERIFIED — fetched]. (This
thesis reproduces the full text of the paywalled *High-Performance FFT Code Generation via MLIR
Linalg Dialect and SIMD Micro-Kernels*, He & Markidis, IEEE CLUSTER 2024, so its numbers are
citable.) It measures the *cost of getting the layout wrong* on x86:

> "under the conventional interleaved layout, the LLVM vectorizer first emits **gather**
> instructions to load the real and imaginary parts into separate SIMD registers… and subsequently
> emits **scatter** operations to write the computed results back into the interleaved memory layout.
> **This pattern is repeated for each arithmetic step**, introducing significant memory access
> overhead."

Their Table 4.3, gather versus in-register permute:

| input size, stride | approach | instructions | **IPC** | cache refs | L1 miss | **speed** |
|---|---|---|---|---|---|---|
| 64, stride 8 | gather | 1× | **0.13** | 1× | 0.08 % | 1× |
| 64, stride 8 | permute | 2.19× | **2.59** | 0.27× | 0.01 % | **9.25×** |
| 256, stride 16 | gather | 3.64× | 0.12 | 3.55× | 0.11 % | 0.25× |
| 256, stride 16 | permute | 13.29× | 2.19 | 0.90× | 0.01 % | 1.30× |

**IPC 0.13 → 2.59 and 9.25× the speed, for 2.19× more instructions.** That is the single most
vivid published demonstration of "instructions are not the currency when one of them is a gather",
and it is a direct measurement of the hazard §04 §8.4 warns about. Their fix is a **block-interleaved
layout converted once at micro-kernel entry and reverted at exit** — citing Popovici's HPEC 2017
paper — which is exactly §1.10's recommendation.

Their other findings, all actionable: **"the innermost loop vectorizer SLP with interleaved memory
access optimization outperforms the outermost loop Vectorizer VPlan. VPlan cannot work with
interleaved memory access optimization and generates gather/scatter instructions"**; the MLIR-native
vector dialect was a dead end ("`vector.transfer read/write` is scalarized when lowering down");
they had to **patch the x86 backend's `PreferVectorWidth`** because "in our FFT computation, wider
vectors brings more performance" (see §4.6 — this is the same 256-bit default); single-threaded
against FFTW/FFTE/SPIRAL/NumPy at sizes 64–2048 they report "a consistent **5× speedup over
NumPy**" and "**2× to 3× speedup**" over scalar FFTW while remaining short of AVX-512 FFTW; and
their own diagnosis of the residual gap names something we must avoid: "**certain optimizations
involving constant elimination are hampered when FFT kernels contain divergent constants across
SIMD lanes — e.g., mixed zeros and ones — limiting vectorization efficiency.**" *If you build a
twiddle vector containing literal 0.0 and 1.0 in different lanes — which happens naturally in a
radix-4 or radix-8 stage — you block constant folding. Specialise those butterflies instead.*

**(iii) ducc0's source code does the same thing**, see §5.5.

**Verdict: `LITERATURE.md` §4.4 ("split vs interleaved: strongly motivated, unproven") can be
closed.** There are now three independent sources — one measuring 1.3–2× on `DFT_n ⊗ I_ν` kernels in
double precision on x86, one measuring 9.25× for the permute-versus-gather core of the same
decision, and one production library — all landing on split/block-interleaved. §04's verdict was
right and the panel's layout is right.

### 5.5 ducc0 / pocketfft: the design notes are the source, and they answer the L=17 question differently

Martin Reinecke's ducc0 has no design paper, but its README and source are explicit. All
[VERIFIED — fetched]: https://raw.githubusercontent.com/mreineck/ducc/master/README.md,
https://raw.githubusercontent.com/mreineck/pocketfft/master/README.md,
https://raw.githubusercontent.com/mreineck/ducc/master/src/ducc0/fft/fft.h,
`.../fft1d_impl.h`, `.../fftnd_impl.h`, `.../ChangeLog`, `python/demos/fft_bench.py`.

**His own performance claims**, verbatim: "1D transforms are **somewhat slower** than those provided
by FFTW (if FFTW's plan generation overhead is ignored)"; "multi-D transforms in double precision
perform **fairly similar to FFTW with FFTW_MEASURE**; in single precision `ducc.fft` **can be
significantly faster**"; "makes use of CPU vector instructions, **except for short 1D transforms**";
"supports prime-length transforms without degrading to O(N**2) performance"; "the storage
requirement for a plan only scales with the **square root of the FFT length**".

**Five design decisions worth copying, with the code:**

1. **L=17 gets neither Rader nor Bluestein.** `cfftpass::factorize(N)` greedily extracts 8, then 4,
   then at most one 2 (moved to the front of the factor list), then odd primes ascending — so
   L=8 → `[8]`, L=6 → `[2,3]`, L=36 → `[4,3,3]`. For a single prime factor the dispatch is:
   `if (ip<110) return cfftpg(...); else return cfftpblue(...)`. **17 < 110, so ducc0 uses its
   generic pass, and there is no Rader implementation in this path at all.** Compare VkFFT, whose
   `fixMinRaderPrimeMult` and `fixMinRaderPrimeFFT` defaults are both **exactly 17** (§5.7). Two of
   the best-engineered FFT libraries in the world disagree about what to do at precisely our prime.
   **`LITERATURE.md` §4.2 is therefore not a gap in our knowledge; it is a genuinely open question in
   the field, and the panel is in a position to answer it.**
2. **Short 1D transforms are deliberately not vectorised**:
   `if (vectorize && (ip>300) && (ip<=100000) && (l1==1) && (ido==1))` — the vectorised pass is
   gated to prime factors above 300. Everything small goes through scalar-shaped code, vectorised
   only across the *multi-dimensional* / batch loop. That is the `A ⊗ I_ν` doctrine implemented as a
   policy.
3. **SIMD width is deliberately capped below AVX-512 for FFT**:
   ```
   template<> constexpr inline size_t fft_simdlen<double> = min<size_t>(4, native_simd<double>::size());
   ```
   **Four doubles — AVX2 width — even on AVX-512 hardware.** The source gives no reason. Given §4.1
   (no frequency penalty on our SKU at one core) and §4.2 (identical FP throughput), this is a
   deliberate choice by a careful author that our analysis says should not matter here — worth one
   A/B rather than an assumption in either direction.
4. **Critical-stride avoidance, verbatim from `TmpStorage`:**
   ```c
   // critical stride avoidance
   if ((dstride&256)==0) dstride += 16;
   if ((dofs   &256)==0) dofs    += 16;
   ```
   *Pad any scratch stride that is a multiple of 256 elements by 16 elements.* Compare §04 §7.3's
   odd-cache-line rule and the corpus's L=8/L=36 padding disagreement (`LITERATURE.md` §4.5): this
   is a production library doing the same thing with a one-line guard, and it is the cheapest
   possible implementation of that advice. ducc also ships a public `make_noncritical()` helper and
   calls it on the work buffer in its own benchmark script.
5. **Sort the axes by output stride**, verbatim from `multi_iter`: "**Sort the extraneous dimensions
   in order of ascending output stride**; this should improve overall cache re-use and avoid clashes
   between threads as much as possible", followed by an explicit loop that collapses contiguous
   dimensions. Compare Popovici et al.'s "pick the x dimension first" (§2.2) and
   `L17_matrixsimd`'s X-first reorder worth 10.8 %: **three independent arrivals at the same rule.**

Twiddle generation, from the pocketfft README, is also worth copying: "all angles are reduced to the
range `[0; pi/4]`", using an adapted `sincospi()` that computes `sin(x)` and `(cos(x)-1)`, and "if
`n` sin/cos pairs are required, the adjusted `sincospi()` is only called `2*sqrt(n)` times; the
remaining values are obtained by evaluating the angle addition theorems." Our tables are tiny enough
that we compute them offline in extended precision instead (§07 §5.3) — but the `[0, π/4]` reduction
is exactly how to generate them accurately.

**And the one fetchable cross-library benchmark with small-size numbers**: Marcin Wojdyr,
`project-gemmi/benchmarking-fft` —
https://raw.githubusercontent.com/project-gemmi/benchmarking-fft/master/README.md
[VERIFIED — fetched]. Old (≈2019, Intel i7-5600U, GCC 8 `-O3`) but the only such source either pass
found. 1D complex-to-complex, single precision except pocketfft (double):

| | n=256 | n=384 | n=480 | n=512 |
|---|---|---|---|---|
| fftw3 estimate | 321 ns | 499 ns | 1538 ns | 663 ns |
| fftw3 measure | 274 ns | 443 ns | 883 ns | 588 ns |
| pffft | 585 ns | 1014 ns | 1329 ns | 1255 ns |
| fftw3 measure, **SIMD off** | 1699 ns | 2748 ns | 3855 ns | 3832 ns |
| pocketfft (double) | 1690 ns | 3035 ns | 3633 ns | 4009 ns |
| kissfft | 2536 ns | 4929 ns | 6030 ns | 6553 ns |

Author's summary: "**To a first approximation, SSE1 gives 3x speedup, AVX -- 6x**"; "`-ffast-math`
doesn't seem to make a significant difference"; "**When using Clang 8 instead of GCC, PocketFFT is
~12 % faster**". Plan-generation cost at n=512: pocketfft **1 274 ns**, fftw3 estimate 9 693 ns,
**fftw3 measure 25 610 ns** — relevant because our `create()` is untimed but the r3 verdict notes
plan time growing to 1.0–1.6 s, which is 10⁵× FFTW's.

**One number from that page every L=36 implementer should see:** a 3D transpose of `complex<float>`
at 256³ costs 22 ms as a plain assign, 25 ms `naive yxz`, 49 ms `tiled zxy`, 89–91 ms for
`naive xzy` / `naive zxy` / in-place, and **202–204 ms for `naive zyx` / `naive yzx`. A bad axis
order costs 9× over a straight copy.**

### 5.6 Compiler reality on x86, 2020–2026

* **No compiler is reliably best, and half of all loops don't vectorise.** Nazmus Sakib, Tarun
  Prabhu, Nandakishore Santhi, John Shalf, Abdel-Hameed A. Badawy, *Comparison of Vectorization
  Capabilities of Different Compilers for X86 and ARM CPUs*, 17 February 2025, arXiv:2502.11906 —
  https://arxiv.org/pdf/2502.11906 [VERIFIED — fetched]. **GCC 14.1.1, Clang 18.1.8, ICX 2024.0.2**
  on an **Intel Xeon Gold 6152**, `-O3 -march=skylake-avx512 -mprefer-vector-width=512`. Of 151
  TSVC2 loops: **GCC failed to vectorise 46 %, Clang 54 %, ICX 50 %**; "the code generated by **ICX
  was fastest for 40 %** of the loops, **GCC for 39 %**, **Clang for 21 %**"; "the vectorized code
  did not always outperform the unvectorized code"; and the verdict — "**we cannot definitively say
  if any one compiler is significantly better than the others**." *Note their methodology detail:
  they deliberately hid the compile-time array sizes because the original suite's compile-time-known
  bounds are "not representative of scientific applications." Our sizes* are *compile-time constants,
  which is a real advantage — but §5.4(ii) shows it does not rescue straight-line complex arithmetic.*
* **`-ffast-math` will silently break a complex butterfly, and it did so in FFTW.** FFTW 3.3.8
  release note (verified above), verbatim: "Fixed AVX, AVX2 for gcc-8. By default, FFTW 3.3.7 was
  broken with gcc-8. **AVX and AVX2 code assumed that the compiler honors the distinction between
  +0 and -0, but gcc-8 `-ffast-math` does not. The default CFLAGS included `-ffast-math`.**" Any
  kernel using `addsub`-shaped patterns or sign-folded constants (§6.7) is exposed. The panel builds
  with `-O3 -march=native -mtune=native -std=gnu11` and no `-ffast-math`, which is the right choice;
  **any entry that adds it must A/B for bit-exactness on the ±0 cases, not just for the 1e-12
  tolerance.**
* **`-mprefer-vector-width=512` must be passed explicitly.** §4.6 has the compiler-policy citations;
  §5.4(ii) has an FFT project that had to patch the backend to get zmm. The panel writes intrinsics,
  so this affects only the surrounding loop code — but it affects it.
* **Mark input and output `restrict`.** ducc0 annotates every hot pointer
  (`Cmplx<Tsimd> *DUCC0_RESTRICT dst`, `const Cmplx<...> *DUCC0_RESTRICT ptr`); FFTE-SVE used
  Fujitsu's `-Krestp=all`, which "specifies to perform the optimization of restricted pointers
  assuming that the `restrict` qualifier is specified for all pointers". For an out-of-place batched
  kernel this is free and removes a real aliasing question.
* **Compiler choice is worth ~10 % on identical FFT source** — gemmi's "PocketFFT is ~12 % faster"
  with Clang 8 over GCC; KFR's README says "use Clang or GCC when maximum performance is required"
  and ships "Bit-index permutation optimizations **for Clang** and the generic backend"
  (https://raw.githubusercontent.com/kfrlib/kfr/master/README.md [VERIFIED — fetched]); and
  FFTE-SVE's cross-compiler inversion, where auto-vectorised Fortran won under Fujitsu's compiler
  and intrinsics won under ARM's. **`LITERATURE.md` §5 item 8's "compiler flags as a searched
  dimension" is confirmed, and the compiler *identity* belongs in that search too.**
* **`-O3` is not a fixed point.** Zhengyang Liu, Stefan Mada, John Regehr, *Minotaur: A
  SIMD-Oriented Synthesizing Superoptimizer*, arXiv:2306.00229 —
  https://arxiv.org/pdf/2306.00229 [VERIFIED — fetched] — a synthesis-based superoptimizer over
  "165 vector intrinsics mapping to SSE, AVX, AVX2 and AVX-512", every rewrite formally verified by
  an extended Alive2, measured on an **Intel Xeon Gold 6210U (Cascade Lake, AVX-512)**: **average
  7.3 % speedup on GMP's benchmark suite, max 13 %**; SPEC CPU 2017 average 1.5 %, max 4.5 %;
  324 rewrites at geomean 1.0610×; compile-time cost "with a warm cache it is just 3 % slower".
  **Caveat that matters for us: integer-vector rewrites contribute 75.57 % of the speedup and
  FP-vector only 1.64 %**, so do not expect much on pure FP butterflies — but it runs on our exact
  microarchitecture and costs one build.
* **I could not find a rigorous, fetchable 2020–2026 measurement of `-O2` versus `-O3` on small
  numeric kernels.** Neither research pass found one. §06 §3.4's >3× spread across random flag sets
  remains the corpus's best datum. Given everything above, the practical recommendation is a
  **{-O2, -O3} × {prefer-vector-width 256, 512} × {gcc, clang, icx} × {±ffast-math}** grid, which is
  24 builds and, on this evidence, likely to move performance more than most algorithmic variants.

### 5.7 VkFFT and cuFFTDx: the fusion argument, and the only published number for it

Both are GPU. **The memory-hierarchy argument and the decomposition policy transfer; the absolute
numbers do not.**

**The fusion claim, quantified — and this is the only quantified version of it anywhere.** Dmitrii
Tolmachev, VkFFT SC22 technical poster —
https://sc22.supercomputing.org/proceedings/tech_poster/poster_files/rpost143s3-file2.pdf
[VERIFIED — fetched], verbatim:

> "FFT is an extremely global memory bandwidth-limited algorithm, which means **having two global
> memory data round trips instead of one often decreases performance by a factor of two.**"

VkFFT's *API guide* — https://raw.githubusercontent.com/DTolm/VkFFT/master/documentation/VkFFT_API_guide.pdf
[VERIFIED — fetched] — states the design rule in a form that is architecture-neutral: "**GPUs and
CPUs have a hierarchical memory model** – the closer memory to the unit that performs the
computations, the faster its speed and the lower the size. So it is advantageous to split FFTs,
**not to the lowest primes, but to some bigger multiplication of those primes**, then upload this
subsequence to the closest cache level to the cores and do the final prime split there." Plus
"Register overutilization… use a register file (which is often bigger than the amount of shared
memory) to store the sequence and use shared memory only as a communication buffer."

**Translated to our machine** *(mine)*: the register file is 32 × 512 bits = 2 KiB and L1d is
32 KiB, so "one round trip" means *the whole batch tile lives in registers plus L1 and is written to
memory exactly once*. At L=6 (3.375 KiB/volume) and L=8 (8 KiB/volume) a granule of 8 volumes is
27 KiB and 64 KiB — the first fits L1, the second needs L2. At L=36 (746 KiB/volume) it is
impossible, and VkFFT's own prescription applies: a four-step-style split with **inlined twiddle
multiplication** rather than a separate transpose pass. Note the important caveat from §1.9 and the
r3 verdict: on a CPU the L1↔L2 gap is 2.6×, not the 10–20× global-memory gap this factor-of-two
claim is measured against, so **the CPU payoff is the *L2↔DRAM* version of the argument, not the
L1↔L2 version.**

VkFFT's other documented techniques, verbatim: "VkFFT does multiple small FFTs at once – increases
thread block size and allows to access continuous blocks of memory"; "**VkFFT does not have a
transposition for the strided FFT axes – they are done by coalescing the neighboring sequences**"
(the GPU analogue of never materialising a transpose, which is §05 §3.2's point); and
"Removing additional last forward FFT/first inverse FFT memory requests for convolutions by inlining
kernel multiplication in the generated code. **Removes one data round-trip.**"

**VkFFT's documented prime policy — and 17 is exactly the threshold.** Radices implemented:
**2/3/4/5/7/8/11/13**, with "Sequences using radix 3, 5, 7, 11 and 13 have comparable performance to
that of powers of 2." Rader "for primes from **17 up to max shared memory length (~10000)**.
**Inlined and done without additional memory transfers.**" Parameters:
`fixMinRaderPrimeMult` — "start direct multiplication Rader's algorithm for radix primes from this
number… **Default is 17**"; `fixMaxRaderPrimeMult` — "realistically you would want to switch
somewhere on **30-100 range**. Default is vendor-specific (currently ~40)";
`fixMinRaderPrimeFFT` — "start FFT convolution version of Rader… **Better than direct
multiplication version for almost all primes (except small ones, like 17-23 on some GPUs)**.
Default **29 on AMD and 17 on other GPUs**"; `fixMaxRaderPrimeFFT` — "switch to Bluestein's
algorithm for radix primes from this number… Default is **16384**."

**So VkFFT's own documentation says 17 is a coin-flip between Rader-with-FFT-convolution and
Rader-with-direct-multiplication, and ducc0 (§5.5) uses neither.** Three defensible answers at our
one prime. That is the single clearest statement of why L=17 is the panel's opportunity, and it
supersedes `LITERATURE.md` §4.2's framing: the question is not "which does the literature say", it
is "nobody knows".

VkFFT's poster also documents the competition, useful for calibration: "cuFFT only uses Rader's
algorithm for **primes up to 127** and implements it as a **direct matrix multiplication**"; "cuFFT
does not use Rader's algorithm in FP32 and switches to Bluestein's algorithm for **primes after
17**." And on accuracy: VkFFT is verified against **FP128 FFTW** over lengths [2, 100000]; "With
FP128 precomputation VkFFT is more precise than cuFFT and rocFFT"; "With FP32 twiddle factors VkFFT
is slightly less precise in Bluestein's and Rader's algorithms" — corroborating §07 §5.2.

**cuFFTDx publishes the fusion argument and no number for it.**
https://docs.nvidia.com/cuda/cufftdx/index.html and
https://docs.nvidia.com/cuda/cufftdx/performance.html [both VERIFIED — fetched]: "Ability to fuse
FFT kernels with other operations in order to **save global memory trips**"; "**Merge adjacent
memory bound kernels (pre- and post-processing) with an FFT kernel to save global memory trips**";
"Use shared memory or extra registers to store the temporary data"; and, worth noting for §3,
"**Best parameters for compute bound and memory bound kernels might not be identical.**" Neither
page contains any quantitative comparison against cuFFT.

**rocFFT's Bluestein design note** — https://rocm.docs.amd.com/projects/rocFFT/en/docs-6.1.1/design/bluestein.html
[VERIFIED — fetched] — supplies one useful admission: "**the Bluestein DFT computation is much
slower than directly computing the DFT equation via an FFT with a supported length, even though both
computations posses the same complexity of O(N log N)**", and their optimised version "reduces kernel
count from 8 (default) to 6 kernels… by fusing operations and precomputing the chirp sequence at the
plan creation phase". rocFFT's radix support is "combinations of powers of 2, 3, and 5" — so 7, 11,
13 and 17 are second-class there. Corroborates §02 §4's "do not use Bluestein at L=17".

**oneMKL's internals are not publicly documented.** The oneAPI specification page
(https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/elements/onemkl/source/domains/dft/dft.html
[VERIFIED — fetched]) documents batched DFTs and strides but specifies **no radix set, no
prime-length policy and no internals**; its only performance guidance is that descriptors should be
"created, configured and committed **outside** applications' hotpath(s)". `LITERATURE.md` §4.8
gap 4 ("no primary source for MKL's prime-size algorithm") is therefore not a search failure — the
information is not public. Treat MKL as a black box and expect FFTX's "2× to 10× or more" variance
across sizes.

### 5.8 Autotuning at tiny sizes: how much search, and which searcher

**Source.** Richard Schoonhoven, Ben van Werkhoven, K. Joost Batenburg, *Benchmarking optimization
algorithms for auto-tuning GPU kernels*, IEEE Transactions on Evolutionary Computation,
arXiv:2210.01465 — https://arxiv.org/pdf/2210.01465 [VERIFIED — fetched]. GPU kernels, but this is
the most rigorous "how much search do you need" study either pass found: 26 kernel spaces × 9 GPUs ×
16 algorithms, with every space **brute-forced into cache files** so the true optimum is known.

* Space sizes: Convolution **18 432 points, 89 local minima, 6 tunable variables**; GEMM **82 944
  points, 64 local minima, 5 variables**; point-in-polygon 8 184 points, 220 local minima, 10
  variables. And the number that will feel familiar: "For convolution and GEMM the majority of the
  points in the kernel space **fail to compile, 68 % and 78 %** respectively."
* Budgets: 25 … 1600 function evaluations, each repeated 50 times.
* Verdict, verbatim: "**dual annealing performs best as GPU kernel tuner when a limited amount of
  function evaluations is desirable. When more evaluations are possible, first-improvement local
  searchers such as FirstILS proved the best.**" Also: "best-improvement local search algorithms
  performed significantly worse than the first-improvement variants."
* **Models lost to search here**: "SMAC consistently achieves a lower fraction of optimality than
  the competing algorithms across kernels and budgets… we hypothesize that this is because of the
  high number of fail fitnesses in the search spaces."
* Treat tuning as deterministic: "the average (normalized) runtime and standard deviation is
  **1.000 ± 0.011**."

**Read across to the panel's plan-time tuners.** Our space — radix factorisation order, batch-tile
width, register-blocking factor, twiddle load strategy, pass count, store policy, prefetch distance,
axis order — is plausibly 10³–10⁵ points with a large invalid fraction, i.e. structurally the same as
theirs. **Use dual annealing at small budgets or first-improvement iterated local search at larger
ones; do not build a surrogate model; and treat the measurement as deterministic (which the r3
verdict's tuner-instability finding says the panel currently does not — see §4.3).**

The counterpoint, for completeness: a learned cost model *can* beat search when the space is
scheduling and you can afford the training data — Andrew Adams et al., *Learning to Optimize Halide
with Tree Search and Random Programs*, **ACM TOG 38(4), July 2019** —
https://halide-lang.org/papers/halide_autoscheduler_2019.pdf [VERIFIED — fetched] — geomean
**1.37×** (greedy) and **2.29×** (beam search) over the previous Halide autoscheduler on 100 unseen
pipelines on an Intel Core i9-7960X, "the first automatic scheduling algorithm to significantly
outperform human experts on average", cost model R² = 0.96, and "**We found no benefit in autotuning
on the random pipelines**" — at a training cost of "hundreds of thousands of random programs".
Not a route available to this panel.

Two more datapoints on search cost at genuinely tiny sizes. Alexa VanHattum, Rachit Nigam, Vincent
T. Lee, James Bornholt, Adrian Sampson, *Vectorization for Digital Signal Processors via Equality
Saturation*, **ASPLOS '21** — https://cs.wellesley.edu/~avh/diospyros-asplos-2021-preprint.pdf
[VERIFIED — fetched] synthesises the *data movement* as well as the arithmetic via equality
saturation, and its motivation is our problem: existing auto-vectorisers "struggle to invent the
complex data movements necessary to optimize small kernels"; small kernels are "not much larger than
the vector width". Geomean **3.1×** over the best non-expert baseline, and "**within 8 % of expert
performance, compiled in 2.7 s**" against one hand-tuned matmul. But the search-cost table is the
number to internalise: MatMul 2×2 **1.9 s**, 3×3 **2.7 s**, 4×4 **5.8 s**, **8×8 timed out at 180 s
using 4.0 GB**, 16×16 timed out; QRDecomp 4×4 ran **4 h 25 m using 35.4 GB** and still timed out.
*(Target is a Tensilica Fusion G3 DSP, not x86 — the absolute numbers do not transfer, the
tractability boundary does.)* **Full synthesis of shuffle networks is tractable at about 4×4 and
falls off a cliff by 8×8: synthesise the radix codelet, hand-schedule everything around it.**

For completeness on the frameworks the brief named: Tiramisu (arXiv:1804.10694
[VERIFIED — fetched]) "outperforms Intel MKL by up to 2.3×" on a dual-socket 24-core Xeon E5-2680v3
and notably includes **Baryon, "a dense tensor contraction code for constructing Baryon Building
Blocks"** — LQCD-adjacent. Ansor (arXiv:2006.06762 [VERIFIED — fetched]) beats other search
frameworks by 1.1–22.5× and manual libraries by up to 3.8× on an 18-core Intel Platinum 8269CY, with
"up to 1,000 measurement trials per test case". MDH/ATF (arXiv:2405.05118 [VERIFIED — fetched])
reports speedups over oneMKL **on an Intel Xeon Skylake Gold-6140 @ 2.30 GHz** of **6.27× for
`MatMul(10,500,64)`** and 3.42× for `MatMulT(10,500,64)` — but 0.69× for `MatMul(1024³)` and 0.42×
for `MatVec(4096,4096)`; **again: wins at skinny/tiny shapes, loses at the shapes vendors tune** —
at a cost of "a uniform 12 h per tuning run". ATL (Liu, Bernstein, Chlipala, Ragan-Kelley, POPL
2022 — https://people.csail.mit.edu/lamanda/assets/documents/LiuPOPL2022.pdf [VERIFIED — fetched])
gives Coq-verified schedule rewrites at "comparable performance to a roughly equivalent schedule
programmed in Halide", but with the disqualifying caveat for us: "**vectorization was left to the
downstream C compiler for ATL programs**" — which §5.2 and §5.4 say is the wrong end of the problem.
RISE/Shine (arXiv:2201.03611 [VERIFIED — fetched]) is evaluated on OpenCL GPUs only.

### 5.9 What the LLM/agent kernel-generation literature says about how this panel should be run

This is meta, but the brief asked, and the findings are specific enough to act on. All
[VERIFIED — fetched].

* **Serial refinement beats parallel sampling at equal budget.** Carlo Baronio, Pietro Marsella
  et al., *Kevin: Multi-Turn RL for Generating CUDA Kernels*, arXiv:2507.11948 —
  https://arxiv.org/pdf/2507.11948. Isobudget ablation at total budget 128: **16 parallel
  trajectories × 8 serial refinement turns → 1.10× mean speedup / 82 % correct**, versus **32
  parallel × 4 turns → 1.02× / 83 %**. Their summary: "**we found scaling serial refinement more
  beneficial than parallel sampling.**" Overall training moved correctness 56 % → 82 % and mean
  speedup 0.53× → 1.10× over PyTorch eager. **The panel's structure — eleven agents, six rounds,
  each round reading the previous round's records — is the serial-refinement shape, and this says
  that is the right call.**
* **Execution *and profiler* feedback is worth ~2× on success rate.** Anne Ouyang, Simon Guo, Simran
  Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini, *KernelBench: Can LLMs Write
  Efficient GPU Kernels?*, arXiv:2502.10517 — https://arxiv.org/pdf/2502.10517. At a fixed budget of
  10 calls, `fast_1` for DeepSeek-R1 on Level 2: single attempt **36 %**, + prior generation **44 %**,
  + execution feedback **62 %**, **+ profiler output 72 %**. "iterative refinement being more
  effective in 5 of the 6 cases" than repeated sampling. **The panel gives implementers `tryout.sh`
  (execution feedback) but not profiler output. This measurement says adding `perf stat` to
  `tryout.sh`'s output is worth 10 percentage points of success rate** — and §1's counters
  (`ld_blocks_partial.address_alias`, `L1D_PEND_MISS.FB_FULL`, `cycles`/`ref-cycles`) are exactly
  the ones to add.
* **Agentic feedback is worth ~4× on correctness.** Jianghui Wang, Vinay Joshi et al. (AMD),
  *GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks*, arXiv:2507.23194 —
  https://arxiv.org/pdf/2507.23194: "correct kernel generation rate up to **54.89 %** on
  TritonBench-revised and **63.33 %** on the ROCm Triton benchmark—compared to **less than 15 % when
  directly prompting strong LLMs without agentic feedback**", with "average speedup of up to
  **2.59×**".
* **Two things that *backfired*, both directly relevant to how `PANEL_BRIEF.md` is written.**
  KernelBench §5.2: "**In-context examples degrade the LM's overall `fast_1` score since LMs attempt
  more aggressive optimization strategies, but result in more execution failures.** OpenAI o1's
  generations are 25 % longer on average"; and on hardware documentation — "**Providing hardware
  information does not significantly impact the outputs**… LMs are better at adjusting their
  approaches when provided with few-shot examples… than with hardware information." *Read that
  against this document, which is 2 000 lines of hardware information. The honest conclusion is that
  the corpus earns its place only insofar as it converts to **specific numbered actions with
  measurements attached** — which is why §0 exists and why `LITERATURE.md`'s per-size priority lists
  matter more than the sections behind them.*
* **Verify, don't trust.** Jubi Taneja, Avery Laird, Cong Yan, Madan Musuvathi, Shuvendu K. Lahiri
  (Microsoft Research / Toronto), *LLM-Vectorizer: LLM-based Verified Loop Vectorizer*,
  arXiv:2406.04693 — https://arxiv.org/pdf/2406.04693: LLM-generated intrinsics achieved "run-time
  speedup ranging from **1.1x to 9.4x** as compared to the state-of-the-art compilers such as Intel
  Compiler, GCC, and Clang" — but they could only formally verify "**38.2 %** of vectorizations as
  correct" with Alive2. **We have a far better oracle than they did:** `check.py` against
  `numpy.fft.fftn` at 1e-12 on the full batched volume, plus the bit-identical re-run check. VkFFT's
  practice (verify against FP128 FFTW across the whole length range) is the standard to aim at, and
  the panel already meets a weaker version of it.
* **A cheaper first pass exists: have the agent rewrite C so the compiler can vectorise it.**
  Zhongchun Zheng, Kan Wu, Long Cheng, Lu Li, Rodrigo C. O. Rocha, Tianyi Liu et al., *VecTrans:
  Enhancing Compiler Auto-Vectorization through LLM-Assisted Code Transformations*, arXiv:2503.19449
  — https://arxiv.org/pdf/2503.19449, on a **Xeon Gold 6132** with GCC 14.2 / Clang 20 / ICX 2023.2
  at `-O3 -ffast-math`: **1.77× geometric mean over `-O3`, peaking at 5.21×**, and 1.2× geomean
  under ICX. Useful context: of TSVC-2's 149 functions, "**50 functions (approximately 33.6 %)
  resist effective vectorization** by … mainstream compilers". For *our* kernels §5.2's 5–6×
  intrinsics multiplier dominates this, so it is a fallback, not a strategy.
* **Multi-agent beats single-agent, modestly.** Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang
  Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, Alex Aiken, *Astra: A Multi-Agent System for GPU
  Kernel Performance Optimization*, arXiv:2509.07506 — https://arxiv.org/pdf/2509.07506: "average
  speedup of **1.32×** using zero-shot prompting" versus "a single-agent baseline, which attains only
  **1.08×**" on three SGLang kernels. And Jinwu Chen, Qidie Wu, Bin Li et al., *cuPilot: A
  Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution*, arXiv:2512.16465 —
  https://arxiv.org/pdf/2512.16465: "**3.09× speedup over PyTorch**" on KernelBench's 100 kernels,
  combining an evolutionary framework with **roofline-guided analysis** — the same
  compute-the-roofline-then-choose discipline §3.7 recommends.

**Two process changes this literature supports, both cheap:**

1. **Add `perf stat` output to `tryout.sh`** (KernelBench: +10 points of success rate from profiler
   feedback). The counters §1 needs are `cycles`, `ref-cycles`, `ld_blocks_partial.address_alias`,
   `L1D_PEND_MISS.FB_FULL`, `L1-dcache-load-misses`, `MEM_LOAD_RETIRED.L3_HIT`. *Note: only the
   monitor can run these on the scored node; on wallaby they are available to implementers.*
2. **Keep candidates, don't replace them, and make each candidate's timing licence-clean** (§4.3).
   The r3 verdict reached the first half independently; §4.3 supplies the mechanism for the second.

---

## 6. Target 6 — complex-arithmetic microoptimisation

Notation: **u** is the unit roundoff, u = 2⁻⁵³ for binary64. Bounds below are normwise relative
unless stated. Everything in this section was fetched and read; the ledger is in §7.

### 6.1 The FMA complex multiply is provably optimal at 2u. Stop looking for a better one.

**Source.** Claude-Pierre Jeannerod, Peter Kornerup, Nicolas Louvet, Jean-Michel Muller,
*Error bounds on complex floating-point multiplication with an FMA*, **Mathematics of
Computation 86(304):881–898 (2017)**, DOI 10.1090/mcom/3123 —
https://inria.hal.science/hal-00867040v5/document [VERIFIED — fetched].

Their four algorithms and exact results:

| scheme | form | ops | bound |
|---|---|---|---|
| **A₀** no FMA | `RN(RN(ac)−RN(bd)) + i·RN(RN(ad)+RN(bc))` | 6 | `√5·u`, asymptotically optimal |
| **A₁** naive FMA | `RN(ac−RN(bd)) + i·RN(ad+RN(bc))` | **4** (2 mul + 2 FMA) | **`2u`**, and *sharp* |
| **A₂** Cornea–Harrison–Tang compensated | — | 14 | `2u` |
| **A₃** Kahan compensated per component | — | 8 | `2u` |

The load-bearing sentence, verbatim:

> "The availability of an FMA makes it possible to replace the classical accuracy bound √5u by
> 2u, and this new bound is sharp when the FMA is used in the conventional way, as in algorithm
> A₁. The term 2u **cannot be reduced further** by FMA-based, compensated schemes like
> algorithms A₂ and A₃."

They also prove A₁'s bound is attained: "by constructing inputs a,b,c,d for which
|ẑ₁/z − 1| = 2u − O(u^1.5) as u → 0, that the relative error bound (1.1) is asymptotically
optimal", and it holds for **any** of the four choices of which product to fuse. Compensation
buys only *componentwise* accuracy: "algorithms A₂ and A₃ guarantee a tiny componentwise
relative error, while algorithms A₀ and A₁ can both be highly inaccurate due to possible
catastrophic cancellations in the real or imaginary part."

**Consequence for us.** `check.py` measures a normwise (relative L2) error, and every panel
entry reports 1.3e-16 … 2.6e-16 against a 1e-12 tolerance — four orders of magnitude of
headroom. **A₁ (2 mul + 2 FMA in split layout) is the right answer, it is provably optimal for
the metric we are scored on, and compensated complex multiplication is 2–3.5× the instructions
for exactly zero normwise gain.** This confirms §07 §6.1 and closes it.

Related exact bound, if you ever need a 2×2 determinant (it appears in the Rader kernel's
constant folding and in any `a²+b²`): Claude-Pierre Jeannerod, Nicolas Louvet, Jean-Michel
Muller, *Further analysis of Kahan's algorithm for the accurate computation of 2×2
determinants*, **Mathematics of Computation 82(284):2245–2264 (2013)** —
https://ens-lyon.hal.science/ensl-00649347/document [VERIFIED — fetched]. Kahan's algorithm is
"one multiplication, two independent FMA operations, and one addition"; Theorem 1.2 gives
`|x̂ − x| ⩽ 2u|x|`, and Proposition 6.1 tightens it to `|x̂ − x| ⩽ ulp(x)` when the two products
have opposite signs.

### 6.2 The 3-multiply (Karatsuba/Gauss/3M) complex multiply loses. Two library authors say so in print.

**Operation count.** Oscar Gustafsson, *On Lifting-Based Fixed-Point Complex Multiplications and
Rotations*, ARITH-24 (London, 24–26 July 2017), pp. 43–49 —
https://www.diva-portal.org/smash/get/diva2:1121297/FULLTEXT02 [VERIFIED — fetched] states it
precisely: "three real-valued multiplications are enough at the expense of **five** real-valued
additions, which can be reduced to **three** real-valued additions if the sum and difference of
the real and imaginary values of one of the terms is pre-computed and stored."

So the best case is **3 mul + 3 add = 6 instructions**, versus A₁'s **4**. And the three products
must all exist separately before the final adds, so **nothing can be fused**. It is 50 % more
instructions for a strictly weaker error bound. There is no configuration in which it wins in
our kernels.

**Two authoritative print statements.** Field G. Van Zee, Tyler M. Smith, *Implementing
high-performance complex matrix multiplication via the 3m and 4m methods*, **ACM TOMS (2017)**,
DOI 10.1145/3086466 — https://www.cs.utexas.edu/~flame/pubs/blis5_toms_rev2.pdf
[VERIFIED — fetched]:

> "Note that implementing 3M **at the scalar level forfeits this advantage**… modern computers
> tend to favor computations with a balanced number of MULTIPLY and ADD instructions; thus,
> trading a MULTIPLY for two additional ADD instructions may be counterproductive… In general,
> the 3M method becomes advantageous only when the cost of a multiplication is significantly
> higher than that of an addition."

(At the *matrix* level 3M does win — "nearly 25 % fewer flops", and on a 2×8-core Xeon E5-2680
they measure implementations that "exceed the theoretical peak of the processor core". That is a
GEMM-blocking result and does not transfer to a butterfly.)

Steven G. Johnson, Matteo Frigo, *A Modified Split-Radix FFT With Fewer Arithmetic Operations*,
**IEEE Transactions on Signal Processing 55(1):111–119 (2007)** — https://www.fftw.org/newsplit.pdf
[VERIFIED — fetched]: "**this tradeoff no longer appears to be beneficial on CPUs with hardware
multipliers (and especially those with fused multiply-adders).**"

**The accuracy reason, with the exact counterexample.** Nicholas J. Higham, *Stability of a
method for multiplying complex matrices with three real matrix multiplications*, **SIAM J.
Matrix Anal. Appl. 13(3):681–687 (1992)**, MIMS EPrint 2006.169 —
https://eprints.maths.manchester.ac.uk/348/1/covered/MIMS_ep2006_169.pdf [VERIFIED — fetched].
For `z = (θ + i/θ)² = θ² − 1/θ² + 2i`, the conventional formula gives relative error O(u) in the
imaginary part while 3M computes it as `(θ + 1/θ)² − θ² − 1/θ²` and gets O(uθ²): "If |θ| is
large this formula expresses a number of order 1 as the difference of large numbers." The
structural reason: "there are always subtractions of like-signed numbers in the 3M method", and
"It does not seem possible to improve the stability of the 3M method by 'tinkering' with the
basic formula." **This settles §5 item 6 of `LITERATURE.md` with a primary source.**

### 6.3 The one genuinely promising new idea: the 3-FMA lifting / half-angle rotation

This is the highest-value untried arithmetic idea either research pass turned up.

**Source.** Gustafsson, ARITH-24 2017 (verified above), eq. (12)–(13):

```
R = [1 g][1 0][1 g]        with   g = (c − 1)/d,   c = cos θ,  d = sin θ
    [0 1][d 1][0 1]
```

> "these three matrix multiplications only require **three real-valued multiplications and three
> real-valued additions in total**… Furthermore, the complex multiplication can be realized
> using **three multiply-add operations**"

and the coefficient bound: "Depending on the angle of rotation, a structure can be selected to
yield **|g| ⩽ (1 − cos(π/4))/sin(π/4) ≈ 0.414** and low round-off noise."

**Note that `g = (cos θ − 1)/sin θ = −tan(θ/2)`, so this is the half-angle tangent
representation**, and the 0.414 bound is exactly tan(π/8).

**The code, in split-complex form, with `g` and `d` as compile-time broadcast constants**
*(this instruction sequence and its verification are mine, derived from Gustafsson's
factorisation; the factorisation and the |g| bound are his)*:

```c
/* multiply (xr + i*xi) by exp(i*theta), constants g = -tan(theta/2), d = sin(theta) */
t1 = fma( g, xi, xr );      /* t1 = xr + g*xi                */
t2 = fma( d, t1, xi );      /* t2 = xi + d*t1  = d*xr + c*xi */   /* = out_i */
or = fma( g, t2, t1 );      /* or = t1 + g*t2  = c*xr - d*xi */   /* = out_r */
```

Verified algebraically: `gd = c − 1`, so `t2 = d·xr + (1+gd)·xi = d·xr + c·xi` ✓, and
`or = xr(1+gd) + g·xi(1+c) = c·xr + ((c²−1)/d)·xi = c·xr − d·xi` ✓. Exactly a rotation.

| | instructions | constants stored | dependency depth |
|---|---|---|---|
| A₁ (2 mul + 2 FMA) | **4** | cos, sin | 2 (two independent chains) |
| **lifting / half-angle** | **3** | −tan(θ/2), sin θ | 3 (one serial chain) |

**A 25 % reduction in the instruction count of every non-trivial twiddle multiply**, on ports
that are our binding constraint at B=1 (§4.1). The latency cost — a depth-3 serial chain,
≈12 cycles at 4-cycle FMA latency versus A₁'s ≈8 — is irrelevant to us **because we vectorise
across the batch**: there are thousands of independent chains, so the kernel is
throughput-bound, which is precisely the regime where lifting should win. This is the exact
situation §04 §3.1's `A ⊗ I_ν` argument creates.

**Where it applies in our four geometries.** Nowhere at L=6 or L=36-under-PFA, because
Good–Thomas removes the twiddles entirely, and only once at L=8 (the single 1/√2). **It applies
at L=17, which is the geometry with the most arithmetic**, in both the dense
conjugate-symmetric kernel (which is a sum of constant rotations) and the Rader convolution
kernel; and inside the **9-point module at L=36**, which §02 §5.4 puts at 79 % of PFA-36's
arithmetic. Those are exactly the two places the corpus says a better module could pay.

**Two honest caveats.**
1. **The error status is weaker than A₁'s.** Gustafsson notes the realised transform is
   input-dependent: unlike a direct rotation, which has constant magnitude `√(ĉ²+d̂²)` and
   constant angle, in lifting "both the magnitude and the angle of rotation is input signal
   dependent". With rounded constants `1 + ĝd̂ ≠ ĉ` exactly, so you get an O(u) input-dependent
   magnitude/angle perturbation rather than A₁'s clean sharp 2u. In binary64 with 1e-12 of
   tolerance and 1e-16 of measured error this is very unlikely to matter, but it must be
   *checked*, not assumed — and the harness already checks it.
2. **r2's L=17 experiment is a warning.** The r3 verdict records that an 11.9 % FP-op reduction
   at L=17 bought 0.8 % of time. A 25 % reduction on the *twiddle multiplies only* — not on the
   whole op count — could easily land in the same place. Treat this as a one-hour experiment
   with a clear measurement, not as a round's plan.

(Gustafsson's own motivation, exact invertibility of the quantised transform for lossless
coding, is irrelevant to us: "R R⁻¹ = I holds independent of coefficient quantization errors".
We want it purely for the instruction count.)

### 6.4 The 6-FMA (Linzer–Feig) radix-2 butterfly

**Source.** Elliot Linzer, Ephraim Feig, *Modified FFTs for fused multiply-add architectures*,
**Mathematics of Computation 60(201):347–361 (1993)** —
https://www.ams.org/journals/mcom/1993-60-201/S0025-5718-1993-1159169-0/S0025-5718-1993-1159169-0.pdf
[VERIFIED — fetched]. Theorem 1: `π_new(n) = (8/3)nm − (16/9)n + 2 − (2/9)(−1)^m` multiply-adds
for n = 2^m, and verbatim: "as n tends to infinity the new algorithm uses only **80 %** of the
number of operations used by a regular split-radix algorithm… when n = 1024 the new algorithm
uses **25488** m/a-ops, which is **83.6 %** of the **30496** m/a-ops used by a regular
split-radix FFT."

The explicit butterfly, with `t = ω_r/ω_i = cot θ` (transcribed from a 2026 preprint that
reproduces it, see below):

```
s₁ = b_i − t·b_r      s₂ = t·b_r + b_i
A_r = a_r − s₁·ω_i    A_i = a_i + s₂·ω_i
B_r = a_r + s₁·ω_i    B_i = a_i − s₂·ω_i          -> 6 FMAs
```

against a standard butterfly's 4 multiplies + 6 additions = 10 flops, which with FMA is
**8 instructions** (2 mul + 2 FMA + 4 add/sub). So **6 versus 8 — a 25 % instruction reduction
per radix-2 butterfly** *(the 8-instruction figure is my accounting, not the paper's)*.

Crucially, Linzer & Feig also solve the constant-magnitude problem themselves. Their **Theorem
2**: "All of the real multiplicative constants used in (30) have absolute value at most 1. For
n > 32 the smallest multiplicative constant is larger than the smallest multiplicative constant
used in regular Cooley-Tukey or split-radix algorithms" — obtained by normalising with
`q(S,j) = max_k{max(|Re(S_{j,k})|, |Im(S_{j,k})|)}`, i.e. selecting per twiddle between the
tangent and cotangent forms so the stored ratio never exceeds 1. They also warn against the
alternative: "the Rader-Brenner FFT uses multiplicative constants that are larger than one, and,
for large n, the multiplicative constants are so large as to cause numerical instabilities."

**A caveat about a recent preprint on this.** Mohamed Amine Bergach, *Dual-Select FMA Butterfly
for FFT: Eliminating Twiddle Factor Singularities with Bounded Precomputed Ratios*,
arXiv:2604.00567, 1 April 2026 — https://arxiv.org/abs/2604.00567 [VERIFIED — fetched, abstract
and full text] presents the max(|cos|,|sin|) selection rule as new relative to Linzer–Feig,
reporting for N = 1024 a worst-case ratio reduced "from 163 to 1.0" and "a **235× tighter error
bound in FP16 arithmetic over 10 FFT passes**". Two things to know before using it: (i) the
selection rule appears to be **Linzer & Feig's own Theorem 2 construction**, so the novelty
claim may be a mis-attribution (the *Math. Comp.* paper above says so in print; the preprint
cites their other, IEEE TSP, paper which neither research pass fetched); and (ii) the paper's
own null result kills its accuracy motivation for us — verbatim: "**FP32 precision.** In float32
(ε = 5.96×10⁻⁸), both strategies produce equivalent roundtrip error (∼10⁻⁷ relative L2),
confirming that the advantage is specific to **low-precision arithmetic** where the ratio
amplification exceeds the precision floor." In binary64 the accuracy argument is moot. There are
no wall-clock measurements in the paper. **Take the 6-FMA butterfly for its instruction count;
take the max(|cos|,|sin|) selection because it is free insurance; do not expect an accuracy
change.**

Applicability here: L=8's radix-2/radix-8 stages and the radix-2/radix-4 parts of L=36. At L=6
and under PFA at L=36 there are no twiddles to represent, so it does not apply.

### 6.5 The (1, tan) twiddle: 2 FMAs, if you have somewhere to put the scale

Johnson & Frigo 2007 (verified above), verbatim, with `t_{N,k} = 1 − i·tan(2πk/N)`:

> "Multiplying ω^k_N z_k requires 4 real multiplications and 2 real additions (6 flops) for
> general k, but multiplying t_{N,k} z_k requires only **2 real multiplications and 2 real
> additions (4 flops)**"

and "t_{N,k} is only **1 nontrivial real constant** vs. 2 for ω^k_N". Under FMA accounting this
is better still: `(1 − it)(x + iy) = (x + ty) + i(y − tx)` is **2 FMAs** — half of A₁'s four
instructions — with one stored constant. Their measured flop counts: N=1024, 33968 versus
Yavne's 34824; N=16384, 791264 versus 819208; and savings begin only at **N = 64** for the
unscaled DFT.

**The catch, and it is the whole story:** the `1/cos` factor must be absorbed somewhere. Their
machinery absorbs it recursively — "we can rescale the size-N/4 sub-transforms by any factor
1/s_{N/4,k} desired, and absorb the scale factor into ω^k_N s_{N/4,k} at no cost". At N = 6, 8,
17, 36 that machinery does not apply.

**But it may apply *across our three axes*, which is a structure their paper does not consider.**
A 3D transform makes three passes; a per-line scale introduced in the x pass can in principle be
absorbed into the y or z pass's constants, or into a final scaling the kernel is doing anyway.
That is the actionable version, and it is unproven — flag it as an idea, not a recommendation.

**The accuracy of the tan form is measured and it is fine**, provided you obey one rule. Johnson
& Frigo, Fig. 2 (64-bit double, N = 16…2²⁰, uniform random inputs, Pentium IV/gcc 3.3.5): "our
new FFT has errors within **10 %** of the standard conjugate-pair split-radix algorithm, both
growing roughly as ~√log N", RMS relative error staying in 1.5×10⁻¹⁶–3×10⁻¹⁶. Their stated
reason is the rule: "we never add scaled values with unscaled values, so that whenever standard
split radix computes a + b our new FFT merely computes s·(a + b)." And the counterexample that
defines the boundary: the old "real-factor" FFT used a `csc` twiddle and "proved numerically
ill-behaved… because of the singular csc function—unlike in our algorithm, c_k was not scaled by
any sin function that would cancel the csc singularity, and thus the addition with the unscaled
u_k exacerbates roundoff."

> **Rule: a tan/cot/sec/csc twiddle is safe if and only if the compensating cos/sin scale is
> applied to everything it is subsequently added to.** Break that and you get the real-factor
> FFT's instability.

### 6.6 Twiddle precision: do not touch it in either direction

* **Compensated / double-word twiddles cost ~2× and buy u instead of 2u.** Vincent Lefèvre,
  Jean-Michel Muller, *Accurate Complex Multiplication in Floating-Point Arithmetic*,
  **ARITH-26 (Kyoto, 2019), pp. 1–7** — https://inria.hal.science/hal-02001080/document
  [VERIFIED — fetched]. Their motivation is literally ours: "This is of interest, for instance,
  when that operand is a root of one (which is the case in fast Fourier transforms),
  pre-computed and stored in higher precision than standard floating-point precision."
  Theorem 1 (double-word ω, FP output): `η < u + 33u²`, essentially the best any FP-output
  algorithm can do. Measured, Table I, **Intel Xeon E5-2609 v3**, GCC 8.2.0
  `-march=native -O3 -static`, 2²⁶ operations: Algorithm 3 **0.91–1.02 s** against the naive
  binary64 formula's **0.61–0.62 s** — "about two times as fast as our implementation of
  Algorithm 3, but it is significantly less accurate" — while `__float128` takes **20.90–21.28 s**
  ("from 19 to 25 times as slow" on x86-64) and MPFR 12–24 s.
* **Reduced-precision twiddles buy nothing here.** The only mechanism by which lower twiddle
  precision buys anything is table size, and §07 §5.3 already establishes that all twiddles for
  all four of our sizes total under 3 KiB. There is no table pressure and no recurrence.
* **Error-free FFT machinery costs 2–3 orders of magnitude.** Shota Kawakami, Daisuke Takahashi,
  *Computing FFTs at Target Precision Using Lower-Precision FFTs*, arXiv:2603.29129 (2026) —
  https://arxiv.org/abs/2603.29129 [VERIFIED — fetched, abstract]: an Ozaki-scheme split of the
  Bluestein convolution evaluated exactly by NTT+CRT, on an **Intel Xeon Platinum 8468** for
  n = 2¹⁰–2¹⁸, "the execution time is approximately **107–1315× that of FFTW's double-precision
  FFT**". This independently reproduces the 107–1315× figure §07 §6.3 already cites; it is now
  sourced to a fetchable primary.
* **The baseline growth rates, for the record.** FFTW's own accuracy page —
  https://www.fftw.org/accuracy/comments.html [VERIFIED — fetched]: error "grows as O(log N) in
  the worst case (Gentleman & Sande, 1966) and as O(√log N) on average (Schatzman, 1996)", and
  "**Inaccurate twiddle factors are the most likely reason for the inaccuracy of an FFT
  routine.**" With log₂ 36 ≈ 5.2, our per-axis accumulation factor is ≈2.3. We have four orders
  of magnitude of margin. Spend none of it, and gain nothing by trying to increase it.

### 6.7 Positive constants: 10–15 %, and it is still the cheapest arithmetic win

Johnson & Frigo, *Implementing FFTs in Practice* (ch. 11 of *Fast Fourier Transforms*, ed.
C. S. Burrus; the arXiv posting arXiv:2602.23525 was fetched, and the MIT copy
https://math.mit.edu/~stevenj/papers/JohnsonFr08-burrus.pdf independently) [VERIFIED — fetched]:

> "multiplicative constants in FFT algorithms often come in positive/negative pairs, but every C
> compiler we are aware of will generate separate load instructions for positive and negative
> versions of the same constants… We thus obtained a **10–15 % speedup** by making all constants
> positive, which involves propagating minus signs to change additions into subtractions or vice
> versa elsewhere in the dag."

This corroborates §01 §1.2 from a second fetched source. On AVX-512 there is an additional
reason to care: a positive-canonicalised constant set is smaller, so more of it fits in the
`{1toN}` embedded-broadcast operands of §4.1 item 4, and none of it needs a register.

### 6.8 x86 has no double-precision complex-arithmetic instruction, and split layout means you are not missing one

* **What exists.** `VFMADDSUB132/213/231PD` and the `VFMSUBADD` counterparts — "adds the odd
  double precision floating-point elements and subtracts the even double precision floating-point
  values" (https://www.felixcloutier.com/x86/vfmaddsub132pd:vfmaddsub213pd:vfmaddsub231pd
  [VERIFIED — fetched]), CPUID `FMA` (VEX) or `AVX512F`+`AVX512VL` (EVEX). Its rounding pattern
  is A₁'s, so it inherits the sharp 2u bound.
* **What does not exist.** The AVX512-FP16 complex family — `VFMULCPH`, `VFCMULCPH`,
  `VFMADDCPH`, `VFCMADDCPH` (https://www.felixcloutier.com/x86/vfcmulcph:vfmulcph
  [VERIFIED — fetched]) — is **FP16 only**. There is no FP32 or FP64 complex-arithmetic
  instruction in any AVX-512 or AVX10 subset. Note also that the FP16 ones do *not* avoid
  intermediate rounding: "Rounding is performed at every FMA (fused multiply and add) boundary."
* **The contrast, and why it is not a loss.** Arm SVE `FCMLA (vectors)` does support double
  precision and multiplies "without intermediate rounding", and "these transformations permit
  the creation of a variety of multiply-add and multiply-subtract operations on complex numbers
  by combining **two** of these instructions" (https://www.scs.stanford.edu/~zyedidia/arm64/fcmla_z_p_zzz.html
  [VERIFIED — fetched]). Count it out: 2 FCMLA per 4 complex doubles in a 512-bit vector = **0.5
  instructions per complex FMA**. Our split-complex batched form is 4 FMAs per 8 complex FMAs =
  **also 0.5 instructions per complex FMA** *(this comparison is mine)*. **FCMLA's advantages are
  that it works on interleaved data with zero shuffles and that it avoids one rounding — not
  throughput.** Split-complex batching on x86 already matches dedicated complex-FMA hardware in
  instruction count, which is a strong independent justification for §04's layout verdict, and
  it explains why Intel never added FP64 complex instructions.

FFTW's authors state the same layout conclusion, from Frigo & Johnson, *The Design and
Implementation of FFTW3*, **Proceedings of the IEEE 93(2):216–231 (2005)** —
https://www.fftw.org/fftw-paper-ieee.pdf [VERIFIED — fetched]: "We view a complex DFT as a pair
of real DFTs: DFT(A + i·B) = DFT(A) + i·DFT(B)… Our algorithm computes the two real DFTs in
parallel using SIMD instructions", and "SIMD instructions are better seen as a restricted form
of instruction-level parallelism than as a degenerate flavor of vector parallelism."

### 6.9 Summary table — every twiddle-multiply form, ranked

*(Instruction counts and dependency depths are my accounting; bounds and op counts are sourced
above.)*

| form | constants stored | instructions | dep. depth | leftover scale | error status |
|---|---|---|---|---|---|
| A₀, no FMA | cos, sin | 6 | 2 | none | `√5·u`, sharp |
| **A₁ = 2 mul + 2 FMA** | cos, sin | **4** | 2 | none | **`2u`, sharp, optimal** |
| **lifting / half-angle (§6.3)** | −tan(θ/2), sin θ | **3** | 3 | none | O(u), input-dependent |
| (1, tan) (§6.5) | tan θ | **2** | 2 | **1/cos θ** | within 10 % of split-radix (measured) |
| Kahan A₃ | cos, sin | 8 | 3 | none | `2u` — no normwise gain |
| CHT A₂ | cos, sin | 14 | 4 | none | `2u` — no normwise gain |
| double-word (Lefèvre–Muller Alg. 3) | 4 words | ~18 ops, ~2× measured | deep | none | `u + 33u²` |

**Do:** A₁ everywhere by default; try the 3-FMA lifting form at L=17 and in L=36's 9-point
module; canonicalise every constant positive; keep every constant a compile-time correctly
rounded binary64 literal and let AVX-512 embed it as a broadcast memory operand.
**Do not:** 3M/Karatsuba, compensated complex multiply, double-word twiddles, reduced-precision
twiddles, or any error-free-transformation machinery.

---

## 7. Unsourced engineering notes (my own analysis, attributed to nobody)

Everything above is sourced. These are inferences, arithmetic and suggestions that no source
supports directly. They are separated so nobody cites them as literature.

1. **The panel's 12.3 GB/s and the published 11.5 GB/s are the same number.** The r3 verdict's
   "fastest single-core stream on the board" is, to within the difference between a read-only
   microbenchmark and a read+write kernel, exactly what Velten et al. and Alappat et al. measure
   for one core of this microarchitecture. That is a strong validation of the panel's measurement
   discipline and a hard ceiling on the large-batch cells. It is also the reason §1.12's table shows
   two cells *beating* the ceiling — which can only be L3 hits.

2. **Machine balance for this node, and the arithmetic intensity of all four geometries.**
   Peak is 16 DP flops/cycle (one 512-bit FMA unit, §4.2) × 2.9 GHz (§4.1) = **46.4 GFlop/s**.
   Against the single-core bandwidths of §1.2 that gives a machine balance of
   **46.4 / 11.5 = 4.03 flops/byte at DRAM**, **46.4 / 18.2 = 2.55 at L3**, and
   **46.4 / 87.3 = 0.53 at L2**.

   Our arithmetic intensity, in the same nominal-flop convention the corpus and the leaderboard
   use (5·N·log₂N flops for N = L³ points, against 2 × 16N bytes of compulsory out-of-place
   traffic, i.e. `AI = 5·log₂N / 32`):

   | | L=6 (N=216) | L=8 (N=512) | L=17 (N=4913) | L=36 (N=46656) |
   |---|---|---|---|---|
   | AI, flops/byte | **1.21** | **1.41** | **1.92** | **2.42** |

   *(The L=36 figure reproduces §05 §5.4's "one fused pass gets to 2.42" exactly, which is a
   useful consistency check on both.)*

   **Every geometry sits below the DRAM balance of 4.03 and above the L2 balance of 0.53.** That
   is this whole document in two lines: streaming from DRAM, we are memory-bound and no arithmetic
   will help; holding the working set in L2, we are compute-bound and the arithmetic is all that
   matters. It also means the tile size in §1.9 is not an optimisation — **it is the parameter that
   decides which regime you are in.** And it says something sharper about L=36: at AI 2.42 against
   an L3 balance of 2.55, an L=36 volume that is *L3*-resident is almost exactly at the balance
   point, i.e. neither clearly bound — which is precisely the marginal regime where the r3 verdict
   found prefetch decisions inverting between wallaby and the node.

3. **The prefetch distances in §1.6 are derived, not published.** They use Velten et al.'s 80 ns
   DRAM latency, the panel's own measured times, and Mowry's formula. If the clock is 2.9 GHz rather
   than 2.3 (§4.1), l ≈ 232 cycles rather than 184 and every D rises by 26 % — so ~20 lines rather
   than ~16. The point is the order of magnitude: **one kibibyte ahead, not one volume ahead.**

4. **The 4 KiB-alias exposure table in §1.8 is arithmetic on our volume sizes, not a measurement.**
   It assumes `out − in ≡ 0 (mod 4096)`, which is likely but not certain — `aligned_alloc` of tens
   of megabytes almost always comes from `mmap` and is page-aligned, but a small allocator offset
   would break the alignment and the whole concern with it. **Print `((uintptr_t)out - (uintptr_t)in) % 4096`
   from `create()` once and the question is settled in one line of output.**

5. **The "re-read the X-first reorder as an alias fix" hypothesis (§1.8) is mine and may be wrong.**
   The competing explanation — that spreading the writes raises store-buffer and memory-level
   parallelism — is equally consistent with the 10.8 % and with §1.1's concurrency framing. The
   counter distinguishes them: if `ld_blocks_partial.address_alias` drops sharply between the two
   variants, it was aliasing; if it does not move but `L1D_PEND_MISS.PENDING` rises, it was
   concurrency. Either way the technique works and should be applied at the other three geometries.

6. **512-bit vs 256-bit on this SKU: the conclusion is derived from two documents, not measured.**
   §4.2's per-cycle table combines Microway's and `vpu-count`'s FMA-unit count with Agner's port
   description and Intel's turbo table. Every input is verified; the composition is mine. The r3
   verdict's L=8 floor independently assumed the same 1-FMA figure and landed within 5 % of
   measurement, which is corroboration but not proof. **`avx-turbo` on the node would make all of
   §4 measured rather than inferred, and it is one job.**

7. **The three-GEMM (μ-mode) idea at L=17 is untested by anyone, including me.** §2.3 gives the
   shape arithmetic and the libxsmm prior; it does not give a reason to believe it beats a
   batch-vectorised Rader or dense-symmetric kernel. Treat it as the one genuinely different
   structure worth a round, not as a recommendation. Note in particular that our kernels already
   *are* GEMM-shaped at the innermost level; what a GEMM formulation would add is a different
   blocking of the same arithmetic, and per §1.9 the blocking is where the money is.

8. **A speculation worth exactly one experiment at L=36.** §1.3's victim-L3 result says CLX retains
   reuse at 4–10× L3. §1.9 says tile the batch into L2. At L=36 one volume (746 KiB) already fills
   three quarters of L2, so batch tiling is impossible and the tile must be *within* the volume —
   e.g. a 36×36×8 slab (166 KiB) or a 36×8-pencil group. That means L=36 is the one geometry where
   the corpus's plane/pencil blocking advice and this section's batch-tiling advice point at
   different constructions, and where a two-level scheme (pencil groups inside L2, whole volumes
   relying on the victim L3) might be the answer. Nothing in the literature covers this.

9. **On the honest status of the 2026 preprints cited in §6.** arXiv:2604.00567 and arXiv:2603.29129
   were fetched and read in this session, but they are preprints, one of them appears to
   mis-attribute its central construction (§6.4), and neither has wall-clock CPU measurements. The
   *classical* sources behind the same techniques — Linzer & Feig 1993, Gustafsson 2017,
   Jeannerod et al. 2013/2017, Johnson & Frigo 2007 — are peer-reviewed and are what the
   recommendations rest on.

10. **What I would drop from the corpus's Tier-4 "do not do" list, and what I would add.** Drop
    nothing — every item survives. **Add:** do not build a runtime tuner that races candidates of
    mixed SIMD width without per-candidate warm-up (§4.3); do not prefetch a whole volume ahead
    (§1.6); do not aim a blocking scheme at L3 (§1.4); do not put literal 0.0/1.0 in the same
    constant vector as real twiddles, because it defeats constant folding (§5.4(ii)); and do not add
    `-ffast-math` without a ±0 bit-exactness check (§5.6).

---

## 8. Citation ledger

**Method.** Every source below was fetched in this session — by me directly, or by one of four
parallel research passes working under the same rule: *only cite a URL you actually fetched, quote
numbers verbatim, name the machine.* Where a PDF's text could not be decoded by the fetch tool it
was retrieved with `curl` and extracted locally with `pypdf`; those are marked *(pypdf)*. Sources
that could not be fetched are listed separately and are **not** used to support any claim in this
section.

### 8.1 Verified — fetched and read in this session

**Cascade Lake / Skylake-SP memory system and microarchitecture (§1, §4)**

1. Christie L. Alappat, Johannes Hofmann, Georg Hager, Holger Fehske, Alan R. Bishop, Gerhard
   Wellein, *Understanding HPC Benchmark Performance on Intel Broadwell and Cascade Lake
   Processors*, ISC High Performance 2020, arXiv:2002.03344 — https://arxiv.org/abs/2002.03344,
   https://arxiv.org/pdf/2002.03344 *(pypdf)*. Xeon Gold 6248.
2. Markus Velten, Robert Schöne, Thomas Ilsche, Daniel Hackenberg, *Memory Performance of AMD EPYC
   Rome and Intel Cascade Lake SP Server Processors*, ICPE '22, arXiv:2204.03290 —
   https://arxiv.org/pdf/2204.03290 *(pypdf)*. Xeon Gold 6248.
3. Intel Corporation, *Intel 64 and IA-32 Architectures Optimization Reference Manual, Volume 1*,
   doc 248966-049US — https://cdrdv2-public.intel.com/814198/248966-Optimization-Reference-Manual-V1-049.pdf
   *(pypdf)*; and rev 048 — https://cdrdv2-public.intel.com/671488/248966-Software-Optimization-Manual-V1-048.pdf
   *(pypdf)*. §2.1, §2.3, §2.4.1.6, §2.5, §2.5.1.3, §2.5.2, §2.5.3 Table 2-10, Table 2-9,
   Figure 2-7, §3.6.11, §18.2.6, §18.9.1, §18.9.2, §18.15.7.
4. Intel Corporation, *2nd Gen Intel Xeon Scalable Processors Specification Update*,
   doc 338848-028US, October 2023 —
   https://cdrdv2-public.intel.com/338848/338848_2nd%20Gen%20Intel%C2%AE%20Xeon%C2%AE%20Scalable%20Processors%20Specification%20Update_Rev028US.pdf
   *(pypdf)*. **Figures 1–6 carry the Xeon Gold 5218 turbo tables used in §4.1.**
5. Intel Corporation, *4th Gen Intel Xeon Scalable Processor (Sapphire Rapids Edge Enhanced)
   Specification Update*, doc 784461-002 —
   https://cdrdv2-public.intel.com/784461/784461_SapphireRapidsEE_SpecificationUpdate_002.pdf
   *(pypdf)*. Cited for the **absence** of AVX turbo figures.
6. Agner Fog, *The microarchitecture of Intel, AMD and VIA CPUs* —
   https://www.agner.org/optimize/microarchitecture.pdf *(pypdf)*. Ch. 11 (Skylake/Cascade Lake),
   ch. 12 (Ice Lake).
7. Roland Kühn, Jan Mühlig, Jens Teubner, *How to Be Fast and Not Furious: Looking Under the Hood
   of CPU Cache Prefetching*, DaMoN '24 —
   https://dbis.cs.tu-dortmund.de/storages/dbis-cs/r/papers/2024/sw-prefetching-survey/sw-prefetching.pdf
   *(pypdf)*. **Xeon Gold 6226, Cascade Lake: exactly 10 LFB slots.**
8. Jan Laukemann, Thomas Gruber, Georg Hager, Dossay Oryspayev, Gerhard Wellein, *CloverLeaf on
   Intel Multi-Core CPUs: A Case Study in Write-Allocate Evasion*, arXiv:2311.04797 —
   https://arxiv.org/pdf/2311.04797 *(pypdf)*.
9. Jan Laukemann, Georg Hager, Gerhard Wellein, *Microarchitectural comparison and in-core modeling
   of state-of-the-art CPUs: Grace, Sapphire Rapids, and Genoa*, arXiv:2409.08108 —
   https://arxiv.org/pdf/2409.08108 *(pypdf)*.
10. Johannes Hofmann, Dietmar Fey, Jan Eitzinger, Georg Hager, Gerhard Wellein, *Analysis of Intel's
    Haswell Microarchitecture Using The ECM Model and Microbenchmarks*, arXiv:1511.03639 —
    https://arxiv.org/pdf/1511.03639 *(pypdf)*.
11. Johannes Hofmann, Dietmar Fey, *An ECM-based energy-efficiency optimization approach for
    bandwidth-limited streaming kernels on recent Intel Xeon processors*, arXiv:1609.03347 —
    https://arxiv.org/pdf/1609.03347 *(pypdf)*.
12. Johannes Hofmann, Jan Eitzinger, Dietmar Fey, *Execution-Cache-Memory Performance Model:
    Introduction and Validation*, arXiv:1509.03118 — https://arxiv.org/abs/1509.03118
    (abstract/metadata only).
13. John D. McCalpin, *Notes on "non-temporal" (aka "streaming") stores*, 1 January 2018 — original
    URL `sites.utexas.edu/jdm4372/2018/01/01/notes-on-non-temporal-aka-streaming-stores/` now
    retired; retrieved via `web.archive.org`.
14. John D. McCalpin, *The evolution of single-core bandwidth in multicore processors*, 25 April
    2023, and *…update*, 19 December 2023 — both via `web.archive.org`.
15. Georg Hager, *The McCalpin STREAM benchmark: How to do it right and interpret the results*,
    9 March 2019 — https://blogs.fau.de/hager/archives/8263; *Write-allocate evasion has finally
    arrived at Intel – or has it?*, 27 October 2021 — https://blogs.fau.de/hager/archives/8997;
    *A case for the non-temporal store*, 4 September 2008 — https://blogs.fau.de/hager/archives/2103.
16. Ioan Hadade, Timothy M. Jones, Feng Wang, Luca di Mare, *Software Prefetching for Unstructured
    Mesh Applications*, ACM TOPC 7(1) art. 3, March 2020 —
    https://api.repository.cam.ac.uk/server/api/core/bitstreams/9dd57510-1512-4ef5-9936-0bfdd47a9f38/content
    *(pypdf)*. **Xeon Gold 6152: 1.29× single-core from software prefetch.**
17. Jaekyu Lee, Hyesoon Kim, Richard Vuduc, *When Prefetching Works, When It Doesn't, and Why*,
    ACM TACO 9(1) art. 2, March 2012 — https://vuduc.org/pubs/lee2012-taco.pdf and
    https://faculty.cc.gatech.edu/~hyesoon/lee_taco12.pdf *(pypdf)*.
18. Daniel Lemire, *Memory-level parallelism: Intel Skylake versus Intel Cannonlake*, 13 January
    2019 — https://lemire.me/blog/2019/01/01/memory-level-parallelism-intel-skylake-versus-intel-cannonlake/
19. Daniel Lemire, *Estimating your memory bandwidth*, 13 January 2024 —
    https://lemire.me/blog/2024/01/13/estimating-your-memory-bandwidth/
20. Daniel Lemire, *AVX-512: when and how to use these new instructions*, 7 September 2018 —
    https://lemire.me/blog/2018/09/07/avx-512-when-and-how-to-use-these-new-instructions/ ; and
    *AVX-512 throttling: heavy instructions are maybe not so dangerous*, 25 August 2018 —
    https://lemire.me/blog/2018/08/25/avx-512-throttling-heavy-instructions-are-maybe-not-so-dangerous/
21. Travis Downs, *Gathering Intel on Intel AVX-512 Transitions*, 17 January 2020 —
    https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html
22. Travis Downs, *Ice Lake AVX-512 Downclocking*, 19 August 2020 —
    https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html
23. Travis Downs, *AVX-512 Mask Registers, Again*, 26 May 2020 —
    https://travisdowns.github.io/blog/2020/05/26/kreg2.html
24. Travis Downs, *Dirty upper 256 causes everything to run at AVX-512 frequencies*, Real World
    Tech forum, 25 August 2018 — https://www.realworldtech.com/forum/?threadid=179700&curpostid=179700
25. Travis Downs, `avx-turbo` — https://github.com/travisdowns/avx-turbo
26. Robert Schöne, Thomas Ilsche, Mario Bielert, Andreas Gocht, Daniel Hackenberg, *Energy Efficiency
    Features of the Intel Skylake-SP Processor and Their Impact on Performance*, HPCS 2019,
    arXiv:1905.12468 — https://arxiv.org/pdf/1905.12468 *(pypdf)*.
27. Mathias Gottschlag, Peter Brantsch, Frank Bellosa, *Automatic Core Specialization for AVX-512
    Applications*, SYSTOR '20 —
    https://os.itec.kit.edu/downloads/Automatic_Core_Specialiszation_for_AVX_512_Application.pdf
    *(pypdf)*; project page https://os.itec.kit.edu/3616_3604.php
28. Vlad Krasnov (Cloudflare), *On the dangers of Intel's frequency scaling*, 10 November 2017 —
    https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/
29. Microway, *Detailed Specifications of the "Cascade Lake SP" Intel Xeon Processor Scalable Family
    CPUs*, 2 April 2019 —
    https://www.microway.com/knowledge-center-articles/detailed-specifications-of-the-cascade-lake-sp-intel-xeon-processor-scalable-family-cpus/
    ; and the Skylake-SP equivalent, 11 July 2017.
30. Jeff Hammond, `vpu-count` — https://github.com/jeffhammond/vpu-count
31. Antonio Gómez-Iglesias, Feng Chen, Lei Huang, Hang Liu, Si Liu, Carlos Rosales, *Benchmarking the
    Intel Xeon Platinum 8160 Processor*, TACC Technical Report **TR-17-01**, 10 August 2017 —
    https://repositories.lib.utexas.edu/bitstream/handle/2152/61472/SKX_Benchmarking.pdf?sequence=2
    *(pypdf)*. Cited for the FLASH 64³ finding that "when not -no-vec is specified, -xCORE-AVX2
    clearly outperforms -xCORE-AVX512", and for 2 AVX-512 FMA units on the 8160.
32. Dean Sullivan, Orlando Arias, Travis Meade, Yier Jin, *Microarchitectural Minefields:
    4K-Aliasing Covert Channel and Multi-Tenant Detection in IaaS Clouds*, NDSS 2018 —
    https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_06A-3_Sullivan_paper.pdf
    *(pypdf)*.
33. Jakub Beránek, `hardware-effects/4k-aliasing` —
    https://github.com/Kobzol/hardware-effects/blob/master/4k-aliasing/README.md
34. `trackreco/mkFit` issue #170, *no difference in computational performance between AVX-512 and
    AVX2?* — https://github.com/trackreco/mkFit/issues/170
35. twest820, *AVX-512: documentation beyond what Intel provides* — https://github.com/twest820/AVX-512
36. Cornell Virtual Workshop, *Turbo Boost* — https://cvw.cac.cornell.edu/vector/performance/performance-turbo
    ; *Enabling vectorization* — https://cvw.cac.cornell.edu/vector/compilers/enabling-vector
37. Wikipedia, *AVX-512* — https://en.wikipedia.org/wiki/AVX-512 ; *List of Intel Xeon processors
    (Cascade Lake-based)* — https://en.wikipedia.org/wiki/List_of_Intel_Xeon_processors_(Cascade_Lake-based)
38. *First Impressions of the Sapphire Rapids Processor with HBM for Scientific Workloads*,
    SN Computer Science, 2024, DOI 10.1007/s42979-024-02958-3 —
    https://link.springer.com/article/10.1007/s42979-024-02958-3
39. Ayesha Afzal, Georg Hager, Gerhard Wellein, *SPEChpc 2021 Benchmarks on Ice Lake and Sapphire
    Rapids Infiniband Clusters*, PMBS23/SC'23, arXiv:2309.05373 — https://arxiv.org/pdf/2309.05373
    *(pypdf)*.
40. Chester Lam, *Sapphire Rapids: Golden Cove Hits Servers*, Chips and Cheese, 12 March 2023 —
    https://chipsandcheese.com/p/a-peek-at-sapphire-rapids
41. LLVM review **D67259**, *[X86] Enable -mprefer-vector-width=256 by default for Skylake-avx512 and
    later Intel CPUs* — https://reviews.llvm.org/D67259 ; LLVM issue **#102047** —
    https://github.com/llvm/llvm-project/issues/102047
42. H. J. Lu, glibc commit *x86: Use AVX2 memcpy/memset on Skylake server [BZ #21396]* —
    https://git.zx2c4.com/glibc/commit/?id=4cb334c4d6249686653137ec273d081371b3672d ; Andy
    Lutomirski, *Why does glibc use AVX-512?*, libc-alpha 2021-03-26 —
    https://sourceware.org/pipermail/libc-alpha/2021-March/124414.html ; Undo, *AVX-512 recording
    portability* — https://docs.undo.io/GlibcAVX512.html ; GCC *x86 Options* —
    https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html

**Batched small transforms, layout, fusion, roofline (§1, §2, §3, §5)**

43. Doru Thom Popovici, Francis P. Russell, Karl Wilkinson, Chris-Kriton Skylaris, Paul H. J. Kelly,
    Franz Franchetti, *Generating Optimized Fourier Interpolation Routines for Density Functional
    Theory using SPIRAL*, IPDPS 2015, 743–752 — https://users.ece.cmu.edu/~franzf/papers/ipdps15.pdf
    *(pypdf)*. **The closest published measurement to our regime; 2–3× over FFTW and MKL.**
44. Doru T. Popovici, Franz Franchetti, Tze Meng Low, *Mixed Data Layout Kernels for Vectorized
    Complex Arithmetic*, IEEE HPEC 2017 —
    https://aiichironakano.github.io/cs653/Popovici-ComplexSIMD-HPEC17.pdf *(pypdf)*.
45. Doru Thom Popovici, Tze Meng Low, Franz Franchetti, *Large Bandwidth-Efficient FFTs on Multicore
    and Multi-Socket Systems*, IPDPS 2018 — https://users.ece.cmu.edu/~franzf/papers/ipdps2018_dtp.pdf
46. Doru Thom Popovici, Martin D. Schatz, Franz Franchetti, Tze Meng Low, *A Flexible Framework for
    Parallel Multi-Dimensional DFTs*, arXiv:1904.10119 — https://arxiv.org/pdf/1904.10119
47. Franz Franchetti, Tze Meng Low, Doru Thom Popovici, Richard M. Veras, Daniele G. Spampinato,
    Jeremy R. Johnson, Markus Püschel, James C. Hoe, José M. F. Moura, *SPIRAL: Extreme Performance
    Portability*, Proceedings of the IEEE 106(11):1935–1968, 2018 —
    https://www.spiral.net/doc/papers/SPIRAL_IEEE_2018.pdf *(pypdf)*.
48. Franz Franchetti et al., *FFTX and SpectralPack: A First Look*, HiPC 2018 —
    https://users.ece.cmu.edu/~franzf/papers/hipc_2018.pdf *(pypdf)* ; FFTX project page
    https://www.spiral.net/software/fftx.html ; FFTX README
    https://raw.githubusercontent.com/spiral-software/fftx/master/README.md ; SPIRAL publication
    index https://spiral.ece.cmu.edu/pub-spiral/allabstract.jsp
49. Daniel S. McFarlin, Volodymyr Arbatov, Franz Franchetti, Markus Püschel, *Automatic SIMD
    Vectorization of Fast Fourier Transforms for the Larrabee and AVX Instruction Sets*, ICS 2011 —
    https://users.ece.cmu.edu/~franzf/papers/ics2011.pdf *(pypdf)*.
50. Daisuke Takahashi, Franz Franchetti, *FFTE on SVE: SPIRAL-Generated Kernels*, HPCAsia 2020 —
    https://www.spiral.net/doc/papers/ffte-spiral.pdf *(pypdf)*.
51. Aleksandar Zlateski, Zhen Jia, Kai Li, Fredo Durand, *FFT Convolutions are Faster than Winograd
    on Modern CPUs, Here's Why*, arXiv:1809.07851 — https://arxiv.org/pdf/1809.07851 *(pypdf)* and
    https://ar5iv.labs.arxiv.org/html/1809.07851 ; and *A Deeper Look at FFT and Winograd
    Convolutions*, SysML 2018 — https://mlsys.org/Conferences/doc/2018/28.pdf
52. Rati Gelashvili, Nir Shavit, Aleksandar Zlateski, *L3 Fusion: Fast Transformed Convolutions on
    CPUs*, arXiv:1912.02165 — https://arxiv.org/pdf/1912.02165 *(pypdf)*.
53. Fabian Giesen, *Notes on FFTs: for implementers*, 19 March 2023 —
    https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/
54. Vedran Novaković, *Batched computation of the singular value decompositions of order two by the
    AVX-512 vectorization*, Parallel Processing Letters 2020, arXiv:2005.07403 —
    https://arxiv.org/pdf/2005.07403 *(pypdf)*.
55. N. Malapally et al., *Scalability of 3D-DFT by block tensor-matrix multiplication on the JUWELS
    Cluster*, arXiv:2303.13337 — https://arxiv.org/pdf/2303.13337 *(pypdf)*.
56. Arjun Ramaswami, Tobias Kenter, Thomas D. Kühne, Christian Plessl, *Efficient Ab-Initio
    Molecular Dynamic Simulations by Offloading Fast Fourier Transformations to FPGAs*,
    arXiv:2006.08435 — https://arxiv.org/pdf/2006.08435 *(pypdf)*.
57. *Breaking Down the Parallel Performance of GROMACS*, arXiv:2208.13658 —
    https://arxiv.org/pdf/2208.13658 *(pypdf)*.
58. *Intel Cilk Plus for Complex Parallel Algorithms: "Enormous Fast Fourier Transform" (EFFT)
    Library*, arXiv:1409.5757 — https://arxiv.org/pdf/1409.5757 *(pypdf)*.
59. Berkin Akın, Peter A. Milder, Franz Franchetti, James C. Hoe, *Memory Bandwidth Efficient
    Two-Dimensional FFT…*, FCCM 2012 — https://users.ece.cmu.edu/~franzf/papers/akin-fccm12.pdf
    *(FPGA)*.
60. FFTW documentation, *Advanced Complex DFTs* — https://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html
    ; *Release Notes* — https://www.fftw.org/release-notes.html ; *Installation on Unix* —
    https://www.fftw.org/fftw3_doc/Installation-on-Unix.html
61. Marcin Wojdyr, `project-gemmi/benchmarking-fft` —
    https://raw.githubusercontent.com/project-gemmi/benchmarking-fft/master/README.md

**Beating row-column, kernel selection, small-matrix libraries (§2, §3)**

62. Nicolas Venkovic, Hartwig Anzt, *Permutation-Avoiding FFT-Based Convolution*, arXiv:2506.12718v3
    (2026) — https://arxiv.org/html/2506.12718v3, https://arxiv.org/pdf/2506.12718
63. Keun-Yung Byun, Chun-Su Park, Jee-Young Sun, Sung-Jea Ko, *Vector Radix 2 × 2 Sliding Fast
    Fourier Transform*, Mathematical Problems in Engineering, 2016 —
    https://onlinelibrary.wiley.com/doi/10.1155/2016/2416286
64. Ryszard M. Stasiński, *Fast Discrete Fourier Transform algorithms requiring less than O(N log N)
    multiplications*, arXiv:2303.02647 — https://arxiv.org/html/2303.02647
65. Marco Caliari, Fabio Cassini, Franco Zivcovich, *A μ-mode BLAS approach for multidimensional
    tensor-structured problems*, arXiv:2112.11238 — https://ar5iv.labs.arxiv.org/html/2112.11238
66. Marco Caliari, Fabio Cassini, Lukas Einkemmer, Alexander Ostermann, Franco Zivcovich, *A μ-mode
    integrator for solving evolution equations in Kronecker form*, arXiv:2103.01691 —
    https://ar5iv.labs.arxiv.org/html/2103.01691
67. Paul Springer, Paolo Bientinesi, *Design of a High-Performance GEMM-like Tensor–Tensor
    Multiplication* (GETT/TCCG), arXiv:1607.00145 — https://arxiv.org/abs/1607.00145
68. Alexander Heinecke, Hans Pabst, Greg Henry, *LIBXSMM: A High Performance Library for Small Matrix
    Multiplications*, SC15 poster —
    https://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file2.pdf
    *(pypdf)* ; libxsmm documentation https://libxsmm.readthedocs.io/en/latest/ and
    https://github.com/libxsmm/libxsmm/wiki/Q&A
69. Gianluca Frison, Dimitris Kouzoupis, Tommaso Sartor, Andrea Zanelli, Moritz Diehl, *BLASFEO*,
    ACM TOMS 44(4) 2018, arXiv:1704.02457 — https://arxiv.org/pdf/1704.02457 *(pypdf)*.
70. Jianyu Yao, Boqian Shi, Chunyang Xiang, Haipeng Jia, Chendi Li, Hang Cao, Yunquan Zhang, *IAAT:
    A Input-Aware Adaptive Tuning framework for Small GEMM*, arXiv:2208.09822 —
    https://arxiv.org/pdf/2208.09822 *(pypdf)*.
71. *The Design and Performance of Batched BLAS on Modern High-Performance Computing Systems*,
    ICCS 2017, Procedia Computer Science 108C:495–504 —
    https://www.netlib.org/utk/people/JackDongarra/PAPERS/batched-blas-iccs17.pdf *(pypdf)*.
    Authorship as recorded in §3.3.
72. Sameer Deshmukh, Rio Yokota, George Bosilca, *Cache Optimization and Performance Modeling of
    Batched, Small, and Rectangular Matrix Multiplication…*, arXiv:2311.07602 —
    https://arxiv.org/html/2311.07602v1
73. Ahmad Abdelfattah, Stanimire Tomov, Jack Dongarra, *Matrix multiplication on batches of small
    matrices in half and half-complex precisions*, JPDC 145 (2020) 188–201 —
    https://icl.utk.edu/files/publications/2020/icl-utk-1411-2020.pdf *(GPU)*.
74. Alexander O. Korotkevich, *A new set of efficient SMP-parallel 2D Fourier subroutines*,
    arXiv:2008.07031 — https://ar5iv.labs.arxiv.org/html/2008.07031
75. Stephen Fegan, `dft_simd` — https://github.com/sfegan/dft_simd and
    https://raw.githubusercontent.com/sfegan/dft_simd/master/README.md
76. Maron Schlemon, Jamin Naghmouchi, *FFT optimizations and performance assessment targeted towards
    satellite and airborne radar processing* (DLR) — https://elib.dlr.de/140530/2/conference_101719.pdf
    *(pypdf)*.
77. James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontañón, *FNet: Mixing Tokens with
    Fourier Transforms*, arXiv:2105.03824 — https://arxiv.org/pdf/2105.03824 *(GPU/TPU)*.
78. Tianjian Lu, Yi-Fan Chen, Blake Hechtman, Tao Wang, John Anderson, *Large-Scale Discrete Fourier
    Transform on TPUs*, arXiv:2002.03260 — https://arxiv.org/html/2002.03260 *(TPU)*.
79. Zixuan Jiang, Jiaqi Gu, David Z. Pan, *A New Acceleration Paradigm for Discrete Cosine Transform
    and Other Fourier-Related Transforms*, arXiv:2110.01172 —
    https://ar5iv.labs.arxiv.org/html/2110.01172 *(GPU)*.
80. Daniele G. Spampinato, Diego Fabregat-Traver, Paolo Bientinesi, Markus Püschel, *Program
    Generation for Small-Scale Linear Algebra Applications*, CGO 2018, arXiv:1805.04775 —
    https://ar5iv.labs.arxiv.org/html/1805.04775
81. Evangelos Georganas et al., *Anatomy Of High-Performance Deep Learning Convolutions On SIMD
    Architectures*, SC18, arXiv:1808.05567 — https://ar5iv.labs.arxiv.org/html/1808.05567
82. Doru Thom Popovici, Mauro Del Ben, Osni Marques, Andrew Canning, *Flexible Multi-Dimensional FFTs
    for Plane Wave Density Functional Theory Codes*, arXiv:2406.05577v1 —
    https://arxiv.org/html/2406.05577v1 *(GPU/distributed)*.
83. Pascal Giorgi, *On Polynomial Multiplication in Chebyshev Basis*, arXiv:1009.4597v2 —
    https://arxiv.org/pdf/1009.4597

**Code generation, autotuning, library design (§5)**

84. Yuka Ikarashi, Kevin Qian, Samir Droubi, Alex Reinking, Gilbert Louis Bernstein, Jonathan
    Ragan-Kelley, *Exo 2: Growing a Scheduling Language*, ASPLOS '25, arXiv:2411.07211 —
    https://arxiv.org/pdf/2411.07211, https://arxiv.org/abs/2411.07211 ; Exo README
    https://raw.githubusercontent.com/exo-lang/exo/main/README.md
85. *Exocompilation for Productive Programming of Hardware Accelerators*, Ikarashi, Bernstein,
    Reinking, Genc, Ragan-Kelley, PLDI 2022 — **metadata only**, from
    https://pldi22.sigplan.org/details/pldi-2022-pldi/21/Exocompilation-for-Productive-Programming-of-Hardware-Accelerators
86. Amanda Liu, Gilbert Louis Bernstein, Adam Chlipala, Jonathan Ragan-Kelley, *Verified
    Tensor-Program Optimization Via High-Level Scheduling Rewrites* (ATL), POPL 2022 —
    https://people.csail.mit.edu/lamanda/assets/documents/LiuPOPL2022.pdf *(pypdf)*.
87. Riyadh Baghdadi et al., *TIRAMISU*, arXiv:1804.10694 — https://arxiv.org/pdf/1804.10694 *(pypdf)*.
88. Lianmin Zheng et al., *Ansor: Generating High-Performance Tensor Programs for Deep Learning*,
    OSDI 2020, arXiv:2006.06762 — https://arxiv.org/pdf/2006.06762 *(pypdf)*.
89. Andrew Adams, Karima Ma, Luke Anderson, Riyadh Baghdadi, Tzu-Mao Li et al., *Learning to
    Optimize Halide with Tree Search and Random Programs*, ACM TOG 38(4), July 2019 —
    https://halide-lang.org/papers/halide_autoscheduler_2019.pdf *(pypdf)*.
90. Michel Steuwer, Thomas Kœhler, Bastian Köpcke, Federico Pizzuti, *RISE & Shine*,
    arXiv:2201.03611 — https://arxiv.org/pdf/2201.03611 *(GPU-only evaluation)*.
91. Ari Rasch, *(De/Re)-Composition of Data-Parallel Computations via Multi-Dimensional
    Homomorphisms*, arXiv:2405.05118 — https://arxiv.org/pdf/2405.05118 *(pypdf)*.
92. Zhengyang Liu, Stefan Mada, John Regehr, *Minotaur: A SIMD-Oriented Synthesizing
    Superoptimizer*, arXiv:2306.00229 — https://arxiv.org/pdf/2306.00229 *(pypdf)*.
93. Alexa VanHattum, Rachit Nigam, Vincent T. Lee, James Bornholt, Adrian Sampson, *Vectorization for
    Digital Signal Processors via Equality Saturation* (Diospyros), ASPLOS '21 —
    https://cs.wellesley.edu/~avh/diospyros-asplos-2021-preprint.pdf *(pypdf)*.
94. Yifei He, Artur Podobas, Måns I. Andersson, Stefano Markidis, *FFTc: An MLIR Dialect for
    Developing HPC Fast Fourier Transform Libraries*, Euro-Par 2022 Workshops, arXiv:2207.06803 —
    https://arxiv.org/pdf/2207.06803 *(pypdf)* ; and He, Podobas, Markidis, *Leveraging MLIR for Loop
    Vectorization and GPU Porting of FFT Libraries*, Euro-Par 2023 Workshops, arXiv:2308.00497 —
    https://arxiv.org/pdf/2308.00497 *(pypdf)*.
95. Yifei He, *Domain-Specific Compilation Framework with High-Level Tensor Abstraction for Fast
    Fourier Transform and Finite-Difference Time-Domain Methods*, KTH Doctoral Thesis,
    TRITA-EECS-AVL-2025:67, defended 12 June 2025 —
    https://www.diva-portal.org/smash/get/diva2:1958857/FULLTEXT01.pdf *(pypdf)*. **Contains the full
    text of the paywalled CLUSTER 2024 FFTc 2.0 paper.**
96. *ProtoX: A First Look*, arXiv:2307.07931 — https://arxiv.org/pdf/2307.07931 *(pypdf)*.
97. Martin Reinecke, **ducc0** — README
    https://raw.githubusercontent.com/mreineck/ducc/master/README.md ; ChangeLog ; `src/ducc0/fft/fft.h` ;
    `src/ducc0/fft/fft1d_impl.h` ; `src/ducc0/fft/fftnd_impl.h` ; `python/demos/fft_bench.py` ; and
    **pocketfft** README https://raw.githubusercontent.com/mreineck/pocketfft/master/README.md
98. KFR — https://raw.githubusercontent.com/kfrlib/kfr/master/README.md ; RustFFT —
    https://raw.githubusercontent.com/ejmahler/RustFFT/master/README.md ; PFFFT —
    https://raw.githubusercontent.com/marton78/pffft/master/README.md ; FFTS —
    https://raw.githubusercontent.com/anthonix/ffts/master/README.md
99. LIGO/Virgo/KAGRA, *Modernizing gravitational wave analysis with ROOT+ and KFR* (ACAT2024),
    arXiv:2503.14292 — https://arxiv.org/pdf/2503.14292 *(MFLOPS values figure-only)*.
100. Dmitrii Tolmachev, **VkFFT** — README
     https://raw.githubusercontent.com/DTolm/VkFFT/master/README.md ; API guide
     https://raw.githubusercontent.com/DTolm/VkFFT/master/documentation/VkFFT_API_guide.pdf
     *(pypdf)* ; SC22 technical poster
     https://sc22.supercomputing.org/proceedings/tech_poster/poster_files/rpost143s3-file2.pdf
     *(pypdf)*.
101. NVIDIA **cuFFTDx** documentation — https://docs.nvidia.com/cuda/cufftdx/index.html and
     https://docs.nvidia.com/cuda/cufftdx/performance.html
102. AMD **rocFFT** Bluestein design document —
     https://rocm.docs.amd.com/projects/rocFFT/en/docs-6.1.1/design/bluestein.html
103. **oneAPI oneMKL** DFT specification —
     https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/elements/onemkl/source/domains/dft/dft.html
104. Richard Schoonhoven, Ben van Werkhoven, K. Joost Batenburg, *Benchmarking optimization
     algorithms for auto-tuning GPU kernels*, IEEE TEVC, arXiv:2210.01465 —
     https://arxiv.org/pdf/2210.01465 *(pypdf)*.
105. Floris-Jan Willemsen, Niki van Stein, Ben van Werkhoven, *Automated Algorithm Design for
     Auto-Tuning Optimizers*, arXiv:2510.17899 — https://arxiv.org/pdf/2510.17899 *(pypdf)*.
106. Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia
     Mirhoseini, *KernelBench: Can LLMs Write Efficient GPU Kernels?*, arXiv:2502.10517 —
     https://arxiv.org/pdf/2502.10517 *(pypdf)*.
107. Carlo Baronio, Pietro Marsella et al., *Kevin: Multi-Turn RL for Generating CUDA Kernels*,
     arXiv:2507.11948 — https://arxiv.org/pdf/2507.11948 *(pypdf)*.
108. Jianghui Wang, Vinay Joshi et al. (AMD), *GEAK: Introducing Triton Kernel AI Agent & Evaluation
     Benchmarks*, arXiv:2507.23194 — https://arxiv.org/pdf/2507.23194 *(pypdf)*.
109. Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang,
     Alex Aiken, *Astra: A Multi-Agent System for GPU Kernel Performance Optimization*,
     arXiv:2509.07506 — https://arxiv.org/pdf/2509.07506 *(pypdf)*.
110. Jinwu Chen, Qidie Wu, Bin Li et al., *cuPilot: A Strategy-Coordinated Multi-agent Framework for
     CUDA Kernel Evolution*, arXiv:2512.16465 — https://arxiv.org/pdf/2512.16465 *(pypdf)*.
111. Jubi Taneja, Avery Laird, Cong Yan, Madan Musuvathi, Shuvendu K. Lahiri, *LLM-Vectorizer:
     LLM-based Verified Loop Vectorizer*, arXiv:2406.04693 — https://arxiv.org/pdf/2406.04693
     *(pypdf)*.
112. Zhongchun Zheng, Kan Wu, Long Cheng, Lu Li, Rodrigo C. O. Rocha, Tianyi Liu et al., *VecTrans*,
     arXiv:2503.19449 — https://arxiv.org/pdf/2503.19449 *(pypdf)*.
113. Nazmus Sakib, Tarun Prabhu, Nandakishore Santhi, John Shalf, Abdel-Hameed A. Badawy,
     *Comparison of Vectorization Capabilities of Different Compilers for X86 and ARM CPUs*,
     arXiv:2502.11906 — https://arxiv.org/pdf/2502.11906 *(pypdf)*.

**Complex arithmetic and accuracy (§6)**

114. Claude-Pierre Jeannerod, Peter Kornerup, Nicolas Louvet, Jean-Michel Muller, *Error bounds on
     complex floating-point multiplication with an FMA*, Mathematics of Computation 86(304):881–898
     (2017), DOI 10.1090/mcom/3123 — https://inria.hal.science/hal-00867040v5/document *(pypdf)*.
115. Claude-Pierre Jeannerod, Nicolas Louvet, Jean-Michel Muller, *Further analysis of Kahan's
     algorithm for the accurate computation of 2×2 determinants*, Mathematics of Computation
     82(284):2245–2264 (2013) — https://ens-lyon.hal.science/ensl-00649347/document *(pypdf)*.
116. Claude-Pierre Jeannerod, Nicolas Louvet, Jean-Michel Muller, *On the componentwise accuracy of
     complex floating-point division with an FMA*, ARITH-21 (2013), 83–90 —
     https://inria.hal.science/ensl-00734339/document *(pypdf)*.
117. Claude-Pierre Jeannerod, *Exploiting structure in floating-point arithmetic*, MACIS 2015,
     Springer LNCS 9582 — https://inria.hal.science/hal-01247059/document *(pypdf)*.
118. Nicholas J. Higham, *Stability of a method for multiplying complex matrices with three real
     matrix multiplications*, SIAM J. Matrix Anal. Appl. 13(3):681–687 (1992), MIMS EPrint 2006.169 —
     https://eprints.maths.manchester.ac.uk/348/1/covered/MIMS_ep2006_169.pdf *(pypdf)*.
119. Field G. Van Zee, Tyler M. Smith, *Implementing high-performance complex matrix multiplication
     via the 3m and 4m methods*, ACM TOMS (2017), DOI 10.1145/3086466 —
     https://www.cs.utexas.edu/~flame/pubs/blis5_toms_rev2.pdf *(pypdf)*.
120. Steven G. Johnson, Matteo Frigo, *A Modified Split-Radix FFT With Fewer Arithmetic Operations*,
     IEEE Trans. Signal Processing 55(1):111–119 (2007) — https://www.fftw.org/newsplit.pdf *(pypdf)*.
121. Elliot Linzer, Ephraim Feig, *Modified FFTs for fused multiply-add architectures*, Mathematics
     of Computation 60(201):347–361 (1993) —
     https://www.ams.org/journals/mcom/1993-60-201/S0025-5718-1993-1159169-0/S0025-5718-1993-1159169-0.pdf
     *(pypdf)*.
122. Daniel J. Bernstein, *The tangent FFT*, 2007 — https://cr.yp.to/arith/tangentfft-20070809.pdf
     *(pypdf)*.
123. Mohamed Amine Bergach, *Dual-Select FMA Butterfly for FFT…*, arXiv:2604.00567, 1 April 2026 —
     https://arxiv.org/abs/2604.00567 and https://arxiv.org/html/2604.00567v1 *(preprint; see §6.4
     for the attribution caveat)*.
124. Mohamed Amine Bergach, *Shortest-Path FFT: Optimal SIMD Instruction Scheduling via Graph
     Search*, arXiv:2604.04311 — https://arxiv.org/abs/2604.04311 *(abstract only; Arm NEON)*.
125. Vincent Lefèvre, Jean-Michel Muller, *Accurate Complex Multiplication in Floating-Point
     Arithmetic*, ARITH-26 (Kyoto, 2019), 1–7 — https://inria.hal.science/hal-02001080/document
     *(pypdf)*.
126. Oscar Gustafsson, *On Lifting-Based Fixed-Point Complex Multiplications and Rotations*,
     ARITH-24 (2017), 43–49, DOI 10.1109/ARITH.2017.10 —
     https://www.diva-portal.org/smash/get/diva2:1121297/FULLTEXT02 *(pypdf)*.
127. FFTW project, *FFT Accuracy Benchmark Comments* — https://www.fftw.org/accuracy/comments.html
128. Steven G. Johnson, Matteo Frigo, *Implementing FFTs in Practice* (ch. 11 of *Fast Fourier
     Transforms*, ed. C. S. Burrus) — https://math.mit.edu/~stevenj/papers/JohnsonFr08-burrus.pdf
     *(pypdf)* and the arXiv posting https://arxiv.org/html/2602.23525v1
129. Matteo Frigo, Steven G. Johnson, *The Design and Implementation of FFTW3*, Proceedings of the
     IEEE 93(2):216–231 (2005) — https://www.fftw.org/fftw-paper-ieee.pdf *(pypdf)*.
130. Shota Kawakami, Daisuke Takahashi, *Computing FFTs at Target Precision Using Lower-Precision
     FFTs*, arXiv:2603.29129 (2026) — https://arxiv.org/abs/2603.29129 *(abstract only)*.
131. Nikhil Deveshwar, Abhejit Rajagopal, Peder E. Z. Larson, *A Mixed Precision FFT with
     applications in MRI*, arXiv:2512.04317, ICASSP 2026 — https://arxiv.org/html/2512.04317v2
132. Intel SDM mirrors: `VFMADDSUB132PD/213PD/231PD` —
     https://www.felixcloutier.com/x86/vfmaddsub132pd:vfmaddsub213pd:vfmaddsub231pd ;
     `VFCMULCPH/VFMULCPH` — https://www.felixcloutier.com/x86/vfcmulcph:vfmulcph
133. Arm A64 ISA (DDI0602) mirrors: `FCMLA (vectors)` —
     https://www.scs.stanford.edu/~zyedidia/arm64/fcmla_z_p_zzz.html ; `FCMLA (indexed)` —
     https://www.scs.stanford.edu/~zyedidia/arm64/fcmla_z_zzzi.html

### 8.2 Could not fetch — do not rely on any number attributed to these

Listed so nobody repeats the attempt, and so no claim in this section rests on them.

* **Intel ARK product page for the Xeon Gold 5218** — HTTP 403 from every route. The 1-FMA-unit
  figure is corroborated instead by Microway and `vpu-count` (both fetched).
* **WikiChip** (Xeon Gold 5218 page, "Frequency behavior", Cascade Lake microarchitecture) —
  `ECONNREFUSED` / timeout throughout the session. Unreachable from this environment.
* **Phoronix** AVX-512 reviews (Emerald Rapids 8592+, Sapphire Rapids vs Genoa) — HTTP 403. The
  quoted Emerald Rapids "2.95 vs 3.01 GHz" and Ice Lake "~175 MHz average drop" figures are
  therefore **[UNVERIFIED]**; the latter appears only second-hand inside LLVM issue #102047.
* **Intel Community forum threads** (single-core bandwidth Haswell/Broadwell/Skylake; MLC single-core
  reads; AVX-512 vs AVX2 with MKL dgemm on a Gold 5118; 3rd/4th Gen turbo frequencies) — HTTP 403.
  These reportedly contain McCalpin's per-generation single-core numbers and Intel's statement that
  3rd/4th-Gen per-licence turbo tables are NDA-only.
* **John McCalpin's live blog** at `sites.utexas.edu/jdm4372` — now HTTP 404 (site retired). All
  McCalpin material used here came via `web.archive.org`. His single-thread read-bandwidth
  optimisation series (the "Version002 → Version012, 12.8 → 16.6 GB/s from interleaving across
  4 KiB pages" result) could **not** be retrieved in any form; that number is
  **[UNVERIFIED — could not fetch]** and §1.1's concurrency argument does not depend on it.
* **Exo 1 / PLDI 2022 PDF** — 403/405 from ACM, MIT DSpace and Zenodo. Its "80–95 % of peak" figures
  are **[UNVERIFIED]**.
* **He & Markidis, CLUSTER 2024 (FFTc 2.0)** — IEEE paywall. *Worked around*: full text reproduced
  in the author's KTH thesis, which was fetched (ledger #95).
* **Tolmachev, VkFFT, IEEE Access 11 (2023)** — paywalled. *Worked around* via the SC22 poster and
  the official API guide.
* **Popovici, Schatz, Franchetti, Low, *A Flexible Framework for Multi-Dimensional DFTs*, SIAM
  J. Sci. Comput. 2020** — paywalled; the SPIRAL abstract page carries no numbers. Its
  "1.2× to 3× over MKL and FFTW" figure is **[UNVERIFIED]**.
* **Heinecke, Henry, Hutchinson, Pabst, *LIBXSMM…*, SC16, 981–991** — IEEE, not fetched. All libxsmm
  numbers here come from the SC15 poster and the current documentation.
* **oneMKL Developer Guide, Fourier Transform Functions** — HTTP 403. MKL's FFT internals remain
  undocumented publicly.
* **Anthony M. Blake, Ian H. Witten, Michael J. Cree, *The Fastest Fourier Transform in the South*,
  IEEE TSP 61(19), 2013** — 403 from every route, as in §01/§06. Its ">10 % speedup from streaming
  stores" claim is **[UNVERIFIED]** and is the most relevant unverified claim in §1.5.
* **Zhen Jia, Aleksandar Zlateski, Fredo Durand, Kai Li, *Optimizing N-dimensional, Winograd-based
  convolution for manycore CPUs*, PPoPP 2018** — ACM paywall. Directly relevant to §2; worth
  obtaining if anyone has access.
* **Tun Chen et al., *OpenFFT: An Adaptive Tuning Framework for 3D FFT on ARM Multicore CPUs*,
  ICS 2023** — ACM paywall. Directly relevant to §3.
* **Huo, Zhao, Liu, *Implementation and optimization of batch 3D FFT for sunway many-core
  processor*, CCF Trans. HPC 2025** — Springer, not fetched. Reportedly selects among three
  strategies by working-set size, i.e. exactly §3's question.
* **"Radix-2 multi-dimensional transposition-free FFT algorithm for Modern SIMD architectures"** —
  IEEE, not fetched. A search snippet claims "59.5 %–198 % faster than FFTW for complex 3D input";
  if that is real it is the closest thing to a modern multidimensional-structure CPU win and would
  contradict §2.1. **[UNVERIFIED — chase this down if anyone has IEEE access.]**
* **Lee, Akleylek, Wong, Yap, Goi, Hwang, *Parallel implementation of Nussbaumer algorithm and NTT
  on a GPU platform*, J. Supercomputing 77:3289–3314 (2021)** — Springer 303. The only modern
  measured Nussbaumer number located, and it is unverified.
* **Williams, Waterman, Patterson, *Roofline*, CACM 2009** — PDF fetched but font-scrambled and
  unusable for quotation. The roofline formulations quoted in §3.7 come from Zlateski et al. and
  Gelashvili et al. instead.
* **Takahashi, *A Blocking Algorithm for FFT on Cache-Based Processors*** and *Memory Efficient
  Two-Pass 3D FFT Algorithm for Intel Xeon Phi* — paywalled, as in §05.
* **Johnson & Burrus, TR 8105 (the canonical 17-point Winograd module)**, **Blake/Witten/Cree 2013**,
  **Sansone & Cococcioni 2023** — the three papers `LITERATURE.md` §6 names as unread across the
  whole corpus. Still unread. Two independent passes tried again this session and failed.
* **A rigorous, fetchable 2020–2026 measurement of `-O2` versus `-O3` on small numeric kernels**, and
  **any SPIRAL/autotuning study quantifying transfer loss across Intel *server* generations**, and
  **any FFTW/OpenBLAS/BLIS/NumPy published policy statement on 512-bit versus 256-bit** — searched
  for repeatedly by two passes; none exist in fetchable form. These are genuine gaps in the
  literature, not search failures.

### 8.3 Where this section contradicts the existing corpus

Stated explicitly, with both sides cited, per the curation rules.

| corpus claim | where | this section | verdict |
|---|---|---|---|
| AVX-512 downclocking on this SKU is severe (2.7 → 2.3 → 1.6 GHz) and a 512-bit kernel may lose to a 256-bit one | §04 §8.2; `LITERATURE.md` §4.8 gap 6; `PANEL_BRIEF.md` | Intel's own table: Gold 5218 at **1 active core** = 3.9 non-AVX / **2.9 AVX2 / 2.9 AVX-512**; L1 = L2 up to 8 cores | **Corpus wrong for our workload.** The 1.6 GHz figure is a 9+-core number. §4.1 |
| The leaders are within 4–13 % of their FP-port floors at B=1, so the arithmetic is finished | `panel_r3/VERDICT.md` §3, §6 | Those floors used 2.30 GHz; at the 1-core AVX-512 licence of 2.9 GHz the margins are **1.31–1.43×** | **Corpus number superseded** (and the verdict flagged the possibility itself). §4.1 |
| "Do not tune to this node's 256 KiB L2 — block to L1 by planes, which is safe everywhere" | §05 §10.6 | L1 blocking is safe but leaves 87.3 GB/s of L2 unexploited on a 1 MiB L2; Intel and Alappat et al. both recommend **blocking to L2** on Skylake Server | **Corpus advice outdated** for this part. §1.4, §1.9 |
| Working set > L3 ⇒ streams from DRAM (implicit throughout §05's working-set tables) | §05 §5.4 | CLX's non-inclusive victim L3 with a streaming-aware policy retains **20 % hit rate at 4× L3** and detectable reuse at **10× L3** | **Corpus model incomplete.** Explains why two of our large-batch cells beat a pure-DRAM bound. §1.3, §1.12 |
| Split vs interleaved complex is "strongly motivated but unproven" | `LITERATURE.md` §4.4; §04's own gap list | Three independent measured sources: `DFT_n ⊗ I_ν` split kernels **1.3–2× faster than FFTW** (double, x86); permute-vs-gather **IPC 0.13 → 2.59, 9.25× speed**; ducc0 ships split | **Question closed in the corpus's favour.** §5.4 |
| There is no published measurement of a small fixed 3D complex-double cube on a single CPU core, by anyone | §03 §7; §07 gap 7; `LITERATURE.md` §4.8 gap 1 and §6 | Popovici et al., IPDPS 2015: odd cubes edge 7–119, complex double, **single core**, vs MKL and FFTW, **2–3×** — and the win came from layout and loop merging, not the 1D kernel | **Gap partially closed, and the finding redirects our effort.** §2.2 |
| L=17: the corpus's four positions on dense-symmetric vs Rader vs Winograd, framed as something the literature could settle | `LITERATURE.md` §4.2 | ducc0 uses **neither** (generic pass, because 17 < 110); VkFFT's Rader thresholds are **both exactly 17** with an explicit hedge for 17–23; cuFFT switches to Bluestein after 17 | **Not a gap in our reading — an open question in the field.** §5.5, §5.7 |
| Pass fusion is worth single-digit percent, sometimes negative | `panel_r3/VERDICT.md` §5 (settling §4.3) | Confirmed **for L1↔L2 intermediates**. Untested for L2↔DRAM, where the bandwidth gap is 7× rather than 2.6× and Tolmachev's rule predicts a large payoff | **Corpus right, but only in the regime it tested.** §1.9 |
| Non-temporal stores are worth "up to the write-allocate share (~1/3 of a read-1-write-1 stream)" | §05 §8; `LITERATURE.md` §5 item 11 | Ratio confirmed (3:2 for a copy-shaped kernel, i.e. 1.5×, not 4:3) — **but** Intel's manual documents a Skylake-Server-specific core-resource-occupancy regression that can cap per-core NT write bandwidth, and McCalpin measured single-core NT *losing* on an earlier part | **Corpus arithmetic right, risk understated.** §1.5 |
| McFarlin et al.'s "direct computation preferable up to n ≈ 20" | §03 §6.4, §06 §6.4a | Correct, but it is a **Larrabee** result, and the stated mechanism is broadcast hardware + FMA — which means it transfers **better to AVX-512 than to AVX2** | **Corpus citation imprecise; the correction favours our L=17 dense kernel.** §3.1 |
| 3-multiply Karatsuba complex multiply "has no 2u guarantee" and was rejected by FFTW on throughput grounds | §07 §6.1; `LITERATURE.md` §5 item 6 | Confirmed and strengthened with primary sources: 6 instructions vs 4, cannot fuse, and Higham's exact counterexample | **Corpus right; now properly sourced.** §6.2 |
