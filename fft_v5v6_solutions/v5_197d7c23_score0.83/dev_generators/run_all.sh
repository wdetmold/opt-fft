#!/bin/sh
# Regenerate implementation.c exactly as attempt 197d7c23 built it.
# Replays the file-affecting steps of the session in order:
#   stages (heredocs)  ->  patch scripts (the exact python the agent piped in).
# Output: ../implementation.c  (also leaves build/impl_head.c, build/impl_tail.c)
set -e
cd "$(dirname "$0")"
rm -rf build && mkdir build

# S01+S02 (01:09-01:11): initial implementation.c (part2 truncated to the head-relevant
# tr8 section + buffers marker; the rest of part2 was discarded by the S03/S04 split).
cat stage_s01_part1.c stage_s02_tr8.c > build/implementation.c

cd build
python3 ../p03_split_head.py                 # S03  01:17:13  split -> impl_head.c
cp ../stage_s04_impl_tail_v1.c impl_tail.c   # S04  01:17:13  tail v1
python3 ../p04_prime_kernel_k4.py            # S05  01:22:06  prime kernel 4-way k-blocked
python3 ../p05_pk_macro_hygiene.py           # S06  01:22:18  PK_ACC4/PK_OUT hygiene
python3 ../p06_vmap_v2.py                    # S07  01:24:50  vmap rsqrt+divide
python3 ../p07_genpv_overlap_strip.py        # S08  01:25:56  GEN_PV overlap-strip (6-param)
python3 ../p08_k64_pragma.py                 # S09  01:28:58  k64 unroll pragmas (truncates)
python3 ../p09_tr8_restore.py                # S10  01:29:17  tr8 restored
# S11/S12 (fused-map + prefetch GEN_PV experiments, 01:35-01:37) are omitted:
# every byte they touched lies inside the region p12 replaces wholesale.
# --- container crash at 02:08:56; transcript truncated to the first 53 actions,
# --- replayed identically; the run resumed at 02:10:34 with the steps below.
python3 ../p12_genpv_plain.py                # T1   02:10:34  GEN_PV plain (ships)
python3 ../p13_api_hybrid.py                 # T2   02:15:42  hybrid BL/PV dispatch (ships)
python3 ../p14_pv_in_nomemset.py             # B6   02:19:40  pv_in without memset
python3 ../p15_unroll_k36_k45_k8.py          # B9   02:20:58  k36/k45/k8 unroll pragmas
python3 ../p16_splat_set1.py                 # B14  02:25:00  splat -> _mm512_set1_pd
python3 ../p17_pk_unroll16.py                # B16  02:25:57  prime-kernel j-loop unroll
python3 ../p18_map_blocks.py                 # B19  02:29:28  4-way-ILP map_blocks
python3 ../p19_cbuf_offset.py                # B25  02:35:23  CBUF set-decorrelation offset
python3 ../p20_k64_loadfirst.py              # B32  02:40:44  k64 load-first
python3 ../p21_k36_k45_loadfirst.py          # B33  02:41:26  k36/k45 load-first
python3 ../p22_header.py                     # B39  02:46:25  documentation header
python3 ../p23_kgen_macroize.py              # B49  02:58:02  K64GEN/K36GEN/K45GEN + d-variants
python3 ../p24_genpv_kfnd.py                 # B50  02:58:47  GEN_PV KFND (dense-strip kernel)
python3 ../p25_k23_k3block.py                # B60  03:11:02  K3 tail block (k23 = K4K4K3)

cat impl_head.c impl_tail.c > ../../implementation.c
echo "wrote $(cd ../.. && pwd)/implementation.c"
wc -l ../../implementation.c
