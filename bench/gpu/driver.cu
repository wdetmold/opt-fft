/* Timing and correctness driver for the single-A100 competition.
 *
 * Deliberately the same shape as the CPU driver, so the two phases' methodology matches:
 * compilation and plan setup excluded, warmup discarded, inner repeat count auto-calibrated
 * so short cases clear the timer, many samples, several independent processes, batched and
 * non-batched measured separately.
 *
 * What differs, because it is a GPU:
 *   * The timed region is device-resident: the input is copied to the device once, before
 *     timing, and stays there. H2D and D2H are measured separately and reported, so the
 *     end-to-end cost is visible without contaminating the kernel comparison.
 *   * Timing uses CUDA events around the whole inner loop, plus an explicit synchronize, so
 *     asynchronous launches cannot be counted as free.
 *   * cudaGetLastError() is checked after the transform: a kernel that faults must fail the
 *     entry rather than silently returning a zero-filled buffer that happens to be fast.
 *
 * ON SAMPLE LENGTH, which turns out to matter a great deal here. We have no permission to
 * lock clocks on this cluster (`nvidia-smi -lgc` returns "The current user does not have
 * permission"), so the GPU's own boost behaviour decides the clock. Measured on the A100
 * with cuFFT at L=8, B=64, varying only the sample length:
 *
 *      --min-sample-ms  3   ->  22.4 us per execute
 *      --min-sample-ms 10   ->  20.9 us
 *      --min-sample-ms 20   ->  12.3 us      <-- SM clock pinned at 1410 MHz
 *      --min-sample-ms 50   ->  12.4 us
 *
 * That is a 1.7x cliff, not a gradual ramp: below ~20 ms of continuous work the GPU never
 * reaches its boost clock, and every number comes out slow. So the default is 20 ms and
 * should not be lowered. The literature's opposite warning (tight-loop repetition inflating
 * times by 20-25%) applies to parts whose clocks are already locked; here the ramp dominates
 * and the sustained number is also the representative one, since the target workload runs
 * transforms back to back.
 */
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fft3d_gpu_api.h"

#define CUDA_OK(call)                                                                    \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "driver: %s failed at %s:%d: %s\n", #call, __FILE__,          \
                    __LINE__, cudaGetErrorString(err_));                                  \
            exit(2);                                                                      \
        }                                                                                 \
    } while (0)

static int compare_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s --L N --batch B --in FILE [options]\n"
            "  --samples N        timed samples (default 30)\n"
            "  --warmup N         discarded warmup executes (default 5)\n"
            "  --min-sample-ms X  auto-calibrate inner reps to exceed this (default 20;\n"
            "                     shorter samples measure a downclocked GPU -- see header)\n"
            "  --out FILE         write the transformed batch here (for the checker)\n"
            "  --json FILE        write timing results here\n"
            "  --run-index N      label for this process\n",
            argv0);
    exit(2);
}

int main(int argc, char **argv)
{
    int L = 0, batch = 0, samples = 30, warmup = 5, run_index = 0;
    double min_sample_ms = 20.0;  /* see the clock-ramp note in the header */
    const char *in_path = NULL, *out_path = NULL, *json_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--L") && i + 1 < argc) L = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--samples") && i + 1 < argc) samples = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-sample-ms") && i + 1 < argc) min_sample_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else if (!strcmp(argv[i], "--run-index") && i + 1 < argc) run_index = atoi(argv[++i]);
        else usage(argv[0]);
    }
    if (L <= 0 || batch <= 0 || !in_path) usage(argv[0]);

    if (!fft3d_gpu_supports(L)) {
        fprintf(stderr, "%s: does not support L=%d\n", fft3d_gpu_name(), L);
        if (json_path) {
            FILE *f = fopen(json_path, "w");
            if (f) {
                fprintf(f, "{\"name\":\"%s\",\"L\":%d,\"batch\":%d,\"supported\":false}\n",
                        fft3d_gpu_name(), L, batch);
                fclose(f);
            }
        }
        return 3;
    }

    const size_t volume = (size_t)L * L * L;
    const size_t count = volume * (size_t)batch;
    const size_t bytes = count * sizeof(double2);

    /* ---- host side: read the shared input ---- */
    double2 *h_in = (double2 *)malloc(bytes);
    double2 *h_out = (double2 *)malloc(bytes);
    if (!h_in || !h_out) { fprintf(stderr, "driver: out of host memory\n"); return 2; }

    FILE *f = fopen(in_path, "rb");
    if (!f) { perror(in_path); return 2; }
    if (fread(h_in, sizeof(double2), count, f) != count) {
        fprintf(stderr, "driver: %s too short for L=%d batch=%d\n", in_path, L, batch);
        fclose(f);
        return 2;
    }
    fclose(f);

    int device = 0;
    CUDA_OK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, device));

    double2 *d_in = NULL, *d_out = NULL;
    CUDA_OK(cudaMalloc((void **)&d_in, bytes));
    CUDA_OK(cudaMalloc((void **)&d_out, bytes));
    CUDA_OK(cudaMemset(d_out, 0, bytes));

    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));

    /* ---- transfers, measured but excluded from the transform time ---- */
    float h2d_ms = 0.0f, d2h_ms = 0.0f;
    CUDA_OK(cudaEventRecord(ev_start));
    CUDA_OK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(ev_stop));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    CUDA_OK(cudaEventElapsedTime(&h2d_ms, ev_start, ev_stop));

    /* ---- setup, timed but reported separately ---- */
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(ev_start));
    fft3d_gpu_plan *plan = fft3d_gpu_create(L, batch);
    CUDA_OK(cudaEventRecord(ev_stop));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    float setup_ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&setup_ms, ev_start, ev_stop));
    if (!plan) {
        fprintf(stderr, "%s: create failed for L=%d batch=%d\n", fft3d_gpu_name(), L, batch);
        return 3;
    }

    /* ---- warmup: JIT, cache warming, first-launch cost -- all discarded ---- */
    for (int i = 0; i < warmup; ++i) fft3d_gpu_execute(plan, d_in, d_out);
    CUDA_OK(cudaDeviceSynchronize());
    cudaError_t warm_err = cudaGetLastError();
    if (warm_err != cudaSuccess) {
        fprintf(stderr, "%s: kernel error during warmup: %s\n", fft3d_gpu_name(),
                cudaGetErrorString(warm_err));
        return 4;
    }

    /* ---- calibrate the inner repeat count ---- */
    long inner = 1;
    for (;;) {
        CUDA_OK(cudaEventRecord(ev_start));
        for (long i = 0; i < inner; ++i) fft3d_gpu_execute(plan, d_in, d_out);
        /* An implementation may launch on its own stream, and a stream created with
           cudaStreamNonBlocking does NOT synchronize with the NULL stream -- so an event
           recorded here would not await it and the work would be timed as free. */
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaEventRecord(ev_stop));
        CUDA_OK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        if (ms >= min_sample_ms || inner >= (1L << 24)) break;
        double growth = ms > 0.0f ? min_sample_ms / (double)ms : 8.0;
        long next = (long)((double)inner * (growth > 2.0 ? growth : 2.0));
        inner = next > inner ? next : inner * 2;
    }

    /* ---- the measurement ---- */
    double *per_execute = (double *)malloc((size_t)samples * sizeof(double));
    if (!per_execute) { fprintf(stderr, "driver: out of memory\n"); return 2; }
    for (int s = 0; s < samples; ++s) {
        CUDA_OK(cudaEventRecord(ev_start));
        for (long i = 0; i < inner; ++i) fft3d_gpu_execute(plan, d_in, d_out);
        CUDA_OK(cudaDeviceSynchronize());   /* see the note in the calibration loop */
        CUDA_OK(cudaEventRecord(ev_stop));
        CUDA_OK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        per_execute[s] = (double)ms * 1e-3 / (double)inner;
    }

    /* ---- the checked output comes from the same code path ---- */
    CUDA_OK(cudaMemset(d_out, 0, bytes));
    fft3d_gpu_execute(plan, d_in, d_out);
    CUDA_OK(cudaDeviceSynchronize());
    cudaError_t run_err = cudaGetLastError();
    if (run_err != cudaSuccess) {
        fprintf(stderr, "%s: kernel error: %s\n", fft3d_gpu_name(), cudaGetErrorString(run_err));
        return 4;
    }

    CUDA_OK(cudaEventRecord(ev_start));
    CUDA_OK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaEventRecord(ev_stop));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    CUDA_OK(cudaEventElapsedTime(&d2h_ms, ev_start, ev_stop));

    if (out_path) {
        FILE *g = fopen(out_path, "wb");
        if (!g) { perror(out_path); return 2; }
        if (fwrite(h_out, sizeof(double2), count, g) != count) {
            fprintf(stderr, "driver: short write to %s\n", out_path);
            fclose(g);
            return 2;
        }
        fclose(g);
    }

    double mean = 0.0;
    for (int s = 0; s < samples; ++s) mean += per_execute[s];
    mean /= (double)samples;
    double variance = 0.0;
    for (int s = 0; s < samples; ++s) {
        double d = per_execute[s] - mean;
        variance += d * d;
    }
    double sd = samples > 1 ? sqrt(variance / (double)(samples - 1)) : 0.0;

    qsort(per_execute, (size_t)samples, sizeof(double), compare_double);
    double best = per_execute[0];
    double median = per_execute[samples / 2];
    double worst = per_execute[samples - 1];

    /* Nominal 5 N log2 N per volume: the same yardstick the CPU phases quote. */
    double nominal_flops = 5.0 * (double)volume * log2((double)volume) * (double)batch;
    /* Minimum traffic for an out-of-place transform: read in, write out. */
    double min_bytes = 2.0 * (double)bytes;

    if (json_path) {
        FILE *g = fopen(json_path, "w");
        if (!g) { perror(json_path); return 2; }
        fprintf(g,
                "{\"name\":\"%s\",\"description\":\"%s\",\"supported\":true,"
                "\"device\":\"%s\",\"L\":%d,\"batch\":%d,\"volume\":%zu,\"run_index\":%d,"
                "\"samples\":%d,\"inner\":%ld,\"warmup\":%d,"
                "\"setup_seconds\":%.9g,"
                "\"h2d_seconds\":%.9g,\"d2h_seconds\":%.9g,"
                "\"per_execute_seconds\":{\"min\":%.9g,\"median\":%.9g,\"mean\":%.9g,"
                "\"sd\":%.9g,\"max\":%.9g},"
                "\"per_transform_seconds_min\":%.9g,"
                "\"gflops_from_min\":%.6f,\"gflops_from_median\":%.6f,"
                "\"effective_gbytes_per_s\":%.6f}\n",
                fft3d_gpu_name(), fft3d_gpu_description(), prop.name, L, batch, volume,
                run_index, samples, inner, warmup, setup_ms * 1e-3,
                h2d_ms * 1e-3, d2h_ms * 1e-3,
                best, median, mean, sd, worst,
                best / (double)batch,
                nominal_flops / best / 1e9, nominal_flops / median / 1e9,
                min_bytes / best / 1e9);
        fclose(g);
    }

    printf("%-24s L=%-3d B=%-6d inner=%-7ld min=%10.3f us  median=%10.3f us  sd=%5.2f%%  "
           "%8.2f GF/s  %8.1f GB/s  setup=%.3f s  h2d=%.3f ms d2h=%.3f ms\n",
           fft3d_gpu_name(), L, batch, inner, best * 1e6, median * 1e6,
           mean > 0 ? 100.0 * sd / mean : 0.0, nominal_flops / best / 1e9,
           min_bytes / best / 1e9, setup_ms * 1e-3, h2d_ms, d2h_ms);

    free(per_execute);
    fft3d_gpu_destroy(plan);
    CUDA_OK(cudaFree(d_in));
    CUDA_OK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    return 0;
}
