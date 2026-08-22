/* Timing and correctness driver for the multi-GPU competition (phase 4).
 *
 * Same methodology as the earlier phases -- compilation and setup excluded, warmup
 * discarded, inner repeat count auto-calibrated, many samples, several independent
 * processes -- with the differences a distributed transform forces:
 *
 *   * The backend owns its device memory and its own distribution (see fft3d_mgpu_api.h),
 *     so the driver measures upload/execute/download as three separate things and scores
 *     only execute(). Distribution and gathering are reported, not scored.
 *   * Timing is wall clock around the inner loop with EVERY device synchronized before the
 *     clock stops. CUDA events belong to one device and would miss work on the others.
 *   * The anti-memoization check re-uploads a perturbed input, so an implementation that
 *     cached its answer is caught even though every timed call sees the same data.
 */
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fft3d_mgpu_api.h"

#define CUDA_OK(call)                                                                    \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "driver: %s failed at %s:%d: %s\n", #call, __FILE__,          \
                    __LINE__, cudaGetErrorString(err_));                                  \
            exit(2);                                                                      \
        }                                                                                 \
    } while (0)

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int compare_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int g_ngpus = 0;
static int g_devices[64];

/* Every device, not just device 0: a distributed transform leaves work on all of them. */
static void sync_all(void)
{
    for (int g = 0; g < g_ngpus; ++g) {
        CUDA_OK(cudaSetDevice(g_devices[g]));
        CUDA_OK(cudaDeviceSynchronize());
    }
}

static void check_all_kernels(const char *what)
{
    for (int g = 0; g < g_ngpus; ++g) {
        CUDA_OK(cudaSetDevice(g_devices[g]));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: kernel error on device %d during %s: %s\n",
                    fft3d_mgpu_name(), g_devices[g], what, cudaGetErrorString(err));
            exit(4);
        }
    }
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s --L N --batch B --in FILE [options]\n"
            "  --ngpus N          GPUs to use (default: all visible)\n"
            "  --samples N        timed samples (default 20)\n"
            "  --warmup N         discarded warmup executes (default 5)\n"
            "  --min-sample-ms X  auto-calibrate inner reps to exceed this (default 20;\n"
            "                     shorter samples measure a downclocked GPU)\n"
            "  --out FILE         write the transformed batch here (for the checker)\n"
            "  --json FILE        write timing results here\n"
            "  --run-index N      label for this process\n",
            argv0);
    exit(2);
}

int main(int argc, char **argv)
{
    int L = 0, batch = 0, samples = 20, warmup = 5, run_index = 0, ngpus = 0;
    double min_sample_ms = 20.0;
    const char *in_path = NULL, *out_path = NULL, *json_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--L") && i + 1 < argc) L = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ngpus") && i + 1 < argc) ngpus = atoi(argv[++i]);
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

    int visible = 0;
    CUDA_OK(cudaGetDeviceCount(&visible));
    if (ngpus <= 0) ngpus = visible;
    if (ngpus > visible) {
        fprintf(stderr, "driver: asked for %d GPUs but only %d are visible\n", ngpus, visible);
        return 2;
    }
    g_ngpus = ngpus;
    for (int g = 0; g < ngpus; ++g) g_devices[g] = g;

    if (!fft3d_mgpu_supports(L, ngpus)) {
        fprintf(stderr, "%s: does not support L=%d on %d GPUs\n",
                fft3d_mgpu_name(), L, ngpus);
        if (json_path) {
            FILE *f = fopen(json_path, "w");
            if (f) {
                fprintf(f, "{\"name\":\"%s\",\"L\":%d,\"batch\":%d,\"ngpus\":%d,"
                           "\"supported\":false}\n", fft3d_mgpu_name(), L, batch, ngpus);
                fclose(f);
            }
        }
        return 3;
    }

    const size_t volume = (size_t)L * L * L;
    const size_t count = volume * (size_t)batch;
    const size_t bytes = count * sizeof(double2);

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

    cudaDeviceProp prop;
    CUDA_OK(cudaSetDevice(g_devices[0]));
    CUDA_OK(cudaGetDeviceProperties(&prop, g_devices[0]));

    /* ---- setup, reported separately ---- */
    double t0 = now_seconds();
    fft3d_mgpu_plan *plan = fft3d_mgpu_create(L, batch, ngpus, g_devices);
    sync_all();
    double setup_seconds = now_seconds() - t0;
    if (!plan) {
        fprintf(stderr, "%s: create failed for L=%d batch=%d ngpus=%d\n",
                fft3d_mgpu_name(), L, batch, ngpus);
        return 3;
    }

    /* ---- distribution, reported separately: it is not the transform ---- */
    t0 = now_seconds();
    fft3d_mgpu_upload(plan, h_in);
    sync_all();
    double upload_seconds = now_seconds() - t0;
    check_all_kernels("upload");

    /* ---- warmup, discarded ---- */
    for (int i = 0; i < warmup; ++i) fft3d_mgpu_execute(plan);
    sync_all();
    check_all_kernels("warmup");

    /* ---- calibrate ---- */
    long inner = 1;
    for (;;) {
        double t = now_seconds();
        for (long i = 0; i < inner; ++i) fft3d_mgpu_execute(plan);
        sync_all();
        double elapsed = now_seconds() - t;
        if (elapsed * 1e3 >= min_sample_ms || inner >= (1L << 22)) break;
        double growth = elapsed > 0 ? (min_sample_ms * 1e-3) / elapsed : 8.0;
        long next = (long)((double)inner * (growth > 2.0 ? growth : 2.0));
        inner = next > inner ? next : inner * 2;
    }

    /* ---- the measurement ---- */
    double *per_execute = (double *)malloc((size_t)samples * sizeof(double));
    if (!per_execute) { fprintf(stderr, "driver: out of memory\n"); return 2; }
    for (int s = 0; s < samples; ++s) {
        double t = now_seconds();
        for (long i = 0; i < inner; ++i) fft3d_mgpu_execute(plan);
        sync_all();
        per_execute[s] = (now_seconds() - t) / (double)inner;
    }
    check_all_kernels("the timed region");

    /* ---- the checked output comes from the same code path ---- */
    fft3d_mgpu_execute(plan);
    sync_all();
    t0 = now_seconds();
    fft3d_mgpu_download(plan, h_out);
    sync_all();
    double download_seconds = now_seconds() - t0;
    check_all_kernels("download");

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

    /* ---- anti-memoization: the answer must depend on the input ---- */
    {
        double2 *reference = (double2 *)malloc(bytes);
        if (!reference) { fprintf(stderr, "driver: out of memory\n"); return 2; }
        memcpy(reference, h_out, bytes);
        h_in[0].x += 1.0;
        h_in[0].y -= 2.0;
        fft3d_mgpu_upload(plan, h_in);
        sync_all();
        fft3d_mgpu_execute(plan);
        sync_all();
        fft3d_mgpu_download(plan, h_out);
        sync_all();
        int changed = memcmp(reference, h_out, bytes) != 0;
        free(reference);
        if (!changed) {
            fprintf(stderr, "%s: output did not change when the input changed -- the "
                            "transform is not being computed from the input\n",
                    fft3d_mgpu_name());
            return 5;
        }
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

    double nominal_flops = 5.0 * (double)volume * log2((double)volume) * (double)batch;
    double min_bytes = 2.0 * (double)bytes;   /* read in, write out, at minimum */

    if (json_path) {
        FILE *g = fopen(json_path, "w");
        if (!g) { perror(json_path); return 2; }
        fprintf(g,
                "{\"name\":\"%s\",\"description\":\"%s\",\"supported\":true,"
                "\"device\":\"%s\",\"ngpus\":%d,\"L\":%d,\"batch\":%d,\"volume\":%zu,"
                "\"run_index\":%d,\"samples\":%d,\"inner\":%ld,\"warmup\":%d,"
                "\"setup_seconds\":%.9g,\"upload_seconds\":%.9g,\"download_seconds\":%.9g,"
                "\"per_execute_seconds\":{\"min\":%.9g,\"median\":%.9g,\"mean\":%.9g,"
                "\"sd\":%.9g,\"max\":%.9g},"
                "\"per_transform_seconds_min\":%.9g,"
                "\"gflops_from_min\":%.6f,\"gflops_from_median\":%.6f,"
                "\"effective_gbytes_per_s\":%.6f}\n",
                fft3d_mgpu_name(), fft3d_mgpu_description(), prop.name, ngpus, L, batch,
                volume, run_index, samples, inner, warmup,
                setup_seconds, upload_seconds, download_seconds,
                best, median, mean, sd, worst,
                best / (double)batch,
                nominal_flops / best / 1e9, nominal_flops / median / 1e9,
                min_bytes / best / 1e9);
        fclose(g);
    }

    printf("%-22s L=%-4d B=%-6d n=%d inner=%-6ld min=%10.3f us median=%10.3f us sd=%5.2f%%  "
           "%8.2f GF/s %8.1f GB/s  setup=%.2fs up=%.3fs down=%.3fs\n",
           fft3d_mgpu_name(), L, batch, ngpus, inner, best * 1e6, median * 1e6,
           mean > 0 ? 100.0 * sd / mean : 0.0, nominal_flops / best / 1e9,
           min_bytes / best / 1e9, setup_seconds, upload_seconds, download_seconds);

    free(per_execute);
    fft3d_mgpu_destroy(plan);
    free(h_in);
    free(h_out);
    return 0;
}
