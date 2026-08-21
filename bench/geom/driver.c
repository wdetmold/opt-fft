/* Timing and correctness driver: one process measures one (backend, L, batch).
 *
 * Every backend is linked against this same driver, so all of them see bit-identical
 * input, the same buffers, the same alignment and the same timing method.
 *
 * Methodology (see fft3d_api.h for the contract):
 *   1. Read the input volume(s) from a file the generator wrote, so every backend and
 *      the numpy checker agree on the data exactly.
 *   2. fft3d_create() is timed separately and reported as setup_seconds; it is NOT
 *      part of the transform time.  Compilation is obviously outside this process.
 *   3. Warmup executes run first and are discarded (first-touch faults, table paging,
 *      branch predictor and cache state).
 *   4. The inner repeat count is auto-calibrated by doubling until one timed sample
 *      exceeds --min-sample-ms, so tiny cases (a single 6^3 volume takes microseconds)
 *      are not measured against clock resolution.
 *   5. Many samples are then taken and the full distribution reported; the runner
 *      additionally repeats this whole process several times.
 *   6. A final separate execute produces the output written for correctness checking,
 *      so what is verified is what the timed code path computes.
 */
#define _POSIX_C_SOURCE 200809L

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fft3d_api.h"

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

static void *aligned_or_die(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0 || p == NULL) {
        fprintf(stderr, "driver: out of memory (%zu bytes)\n", bytes);
        exit(2);
    }
    return p;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s --L N --batch B --in FILE [options]\n"
            "  --samples N        timed samples to collect (default 30)\n"
            "  --warmup N         discarded warmup executes (default 5)\n"
            "  --min-sample-ms X  auto-calibrate inner reps to exceed this (default 20)\n"
            "  --out FILE         write the transformed batch here (for the checker)\n"
            "  --json FILE        write timing results here\n"
            "  --run-index N      label for this process, when the runner repeats it\n",
            argv0);
    exit(2);
}

int main(int argc, char **argv)
{
    int L = 0, batch = 0, samples = 30, warmup = 5, run_index = 0;
    double min_sample_ms = 20.0;
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

    if (!fft3d_supports(L)) {
        /* Not an error: implementations are allowed to specialize for one size. */
        fprintf(stderr, "%s: does not support L=%d\n", fft3d_name(), L);
        if (json_path) {
            FILE *f = fopen(json_path, "w");
            if (f) {
                fprintf(f, "{\"name\":\"%s\",\"L\":%d,\"batch\":%d,\"supported\":false}\n",
                        fft3d_name(), L, batch);
                fclose(f);
            }
        }
        return 3;
    }

    const size_t volume = (size_t)L * L * L;
    const size_t count = volume * (size_t)batch;
    const size_t bytes = count * sizeof(double _Complex);

    double _Complex *in = aligned_or_die(bytes);
    double _Complex *out = aligned_or_die(bytes);
    memset(out, 0, bytes);

    FILE *f = fopen(in_path, "rb");
    if (!f) { perror(in_path); return 2; }
    if (fread(in, sizeof(double _Complex), count, f) != count) {
        fprintf(stderr, "driver: %s is too short for L=%d batch=%d\n", in_path, L, batch);
        fclose(f);
        return 2;
    }
    fclose(f);

    /* ---- setup, timed but reported separately ---- */
    double t0 = now_seconds();
    fft3d_plan *plan = fft3d_create(L, batch);
    double setup_seconds = now_seconds() - t0;
    if (!plan) {
        fprintf(stderr, "%s: fft3d_create failed for L=%d batch=%d\n",
                fft3d_name(), L, batch);
        return 3;
    }

    /* ---- warmup: discarded ---- */
    for (int i = 0; i < warmup; ++i) fft3d_execute(plan, in, out);

    /* ---- calibrate the inner repeat count ---- */
    long inner = 1;
    for (;;) {
        double t = now_seconds();
        for (long i = 0; i < inner; ++i) fft3d_execute(plan, in, out);
        double elapsed = now_seconds() - t;
        if (elapsed * 1e3 >= min_sample_ms || inner >= (1L << 30)) break;
        /* Grow to roughly hit the target, with a floor of 2x so it always advances. */
        double growth = elapsed > 0 ? (min_sample_ms * 1e-3) / elapsed : 8.0;
        long next = (long)((double)inner * (growth > 2.0 ? growth : 2.0));
        inner = next > inner ? next : inner * 2;
    }

    /* ---- the measurement ---- */
    double *per_execute = malloc((size_t)samples * sizeof(double));
    if (!per_execute) { fprintf(stderr, "driver: out of memory\n"); return 2; }
    for (int s = 0; s < samples; ++s) {
        double t = now_seconds();
        for (long i = 0; i < inner; ++i) fft3d_execute(plan, in, out);
        per_execute[s] = (now_seconds() - t) / (double)inner;
    }

    /* ---- the output that gets checked is produced by the same code path ---- */
    memset(out, 0, bytes);
    fft3d_execute(plan, in, out);
    if (out_path) {
        FILE *g = fopen(out_path, "wb");
        if (!g) { perror(out_path); return 2; }
        if (fwrite(out, sizeof(double _Complex), count, g) != count) {
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

    /* Nominal 5 N log2 N per volume: the yardstick every FFT library quotes.  It is
     * not this implementation's real operation count -- it is the same model for
     * every backend, so the ratios are what carry meaning. */
    double nominal_flops = 5.0 * (double)volume * log2((double)volume) * (double)batch;

    if (json_path) {
        FILE *g = fopen(json_path, "w");
        if (!g) { perror(json_path); return 2; }
        fprintf(g,
                "{\"name\":\"%s\",\"description\":\"%s\",\"supported\":true,"
                "\"L\":%d,\"batch\":%d,\"volume\":%zu,\"run_index\":%d,"
                "\"samples\":%d,\"inner\":%ld,\"warmup\":%d,"
                "\"setup_seconds\":%.9g,"
                "\"per_execute_seconds\":{\"min\":%.9g,\"median\":%.9g,\"mean\":%.9g,"
                "\"sd\":%.9g,\"max\":%.9g},"
                "\"per_transform_seconds_min\":%.9g,"
                "\"gflops_from_min\":%.6f,\"gflops_from_median\":%.6f}\n",
                fft3d_name(), fft3d_description(), L, batch, volume, run_index,
                samples, inner, warmup, setup_seconds,
                best, median, mean, sd, worst,
                best / (double)batch,
                nominal_flops / best / 1e9, nominal_flops / median / 1e9);
        fclose(g);
    }

    printf("%-24s L=%-3d B=%-5d inner=%-8ld min=%10.3f us  median=%10.3f us  "
           "sd=%6.2f%%  %8.2f GF/s  setup=%.3f s\n",
           fft3d_name(), L, batch, inner, best * 1e6, median * 1e6,
           mean > 0 ? 100.0 * sd / mean : 0.0, nominal_flops / best / 1e9, setup_seconds);

    free(per_execute);
    fft3d_destroy(plan);
    free(in);
    free(out);
    return 0;
}
