/* =============================================================================
 * D1_RACE -- plan-time race + per-host wisdom for the 1D campaign (d1_r1)
 * =============================================================================
 *
 * LIBRARY LAYER, scored by ADOPTION (the 1D generalization of the 3D
 * campaign's gen_race, gen_r9-era core).  Class owners: adopt with
 *
 *     #define D1_RACE_LIB_ONLY
 *     #include "d1_race.c"        // impl/ is the include dir; everything is
 *                                 // `static`, prefix gr_, no symbols leak
 *
 * Do NOT include this and bench/gen's gen_race.c in one translation unit --
 * they share the gr_ prefix by design (same API, same doctrine, 1D pins).
 *
 * WHAT THE LIBRARY GIVES YOU (all proven in the 3D campaign, ported verbatim):
 *   1. gr_time_run  -- the timing methodology alone: warmups, inner-rep
 *      calibration until a sample clears timer resolution, min over samples.
 *   2. gr_race      -- race N candidate thunks.  NOISE-AWARE: candidates
 *      within noise_rel of the fastest form a tie group and the LOWEST-INDEX
 *      member wins (put your primary first; a noise-level rival cannot
 *      displace it).  INTERLEAVED sample-major timing (round-robin rounds,
 *      min-of-rounds per candidate): candidate-major timing let 10-15%
 *      within-lease core-state drift decide races (three independent gen_r4
 *      records).  D1_RACE_SEQ=1 restores candidate-major.
 *      NOISE GATE (gen_r9): an upset over the primary ships outright only
 *      when its margin clears max(observed jitter, upset_floor 6%); a
 *      sub-floor upset must win a confirmation phase on fresh evidence or it
 *      REVERTS to the primary.  Reverted upsets are never persisted.
 *   3. gr_pick      -- gr_race behind the per-host wisdom cache
 *      (results/wisdom1d_<host>.json).  Keys carry an FNV signature of the
 *      candidate-name list, so a changed candidate set invalidates stale
 *      wisdom automatically.  flock + write-temp + rename keeps concurrent
 *      implementer binaries from corrupting the shared file.
 *   4. gr_pick_value -- integer-knob sweep (tile width, block size...) over
 *      the same race + wisdom machinery.
 *   5. gr_pick_plan  -- whole-plan race over {create,execute,destroy} vtables.
 *   6. gr_wisdom_get_str / gr_wisdom_put_str -- string wisdom under your own
 *      key (gate verdicts, chosen factorization names, knob packs).
 *   7. gr_wisdom_drop_prefix -- round-end wisdom strip, one flock-safe call.
 *
 * ENVIRONMENT PINS (all optional):
 *     D1_RACE_NO_RACE=1    always take candidate 0 (the primary)
 *     D1_RACE_FORCE=name   take the candidate with this name where present
 *     D1_RACE_REFRESH=1    ignore cached wisdom, re-race, overwrite
 *     D1_RACE_NO_WISDOM=1  race every time, never read or write the file
 *     D1_RACE_WISDOM=path  wisdom file override
 *     D1_RACE_VERBOSE=1    print every decision to stderr
 *     D1_RACE_SEQ=1        candidate-major timing order (A/B the racer itself)
 *     D1_RACE_NO_ENG=1     demo entry only: skip the cross-entry stage
 *
 * THE DEMO ENTRY (when not D1_RACE_LIB_ONLY) is the assembled-library trunk
 * in miniature, the gen_r8 "eng" stage brought to 1D on day one:
 *   - supports every L in [2, 2^20]; own safe engine = Stockham pow2 +
 *     Bluestein (pad to pow2 >= 2L-1, chirp k^2 reduced mod 2L in integers).
 *   - at plan time it compiles the sibling class entries (d1_pow2, d1_prime,
 *     d1_composite, d1_rader, d1_bluestein, d1_batchlane, d1_planner,
 *     d1_twiddle) as per-host shared objects (cached by source hash under
 *     build/race1d/<host>/), dlopens them RTLD_LOCAL, and GATES each in a
 *     FORKED CHILD (a crashing or hanging mid-round sibling cannot take this
 *     binary down; rel L2 vs the in-file reference < 2e-12, verdict cached
 *     per (L, B, source-hash)).
 *   - stage "exe.r2": whole-batch fft1d_execute raced, self = candidate 0
 *     (self sits out the race at big cells where its scalar cost would eat
 *     the budget; it remains the unconditional ship-time fallback).
 *   - stage "chn.r1": the graded map chain raced at an honest m from
 *     cases.txt (work-capped); candidate 0 = exec-winner + the exact driver
 *     map, plus every gated arm that exports a native fft1d_chain.
 *   - winners ship by vtable forwarding; wisdom pins run 2 of the driver's
 *     two-process repeatability check to run 1's choice.
 * Source churn re-keys everything: candidate names carry the source hash.
 * ============================================================================= */

#ifndef D1_RACE_C_INCLUDED
#define D1_RACE_C_INCLUDED

#if !defined(_GNU_SOURCE) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE 1
#endif

#include <complex.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define D1R_DIR "/home/lqcd/wdetmold/fft/bench/d1"

/* ---- 1. timing methodology ------------------------------------------------ */

typedef struct {
    int warmups;          /* untimed calls before anything (default 2)          */
    int samples;          /* timed samples after calibration (default 5)        */
    double min_sample_us; /* calibrate inner reps until a sample >= this (200)  */
    double budget_us;     /* soft cap for one whole gr_race call (default 5e6)  */
    double noise_rel;     /* statistical-tie width, relative (default 0.02)     */
    double upset_floor;   /* noise gate: an upset ships unconfirmed only past
                             max(observed jitter, this floor) (default 0.06)    */
    int confirm;          /* confirmation rounds for a sub-floor upset (0=samples) */
} gr_opts;

static inline gr_opts gr_default_opts(void)
{
    gr_opts o;
    o.warmups = 2;
    o.samples = 5;
    o.min_sample_us = 200.0;
    o.budget_us = 5e6;
    o.noise_rel = 0.02;
    o.upset_floor = 0.06;
    o.confirm = 0;
    return o;
}

typedef struct {
    double us;         /* min over samples, per single run() call  */
    double spread_rel; /* (max-min)/min over the timed samples     */
    int reps;          /* inner reps per sample after calibration  */
    int ok;            /* 0: setup failed / skipped, timing invalid */
} gr_timing;

static inline double gr_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e6 * (double)ts.tv_sec + 1e-3 * (double)ts.tv_nsec;
}

static inline void gr_time_run(void (*run)(void *), void *state, const gr_opts *o,
                               double deadline_us, gr_timing *t)
{
    t->ok = 1;
    for (int w = 0; w < o->warmups; ++w) run(state);

    int reps = 1;
    double per;
    for (;;) {
        double t0 = gr_now_us();
        for (int r = 0; r < reps; ++r) run(state);
        double el = gr_now_us() - t0;
        if (el >= o->min_sample_us || reps >= (1 << 24)) { per = el / reps; break; }
        reps = (el <= 1.0) ? reps * 8
                           : (int)((double)reps * (o->min_sample_us * 1.4 / el)) + 1;
    }
    double best = per, worst = per; /* the calibration pass is sample 0 */
    for (int s = 1; s < o->samples; ++s) {
        if (deadline_us > 0.0 && gr_now_us() > deadline_us) break;
        double t0 = gr_now_us();
        for (int r = 0; r < reps; ++r) run(state);
        per = (gr_now_us() - t0) / reps;
        if (per < best) best = per;
        if (per > worst) worst = per;
    }
    t->us = best;
    t->spread_rel = best > 0.0 ? (worst - best) / best : 0.0;
    t->reps = reps;
}

/* ---- 2. the race ----------------------------------------------------------- */

typedef struct {
    const char *name;      /* short, no quotes/spaces: the wisdom identity      */
    void *(*setup)(void *ctx);   /* optional; NULL state => candidate skipped.  */
                                 /* NULL fn => ctx passed straight to run().    */
    void (*run)(void *state);    /* one unit of YOUR graded workload            */
    void (*teardown)(void *state); /* optional                                  */
    void *ctx;
} gr_cand;

typedef struct {
    int widx;         /* winning candidate index (what gr_pick returns)     */
    int tie;          /* 1: runner-up within noise_rel; stability pick      */
    int from_wisdom;  /* 1: no race ran, cached answer                       */
    double us;        /* winner's time (cached or fresh)                     */
    double margin;    /* (runnerup - winner)/winner, 0 if none/cached        */
    int noisy;        /* 1: an upset failed the noise gate and was REVERTED;
                         gr_pick does NOT store it, next create re-races      */
} gr_pick_info;

static inline int gr_race(const gr_cand *c, int n, const gr_opts *o_in,
                          gr_timing *tm, gr_pick_info *pi)
{
    gr_opts o = o_in ? *o_in : gr_default_opts();
    gr_timing local[32];
    gr_timing *t = tm;
    if (!t && n <= 32) t = local;
    if (!t) return n > 0 ? 0 : -1;

    int gate_forced = -1;
    int gate_noisy = 0;
    double deadline = o.budget_us > 0.0 ? gr_now_us() + o.budget_us : 0.0;
    if (getenv("D1_RACE_SEQ") || n > 32) {
        for (int i = 0; i < n; ++i) {
            t[i].ok = 0;
            t[i].us = 0.0;
            t[i].spread_rel = 0.0;
            t[i].reps = 0;
            if (deadline > 0.0 && gr_now_us() > deadline && i > 0) continue;
            void *state = c[i].setup ? c[i].setup(c[i].ctx) : c[i].ctx;
            if (c[i].setup && !state) continue;
            gr_time_run(c[i].run, state, &o, deadline, &t[i]);
            if (c[i].teardown) c[i].teardown(state);
        }
    } else {
        void *state[32];
        double worst[32];
        for (int i = 0; i < n; ++i) {
            t[i].ok = 0;
            t[i].us = 0.0;
            t[i].spread_rel = 0.0;
            t[i].reps = 0;
            state[i] = NULL;
            worst[i] = 0.0;
            if (deadline > 0.0 && gr_now_us() > deadline && i > 0) continue;
            state[i] = c[i].setup ? c[i].setup(c[i].ctx) : c[i].ctx;
            if (c[i].setup && !state[i]) continue;
            t[i].ok = 1;
        }
        for (int i = 0; i < n; ++i) {
            if (!t[i].ok) continue;
            for (int w = 0; w < o.warmups; ++w) c[i].run(state[i]);
            int reps = 1;
            double per;
            for (;;) {
                double t0 = gr_now_us();
                for (int r = 0; r < reps; ++r) c[i].run(state[i]);
                double el = gr_now_us() - t0;
                if (el >= o.min_sample_us || reps >= (1 << 24)) { per = el / reps; break; }
                reps = (el <= 1.0) ? reps * 8
                                   : (int)((double)reps * (o.min_sample_us * 1.4 / el)) + 1;
            }
            t[i].us = per;
            worst[i] = per;
            t[i].reps = reps;
        }
        for (int s = 1; s < o.samples; ++s)
            for (int i = 0; i < n; ++i) {
                if (!t[i].ok) continue;
                if (deadline > 0.0 && gr_now_us() > deadline) { s = o.samples; break; }
                double t0 = gr_now_us();
                for (int r = 0; r < t[i].reps; ++r) c[i].run(state[i]);
                double per = (gr_now_us() - t0) / t[i].reps;
                if (per < t[i].us) t[i].us = per;
                if (per > worst[i]) worst[i] = per;
            }
        for (int i = 0; i < n; ++i) {
            if (!t[i].ok) continue;
            t[i].spread_rel = t[i].us > 0.0 ? (worst[i] - t[i].us) / t[i].us : 0.0;
        }
        /* noise gate (gen_r9): sub-floor upsets must confirm on fresh evidence */
        {
            int fb = -1;
            for (int i = 0; i < n; ++i)
                if (t[i].ok && (fb < 0 || t[i].us < t[fb].us)) fb = i;
            int prim = -1;
            for (int i = 0; i < n; ++i)
                if (t[i].ok) { prim = i; break; }
            int wx = fb;
            if (fb >= 0)
                for (int i = 0; i < fb; ++i)
                    if (t[i].ok && t[i].us <= t[fb].us * (1.0 + o.noise_rel)) { wx = i; break; }
            if (wx >= 0 && prim >= 0 && wx != prim) {
                double up = (t[prim].us - t[wx].us) / t[wx].us;
                double jit = t[wx].spread_rel > t[prim].spread_rel
                                 ? t[wx].spread_rel : t[prim].spread_rel;
                if (jit < o.noise_rel) jit = o.noise_rel;
                double bar = jit > o.upset_floor ? jit : o.upset_floor;
                if (up < bar) {
                    int R = o.confirm > 0 ? o.confirm : o.samples;
                    double cp = 0.0, cw = 0.0;
                    int ran = 0;
                    for (int s = 0; s < R; ++s) {
                        if (deadline > 0.0 && gr_now_us() > deadline && ran)
                            break;
                        const int pair[2] = { prim, wx };
                        for (int q = 0; q < 2; ++q) {
                            const int i = pair[q];
                            double t0 = gr_now_us();
                            for (int r = 0; r < t[i].reps; ++r)
                                c[i].run(state[i]);
                            double per = (gr_now_us() - t0) / t[i].reps;
                            if (q == 0) { if (!ran || per < cp) cp = per; }
                            else        { if (!ran || per < cw) cw = per; }
                        }
                        ++ran;
                    }
                    if (ran && cw < cp / (1.0 + o.noise_rel)) {
                        if (cw < t[wx].us) t[wx].us = cw;
                        if (cp < t[prim].us) t[prim].us = cp;
                    } else {
                        gate_forced = prim;
                        gate_noisy = 1;
                    }
                }
            }
        }
        for (int i = 0; i < n; ++i)
            if (t[i].ok && c[i].teardown) c[i].teardown(state[i]);
    }

    int best = -1;
    for (int i = 0; i < n; ++i)
        if (t[i].ok && (best < 0 || t[i].us < t[best].us)) best = i;
    if (best < 0) return -1;

    int widx = best;
    for (int i = 0; i < best; ++i)
        if (t[i].ok && t[i].us <= t[best].us * (1.0 + o.noise_rel)) { widx = i; break; }
    if (gate_forced >= 0) widx = gate_forced;

    if (pi) {
        double runner = -1.0;
        for (int i = 0; i < n; ++i)
            if (t[i].ok && i != widx && (runner < 0 || t[i].us < runner)) runner = t[i].us;
        pi->widx = widx;
        pi->us = t[widx].us;
        pi->margin = runner > 0.0 ? (runner - t[widx].us) / t[widx].us : 0.0;
        pi->tie = (runner > 0.0 && pi->margin < o.noise_rel) ? 1 : 0;
        pi->from_wisdom = 0;
        pi->noisy = gate_noisy;
    }
    return widx;
}

/* ---- 3. wisdom cache ------------------------------------------------------- */

#define GR_KEY_MAX 160
#define GR_NAME_MAX 128

static inline void gr_keyf(char *buf, size_t cap, const char *entry,
                           const char *tag, int L, int B)
{
    snprintf(buf, cap, "%s/%s/L%d/B%d", entry, tag, L, B);
}

/* Power-of-two batch bucket, capped at 512 (the 1D graded batches reach 512). */
static inline int gr_bucket(int b)
{
    int p = 1;
    while (p * 2 <= b && p < 512) p *= 2;
    return p;
}

static inline unsigned gr_sig(const gr_cand *c, int n)
{
    unsigned h = 2166136261u;
    for (int i = 0; i < n; ++i) {
        const char *s = c[i].name ? c[i].name : "?";
        while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
        h ^= 0xffu; h *= 16777619u;
    }
    return h;
}

static inline void gr__host(char *buf, size_t cap)
{
    if (gethostname(buf, cap) != 0) snprintf(buf, cap, "unknown");
    buf[cap - 1] = 0;
    for (char *p = buf; *p; ++p)
        if (*p == '.') { *p = 0; break; }
}

/* Wisdom file: D1_RACE_WISDOM override, else the campaign's absolute results/
 * (never cwd-relative: a binary run from another campaign's directory must not
 * write into that campaign's wisdom). */
static inline int gr_wisdom_path(char *buf, size_t cap)
{
    const char *env = getenv("D1_RACE_WISDOM");
    if (env && *env) { snprintf(buf, cap, "%s", env); return 1; }
    char host[64];
    gr__host(host, sizeof host);
    struct stat st;
    if (stat(D1R_DIR "/results", &st) == 0 && S_ISDIR(st.st_mode)) {
        snprintf(buf, cap, D1R_DIR "/results/wisdom1d_%s.json", host);
        return 1;
    }
    return 0;
}

static inline char *gr__read_all(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || n > (16 << 20)) { fclose(f); return NULL; }
    rewind(f);
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[rd] = 0;
    if (len) *len = rd;
    return buf;
}

static inline int gr__lockfd(const char *path, int exclusive)
{
    char lockp[560];
    snprintf(lockp, sizeof lockp, "%s.lock", path);
    int fd = open(lockp, O_CREAT | O_RDWR, 0666);
    if (fd >= 0) flock(fd, exclusive ? LOCK_EX : LOCK_SH); /* best effort */
    return fd;
}

static inline int gr_wisdom_lookup(const char *fullkey, char *wname, size_t wcap,
                                   int *widx, int *tie, double *us)
{
    char path[512];
    if (!gr_wisdom_path(path, sizeof path)) return 0;
    int lfd = gr__lockfd(path, 0);
    char *buf = gr__read_all(path, NULL);
    if (lfd >= 0) { flock(lfd, LOCK_UN); close(lfd); }
    if (!buf) return 0;

    int hit = 0;
    char needle[GR_KEY_MAX + 24];
    snprintf(needle, sizeof needle, "\"%s\":{", fullkey);
    char *p = strstr(buf, needle);
    if (p) {
        p += strlen(needle);
        char *end = strchr(p, '}');
        if (end) {
            *end = 0;
            char *w = strstr(p, "\"winner\":\"");
            char *x = strstr(p, "\"widx\":");
            char *ti = strstr(p, "\"tie\":");
            char *u = strstr(p, "\"us\":");
            if (w && x) {
                w += 10;
                char *we = strchr(w, '"');
                if (we && (size_t)(we - w) < wcap) {
                    memcpy(wname, w, (size_t)(we - w));
                    wname[we - w] = 0;
                    *widx = atoi(x + 7);
                    if (tie) *tie = ti ? atoi(ti + 6) : 0;
                    if (us) *us = u ? atof(u + 5) : 0.0;
                    hit = 1;
                }
            }
        }
    }
    free(buf);
    return hit;
}

/* Layout-agnostic entry scan (the gen_r9 fix: line-oriented parsers silently
 * wiped compacted files). Values are flat objects; the value ends at '}'. */
static inline int gr__next_entry(char **cur, char **k0, size_t *klen,
                                 char **v0, char **v1)
{
    char *p = *cur;
    for (;;) {
        char *q = strchr(p, '"');
        if (!q) return 0;
        char *ke = strchr(q + 1, '"');
        if (!ke) return 0;
        char *col = ke + 1;
        while (*col == ' ' || *col == '\t') ++col;
        if (*col != ':') { p = ke + 1; continue; }
        ++col;
        while (*col == ' ' || *col == '\t' || *col == '\n' || *col == '\r') ++col;
        if (*col != '{') { p = ke + 1; continue; }
        char *end = strchr(col, '}');
        if (!end) return 0;
        *k0 = q + 1;
        *klen = (size_t)(ke - q - 1);
        *v0 = col;
        *v1 = end;
        *cur = end + 1;
        return 1;
    }
}

static inline char *gr__entries_start(char *buf)
{
    char *p = strstr(buf, "\"entries\"");
    if (!p) return NULL;
    p += 9;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':') ++p;
    return *p == '{' ? p + 1 : NULL;
}

static inline void gr__emit_entry(FILE *f, const char *k0, size_t klen,
                                  const char *v0, const char *v1)
{
    fprintf(f, "\"%.*s\":", (int)klen, k0);
    for (const char *p = v0; p <= v1; ++p)
        if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') fputc(*p, f);
}

static inline void gr_wisdom_store(const char *fullkey, const char *winner,
                                   int widx, int tie, double us, double margin)
{
    char path[512];
    if (!gr_wisdom_path(path, sizeof path)) return;
    int lfd = gr__lockfd(path, 1);
    char *old = gr__read_all(path, NULL);

    char tmp[560];
    snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
    FILE *f = fopen(tmp, "w");
    if (f) {
        char host[64];
        gr__host(host, sizeof host);
        fprintf(f, "{\"host\":\"%s\",\"format\":1,\"entries\":{\n", host);
        fprintf(f, "\"%s\":{\"winner\":\"%s\",\"widx\":%d,\"tie\":%d,"
                   "\"us\":%.6g,\"margin\":%.4g}",
                fullkey, winner, widx, tie, us, margin);
        if (old) {
            char *cur = gr__entries_start(old);
            if (cur) {
                char *k0, *v0, *v1;
                size_t kl;
                const size_t klen = strlen(fullkey);
                while (gr__next_entry(&cur, &k0, &kl, &v0, &v1))
                    if (!(kl == klen && memcmp(k0, fullkey, klen) == 0)) {
                        fprintf(f, ",\n");
                        gr__emit_entry(f, k0, kl, v0, v1);
                    }
            }
        }
        fprintf(f, "\n}}\n");
        fclose(f);
        rename(tmp, path);
    }
    free(old);
    if (lfd >= 0) { flock(lfd, LOCK_UN); close(lfd); }
}

/* string-valued wisdom under your own key (no quotes/spaces/braces in val) */
static inline int gr_wisdom_get_str(const char *key, char *out, size_t cap)
{
    if (getenv("D1_RACE_NO_WISDOM") || getenv("D1_RACE_REFRESH")) return 0;
    int widx, tie;
    double us;
    return gr_wisdom_lookup(key, out, cap, &widx, &tie, &us);
}

static inline void gr_wisdom_put_str(const char *key, const char *val)
{
    if (getenv("D1_RACE_NO_WISDOM")) return;
    gr_wisdom_store(key, val, -1, 0, 0.0, 0.0);
}

static inline int gr_wisdom_drop_prefix(const char *prefix)
{
    char path[512];
    if (!gr_wisdom_path(path, sizeof path)) return 0;
    size_t plen = strlen(prefix);
    int dropped = 0;
    int lfd = gr__lockfd(path, 1);
    char *old = gr__read_all(path, NULL);
    if (old) {
        char tmp[560];
        snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
        FILE *f = fopen(tmp, "w");
        if (f) {
            char host[64];
            gr__host(host, sizeof host);
            fprintf(f, "{\"host\":\"%s\",\"format\":1,\"entries\":{", host);
            int first = 1;
            char *cur = gr__entries_start(old);
            if (cur) {
                char *k0, *v0, *v1;
                size_t kl;
                while (gr__next_entry(&cur, &k0, &kl, &v0, &v1)) {
                    if (kl >= plen && memcmp(k0, prefix, plen) == 0) {
                        ++dropped;
                    } else {
                        fprintf(f, "%s\n", first ? "" : ",");
                        gr__emit_entry(f, k0, kl, v0, v1);
                        first = 0;
                    }
                }
            }
            fprintf(f, "\n}}\n");
            fclose(f);
            rename(tmp, path);
        }
    }
    free(old);
    if (lfd >= 0) { flock(lfd, LOCK_UN); close(lfd); }
    return dropped;
}

/* ---- 4. gr_pick: the race behind the cache --------------------------------- */

static inline int gr_pick(const char *key, const gr_cand *c, int n,
                          const gr_opts *o, gr_pick_info *pi)
{
    gr_pick_info local;
    if (!pi) pi = &local;
    pi->widx = 0; pi->tie = 0; pi->from_wisdom = 0; pi->us = 0.0; pi->margin = 0.0;
    pi->noisy = 0;
    if (n <= 0) return -1;
    if (n == 1) return 0;

    const char *force = getenv("D1_RACE_FORCE");
    if (force && *force)
        for (int i = 0; i < n; ++i)
            if (c[i].name && strcmp(c[i].name, force) == 0) { pi->widx = i; return i; }
    if (getenv("D1_RACE_NO_RACE")) return 0;

    char fullkey[GR_KEY_MAX + 16];
    snprintf(fullkey, sizeof fullkey, "%s#%08x", key, gr_sig(c, n));

    int use_wisdom = !getenv("D1_RACE_NO_WISDOM");
    if (use_wisdom && !getenv("D1_RACE_REFRESH")) {
        char wname[GR_NAME_MAX];
        int widx = -1, tie = 0;
        double us = 0.0;
        if (gr_wisdom_lookup(fullkey, wname, sizeof wname, &widx, &tie, &us)) {
            int found = -1;
            if (widx >= 0 && widx < n && c[widx].name && !strcmp(c[widx].name, wname))
                found = widx;
            else
                for (int i = 0; i < n; ++i)
                    if (c[i].name && !strcmp(c[i].name, wname)) { found = i; break; }
            if (found >= 0) {
                pi->widx = found; pi->tie = tie; pi->from_wisdom = 1; pi->us = us;
                if (getenv("D1_RACE_VERBOSE"))
                    fprintf(stderr, "d1_race: %s -> %s (wisdom, %.3g us)\n",
                            fullkey, wname, us);
                return found;
            }
        }
    }

    int widx = gr_race(c, n, o, NULL, pi);
    if (widx < 0) { pi->widx = 0; return 0; }
    if (use_wisdom && !pi->noisy)
        gr_wisdom_store(fullkey, c[widx].name ? c[widx].name : "?",
                        widx, pi->tie, pi->us, pi->margin);
    if (getenv("D1_RACE_VERBOSE"))
        fprintf(stderr, "d1_race: %s -> %s (raced, %.3g us, margin %.1f%%%s%s)\n",
                fullkey, c[widx].name ? c[widx].name : "?", pi->us,
                100.0 * pi->margin, pi->tie ? ", TIE: stability pick" : "",
                pi->noisy ? ", NOISY: upset reverted, not stored" : "");
    return widx;
}

/* ---- 5. integer-knob sweep over the same machinery ------------------------- */

typedef struct {
    void (*run)(void *ctx, long v);
    void *ctx;
    long v;
    char name[24];
} gr__valstate;

static inline void gr__valrun(void *state)
{
    gr__valstate *s = (gr__valstate *)state;
    s->run(s->ctx, s->v);
}

static inline long gr_pick_value(const char *key, const long *vals, int n,
                                 void (*run)(void *ctx, long v), void *ctx,
                                 const gr_opts *o, gr_pick_info *pi)
{
    enum { CAP = 24 };
    gr__valstate st[CAP];
    gr_cand c[CAP];
    if (n <= 0) return 0;
    if (n > CAP) n = CAP;
    for (int i = 0; i < n; ++i) {
        st[i].run = run; st[i].ctx = ctx; st[i].v = vals[i];
        snprintf(st[i].name, sizeof st[i].name, "v%ld", vals[i]);
        c[i].name = st[i].name; c[i].setup = NULL; c[i].run = gr__valrun;
        c[i].teardown = NULL; c[i].ctx = &st[i];
    }
    int w = gr_pick(key, c, n, o, pi);
    return vals[w < 0 ? 0 : w];
}

/* ---- 6. whole-plan race ----------------------------------------------------- */

typedef struct {
    const char *name;
    void *(*create)(int L, int batch); /* NULL plan: will not plan -> skipped   */
    void (*execute)(void *plan, const double _Complex *in, double _Complex *out);
    void (*destroy)(void *plan);
} gr_plan_cand;

typedef struct {
    const gr_plan_cand *k;
    void *plan;
    int L, race_batch;
    const double _Complex *in;
    double _Complex *out;
} gr__planstate;

static inline void *gr__plansetup(void *ctx)
{
    gr__planstate *s = (gr__planstate *)ctx;
    s->plan = s->k->create(s->L, s->race_batch);
    return s->plan ? ctx : NULL;
}

static inline void gr__planrun(void *state)
{
    gr__planstate *s = (gr__planstate *)state;
    s->k->execute(s->plan, s->in, s->out);
}

static inline void gr__planteardown(void *state)
{
    gr__planstate *s = (gr__planstate *)state;
    s->k->destroy(s->plan);
    s->plan = NULL;
}

/* Race whole plans at (L, batch); race buffers are the caller's (deterministic
 * fill is the caller's job).  Candidate 0 is the existence fallback. */
static inline int gr_pick_plan(const char *key, const gr_plan_cand *k, int n,
                               int L, int batch, const double _Complex *in,
                               double _Complex *out, const gr_opts *o,
                               gr_pick_info *pi)
{
    enum { CAP = 16 };
    gr__planstate st[CAP];
    gr_cand c[CAP];
    if (n <= 0) return -1;
    if (n > CAP) n = CAP;
    if (n == 1) return 0;
    for (int i = 0; i < n; ++i) {
        st[i].k = &k[i]; st[i].plan = NULL; st[i].L = L; st[i].race_batch = batch;
        st[i].in = in; st[i].out = out;
        c[i].name = k[i].name; c[i].setup = gr__plansetup; c[i].run = gr__planrun;
        c[i].teardown = gr__planteardown; c[i].ctx = &st[i];
    }
    int w = gr_pick(key, c, n, o, pi);
    return w < 0 ? 0 : w;
}

#endif /* D1_RACE_C_INCLUDED */

/* ============================================================================
 * THE DEMO ENTRY
 * ============================================================================ */
#ifndef D1_RACE_LIB_ONLY

#include <dlfcn.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "../fft1d_api.h"

/* ---------------- self engine: Stockham pow2 + Bluestein ------------------- */
/* The in-file trust anchor: correct for every supported L, used as the gate
 * reference and the existence fallback.  Deliberately simple scalar code with
 * manual complex multiplies (no C99 complex-mul NaN branches). */

static inline double _Complex d1r_cmul(double _Complex a, double _Complex b)
{
    return (creal(a) * creal(b) - cimag(a) * cimag(b))
         + I * (creal(a) * cimag(b) + cimag(a) * creal(b));
}

typedef struct {
    int L, M, pow2;
    double _Complex *w;     /* M/2 twiddles e^{-2 pi i k / M}                  */
    double _Complex *chirp; /* L entries e^{-i pi k^2 / L} (Bluestein only)    */
    double _Complex *bf;    /* M: DFT(chirp-kernel) with 1/M folded in         */
    double _Complex *s0, *s1; /* scratch, M each                               */
} d1r_self;

/* Stockham DIF, natural-order output; returns the buffer holding the result. */
static double _Complex *d1r_p2core(int n, double _Complex *a, double _Complex *b,
                                   const double _Complex *w)
{
    int s = 1;
    double _Complex *src = a, *dst = b;
    for (int h = n >> 1; h >= 1; h >>= 1, s <<= 1) {
        for (int p = 0; p < h; ++p) {
            const double _Complex wp = w[(size_t)p * s];
            const double _Complex *u = src + (size_t)s * p;
            const double _Complex *v = src + (size_t)s * (p + h);
            double _Complex *d0 = dst + (size_t)s * 2 * p;
            double _Complex *d1 = dst + (size_t)s * (2 * p + 1);
            for (int q = 0; q < s; ++q) {
                double _Complex uu = u[q], vv = v[q];
                d0[q] = uu + vv;
                d1[q] = d1r_cmul(uu - vv, wp);
            }
        }
        double _Complex *t = src; src = dst; dst = t;
    }
    return src;
}

static void d1r_self_destroy(d1r_self *s)
{
    if (!s) return;
    free(s->w); free(s->chirp); free(s->bf); free(s->s0); free(s->s1);
    free(s);
}

static d1r_self *d1r_self_create(int L)
{
    d1r_self *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->L = L;
    s->pow2 = (L >= 2 && (L & (L - 1)) == 0);
    int M = L;
    if (!s->pow2) {
        M = 1;
        while (M < 2 * L - 1) M <<= 1;
    }
    s->M = M;
    s->w = malloc((size_t)(M / 2) * sizeof *s->w);
    s->s0 = malloc((size_t)M * sizeof *s->s0);
    s->s1 = malloc((size_t)M * sizeof *s->s1);
    if (!s->w || !s->s0 || !s->s1) { d1r_self_destroy(s); return NULL; }
    for (int k = 0; k < M / 2; ++k) {
        double ph = -2.0 * M_PI * (double)k / (double)M;
        s->w[k] = cos(ph) + I * sin(ph);
    }
    if (!s->pow2) {
        s->chirp = malloc((size_t)L * sizeof *s->chirp);
        s->bf = malloc((size_t)M * sizeof *s->bf);
        if (!s->chirp || !s->bf) { d1r_self_destroy(s); return NULL; }
        for (int k = 0; k < L; ++k) {
            unsigned long long k2 = ((unsigned long long)k * (unsigned long long)k)
                                    % (2ULL * (unsigned long long)L);
            double ph = -M_PI * (double)k2 / (double)L;
            s->chirp[k] = cos(ph) + I * sin(ph);
        }
        /* kernel b_j = e^{+i pi j^2/L}, wrapped; spectrum with 1/M folded in */
        memset(s->s0, 0, (size_t)M * sizeof *s->s0);
        s->s0[0] = 1.0;
        for (int j = 1; j < L; ++j) {
            double _Complex v = conj(s->chirp[j]);
            s->s0[j] = v;
            s->s0[M - j] = v;
        }
        double _Complex *fb = d1r_p2core(M, s->s0, s->s1, s->w);
        double invm = 1.0 / (double)M;
        for (int k = 0; k < M; ++k) s->bf[k] = fb[k] * invm;
    }
    return s;
}

static void d1r_self_exec1(d1r_self *s, const double _Complex *x, double _Complex *y)
{
    if (s->pow2) {
        memcpy(s->s0, x, (size_t)s->L * sizeof *s->s0);
        double _Complex *r = d1r_p2core(s->L, s->s0, s->s1, s->w);
        memcpy(y, r, (size_t)s->L * sizeof *y);
        return;
    }
    const int L = s->L, M = s->M;
    for (int j = 0; j < L; ++j) s->s0[j] = d1r_cmul(x[j], s->chirp[j]);
    memset(s->s0 + L, 0, (size_t)(M - L) * sizeof *s->s0);
    double _Complex *fa = d1r_p2core(M, s->s0, s->s1, s->w);
    double _Complex *ob = (fa == s->s0) ? s->s1 : s->s0;
    for (int k = 0; k < M; ++k) fa[k] = conj(d1r_cmul(fa[k], s->bf[k]));
    double _Complex *fc = d1r_p2core(M, fa, ob, s->w);
    for (int k = 0; k < L; ++k) y[k] = d1r_cmul(conj(fc[k]), s->chirp[k]);
}

static void d1r_self_exec(d1r_self *s, const double _Complex *in,
                          double _Complex *out, int B)
{
    for (int b = 0; b < B; ++b)
        d1r_self_exec1(s, in + (size_t)b * s->L, out + (size_t)b * s->L);
}

/* the driver's map, op for op:  z -> (z+c)/(1+|z+c|) */
static void d1r_map(const double _Complex *z, const double _Complex *cf,
                    double _Complex *dst, size_t n)
{
    const double *zr = (const double *)z;
    const double *cr = (const double *)cf;
    double *o = (double *)dst;
    for (size_t i = 0; i < n; ++i) {
        double re = zr[2 * i] + cr[2 * i];
        double im = zr[2 * i + 1] + cr[2 * i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        o[2 * i] = re * sc;
        o[2 * i + 1] = im * sc;
    }
}

/* ---------------- foreign arms: compile + dlopen + fork-gate ---------------- */

/* Arm roster: the eight sibling sources, plus VARIANT LANES (gen_r12's -D knob
 * mechanism, d1_r2): same source compiled with extra -D flags, raced as a
 * separate candidate.  A variant's hash mixes the flag string into the source
 * hash, so its wisdom identity is distinct from the base arm's and re-keys on
 * either a source or a flag change.  exe_only=1 keeps a variant out of the
 * chain race when its knob cannot change the chain (budget discipline). */
typedef struct {
    const char *base;   /* impl/<base>.c                                   */
    const char *flags;  /* extra gcc flags, "" for the plain build         */
    const char *suffix; /* appended to base for the label/.so/wisdom name  */
    int exe_only;       /* 1: skip in the chain race                       */
} d1r_armdef;

static const d1r_armdef d1r_defs[] = {
    { "d1_pow2",      "", "", 0 },
    { "d1_batchlane", "", "", 0 },
    { "d1_composite", "", "", 0 },
    { "d1_prime",     "", "", 0 },
    { "d1_rader",     "", "", 0 },
    { "d1_bluestein", "", "", 0 },
    { "d1_planner",   "", "", 0 },
    { "d1_twiddle",   "", "", 0 },
    /* d1_composite's record (d1_r1): zmm 4-transform batched kernel kept under
     * -DUSE_ZMM4, statistical tie with ymm2 on wallaby (SPR) -- the per-host
     * race on the scoring node is the arbiter that A/B was waiting for.  The
     * knob swaps the batched execute only, so exe_only. */
    { "d1_composite", "-DUSE_ZMM4", "+zmm4", 1 },
};
#define D1R_NARMS ((int)(sizeof d1r_defs / sizeof d1r_defs[0]))

typedef struct {
    char base[24];
    char label[40];             /* "<base><suffix>", the human identity   */
    char name[64];              /* "<label>@<hash16>", the wisdom identity */
    const char *flags;          /* extra gcc flags for this lane          */
    int exe_only;
    unsigned long long hash;    /* 0: source missing -> dead arm          */
    void *dl;
    int (*supports)(int);
    fft1d_plan *(*create)(int, int);
    void (*execute)(fft1d_plan *, const double _Complex *, double _Complex *);
    void (*chainfn)(fft1d_plan *, const double _Complex *, const double _Complex *,
                    double _Complex *, int);
    void (*destroy)(fft1d_plan *);
    fft1d_plan *plan;           /* materialized at this (L,B)             */
    int gate_chain;             /* native chain verified                  */
    int dead;                   /* permanently skipped this create        */
} d1r_arm;

typedef struct d1r_ctx d1r_ctx;
struct d1r_ctx {
    int L, B;
    size_t count, bytes;
    int gate_m, m_race;
    d1r_self *self;             /* lazily created */
    d1r_arm *xarm;              /* shipped execute arm (NULL = self) */
    d1r_arm arms[D1R_NARMS];
    /* race buffers, lazily allocated (a wisdom-warm create never touches them) */
    int bufs, refs;
    double _Complex *rin, *rcf, *rout, *rstate, *gout, *ref_exec, *ref_chain;
};

static unsigned long long d1r_hash_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned long long h = 1469598103934665603ULL;
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0)
        for (size_t i = 0; i < n; ++i) { h ^= buf[i]; h *= 1099511628211ULL; }
    fclose(f);
    if (h == 0) h = 1;
    return h;
}

static void d1r_mkdirs(void)
{
    char host[64], d[512];
    gr__host(host, sizeof host);
    mkdir(D1R_DIR "/build", 0777);
    mkdir(D1R_DIR "/build/race1d", 0777);
    snprintf(d, sizeof d, D1R_DIR "/build/race1d/%s", host);
    mkdir(d, 0777);
}

/* Ensure the per-host .so for (label, hash) exists.  Atomic via tmp+rename;
 * compile failures leave a .bad marker so they are not retried every create. */
static int d1r_ensure_so(const d1r_arm *a, char *so, size_t cap)
{
    char host[64];
    gr__host(host, sizeof host);
    snprintf(so, cap, D1R_DIR "/build/race1d/%s/%s_%016llx.so", host, a->label, a->hash);
    if (access(so, R_OK) == 0) return 0;
    char bad[600];
    snprintf(bad, sizeof bad, "%s.bad", so);
    if (access(bad, R_OK) == 0) return -1;
    d1r_mkdirs();
    char tmp[600];
    snprintf(tmp, sizeof tmp, "%s.tmp.%ld", so, (long)getpid());
    char cmd[2048];
    snprintf(cmd, sizeof cmd,
             "command -v gcc >/dev/null 2>&1 || exit 9; "
             "gcc -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno "
             "-funroll-loops %s -shared -fPIC -Wl,-Bsymbolic -I'%s' "
             "-o '%s' '%s/impl/%s.c' -lm > '%s.log' 2>&1",
             a->flags, D1R_DIR, tmp, D1R_DIR, a->base, so);
    int rc = system(cmd);
    if (rc != 0) {
        unlink(tmp);
        FILE *g = fopen(bad, "w");
        if (g) { fprintf(g, "compile failed rc=%d\n", rc); fclose(g); }
        return -1;
    }
    if (rename(tmp, so) != 0) { unlink(tmp); return access(so, R_OK) == 0 ? 0 : -1; }
    return 0;
}

static int d1r_load(d1r_arm *a)
{
    if (a->dl) return 0;
    char so[600];
    if (d1r_ensure_so(a, so, sizeof so) != 0) return -1;
    a->dl = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (!a->dl) return -1;
    *(void **)&a->supports = dlsym(a->dl, "fft1d_supports");
    *(void **)&a->create = dlsym(a->dl, "fft1d_create");
    *(void **)&a->execute = dlsym(a->dl, "fft1d_execute");
    *(void **)&a->chainfn = dlsym(a->dl, "fft1d_chain"); /* optional */
    *(void **)&a->destroy = dlsym(a->dl, "fft1d_destroy");
    if (!a->supports || !a->create || !a->execute || !a->destroy) {
        dlclose(a->dl);
        a->dl = NULL;
        return -1;
    }
    return 0;
}

static double d1r_rel_l2(const double _Complex *x, const double _Complex *ref, size_t n)
{
    double dd = 0.0, rr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double _Complex d = x[i] - ref[i];
        dd += creal(d) * creal(d) + cimag(d) * cimag(d);
        rr += creal(ref[i]) * creal(ref[i]) + cimag(ref[i]) * cimag(ref[i]);
    }
    return rr > 0.0 ? sqrt(dd / rr) : sqrt(dd);
}

static void d1r_fill(double _Complex *x, size_t n, unsigned long long seed, double scale)
{
    unsigned long long s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        double a = ((double)(s >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        double b = ((double)(s >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
        x[i] = scale * (a + I * b);
    }
}

static void *d1r_alloc(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

static int d1r_ensure_self(d1r_ctx *cx)
{
    if (!cx->self) cx->self = d1r_self_create(cx->L);
    return cx->self ? 0 : -1;
}

static int d1r_ensure_bufs(d1r_ctx *cx)
{
    if (cx->bufs) return 0;
    cx->rin = d1r_alloc(cx->bytes);
    cx->rcf = d1r_alloc(cx->bytes);
    cx->rout = d1r_alloc(cx->bytes);
    cx->rstate = d1r_alloc(cx->bytes);
    cx->gout = d1r_alloc(cx->bytes);
    cx->ref_exec = d1r_alloc(cx->bytes);
    cx->ref_chain = d1r_alloc(cx->bytes);
    if (!cx->rin || !cx->rcf || !cx->rout || !cx->rstate || !cx->gout ||
        !cx->ref_exec || !cx->ref_chain)
        return -1;
    d1r_fill(cx->rin, cx->count, 42ULL, 1.0);
    d1r_fill(cx->rcf, cx->count, 900042ULL, 0.1);
    cx->bufs = 1;
    return 0;
}

/* gate references: one self execute + a gate_m-step self chain */
static int d1r_ensure_refs(d1r_ctx *cx)
{
    if (cx->refs) return 0;
    if (d1r_ensure_bufs(cx) != 0 || d1r_ensure_self(cx) != 0) return -1;
    d1r_self_exec(cx->self, cx->rin, cx->ref_exec, cx->B);
    memcpy(cx->ref_chain, cx->rin, cx->bytes);
    for (int s = 0; s < cx->gate_m; ++s) {
        d1r_self_exec(cx->self, cx->ref_chain, cx->rout, cx->B);
        d1r_map(cx->rout, cx->rcf, cx->ref_chain, cx->count);
    }
    cx->refs = 1;
    return 0;
}

/* Gate one arm in a FORKED child: its create/execute/chain cannot hang or
 * crash this binary (10-15 s watchdog, SIGKILL).  Verdict written to `v`:
 * "e1c1" exec+chain ok, "e1c0" exec ok / no usable chain, "e0" bad. */
static void d1r_fork_gate(d1r_ctx *cx, d1r_arm *a, char *v, size_t vc)
{
    snprintf(v, vc, "e0");
    if (d1r_ensure_refs(cx) != 0) return;
    int fd[2];
    if (pipe(fd) != 0) return;
    pid_t pid = fork();
    if (pid < 0) { close(fd[0]); close(fd[1]); return; }
    if (pid == 0) {
        close(fd[0]);
        char res[8] = "e0";
        fft1d_plan *pl = a->create(cx->L, cx->B);
        if (pl) {
            memset(cx->gout, 0, cx->bytes);
            a->execute(pl, cx->rin, cx->gout);
            if (d1r_rel_l2(cx->gout, cx->ref_exec, cx->count) < 2e-12) {
                if (a->chainfn) {
                    a->chainfn(pl, cx->rin, cx->rcf, cx->gout, cx->gate_m);
                    snprintf(res, sizeof res,
                             d1r_rel_l2(cx->gout, cx->ref_chain, cx->count) < 2e-12
                                 ? "e1c1" : "e1c0");
                } else {
                    snprintf(res, sizeof res, "e1c0");
                }
            }
        }
        ssize_t wr = write(fd[1], res, 5);
        (void)wr;
        _exit(0);
    }
    close(fd[1]);
    struct pollfd pf = { .fd = fd[0], .events = POLLIN };
    int ms = 15000;
    double t0 = gr_now_us();
    for (;;) {
        int pr = poll(&pf, 1, ms);
        if (pr > 0) {
            char buf[8] = { 0 };
            if (read(fd[0], buf, 5) >= 4) snprintf(v, vc, "%s", buf);
            break;
        }
        if (pr == 0) { kill(pid, SIGKILL); break; }
        /* pr < 0: EINTR or worse -- recompute the remaining budget */
        ms = 15000 - (int)((gr_now_us() - t0) / 1000.0);
        if (ms <= 0) { kill(pid, SIGKILL); break; }
    }
    close(fd[0]);
    int st;
    waitpid(pid, &st, 0);
}

/* Materialize an arm at (L,B): compile, load, supports, gate (cached verdict
 * per (L, B, source hash)), in-process plan.  0 on success. */
static int d1r_arm_ready(d1r_ctx *cx, d1r_arm *a, int need_chain)
{
    if (a->hash == 0 || a->dead) return -1;
    if (!a->plan) {
        if (d1r_load(a) != 0) { a->dead = 1; return -1; }
        if (!a->supports(cx->L)) { a->dead = 1; return -1; }
        char gkey[GR_KEY_MAX], v[16];
        snprintf(gkey, sizeof gkey, "d1_race/gate.r1/L%d/B%d/%s", cx->L, cx->B, a->name);
        if (!gr_wisdom_get_str(gkey, v, sizeof v)) {
            d1r_fork_gate(cx, a, v, sizeof v);
            gr_wisdom_put_str(gkey, v);
        }
        if (strncmp(v, "e1", 2) != 0) { a->dead = 1; return -1; }
        a->gate_chain = (strstr(v, "c1") != NULL);
        a->plan = a->create(cx->L, cx->B);
        if (!a->plan) { a->dead = 1; return -1; }
    }
    if (need_chain && (!a->chainfn || !a->gate_chain)) return -1;
    return 0;
}

/* ---------------- race thunks ---------------------------------------------- */

typedef struct { d1r_ctx *cx; d1r_arm *a; } d1r_armref;

static void *d1r_su_self(void *ctx)
{
    d1r_ctx *cx = ctx;
    if (d1r_ensure_bufs(cx) != 0 || d1r_ensure_self(cx) != 0) return NULL;
    return ctx;
}

static void d1r_t_selfexec(void *state)
{
    d1r_ctx *cx = state;
    d1r_self_exec(cx->self, cx->rin, cx->rout, cx->B);
}

static void *d1r_su_exe(void *ctx)
{
    d1r_armref *r = ctx;
    if (d1r_ensure_bufs(r->cx) != 0) return NULL;
    return d1r_arm_ready(r->cx, r->a, 0) == 0 ? ctx : NULL;
}

static void d1r_t_armexec(void *state)
{
    d1r_armref *r = state;
    r->a->execute(r->a->plan, r->cx->rin, r->cx->rout);
}

static void d1r_shipped_exec(d1r_ctx *cx, const double _Complex *in, double _Complex *out)
{
    if (cx->xarm) cx->xarm->execute(cx->xarm->plan, in, out);
    else d1r_self_exec(cx->self, in, out, cx->B);
}

static void *d1r_su_loop(void *ctx)
{
    d1r_ctx *cx = ctx;
    if (d1r_ensure_bufs(cx) != 0) return NULL;
    if (!cx->xarm && d1r_ensure_self(cx) != 0) return NULL;
    return ctx;
}

static void d1r_t_loopchain(void *state)
{
    d1r_ctx *cx = state;
    memcpy(cx->rstate, cx->rin, cx->bytes);
    for (int s = 0; s < cx->m_race; ++s) {
        d1r_shipped_exec(cx, cx->rstate, cx->rout);
        d1r_map(cx->rout, cx->rcf, cx->rstate, cx->count);
    }
}

static void *d1r_su_chn(void *ctx)
{
    d1r_armref *r = ctx;
    if (d1r_ensure_bufs(r->cx) != 0) return NULL;
    return d1r_arm_ready(r->cx, r->a, 1) == 0 ? ctx : NULL;
}

static void d1r_t_armchain(void *state)
{
    d1r_armref *r = state;
    r->a->chainfn(r->a->plan, r->cx->rin, r->cx->rcf, r->cx->rstate, r->cx->m_race);
}

/* honest chain length: the graded m for this (L,B) from cases.txt, work-capped
 * so one race unit stays ~10^6.5 points (the gen_r13 lesson: race an honest m,
 * or once-per-chain costs are charged wrongly) */
static int d1r_m_race(int L, int B)
{
    int m = 16;
    FILE *f = fopen(D1R_DIR "/cases.txt", "r");
    if (f) {
        char ln[128];
        while (fgets(ln, sizeof ln, f)) {
            int l, b, mm;
            if (sscanf(ln, "%d:%d:%d", &l, &b, &mm) == 3 && l == L && b == B && mm > 1) {
                m = mm;
                break;
            }
        }
        fclose(f);
    }
    long cap = 4000000L / ((long)L * (long)B);
    if (cap < 2) cap = 2;
    if (m > cap) m = (int)cap;
    if (m > 1024) m = 1024;
    if (m < 2) m = 2;
    return m;
}

/* ---------------- the entry ------------------------------------------------- */

struct fft1d_plan {
    /* flattened dispatch (d1_r2): what ships is EXACTLY what was raced -- one
     * indirect call, no branch, no pointer chase through cx/arm.  The r1
     * forwarding paid a branch + two dependent loads (plan -> xarm -> its
     * execute/plan fields, a second cache line) on every scored call.  These
     * four fields sit FIRST so the whole dispatch lives in the plan's first
     * cache line (placed after cx they measurably cost ~1 ns at L=13 B=1). */
    void (*exec_fn)(fft1d_plan *, const double _Complex *, double _Complex *);
    fft1d_plan *exec_arg;
    void (*chain_fn)(fft1d_plan *, const double _Complex *, const double _Complex *,
                     double _Complex *, int);
    fft1d_plan *chain_arg;
    double _Complex *tmp;       /* chain fallback FFT-output scratch          */
    d1r_arm *carm;              /* native chain winner (NULL = loop fallback) */
    d1r_ctx cx;
};

/* self/loop adapters: the arg is the outer plan itself */
static void d1r_exec_self(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    d1r_self_exec(p->cx.self, in, out, p->cx.B);
}

static void d1r_chain_loop(fft1d_plan *p, const double _Complex *x0,
                           const double _Complex *c, double _Complex *final_out, int m)
{
    memcpy(final_out, x0, p->cx.bytes);
    for (int s = 0; s < m; ++s) {
        p->exec_fn(p->exec_arg, final_out, p->tmp);
        d1r_map(p->tmp, c, final_out, p->cx.count);
    }
}

const char *fft1d_name(void) { return "d1_race"; }

const char *fft1d_description(void)
{
    return "LIBRARY LAYER (adoption-scored): plan-time race + per-host wisdom "
           "(gen_race generalized to 1D); demo entry fork-gates and races the "
           "sibling class entries per (L,B) and ships the winner by vtable";
}

int fft1d_supports(int L) { return L >= 2 && L <= (1 << 20); }

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L) || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    d1r_ctx *cx = &p->cx;
    cx->L = L;
    cx->B = batch;
    cx->count = (size_t)L * (size_t)batch;
    cx->bytes = cx->count * sizeof(double _Complex);
    cx->gate_m = 2;
    cx->m_race = d1r_m_race(L, batch);
    p->tmp = d1r_alloc(cx->bytes);
    if (!p->tmp) { free(p); return NULL; }

    int eng = !getenv("D1_RACE_NO_ENG");
    if (eng) {
        for (int i = 0; i < D1R_NARMS; ++i) {
            d1r_arm *a = &cx->arms[i];
            const d1r_armdef *d = &d1r_defs[i];
            snprintf(a->base, sizeof a->base, "%s", d->base);
            snprintf(a->label, sizeof a->label, "%s%s", d->base, d->suffix);
            a->flags = d->flags;
            a->exe_only = d->exe_only;
            char src[512];
            snprintf(src, sizeof src, D1R_DIR "/impl/%s.c", a->base);
            a->hash = d1r_hash_file(src);
            if (a->hash && d->flags[0]) { /* mix the flags into the identity */
                unsigned long long h = a->hash;
                for (const char *s = d->flags; *s; ++s) {
                    h ^= (unsigned char)*s;
                    h *= 1099511628211ULL;
                }
                a->hash = h ? h : 1;
            }
            snprintf(a->name, sizeof a->name, "%s@%016llx", a->label, a->hash);
        }

        /* ---- stage exe.r2: whole-batch execute, self = candidate 0 ----
         * exe.r2 (d1_r2): at big cells the scalar self engine is excluded from
         * the race.  r1 measured self at ~75 ms/vector at L=100003 (Bluestein
         * M=262144, three scalar passes); as candidate 0 it ate ~3 s of the
         * 5 s race budget and squeezed the real contenders' samples exactly
         * where their margins are ~5%.  Self never wins those cells -- it only
         * needs to EXIST (the ship-time fallback below is unconditional).
         * Cost model: pow2 self ~ L points/pass, Bluestein ~ 3 passes at
         * M<=4L -> ~12L; skip the race lane when est points x batch > 2^20. */
        size_t selfcost = (size_t)(((L & (L - 1)) == 0) ? L : 12 * L)
                        * (size_t)batch;
        int race_self = selfcost <= ((size_t)1 << 20);
        d1r_armref xr[D1R_NARMS];
        gr_cand xc[1 + D1R_NARMS];
        int nx = 0, na = 0;
        if (race_self) {
            xc[0].name = "self";
            xc[0].setup = d1r_su_self;
            xc[0].run = d1r_t_selfexec;
            xc[0].teardown = NULL;
            xc[0].ctx = cx;
            nx = 1;
        }
        for (int i = 0; i < D1R_NARMS; ++i) {
            if (cx->arms[i].hash == 0) continue;
            xr[na].cx = cx;
            xr[na].a = &cx->arms[i];
            xc[nx].name = cx->arms[i].name;
            xc[nx].setup = d1r_su_exe;
            xc[nx].run = d1r_t_armexec;
            xc[nx].teardown = NULL;
            xc[nx].ctx = &xr[na];
            ++na;
            ++nx;
        }
        if (nx > 0) {
            char key[GR_KEY_MAX];
            gr_keyf(key, sizeof key, "d1_race", "exe.r2", L, gr_bucket(batch));
            gr_pick_info pi;
            int wx = gr_pick(key, xc, nx, NULL, &pi);
            if (wx >= 0 && !(race_self && wx == 0)) {
                d1r_arm *a = ((d1r_armref *)xc[wx].ctx)->a;
                if (d1r_arm_ready(cx, a, 0) == 0) cx->xarm = a;
            }
        }

        /* ---- stage chn.r1: the graded map chain at an honest m ---- */
        char loopname[64];
        snprintf(loopname, sizeof loopname, "loop:%s",
                 cx->xarm ? cx->xarm->name : "self");
        d1r_armref cr[D1R_NARMS];
        char cname[D1R_NARMS][56];
        gr_cand cc[1 + D1R_NARMS];
        cc[0].name = loopname;
        cc[0].setup = d1r_su_loop;
        cc[0].run = d1r_t_loopchain;
        cc[0].teardown = NULL;
        cc[0].ctx = cx;
        int nc = 1;
        for (int i = 0; i < D1R_NARMS; ++i) {
            if (cx->arms[i].hash == 0 || cx->arms[i].exe_only) continue;
            cr[nc - 1].cx = cx;
            cr[nc - 1].a = &cx->arms[i];
            snprintf(cname[nc - 1], sizeof cname[0], "c:%s", cx->arms[i].name);
            cc[nc].name = cname[nc - 1];
            cc[nc].setup = d1r_su_chn;
            cc[nc].run = d1r_t_armchain;
            cc[nc].teardown = NULL;
            cc[nc].ctx = &cr[nc - 1];
            ++nc;
        }
        char ckey[GR_KEY_MAX];
        char ctag[32];
        snprintf(ctag, sizeof ctag, "chn.r1.m%d", cx->m_race);
        gr_keyf(ckey, sizeof ckey, "d1_race", ctag, L, gr_bucket(batch));
        gr_opts co = gr_default_opts();
        co.budget_us = 8e6;
        gr_pick_info ci;
        int wc = gr_pick(ckey, cc, nc, &co, &ci);
        if (wc > 0) {
            d1r_arm *a = ((d1r_armref *)cc[wc].ctx)->a;
            if (d1r_arm_ready(cx, a, 1) == 0) p->carm = a;
        }
    }

    /* ship-time guarantee: execute must always have an engine */
    if (!cx->xarm && d1r_ensure_self(cx) != 0) {
        free(p->tmp);
        free(p);
        return NULL;
    }

    /* flatten the winners into single indirect calls */
    if (cx->xarm) {
        p->exec_fn = cx->xarm->execute;
        p->exec_arg = cx->xarm->plan;
    } else {
        p->exec_fn = d1r_exec_self;
        p->exec_arg = p;
    }
    if (p->carm) {
        p->chain_fn = p->carm->chainfn;
        p->chain_arg = p->carm->plan;
    } else {
        p->chain_fn = d1r_chain_loop;
        p->chain_arg = p;
    }

    /* race buffers are plan-time only */
    free(cx->rin); free(cx->rcf); free(cx->rout); free(cx->rstate);
    free(cx->gout); free(cx->ref_exec); free(cx->ref_chain);
    cx->rin = cx->rcf = cx->rout = cx->rstate = NULL;
    cx->gout = cx->ref_exec = cx->ref_chain = NULL;
    cx->bufs = cx->refs = 0;

    if (getenv("D1_RACE_VERBOSE"))
        fprintf(stderr, "d1_race: L=%d B=%d ships exec=%s chain=%s (m_race=%d)\n",
                L, batch, cx->xarm ? cx->xarm->name : "self",
                p->carm ? p->carm->name : "loop", cx->m_race);
    return p;
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    p->exec_fn(p->exec_arg, in, out);
}

void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    p->chain_fn(p->chain_arg, x0, c, final_out, m);
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    d1r_ctx *cx = &p->cx;
    for (int i = 0; i < D1R_NARMS; ++i) {
        d1r_arm *a = &cx->arms[i];
        if (a->plan) a->destroy(a->plan);
        if (a->dl) dlclose(a->dl);
    }
    d1r_self_destroy(cx->self);
    free(cx->rin); free(cx->rcf); free(cx->rout); free(cx->rstate);
    free(cx->gout); free(cx->ref_exec); free(cx->ref_chain);
    free(p->tmp);
    free(p);
}

#endif /* !D1_RACE_LIB_ONLY */
