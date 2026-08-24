/* =============================================================================
 * GEN_RACE -- the plan-time race + per-host wisdom cache library layer (gen_r1)
 * =============================================================================
 *
 * LIBRARY LAYER, scored by ADOPTION.  Class owners: adopt with
 *
 *     #define GEN_RACE_LIB_ONLY
 *     #include "gen_race.c"          // impl/ is the include dir; everything is
 *                                    // `static`, prefix gr_, no symbols leak
 *
 * (same pattern as gen_layout -- the two compose; include either or both).
 *
 * WHAT THIS SOLVES.  The seed (seed_fft3d_best/fft3d_best.c) proved the choice
 * of kernel is a property of the MACHINE, not the algorithm: the L=64 pair
 * inverts by 4.3x between Haswell and Cascade Lake, and the Ice Lake grading
 * winner differed from the Cascade Lake winner at three of eight sizes.  This
 * campaign adds the plan-time budget: <= 60 s for a never-seen (L,B) INCLUDING
 * racing, <= 50 ms with persisted wisdom, and a cross-arch guard every second
 * round that the race must win on three microarchitectures.  So the layer is:
 *
 *   1. gr_time_run   -- the timing methodology, alone.  Warmup calls, inner-rep
 *      calibration until a sample clears min_sample_us of timer resolution,
 *      min over several samples, spread reported.  The same discipline the
 *      driver grades with, so plan-time decisions rank the way scores rank.
 *   2. gr_race       -- race N candidate thunks (optional setup/teardown, e.g.
 *      plan create/destroy).  NOISE-AWARE: all candidates within noise_rel of
 *      the fastest form a statistical tie group and the LOWEST-INDEX member
 *      wins.  That is the seed's L8/L36 lesson verbatim: "selecting on batch
 *      there would encode measurement noise" -- put your primary first and a
 *      noise-level rival cannot displace it, so the choice is stable across
 *      runs even without wisdom.
 *   3. gr_pick       -- gr_race behind the per-host wisdom cache
 *      (results/wisdom_<host>.json).  Cache hit: no candidate is even built,
 *      just a file read -- microseconds against the 50 ms budget.  Keys carry
 *      an FNV signature of the candidate-name list, so changing your candidate
 *      set next round invalidates stale wisdom automatically instead of
 *      silently replaying it.  flock + write-temp + rename keep 12 implementer
 *      binaries from corrupting the shared file.
 *   4. gr_pick_value -- integer-knob sweep sugar (block size, tile shape, map
 *      variant...) over the same race + wisdom machinery.  gen_dense_prime's
 *      "race for the map variant + BC/tile knobs" ask is this call.
 *   5. gr_pick_plan  -- whole-plan race: N {create,execute,destroy} vtables,
 *      capped race batch (32 MB buffers max, the seed's cap), deterministic
 *      input fill, candidates that fail to plan are skipped, Bluestein-style
 *      existence fallback = candidate 0.  This is the generalized
 *      fft3d_best choose()/trial() that the round-6 assembled library
 *      (planner enumerates -> race picks) will sit on.
 *
 * ENVIRONMENT PINS (all optional, for measuring something else):
 *     GEN_RACE_NO_RACE=1    always take candidate 0 (the primary)
 *     GEN_RACE_FORCE=name   take the candidate with this name where present
 *     GEN_RACE_REFRESH=1    ignore cached wisdom, re-race, overwrite
 *     GEN_RACE_NO_WISDOM=1  race every time, never read or write the file
 *     GEN_RACE_WISDOM=path  wisdom file override (default results/wisdom_<host>.json)
 *     GEN_RACE_VERBOSE=1    print every decision to stderr
 *
 * HOW TO ADOPT (60 seconds).  In your create(), for an in-plan variant knob:
 *
 *     #define GEN_RACE_LIB_ONLY
 *     #include "gen_race.c"
 *
 *     struct myctx { my_plan *p; int variant; ... };   // your thunk state
 *     static void run_v(void *s) { struct myctx *c = s; my_step(c->p, c->variant); }
 *
 *     char key[GR_KEY_MAX];
 *     gr_keyf(key, sizeof key, "gen_powp", "ypass_tile", L, gr_bucket(batch));
 *     gr_cand c[2] = { { "tile2", 0, run_v, 0, &ctx_a },     // primary FIRST
 *                      { "tile4", 0, run_v, 0, &ctx_b } };
 *     int v = gr_pick(key, c, 2, NULL, NULL);               // NULL = defaults
 *
 * Race your GRADED workload (your chain step), not a proxy: the thunk is yours.
 * Allocate race-only buffers in a setup() callback, not before gr_pick -- on a
 * wisdom hit setup() never runs and create() stays inside the 50 ms budget.
 *
 * THE DEMO ENTRY (when not GEN_RACE_LIB_ONLY): a generic any-L (2..128) dense
 * row-column DFT carrying THREE genuinely different contraction variants
 * (scalar k-outer, j-outer axpy, k-quad x 8-wide register tile) picked per
 * (L, B-bucket, host) by gr_pick at create().  Deliberately O(L^4) floor
 * class -- it is the layer's living test bench, NOT a contender.  What it
 * demonstrates under the real driver: cold create() races and persists, warm
 * create() is a file read, and the driver's two-process repeatability cmp
 * passes BECAUSE wisdom pins run 2 to run 1's winner (different variants
 * round differently, so an unpersisted noise-flip would show NOT REPEATABLE).
 * ============================================================================= */

#ifndef GEN_RACE_C_INCLUDED
#define GEN_RACE_C_INCLUDED

#if !defined(_GNU_SOURCE) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
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

/* ---- 1. timing methodology ------------------------------------------------ */

typedef struct {
    int warmups;          /* untimed calls before anything (default 2)          */
    int samples;          /* timed samples after calibration (default 5)        */
    double min_sample_us; /* calibrate inner reps until a sample >= this (200)  */
    double budget_us;     /* soft cap for one whole gr_race call (default 5e6)  */
    double noise_rel;     /* statistical-tie width, relative (default 0.02)     */
} gr_opts;

static inline gr_opts gr_default_opts(void)
{
    gr_opts o;
    o.warmups = 2;
    o.samples = 5;
    o.min_sample_us = 200.0;
    o.budget_us = 5e6;
    o.noise_rel = 0.02;
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

/* Time one thunk: warmups, rep calibration to clear timer resolution, then
 * min/max over `samples` samples.  `deadline_us` (absolute, 0 = none) can cut
 * sampling short but never below one calibrated sample. */
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
    int tie;          /* 1: runner-up within noise_rel; pick is stability,  */
                      /*    not measurement                                  */
    int from_wisdom;  /* 1: no race ran, cached answer                       */
    double us;        /* winner's time (cached or fresh)                     */
    double margin;    /* (runnerup - winner)/winner, 0 if none/cached        */
} gr_pick_info;

/* Race n candidates.  Winner: lowest-index member of the statistical tie group
 * around the fastest (noise doctrine above).  Returns -1 if nothing ran.
 * `tm` (may be NULL) receives per-candidate timings. */
static inline int gr_race(const gr_cand *c, int n, const gr_opts *o_in,
                          gr_timing *tm, gr_pick_info *pi)
{
    gr_opts o = o_in ? *o_in : gr_default_opts();
    gr_timing local[32];
    gr_timing *t = tm;
    if (!t && n <= 32) t = local;
    if (!t) return n > 0 ? 0 : -1; /* too many candidates, no scratch: primary */

    double deadline = o.budget_us > 0.0 ? gr_now_us() + o.budget_us : 0.0;
    for (int i = 0; i < n; ++i) {
        t[i].ok = 0;
        t[i].us = 0.0;
        t[i].spread_rel = 0.0;
        t[i].reps = 0;
        if (deadline > 0.0 && gr_now_us() > deadline && i > 0) continue; /* budget: skip */
        void *state = c[i].setup ? c[i].setup(c[i].ctx) : c[i].ctx;
        if (c[i].setup && !state) continue; /* will not plan: skipped; NULL ctx without setup is legal */
        gr_time_run(c[i].run, state, &o, deadline, &t[i]);
        if (c[i].teardown) c[i].teardown(state);
    }

    int best = -1;
    for (int i = 0; i < n; ++i)
        if (t[i].ok && (best < 0 || t[i].us < t[best].us)) best = i;
    if (best < 0) return -1;

    /* statistical tie group -> lowest index wins */
    int widx = best;
    for (int i = 0; i < best; ++i)
        if (t[i].ok && t[i].us <= t[best].us * (1.0 + o.noise_rel)) { widx = i; break; }

    if (pi) {
        double runner = -1.0;
        for (int i = 0; i < n; ++i)
            if (t[i].ok && i != widx && (runner < 0 || t[i].us < runner)) runner = t[i].us;
        pi->widx = widx;
        pi->us = t[widx].us;
        pi->margin = runner > 0.0 ? (runner - t[widx].us) / t[widx].us : 0.0;
        pi->tie = (runner > 0.0 && pi->margin < o.noise_rel) ? 1 : 0;
        pi->from_wisdom = 0;
    }
    return widx;
}

/* ---- 3. wisdom cache ------------------------------------------------------- */

#define GR_KEY_MAX 160

/* Canonical key: "<entry>/<tag>/L<L>/B<B>".  Key on gr_bucket(batch), not the
 * exact batch, or every B races separately. */
static inline void gr_keyf(char *buf, size_t cap, const char *entry,
                           const char *tag, int L, int B)
{
    snprintf(buf, cap, "%s/%s/L%d/B%d", entry, tag, L, B);
}

/* Power-of-two batch bucket, capped at 128. */
static inline int gr_bucket(int b)
{
    int p = 1;
    while (p * 2 <= b && p < 128) p *= 2;
    return p;
}

/* FNV-1a over the candidate-name list: the candidate-set signature.  A changed
 * list next round makes a different key -> stale wisdom misses, no lies. */
static inline unsigned gr_sig(const gr_cand *c, int n)
{
    unsigned h = 2166136261u;
    for (int i = 0; i < n; ++i) {
        const char *s = c[i].name ? c[i].name : "?";
        while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
        h ^= 0xffu; h *= 16777619u; /* name separator */
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

/* Wisdom file: GEN_RACE_WISDOM, else results/wisdom_<host>.json under cwd if
 * results/ exists (the driver runs from the gen dir), else the campaign's
 * absolute results/ on the shared FS, else none (returns 0: race every time). */
static inline int gr_wisdom_path(char *buf, size_t cap)
{
    const char *env = getenv("GEN_RACE_WISDOM");
    if (env && *env) { snprintf(buf, cap, "%s", env); return 1; }
    char host[64];
    gr__host(host, sizeof host);
    struct stat st;
    if (stat("results", &st) == 0 && S_ISDIR(st.st_mode)) {
        snprintf(buf, cap, "results/wisdom_%s.json", host);
        return 1;
    }
    const char *fixed = "/home/lqcd/wdetmold/fft/bench/gen/results";
    if (stat(fixed, &st) == 0 && S_ISDIR(st.st_mode)) {
        snprintf(buf, cap, "%s/wisdom_%s.json", fixed, host);
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

/* Look up fullkey.  On hit: winner name (into wname), stored index, tie flag,
 * stored time.  Returns 1 on hit. */
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

/* Persist one decision.  Read-merge-rewrite of the whole file (entries are one
 * line each), under flock, written to a temp and rename()d -- atomic on the
 * shared FS, so concurrent implementer binaries cannot corrupt it. */
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
            char *p = strstr(old, "\"entries\":{");
            if (p) {
                p = strchr(p, '\n');
                char *line = p ? strtok(p, "\n") : NULL;
                size_t klen = strlen(fullkey);
                while (line) {
                    char *q = line;
                    while (*q == ' ' || *q == '\t') ++q;
                    if (*q == '"') {
                        char *k0 = q + 1;
                        char *k1 = strchr(k0, '"');
                        if (k1 && !((size_t)(k1 - k0) == klen &&
                                    memcmp(k0, fullkey, klen) == 0)) {
                            size_t ll = strlen(q);
                            while (ll && (q[ll - 1] == ',' || q[ll - 1] == ' ' ||
                                          q[ll - 1] == '\r'))
                                q[--ll] = 0;
                            if (ll) fprintf(f, ",\n%s", q);
                        }
                    }
                    line = strtok(NULL, "\n");
                }
            }
        }
        fprintf(f, "\n}}\n");
        fclose(f);
        rename(tmp, path); /* failure leaves the old file intact */
    }
    free(old);
    if (lfd >= 0) { flock(lfd, LOCK_UN); close(lfd); }
}

/* ---- 4. gr_pick: the race behind the cache --------------------------------- */

static inline int gr_pick(const char *key, const gr_cand *c, int n,
                          const gr_opts *o, gr_pick_info *pi)
{
    gr_pick_info local;
    if (!pi) pi = &local;
    pi->widx = 0; pi->tie = 0; pi->from_wisdom = 0; pi->us = 0.0; pi->margin = 0.0;
    if (n <= 0) return -1;
    if (n == 1) return 0;

    const char *force = getenv("GEN_RACE_FORCE");
    if (force && *force)
        for (int i = 0; i < n; ++i)
            if (c[i].name && strcmp(c[i].name, force) == 0) { pi->widx = i; return i; }
    if (getenv("GEN_RACE_NO_RACE")) return 0;

    char fullkey[GR_KEY_MAX + 16];
    snprintf(fullkey, sizeof fullkey, "%s#%08x", key, gr_sig(c, n));

    int use_wisdom = !getenv("GEN_RACE_NO_WISDOM");
    if (use_wisdom && !getenv("GEN_RACE_REFRESH")) {
        char wname[64];
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
                if (getenv("GEN_RACE_VERBOSE"))
                    fprintf(stderr, "gen_race: %s -> %s (wisdom, %.3g us)\n",
                            fullkey, wname, us);
                return found;
            }
        }
    }

    int widx = gr_race(c, n, o, NULL, pi);
    if (widx < 0) { pi->widx = 0; return 0; } /* nothing ran: primary, unpersisted */
    if (use_wisdom)
        gr_wisdom_store(fullkey, c[widx].name ? c[widx].name : "?",
                        widx, pi->tie, pi->us, pi->margin);
    if (getenv("GEN_RACE_VERBOSE"))
        fprintf(stderr, "gen_race: %s -> %s (raced, %.3g us, margin %.1f%%%s)\n",
                fullkey, c[widx].name ? c[widx].name : "?", pi->us,
                100.0 * pi->margin, pi->tie ? ", TIE: stability pick" : "");
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

/* Race the same thunk over `vals` (e.g. tile widths).  Returns the winning
 * VALUE (vals[0] on any failure).  Candidate names are "v<val>". */
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

/* ---- 6. whole-plan race: the generalized fft3d_best choose()/trial() ------- */

typedef struct {
    const char *name;
    void *(*create)(int L, int batch); /* NULL: will not plan -> skipped        */
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

/* Race whole plans at (L, batch).  The race batch is capped at ~32 MB per
 * buffer (the seed's cap: the ranking was stable across batch on a machine, so
 * planning stays cheap).  Deterministic input fill.  Returns the winning
 * index; 0 if nothing planned or buffers failed (caller's candidate 0 is the
 * existence fallback -- put Bluestein or your safest variant there). */
static inline int gr_pick_plan(const char *key, const gr_plan_cand *k, int n,
                               int L, int batch, const gr_opts *o, gr_pick_info *pi)
{
    enum { CAP = 16 };
    gr__planstate st[CAP];
    gr_cand c[CAP];
    if (n <= 0) return -1;
    if (n > CAP) n = CAP;
    if (n == 1) return 0;

    long vol = (long)L * L * L;
    int rb = (int)(2097152L / (vol > 0 ? vol : 1));
    if (rb < 1) rb = 1;
    if (rb > batch) rb = batch;

    /* On a wisdom hit no buffer is needed; allocate only if we might race. */
    char fullkey[GR_KEY_MAX + 16];
    {
        gr_cand probe[CAP];
        for (int i = 0; i < n; ++i) { probe[i] = (gr_cand){ k[i].name, 0, 0, 0, 0 }; }
        snprintf(fullkey, sizeof fullkey, "%s#%08x", key, gr_sig(probe, n));
    }
    if (!getenv("GEN_RACE_NO_WISDOM") && !getenv("GEN_RACE_REFRESH") &&
        !getenv("GEN_RACE_FORCE") && !getenv("GEN_RACE_NO_RACE")) {
        char wname[64];
        int widx = -1, tie = 0;
        double us = 0.0;
        if (gr_wisdom_lookup(fullkey, wname, sizeof wname, &widx, &tie, &us)) {
            for (int i = 0; i < n; ++i)
                if (k[i].name && !strcmp(k[i].name, wname)) {
                    if (pi) { pi->widx = i; pi->tie = tie; pi->from_wisdom = 1;
                              pi->us = us; pi->margin = 0.0; }
                    return i;
                }
        }
    }

    size_t count = (size_t)vol * (size_t)rb;
    size_t bytes = count * sizeof(double _Complex);
    double _Complex *in = NULL, *out = NULL;
    if (posix_memalign((void **)&in, 64, bytes) != 0 || !in) return 0;
    if (posix_memalign((void **)&out, 64, bytes) != 0 || !out) { free(in); return 0; }
    memset(out, 0, bytes);
    for (size_t j = 0; j < count; ++j)
        in[j] = (double)((j * 2654435761u) % 1000) / 500.0 - 1.0;

    for (int i = 0; i < n; ++i) {
        st[i].k = &k[i]; st[i].plan = NULL; st[i].L = L; st[i].race_batch = rb;
        st[i].in = in; st[i].out = out;
        c[i].name = k[i].name; c[i].setup = gr__plansetup; c[i].run = gr__planrun;
        c[i].teardown = gr__planteardown; c[i].ctx = &st[i];
    }
    int w = gr_pick(key, c, n, o, pi);
    free(in);
    free(out);
    return w < 0 ? 0 : w;
}

/* =============================================================================
 * DEMO ENTRY -- compiled only when this file IS the entry TU.
 * Any-L dense row-column DFT, three contraction variants raced per
 * (L, B-bucket, host).  The layer's living test bench; not a contender.
 * ============================================================================= */
#ifndef GEN_RACE_LIB_ONLY

#include <math.h>

#include "../fft3d_api.h"

struct fft3d_plan {
    int L, batch, variant;
    double _Complex *w; /* L x L DFT matrix, row-major, long-double built */
    double *wr, *wi;    /* split copies (tile variant) */
    double _Complex *tmp; /* one volume of scratch */
};

const char *fft3d_name(void) { return "gen_race"; }
const char *fft3d_description(void)
{
    return "LIBRARY LAYER (scored by adoption): plan-time candidate race + per-host "
           "wisdom cache, results/wisdom_<host>.json (adopt: #define GEN_RACE_LIB_ONLY "
           "+ #include gen_race.c); demo races 3 dense any-L variants at create()";
}
int fft3d_supports(int L) { return L >= 2 && L <= 128; }

/* out[k*inner+c] = sum_j w[k*L+j] * in[j*inner+c] -- contract the slowest axis
 * of an (L, inner) block.  Three variants, same arithmetic, different shapes. */

static void pass_scalar(const fft3d_plan *p, const double _Complex *in,
                        double _Complex *out, int inner)
{
    const int L = p->L;
    for (int k = 0; k < L; ++k)
        for (int c = 0; c < inner; ++c) {
            double _Complex acc = 0.0;
            for (int j = 0; j < L; ++j)
                acc += p->w[(size_t)k * L + j] * in[(size_t)j * inner + c];
            out[(size_t)k * inner + c] = acc;
        }
}

static void pass_axpy(const fft3d_plan *p, const double _Complex *in,
                      double _Complex *out, int inner)
{
    const int L = p->L;
    memset(out, 0, (size_t)L * inner * sizeof *out);
    for (int j = 0; j < L; ++j)
        for (int k = 0; k < L; ++k) {
            const double _Complex wkj = p->w[(size_t)k * L + j];
            const double _Complex *src = in + (size_t)j * inner;
            double _Complex *dst = out + (size_t)k * inner;
            for (int c = 0; c < inner; ++c) dst[c] += wkj * src[c];
        }
}

/* k-quads x 8-wide c-blocks, j innermost, split-real accumulators local (the
 * register-tile shape gen_dense_prime measured 155 -> 33 us/pass with). */
static void pass_tile(const fft3d_plan *p, const double _Complex *in,
                      double _Complex *out, int inner)
{
    enum { KT = 4, CB = 8 };
    const int L = p->L;
    const double *din = (const double *)in;
    double *dout = (double *)out;
    for (int k0 = 0; k0 < L; k0 += KT) {
        const int kt = (L - k0 < KT) ? L - k0 : KT;
        for (int c0 = 0; c0 < inner; c0 += CB) {
            const int cb = (inner - c0 < CB) ? inner - c0 : CB;
            double ar[KT][CB], ai[KT][CB];
            for (int t = 0; t < kt; ++t)
                for (int q = 0; q < cb; ++q) { ar[t][q] = 0.0; ai[t][q] = 0.0; }
            for (int j = 0; j < L; ++j) {
                const double *row = din + 2 * ((size_t)j * inner + c0);
                double re[CB], im[CB];
                for (int q = 0; q < cb; ++q) { re[q] = row[2 * q]; im[q] = row[2 * q + 1]; }
                for (int t = 0; t < kt; ++t) {
                    const double a = p->wr[(size_t)(k0 + t) * L + j];
                    const double b = p->wi[(size_t)(k0 + t) * L + j];
                    for (int q = 0; q < cb; ++q) {
                        ar[t][q] += a * re[q] - b * im[q];
                        ai[t][q] += a * im[q] + b * re[q];
                    }
                }
            }
            for (int t = 0; t < kt; ++t) {
                double *drow = dout + 2 * ((size_t)(k0 + t) * inner + c0);
                for (int q = 0; q < cb; ++q) {
                    drow[2 * q] = ar[t][q];
                    drow[2 * q + 1] = ai[t][q];
                }
            }
        }
    }
}

typedef void (*pass_fn)(const fft3d_plan *, const double _Complex *,
                        double _Complex *, int);
static const pass_fn PASSES[3] = { pass_scalar, pass_axpy, pass_tile };

static void demo_execute_v(fft3d_plan *p, int variant, const double _Complex *in,
                           double _Complex *out, int vols)
{
    const int L = p->L;
    const size_t LL = (size_t)L * L, volume = LL * L;
    pass_fn pass = PASSES[variant];
    for (int b = 0; b < vols; ++b) {
        const double _Complex *src = in + (size_t)b * volume;
        double _Complex *dst = out + (size_t)b * volume;
        pass(p, src, dst, (int)LL);                       /* axis 0 */
        for (int x = 0; x < L; ++x)                       /* axis 1 */
            pass(p, dst + (size_t)x * LL, p->tmp + (size_t)x * LL, L);
        for (size_t r = 0; r < LL; ++r)                   /* axis 2 */
            pass(p, p->tmp + r * (size_t)L, dst + r * (size_t)L, 1);
    }
}

/* Race thunks: buffers allocated LAZILY in setup(), so a wisdom hit allocates
 * nothing and warm create() is a file read. */
struct demo_race {
    fft3d_plan *p;
    double _Complex *rin, *rout;
    int vols;
    int alloc_failed;
};
struct demo_ctx { struct demo_race *r; int variant; };

static void *demo_setup(void *ctx)
{
    struct demo_ctx *c = ctx;
    struct demo_race *r = c->r;
    if (r->alloc_failed) return NULL;
    if (!r->rin) {
        const size_t vol = (size_t)r->p->L * r->p->L * r->p->L;
        const size_t count = vol * (size_t)r->vols;
        const size_t bytes = count * sizeof(double _Complex);
        if (posix_memalign((void **)&r->rin, 64, bytes) != 0 ||
            posix_memalign((void **)&r->rout, 64, bytes) != 0) {
            free(r->rin); r->rin = NULL; r->alloc_failed = 1;
            return NULL;
        }
        memset(r->rout, 0, bytes);
        for (size_t j = 0; j < count; ++j)
            r->rin[j] = (double)((j * 2654435761u) % 1000) / 500.0 - 1.0;
    }
    return ctx;
}

static void demo_run(void *state)
{
    struct demo_ctx *c = state;
    demo_execute_v(c->r->p, c->variant, c->r->rin, c->r->rout, c->r->vols);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch <= 0) return NULL;
    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->variant = 2; /* primary: the tile form (fastest nearly everywhere) */
    p->w = malloc((size_t)L * L * sizeof *p->w);
    p->wr = malloc((size_t)L * L * sizeof *p->wr);
    p->wi = malloc((size_t)L * L * sizeof *p->wi);
    p->tmp = malloc((size_t)L * L * L * sizeof *p->tmp);
    if (!p->w || !p->wr || !p->wi || !p->tmp) {
        free(p->w); free(p->wr); free(p->wi); free(p->tmp); free(p);
        return NULL;
    }
    const long double TWO_PI = 6.283185307179586476925286766559005768L;
    for (int k = 0; k < L; ++k)
        for (int j = 0; j < L; ++j) {
            long double ph = -TWO_PI * (long double)((k * j) % L) / (long double)L;
            double cr = (double)cosl(ph), ci = (double)sinl(ph);
            p->w[(size_t)k * L + j] = cr + ci * I;
            p->wr[(size_t)k * L + j] = cr;
            p->wi[(size_t)k * L + j] = ci;
        }

    /* Race the three variants on the graded operation (full 3-axis execute) at
     * a capped batch; wisdom key on (L, B-bucket, host). */
    long vol = (long)L * L * L;
    int rb = (int)(2097152L / vol);
    if (rb < 1) rb = 1;
    if (rb > batch) rb = batch;
    struct demo_race r = { p, NULL, NULL, rb, 0 };
    struct demo_ctx cx[3] = { { &r, 2 }, { &r, 0 }, { &r, 1 } };
    gr_cand cands[3] = {
        { "tile4x8",    demo_setup, demo_run, NULL, &cx[0] }, /* primary first */
        { "kcj_scalar", demo_setup, demo_run, NULL, &cx[1] },
        { "jk_axpy",    demo_setup, demo_run, NULL, &cx[2] },
    };
    gr_opts o = gr_default_opts();
    o.warmups = 1;
    o.samples = 3;
    o.min_sample_us = 300.0;
    o.budget_us = 20e6; /* worst case (L=128 dense) stays far inside the 60 s plan budget */
    char key[GR_KEY_MAX];
    gr_keyf(key, sizeof key, "gen_race", "exec", L, gr_bucket(batch));
    int w = gr_pick(key, cands, 3, &o, NULL);
    p->variant = cx[w < 0 ? 0 : w].variant;
    free(r.rin);
    free(r.rout);
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    demo_execute_v(p, p->variant, in, out, p->batch);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->w);
    free(p->wr);
    free(p->wi);
    free(p->tmp);
    free(p);
}

#endif /* GEN_RACE_LIB_ONLY */
#endif /* GEN_RACE_C_INCLUDED */
