/* =============================================================================
 * GEN_RACE -- the plan-time race + per-host wisdom cache library layer (gen_r3)
 * =============================================================================
 *
 * LIBRARY LAYER, scored by ADOPTION.  Class owners: adopt with
 *
 *     #define GEN_RACE_LIB_ONLY
 *     #include "gen_race.c"          // impl/ is the include dir; everything is
 *                                    // `static`, prefix gr_, no symbols leak
 *
 * (same pattern as gen_layout -- the layers compose; include any subset).
 *
 * WHAT THIS SOLVES.  The seed (seed_fft3d_best/fft3d_best.c) proved the choice
 * of kernel is a property of the MACHINE, not the algorithm: the L=64 pair
 * inverts by 4.3x between Haswell and Cascade Lake, and the Ice Lake grading
 * winner differed from the Cascade Lake winner at three of eight sizes.  This
 * campaign adds the plan-time budget: <= 60 s for a never-seen (L,B) INCLUDING
 * racing, <= 50 ms with persisted wisdom, and a cross-arch guard every second
 * round that the race must win on three microarchitectures.  The layer:
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
 *      INTERLEAVED since gen_r4: samples are taken SAMPLE-MAJOR (round-robin
 *      across candidates, min-of-rounds per candidate), not candidate-major.
 *      Three r4 records (gen_batchlane, gen_pfa_small, gen_pfa_large) found
 *      independently that this node's cores flip between sustained states
 *      10-15% apart WITHIN a lease, so timing candidate A to completion and
 *      then candidate B is no longer an A/B -- the drift becomes the verdict.
 *      Round-robin adjacency is gen_pfa_large's "alternate within the lease,
 *      compare adjacent pairs" and gen_batchlane's six-pair interleave, done
 *      inside one gr_race call.  Same signatures, same total work, same tie
 *      doctrine; GEN_RACE_SEQ=1 restores the r3 candidate-major order.
 *      NOTE for adopters: all candidates' setup() states now COEXIST for the
 *      duration of the race -- share big race buffers through ctx (the demo's
 *      demo_share shape) rather than allocating per candidate.
 *   3. gr_pick       -- gr_race behind the per-host wisdom cache
 *      (results/wisdom_<host>.json).  Cache hit: no candidate is even built,
 *      just a file read -- milliseconds against the 50 ms budget.  Keys carry
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
 *   6. gr_wisdom_get_str / gr_wisdom_put_str  (NEW in gen_r2) -- string-valued
 *      wisdom under your own key, same file, same locking.  Cache a chosen
 *      factorization name, a gate verdict, a knob pack -- the gen_planner
 *      "default pick = last raced winner" hook (their r1 next-list #5) and the
 *      plan-string cache, in two calls.
 *   7. gr_wisdom_drop_prefix  (NEW in gen_r5) -- round-end wisdom strip as one
 *      flock-safe call: drop every entry whose key starts with your prefix.
 *      The campaign-wide round-end protocol (gen_pfa_large r3), no hand-rolled
 *      file surgery.
 *
 * ENVIRONMENT PINS (all optional, for measuring something else):
 *     GEN_RACE_NO_RACE=1    always take candidate 0 (the primary)
 *     GEN_RACE_FORCE=name   take the candidate with this name where present
 *     GEN_RACE_REFRESH=1    ignore cached wisdom, re-race, overwrite
 *     GEN_RACE_NO_WISDOM=1  race every time, never read or write the file
 *     GEN_RACE_WISDOM=path  wisdom file override (default results/wisdom_<host>.json)
 *     GEN_RACE_VERBOSE=1    print every decision to stderr
 *     GEN_RACE_SEQ=1        r3 candidate-major timing order (A/B the racer itself)
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
 * THE DEMO ENTRY (when not GEN_RACE_LIB_ONLY): the round-6 trunk in miniature.
 * It adopts gen_planner's library (GEN_PLANNER_LIB include, as their record
 * invites): pln_enumerate turns L into candidate algorithm trees, gr_pick
 * races the top trees ON THE GRADED CHAIN STEP (gen_pfa_large's lesson: raw
 * execute can order candidates differently) and persists the winner per
 * (L, B-bucket, host).  gen_r3: a second gr_pick then races the winning
 * tree's scratch-row TILE width {32,16,64} (gen_planner's GEN_PLANNER_TILE
 * knob) -- on Ice Lake 32 wins per their calibration, but the tile is
 * exactly the kind of knob the cross-arch guard exists for, and now it is
 * raced + persisted per host instead of hardcoded.  gen_r4: the tree race
 * widens to 12 candidates to consume gen_planner's sub-tree diversity
 * enumeration (their r4 change, emitted explicitly "for the race"), and all
 * races run on the interleaved sample-major racer above.  gen_r5: at
 * batch >= 8, gen_planner's NEW split-group batch-lane engine (SoA-8, zero
 * shuffles, batch = vector dimension) joins the race as up to 4 extra arms
 * "<tree>@s<lev>"; a group arm's thunk is one group step (8 volumes), pv
 * thunks run 8 volume-major steps, so the race compares equal work in each
 * arm's real chain pattern.  A group winner gets its own cached two-step
 * gate and the chain runs pack -> m steps -> unpack per 8 volumes with the
 * B%8 remainder per-volume.  Wisdom tags carry an engine-generation salt
 * (now "chain7"/"tile7"/"chaingate7"/"fm7"/"p47"): the r2->r3 lesson was
 * that engine changes with unchanged candidate NAMES defeat gr_sig (the
 * stale r2 verdict would have cost ~2x at L=31); bumped every round the
 * planner's engine generation moved (r4: fused CT + interleaved racer;
 * r5: the split-group engine; r6: fused-map exits + pair-packed ladder +
 * @s4; r7: fused Good-Thomas codelets + the d7 hard leaf).
 * gen_r6: (1) the group probe loop covers planner's NEW level 4 (staged
 * in-place two-pass CT, "<tree>@s4") -- their own entry races lev <= 3
 * only, so the @s4-vs-@s2 verdict at 27/32/40 is this layer's to measure;
 * group-arm cap 4 -> 6 so three trees can contribute arms.  (2) a THIRD
 * race stage "fm6" on the SHIPPED engine: both planner engines carry a
 * runtime fusemap field gating where the graded map runs (pv: fused
 * transpose-out exit, default L > 12; group: fused final-axis stores,
 * default L^3 > 1728) -- host-tuned ICX boundaries, flipped in place per
 * thunk (a plain int consulted per step, no rebuild), primary = engine
 * default per the tie doctrine.  The two pv forms round differently
 * (fused exit = one exact vdivpd, separate pass = rcp14 ladder), so the
 * picked name carries "@fm<0|1>" and the gate verdict is cached per form.
 * gen_r7: (1) salt bump chain7/tile7/chaingate7/fm7 (planner's generation
 * moved again: fused Good-Thomas codelets, DFT7 hard leaf); (2) a FOURTH
 * race stage "p47" on @s4 winners: the stage-block width p4 (planner's
 * {n<=40:4, n<=64:2, else 1} table -- another ICX-tuned L1-residency
 * constant) flips in place per thunk like fm; wider-than-built widths
 * grow stage4 lazily in setup().  Blocking-only, numerics identical, so
 * the group gate verdict holds for any width.  (3) the pv tile race
 * widens {32,16,64} -> {32,16,48,64}.
 * Warm create() is a wisdom read + one engine build.
 * The entry owns fft3d_chain via pln_p3d_step (map fused per x-plane), gated
 * at create() against execute + the exact scalar map; the gate verdict is
 * itself cached through gr_wisdom_put_str so the warm path never re-runs it.
 * The driver's two-process repeatability cmp passes BECAUSE wisdom pins run 2
 * to run 1's winner (different trees round differently, so an unpersisted
 * noise-flip would show NOT REPEATABLE).
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
 * `tm` (may be NULL) receives per-candidate timings.
 *
 * Timing order is SAMPLE-MAJOR (gen_r4): every candidate is set up, warmed and
 * calibrated first (that calibration pass is its sample 0), then the remaining
 * samples run as round-robin ROUNDS over all live candidates, min-of-rounds
 * per candidate.  Candidate-major timing (r1-r3) let the node's 10-15%
 * within-lease core-state drift decide races; round-robin adjacency samples
 * every candidate in every state the window passes through.  GEN_RACE_SEQ=1
 * restores the old order.  All setup() states coexist until the race ends. */
static inline int gr_race(const gr_cand *c, int n, const gr_opts *o_in,
                          gr_timing *tm, gr_pick_info *pi)
{
    gr_opts o = o_in ? *o_in : gr_default_opts();
    gr_timing local[32];
    gr_timing *t = tm;
    if (!t && n <= 32) t = local;
    if (!t) return n > 0 ? 0 : -1; /* too many candidates, no scratch: primary */

    double deadline = o.budget_us > 0.0 ? gr_now_us() + o.budget_us : 0.0;
    if (getenv("GEN_RACE_SEQ") || n > 32) {
        /* r3 path: candidate-major (also the >32-candidate fallback, which
         * cannot hold 32+ coexisting states in the fixed arrays below) */
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
    } else {
        void *state[32];
        double worst[32];
        /* setup every candidate first (lazy allocations in setup() still only
         * happen when a race actually runs -- the wisdom hit path never gets
         * here); a NULL from setup() means "will not plan": skipped */
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
        /* warmup + rep calibration per candidate = its sample 0 (guaranteed
         * even past the deadline, like gr_time_run's first sample) */
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
        /* rounds 1..samples-1: one sample per candidate per round, adjacent
         * in time -- the corpus's alternation protocol, min-of-rounds */
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
            if (c[i].teardown) c[i].teardown(state[i]);
        }
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
/* Longest winner name storable/retrievable.  128 covers gen_planner's 96-char
 * canonical tree names; gen_r1's 64 made wisdom silently MISS on long names
 * (lookup rejects a name that does not fit) => permanent re-race every create. */
#define GR_NAME_MAX 128

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

/* ---- 3b. string-valued wisdom (NEW in gen_r2) ------------------------------- */
/* Cache an arbitrary short string under YOUR OWN key: a chosen factorization
 * name (gen_planner's "default pick = last raced winner" hook), a gate
 * verdict, a serialized knob pack.  Same file, same locking, same atomicity.
 * `val` must contain no quotes, spaces, braces or newlines (stored verbatim
 * as a JSON string, max GR_NAME_MAX-1 chars).  Honors GEN_RACE_NO_WISDOM and
 * (on reads) GEN_RACE_REFRESH, like gr_pick.  Use keys that cannot collide
 * with gr_pick's (those always end in "#<8 hex>"). */
static inline int gr_wisdom_get_str(const char *key, char *out, size_t cap)
{
    if (getenv("GEN_RACE_NO_WISDOM") || getenv("GEN_RACE_REFRESH")) return 0;
    int widx, tie;
    double us;
    return gr_wisdom_lookup(key, out, cap, &widx, &tie, &us);
}

static inline void gr_wisdom_put_str(const char *key, const char *val)
{
    if (getenv("GEN_RACE_NO_WISDOM")) return;
    gr_wisdom_store(key, val, -1, 0, 0.0, 0.0);
}

/* Drop every wisdom entry whose key starts with `prefix` (NEW in gen_r5, the
 * round-end strip protocol as one flock-safe call: gen_pfa_large's r3 protocol
 * is now campaign-wide and three entries hand-roll it).  Call it once at round
 * end with your entry's prefix ("gen_race/") so the monitor's scoring window
 * cold-races fresh verdicts.  Returns the number of entries removed. */
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
            char *p = strstr(old, "\"entries\":{");
            if (p) {
                p = strchr(p, '\n');
                char *line = p ? strtok(p, "\n") : NULL;
                while (line) {
                    char *q = line;
                    while (*q == ' ' || *q == '\t') ++q;
                    if (*q == '"') {
                        char *k0 = q + 1;
                        char *k1 = strchr(k0, '"');
                        if (k1 && (size_t)(k1 - k0) >= plen &&
                            memcmp(k0, prefix, plen) == 0) {
                            ++dropped;
                        } else if (k1) {
                            size_t ll = strlen(q);
                            while (ll && (q[ll - 1] == ',' || q[ll - 1] == ' ' ||
                                          q[ll - 1] == '\r'))
                                q[--ll] = 0;
                            if (ll) { fprintf(f, "%s\n%s", first ? "" : ",", q); first = 0; }
                        }
                    }
                    line = strtok(NULL, "\n");
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
        char wname[GR_NAME_MAX];
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
 * The round-6 trunk in miniature: gen_planner's library enumerates candidate
 * trees for L, gr_pick races the top trees ON THE GRADED CHAIN STEP and
 * persists the per-(L, B-bucket, host) winner; execute/chain run through
 * gen_planner's generic strided-row engine (pln_p3d_exec / pln_p3d_step).
 * ============================================================================= */
#ifndef GEN_RACE_LIB_ONLY

#include <math.h>

#define GEN_PLANNER_LIB 1
#include "gen_planner.c"   /* adopted library: static pln_* only, no entry */

#include "../fft3d_api.h"

struct fft3d_plan {
    int L, batch;
    int chain_ok;                /* create()-time gate passed: own the chain */
    pln_p3d *p3;                 /* per-volume engine: execute, remainder    */
#if PLN_SIMD
    pln_s8 *s8;                  /* split-group engine (gen_r5): raced in at
                                    batch >= 8 AND its own chain gate passed */
    double *G, *G2, *Cg;         /* group state / lev-2 alternate / packed c */
#endif
    double _Complex *scratch;    /* one volume, only for the !chain_ok path  */
    char picked[GR_NAME_MAX + 8];  /* tree name + "@t<w>" or "@s<lev>"        */
};

const char *fft3d_name(void) { return "gen_race"; }
const char *fft3d_description(void)
{
    return "LIBRARY LAYER (scored by adoption): plan-time candidate race (interleaved "
           "sample-major since r4: core-state-drift immune) + per-host wisdom cache "
           "incl. string wisdom + round-end drop_prefix (adopt: #define GEN_RACE_LIB_ONLY "
           "+ #include gen_race.c); demo = round-6 trunk: pln_enumerate trees (r7: incl. "
           "fused-GT + d7-leaf trees) + gen_planner split-group batch-lane arms "
           "(@s1/2/3/4, batch>=8, 6-arm cap) + tile width {32,16,48,64} + fm race + NEW "
           "r7 p4 race (@s4 stage-block width, planner's ICX L1 table, flipped in place "
           "per host), all raced on the graded chain step by gr_pick, persisted, fused "
           "chain; salts chain7/tile7/chaingate7/fm7/p47";
}
int fft3d_supports(int L) { return L >= 2 && L <= 128; }

/* --- race thunks: one graded chain step on one volume ------------------------
 * Buffers are shared across candidates and allocated LAZILY in setup(), so a
 * wisdom hit allocates nothing and builds no engine: warm create() is a file
 * read plus ONE winner build.  Racing the chain step, not raw execute, is
 * gen_pfa_large's gen_r1 lesson (the two can order candidates differently). */
struct demo_share {
    int L;
    double *st, *cf;             /* state + coefficient volume               */
    int alloc_failed;
#if PLN_SIMD
    double *G, *G2, *Cg;         /* split-group race buffers (gen_r5), 16*vol
                                    doubles each, shared by ALL group arms
                                    (the coexistence rule in the r4 header)  */
    int need_g2;                 /* any lev-2 arm in the candidate set       */
    int galloc_failed;
#endif
};
struct demo_cand {
    struct demo_share *sh;
    const pln_node *t;
    pln_p3d *eng;
    int tile;                    /* 0: engine default; else GEN_PLANNER_TILE */
    int nrun;                    /* pv chain steps per run(); 8 when group
                                    arms race (1 group step == 8 volumes, so
                                    both thunk kinds do 8 volume-steps)      */
#if PLN_SIMD
    pln_s8 *s8;                  /* probe-built group engine; owned by
                                    create(), NOT freed in teardown          */
    double *ga, *gb;             /* this arm's ping-pong view of sh->G/G2    */
#endif
};

/* Build a pln_p3d with a specific scratch-row tile width, through gen_planner's
 * documented GEN_PLANNER_TILE dev knob (create()-time only, single-threaded, so
 * the set/restore around the build is safe).  Copying pln_p3d_build's body with
 * a tile arg would fork their internals; the env hook survives their changes. */
static pln_p3d *demo_build_tiled(int L, const pln_node *t, int tile)
{
    if (tile <= 0) return pln_p3d_build(L, t);
    char old[24];
    const char *prev = getenv("GEN_PLANNER_TILE");
    if (prev) snprintf(old, sizeof old, "%s", prev);
    char v[16];
    snprintf(v, sizeof v, "%d", tile);
    setenv("GEN_PLANNER_TILE", v, 1);
    pln_p3d *p = pln_p3d_build(L, t);
    if (prev) setenv("GEN_PLANNER_TILE", old, 1);
    else unsetenv("GEN_PLANNER_TILE");
    return p;
}

static void demo_fill(double *p, size_t n, unsigned long s, double scale)
{
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005UL + 1442695040888963407UL;
        p[i] = scale * ((double)(s >> 12) / (double)(1UL << 52) - 0.5);
    }
}

static void *demo_setup(void *ctx)
{
    struct demo_cand *c = ctx;
    struct demo_share *sh = c->sh;
#if PLN_SIMD
    if (c->s8) {                 /* group arm: shared 8-volume buffers, lazy */
        if (sh->galloc_failed) return NULL;
        if (!sh->G) {
            const size_t gn = 16 * (size_t)sh->L * sh->L * sh->L;
            if (posix_memalign((void **)&sh->G, 64, gn * sizeof(double)) != 0 ||
                posix_memalign((void **)&sh->Cg, 64, gn * sizeof(double)) != 0 ||
                (sh->need_g2 &&
                 posix_memalign((void **)&sh->G2, 64, gn * sizeof(double)) != 0)) {
                free(sh->G); free(sh->Cg); sh->G = sh->Cg = NULL;
                sh->galloc_failed = 1;   /* group arms drop out; pv arms race */
                return NULL;
            }
            demo_fill(sh->G, gn, 5551, 1.0);
            demo_fill(sh->Cg, gn, 77713, 0.1);
            if (sh->G2) demo_fill(sh->G2, gn, 424243, 1.0);
        }
        c->ga = sh->G; c->gb = sh->G2;
        return ctx;
    }
#endif
    if (sh->alloc_failed) return NULL;
    if (!sh->st) {
        const size_t n2 = 2 * (size_t)sh->L * sh->L * sh->L;
        if (posix_memalign((void **)&sh->st, 64, n2 * sizeof(double)) != 0 ||
            posix_memalign((void **)&sh->cf, 64, n2 * sizeof(double)) != 0) {
            free(sh->st); sh->st = NULL; sh->alloc_failed = 1;
            return NULL;
        }
        demo_fill(sh->st, n2, 12346, 1.0);
        demo_fill(sh->cf, n2, 54322, 0.1);
    }
    c->eng = demo_build_tiled(sh->L, c->t, c->tile);
    return c->eng ? ctx : NULL;
}

static void demo_run(void *state)
{
    struct demo_cand *c = state;
#if PLN_SIMD
    if (c->s8) {                 /* one group step = 8 volume-steps */
        pln_s8_step(c->s8, &c->ga, &c->gb, c->sh->Cg);
        return;
    }
#endif
    for (int r = 0; r < c->nrun; ++r)           /* volume-major: steps
                                                   back-to-back on one volume */
        pln_p3d_step(c->eng, c->sh->st, c->sh->cf); /* in place; map keeps |z|<1 */
}

static void demo_teardown(void *state)
{
    struct demo_cand *c = state;
    pln_p3d_free(c->eng);        /* group arms have eng == NULL: no-op; their
                                    probe s8 engines are freed by create()   */
    c->eng = NULL;
}

#if PLN_SIMD
/* --- stage-3 fm race thunks (gen_r6): both planner engines carry a runtime
 * fusemap field (where the graded map runs); it is a plain int consulted
 * once per step, so the two candidates share ONE engine and the race flips
 * the field in place -- zero extra builds.  pv arms use the shared race
 * volume (lazy: a wisdom hit allocates nothing); group arms run on the
 * plan's own group buffers, deterministically re-filled in candidate 0's
 * setup because on a partially-warm create (tree race hit wisdom, fm did
 * not) they are fresh, uninitialized allocations. */
struct fm_cand {
    struct demo_share *sh;
    pln_p3d *p3;                 /* pv engine, NULL for a group winner   */
    pln_s8 *s8;                  /* group engine, NULL for a pv winner   */
    double *ga, *gb;             /* group ping-pong view (lev 2 swaps)   */
    double *cg;
    int fm;                      /* the form this candidate runs          */
    int fill;                    /* candidate 0 only: init group buffers  */
};

static void *fm_setup(void *ctx)
{
    struct fm_cand *c = ctx;
    struct demo_share *sh = c->sh;
    if (c->s8) {
        if (c->fill) {
            const size_t gn = 16 * (size_t)sh->L * sh->L * sh->L;
            demo_fill(c->ga, gn, 5551, 1.0);
            demo_fill(c->cg, gn, 77713, 0.1);
        }
        return ctx;
    }
    if (sh->alloc_failed) return NULL;
    if (!sh->st) {
        const size_t n2 = 2 * (size_t)sh->L * sh->L * sh->L;
        if (posix_memalign((void **)&sh->st, 64, n2 * sizeof(double)) != 0 ||
            posix_memalign((void **)&sh->cf, 64, n2 * sizeof(double)) != 0) {
            free(sh->st); sh->st = NULL; sh->alloc_failed = 1;
            return NULL;
        }
        demo_fill(sh->st, n2, 12346, 1.0);
        demo_fill(sh->cf, n2, 54322, 0.1);
    }
    return ctx;
}

static void fm_run(void *state)
{
    struct fm_cand *c = state;
    if (c->s8) {
        c->s8->fusemap = c->fm;
        pln_s8_step(c->s8, &c->ga, &c->gb, c->cg);
    } else {
        c->p3->fusemap = c->fm;
        pln_p3d_step(c->p3, c->sh->st, c->sh->cf);
    }
}

/* --- stage-4 p4 race thunks (gen_r7): the @s4 engine's stage-block width
 * (pencils per L1 staging pass) is a build-time TABLE on planner's side
 * ({n<=40:4, n<=64:2, else 1} -- an ICX-tuned constant like the fm
 * boundaries) but a RUNTIME field in the step: pln_s8_ct1_set derives the
 * stage stride 16*p4+8 per call, and p4 is pure blocking (the p0 loop
 * carries the remainder), so per-pencil arithmetic -- and therefore the
 * gate verdict -- is identical for every p4.  It races by in-place flip
 * like fm, with one wrinkle: stage4 was allocated for the BUILT width, so
 * a WIDER candidate must grow the allocation first (once, lazily in its
 * setup, so a wisdom hit allocates nothing).  Growth always sizes for
 * P=8; every 16P+8 stride stays an odd cache-line count, preserving
 * planner's anti-4K row pad at all widths. */
struct p4_cand {
    pln_s8 *s8;
    double *ga, *gb, *cg;        /* the plan's group buffers               */
    int L;
    int p4;                      /* the width this candidate runs          */
    int built;                   /* width stage4 was allocated for         */
    int *grown;                  /* shared flag: stage4 already at P=8     */
    int fill;                    /* candidate 0 only: init group buffers   */
};

static void *p4_setup(void *ctx)
{
    struct p4_cand *c = ctx;
    if (c->p4 > c->built && !*c->grown) {
        const int n = c->s8->r * c->s8->m;   /* @s4 root length            */
        double *ns;
        if (posix_memalign((void **)&ns, 64,
                           (size_t)n * (16 * 8 + 8) * sizeof(double)))
            return NULL;                     /* cannot grow: skip this arm */
        free(c->s8->stage4);
        c->s8->stage4 = ns;
        *c->grown = 1;
    }
    if (c->fill) {               /* same reason as the fm race: on a
                                    partially-warm create these are fresh,
                                    uninitialized allocations */
        const size_t gn = 16 * (size_t)c->L * c->L * c->L;
        demo_fill(c->ga, gn, 5551, 1.0);
        demo_fill(c->cg, gn, 77713, 0.1);
    }
    return ctx;
}

static void p4_run(void *state)
{
    struct p4_cand *c = state;
    c->s8->p4 = c->p4;
    pln_s8_step(c->s8, &c->ga, &c->gb, c->cg);
}
#endif

/* Two fused chain steps vs execute + the exact scalar map on a random volume
 * (the ice L17_rader r5 discipline: a fast wrong chain must be structurally
 * impossible to ship).  Deterministic given (tree, host), so the verdict is
 * cached via string wisdom and the warm path never re-runs it. */
static int demo_gate(pln_p3d *p3, int L)
{
    size_t vol = (size_t)L * L * L;
    double *a = NULL, *st = NULL, *rf = NULL, *cf = NULL;
    int ok = !posix_memalign((void **)&a, 64, 2 * vol * sizeof(double)) &&
             !posix_memalign((void **)&st, 64, 2 * vol * sizeof(double)) &&
             !posix_memalign((void **)&rf, 64, 2 * vol * sizeof(double)) &&
             !posix_memalign((void **)&cf, 64, 2 * vol * sizeof(double));
    int pass = 0;
    if (ok) {
        unsigned long s = 987654321;
        for (size_t i = 0; i < 2 * vol; ++i) {
            s = s * 6364136223846793005UL + 1442695040888963407UL;
            a[i] = (double)(s >> 12) / (double)(1UL << 52) - 0.5;
            s = s * 6364136223846793005UL + 1442695040888963407UL;
            cf[i] = 0.1 * ((double)(s >> 12) / (double)(1UL << 52) - 0.5);
        }
        memcpy(st, a, 2 * vol * sizeof(double));
        memcpy(rf, a, 2 * vol * sizeof(double));
        double num = 0, den = 0;
        for (int step = 0; step < 2; ++step) {
            pln_p3d_step(p3, st, cf);
            double *z = a; /* reuse a as the reference FFT output */
            pln_p3d_exec(p3, (const double _Complex *)rf, (double _Complex *)z);
            for (size_t i = 0; i < 2 * vol; i += 2) {
                double re = z[i] + cf[i], im = z[i + 1] + cf[i + 1];
                double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                rf[i] = re * sc; rf[i + 1] = im * sc;
            }
        }
        for (size_t i = 0; i < 2 * vol; ++i) {
            double d = st[i] - rf[i];
            num += d * d; den += rf[i] * rf[i];
        }
        pass = (den > 0 && sqrt(num / den) < 1e-12);
    }
    free(a); free(st); free(rf); free(cf);
    return pass;
}

#if PLN_SIMD
/* The split-group engine gets its OWN two-step gate (gen_planner r5's
 * discipline, adopted with the engine): pack -> 2 group steps -> unpack on 8
 * random volumes vs per-volume execute + the exact scalar map.  Exercises
 * pack, all three group axis passes, ping-pong parity and the split map end
 * to end.  Reuses the plan's group buffers (contents are rebuilt by chain's
 * pack anyway).  Verdict cached like the pv gate. */
static int demo_gate8(pln_s8 *s8, pln_p3d *p3, int L,
                      double *G, double *G2, double *Cg)
{
    size_t vol = (size_t)L * L * L;
    double *a = NULL, *rf = NULL, *cf = NULL, *ou = NULL, *z = NULL;
    int pass = 0;
    int mem = !posix_memalign((void **)&a, 64, 16 * vol * sizeof(double)) &&
              !posix_memalign((void **)&rf, 64, 16 * vol * sizeof(double)) &&
              !posix_memalign((void **)&cf, 64, 16 * vol * sizeof(double)) &&
              !posix_memalign((void **)&ou, 64, 16 * vol * sizeof(double)) &&
              !posix_memalign((void **)&z, 64, 2 * vol * sizeof(double));
    if (mem) {
        demo_fill(a, 16 * vol, 987654322, 1.0);
        demo_fill(cf, 16 * vol, 13131, 0.1);
        memcpy(rf, a, 16 * vol * sizeof(double));
        pln_s8_pack(a, vol, G);
        pln_s8_pack(cf, vol, Cg);
        double *ga = G, *gb = G2;
        pln_s8_step(s8, &ga, &gb, Cg);
        pln_s8_step(s8, &ga, &gb, Cg);
        pln_s8_unpack(ga, vol, ou);
        for (int v = 0; v < 8; ++v) {
            double *st = rf + 2 * vol * (size_t)v;
            const double *cv = cf + 2 * vol * (size_t)v;
            for (int step = 0; step < 2; ++step) {
                pln_p3d_exec(p3, (const double _Complex *)st, (double _Complex *)z);
                for (size_t i = 0; i < 2 * vol; i += 2) {
                    double re = z[i] + cv[i], im = z[i + 1] + cv[i + 1];
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    st[i] = re * sc; st[i + 1] = im * sc;
                }
            }
        }
        double num = 0, den = 0;
        for (size_t i = 0; i < 16 * vol; ++i) {
            double d = ou[i] - rf[i];
            num += d * d; den += rf[i] * rf[i];
        }
        pass = (den > 0 && sqrt(num / den) < 1e-12);
    }
    free(a); free(rf); free(cf); free(ou); free(z);
    return pass;
}
#endif

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch <= 0) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    pln_arena *A = calloc(1, sizeof *A);
    if (!p || !A) { free(p); free(A); return NULL; }
    p->L = L;
    p->batch = batch;

    pln_cand cand[12];
    int k = pln_enumerate(A, L, cand, 12);
    if (k <= 0) { free(A); free(p); return NULL; }

    /* Race the top trees on the graded chain step; model-best is candidate 0
     * (primary: the tie doctrine makes it win statistical ties, so quiet hosts
     * pick stably -- same philosophy as gen_planner's 2% hysteresis, but the
     * verdict persists per host instead of re-racing every process).
     * NK_MAX 8 -> 12 in gen_r4 (gen_planner's sub-tree diversity).
     * gen_r5: at batch >= 8, gen_planner's split-group batch-lane engine
     * enters the race -- up to 4 extra arms "<tree>@s<lev>" (lev 1/2/3),
     * probe-built here so the candidate NAME LIST (and with it gr_sig) is
     * deterministic per (L, batch-regime, host).  A group step runs 8 volumes
     * at once, so pv arms run 8 chain steps per thunk call when group arms
     * are present: every candidate's run() is 8 volume-steps, in its own real
     * chain pattern (pv = volume-major back-to-back, group = 8-wide SoA). */
    enum { NK_MAX = 12, NS8_MAX = 6 };  /* r6: 6 group arms (was 4) -- lev 4
                                           joined the probe set, and per tree
                                           at most two levels build ({1,2} or
                                           {4,2} or {3}), so 6 lets three
                                           trees contribute; 12+6 = 18 stays
                                           inside the racer's 32-state cap */
    int nk = k < NK_MAX ? k : NK_MAX;
    struct demo_share sh;
    memset(&sh, 0, sizeof sh);
    sh.L = L;
    struct demo_cand rc[NK_MAX + NS8_MAX];
    gr_cand c[NK_MAX + NS8_MAX];
    int n = 0;
    for (int i = 0; i < nk; ++i, ++n) {
        memset(&rc[n], 0, sizeof rc[n]);
        rc[n].sh = &sh; rc[n].t = cand[i].t; rc[n].nrun = 1;
        c[n].name = cand[i].name; c[n].setup = demo_setup; c[n].run = demo_run;
        c[n].teardown = demo_teardown; c[n].ctx = &rc[n];
    }
#if PLN_SIMD
    int ns8 = 0;
    char s8names[NS8_MAX][GR_NAME_MAX + 8];
    if (batch >= 8)
        for (int i = 0; i < nk && ns8 < NS8_MAX; ++i)
            /* lev <= 4 since gen_r6: planner's staged in-place two-pass CT
             * (@s4) races alongside @s1/2/3.  Level order per tree puts the
             * battle-tested @s2 before @s4, so the new form must beat it by
             * more than noise_rel to displace it (tie doctrine). */
            for (int lev = 1; lev <= 4 && ns8 < NS8_MAX; ++lev) {
                pln_s8 *s = pln_s8_build(L, cand[i].t, lev);
                if (!s) continue;
                memset(&rc[n], 0, sizeof rc[n]);
                rc[n].sh = &sh; rc[n].t = cand[i].t; rc[n].nrun = 1;
                rc[n].s8 = s;
                sh.need_g2 |= (lev == 2);
                snprintf(s8names[ns8], sizeof s8names[ns8], "%s@s%d",
                         cand[i].name, lev);
                c[n].name = s8names[ns8]; c[n].setup = demo_setup;
                c[n].run = demo_run; c[n].teardown = NULL; c[n].ctx = &rc[n];
                ++n; ++ns8;
            }
    if (ns8)
        for (int i = 0; i < nk; ++i) rc[i].nrun = 8;
#endif
    gr_opts o = gr_default_opts();
    o.warmups = 1;
    o.samples = 4;      /* 4 interleaved rounds: each candidate sampled in 4
                           temporally-separate core states (bimodality armor) */
    o.min_sample_us = 300.0;
    o.budget_us = 20e6; /* worst case (L=128, 12 trees) stays far inside 60 s */
    char key[GR_KEY_MAX];
    gr_keyf(key, sizeof key, "gen_race", "chain7", L, gr_bucket(batch));
    int w = gr_pick(key, c, n, &o, NULL);
    if (w < 0 || w >= n) w = 0;
    int wlev = 0;
#if PLN_SIMD
    if (w >= nk && rc[w].s8) wlev = rc[w].s8->lev;
#endif

    /* stage 2 (pv winner only): race the winning tree's scratch-row tile
     * width.  Engine default first (tie doctrine); candidates named by tile
     * so gr_sig salts the key.  Group winners skip it: the group engine has
     * no tile knob and the pv engine then only serves execute()/remainder. */
    static const int tiles[4] = { 0, 16, 48, 64 }; /* 0 = engine default (32);
                                    48 joined in gen_r7 (my r5 next-list #4:
                                    widen the set when the engine generation
                                    moves -- it did, twice since) */
    int tile = 0;
    if (wlev == 0) {
        struct demo_cand tc[4];
        gr_cand ct[4];
        static const char *tnames[4] = { "t32", "t16", "t48", "t64" };
        for (int i = 0; i < 4; ++i) {
            memset(&tc[i], 0, sizeof tc[i]);
            tc[i].sh = &sh; tc[i].t = rc[w].t; tc[i].tile = tiles[i];
            tc[i].nrun = 1;
            ct[i].name = tnames[i]; ct[i].setup = demo_setup;
            ct[i].run = demo_run; ct[i].teardown = demo_teardown;
            ct[i].ctx = &tc[i];
        }
        gr_opts ot = o;
        ot.budget_us = 6e6;
        char tkey[GR_KEY_MAX];
        gr_keyf(tkey, sizeof tkey, "gen_race", "tile7", L, gr_bucket(batch));
        int tw = gr_pick(tkey, ct, 4, &ot, NULL);
        if (tw >= 0 && tw < 4) tile = tiles[tw];
    }
    /* sh.st/sh.cf freed after the fm race below -- it reuses them */

    p->p3 = demo_build_tiled(L, rc[w].t, tile);
    int pk = w;
    if (w >= nk)                     /* group winner: its tree's pv index */
        for (int i = 0; i < nk; ++i)
            if (cand[i].t == rc[w].t) { pk = i; break; }
    if (pk >= nk) pk = 0;            /* cannot happen; belt and braces */
    for (int i = 0; !p->p3 && i < nk; ++i) { /* build failure: next-best tree */
        p->p3 = pln_p3d_build(L, cand[i].t);
        pk = i; tile = 0; wlev = 0;
    }
    if (!p->p3) {
#if PLN_SIMD
        for (int i = nk; i < n; ++i) if (rc[i].s8) pln_s8_free(rc[i].s8);
        free(sh.G); free(sh.G2); free(sh.Cg);
#endif
        free(sh.st); free(sh.cf);   /* frees moved past the fm race (r6) */
        free(A); free(p); return NULL;
    }

    char pvname[GR_NAME_MAX + 8];  /* pv identity: gates/fallback key on it */
    snprintf(pvname, sizeof pvname, "%s@t%d", cand[pk].name, tile ? tile : 32);

#if PLN_SIMD
    if (wlev > 0) {
        /* transfer the winning probe engine + the race's group buffers */
        p->s8 = rc[w].s8; rc[w].s8 = NULL;
        p->G = sh.G; p->G2 = sh.G2; p->Cg = sh.Cg;
        sh.G = sh.G2 = sh.Cg = NULL;
        size_t gn = 16 * (size_t)L * L * L;
        int ok = 1;
        if (!p->G && posix_memalign((void **)&p->G, 64, gn * sizeof(double)))
            { p->G = NULL; ok = 0; }
        if (ok && wlev == 2 && !p->G2 &&
            posix_memalign((void **)&p->G2, 64, gn * sizeof(double)))
            { p->G2 = NULL; ok = 0; }
        if (ok && !p->Cg && posix_memalign((void **)&p->Cg, 64, gn * sizeof(double)))
            { p->Cg = NULL; ok = 0; }
        if (!ok) {                  /* no group memory: drop to the pv path */
            pln_s8_free(p->s8); p->s8 = NULL;
            free(p->G); free(p->G2); free(p->Cg);
            p->G = p->G2 = p->Cg = NULL;
            wlev = 0;
        }
    }
    for (int i = nk; i < n; ++i)    /* free the losing probe engines */
        if (rc[i].s8) pln_s8_free(rc[i].s8);
    free(sh.G); free(sh.G2); free(sh.Cg);
#endif

    if (wlev > 0)
        snprintf(p->picked, sizeof p->picked, "%s", c[w].name); /* tree@s<lev> */
    else
        snprintf(p->picked, sizeof p->picked, "%s", pvname);
    free(A); /* engines own copies of everything they need */

#if PLN_SIMD
    /* stage 3 (gen_r6): race the SHIPPED engine's runtime fused-map form
     * ("fm6").  Both engine defaults are host-tuned ICX boundaries (pv:
     * fuse the map into the transpose-out exit at L > 12; group: into the
     * final-axis stores at L^3 > 1728) -- exactly the knob class the
     * cross-arch guard exists for, so it gets raced + persisted per host
     * instead of trusted (the r3 tile move again).  Primary = engine
     * default (tie doctrine).  The two pv forms round differently (fused
     * exit = one exact vdivpd, separate pass = rcp14 ladder), so the
     * picked name carries "@fm<v>" and gate verdicts cache per form.  The
     * scalar build has no fused-map path (planner pins fusemap = 0),
     * hence the PLN_SIMD guard on the whole stage. */
    {
        int def = wlev > 0 ? p->s8->fusemap : p->p3->fusemap;
        struct fm_cand fc[2];
        gr_cand cfm[2];
        char fmn[2][8];
        for (int i = 0; i < 2; ++i) {
            memset(&fc[i], 0, sizeof fc[i]);
            fc[i].sh = &sh;
            fc[i].fm = i ? !def : def;
            fc[i].fill = (i == 0);
            if (wlev > 0) {
                fc[i].s8 = p->s8; fc[i].ga = p->G; fc[i].gb = p->G2;
                fc[i].cg = p->Cg;
            } else {
                fc[i].p3 = p->p3;
            }
            snprintf(fmn[i], sizeof fmn[i], "fm%d", fc[i].fm);
            cfm[i].name = fmn[i]; cfm[i].setup = fm_setup; cfm[i].run = fm_run;
            cfm[i].teardown = NULL; cfm[i].ctx = &fc[i];
        }
        gr_opts of = o;
        of.budget_us = 4e6;
        char fkey[GR_KEY_MAX];
        gr_keyf(fkey, sizeof fkey, "gen_race", "fm7", L, gr_bucket(batch));
        int fw = gr_pick(fkey, cfm, 2, &of, NULL);
        int fmv = (fw == 1) ? !def : def;
        if (wlev > 0) {
            p->s8->fusemap = fmv;
            size_t n_ = strlen(p->picked);
            snprintf(p->picked + n_, sizeof p->picked - n_, "@fm%d", fmv);
        } else {
            p->p3->fusemap = fmv;
            size_t n_ = strlen(pvname);
            snprintf(pvname + n_, sizeof pvname - n_, "@fm%d", fmv);
            snprintf(p->picked, sizeof p->picked, "%s", pvname);
        }
    }

    /* stage 4 (gen_r7): race the @s4 winner's stage-block width p4 ("p47").
     * Planner's {n<=40:4, n<=64:2, else 1} table is another host-tuned ICX
     * constant (the r3 tile / r6 fm move, one level deeper): the stage's
     * L1 residency is a property of the HOST's L1, so CLX/SPR get their own
     * measured verdict instead of an inherited one.  Flip-in-place on the
     * shipped engine (the stage stride is derived from p4 per step call);
     * candidates wider than the built width grow stage4 lazily in setup().
     * Primary = table default (tie doctrine).  p4 is blocking-only --
     * identical numerics -- so the group gate verdict below is valid for
     * whatever width wins; the picked name still carries "@p<w>" when the
     * table was beaten, so wisdom receipts show it.  My r6 next-list #2,
     * unblocked by racing DOWN in place + growing UP lazily instead of the
     * rebuild-per-candidate I thought it needed. */
    if (p->s8 && p->s8->lev == 4 && p->s8->stage4) {
        const int def = p->s8->p4;
        int grown = 0;
        static const int pvals[4] = { 8, 4, 2, 1 };
        struct p4_cand pc[5];
        gr_cand cp4[5];
        char pn[5][8];
        int np = 0;
        for (int i = -1; i < 4; ++i) {
            const int v = i < 0 ? def : pvals[i];
            if (i >= 0 && v == def) continue;
            memset(&pc[np], 0, sizeof pc[np]);
            pc[np].s8 = p->s8; pc[np].ga = p->G; pc[np].gb = p->G2;
            pc[np].cg = p->Cg; pc[np].L = L;
            pc[np].p4 = v; pc[np].built = def; pc[np].grown = &grown;
            pc[np].fill = (np == 0);
            snprintf(pn[np], sizeof pn[np], "p%d", v);
            cp4[np].name = pn[np]; cp4[np].setup = p4_setup;
            cp4[np].run = p4_run; cp4[np].teardown = NULL; cp4[np].ctx = &pc[np];
            ++np;
        }
        gr_opts op = o;
        op.budget_us = 4e6;
        char pkey[GR_KEY_MAX];
        gr_keyf(pkey, sizeof pkey, "gen_race", "p47", L, gr_bucket(batch));
        int pw = gr_pick(pkey, cp4, np, &op, NULL);
        int wp4 = (pw >= 0 && pw < np) ? pc[pw].p4 : def;
        if (wp4 > def && !grown) {  /* wisdom hit on a wider width: grow now */
            const int n_r = p->s8->r * p->s8->m;
            double *ns;
            if (posix_memalign((void **)&ns, 64,
                               (size_t)n_r * (16 * 8 + 8) * sizeof(double)))
                wp4 = def;          /* cannot grow: keep the table width     */
            else { free(p->s8->stage4); p->s8->stage4 = ns; }
        }
        p->s8->p4 = wp4;            /* the race left p4 at its last thunk    */
        if (wp4 != def) {
            size_t n_ = strlen(p->picked);
            snprintf(p->picked + n_, sizeof p->picked - n_, "@p%d", wp4);
        }
    }
#endif
    free(sh.st);
    free(sh.cf);
    sh.st = sh.cf = NULL;

    /* chain-ownership gates, verdicts cached (string wisdom dogfood): warm
     * create is wisdom reads + builds, no gate re-run.
     * The pv gate always runs (execute() and the B%8 remainder use p3); the
     * group gate additionally guards the split-group chain (gen_r5). */
    {
        /* GR_KEY_MAX + 64: "gen_race/chaingate6/L<L>/" + a maximal 96-char
         * tree name + "@t<w>" + "@fm<v>" can pass 160; the lookup needle
         * buffer (GR_KEY_MAX + 24) still fits the longest key + quotes */
        char gk[GR_KEY_MAX + 64], gv[16];
        snprintf(gk, sizeof gk, "gen_race/chaingate7/L%d/%s", L, pvname);
        if (gr_wisdom_get_str(gk, gv, sizeof gv)) {
            p->chain_ok = (gv[0] == 'o');
        } else {
            p->chain_ok = demo_gate(p->p3, L);
            gr_wisdom_put_str(gk, p->chain_ok ? "ok" : "bad");
        }
#if PLN_SIMD
        if (p->s8) {
            int ok8;
            snprintf(gk, sizeof gk, "gen_race/chaingate7/L%d/%s", L, p->picked);
            if (gr_wisdom_get_str(gk, gv, sizeof gv)) {
                ok8 = (gv[0] == 'o');
            } else {
                ok8 = demo_gate8(p->s8, p->p3, L, p->G, p->G2, p->Cg);
                gr_wisdom_put_str(gk, ok8 ? "ok" : "bad");
            }
            if (!ok8) {             /* group gate failed: never ship it */
                pln_s8_free(p->s8); p->s8 = NULL;
                free(p->G); free(p->G2); free(p->Cg);
                p->G = p->G2 = p->Cg = NULL;
                snprintf(p->picked, sizeof p->picked, "%s", pvname);
            }
        }
#endif
        if (getenv("GEN_RACE_VERBOSE"))
            fprintf(stderr, "gen_race: L=%d picked %s (of %d), chain %s\n",
                    L, p->picked, n, p->chain_ok ? "fused" : "fallback");
        if (!p->chain_ok &&
            posix_memalign((void **)&p->scratch, 64,
                           (size_t)L * L * L * sizeof *p->scratch))
            p->scratch = NULL;
    }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int b = 0; b < p->batch; ++b)
        pln_p3d_exec(p->p3, in + (size_t)b * vol, out + (size_t)b * vol);
}

/* per-volume chain for volumes [b0, b1): volume-major, whole chain in place
 * per volume while it is cache resident, map fused per x-plane. */
static void demo_chain_pv(fft3d_plan *p, const double _Complex *x0,
                          const double _Complex *c, double _Complex *final_out,
                          int m, int b0, int b1)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int b = b0; b < b1; ++b) {
        double *st = (double *)(final_out + (size_t)b * vol);
        const double *cf = (const double *)(c + (size_t)b * vol);
        memcpy(st, x0 + (size_t)b * vol, vol * sizeof *x0);
        if (p->chain_ok || !p->scratch) {
            for (int s = 0; s < m; ++s)
                pln_p3d_step(p->p3, st, cf);
        } else {
            double *z = (double *)p->scratch;
            for (int s = 0; s < m; ++s) {
                pln_p3d_exec(p->p3, (const double _Complex *)st, p->scratch);
                for (size_t i = 0; i < 2 * vol; i += 2) {
                    double re = z[i] + cf[i], im = z[i + 1] + cf[i + 1];
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    st[i] = re * sc; st[i + 1] = im * sc;
                }
            }
        }
    }
}

/* Owned graded chain: state <- (FFT3(state) + c) / (1 + |FFT3(state) + c|),
 * m steps.  gen_r5: when the race picked a split-group arm, volumes go
 * through gen_planner's batch-lane engine 8 at a time (pack once, m steps on
 * the group, unpack once -- pack cost amortizes over m); the B % 8 remainder
 * takes the per-volume path.  Falls back to execute + the exact scalar map
 * if the create() gate failed. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
#if PLN_SIMD
    if (p->s8 && p->batch >= 8) {
        const size_t vol = (size_t)p->L * p->L * p->L;
        const int nb = p->batch - p->batch % 8;
        for (int b = 0; b < nb; b += 8) {
            pln_s8_pack((const double *)(x0 + (size_t)b * vol), vol, p->G);
            pln_s8_pack((const double *)(c + (size_t)b * vol), vol, p->Cg);
            double *ga = p->G, *gb = p->G2;
            for (int s = 0; s < m; ++s)
                pln_s8_step(p->s8, &ga, &gb, p->Cg);
            pln_s8_unpack(ga, vol, (double *)(final_out + (size_t)b * vol));
        }
        demo_chain_pv(p, x0, c, final_out, m, nb, p->batch);
        return;
    }
#endif
    demo_chain_pv(p, x0, c, final_out, m, 0, p->batch);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    pln_p3d_free(p->p3);
#if PLN_SIMD
    pln_s8_free(p->s8);
    free(p->G); free(p->G2); free(p->Cg);
#endif
    free(p->scratch);
    free(p);
}

#endif /* GEN_RACE_LIB_ONLY */
#endif /* GEN_RACE_C_INCLUDED */
