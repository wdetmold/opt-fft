# L13_direct — strategy record (ICE panel)

Geometry: **L = 13**, cube 13³ = 2197 complex doubles per volume (34.3 KiB),
forward, unnormalised, out-of-place, batched, single-threaded.
Implementation: `impl/L13_direct.c`. `fft3d_name()` → `L13_direct`.
Graded case (cases.txt): **B = 32, chain m = 1278, unitary** — 2.25 MiB
ping-pong working set, > the node's 1.25 MB L2, ≪ its L3.

This entry arrives at the ice panel as the geom-panel lineage at panel_r11
(conjugate-pair-folded dense 13×13 per axis, lanes = whole lines, all-pinned
register-resident matrix, zsolid-Y + xmm-tail mixed execs, X-first).  The
full history — including every dead end — is
`bench/geom/strategies/L13_direct.md` (rounds panel_r6 … panel_r11); the
multicore fork's record is `bench/mt/strategies/L13_direct.md`.  Do not
rediscover what those already killed.

---

## Round ice_r1 (2026-08-22)

(section under construction this round — measurements pending below)
