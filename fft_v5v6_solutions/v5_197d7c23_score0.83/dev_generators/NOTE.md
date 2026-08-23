# Generator chain for implementation.c (attempt 197d7c23)

`implementation.c` was never written (or printed) in one piece during the
session: the agent kept `impl_head.c` (kernels) and `impl_tail.c`
(batching schemes + dispatch) in `/tmp/exp`, edited them via Python
patch scripts piped into `python3 -`, and rebuilt with
`cat impl_head.c impl_tail.c > implementation.c` after each step.
This directory transcribes that chain; `./run_all.sh` regenerates
`../implementation.c` deterministically.

Regeneration verified: all exact-match asserts inside the patch scripts pass
(they anchor on full-text snippets of the expected intermediate state, so a
transcription error in any earlier stage would trip them), and the result
preprocesses cleanly.

## Fidelity notes

1. **Log decoding.** The session log escapes backslashes (each source `\`
   appears as `\\`) and renders `\n` string escapes as `\` + newline. All
   content here is decoded back to the original bytes (macro continuations are
   single `\`, python sources carry `\\` inside non-raw strings / `\` inside
   r-strings).
2. **`stage_s02_tr8.c`** contains only the part of the original "part 2"
   heredoc that survives into the head (the `tr8` transpose + the
   `// ---- buffers` split marker). The remainder of part 2 (v1 buffers/
   schemes/dispatch) was discarded minutes later by the S03 split + S04 tail
   rewrite and never influences the final file.
3. **Omitted no-ops.** Two experimental GEN_PV rewrites (fused-map + software
   prefetch, log 01:35:38 and 01:36:38) are skipped: every byte they touch
   lies strictly inside the region `p12_genpv_plain.py` replaces wholesale
   (from the 6-parameter `#define GEN_PV(...)` line, introduced by p07, to
   `static void pv_in(`). Benchmark-only files (kbench*, ab64*, probe, etc.)
   and the mklref test harness are not reproduced.
4. **Crash boundary.** The container was discarded at 02:08:56 (429 storm) and
   restarted from the setup snapshot with only the first 53 actions retained
   and replayed. The pre-crash edits from ~01:38–02:08 therefore do NOT feed
   the graded file; the chain here jumps from p09 (01:29) to p12 (02:10),
   exactly as the surviving state did. This is also why the final code keeps
   tail-v1's `pv_run`/`bl_run` (which convert `c` into the blocked CBUF
   layout) rather than the pre-crash raw-`c` variants.
5. The final `/workdir` copy happened at 03:12:42 (after `p25`); the only later
   `/workdir` mutations were an `.so` rebuild in place and `__pycache__`
   removal.
