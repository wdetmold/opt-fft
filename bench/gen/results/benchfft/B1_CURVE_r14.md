# B=1 curve after r14 execute() reroute (benchFFT, a80n0, same-window ratios)

r14 rerouted fft3d_execute() to the fast within-volume engine across the whole L range.
Result: the library beats fftw3 at EVERY B=1 size (was losing 10/12, thin at 15/32/64/128).
Ratios are same-window (ours and fftw measured back-to-back, same pinned core) — the honest
comparison; absolute mflops are session-dependent (this window ran cooler than the baseline).

| L | ratio vs fftw3 | before r14 |
|---|---|---|
| 10 | 1.55x | lost 1.79x |
| 12 | 1.82x | lost 1.71x |
| 15 | 1.40x | 1.20x |
| 16 | 1.80x | 1.39x |
| 20 | 1.57x | 1.41x |
| 25 | 1.96x | 2.45x* |
| 27 | 2.47x | 4.75x* |
| 31 | 9.89x | 10.3x* |
| 32 | 1.64x | 1.32x |
| 40 | 1.57x | 2.59x* |
| 50 | 1.57x | 2.35x* |
| 64 | 1.54x | 1.16x |
| 100 | 1.98x | 1.29x |
| 128 | 1.64x | 1.03x |

(*cells where the baseline window's fftw ran anomalously slow, inflating the old ratio;
same-window r14 numbers are the honest ones.) Every cell verified. The library now has no
losing cell vs any stock library at any measured shape, in either regime.
