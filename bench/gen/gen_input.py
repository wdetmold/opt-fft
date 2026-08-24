"""Generate the shared problem data.  One file, so every backend and the checker
see bit-identical input.  The monitor passes a fresh seed each round."""
import argparse, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--L", type=int, required=True)
p.add_argument("--batch", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--out", required=True)
p.add_argument("--scale", type=float, default=1.0)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
shape = (a.batch, a.L, a.L, a.L)
data = (a.scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))).astype(np.complex128)
data.tofile(a.out)
print(f"wrote {a.out}: shape {shape}, seed {a.seed}, "
      f"{data.nbytes/1024**2:.2f} MiB, checksum {np.abs(data).sum():.12g}")
