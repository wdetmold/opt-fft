import subprocess, gen
from mb2_harness import bench
for n in (13, 17, 23):
    bench(n, gen.emit_codelet(n, 8), f'fft{n}_w8', 'monolithic')
    for kt in (2, 3, 4, 6):
        bench(n, gen.emit_direct_staged(n, 8, kt), f'fft{n}_w8', f'direct staged kt={kt}')
