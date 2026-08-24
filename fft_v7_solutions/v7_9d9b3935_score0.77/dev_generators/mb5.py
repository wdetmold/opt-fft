import subprocess, gen
from mb2_harness import bench
for n in (13, 17, 23):
    bench(n, gen.emit_direct_staged(n, 8, {13:4,17:2,23:4}[n]), f'fft{n}_w8', 'direct staged')
    bench(n, gen.emit_rader(n, 8), f'fft{n}_w8', 'rader blocks')
    bench(n, gen.emit_rader(n, 8, maxblock=5), f'fft{n}_w8', 'rader blocks mb5')
