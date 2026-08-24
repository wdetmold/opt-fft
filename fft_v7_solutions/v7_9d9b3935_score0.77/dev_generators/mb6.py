import subprocess, gen
from mb2_harness import bench
for n in (13, 17, 23):
    for mb in (3, 5, 7, 11, 13):
        if mb >= n: continue
        try:
            bench(n, gen.emit_rader(n, 8, maxblock=mb), f'fft{n}_w8', f'rader mb={mb}')
        except Exception as e:
            print(n, mb, "fail", e)
