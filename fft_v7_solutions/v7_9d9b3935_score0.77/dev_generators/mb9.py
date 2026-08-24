import subprocess, gen
from mb2_harness import bench
for mb in (5, 7):
    bench(23, gen.emit_rader(23, 8, maxblock=mb), 'fft23_w8', f'rader mb={mb} fused')
# force ct(2,11) for 22 by hacking BEST cache
gen.BEST[22] = ('ct', 2, 11)
bench(23, gen.emit_rader(23, 8, maxblock=7), 'fft23_w8', 'rader q=ct(2,11)')
gen.BEST[22] = ('ct', 11, 2)
bench(23, gen.emit_rader(23, 8, maxblock=7), 'fft23_w8', 'rader q=ct(11,2)')
del gen.BEST[22]
