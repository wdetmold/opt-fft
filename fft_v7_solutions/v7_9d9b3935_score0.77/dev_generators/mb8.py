import subprocess, gen
from mb2_harness import bench
bench(23, gen.emit_rader(23, 8, maxblock=7), 'fft23_w8', 'rader mb7 (cur)')
gen.DIRECT_SPLIT.add(11)
bench(23, gen.emit_rader(23, 8, maxblock=11), 'fft23_w8', 'rader mb11 split11')
gen.DIRECT_SPLIT.clear()
