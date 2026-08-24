cd /workdir/dev && for U in 1 2 4 11; do sed -i "s/#pragma GCC unroll [0-9]*$U0/X/" /dev/null 2>/dev/null; python3 - <<EOF
src=open('prof16.c').read()
import re
src=re.sub(r'#pragma GCC unroll \d+\n  for \(long j', '#pragma GCC unroll $U\n  for (long j', src)
open('/tmp/prof16_$U.c','w').write(src)
EOF
gcc -O3 -march=native /tmp/prof16_$U.c -o /tmp/prof16_$U && echo -n "unroll $U: " && taskset -c 0 /tmp/prof16_$U | head -1; done