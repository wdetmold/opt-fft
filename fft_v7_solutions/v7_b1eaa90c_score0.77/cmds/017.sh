cd /tmp/bench && sed -i 's/#include <stdio.h>/#include <stdio.h>\n#include <stdlib.h>/' zig.c && gcc -O2 zig.c -o zig -lm -I. && python3 - <<'EOF'
import numpy as np, subprocess
ok=True
for seed in [13, 1006, 42, 123456, 0, 999999]:
    bg = np.random.default_rng(seed).bit_generator
    st = bg.state['state']
    s, inc = st['state'], st['inc']
    args=[f"{s>>64:x}", f"{s & (2**64-1):x}", f"{inc>>64:x}", f"{inc & (2**64-1):x}"]
    raw = subprocess.run(['./zig']+args+['8','r'], capture_output=True, text=True).stdout.split()
    pyraw = np.random.default_rng(seed).bit_generator.random_raw(8)
    m1 = all(int(a,16)==b for a,b in zip(raw, pyraw))
    out = subprocess.run(['./zig']+args+['200000','n'], capture_output=True, text=True).stdout.split()
    mine = np.array([float.fromhex(x) for x in out])
    ref = np.random.default_rng(seed).standard_normal(200000)
    m2 = np.array_equal(mine, ref)
    print(seed, "raw:", m1, "normal:", m2)
    ok = ok and m1 and m2
print("ALL OK" if ok else "MISMATCH")
EOF