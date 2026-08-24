cd /workdir/dev && cat > core64_test.c <<'EOF'
#include <stdint.h>
#include <immintrin.h>
typedef double V __attribute__((vector_size(64), aligned(64)));
#include "core64only.h"
void kernel(const V* xr, const V* xi, V* yr, V* yi){ fft64_core(xr,xi,yr,yi,8,1); }
EOF
python3 -c "
import codelets
code,_ = codelets.gen_core(64)
open('core64only.h','w').write(code)
"
gcc -O3 -march=native -c core64_test.c -o core64_test.o && objdump -d core64_test.o | awk '{print \$3}' | grep -E "^v" | sort | uniq -c | sort -rn | head -20; echo "TOTAL:"; objdump -d core64_test.o | grep -cE "^\s+[0-9a-f]+:"