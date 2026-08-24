cd /tmp/bench && cat > zig.c <<'EOF'
#include <stdint.h>
#include <stdio.h>
#include <math.h>
typedef __uint128_t u128;
typedef struct { u128 state, inc; } pcg64_t;
static const u128 PCG_MULT = (((u128)2549297995355413924ULL)<<64) | 4865540595714422341ULL;
static inline uint64_t rotr64(uint64_t v, unsigned rot){ return (v >> rot) | (v << ((-rot) & 63)); }
static inline uint64_t pcg64_next(pcg64_t *g){
  g->state = g->state * PCG_MULT + g->inc;
  uint64_t hi = (uint64_t)(g->state >> 64), lo = (uint64_t)g->state;
  return rotr64(hi ^ lo, (unsigned)(g->state >> 122));
}
static inline double next_double(pcg64_t *g){ return (pcg64_next(g) >> 11) * (1.0/9007199254740992.0); }
#include "zigtab.h"  /* wi_double, ki_double, fi_double */
static const double ziggurat_nor_r = 3.6541528853610088;
static const double ziggurat_nor_inv_r = 0.27366123732975828;
static double std_normal(pcg64_t *g){
  uint64_t r, rabs; int sign, idx; double x, xx, yy;
  for (;;) {
    r = pcg64_next(g);
    idx = r & 0xff; r >>= 8;
    sign = r & 0x1;
    rabs = (r >> 1) & 0x000fffffffffffffULL;
    x = rabs * wi_double[idx];
    if (sign & 0x1) x = -x;
    if (rabs < ki_double[idx]) return x;
    if (idx == 0) {
      for (;;) {
        xx = -ziggurat_nor_inv_r * log1p(-next_double(g));
        yy = -log1p(-next_double(g));
        if (yy + yy > xx * xx)
          return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx) : ziggurat_nor_r + xx;
      }
    } else {
      if (((fi_double[idx-1] - fi_double[idx]) * next_double(g) + fi_double[idx]) < exp(-0.5*x*x))
        return x;
    }
  }
}
int main(int argc, char**argv){
  // read state from args: hi lo inchi inclo n
  u128 s = ((u128)strtoull(argv[1],0,16)<<64) | strtoull(argv[2],0,16);
  u128 inc = ((u128)strtoull(argv[3],0,16)<<64) | strtoull(argv[4],0,16);
  long n = atol(argv[5]);
  pcg64_t g = {s, inc};
  if (argv[6][0]=='r') { for(long i=0;i<n;i++) printf("%016lx\n", pcg64_next(&g)); }
  else { for(long i=0;i<n;i++){ double v=std_normal(&g); printf("%a\n", v);} }
  return 0;
}
EOF
python3 - <<'EOF'
import pickle
t=pickle.load(open('/tmp/bench/zig_tables.pkl','rb'))
with open('/tmp/bench/zigtab.h','w') as f:
    f.write("static const double wi_double[256]={"+",".join(float.hex(v) for v in t['wi'])+"};\n")
    f.write("static const uint64_t ki_double[256]={"+",".join(str(v)+"ULL" for v in t['ki'])+"};\n")
    f.write("static const double fi_double[256]={"+",".join(float.hex(v) for v in t['fi'])+"};\n")
EOF
gcc -O2 zig.c -o zig -lm -I. && python3 - <<'EOF'
import numpy as np, subprocess
for seed in [7+6, 1000+6, 42, 123456]:
    bg = np.random.default_rng(seed).bit_generator
    st = bg.state['state']
    s, inc = st['state'], st['inc']
    args=[f"{s>>64:x}", f"{s & (2**64-1):x}", f"{inc>>64:x}", f"{inc & (2**64-1):x}"]
    # raw test
    raw = subprocess.run(['./zig']+args+['8','r'], capture_output=True, text=True).stdout.split()
    nraw = np.random.PCG64(seed)  # hmm default_rng state vs PCG64(seed): same? compare with bg itself
    import numpy.random as nr
    bg2 = np.random.default_rng(seed).bit_generator
    pyraw = bg2.random_raw(8)
    print(seed, "raw match:", all(int(a,16)==b for a,b in zip(raw, pyraw)))
    # normal test
    out = subprocess.run(['./zig']+args+['100000','n'], capture_output=True, text=True).stdout.split()
    mine = np.array([float.fromhex(x) for x in out])
    ref = np.random.default_rng(seed).standard_normal(100000)
    print(seed, "normal bit-exact:", np.array_equal(mine, ref), (mine==ref).mean())
EOF