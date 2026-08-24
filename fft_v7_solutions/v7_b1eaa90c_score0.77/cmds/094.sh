cd /workdir/dev && cat >> kernels.py <<'PYEOF'

def gen_zline64(fname):
    """in-place 64-point complex FFT on one contiguous line (8 V re + 8 V im),
       vertical fft8 / lane-twiddle / 8x8 transpose / vertical fft8. natural order."""
    from codelets import tw, hexf
    # lane-twiddle tables: T[k1] lane s = W64^{s*k1}, k1=1..7
    rows_r = []; rows_i = []
    for k1 in range(1, 8):
        rr = []; ii = []
        for s in range(8):
            c, s_ = tw(s*k1, 64)
            rr.append(c); ii.append(s_)
        rows_r.append(rr); rows_i.append(ii)
    tabs = ("static const V zl64_twr[7] = {" +
            ",".join("{" + ",".join(hexf(v) for v in r) + "}" for r in rows_r) + "};\n" +
            "static const V zl64_twi[7] = {" +
            ",".join("{" + ",".join(hexf(v) for v in r) + "}" for r in rows_i) + "};\n")
    e = E()
    x = [(e.t(f"dr[{t}]"), e.t(f"di[{t}]")) for t in range(8)]
    A = fft_any(e, x, 8)
    st1 = []
    st1.append(f"    Br[0] = {A[0][0]}; Bi[0] = {A[0][1]};")
    for k1 in range(1, 8):
        rr = e.t(f"{A[k1][0]} * zl64_twr[{k1-1}] - {A[k1][1]} * zl64_twi[{k1-1}]")
        ii = e.t(f"{A[k1][1]} * zl64_twr[{k1-1}] + {A[k1][0]} * zl64_twi[{k1-1}]")
        st1.append(f"    Br[{k1}] = {rr}; Bi[{k1}] = {ii};")
    e2 = E()
    y = [(e2.t(f"Cr[{s}]"), e2.t(f"Ci[{s}]")) for s in range(8)]
    D = fft_any(e2, y, 8)
    st2 = "\n".join(f"    dr[{k2}] = {D[k2][0]}; di[{k2}] = {D[k2][1]};" for k2 in range(8))
    return tabs + f"""static __attribute__((noinline)) void {fname}(V* dr, V* di) {{
  V Br[8], Bi[8], Cr[8], Ci[8];
  {{
{e.code(indent="    ")}
{chr(10).join(st1)}
  }}
  tr8x8(Br, Cr);
  tr8x8(Bi, Ci);
  {{
{e2.code(indent="    ")}
{st2}
  }}
}}
"""
PYEOF
python3 - <<'EOF'
# quick numeric unit test of zline64
import subprocess, numpy as np, kernels
code = kernels.gen_zline64("f64_zline")
hdr = open('/workdir/implementation.c').read()
hdr = hdr[:hdr.index('// ---------------- L=6')]  # keep header + pw + tr8x8
test = hdr + code + r'''
#include <stdio.h>
int main(){
  static V dr[8], di[8];
  double in[128];
  FILE*f=fopen("zl_in.bin","rb"); fread(in,8,128,f); fclose(f);
  for(int t=0;t<8;t++) for(int s=0;s<8;s++){ dr[t][s]=in[8*t+s]; di[t][s]=in[64+8*t+s]; }
  f64_zline(dr, di);
  for(int t=0;t<8;t++) for(int s=0;s<8;s++) printf("%.17e %.17e\n", dr[t][s], di[t][s]);
  return 0;
}
'''
open('/tmp/zltest.c','w').write(test)
rng = np.random.default_rng(3)
line = rng.standard_normal(64) + 1j*rng.standard_normal(64)
buf = np.concatenate([line.real, line.imag])
buf.tofile('zl_in.bin')
subprocess.run(['gcc','-O3','-march=native','/tmp/zltest.c','-o','/tmp/zltest','-lm'],check=True)
out = subprocess.run(['/tmp/zltest'],capture_output=True,text=True).stdout.split()
vals = np.array([float(v) for v in out]).reshape(64,2)
got = vals[:,0]+1j*vals[:,1]
ref = np.fft.fft(line)
print("zline64 rel err:", np.linalg.norm(got-ref)/np.linalg.norm(ref))
EOF