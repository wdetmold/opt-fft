import numpy as np, subprocess, codelets
hdr = """#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef double V __attribute__((vector_size(64)));
"""
mains = []
code = hdr
for N in (2,3,4,5,8,9,6,13,17,23,36,45,64):
    c, _ = codelets.gen_core(N)
    code += c
    mains.append(f"""
  {{
    V xr[{N}], xi[{N}], yr[{N}], yi[{N}];
    for (int i = 0; i < {N}; i++) for (int l = 0; l < 8; l++) {{ xr[i][l] = in[2*(i*8+l)]; xi[i][l] = in[2*(i*8+l)+1]; }}
    fft{N}_core(xr, xi, yr, yi, 1, 1);
    for (int i = 0; i < {N}; i++) for (int l = 0; l < 8; l++) {{ printf("%.17e %.17e\\n", yr[i][l], yi[i][l]); }}
    in += 2*8*{N};
  }}""")
code += f"""
int main(void) {{
  long total = 0;
  {" ".join(f"total += 2*8*{N};" for N in (2,3,4,5,8,9,6,13,17,23,36,45,64))}
  double *buf = malloc(total*sizeof(double));
  FILE *f = fopen("testin.bin","rb");
  fread(buf, sizeof(double), total, f); fclose(f);
  const double *in = buf;
  {"".join(mains)}
  return 0;
}}
"""
open("test_cores.c","w").write(code)
subprocess.run(["gcc","-O2","-march=native","test_cores.c","-o","test_cores"], check=True)
rng = np.random.default_rng(0)
Ns = (2,3,4,5,8,9,6,13,17,23,36,45,64)
blocks = [rng.standard_normal((N,8,2)) for N in Ns]
buf = np.concatenate([b.ravel() for b in blocks])
buf.tofile("testin.bin")
out = subprocess.run(["./test_cores"], capture_output=True, text=True).stdout.split()
vals = np.array([float(v) for v in out]).reshape(-1,2)
pos = 0
ok = True
for N, b in zip(Ns, blocks):
    x = b[...,0] + 1j*b[...,1]      # (N, 8)
    ref = np.fft.fft(x, axis=0)
    got = (vals[pos:pos+N*8,0] + 1j*vals[pos:pos+N*8,1]).reshape(N,8)
    pos += N*8
    err = np.linalg.norm(got-ref)/np.linalg.norm(ref)
    print(f"N={N:2d} rel err {err:.3e} {'OK' if err < 3e-15 else 'FAIL'}")
    ok &= err < 3e-15
print("ALL OK" if ok else "FAILURES")
