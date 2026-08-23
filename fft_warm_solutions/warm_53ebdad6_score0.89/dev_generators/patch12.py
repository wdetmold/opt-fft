import re
src = open('implementation.c').read()
# swap sweep order in S_N for SoA sizes 6,8,13,17,23: do the es=16 (K axis, sequential) sweep first
for N in (6,8,13,17,23):
    sS = src.index(f"static void S_{N}(double* X")
    eS = src.index("\nstatic", sS+10)
    body = src[sS:eS]
    a = f"            for(int k=0;k<{N};k++) dftp{N}_v(sl + k*16, sl + k*16 + 8, {N*16});\n"
    b = f"            for(int j=0;j<{N};j++){{ dftp{N}_v(sl + (long)j*{N*16}, sl + (long)j*{N*16} + 8, 16);  }}\n"
    a6 = f"            for(int k=0;k<{N};k++) dft{N}_v(sl + k*16, sl + k*16 + 8, {N*16});\n"
    b6 = f"            for(int j=0;j<{N};j++){{ dft{N}_v(sl + (long)j*{N*16}, sl + (long)j*{N*16} + 8, 16);  }}\n"
    if a in body and b in body:
        body = body.replace(a+b, b+a)
    elif a6 in body and b6 in body:
        body = body.replace(a6+b6, b6+a6)
    else:
        print("WARN: pattern not found for", N)
        continue
    src = src[:sS] + body + src[eS:]
open('implementation.c','w').write(src)
print("patch12 applied")