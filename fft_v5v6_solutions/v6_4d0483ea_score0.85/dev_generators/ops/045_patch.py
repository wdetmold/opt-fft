mi36 = [[(9*n1+4*n2)%36 for n2 in range(9)] for n1 in range(4)]
mo36 = [[(9*q1+28*q2)%36 for q1 in range(4)] for q2 in range(9)]
mi45 = [[(9*n1+5*n2)%45 for n2 in range(9)] for n1 in range(5)]
mo45 = [[(36*q1+10*q2)%45 for q1 in range(5)] for q2 in range(9)]
def fmt(name, t):
    rows = ",\n  ".join("{" + ", ".join(str(v) for v in row) + "}" for row in t)
    return f"static const long {name}[{len(t)}][{len(t[0])}] = {{\n  {rows}}};"
src = open('implementation.c').read()
src = src.replace("""// PFA index tables (input (9*n1+K*n2)%N, output (A*q1+B*q2)%N)
static long MI36[4][9], MO36[9][4];   // 36 = 4 (x) 9
static long MI45[5][9], MO45[9][5];   // 45 = 5 (x) 9""",
"// PFA index tables (input (9*n1+K*n2)%N, output (A*q1+B*q2)%N)\n"
+ fmt("MI36", mi36) + "\n" + fmt("MO36", mo36) + "\n" + fmt("MI45", mi45) + "\n" + fmt("MO45", mo45))
# remove the runtime init of these tables
src = src.replace("""
  for (int n1 = 0; n1 < 4; n1++)
    for (int n2 = 0; n2 < 9; n2++) MI36[n1][n2] = (9 * n1 + 4 * n2) % 36;
  for (int q2 = 0; q2 < 9; q2++)
    for (int q1 = 0; q1 < 4; q1++) MO36[q2][q1] = (9 * q1 + 28 * q2) % 36;
  for (int n1 = 0; n1 < 5; n1++)
    for (int n2 = 0; n2 < 9; n2++) MI45[n1][n2] = (9 * n1 + 5 * n2) % 45;
  for (int q2 = 0; q2 < 9; q2++)
    for (int q1 = 0; q1 < 5; q1++) MO45[q2][q1] = (36 * q1 + 10 * q2) % 45;""", "")
# pragma toggles
src = src.replace("  for (int n1 = 0; n1 < 4; n1++) {\n    const long *mi = MI36[n1];",
                  "  _Pragma(PSTR(GCC unroll U36S1)) for (int n1 = 0; n1 < 4; n1++) {\n    const long *mi = MI36[n1];")
src = src.replace("  for (int q2 = 0; q2 < 9; q2++) {\n    const long *mo = MO36[q2];",
                  "  _Pragma(PSTR(GCC unroll U36S2)) for (int q2 = 0; q2 < 9; q2++) {\n    const long *mo = MO36[q2];")
src = src.replace("  for (int n1 = 0; n1 < 5; n1++) {\n    const long *mi = MI45[n1];",
                  "  _Pragma(PSTR(GCC unroll U45S1)) for (int n1 = 0; n1 < 5; n1++) {\n    const long *mi = MI45[n1];")
src = src.replace("  for (int q2 = 0; q2 < 9; q2++) {\n    const long *mo = MO45[q2];",
                  "  _Pragma(PSTR(GCC unroll U45S2)) for (int q2 = 0; q2 < 9; q2++) {\n    const long *mo = MO45[q2];")
src = src.replace("#ifndef BLK17", """#ifndef U36S1
#define U36S1 1
#endif
#ifndef U36S2
#define U36S2 1
#endif
#ifndef U45S1
#define U45S1 1
#endif
#ifndef U45S2
#define U45S2 1
#endif
#ifndef BLK17""")
open('implementation.c','w').write(src)
print("ok")
