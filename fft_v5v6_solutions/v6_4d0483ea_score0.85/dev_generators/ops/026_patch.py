src = open('implementation.c').read()
src = src.replace("""    PF1(b + (8 + 0) * ld + cb * 8); PF1(b + (8 + 1) * ld + cb * 8);
    PF1(b + (8 + 2) * ld + cb * 8); PF1(b + (8 + 3) * ld + cb * 8);
    PF1(b + (8 + 4) * ld + cb * 8); PF1(b + (8 + 5) * ld + cb * 8);
    PF1(b + (8 + 6) * ld + cb * 8); PF1(b + (8 + 7) * ld + cb * 8);""",
"""    PF1(b + (PFD + 0) * ld + cb * 8); PF1(b + (PFD + 1) * ld + cb * 8);
    PF1(b + (PFD + 2) * ld + cb * 8); PF1(b + (PFD + 3) * ld + cb * 8);
    PF1(b + (PFD + 4) * ld + cb * 8); PF1(b + (PFD + 5) * ld + cb * 8);
    PF1(b + (PFD + 6) * ld + cb * 8); PF1(b + (PFD + 7) * ld + cb * 8);""")
src = src.replace("""#ifndef CPFHINT
#define CPFHINT _MM_HINT_T1
#endif""","""#ifndef CPFHINT
#define CPFHINT _MM_HINT_T1
#endif
#ifndef PFD
#define PFD 8
#endif""")
open('implementation.c','w').write(src)
