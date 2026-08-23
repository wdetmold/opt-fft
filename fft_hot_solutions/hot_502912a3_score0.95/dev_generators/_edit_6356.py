src = open('assemble.py').read()
old = """    shim45 = '''
void run_45(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    static int s39_inited = 0;
    if (!s39_inited) { s39_init_all(); s39_inited = 1; }
    s39_run_45(B, m, x0, c, out1, outm);
}
'''"""
new = """    shim45 = '''
void run_45(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (B < 8) { w00_run_45(x0, c, out1, outm, B, m); return; }
    static int s39_inited = 0;
    if (!s39_inited) { s39_init_all(); s39_inited = 1; }
    s39_run_45(B, m, x0, c, out1, outm);
}
'''"""
assert old in src
open('assemble.py','w').write(src.replace(old,new))
print('ok')
