
// ================= dispatch layer =================
void run_6(const double* x, const double* c, double* o1, double* om, long B, long m){
    b00_run_6(x, c, o1, om, B, m);
}
void run_8(const double* x, const double* c, double* o1, double* om, long B, long m){
    b00_run_8(x, c, o1, om, B, m);
}
void run_13(const double* x, const double* c, double* o1, double* om, long B, long m){
    if (B < 6) { b00_run_13(x, c, o1, om, B, m); return; }
    long rem = B & 7L;
    if (rem == 0 || rem >= 6) { d43_run_13((void*)x, (void*)c, o1, om, B, m); return; }
    long G = B - rem;
    d43_run_13((void*)x, (void*)c, o1, om, G, m);
    long off = G * 2 * 2197;
    b00_run_13(x + off, c + off, o1 + off, om + off, rem, m);
}
void run_17(const double* x, const double* c, double* o1, double* om, long B, long m){
    b00_run_17(x, c, o1, om, B, m);
}
void run_23(const double* x, const double* c, double* o1, double* om, long B, long m){
    b00_run_23(x, c, o1, om, B, m);
}
void run_36(const double* x, const double* c, double* o1, double* om, long B, long m){
    if (B >= 7) s81_run_size(36, B, m, (void*)x, (void*)c, o1, om, 0);
    else b00_run_36(x, c, o1, om, B, m);
}
void run_45(const double* x, const double* c, double* o1, double* om, long B, long m){
    if (B >= 8) s81_run_size(45, B, m, (void*)x, (void*)c, o1, om, 0);
    else b00_run_45(x, c, o1, om, B, m);
}
void run_64(const double* x, const double* c, double* o1, double* om, long B, long m){
    f30_run64(B, m, (void*)x, (void*)c, o1, om);
}
void engines_init(void){ f30_init_tables(); }
