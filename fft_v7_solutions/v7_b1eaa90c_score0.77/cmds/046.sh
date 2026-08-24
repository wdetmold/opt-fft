cd /workdir/dev && python3 - <<'EOF'
src = open('prof6.c').read()
main = '''
int main(){
  for(int i=0;i<64*8;i++) for(int l=0;l<8;l++){ X_r[i][l]=0.1*((i+l)%7)-0.3; X_i[i][l]=0.1*((i*3+l)%5)-0.2; }
  static V S1r[64], S1i[64];
  long R=300000; uint64_t t0,t1;
  t0=rdtscp();
  for(long r=0;r<R;r++) passA(X_r+(r&7)*8, X_i+(r&7)*8, S1r, S1i);
  t1=rdtscp();
  printf("passA: %.1f cyc/call (%.3f c/elem)\\n",(double)(t1-t0)/R,(double)(t1-t0)/R/512);
  t0=rdtscp();
  for(long r=0;r<R;r++) passB(S1r, S1i, Y_r+(r&7)*8, Y_i+(r&7)*8);
  t1=rdtscp();
  printf("passB: %.1f cyc/call (%.3f c/elem)\\n",(double)(t1-t0)/R,(double)(t1-t0)/R/512);
  t0=rdtscp();
  for(long r=0;r<R;r++){ passA(X_r+(r&7)*8, X_i+(r&7)*8, S1r, S1i); passB(S1r, S1i, Y_r+(r&7)*8, Y_i+(r&7)*8); }
  t1=rdtscp();
  printf("A+B:   %.1f cyc/call (%.3f c/elem)\\n",(double)(t1-t0)/R,(double)(t1-t0)/R/512);
  double s=0; for(int i=0;i<64*8;i++) s+=Y_r[i][2]+Y_i[i][6];
  printf("chk %g\\n", s);
  return 0;
}
'''
src = src[:src.index('int main(){')] + main
open('prof6.c','w').write(src)
EOF
gcc -O3 -march=native prof6.c -o prof6 && taskset -c 0 ./prof6