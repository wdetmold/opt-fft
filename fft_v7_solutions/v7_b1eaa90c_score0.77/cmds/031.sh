cd /workdir/dev && python3 -c "
import codelets
code,_ = codelets.gen_core(64)
open('core64only.h','w').write(code)
" && gcc -O3 -march=native -c core64_test.c -o core64_test.o && echo "fft64 instr:" && objdump -d core64_test.o | grep -cE "^\s+[0-9a-f]+:" && objdump -d core64_test.o | grep -oE "\bv[a-z0-9]+pd\b" | sort | uniq -c | sort -rn | head -8 && python3 gen.py && gcc -O3 -march=native prof1.c -o prof1 -lm && taskset -c 0 ./prof1