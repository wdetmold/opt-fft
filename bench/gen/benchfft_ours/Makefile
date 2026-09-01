B := /home/lqcd/wdetmold/fft/ext/benchfft/benchfft-3.1
GEN := /home/lqcd/wdetmold/fft/bench/gen
CC := gcc
CFLAGS := -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops -fcommon
doit: doit.o gen_race.o
	$(CC) $(CFLAGS) -o $@ doit.o gen_race.o $(B)/libbench/libbench.a -lm
doit.o: doit.c
	$(CC) $(CFLAGS) -I$(B) -I$(B)/libbench -c doit.c
gen_race.o: $(GEN)/impl_12/gen_race.c $(GEN)/fft3d_api.h
	$(CC) $(CFLAGS) -I$(GEN) -c $(GEN)/impl_12/gen_race.c -o gen_race.o
clean: ; rm -f *.o doit
