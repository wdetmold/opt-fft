# Build the eight winning kernels into one library with one API.
#
# Each kernel keeps its source byte-identical to what was measured; its six API symbols are
# renamed at compile time so the units can coexist. -march=native matters: these kernels were
# tuned for AVX-512 on a Xeon Gold 5218 and several have runtime-guarded 512-bit paths.
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops
LDLIBS  ?= -lm

RENAME = -Dfft3d_name=$(1)_name -Dfft3d_description=$(1)_description \
         -Dfft3d_supports=$(1)_supports -Dfft3d_create=$(1)_create \
         -Dfft3d_execute=$(1)_execute -Dfft3d_destroy=$(1)_destroy

OBJS = obj/fft3d_best.o obj/k6.o obj/k8.o obj/k13.o obj/k17.o obj/k23.o obj/k36.o \
       obj/k45.o obj/k64.o

.PHONY: all verify clean
all: libfft3d_best.a verify_best

obj:
	@mkdir -p obj

libfft3d_best.a: $(OBJS)
	ar rcs $@ $(OBJS)

obj/fft3d_best.o: fft3d_best.c fft3d_best.h | obj
	$(CC) $(CFLAGS) -I. -c -o $@ $<

# One rule per geometry: same recipe, different source and symbol prefix.
obj/k6.o:  kernels/L6_pfa.c         | obj ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k6)  -c -o $@ $<
obj/k8.o:  kernels/L8_batchsimd.c   | obj ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k8)  -c -o $@ $<
obj/k13.o: kernels/L13_direct.c     | obj ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k13) -c -o $@ $<
obj/k17.o: kernels/L17_matrixsimd.c | obj ; $(CC) $(CFLAGS) -I. -I.. -Ikernels $(call RENAME,k17) -c -o $@ $<
obj/k23.o: kernels/L23_rader.c      | obj ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k23) -c -o $@ $<
obj/k36.o: kernels/L36_mixedradix.c | obj ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k36) -c -o $@ $<
obj/k45.o: kernels/L45_pfa.c        | obj ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k45) -c -o $@ $<
obj/k64.o: kernels/L64_blocked.c    | obj ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k64) -c -o $@ $<

verify_best: verify_best.c libfft3d_best.a
	$(CC) $(CFLAGS) -I. -o $@ $< libfft3d_best.a $(LDLIBS)

verify: verify_best
	./verify_best

clean:
	rm -rf obj libfft3d_best.a verify_best
