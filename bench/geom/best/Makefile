# Build the eight winning kernels into one library with one API.
#
# Each kernel keeps its source byte-identical to what was measured; its six API symbols are
# renamed at compile time so the units can coexist. -march=native matters: these kernels were
# tuned for AVX-512 on a Xeon Gold 5218 and several have runtime-guarded 512-bit paths.
# Per-host build products: this tree is shared and is built on several machines with
# -march=native (Haswell login node, Sapphire Rapids wallaby, Cascade Lake benchmark nodes).
# A shared obj/ would mean whichever built last leaves binaries that SIGILL on the others.
HOST    := $(shell hostname -s)
BUILD   := build/$(HOST)
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops
LDLIBS  ?= -lm

RENAME = -Dfft3d_name=$(1)_name -Dfft3d_description=$(1)_description \
         -Dfft3d_supports=$(1)_supports -Dfft3d_create=$(1)_create \
         -Dfft3d_execute=$(1)_execute -Dfft3d_destroy=$(1)_destroy

OBJS = $(BUILD)/fft3d_best.o $(BUILD)/k6.o $(BUILD)/k8.o $(BUILD)/k13.o $(BUILD)/k17.o $(BUILD)/k23.o $(BUILD)/k36.o \
       $(BUILD)/k45.o $(BUILD)/k64.o $(BUILD)/k8b.o $(BUILD)/k36b.o $(BUILD)/k64b.o

.PHONY: all verify clean
all: $(BUILD)/libfft3d_best.a $(BUILD)/verify_best $(BUILD)/pick_check

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/libfft3d_best.a: $(OBJS)
	ar rcs $@ $(OBJS)

$(BUILD)/fft3d_best.o: fft3d_best.c fft3d_best.h | $(BUILD)
	$(CC) $(CFLAGS) -I. -c -o $@ $<

# One rule per geometry: same recipe, different source and symbol prefix.
$(BUILD)/k6.o:  kernels/L6_pfa.c         | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k6)  -c -o $@ $<
$(BUILD)/k8.o:  kernels/L8_batchsimd.c   | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k8)  -c -o $@ $<
$(BUILD)/k13.o: kernels/L13_direct.c     | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k13) -c -o $@ $<
$(BUILD)/k17.o: kernels/L17_matrixsimd.c | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. -Ikernels $(call RENAME,k17) -c -o $@ $<
$(BUILD)/k23.o: kernels/L23_rader.c      | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k23) -c -o $@ $<
$(BUILD)/k36.o: kernels/L36_mixedradix.c | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k36) -c -o $@ $<
$(BUILD)/k45.o: kernels/L45_pfa.c        | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k45) -c -o $@ $<
$(BUILD)/k64.o: kernels/L64_blocked.c    | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k64) -c -o $@ $<

# Batched alternates: carried so a caller can measure the choice locally, selected only where
# the measurements support it (L=64 today).
$(BUILD)/k8b.o:  kernels/L8_fusedaxes.c    | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k8b)  -c -o $@ $<
$(BUILD)/k36b.o: kernels/L36_pencilfused.c | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k36b) -c -o $@ $<
$(BUILD)/k64b.o: kernels/L64_radix8.c      | $(BUILD) ; $(CC) $(CFLAGS) -I. -I.. $(call RENAME,k64b) -c -o $@ $<

$(BUILD)/verify_best: verify_best.c $(BUILD)/libfft3d_best.a | $(BUILD)
	$(CC) $(CFLAGS) -I. -o $@ $< $(BUILD)/libfft3d_best.a $(LDLIBS)

$(BUILD)/pick_check: pick_check.c $(BUILD)/libfft3d_best.a | $(BUILD)
	$(CC) $(CFLAGS) -I. -o $@ $< $(BUILD)/libfft3d_best.a $(LDLIBS)

verify: $(BUILD)/verify_best
	$(BUILD)/verify_best

clean:
	rm -rf $(BUILD)
