# Builds one executable per backend: $(OBJ) + <backend>.o, so every backend is
# measured by identical driver code.  Compilation is outside the timed region.
#
#   make            build everything available
#   make impls      only the library-free implementations (impl/*.c)
#   make sota       only the library baselines (needs FFTW / MKL in the environment)

FFT_PREFIX ?= /home/lqcd/wdetmold/fft/ext/install
MKLROOT    ?= /opt/software/intel/oneapi/mkl/2022.0.2
# The module-provided MKL is 2022; the venv wheel carries a current one, so both are
# benchmarked -- beating only a four-year-old library would overstate our gains.
MKL2026    ?= /home/lqcd/wdetmold/fft/venv
CC         ?= gcc
# -march=native is deliberate: these binaries are built on the machine that runs them.
CFLAGS     ?= -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops
CXX        ?= g++
CXXFLAGS   ?= -O3 -march=native -mtune=native -std=c++17 -fno-math-errno
DUCC_SRC   ?= /home/lqcd/wdetmold/fft/ext/src/ducc/src
LDLIBS     ?= -lm

# Build products go under build/<hostname>/ because this directory lives on a SHARED
# filesystem and is built from two different machines: the Haswell login node (AVX2) and
# the Cascade Lake benchmark node (AVX-512), both with -march=native.  A single shared
# bin/ means whichever machine built last leaves binaries that SIGILL on the other -- and
# that has already silently corrupted an MKL baseline once.
HOST  := $(shell hostname -s)
BUILD := build/$(HOST)
BIN   := $(BUILD)/bin
OBJ   := $(BUILD)/driver.o
IMPL_SRC  := $(wildcard impl/*.c)
IMPL_BIN  := $(patsubst impl/%.c,$(BIN)/%,$(IMPL_SRC))
SOTA_BIN  := $(BIN)/fftw1d_estimate $(BIN)/fftw1d_measure $(BIN)/fftw1d_patient \
             $(BIN)/mkl1d_dfti

.PHONY: all impls sota clean list
all: impls sota
impls: $(IMPL_BIN)
sota: $(SOTA_BIN)

$(BIN):
	@mkdir -p $(BIN)

$(OBJ): driver.c fft1d_api.h | $(BIN)
	$(CC) $(CFLAGS) -I. -c -o $@ driver.c

# --- library-free implementations -------------------------------------------------
$(BIN)/%: impl/%.c $(OBJ) fft1d_api.h | $(BIN)
	$(CC) $(CFLAGS) -I. -o $@ $< $(OBJ) $(LDLIBS)

# --- library baselines -----------------------------------------------------------
FFTW_FLAGS := -I$(FFT_PREFIX)/include
FFTW_LIBS  := -L$(FFT_PREFIX)/lib -Wl,-rpath,$(FFT_PREFIX)/lib -lfftw3

$(BIN)/fftw3_estimate: sota/fftw3.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -DPLAN_FLAG=FFTW_ESTIMATE -DPLAN_NAME='"fftw3_estimate"' \
	  -o $@ $< $(OBJ) $(FFTW_LIBS) $(LDLIBS)

$(BIN)/fftw3_measure: sota/fftw3.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -DPLAN_FLAG=FFTW_MEASURE -DPLAN_NAME='"fftw3_measure"' \
	  -o $@ $< $(OBJ) $(FFTW_LIBS) $(LDLIBS)

$(BIN)/fftw3_patient: sota/fftw3.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -DPLAN_FLAG=FFTW_PATIENT -DPLAN_NAME='"fftw3_patient"' \
	  -o $@ $< $(OBJ) $(FFTW_LIBS) $(LDLIBS)

$(BIN)/fftw3_guru: sota/fftw3_guru.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -o $@ $< $(OBJ) $(FFTW_LIBS) $(LDLIBS)

$(BIN)/fftw3_custom: sota/fftw3_custom.c sota/genfft_shim.h $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. -o $@ $< $(OBJ) $(LDLIBS)

$(BIN)/fftw3_custom_soa: sota/fftw3_custom.c sota/genfft_shim.h $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -DCUSTOM_SOA -I. -o $@ $< $(OBJ) $(LDLIBS)

$(BIN)/mkl_dfti: sota/mkl_dfti.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. -I$(MKLROOT)/include -o $@ $< $(OBJ) \
	  -L$(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64 \
	  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core $(LDLIBS)

# ducc0 declares c2c in fft.h but instantiates it in fft_inst2.cc (double) --
# that translation unit has to be compiled in, or the link fails.
$(BIN)/ducc0_c2c: sota/ducc0_c2c.cc $(OBJ) | $(BIN)
	$(CXX) $(CXXFLAGS) -I. -I$(DUCC_SRC) -o $@ $< $(DUCC_SRC)/ducc0/fft/fft_inst2.cc \
	  $(DUCC_SRC)/ducc0/infra/mav.cc $(DUCC_SRC)/ducc0/infra/string_utils.cc \
	  $(DUCC_SRC)/ducc0/infra/threading.cc $(DUCC_SRC)/ducc0/infra/system.cc \
	  $(OBJ) -lpthread $(LDLIBS)

# Current MKL from the pip wheel: no .so symlinks in the wheel, so the versioned
# objects are named directly.
$(BIN)/mkl2026_dfti: sota/mkl_dfti.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. -I$(MKL2026)/include -DMKL_BACKEND_NAME='"mkl2026_dfti"' \
	  -DMKL_BACKEND_DESC='"oneMKL 2026.1 (pip wheel) DFTI, sequential, batched"' \
	  -o $@ $< $(OBJ) -Wl,-rpath,$(MKL2026)/lib \
	  $(MKL2026)/lib/libmkl_intel_lp64.so.3 $(MKL2026)/lib/libmkl_sequential.so.3 \
	  $(MKL2026)/lib/libmkl_core.so.3 $(LDLIBS)

list:
	@echo "host:            $(HOST)  (binaries in $(BIN))"
	@echo "implementations: $(notdir $(IMPL_BIN))"
	@echo "sota:            $(notdir $(SOTA_BIN))"

# Only this host's build; another machine's binaries in build/<other>/ are left alone.
clean:
	rm -rf $(BUILD)

clean-all:
	rm -rf build

$(BIN)/fftw1d_estimate: sota/fftw1d.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -DPLAN_FLAG=FFTW_ESTIMATE -DPLAN_NAME='"fftw1d_estimate"' -o $@ $< $(OBJ) $(FFTW_LIBS) $(LDLIBS)
$(BIN)/fftw1d_measure: sota/fftw1d.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -DPLAN_FLAG=FFTW_MEASURE -DPLAN_NAME='"fftw1d_measure"' -o $@ $< $(OBJ) $(FFTW_LIBS) $(LDLIBS)
$(BIN)/fftw1d_patient: sota/fftw1d.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -DPLAN_FLAG=FFTW_PATIENT -DPLAN_NAME='"fftw1d_patient"' -o $@ $< $(OBJ) $(FFTW_LIBS) $(LDLIBS)
$(BIN)/mkl1d_dfti: sota/mkl1d.c $(OBJ) | $(BIN)
	$(CC) $(CFLAGS) -I. -I$(MKLROOT)/include -o $@ $< $(OBJ) -L$(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core $(LDLIBS)
