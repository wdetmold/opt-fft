# Builds one executable per backend: driver.o + <backend>.o, so every backend is
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

BIN := bin
IMPL_SRC  := $(wildcard impl/*.c)
IMPL_BIN  := $(patsubst impl/%.c,$(BIN)/%,$(IMPL_SRC))
SOTA_BIN  := $(BIN)/fftw3_estimate $(BIN)/fftw3_measure $(BIN)/fftw3_patient \
             $(BIN)/mkl_dfti $(BIN)/mkl2026_dfti $(BIN)/ducc0_c2c

.PHONY: all impls sota clean list
all: impls sota
impls: $(IMPL_BIN)
sota: $(SOTA_BIN)

$(BIN) $(BIN)/.keep:
	@mkdir -p $(BIN)

driver.o: driver.c fft3d_api.h
	$(CC) $(CFLAGS) -I. -c -o $@ driver.c

# --- library-free implementations -------------------------------------------------
$(BIN)/%: impl/%.c driver.o fft3d_api.h | $(BIN)
	$(CC) $(CFLAGS) -I. -o $@ $< driver.o $(LDLIBS)

# --- library baselines -----------------------------------------------------------
FFTW_FLAGS := -I$(FFT_PREFIX)/include
FFTW_LIBS  := -L$(FFT_PREFIX)/lib -Wl,-rpath,$(FFT_PREFIX)/lib -lfftw3

$(BIN)/fftw3_estimate: sota/fftw3.c driver.o | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -DPLAN_FLAG=FFTW_ESTIMATE -DPLAN_NAME='"fftw3_estimate"' \
	  -o $@ $< driver.o $(FFTW_LIBS) $(LDLIBS)

$(BIN)/fftw3_measure: sota/fftw3.c driver.o | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -DPLAN_FLAG=FFTW_MEASURE -DPLAN_NAME='"fftw3_measure"' \
	  -o $@ $< driver.o $(FFTW_LIBS) $(LDLIBS)

$(BIN)/fftw3_patient: sota/fftw3.c driver.o | $(BIN)
	$(CC) $(CFLAGS) -I. $(FFTW_FLAGS) -DPLAN_FLAG=FFTW_PATIENT -DPLAN_NAME='"fftw3_patient"' \
	  -o $@ $< driver.o $(FFTW_LIBS) $(LDLIBS)

$(BIN)/mkl_dfti: sota/mkl_dfti.c driver.o | $(BIN)
	$(CC) $(CFLAGS) -I. -I$(MKLROOT)/include -o $@ $< driver.o \
	  -L$(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64 \
	  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core $(LDLIBS)

# ducc0 declares c2c in fft.h but instantiates it in fft_inst2.cc (double) --
# that translation unit has to be compiled in, or the link fails.
$(BIN)/ducc0_c2c: sota/ducc0_c2c.cc driver.o | $(BIN)
	$(CXX) $(CXXFLAGS) -I. -I$(DUCC_SRC) -o $@ $< $(DUCC_SRC)/ducc0/fft/fft_inst2.cc \
	  $(DUCC_SRC)/ducc0/infra/mav.cc $(DUCC_SRC)/ducc0/infra/string_utils.cc \
	  $(DUCC_SRC)/ducc0/infra/threading.cc $(DUCC_SRC)/ducc0/infra/system.cc \
	  driver.o -lpthread $(LDLIBS)

# Current MKL from the pip wheel: no .so symlinks in the wheel, so the versioned
# objects are named directly.
$(BIN)/mkl2026_dfti: sota/mkl_dfti.c driver.o | $(BIN)
	$(CC) $(CFLAGS) -I. -I$(MKL2026)/include -DMKL_BACKEND_NAME='"mkl2026_dfti"' \
	  -DMKL_BACKEND_DESC='"oneMKL 2026.1 (pip wheel) DFTI, sequential, batched"' \
	  -o $@ $< driver.o -Wl,-rpath,$(MKL2026)/lib \
	  $(MKL2026)/lib/libmkl_intel_lp64.so.3 $(MKL2026)/lib/libmkl_sequential.so.3 \
	  $(MKL2026)/lib/libmkl_core.so.3 $(LDLIBS)

list:
	@echo "implementations: $(notdir $(IMPL_BIN))"
	@echo "sota:            $(notdir $(SOTA_BIN))"

clean:
	rm -rf $(BIN) driver.o
