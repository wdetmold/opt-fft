# One executable per backend: driver.cu + <backend>.cu, so every entry is measured by
# identical driver code. Build products go under build/<hostname>/ for the same reason as
# the CPU harness: this tree is shared and is built from more than one machine.
CUDA_HOME ?= /opt/software/cuda-12.2.1
NVCC      ?= $(CUDA_HOME)/bin/nvcc
# A100 only: sm_80. -lineinfo costs nothing and makes ncu/compute-sanitizer usable.
ARCH      ?= sm_80
NVCCFLAGS ?= -O3 -std=c++14 -arch=$(ARCH) -lineinfo
LDLIBS    ?= -lm

HOST  := $(shell hostname -s)
BUILD := build/$(HOST)
BIN   := $(BUILD)/bin

IMPL_SRC := $(wildcard impl/*.cu)
IMPL_BIN := $(patsubst impl/%.cu,$(BIN)/%,$(IMPL_SRC))
SOTA_BIN := $(BIN)/cufft_batchsplit $(BIN)/cufftxt

.PHONY: all impls sota clean list
all: impls sota
impls: $(IMPL_BIN)
sota: $(BIN)/cufft_batchsplit

$(BIN):
	@mkdir -p $(BIN)

# --- library-free entries ---------------------------------------------------------
$(BIN)/%: impl/%.cu driver.cu fft3d_mgpu_api.h | $(BIN)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< driver.cu $(LDLIBS)

# --- library baselines ------------------------------------------------------------
$(BIN)/cufft_batchsplit: sota/cufft_batchsplit.cu driver.cu fft3d_mgpu_api.h | $(BIN)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< driver.cu -lcufft $(LDLIBS)

$(BIN)/cufftxt: sota/cufftxt.cu driver.cu fft3d_mgpu_api.h | $(BIN)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< driver.cu -lcufft $(LDLIBS)

list:
	@echo "host:  $(HOST)  (binaries in $(BIN))"
	@echo "impls: $(notdir $(IMPL_BIN))"

clean:
	rm -rf $(BUILD)
