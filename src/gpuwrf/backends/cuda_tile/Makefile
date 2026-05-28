ROOT := $(shell git rev-parse --show-toplevel)
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
BUILD_DIR ?= $(ROOT)/data/scratch/cuda_tile
TARGET := $(BUILD_DIR)/bench

NVCCFLAGS := -arch=sm_120 -O3 -std=c++17 -Xptxas=-v
LDFLAGS := -lz
SOURCES := host.cpp stencil.cu column.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SOURCES) $(LDFLAGS) -o $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -f $(TARGET)
