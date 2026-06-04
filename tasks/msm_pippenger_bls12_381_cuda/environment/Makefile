NVCC ?= nvcc
TARGET := msm_bls12381

NVCCFLAGS := -O3 -std=c++17 -arch=sm_90 -Xcompiler -Wall,-Wextra

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.cu solve.cu solve.h
	$(NVCC) $(NVCCFLAGS) main.cu solve.cu -o $(TARGET)

clean:
	rm -f $(TARGET)
