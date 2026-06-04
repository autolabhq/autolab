NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=sm_90

TARGET = huffman_decode

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.cu solve.cu solve.h
	$(NVCC) $(NVCCFLAGS) -o $@ main.cu solve.cu

clean:
	rm -f $(TARGET)
