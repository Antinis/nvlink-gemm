NVCC = nvcc
CXXFLAGS = -std=c++11 -O1
NVCCFLAGS = -arch=sm_70 -Xcompiler -fopenmp
TARGET = nvlink_gemm

all: $(TARGET)

$(TARGET): nvlink_gemm.cu
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) -o $(TARGET) nvlink_gemm.cu

clean:
	rm -f $(TARGET)

.PHONY: all clean

