NVCC     := nvcc
CXXFLAGS := -std=c++17 -Xcompiler -fopenmp
LDFLAGS  := -lm -lgomp

.PHONY: all clean

all: conv attention

conv: conv.cu
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

attention: attention.cu
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f conv attention
