NVCC     := nvcc
CXXFLAGS := -O3 -std=c++17 -Xcompiler -fopenmp
LDFLAGS  := -lm -lgomp

.PHONY: all clean

all: conv attention



attention: attention.cu
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

conv: conv.cu
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	# nvcc -O3 -Xcompiler -fopenmp conv-3d-with-cpu.cu -o conv_compare
	./conv

conv_smem_profile: conv_smem_profile.cu
	$(NVCC) -O3 -std=c++17 $< -o $@ $(LDFLAGS)

attention_prefill_profile: attention_prefill_profile.cu
	$(NVCC) -O3 -std=c++17 $< -o $@ $(LDFLAGS)


clean:
	rm -f conv attention
