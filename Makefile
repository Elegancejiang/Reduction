all:
	nvcc -std=c++11 -gencode arch=compute_86,code=sm_86 -O3  reduction.cu -o  reduction  --expt-relaxed-constexpr -w