#pragma once

#include <tigerfmm/containers.hpp>
#include <tigerfmm/cuda.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/tensor.hpp>

struct ewald_const {
	CUDA_EXPORT static int nfour();
	CUDA_EXPORT static int nreal();
	static void init();
	static void init_gpu();
	CUDA_EXPORT static const array<float,NDIM>& real_index(int i);
	CUDA_EXPORT static const array<float,NDIM>& four_index(int i);
	CUDA_EXPORT static const tensor_trless_sym<float,LORDER>& four_expansion(int i);
};
