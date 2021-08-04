#pragma once

#include <cosmictiger/containers.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/tensor.hpp>

struct ewald_const {
	CUDA_EXPORT static int nfour();
	CUDA_EXPORT static int nreal();
	static void init();
	static void init_gpu();
	CUDA_EXPORT static const array<float,NDIM>& real_index(int i);
	CUDA_EXPORT static const array<float,NDIM>& four_index(int i);
	CUDA_EXPORT static const tensor_trless_sym<float,LORDER>& four_expansion(int i);
};
