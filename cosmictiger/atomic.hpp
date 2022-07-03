
#ifndef ATOMIC333_HPP_
#define ATOMIC333_HPP_

#include <cosmictiger/cuda.hpp>

template<class T>
CUDA_EXPORT T atomic_cas(T* this_, T expected, T value) {
#ifdef __CUDA_ARCH__
	return atomicCAS_system(this_, expected, value);
#else
	return __sync_val_compare_and_swap(this_, expected, value);
#endif
}

template<class T>
CUDA_EXPORT T atomic_add(T* this_, T value) {
#ifdef __CUDA_ARCH__
	return atomicAdd_system(this_, value);
#else
	T expected, sum, rc;
	do {
		expected = *this_;
		sum = expected + value;
		rc = atomic_cas(this_, expected, sum);
	} while (rc != expected);
	return expected;
#endif
}

CUDA_EXPORT inline void fence() {
#ifdef __CUDA_ARCH__
	__threadfence_system();
#else
	__sync_synchronize();
#endif
}

template<class T>
CUDA_EXPORT T atomic_max(T* this_, T value) {
#ifdef __CUDA_ARCH__
	return atomicMax_system(this_, value);
#else
	T expected, maxvalue, rc;
	do {
		expected = *this_;
		maxvalue = std::max(expected, value);
		rc = atomic_cas(this_, expected, maxvalue);
	} while (rc != expected);
	return maxvalue;
#endif
}

template<class T>
CUDA_EXPORT T atomic_min(T* this_, T value) {
#ifdef __CUDA_ARCH__
	return atomicMax_system(this_, value);
#else
	T expected, minvalue, rc;
	do {
		expected = *this_;
		minvalue = std::min(expected, value);
		rc = atomic_cas(this_, expected, minvalue);
	} while (rc != expected);
	return minvalue;
#endif
}

template<class T>
CUDA_EXPORT T atomic_exch(T* this_, T value) {
#ifdef __CUDA_ARCH__
	return atomicExch_system(this_, value);
#else
	T expected, rc;
	do {
		expected = *this_;
		rc = atomic_cas(this_, expected, value);
	} while (rc != expected);
	return expected;
#endif
}



#endif /* ATOMIC_HPP_ */
