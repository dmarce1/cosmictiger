/*
 * vector.hpp
 *
 *  Created on: Feb 11, 2021
 *      Author: dmarce1
 */

#ifndef COSMICTIGER_VECTOR_HPP_
#define COSMICTIGER_VECTOR_HPP_

#include <cassert>

#define vectorPOD 1

#ifdef __CUDA_ARCH__
#define BLOCK const unsigned& blocksize = blockDim.x
#define THREAD const unsigned& tid = threadIdx.x
#define FENCE() __threadfence_block()
#define SYNC() __syncthreads()
#else
#define BLOCK constexpr unsigned blocksize = 1
#define THREAD constexpr unsigned tid = 0
#define SYNC()
#define FENCE()
#endif

#include <functional>
#include <cosmictiger/memory.hpp>

template<class T>
class vector {
	T *ptr;
	T *new_ptr;
	unsigned cap;
	unsigned sz;
	bool dontfree;CUDA_EXPORT
	inline
	void destruct(unsigned b, unsigned e) {
		THREAD;
		BLOCK;
		for (unsigned i = b + tid; i < e; i += blocksize) {
			(*this)[i].T::~T();
		}
	}
	CUDA_EXPORT
	inline
	void construct(unsigned b, unsigned e) {
		THREAD;
		BLOCK;
		for (unsigned i = b + tid; i < e; i += blocksize) {
			new (ptr + i) T();
		}
	}
public:
	CUDA_EXPORT
	bool operator==(const vector<T>& other) const {
		if (size() != other.size()) {
			return false;
		} else {
			for (int i = 0; i < size(); i++) {
				if (other[i] != (*this)[i]) {
					return false;
				}
			}
		}
		return true;
	}
	CUDA_EXPORT
	bool operator!=(const vector<T>& other) const {
		return !operator==(other);
	}
#ifndef __CUDA_ARCH__
	template<class A>
	void serialize(A&& arc, unsigned) {
		auto this_sz = sz;
		arc & this_sz;
		resize(this_sz);
		for (unsigned i = 0; i < sz; i++) {
			arc & (*this)[i];
		}
	}
	T* begin() {
		return ptr;
	}
	T* end() {
		return ptr + sz;
	}
	std::function<void()> to_device(cudaStream_t stream) {
		//   assert(cap);
		dontfree = true;
		if (ptr) {
			//      CUDA_CHECK(cudaMemPrefetchAsync(ptr, sizeof(T) * sz, 0, stream));
			auto *dptr = ptr;
			auto sz_ = sz;
			auto func = [dptr, sz_]() {
				unified_allocator alloc;
				if (dptr) {
					for (unsigned i = 0; i < sz_; i++) {
						dptr[i].T::~T();
					}
					alloc.deallocate(dptr);
				}
			};
			//   PRINT( "3\n");
			//     ptr = new_ptr;
			return func;
		} else {
			return []() {
			};
		}
	}
#endif
	CUDA_EXPORT
	inline unsigned capacity() const {
		return cap;
	}
	CUDA_EXPORT
	inline vector() {
		THREAD;
		if (tid == 0) {
			dontfree = false;
			ptr = nullptr;
			cap = 0;
			sz = 0;
		}
//      reserve(1);
	}
	CUDA_EXPORT
	inline vector(unsigned _sz) {
		THREAD;
		if (tid == 0) {
			dontfree = false;
			ptr = nullptr;
			cap = 0;
			sz = 0;
		}
		resize(_sz);
//      reserve(1);
	}
	CUDA_EXPORT
	inline vector(unsigned _sz, T ele) {
		THREAD;
		BLOCK;
		if (tid == 0) {
			dontfree = false;
			ptr = nullptr;
			cap = 0;
			sz = 0;
		}
		resize(_sz);
		SYNC();
		for (unsigned i = tid; i < _sz; i += blocksize) {
			(*this)[i] = ele;
		}
//      reserve(1);
	}
	CUDA_EXPORT
	inline vector(const vector &&other) {
		THREAD;
		if (tid == 0) {
			dontfree = false;
			ptr = nullptr;
			cap = 0;
			sz = 0;
		}
		swap(other);
	}
	CUDA_EXPORT
	inline vector(const vector &other) {
		THREAD;
		BLOCK;
		if (tid == 0) {
			dontfree = false;
			sz = 0;
			ptr = nullptr;
			cap = 0;
		};
		reserve(other.cap);
		if (tid == 0) {
			sz = other.sz;
			cap = other.sz;
		}
		construct(0, other.sz);

		for (unsigned i = tid; i < other.sz; i += blocksize) {
			(*this)[i] = other[i];
		}
	}
	CUDA_EXPORT
	inline vector& operator=(const vector &other) {
		THREAD;
		BLOCK;
		reserve(other.cap);
		resize(other.size());
		for (unsigned i = tid; i < other.size(); i += blocksize) {
			(*this)[i] = other[i];
		}
		return *this;
	}
	CUDA_EXPORT
	inline vector& operator=(vector &&other) {
		THREAD;
		if (tid == 0) {
			swap(other);
		}
		return *this;
	}
	CUDA_EXPORT
	inline vector(vector &&other) {
		THREAD;
		if (tid == 0) {
			dontfree = false;
			sz = 0;
			ptr = nullptr;
			cap = 0;
			swap(other);
		}
	}
	CUDA_EXPORT
	inline
	void reserve(unsigned new_cap) {
		THREAD;
		BLOCK;
		if (new_cap > cap) {
			size_t i = 1;
			while ((i) / sizeof(T) < new_cap) {
				i *= 2;
			}
			new_cap = (i) / sizeof(T);
			//		       PRINT( "INcreasing capacity from %i to %i\n", cap, new_cap);
			if (tid == 0) {
#ifndef __CUDA_ARCH__
				unified_allocator alloc;
				new_ptr = (T*) alloc.allocate(new_cap * sizeof(T));
#else
				CUDA_MALLOC(new_ptr, new_cap);
#endif
			}SYNC();
			for (unsigned i = tid; i < sz; i += blocksize) {
				new (new_ptr + i) T();
				new_ptr[i] = std::move((*this)[i]);
			}
			destruct(0, sz);
			SYNC();
			if (tid == 0) {
				cap = new_cap;
				if (ptr && !dontfree) {
#ifndef __CUDA_ARCH__
					unified_allocator alloc;
					alloc.deallocate(ptr);
#else
					CUDA_FREE(ptr);
#endif
				}
				dontfree = false;
				ptr = new_ptr;
			}
		}
	}
	CUDA_EXPORT
	inline
	void resize(unsigned new_size, unsigned pod = 0) {
		THREAD;
		reserve(new_size);
		auto oldsz = sz;
#ifndef NDEBUG
		pod = 0;
#endif
		if (!pod) {
			destruct(new_size, oldsz);
		}
		if (tid == 0) {
			sz = new_size;
		}
		if (!pod) {
			construct(oldsz, new_size);
		}

	}
	CUDA_EXPORT
	inline T operator[](unsigned i) const {
		assert(i < sz);
		return ptr[i];
	}
	CUDA_EXPORT
	inline T& operator[](unsigned i) {
		assert(i < sz);
		return ptr[i];
	}
	CUDA_EXPORT
	inline unsigned size() const {
		return sz;
	}
	CUDA_EXPORT
	inline
	void push_back(const T &dat) {
		THREAD;
		resize(size() + 1);
		if (tid == 0) {
			ptr[size() - 1] = dat;
		}
	}
	CUDA_EXPORT
	inline
	void push_back(T &&dat) {
		THREAD;
		resize(size() + 1);
		if (tid == 0) {
			ptr[size() - 1] = std::move(dat);
		}
	}
	CUDA_EXPORT
	inline T* data() {
		return ptr;
	}
	CUDA_EXPORT
	inline const T* data() const {
		return ptr;
	}
	CUDA_EXPORT
	inline ~vector() {
		//    PRINT( "destroying\n");
		THREAD;
		destruct(0, sz);
		if (tid == 0 && ptr && !dontfree) {
#ifndef __CUDA_ARCH__
			unified_allocator alloc;
			alloc.deallocate(ptr);
#else
			CUDA_FREE(ptr);
#endif
		}
	}
	CUDA_EXPORT
	inline void pop_back() {
		assert(size());
		resize(size() - 1);
	}
	CUDA_EXPORT
	inline T back() const {
		return ptr[size() - 1];
	}
	CUDA_EXPORT
	inline T& back() {
		return ptr[size() - 1];
	}
	CUDA_EXPORT
	inline
	void swap(vector &other) {
		THREAD;
		if (tid == 0) {
			auto tmp1 = sz;
			auto tmp2 = cap;
			auto tmp3 = ptr;
			auto tmp4 = dontfree;
			sz = other.sz;
			cap = other.cap;
			ptr = other.ptr;
			dontfree = other.dontfree;
			other.sz = tmp1;
			other.cap = tmp2;
			other.ptr = tmp3;
			other.dontfree = tmp4;
		}
	}
};

#endif /* COSMICTIGER_VECTOR_HPP_ */
