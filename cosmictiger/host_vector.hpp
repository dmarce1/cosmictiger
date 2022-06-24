/*
 * host_vector.hpp
 *
 *  Created on: Jun 18, 2022
 *      Author: dmarce1
 */

#ifndef HOST_VECTOR_HPP_
#define HOST_VECTOR_HPP_

#include <cosmictiger/device_vector.hpp>

template<class T>
using host_vector = device_vector<T>;
/*
template<class T>
class host_vector {
	T* ptr;
	int sz;
	int cap;
	inline void initialize() {
		sz = 0;
		cap = 0;
		ptr = nullptr;
	}
	inline void construct(int b, int e) {
		for (int i = b; i < e; i++) {
			new (ptr + i) T;
		}
	}
	inline void destruct(int b, int e) {
		for (int i = b; i < e; i++) {
			(ptr + i)->~T();
		}
	}
public:
	void swap(host_vector&& other) {
		auto* a = other.ptr;
		auto b = other.sz;
		auto c = other.cap;
		other.ptr = ptr;
		other.sz = sz;
		other.cap = cap;
		ptr = a;
		sz = b;
		cap = c;
	}
	inline host_vector() {
		initialize();
	}
	inline host_vector(host_vector<T> && other) {
		initialize();
		swap(std::move(other));
	}
	inline host_vector(const host_vector<T>& other) {
		initialize();
		resize(other.size());
		memcpy(ptr, other.ptr, sizeof(T) * other.size());
	}
	inline host_vector& operator=(host_vector<T> && other) {
		swap(std::move(other));
		return *this;
	}
	inline host_vector& operator=(const host_vector<T>& other) {
		resize(other.size());
		memcpy(ptr, other.ptr, sizeof(T) * other.size());
		return *this;
	}
	inline host_vector(int sz0) {
		initialize();
		resize(sz0);
	}
	inline ~host_vector() {
		if (ptr) {
			CUDA_CHECK(cudaFree(ptr));
		}
	}
	inline T* data() {
		return ptr;
	}
	inline void resize(int new_sz) {
		const int old_sz = sz;
		destruct(new_sz, std::max(old_sz, new_sz));
		if (new_sz <= cap) {
			sz = new_sz;
		} else {
			int new_cap = std::max(1024 / sizeof(T), (size_t) 1);
			while (new_cap < new_sz) {
				new_cap *= 2;
			}
			T* new_ptr;
			CUDA_CHECK(cudaMallocManaged(&new_ptr, new_cap * sizeof(T)));
			if (new_ptr == nullptr) {
				PRINT("OOM in host_vector while requesting %i elements and %lli bytes! \n", new_cap, (long long) new_cap * sizeof(T));
				abort();
			}
			if (ptr) {
				memcpy((void*) new_ptr, (void*) ptr, sz * sizeof(T));
			}
			if (ptr) {
				CUDA_CHECK(cudaFree(ptr));
			}
			ptr = new_ptr;
			sz = new_sz;
			cap = new_cap;
		}
		construct(std::min(old_sz, new_sz), new_sz);
	}
	inline T& back() {
		return ptr[sz - 1];
	}
	inline const T& back() const {
		return ptr[sz - 1];
	}
	inline void pop_back() {
		sz--;
	}
	inline void push_back(const T& item) {
		resize(size() + 1);
		back() = item;
	}
	CUDA_EXPORT
	inline int size() const {
		return sz;
	}
	CUDA_EXPORT
	inline T& operator[](int i) {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in host_vector\n");
			ALWAYS_ASSERT(false);
		}
#endif
		return ptr[i];
	}
	CUDA_EXPORT
	inline const T& operator[](int i) const {
#ifdef CHECK_BOUNDS
		if (i >= sz) {
			PRINT("Bound exceeded in host_vector\n");
			ALWAYS_ASSERT(false);
		}
#endif
		return ptr[i];
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		int this_size;
		arc & this_size;
		resize(this_size);
		for (int i = 0; i < this_size; i++) {
			arc & (*this)[i];
		}
	}
};*/

#endif /* HOST_VECTOR_HPP_ */
