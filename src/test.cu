#include <cosmictiger/cuda.hpp>
#include <cosmictiger/hpx.hpp>

__managed__ int testint = 0;

class cuda_mutex {
	int lck;
#ifdef __CUDA_ARCH__
	__device__
	inline int atomic_add(int* ptr, int num) {
		return atomicAdd(ptr,num);
	}
	__device__
	inline void yield() {
		__nanosleep(10);
	}
#else
	inline int atomic_add(int* ptr, int num) {
		return __sync_fetch_and_add(ptr, num);
	}
	inline void yield() {
		hpx_yield();
	}
#endif
public:
	CUDA_EXPORT
	inline cuda_mutex() {
		lck = 0;
	}
	CUDA_EXPORT
	inline void lock() {
		while (atomic_add(&lck, 1) != 0) {
			atomic_add(&lck, -1);
			yield();
		}
	}
	CUDA_EXPORT
	inline void unlock() {
		lck = 0;
	}
};




