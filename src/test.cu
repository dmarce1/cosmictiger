#include <memory>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_mem.hpp>

#include <nvfunctional>
#include <cuda/semaphore>
template<int...>
struct int_seq {
};

template<int N, int ...S>
struct iota_seq: iota_seq<N - 1, N - 1, S...> {
};

template<int ...S>
struct iota_seq<1, S...> {
	typedef int_seq<S...> type;
};

using binary_semaphore = cuda::binary_semaphore<cuda::thread_scope_device>;

namespace cuda_hpx {

template<class T>
class shared_ptr {
	T* ptr;
	int* ref_count;
	__device__
	void ref_init() {
		ref_count = (int*) cuda_malloc(sizeof(int));
		*ref_count = 1;
	}
	__device__
	void dec_ref() {
		if (ref_count) {
			const int res = atomicAdd(ref_count, -1);
			if (res == 0) {
				if (ptr == nullptr) {
					cuda_free(ptr);
				}
				ptr->~T();
				cuda_free(ref_count);
			}
		}
	}
	__device__
	void inc_ref() {
		atomicAdd(ref_count, 1);
	}
public:
	__device__ shared_ptr() {
		ptr = nullptr;
		ref_count = nullptr;
	}
	__device__ shared_ptr(T* new_ptr) {
		ptr = new_ptr;
		ref_init();
	}
	__device__ shared_ptr(const shared_ptr& other) {
		ptr = other.ptr;
		ref_count = other.ref_count;
		inc_ref();
	}
	__device__ shared_ptr& operator=(const shared_ptr& other) {
		dec_ref();
		ptr = other.ptr;
		ref_count = other.ref_count;
		inc_ref();
	}
	__device__
	bool operator==(const shared_ptr& other) const {
		return ptr == other.ptr;
	}
	__device__
	bool operator!=(const shared_ptr& other) const {
		return ptr != other.ptr;
	}
	__device__ T& operator*() const {
		return *ptr;
	}
	__device__ T* operator->() const {
		return ptr;
	}
	__device__ ~shared_ptr() {
		dec_ref();
	}
};

template<class T, class ...Args>
__device__ shared_ptr<T> make_shared(Args&&...args) {
	auto* ptr = (T*) cuda_malloc(sizeof(T));
	new (ptr) T(std::forward<Args>(args)...);
	shared_ptr<T> rc(ptr);
	return rc;
}

template<class R, class F, class ...Args, int ...S>
__device__ R invoke_function(std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>& tup, int_seq<S...>) {
	return std::get < 0 > (tup)(std::forward<Args>(std::get<S>(tup))...);
}

template<class R, class ...Args>
__device__ R invoke_function(std::tuple<typename std::decay<Args>::type...>& tup) {
	return invoke_function<R, Args...>(tup, typename iota_seq<sizeof...(Args)>::type());
}

template<class T>
struct shared_state {
	T data;
	binary_semaphore signal;
	__device__ shared_state() :
			signal(false) {
	}
};

template<class T>
class promise;

template<class T>
class future {
	shared_ptr<shared_state<T>> state_ptr;
public:
	__device__ T get() {
		state_ptr->signal.acquire();
	}
	friend class promise<T> ;
};

template<class T>
class promise {
	shared_ptr<shared_state<T>> state_ptr;
public:
	__device__ promise() {
		state_ptr = make_shared<shared_state<T>>();
	}
	__device__ future<T> get_future() {
		future<T> fut;
		fut.state_ptr = state_ptr;
		return fut;
	}
	__device__ void set_value(T&& value) {
		state_ptr->data = std::move(value);
		state_ptr->signal.release();
	}
};

namespace detail {

//__device__ lockfree_queue<nvstd::function<void>*> workq;

__device__
void push_work(nvstd::function<void()>&& func) {

}

}

template<class F, class ... Args>
__device__ cuda_hpx::future<typename std::result_of<F(Args...)>::type> async(F &&f, Args &&... args) {
	using tuple_type = std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>;
	using return_type = typename std::result_of<F(Args...)>::type;
	auto tup_ptr = cuda_hpx::make_shared<tuple_type>(std::make_tuple(std::forward < F > (f), std::forward<Args>(args)...));
	shared_ptr<promise<return_type>> prms = cuda_hpx::make_shared<promise<return_type>>();
	auto func = [tup_ptr, prms]() {
		return_type rc = invoke_function<return_type, F, typename std::decay<Args>::type...>(*tup_ptr);
		if( threadIdx.x == 0 ) {
			prms->set_value(std::move(rc));
		}
	};
	detail::push_work(std::move(func));
	return prms->get_future();
}

}

__device__ int test_func(int a) {

}

__global__ void test_kernel() {
	cuda_hpx::async(test_func, 3);
}
