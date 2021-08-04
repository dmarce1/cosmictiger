/*
 * thread_detail.hpp
 *
 *  Created on: Dec 10, 2015
 *      Author: dmarce1
 */

#ifndef THREAD_DETAIL_HPP_
#define THREAD_DETAIL_HPP_

#include <memory>
#include <tuple>

#define HPX_MPI_CALL(code) hpx::detail::check_mpi_return_code((code), __FILE__, __LINE__)

namespace hpx {

namespace detail {

class context;


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

class thread_exit {
private:
	std::function<void()> exit_func;
public:
	thread_exit();
	thread_exit(const thread_exit& other) = default;
	thread_exit(thread_exit&& other) = default;
	thread_exit(const std::function<void()>& func);
	thread_exit(std::function<void()>&& func);
	thread_exit& operator=(const thread_exit& other) = default;
	thread_exit& operator=(thread_exit&& other) = default;
	thread_exit& operator=(const std::function<void()>& func);
	thread_exit& operator=(std::function<void()>&& func);
	virtual ~thread_exit();
};


template<class R, class F, class ...Args, int ...S>
R invoke_function(std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>& tup, int_seq<S...>);

template<class R, class ...Args>
R invoke_function(std::tuple<typename std::decay<Args>::type...>& tup);

void check_mpi_return_code(int, const char*, int);
void thread_initialize();
void thread_finalize();
std::shared_ptr<context> thread_fork(void (*func_ptr)(void*), void* arg_ptr);

template<class R, class F, class ...Args, int ...S>
R invoke_function(std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>& tup, int_seq<S...>) {
	return std::get < 0 > (tup)(std::forward<Args>(std::get<S>(tup))...);
}

template<class R, class ...Args>
R invoke_function(std::tuple<typename std::decay<Args>::type...>& tup) {
	return invoke_function<R, Args...>(tup, typename iota_seq<sizeof...(Args)>::type());
}

}

namespace this_thread {
namespace detail {
void set_exit_function(std::function<void()>&&);
}
}

}

#endif /* THREAD_DETAIL_HPP_ */
