/*
 * thread_impl.hpp
 *
 *  Created on: Dec 10, 2015
 *      Author: dmarce1
 */

#ifndef THREAD_IMPL_HPP_
#define THREAD_IMPL_HPP_

#include <fenv.h>
#include <signal.h>

namespace hpx {


namespace detail {
void stack_trace_handler(int sig);
}


template<class F, class ...Args>
void thread::wrapper(void* arg) {
/*	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);*/
	signal(SIGABRT, detail::stack_trace_handler);
//	signal(SIGINT, detail::stack_trace_handler);
	signal(SIGSEGV, detail::stack_trace_handler);
	signal(SIGFPE, detail::stack_trace_handler);
	signal(SIGILL, detail::stack_trace_handler);
//	signal(SIGTERM, detail::stack_trace_handler);
//	signal(SIGHUP, detail::stack_trace_handler);

	auto tup_ptr = reinterpret_cast<tuple_type<F, Args...>*>(arg);
	typedef typename std::result_of<F(Args...)>::type return_type;
	detail::invoke_function<return_type, F, Args...>(*tup_ptr);
	delete tup_ptr;
}

template<class F, class ...Args>
thread::thread(F&& f, Args&&...args) {
	static std::atomic<unsigned> current_shep(0);
	auto tup_ptr = new tuple_type<F, Args...>(std::make_tuple(std::forward < F > (f), std::forward<Args>(args)...));
	auto void_ptr = reinterpret_cast<void*>(tup_ptr);
	handle = detail::thread_fork(&wrapper<F, Args...>, void_ptr);
}

namespace this_thread {

template<class Rep, class Period>
void sleep_for(const std::chrono::duration<Rep, Period>& pause_len) {
	typedef std::chrono::high_resolution_clock clock;
	auto start_time = clock::now();
	do {
		yield();
	} while ((clock::now() - start_time) > pause_len);
}

template<class Clock, class Duration>
void sleep_until(const std::chrono::time_point<Clock, Duration>& start_time) {
	typedef std::chrono::high_resolution_clock clock;
	while (clock::now() < start_time) {
		yield();
	}
}

}
}

namespace std {
template<class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& ost, hpx::thread::id id) {
	return ost << id.global << ':' << id.local;
}
}

#endif /* THREAD_IMPL_HPP_ */
