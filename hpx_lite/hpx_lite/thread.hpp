/*
 * thread.hpp
 *
 *  Created on: Nov 24, 2015
 *      Author: dmarce1
 */

#ifndef THREAD_HPP_
#define THREAD_HPP_

#include <mpi.h>

#include <atomic>
#include <chrono>
#include <functional>

#include "detail/thread_detail.hpp"

namespace hpx {
class thread {
private:

	template<class F, class...Args>
	using tuple_type = std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>;

	template<class F, class ...Args>
	static void wrapper(void* arg);

	std::shared_ptr<detail::context> handle;
public:

	bool attempt_continuation(std::function<void()>& func);

	class id;
	typedef void native_handle_type;

	thread();
	thread(const thread&) = delete;
	thread(thread&&) = default;
	~thread() = default;

	thread& operator=(const thread&) = delete;
	thread& operator=(thread&&) = default;

	bool joinable() const;
	id get_id() const;
	native_handle_type native_handle();
	void detach();
	void join();
	void swap(thread&);

	template<class F, class ...Args>
	thread(F&& f, Args&&...args);

	static unsigned hardware_concurrency();

};

namespace this_thread {
void yield(const std::shared_ptr<bool> yield_bit=nullptr);
thread::id get_id();

template<class Rep, class Period>
void sleep_for(const std::chrono::duration<Rep, Period>&);

template<class Clock, class Duration>
void sleep_until(const std::chrono::time_point<Clock, Duration>&);

}

class thread::id {
private:
	std::uintptr_t local;
	int global;
public:


	id();
	id(const id&) = default;
	id& operator=(const id&) = default;
	bool operator==( const id& other) const;
	bool operator!=( const id& other) const;
	bool operator<( const id& other) const;
	bool operator>( const id& other) const;
	bool operator<=( const id& other) const;
	bool operator>=( const id& other) const;

	template<class CharT, class Traits>
	friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>&, thread::id);

	friend class std::hash<thread::id>;
	friend class thread;
	friend id this_thread::get_id();
};

}

namespace std {

template<class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& ost, hpx::thread::id id);

template<>
struct hash<hpx::thread::id> {
	size_t operator()(const hpx::thread::id& id) const;
};

void swap(hpx::thread&, hpx::thread&);

}

#include "detail/thread_impl.hpp"

#endif /* QTHREAD_HPP_ */
