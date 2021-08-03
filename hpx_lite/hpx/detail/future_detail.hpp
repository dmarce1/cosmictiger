/*
 * future_detail.hpp
 *
 *  Created on: Dec 11, 2015
 *      Author: dmarce1
 */

#ifndef FUTURE_DETAIL_HPP_
#define FUTURE_DETAIL_HPP_

#include "hpx/thread.hpp"
#include "hpx/mutex.hpp"

namespace hpx {

template<class >
class future;

template<class >
class shared_future;

namespace detail {

template<class T>
class shared_state {
private:
	hpx::mutex mtx;
	T data;
	std::shared_ptr<bool> ready_ptr;
	std::exception_ptr exception;
	hpx::thread this_thread;
public:
	bool join();
	void set_thread(hpx::thread&& thd) {
		this_thread = std::move(thd);
	}
	hpx::thread get_thread() {
		return std::move(this_thread);
	}
	bool set_continuation(std::function<void()>& func) {
		bool rc;
		std::lock_guard<hpx::mutex> lock(mtx);
		if (this_thread.joinable()) {
			rc = this_thread.attempt_continuation(func);
		} else {
			rc = false;
		}
		return rc;
	}
	shared_state();
	shared_state(const shared_state&) = delete;
	shared_state(shared_state&&) = delete;
	~shared_state() = default;
	shared_state& operator=(const shared_state&) = delete;
	shared_state& operator=(shared_state&&) = delete;

	template<class V>
	void set_value(V&& value);

	template<class V>
	void set_value_at_thread_exit(V&& value);

	void set_exception(std::exception_ptr);
	void set_exception_at_thread_exit(std::exception_ptr);
	void wait() const;
	bool is_ready() const;
	T get();
	void set_async();
	bool is_async() const;
};

std::exception_ptr store_exception(const std::exception&);

template<class T>
shared_state<T>::shared_state() :
		ready_ptr(std::make_shared<bool>(false)), exception(nullptr) {
}

template<class T>
template<class V>
void shared_state<T>::set_value(V&& value) {
	data = std::forward < V > (value);
	*ready_ptr = true;
}

template<class T>
template<class V>
void shared_state<T>::set_value_at_thread_exit(V&& value) {
	data = std::forward < V > (value);
	hpx::this_thread::detail::set_exit_function([=]() {
		*ready_ptr = true;
	});
}

template<class T>
void shared_state<T>::set_exception(std::exception_ptr except) {
	exception = except;
	*ready_ptr = true;
}

template<class T>
void shared_state<T>::set_exception_at_thread_exit(std::exception_ptr except) {
	exception = except;
	hpx::this_thread::detail::set_exit_function([=]() {
		*ready_ptr = true;
	});
}

template<class T>
bool shared_state<T>::is_ready() const {
	return *ready_ptr;
}

template<class T>
void shared_state<T>::wait() const {
	while (!*ready_ptr) {
		hpx::this_thread::yield(ready_ptr);
	}
}

template<class T>
T shared_state<T>::get() {
	wait();
	if (exception != nullptr) {
		std::rethrow_exception(exception);
	}
	return std::move(data);
}

template<class T>
struct is_future {
	static constexpr bool value = false;
};

template<class T>
struct is_future<hpx::future<T>> {
	static constexpr bool value = true;
};

template<class T>
struct is_future<hpx::shared_future<T>> {
	static constexpr bool value = true;
};

}

}

#endif /* FUTURE_DETAIL_HPP_ */
