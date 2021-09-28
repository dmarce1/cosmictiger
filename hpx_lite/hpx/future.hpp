/*
 * future.hpp
 *
 *  Created on: Dec 11, 2015
 *      Author: dmarce1
 */

#ifndef HPX_FUTURE_HPP_
#define HPX_FUTURE_HPP_

#include "detail/future_detail.hpp"
#include "mutex.hpp"
#include "serialize.hpp"
#include <future>

#define HPX_THROW( e ) throw( e )

namespace hpx {

using future_error = std::future_error;
using future_errc = std::future_errc;
using future_status = std::future_status;

template<class T>
class promise;

template<class T>
class future;

enum class launch {
	deferred, async
};

template<class Function, class ... Args>
future<typename std::result_of<Function(Args...)>::type> async(Function &&f, Args &&... args);

template<class Function, class ... Args>
void apply(Function &&f, Args &&... args) {
	hpx::async(std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class ... Args>
future<typename std::result_of<Function(Args...)>::type> async(hpx::launch policy, Function &&f, Args &&... args);

template<class T>
class future {
private:
	std::shared_ptr<detail::shared_state<T>> state;
	std::function<T()> deferred;
	void set_deferred(std::function<T()>&& f) {
		deferred = std::move(f);
	}
public:
	future() = default;
	future(const future&) = delete;
	future(future&&) = default;
	~future();
	future& operator=(const future&) = delete;
	future& operator=(future&&) = default;
	void wait();
	bool valid() const;
	bool is_ready() const;
	T get();
	future(hpx::future<hpx::future<T>>&& other);
	future& operator=(hpx::future<hpx::future<T>>&& other);

	template<class Function>
	future<typename std::result_of<Function(hpx::future<T>&&)>::type> then(Function&& function);

	template<class Archive>
	void load(const Archive& arc, const unsigned);

	template<class Archive>
	void save(Archive& arc, const unsigned) const;

	HPX_SERIALIZATION_SPLIT_MEMBER()
	;

	friend class promise<T> ;

	template<class >
	friend class future;

	template<class Function, class ... Args>
	friend future<typename std::result_of<Function(Args...)>::type> async(hpx::launch policy, Function &&f, Args &&... args);

};

template<>
class future<void> : public future<char> {
public:
	future() = default;
	future(const future&) = delete;
	future(future&&) = default;
	future(future<char> &&);
	~future() = default;
	future& operator=(const future&) = delete;
	future& operator=(future&&) = default;

	void get();
	future(hpx::future<hpx::future<void>>&& other);
	future& operator=(hpx::future<hpx::future<void>>&& other);
};

template<class T>
class shared_future {
private:
	struct shared_future_data {
		std::shared_ptr<hpx::mutex> mtx;
		bool ready;
		T data;
		hpx::future<T> fut;
	};
	std::shared_ptr<shared_future_data> ptr;
public:
	shared_future() = default;
	shared_future(const shared_future&) = default;
	shared_future(shared_future&&) = default;
	~shared_future() = default;
	shared_future& operator=(const shared_future&) = default;
	shared_future& operator=(shared_future&&) = default;

	shared_future& operator=(hpx::future<T> &&other) {
		shared_future_data data;
		data.ready = false;
		data.fut = std::move(other);
		data.mtx = std::make_shared<hpx::mutex>();
		ptr = std::make_shared<shared_future_data>(std::move(data));
		return *this;
	}

	shared_future(hpx::future<T> &&other) {
		*this = std::move(other);
	}

	T& get() {
		std::lock_guard<hpx::mutex> lock(*ptr->mtx);
		if (!ptr->ready) {
			ptr->data = ptr->fut.get();
			ptr->ready = true;
		}
		return ptr->data;
	}

};

template<>
class shared_future<void> {
private:
	struct shared_future_data {
		std::shared_ptr<hpx::mutex> mtx;
		bool ready;
		hpx::future<void> fut;
	};
	std::shared_ptr<shared_future_data> ptr;
public:
	shared_future() = default;
	shared_future(const shared_future&) = default;
	shared_future(shared_future&&) = default;
	~shared_future() = default;
	shared_future& operator=(const shared_future&) = default;
	shared_future& operator=(shared_future&&) = default;

	shared_future& operator=(hpx::future<void> &&other) {
		shared_future_data data;
		data.ready = false;
		data.fut = std::move(other);
		data.mtx = std::make_shared<hpx::mutex>();
		ptr = std::make_shared<shared_future_data>(std::move(data));
		return *this;
	}

	shared_future(hpx::future<void> &&other) {
		*this = std::move(other);
	}

	void get() {
		std::lock_guard<hpx::mutex> lock(*ptr->mtx);
		if (!ptr->ready) {
			ptr->fut.get();
			ptr->ready = true;
		}
	}

};

template<class T>
class promise {
private:
	std::shared_ptr<detail::shared_state<T>> state;
	std::shared_ptr<std::atomic<int>> future_retrieved;
	std::shared_ptr<hpx::mutex> set_mtx;
public:
	promise();
	promise(const promise&) = delete;
	promise(promise&&) = default;
	~promise();
	promise& operator=(const promise&) = delete;
	promise& operator=(promise&&) = default;

	template<class V>
	void set_value(V&& v);

	template<class V>
	void set_value(const V& v);

	template<class V>
	void set_value_at_thread_exit(V&& v);

	template<class V>
	void set_value_at_thread_exit(const V& v);

	future<T> get_future() const;
	void set_exception(std::exception_ptr);
	void set_exception_at_thread_exit(std::exception_ptr);

	//	template< class Alloc >
	//	promise( std::allocator_arg_t, const Alloc& );

	// void swap( promise& );
};

template<>
class promise<void> : public promise<char> {
public:
	promise() = default;
	promise(const promise&) = delete;
	promise(promise&&) = default;
	~promise() = default;
	promise& operator=(const promise&) = delete;
	promise& operator=(promise&&) = default;
	void set_value();
	void set_value_at_thread_exit();
	future<void> get_future() const;
};

namespace lcos {
namespace local {
template<class T>
using promise = hpx::promise<T>;
}
}

namespace detail {

template<class T>
typename std::enable_if<!std::is_void<T>::value>::type set_promise(hpx::promise<T>& prms, std::function<T()>&& func);

template<class T>
typename std::enable_if<std::is_void<T>::value>::type set_promise(hpx::promise<T>& prms, std::function<T()>&& func);

}

hpx::future<void> make_ready_future();

template<class Type>
future<typename std::decay<Type>::type> make_ready_future(Type&& arg);

template<class Type>
future<typename std::decay<Type>::type> make_exceptional_future(std::exception exception);

template<class Iterator>
typename std::enable_if<!hpx::detail::is_future<Iterator>::value>::type wait_all(Iterator, Iterator);

void wait_all();

template<class F1, class ...FR>
typename std::enable_if<hpx::detail::is_future<F1>::value>::type wait_all(F1&, FR&...);

template<class Function, class ... Args>
future<typename std::result_of<Function(Args...)>::type> async(Function&& f, Args&&... args);

hpx::future<std::tuple<>> when_all();

template<class F1, class ...FR>
hpx::future<std::tuple<F1, FR...>> when_all(F1&, FR&...);

template<class Iterator>
hpx::future<std::vector<typename std::iterator_traits<Iterator>::value_type>> when_all(Iterator, Iterator);

}
#include "detail/future_impl.hpp"

#endif /* FUTURE_HPP_ */
