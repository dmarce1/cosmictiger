/*
 * future_impl.hpp
 *
 *  Created on: Dec 11, 2015
 *      Author: dmarce1
 */

#ifndef FUTURE_IMPL_HPP_
#define FUTURE_IMPL_HPP_

#include <cassert>
#include <functional>

namespace hpx {

namespace detail {

template<class T>
typename std::enable_if<!std::is_void<T>::value>::type set_promise(hpx::promise<T> &prms, std::function<T()> &&func) {
	prms.set_value(func());
}

template<class T>
typename std::enable_if<std::is_void<T>::value>::type set_promise(hpx::promise<T> &prms, std::function<T()> &&func) {
	func();
	prms.set_value();
}

}

template<class T>
promise<T>::promise() :
		state(std::make_shared<detail::shared_state<T>>()), future_retrieved(std::make_shared<std::atomic<int>>(0)), set_mtx(std::make_shared<hpx::mutex>()) {
}

template<class T>
promise<T>::~promise() {
	if (state != nullptr) {
		if (!state->is_ready()) {
			std::exception_ptr eptr = detail::store_exception(hpx::future_error(hpx::future_errc::broken_promise));
			state->set_exception(eptr);
		}
	}
}

template<class T>
template<class V>
void promise<T>::set_value(V &&v) {
	std::lock_guard < hpx::mutex > lock(*set_mtx);
	if (state != nullptr) {
		if (!state->is_ready()) {
			state->set_value(std::forward < V > (v));
		} else {
			HPX_THROW(hpx::future_error(hpx::future_errc::promise_already_satisfied));
		}
	} else {
		HPX_THROW(hpx::future_error(hpx::future_errc::no_state));
	}
}

template<class T>
void promise<T>::set_exception(std::exception_ptr e) {
	std::lock_guard < hpx::mutex > lock(set_mtx);
	if (state != nullptr) {
		if (!state->is_ready()) {
			state->set_exception(e);
		} else {
			HPX_THROW(hpx::future_error(hpx::future_errc::promise_already_satisfied));
		}
	} else {
		HPX_THROW(hpx::future_error(hpx::future_errc::no_state));
	}
}

template<class T>
template<class V>
void promise<T>::set_value(const V &v) {
	auto tmp = v;
	set_value(std::move(v));
}

template<class T>
template<class V>
void promise<T>::set_value_at_thread_exit(V &&v) {
	std::lock_guard < hpx::mutex > lock(*set_mtx);
	if (state != nullptr) {
		if (!state->is_ready()) {
			state->set_value_at_thread_exit(std::forward < V > (v));
		} else {
			HPX_THROW(hpx::future_error(hpx::future_errc::promise_already_satisfied));
		}
	} else {
		HPX_THROW(hpx::future_error(hpx::future_errc::no_state));
	}
}

template<class T>
void promise<T>::set_exception_at_thread_exit(std::exception_ptr e) {
	std::lock_guard < hpx::mutex > lock(set_mtx);
	if (state != nullptr) {
		if (!state->is_ready()) {
			state->set_exception_at_thread_exit(e);
		} else {
			HPX_THROW(hpx::future_error(hpx::future_errc::promise_already_satisfied));
		}
	} else {
		HPX_THROW(hpx::future_error(hpx::future_errc::no_state));
	}
}

template<class T>
template<class V>
void promise<T>::set_value_at_thread_exit(const V &v) {
	auto tmp = v;
	set_value_at_thread_exit(std::move(v));
}

template<class T>
future<T> promise<T>::get_future() const {
	future < T > fut;
	if ((*future_retrieved)++ != 0) {
		HPX_THROW(hpx::future_error(hpx::future_errc::future_already_retrieved));

	} else if (state == nullptr) {
		HPX_THROW(hpx::future_error(hpx::future_errc::no_state));
	} else {
		fut.state = state;
	}
	return std::move(fut);
}

template<class Type>
hpx::future<typename std::decay<Type>::type> make_ready_future(Type &&arg) {
	typedef typename std::decay<Type>::type data_type;
	hpx::promise<data_type> promise;
	promise.set_value(std::forward<Type>(arg));
	return promise.get_future();
}

template<class Type>
hpx::future<typename std::decay<Type>::type> make_exceptional_future(std::exception exception) {
	typedef typename std::decay<Type>::type data_type;
	hpx::promise<data_type> promise;
	promise.set_exception(exception);
}

template<class T>
bool future<T>::valid() const {
	if (deferred) {
		return true;
	} else {
		return state != nullptr;
	}
}

template<class T>
bool future<T>::is_ready() const {
	if (deferred) {
		return true;
	} else {
		return state->is_ready();
	}
}

template<class T>
void future<T>::wait() {
	if (!deferred) {
		state->wait();
	}
}
namespace detail {
template<class T>
bool shared_state<T>::join() {
	std::lock_guard < hpx::mutex > lock(mtx);
	const bool rc = this_thread.joinable();
	if (rc) {
		this_thread.join();
	}
	return rc;
}
}

template<class T>
T future<T>::get() {
	if (deferred) {
		T res = deferred();
		deferred = nullptr;
		return res;
	} else {
		assert(state != nullptr);
		detail::thread_exit te([=]() {
			state = nullptr;
		});
		state->join();
		return state->get();
	}
}

template<class T>
future<T>::~future() {
}

template<class T>
future<T>::future(hpx::future<hpx::future<T>> &&other) {
	*this = std::move(other);
}

template<class T>
future<T>& future<T>::operator=(hpx::future<hpx::future<T>> &&other) {
	auto promise_ptr = std::make_shared<hpx::promise<T>>();
	auto future_ptr = std::make_shared < hpx::future<hpx::future<T>> > (std::move(other));
	*this = promise_ptr->get_future();
	hpx::thread([=]() {
		hpx::future < T > fut = future_ptr->get();
		promise_ptr->set_value(fut.get());
	}).detach();

	return *this;
}

template<class T>
template<class Archive>
void future<T>::load(const Archive &arc, const unsigned) {
	T data;
	bool is_valid;
	arc >> is_valid;
	if (is_valid) {
		arc >> data;
		*this = hpx::make_ready_future < T > (std::move(data));
	} else {
		*this = hpx::future<T>();
	}
}

template<class T>
template<class Archive>
void future<T>::save(Archive &arc, const unsigned) const {
	bool is_valid;
	is_valid = this->valid();
	arc << is_valid;
	if (is_valid) {
		arc << const_cast<future<T>&>(*this).get();
	}
}

template<class T>
template<class Function>
future<typename std::result_of<Function(hpx::future<T>&&)>::type> future<T>::then(Function &&function) {
	using then_type = typename std::result_of<Function(hpx::future<T>&&)>::type;

	struct continuation_data_t {
		hpx::promise<then_type> prom;
		hpx::future<T> fut;
		Function func;
		continuation_data_t(Function &&_f) :
				func(std::move(_f)) {
		}
	};

	auto cont_ptr = std::make_shared < continuation_data_t > (std::move(function));
	cont_ptr->fut = std::move(*this);
	auto return_future = cont_ptr->prom.get_future();
	auto then_function = std::function < void() > ([=]() {
		cont_ptr->fut.wait();
		hpx::detail::set_promise(cont_ptr->prom, std::function < then_type() > ([&]() {
			return cont_ptr->func(std::move(cont_ptr->fut));
		}));
	});

	if (cont_ptr->fut.state->set_continuation(then_function)) {
		return_future.state->set_thread(cont_ptr->fut.state->get_thread());
	} else {
		return_future.state->set_thread(hpx::thread(std::move(then_function)));
	}

	return std::move(return_future);
}

template<class F, class ... Args>
hpx::future<typename std::result_of<F(Args...)>::type> async(F &&f, Args &&... args) {
	using tuple_type = std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>;
	using return_type = typename std::result_of<F(Args...)>::type;
	auto tup_ptr = std::make_shared<tuple_type>(std::make_tuple(std::forward<F>(f), std::forward<Args>(args)...));
	return make_ready_future().then([=](hpx::future<void>&&) {
		return detail::invoke_function<return_type, F, typename std::decay<Args>::type...>(*tup_ptr);
	});
}

template<class F, class ... Args>
hpx::future<typename std::result_of<F(Args...)>::type> async(hpx::launch policy, F &&f, Args &&... args) {
	using tuple_type = std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>;
	using return_type = typename std::result_of<F(Args...)>::type;
	auto tup_ptr = std::make_shared<tuple_type>(std::make_tuple(std::forward<F>(f), std::forward<Args>(args)...));
	if (policy == launch::async) {
		return make_ready_future().then([=](hpx::future<void>&&) {
			return detail::invoke_function<return_type, F, typename std::decay<Args>::type...>(*tup_ptr);
		});
	} else {
		auto func = [=] {
			return detail::invoke_function<return_type, F, typename std::decay<Args>::type...>(*tup_ptr);
		};
		hpx::future<return_type> future;
		future.set_deferred(std::move(func));
		return future;
	}
}

template<class Iterator>
typename std::enable_if<!hpx::detail::is_future<Iterator>::value>::type wait_all(Iterator begin, Iterator end) {
	for (auto i = begin; i != end; ++i) {
		i->wait();
	}
}

template<class F1, class ...FR>
typename std::enable_if<hpx::detail::is_future<F1>::value>::type wait_all(F1 &f1, FR &...frest) {
	wait_all(frest...);
	f1.wait();
}

template<class F1, class ...FR>
hpx::future<std::tuple<F1, FR...>> when_all(F1 &f1, FR &...frest) {
	return hpx::async([](F1 &&f1, FR &&...frest) {
		f1.wait();
		std::tuple<FR...> tup2 = when_all(frest...).get();
		std::tuple<F1> tup1(std::move(f1));
		return std::tuple_cat<decltype(tup1), decltype(tup2)>(std::move(tup1), std::move(tup2));
	}, std::move(f1), std::move(frest)...);
}

template<class Iterator>
hpx::future<std::vector<typename std::iterator_traits<Iterator>::value_type>> when_all(Iterator begin, Iterator end) {
	auto futs_ptr = std::make_shared<std::vector<typename std::iterator_traits<Iterator>::value_type>>();
	futs_ptr->reserve(std::distance(begin, end));
	for (auto i = begin; i != end; ++i) {
		futs_ptr->push_back(std::move(*i));
	}
	return hpx::async([=]() {
		hpx::wait_all(futs_ptr->begin(), futs_ptr->end());
		return std::move(*futs_ptr);
	});
}

}
#endif /* FUTURE_IMPL_HPP_ */
