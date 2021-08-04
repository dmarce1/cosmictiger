/*
 * future.cpp
 *
 *  Created on: Dec 11, 2015
 *      Author: dmarce1
 */

#include "hpx/future.hpp"

namespace hpx {

hpx::future<std::tuple<>> when_all() {
	return hpx::make_ready_future<std::tuple<>>(std::tuple<>());
}

void wait_all() {
	return;
}

future<void>::future(future<char>&& f) {
	future<char>::operator=(std::move(f));
}


future<void>::future(hpx::future<hpx::future<void>>&& other) {
	auto promise_ptr = std::make_shared<hpx::promise<void>>();
	auto future_ptr = std::make_shared < hpx::future<hpx::future<void>>>(std::move(other));
	*this = promise_ptr->get_future();
	hpx::thread([=]() {
		hpx::future<void> fut = future_ptr->get();
		fut.get();
		promise_ptr->set_value();
	}).detach();

}

void future<void>::get() {
	future<char>::get();
}


void promise<void>::set_value() {
	promise<char>::set_value(char(1));
}


void promise<void>::set_value_at_thread_exit() {
	promise<char>::set_value_at_thread_exit(char(1));
}

future<void> promise<void>::get_future() const {
	future<void> f = promise<char>::get_future();
	return std::move(f);
}

hpx::future<void> make_ready_future() {
	hpx::promise<void> promise;
	promise.set_value();
	return promise.get_future();
}

namespace detail {

std::exception_ptr store_exception(const std::exception& e) {
	std::exception_ptr eptr = nullptr;
	try {
		HPX_THROW(e);
	} catch (...) {
		eptr = std::current_exception();
	}
	return eptr;
}


}

}
