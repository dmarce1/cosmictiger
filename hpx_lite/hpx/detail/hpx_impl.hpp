/*
 * hpx_impl.hpp
 *
 *  Created on: Oct 1, 2015
 *      Author: dmarce1
 */

#ifndef HPX_IMPL_HPP_
#define HPX_IMPL_HPP_

namespace hpx {

template<class Type, class ...Args>
hpx::future<id_type> new_(const id_type& id, Args&&...args) {
#ifndef HPX_ALWAYS_REMOTE
	if (id == find_here()) {
		return hpx::make_ready_future<id_type>(
				hpx::detail::create_component_direct<Type, Args...>(std::forward<Args>(args)...));
	} else {
#endif
		hpx::detail::ibuffer_type send_buffer;
		hpx::future<id_type> future;
		const std::uintptr_t index = hpx::detail::promise_table_generate_future<id_type>(future);
		send_buffer << hpx::detail::invokation_type(NAME);
		send_buffer << Type::class_name();
		send_buffer << index;
		std::uintptr_t promise_function = reinterpret_cast<std::uintptr_t>(&hpx::detail::promise_table_satisfy_promise<
				id_type>);
		send_buffer << promise_function;
		Type tmp(std::forward<Args>(args)...);
		send_buffer << tmp;
//		hpx::detail::pack_args(send_buffer, std::forward<Args>(args)...);
		hpx::detail::remote_action(id.get_rank(), std::move(send_buffer));
		return std::move(future);
#ifndef HPX_ALWAYS_REMOTE
	}
#endif
}


namespace detail {
template<class Type>
std::uintptr_t promise_table_generate_future(hpx::future<Type>& future) {
	auto pointer = new hpx::promise<Type>();
	future = pointer->get_future();
	return reinterpret_cast<std::uintptr_t>(pointer);
}

template<class Type>
void promise_table_satisfy_promise(obuffer_type data) {
	std::uintptr_t index;
	data >> index;
	auto pointer = reinterpret_cast<hpx::promise<Type>*>(index);
	typename std::remove_reference<typename std::remove_const<Type>::type>::type argument;
	data >> argument;
	pointer->set_value(std::move(argument));
	delete pointer;
}

template<class T, class ...Args>
void pack_args(ibuffer_type& buffer, T&& arg1, Args&&...rest) {
	buffer << arg1;
	pack_args(buffer, std::forward<Args>(rest)...);
}

template<class ReturnType, class ...Args, class ...ArgsDone>
typename std::enable_if<std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		ReturnType (*fptr)(Args...), ibuffer_type& obuffer, const obuffer_type& buffer, ArgsDone&&...args_done) {
	assert(fptr);
	obuffer << (*fptr)(std::forward<ArgsDone>(args_done)...);
}

template<class ...Args, class ...ArgsDone>
typename std::enable_if<std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		void (*fptr)(Args...), ibuffer_type& obuffer, const obuffer_type& buffer, ArgsDone&&...args_done) {
	assert(fptr);
	(*fptr)(std::forward<ArgsDone>(args_done)...);
}

template<class ReturnType, class ...Args, class ...ArgsDone>
typename std::enable_if<!std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		ReturnType (*fptr)(Args...), ibuffer_type& ibuffer, const obuffer_type& buffer, ArgsDone&&...args_done) {
	assert(fptr);
	typedef typename variadic_type<sizeof...(ArgsDone), Args...>::type next_type;
	typename std::remove_const<typename std::remove_reference<next_type>::type>::type next_data;
	buffer >> next_data;
	invoke(fptr, ibuffer, buffer, std::forward<ArgsDone>(args_done)..., std::forward<next_type>(next_data));
}

template<class Type, class ...Args>
void create_component(hpx::detail::ibuffer_type& return_buffer, hpx::detail::obuffer_type buffer) {
	id_type id;
	int rank;
	rank = hpx::detail::mpi_comm_rank();
	Type* ptr;
	ptr = new Type();
	buffer >> *ptr;
	auto entry = new detail::agas_entry_t(reinterpret_cast<std::uintptr_t>(ptr), [](void* ptr) {
		auto _ptr = reinterpret_cast<Type*>(ptr);
		delete _ptr;
	});
	std::uintptr_t del_index = reinterpret_cast<std::uintptr_t>(entry);
	auto ptrptr = new (detail::agas_entry_t*)(entry);
	id.set_address(ptrptr, [=](detail::agas_entry_t** ptrptr) {
		hpx::detail::add_ref_count(del_index, -1);
		delete ptrptr;
	});
	id.set_rank(rank);
	return_buffer << id;
}

template<class Type, class ...Args>
id_type create_component_direct(Args ... args) {
	id_type id;
	int rank;
	rank = hpx::detail::mpi_comm_rank();
	Type* ptr;
	ptr = new Type(std::move(args)...);
	auto entry = new detail::agas_entry_t(reinterpret_cast<std::uintptr_t>(ptr), [](void* ptr) {
		auto _ptr = reinterpret_cast<Type*>(ptr);
		delete _ptr;
	});
	std::uintptr_t del_index = reinterpret_cast<std::uintptr_t>(entry);
	auto ptrptr = new (detail::agas_entry_t*)(entry);
	id.set_address(ptrptr, [=](detail::agas_entry_t** ptrptr) {
		hpx::detail::add_ref_count(del_index, -1);
		delete ptrptr;
	});
	id.set_rank(rank);
	return id;
}

}

namespace detail {

template<class ReturnType, class ...Args>
action_base<ReturnType(Args...)>::action_base(const std::string& str) :
		name(str) {
}

template<class ReturnType, class ...Args2>
template<class ...Args>
ReturnType action_base<ReturnType(Args2...)>::operator()(const id_type& id, Args&& ...args) const {
	ReturnType result;
	if (find_here() != id) {
		auto future = get_action_future(id, std::forward<Args>(args)...);
		result = future.get();
	} else {
		result = (*fptr)(std::forward<Args>(args)...);
	}
	return result;
}

template<class ReturnType, class ...Args2>
template<class ...Args>
hpx::future<ReturnType> action_base<ReturnType(Args2...)>::get_action_future(const id_type& id, Args&& ...args) const {
	hpx::detail::ibuffer_type send_buffer;
	hpx::future<ReturnType> future;
	const std::uintptr_t index = promise_table_generate_future<ReturnType>(future);
	send_buffer << invokation_type(NAME);
	send_buffer << name;
	send_buffer << index;
	std::uintptr_t promise_function = reinterpret_cast<std::uintptr_t>(&hpx::detail::promise_table_satisfy_promise<
			ReturnType>);
	send_buffer << promise_function;
	hpx::detail::pack_args(send_buffer, std::forward<Args>(args)...);
	hpx::detail::remote_action(id.get_rank(), std::move(send_buffer));
	return future;
}

template<class ...Args>
action_base<void(Args...)>::action_base(const std::string& str) :
		name(str) {
}

template<class ...Args2>
template<class ...Args>
hpx::future<void> action_base<void(Args2...)>::get_action_future(const id_type& id, Args&& ...args) const {
	hpx::detail::ibuffer_type send_buffer;
	hpx::future<void> future;
	const std::uintptr_t index = promise_table_generate_future(future);
	send_buffer << invokation_type(NAME);
	send_buffer << name;
	send_buffer << index;
	std::uintptr_t promise_function = reinterpret_cast<std::uintptr_t>(&hpx::detail::promise_table_satisfy_void_promise);
	send_buffer << promise_function;
	hpx::detail::pack_args(send_buffer, std::forward<Args>(args)...);
	hpx::detail::remote_action(id.get_rank(), std::move(send_buffer));
	return future;
}

template<class ...Args2>
template<class ...Args>
void action_base<void(Args2...)>::operator()(const id_type& id, Args&& ...args) const {
	if (find_here() != id) {
		auto future = get_action_future(id, std::forward<Args>(args)...);
		future.get();
	} else {
		(*fptr)(std::forward<Args>(args)...);
	}
}

template<class ReturnType, class ...Args>
component_action_base<ReturnType(std::uintptr_t, Args...)>::component_action_base(const std::string& str) :
		action_base<ReturnType(std::uintptr_t, Args...)>::action_base(str) {
}

template<class ReturnType, class ...Args2>
template<class ... Args>
ReturnType component_action_base<ReturnType(std::uintptr_t, Args2...)>::operator()(const id_type& id,
		Args&& ...args) const {
	id_type remote_id;
	remote_id.set_rank(id.get_rank());
	std::uintptr_t agas_ptr = id.get_address();
	return action_base<ReturnType(std::uintptr_t, Args2...)>::operator()(remote_id, agas_ptr,
			std::forward<Args>(args)...);
}

template<class ReturnType, class ...Args2>
template<class ...Args>
future<ReturnType> component_action_base<ReturnType(std::uintptr_t, Args2...)>::get_action_future(const id_type& id,
		Args&& ...args) const {
	id_type remote_id;
	remote_id.set_rank(id.get_rank());
	std::uintptr_t agas_ptr = id.get_address();
	return action_base<ReturnType(std::uintptr_t, Args2...)>::get_action_future(remote_id, std::move(agas_ptr),
			std::forward<Args>(args)...);
}

template<class ...Args2>
template<class ...Args>
future<void> component_action_base<void(std::uintptr_t, Args2...)>::get_action_future(const id_type& id,
		Args&& ...args) const {
	id_type remote_id;
	remote_id.set_rank(id.get_rank());
	std::uintptr_t agas_ptr = id.get_address();
	return action_base<void(std::uintptr_t, Args2...)>::get_action_future(remote_id, std::move(agas_ptr),
			std::forward<Args>(args)...);
}

template<class ...Args>
component_action_base<void(std::uintptr_t, Args...)>::component_action_base(const std::string& str) :
		action_base<void(std::uintptr_t, Args...)>(str) {
}

template<class ...Args2>
template<class ...Args>
void component_action_base<void(std::uintptr_t, Args2...)>::operator()(const id_type& id, Args&& ...args) const {
	id_type remote_id;
	remote_id.set_rank(id.get_rank());
	std::uintptr_t agas_ptr = id.get_address();
	action_base<void(std::uintptr_t, Args2...)>::operator()(remote_id, agas_ptr, std::forward<Args>(args)...);
}

template<class Type, class ...Args>
std::uintptr_t creator<Type, Args...>::call(Args&&...args) {
	return reinterpret_cast<std::uintptr_t>(new Type(std::forward<Args>(args)...));
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...)>
struct component_action<ReturnType (ClassType::*)(Args...), Pointer> : public hpx::detail::component_action_base<
		ReturnType(std::uintptr_t, Args...)> {
	using base_type = hpx::detail::component_action_base<ReturnType(std::uintptr_t, Args...)>;
	typedef ReturnType return_type;
	static ReturnType call(std::uintptr_t _this, Args ...args);
	static void invoke(hpx::detail::ibuffer_type& ibuf, hpx::detail::obuffer_type buffer);
	component_action(const std::string&);
};

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...) const>
struct component_action<ReturnType (ClassType::*)(Args...) const, Pointer> : public hpx::detail::component_action_base<
		ReturnType(std::uintptr_t, Args...)> {
	using base_type = hpx::detail::component_action_base<ReturnType(std::uintptr_t, Args...)>;
	typedef ReturnType return_type;
	static ReturnType call(std::uintptr_t _this, Args ...args);
	static void invoke(hpx::detail::ibuffer_type& ibuf, hpx::detail::obuffer_type buffer);
	component_action(const std::string&);
};

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...)>
struct component_action<void (ClassType::*)(Args...), Pointer> : public hpx::detail::component_action_base<
		void(std::uintptr_t, Args...)> {
	using base_type = hpx::detail::component_action_base<void(std::uintptr_t, Args...)>;
	typedef void return_type;
	static void call(std::uintptr_t _this, Args ...args);
	static void invoke(hpx::detail::ibuffer_type& ibuf, hpx::detail::obuffer_type buffer);
	component_action(const std::string&);

};

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...) const>
struct component_action<void (ClassType::*)(Args...) const, Pointer> : public hpx::detail::component_action_base<
		void(std::uintptr_t, Args...)> {
	using base_type = hpx::detail::component_action_base<void(std::uintptr_t, Args...)>;
	typedef void return_type;
	static void call(std::uintptr_t _this, Args ...args);
	static void invoke(hpx::detail::ibuffer_type& ibuf, hpx::detail::obuffer_type buffer);
	component_action(const std::string&);
};

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...)>
ReturnType component_action<ReturnType (ClassType::*)(Args...), Pointer>::call(std::uintptr_t _this, Args ...args) {
	auto ptr = Pointer;
	auto __this = reinterpret_cast<ClassType*>(reinterpret_cast<hpx::detail::agas_entry_t*>(_this)->pointer);
	return ((*__this).*ptr)(args...);
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...)>
void component_action<ReturnType (ClassType::*)(Args...), Pointer>::invoke(hpx::detail::ibuffer_type& ibuf,
		hpx::detail::obuffer_type buffer) {
	hpx::detail::invoke(call, ibuf, buffer);
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...)>
component_action<ReturnType (ClassType::*)(Args...), Pointer>::component_action(const std::string& str) :
		base_type(str) {
	hpx::detail::component_action_base<ReturnType(std::uintptr_t, Args...)>::fptr = call;
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...) const>
ReturnType component_action<ReturnType (ClassType::*)(Args...) const, Pointer>::call(std::uintptr_t _this,
		Args ...args) {
	auto ptr = Pointer;
	auto __this = reinterpret_cast<const ClassType*>(reinterpret_cast<hpx::detail::agas_entry_t*>(_this)->pointer);
	return ((*__this).*ptr)(args...);
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...) const>
void component_action<ReturnType (ClassType::*)(Args...) const, Pointer>::invoke(hpx::detail::ibuffer_type& ibuf,
		hpx::detail::obuffer_type buffer) {
	hpx::detail::invoke(call, ibuf, buffer);
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...) const>
component_action<ReturnType (ClassType::*)(Args...) const, Pointer>::component_action(const std::string& str) :
		base_type(str) {
	hpx::detail::component_action_base<ReturnType(std::uintptr_t, Args...)>::fptr = call;
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...)>
void component_action<void (ClassType::*)(Args...), Pointer>::call(std::uintptr_t _this, Args ...args) {
	auto ptr = Pointer;
	auto __this = reinterpret_cast<ClassType*>(reinterpret_cast<hpx::detail::agas_entry_t*>(_this)->pointer);
	((*__this).*ptr)(std::forward<Args>(args)...);
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...)>
void component_action<void (ClassType::*)(Args...), Pointer>::invoke(hpx::detail::ibuffer_type& ibuf,
		hpx::detail::obuffer_type buffer) {
	hpx::detail::invoke(call, ibuf, buffer);
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...)>
component_action<void (ClassType::*)(Args...), Pointer>::component_action(const std::string& str) :
		base_type(str) {
	hpx::detail::component_action_base<void(std::uintptr_t, Args...)>::fptr = call;
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...) const>
void component_action<void (ClassType::*)(Args...) const, Pointer>::call(std::uintptr_t _this, Args ...args) {
	auto ptr = Pointer;
	auto __this = reinterpret_cast<const ClassType*>(reinterpret_cast<hpx::detail::agas_entry_t*>(_this)->pointer);
	((*__this).*ptr)(args...);
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...) const>
void component_action<void (ClassType::*)(Args...) const, Pointer>::invoke(hpx::detail::ibuffer_type& ibuf,
		hpx::detail::obuffer_type buffer) {
	hpx::detail::invoke(call, ibuf, buffer);
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...) const>
component_action<void (ClassType::*)(Args...) const, Pointer>::component_action(const std::string& str) :
		base_type(str) {
	hpx::detail::component_action_base<void(std::uintptr_t, Args...)>::fptr = call;
}

}

}

#endif /* HPX_IMPL_HPP_ */
