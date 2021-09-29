/*
 * hpx_decl.hpp
 *
 *  Created on: Oct 1, 2015
 *      Author: dmarce1
 */

#ifndef HPX_DECL_HPP_
#define HPX_DECL_HPP_

#include "hpx_fwd.hpp"

namespace hpx {
/****************************************************Function decs***************************/

int get_locality_id();
std::vector<hpx::id_type> find_all_localities();
int finalize();
hpx::id_type find_here();

namespace naming {
int get_locality_id_from_gid(const gid_type&);
}

namespace detail {

void register_action_name(const std::string& strname, naked_action_type fptr);
naked_action_type lookup_action(const std::string& strname);

void pack_args(ibuffer_type& buffer);
int server(int, char*[]);
void remote_action(int target, obuffer_type&& buffer );
void set_exit_signal();
std::uintptr_t promise_table_generate_future(hpx::future<void>&);
void promise_table_satisfy_void_promise(obuffer_type data);
}

template<class Type, class ...Args>
hpx::future<id_type> new_(const id_type& id, Args&&...args);

namespace detail {

template<class Type>
std::uintptr_t promise_table_generate_future(hpx::future<Type>& future);

template<class Type>
void promise_table_satisfy_promise(obuffer_type data);

template<class T, class ...Args>
void pack_args(ibuffer_type& buffer, T&& arg1, Args&&...rest);

template<class ReturnType, class ...Args, class ...ArgsDone>
typename std::enable_if<std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		ReturnType (*fptr)(Args...), ibuffer_type& obuffer, const obuffer_type& buffer, ArgsDone&&...args_done);

template<class ...Args, class ...ArgsDone>
typename std::enable_if<std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		void (*fptr)(Args...), ibuffer_type& obuffer, const obuffer_type& buffer, ArgsDone&&...args_done);

template<class ReturnType, class ...Args, class ...ArgsDone>
typename std::enable_if<!std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		ReturnType (*fptr)(Args...), ibuffer_type& ibuffer, const obuffer_type& buffer, ArgsDone&&...args_done);

template<class Type, class ...Args>
void create_component(hpx::detail::ibuffer_type& return_buffer, hpx::detail::obuffer_type buffer);

template<class Type, class ...Args>
id_type create_component_direct(Args ... args);
}

namespace components {
template<class Base>
struct managed_component_base {
	static std::string& class_name() {
		static std::string a;
		return a;
	}
	static hpx::detail::naked_action_type& fptr() {
		static hpx::detail::naked_action_type a;
		return a;
	}
	static void register_(std::string _class_name, hpx::detail::naked_action_type _fptr) {
		class_name() = _class_name;
		fptr() = _fptr;
		hpx::detail::register_action_name(class_name(), fptr());
	}
};

template<class Base>
using managed_component = managed_component_base<Base>;

}


namespace detail {

template<class ReturnType, class ...Args>
struct component_action_base<ReturnType(std::uintptr_t, Args...)> : public action_base<
		ReturnType(std::uintptr_t, Args...)> {
	component_action_base(const std::string& str);

	template<class ...Args2>
	ReturnType operator()(const id_type& id, Args2&& ...args) const;

	template<class...Args2>
	hpx::future<ReturnType> get_action_future(const id_type& id, Args2&& ...args) const;
};

template<class ...Args>
struct component_action_base<void(std::uintptr_t, Args...)> : public action_base<void(std::uintptr_t, Args...)> {
	component_action_base(const std::string& str);

	template<class ...Args2>
	void operator()(const id_type& id, Args2&& ...args) const;

	template<class...Args2>
	hpx::future<void> get_action_future(const id_type& id, Args2&& ...args) const;
}
;

template<int Index, class NextType, class ...Args>
struct variadic_type<Index, NextType, Args...> {
	typedef typename variadic_type<Index - 1, Args...>::type type;
};

template<int Index>
struct variadic_type<Index> {
	typedef void type;
};

template<class NextType, class ...Args>
struct variadic_type<0, NextType, Args...> {
	typedef NextType type;
};

template<class Type, class ...Args>
struct creator {
	static std::uintptr_t call(Args&&...args);
};

}

}

#endif /* HPX_DECL_HPP_ */
