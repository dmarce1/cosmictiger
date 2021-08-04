/*
 * id_type_detail.hpp
 *
 *  Created on: Dec 15, 2015
 *      Author: dmarce1
 */

#ifndef ID_TYPE_DETAIL_HPP_
#define ID_TYPE_DETAIL_HPP_

namespace hpx {

class id_type;

namespace detail {

struct agas_entry_t;


template<class T>
struct action_base;


typedef void (*deleter_type)(void*);


struct agas_entry_t {
	std::uintptr_t pointer;
	hpx::mutex mtx;
	int reference_count;
	deleter_type deleter;

	agas_entry_t(std::uintptr_t, deleter_type);
};

void add_ref_count(std::uintptr_t, int change);


template<class ReturnType, class ...Args>
struct action_base<ReturnType(Args...)> {
	typedef ReturnType return_type;
	const std::string name;
	ReturnType (*fptr)(Args...);

	action_base(const std::string&);

	template<class ...Args2>
	ReturnType operator()(const id_type& id, Args2&& ...args) const;

	template<class...Args2>
	hpx::future<ReturnType> get_action_future(const id_type& id, Args2&& ...args) const;
};

template<class ...Args>
struct action_base<void(Args...)> {
	typedef void return_type;

	const std::string name;

	void (*fptr)(Args...);

	action_base(const std::string& str);

	template<class ...Args2>
	void operator()(const id_type& id, Args2&& ...args) const;

	template<class...Args2>
	hpx::future<void> get_action_future(const id_type& id, Args2&& ...args) const;
};

}
}


#endif /* ID_TYPE_DETAIL_HPP_ */
