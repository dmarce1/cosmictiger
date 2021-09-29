/*
 * id_type_impl.hpp
 *
 *  Created on: Dec 15, 2015
 *      Author: dmarce1
 */

#ifndef ID_TYPE_IMPL_HPP_
#define ID_TYPE_IMPL_HPP_

#include "hpx_fwd.hpp"

namespace hpx {

template<class Archive>
void id_type::load(const Archive& arc, const unsigned) {
	std::uintptr_t ptr;
	arc >> rank;
	arc >> ptr;
	if (ptr) {
		action_add_ref_count act;
		auto ptrptr = new (detail::agas_entry_t*)(reinterpret_cast<detail::agas_entry_t*>(ptr));
		int remote_rank = rank;
		set_address(ptrptr, [=](detail::agas_entry_t** ptrptr) {
			id_type remote;
			remote.set_rank(remote_rank);
			act(remote, reinterpret_cast<std::uintptr_t>(*ptrptr), -1);
			delete ptrptr;
		});
	} else {
		address = nullptr;
	}
}

template<class Archive>
void id_type::save(Archive& arc, const unsigned) const {
	arc << rank;
	if (address != nullptr) {
		auto ptr = reinterpret_cast<std::uintptr_t>(*(address));
		arc << ptr;
		action_add_ref_count act;
		id_type remote;
		remote.set_rank(rank);
		act(remote, ptr, +1);
	} else {
		arc << std::uintptr_t(0);
	}
}



template<class Function, class ... Args>
future<typename Function::return_type> async(const id_type& id, Args&& ... args) {
	Function function;
	future<typename Function::return_type> future;
	if (id.get_rank() == hpx::detail::mpi_comm_rank()) {
		future = hpx::async(function, id, std::forward<Args>(args)...);
	} else {
		future = function.get_action_future(id, std::forward<Args>(args)...);
	}
	return future;
}


}

#endif /* ID_TYPE_IMPL_HPP_ */
