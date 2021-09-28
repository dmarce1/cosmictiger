/*
 * id_type.cpp
 *
 *  Created on: Dec 15, 2015
 *      Author: dmarce1
 */


#include "hpx/id_type.hpp"

namespace hpx {


gid_type id_type::get_gid() const {
	return *this;
}


void id_type::set_address(detail::agas_entry_t** data_ptr, const std::function<void(detail::agas_entry_t**)>& deleter) {
	address = std::shared_ptr<detail::agas_entry_t*>(data_ptr, deleter );

}


std::uintptr_t id_type::get_address() const {
	if (address != nullptr) {
		return reinterpret_cast<std::uintptr_t>(*address);
	} else {
		return 0;
	}
}

void id_type::set_rank(int r) {
	rank = r;
}

int id_type::get_rank() const {
	return rank;
}

id_type::id_type() :
		rank(-1), address(nullptr) {
}

id_type& id_type::operator=(const id_type& other) {
	rank = other.rank;
	address = other.address;
	return *this;
}

id_type& id_type::operator=(id_type&& other ) {
	rank = std::move(other.rank);
	address = std::move(other.address);
	other.rank = -1;
	other.address = nullptr;
	return *this;
}

id_type::id_type(const id_type& other) {
	*this = other;
}

id_type::id_type(id_type&& other) {
	*this = std::move(other);
}


namespace detail {

void add_ref_count(std::uintptr_t id, int change) {
	auto ptr = reinterpret_cast<agas_entry_t*>(id);
	std::unique_lock<hpx::mutex> lock(ptr->mtx);
	ptr->reference_count += change;
	assert(ptr->reference_count >= 0);
	if (ptr->reference_count == 0) {
		auto del_ptr = ptr->deleter;
		(*del_ptr)(reinterpret_cast<void*>(ptr->pointer));
		lock.unlock();
		delete ptr;
	}
}

agas_entry_t::agas_entry_t(std::uintptr_t id, deleter_type _deleter) {
	reference_count = 1;
	pointer = id;
	deleter = _deleter;
}


}


}



