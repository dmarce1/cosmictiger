/*
 * id_type.hpp
 *
 *  Created on: Dec 15, 2015
 *      Author: dmarce1
 */

#ifndef ID_TYPE_HPP_
#define ID_TYPE_HPP_

#include "serialize.hpp"
#include "future.hpp"
#include "mutex.hpp"


#define HPX_DEFINE_PLAIN_ACTION(func, action_name)                                                \
		struct action_name : public hpx::detail::action_base<decltype(func)>  {                   \
			static void invoke( hpx::detail::ibuffer_type&, hpx::detail::obuffer_type);      \
			typedef hpx::detail::action_base<decltype(func)> base_type;                           \
			action_name() :                                                                       \
					base_type( #action_name ) {        \
			   fptr = &func;                                                                      \
			}                                                                                     \
		}                                                                                         \


#define HPX_REGISTER_PLAIN_ACTION( action_name )                                                         \
	    __attribute__((constructor))  \
	    static void register_##action_name() { \
			hpx::detail::register_action_name( #action_name,  & action_name ::invoke); \
		} \
		void action_name ::invoke(hpx::detail::ibuffer_type& ibuf,  hpx::detail::obuffer_type buffer) {  \
			action_name act;                                                                             \
			hpx::detail::invoke(act.fptr, ibuf, buffer );                                    \
        }

#define HPX_PLAIN_ACTION2(func, name)           \
		HPX_DEFINE_PLAIN_ACTION(func, name );  \
		HPX_REGISTER_PLAIN_ACTION( name )

#define HPX_PLAIN_ACTION1(func)           \
		HPX_DEFINE_PLAIN_ACTION(func, func##_action );  \
		HPX_REGISTER_PLAIN_ACTION( func##_action )

#define HPX_GET_PLAIN_ACTION(_1,_2,NAME,...) NAME

#define HPX_PLAIN_ACTION(...) HPX_GET_PLAIN_ACTION(__VA_ARGS__, HPX_PLAIN_ACTION2, HPX_PLAIN_ACTION1)(__VA_ARGS__)


#include "hpx/detail/id_type_detail.hpp"

namespace hpx {

typedef id_type gid_type;

}

HPX_DEFINE_PLAIN_ACTION(hpx::detail::add_ref_count, action_add_ref_count);

namespace hpx {
class id_type {
	int rank;
	std::shared_ptr<detail::agas_entry_t*> address;

public:

	void set_address(detail::agas_entry_t** data_ptr, const std::function<void(detail::agas_entry_t**)>& deleter);
	void set_rank(int);
	std::uintptr_t get_address() const;
	int get_rank() const;

	id_type();
	id_type(const id_type& other);
	id_type(id_type&& other);
	~id_type() = default;
	id_type& operator=(const id_type& other);
	id_type& operator=(id_type&& other);
	bool operator==(const id_type&) const;
	bool operator!=(const id_type&) const;
	gid_type get_gid() const;

	template<class Archive>
	void load(const Archive& arc, const unsigned);

	template<class Archive>
	void save(Archive& arc, const unsigned) const;

	HPX_SERIALIZATION_SPLIT_MEMBER()
	;


};


template<class Function, class ... Args>
future<typename Function::return_type> async(const id_type& id, Args&& ... args);

}

#include "hpx/detail/id_type_impl.hpp"

#endif /* ID_TYPE_HPP_ */
