/*
 * hpx_fwd.hpp
 *
 *  Created on: Dec 21, 2015
 *      Author: dmarce1
 */

#ifndef HPX_FWD_HPP_
#define HPX_FWD_HPP_

/************FORWARD DECLARATIONS******************/

namespace hpx {

namespace detail {

typedef char invokation_type;
#define NAME  (char) 1
#define ADDRESS (char) 2
#define PROMISE (char) 3

struct agas_entry_t;
}

namespace detail {

int& mpi_comm_rank();

int& mpi_comm_size();
template<int Index, class ...Args>
struct variadic_type;

template<class T>
struct component_action_base;

template<class MethodType, MethodType MethodPtr>
struct component_action;

}


/****************************************************typedefs***************************/
namespace detail {
typedef void (*naked_action_type)(ibuffer_type&, obuffer_type);
typedef void (*promise_action_type)(obuffer_type);
}

}
#endif /* HPX_FWD_HPP_ */
