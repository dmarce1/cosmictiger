/*
 * hpx.hpp
 *
 *  Created on: Sep 17, 2015
 *      Author: dmarce1
 */

#ifndef HPX_HPP_
#define HPX_HPP_

#include <mpi.h>

#include "id_type.hpp"
#include "future.hpp"
#include "mutex.hpp"
#include "serialize.hpp"
#include "thread.hpp"

#define HPX_DEFINE_COMPONENT_ACTION1(class_name, method, action_name ) \
		class action_name : public hpx::detail::component_action< decltype(& class_name :: method), & class_name :: method > {\
			using base_type = hpx::detail::component_action< decltype(& class_name :: method), & class_name :: method >; \
		public: \
			action_name() : base_type( #action_name ) {} \
			static void register_me() { \
				hpx::detail::register_action_name( #action_name,  & action_name ::invoke); \
			} \
		};

#define HPX_DEFINE_COMPONENT_ACTION(class_name, method) \
		HPX_DEFINE_COMPONENT_ACTION1(class_name, method, method##_action)

#define HPX_DEFINE_COMPONENT_DIRECT_ACTION HPX_DEFINE_COMPONENT_ACTION

#define HPX_REGISTER_ACTION( action_type)  \
		__attribute__((constructor)) \
		static void register_##action_type() { \
			action_type :: register_me(); \
		}
#define HPX_REGISTER_ACTION_DECLARATION( action_type)
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(base_type, class_name) \
	__attribute__((constructor)) \
	static void register_constructor_##class_name () { \
		base_type::register_( #class_name , &hpx::detail::create_component< class_name > );\
	}

#define HPX_ACTION_USES_HUGE_STACK(a)
#define HPX_ACTION_USES_LARGE_STACK(a)
#define HPX_ACTION_USES_MEDIUM_STACK(a)
#define HPX_ACTION_USES_SMALL_STACK(a)

#define QCHECK(call)                                         \
	if( (call) != 0 ) {                                        \
		printf( "Qthread returned error\n");                 \
		printf( "File: %s, Line: %i\n", __FILE__, __LINE__); \
		abort();                                             \
	}

#include "detail/hpx_decl.hpp"
#include "detail/hpx_impl.hpp"

namespace hpx {
const static id_type invalid_id;
}

namespace hpx {
namespace lcos {
namespace local {
class counting_semaphore {
	std::atomic<int> counter;
public:
	counting_semaphore() :
			counter(0) {
	}
	counting_semaphore(int cnt) :
			counter(cnt) {
	}
	void wait() {
		while (counter-- < 0) {
			counter++;
			hpx::this_thread::yield();
		}
	}
	void signal() {
		counter++;
	}
};
}
}

enum class execution {
	par
};

namespace parallel {
template<class T, class F>
future<void> sort(execution exec_type, T&& begin, T&& end, F&& less) {
	std::sort(std::forward < T > (begin), std::forward < T > (end), std::forward < F > (less));
	return hpx::make_ready_future();
}

template<class T>
future<void> sort(execution exec_type, T&& begin, T&& end) {
	std::sort(std::forward < T > (begin), std::forward < T > (end));
	return hpx::make_ready_future();
}


template<class T, class V, class U>
future<void> copy(execution exec_type, T&& a1, V&& a2, U&& a3) {
	std::copy(std::forward < T > (a1), std::forward < V > (a2), std::forward<U>(a3));
	return hpx::make_ready_future();
}
}

}

#endif /* HPX_HPP_ */
