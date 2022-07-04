/*
 * lockfree_queue.hpp
 *
 *  Created on: Jul 3, 2022
 *      Author: dmarce1
 */

#ifndef LOCKFREE_QUEUE_HPP_
#define LOCKFREE_QUEUE_HPP_


#include <cosmictiger/atomic.hpp>
#include <cosmictiger/containers.hpp>

template<class T, int N>
class lockfree_queue {
	using itype = unsigned long long int;
	array<T*, N> q;
	itype qin;
	itype qout;
public:
	lockfree_queue() {
		qin = qout = 0;
		for( int i = 0; i < N; i++) {
			q[i] = nullptr;
		}
	}
	CUDA_EXPORT
	void push(T* ptr) {
		auto in = qin;
		const auto& out = qout;
		ALWAYS_ASSERT (!(in - out >= N && in > out));
		while (atomic_cas((itype*) &q[in % N], (itype) 0, (itype) ptr) != 0) {
			in++;
			ALWAYS_ASSERT (!(in - out >= N && in > out));
		}
		in++;
		atomic_max((itype*) &qin, (itype) in);
	}
	CUDA_EXPORT
	T* pop() {
		const auto& in = qin;
		auto out = qout;
		if (out >= in) {
			return nullptr;
		}
		T* ptr;
		while ((ptr = (T*) atomic_exch((itype*) &q[out % N], (itype) 0)) == nullptr) {
			if (out >= in) {
				return nullptr;
			}
			out++;
		}
		out++;
		atomic_max((itype*) &qout, (itype) out);
		return ptr;
	}
};


#endif /* LOCKFREE_QUEUE_HPP_ */
