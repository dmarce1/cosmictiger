/*
 * semaphore.hpp
 *
 *  Created on: Oct 27, 2021
 *      Author: dmarce1
 */

#ifndef SEMAPHORE_HPP_
#define SEMAPHORE_HPP_


#include <cosmictiger/hpx.hpp>

class semaphore {
	std::atomic<int> available;
public:
	semaphore(int cnt) {
		available = cnt;
	}
	semaphore() {
		available = 0;
	}
	void wait() {
		while( --available < 0 ) {
			available++;
			hpx_yield();
		}

	}
	void signal() {
		available++;
	}
};


#endif /* SEMAPHORE_HPP_ */
