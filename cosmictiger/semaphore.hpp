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
	void wait(int count) {
		while ((available -= count) < 0) {
			available += count;
			hpx_yield();
		}
	}
	bool try_wait(int count) {
		if ((available -= count) < 0) {
			available += count;
			return false;
		} else {
			return true;
		}
	}
	void signal(int count) {
		available += count;
	}
};

#endif /* SEMAPHORE_HPP_ */
