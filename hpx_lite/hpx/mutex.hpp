/*
 * mutex.hpp
 *
 *  Created on: Dec 11, 2015
 *      Author: dmarce1
 */

#ifndef HPX_MUTEX_HPP_
#define HPX_MUTEX_HPP_

#include <atomic>
#include <mutex>
#include <memory>

namespace hpx {

class mutex {
private:
	std::atomic<int> locked;
public:
	constexpr mutex() :
			locked(0) {
	}
	mutex(const mutex&) = delete;
	mutex(mutex&&) = delete;
	~mutex() = default;
	mutex& operator=(const mutex&) = delete;
	mutex& operator=(mutex&&) = delete;
	void lock();
	void unlock();
};

class spinlock {
private:
	std::atomic<int> locked;
public:
	constexpr spinlock() :
			locked(0) {
	}
	spinlock(const spinlock&) = delete;
	spinlock(spinlock&&) = delete;
	~spinlock() = default;
	spinlock& operator=(const spinlock&) = delete;
	spinlock& operator=(spinlock&&) = delete;
	void lock();
	void unlock();
};

namespace lcos {
namespace local {
using spinlock = hpx::spinlock;
using mutex = hpx::mutex;
}
}
}

#include "hpx/thread.hpp"


namespace hpx {
namespace lcos {
namespace local {
class shared_mutex {
private:
	mutex mtx;
	int read_cnt;
	bool writing;
public:
	constexpr shared_mutex() :
			read_cnt(0), writing(false) {
	}
	shared_mutex(const mutex&) = delete;
	shared_mutex(mutex&&) = delete;
	~shared_mutex() = default;
	shared_mutex& operator=(const mutex&) = delete;
	shared_mutex& operator=(mutex&&) = delete;
	void lock() {
		bool done = false;
		mtx.lock();
		writing = true;
		mtx.unlock();
		do {
			mtx.lock();
			if (read_cnt == 0) {
				done = true;
			}
			mtx.unlock();
			hpx::this_thread::yield();
		} while (!done);

	}
	void unlock() {
		mtx.lock();
		writing = false;
		mtx.unlock();
	}
	void lock_shared() {
		bool done = false;
		do {
			mtx.lock();
			if (!writing) {
				read_cnt++;
				done = true;
			}
			mtx.unlock();
			hpx::this_thread::yield();
		} while (!done);
	}
	void unlock_shared() {
		mtx.lock();
		read_cnt--;
		mtx.unlock();
	}
};
}}}

namespace hpx {

inline void mutex::lock() {
	while (locked++ != 0) {
		locked--;
		hpx::this_thread::yield();
	}
}

inline void mutex::unlock() {
	locked--;
}

inline void spinlock::lock() {
	while (locked++ != 0) {
		locked--;
		hpx::this_thread::yield();
	}
}

inline void spinlock::unlock() {
	locked--;
}

}

#endif /* MUTEX_HPP_ */
