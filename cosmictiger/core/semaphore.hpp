/*
 * semaphore.hpp
 *
 *  Created on: Oct 27, 2021
 *      Author: dmarce1
 */

#ifndef SEMAPHORE_HPP_
#define SEMAPHORE_HPP_

#include <cosmictiger/core/hpx.hpp>

class Semaphore {
    std::atomic<int> available_;

public:
    /// Initialize the semaphore's available-permit count.
    Semaphore(int cnt) {
        available_ = cnt;
    }
    /// Initialize the semaphore's available-permit count.
    Semaphore() {
        available_ = 0;
    }
    /// Acquire count permits, yielding execution and retrying while fewer permits are
    /// available.
    void wait(int count) {
        while ((available_ -= count) < 0) {
            available_ += count;
            hpxYield();
        }
    }
    /// Attempt to acquire count permits without waiting; restore the count and return
    /// false on failure.
    bool tryWait(int count) {
        if ((available_ -= count) < 0) {
            available_ += count;
            return false;
        } else {
            return true;
        }
    }
    /// Release count permits for waiting or future callers.
    void signal(int count) {
        available_ += count;
    }
};

#endif /* SEMAPHORE_HPP_ */
