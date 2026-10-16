/*
 * semaphore.hpp
 *
 *  Created on: Oct 27, 2021
 *      Author: dmarce1
 */

#ifndef SEMAPHORE_HPP_
#define SEMAPHORE_HPP_

#include <cosmictiger/hpx.hpp>

class Semaphore {
    std::atomic<int> available_;

public:
    Semaphore(int cnt) {
        available_ = cnt;
    }
    Semaphore() {
        available_ = 0;
    }
    void wait(int count) {
        while ((available_ -= count) < 0) {
            available_ += count;
            hpxYield();
        }
    }
    bool tryWait(int count) {
        if ((available_ -= count) < 0) {
            available_ += count;
            return false;
        } else {
            return true;
        }
    }
    void signal(int count) {
        available_ += count;
    }
};

#endif /* SEMAPHORE_HPP_ */
