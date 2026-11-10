/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#pragma once

#include <chrono>

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;
    double time_;

public:
    /// Initialize a stopwatch with zero accumulated elapsed time.
    inline Timer() {
        time_ = 0.0;
    }
    /// Add the interval since the last start to the accumulated elapsed time.
    inline void stop() {
        std::chrono::time_point<std::chrono::high_resolution_clock> stopTime =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = stopTime - startTime_;
        time_ += dur.count();
    }
    /// Record the starting timestamp of the next measured interval.
    inline void start() {
        startTime_ = std::chrono::high_resolution_clock::now();
    }
    /// Clear accumulated elapsed time.
    inline void reset() {
        time_ = 0.0;
    }
    /// Return the accumulated elapsed time in seconds.
    inline double read() {
        return time_;
    }
};
