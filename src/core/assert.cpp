/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#include <fstream>
#include <iostream>
#include <memory>

#include <boost/stacktrace.hpp>

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/core/diagnostic_output.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/safe_io.hpp>

namespace detail {

std::shared_ptr<std::string const> stackTracePath =
    std::make_shared<std::string const>("stack_trace.txt");

}    // namespace detail

void setStackTraceOutputPath(std::string const& filename) {
    if (*std::atomic_load(&detail::stackTracePath) != filename) {
        std::atomic_store(&detail::stackTracePath, std::make_shared<std::string const>(filename));
    }
}

std::string stackTraceOutputPath() {
    return *std::atomic_load(&detail::stackTracePath);
}

void cosmictigerAssert(char const* condStr, bool cond, char const* filename, int line) {
    if (!cond) {
        std::cout << "\nCondition \"" << condStr << "\" failed in " << filename << " on line "
                  << std::to_string(line) << " on rank " << hpxRank() << "\n";
        std::cout << "\nStack trace:\n" << boost::stacktrace::stacktrace() << "\n";
        std::ofstream fp(stackTraceOutputPath(), std::ios::app);
        fp << "\nCondition \"" << condStr << "\" failed in " << filename << " on line "
           << std::to_string(line) << " on rank " << hpxRank() << "\n";
        fp << "\nStack trace:\n %s\n" << boost::stacktrace::stacktrace() << "\n";
        fp.close();
        abort();
    }
}
