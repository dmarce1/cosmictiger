/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#include <fstream>
#include <iostream>

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/core/diagnostic_output.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/stack_trace.hpp>

void cosmictigerAssert(char const* condStr, bool cond, char const* filename, int line) {
    if (!cond) {
        std::cout << "\nCondition \"" << condStr << "\" failed in " << filename << " on line "
                  << std::to_string(line) << " on rank " << hpxRank() << "\n";
        auto const trace = stackTrace();
        std::cout << "\nStack trace:\n" << trace << "\n";
        std::ofstream fp(stackTraceOutputPath(), std::ios::app);
        fp << "\nCondition \"" << condStr << "\" failed in " << filename << " on line "
           << std::to_string(line) << " on rank " << hpxRank() << "\n";
        fp << "\nStack trace:\n" << trace << "\n";
        fp.close();
        abort();
    }
}
