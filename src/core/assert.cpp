/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include <fstream>
#include <iostream>

#include <boost/stacktrace.hpp>

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/safe_io.hpp>

void cosmictigerAssert(char const* condStr, bool cond, char const* filename, int line) {
    if (!cond) {
        std::cout << "\nCondition \"" << condStr << "\" failed in " << filename << " on line "
                  << std::to_string(line) << " on rank " << hpxRank() << "\n";
        std::cout << "\nStack trace:\n" << boost::stacktrace::stacktrace() << "\n";
        std::ofstream fp("stack_trace.txt", std::ios::app);
        fp << "\nCondition \"" << condStr << "\" failed in " << filename << " on line "
           << std::to_string(line) << " on rank " << hpxRank() << "\n";
        fp << "\nStack trace:\n %s\n" << boost::stacktrace::stacktrace() << "\n";
        fp.close();
        abort();
    }
}
