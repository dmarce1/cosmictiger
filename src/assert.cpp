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

#include <cosmictiger/assert.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/safe_io.hpp>

#include <iostream>
#include <fstream>

#include <boost/stacktrace.hpp>

void cosmictiger_assert(const char* cond_str, bool cond, const char* filename, int line) {
	if (!cond) {
		std::cout << "\nCondition \"" << cond_str << "\" failed in " << filename << " on line " << std::to_string(line) << " on rank " << hpx_rank() << "\n";
		std::cout << "\nStack trace:\n" << boost::stacktrace::stacktrace() << "\n";
		std::ofstream fp("stack_trace.txt", std::ios::app);
		fp << "\nCondition \"" << cond_str << "\" failed in " << filename << " on line " << std::to_string(line) << " on rank " << hpx_rank() << "\n";
		fp << "\nStack trace:\n %s\n" << boost::stacktrace::stacktrace() << "\n";
		fp.close();
		abort();
	}
}
