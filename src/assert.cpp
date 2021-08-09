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
