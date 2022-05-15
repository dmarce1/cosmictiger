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

#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/hpx.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <boost/stacktrace.hpp>

static void handler(int sig) {
	std::cout << "\nSIGNAL \"" << std::to_string(sig) << " on rank " << hpx_rank() << "\n";
	switch (sig) {
	case SIGSEGV:
		PRINT("SIGSEGV\n");
		break;
	case SIGFPE:
		PRINT("SIGFPE\n");
		break;
	case SIGILL:
		PRINT("SIGILL\n");
		break;
	case SIGBUS:
		PRINT("SIGBUS\n");
		break;
	case SIGABRT:
		PRINT("SIGBUS\n");
		break;
	}
	std::cout << "\nStack trace:\n" << boost::stacktrace::stacktrace() << "\n";
	std::ofstream fp("stack_trace.txt", std::ios::app);
	fp << "\nStack trace:\n %s\n" << boost::stacktrace::stacktrace() << "\n";
	fp.close();
	exit(1);
}

struct stack_trace_activator {
	stack_trace_activator() {
		signal(SIGABRT, handler);
		signal(SIGSEGV, handler);
		signal(SIGFPE, handler);
		signal(SIGILL, handler);
		signal(SIGBUS, handler);
	}
	inline void touch() {
	}

};

static thread_local stack_trace_activator activator;

void stack_trace_activate() {
	activator.touch();
}
