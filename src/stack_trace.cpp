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
