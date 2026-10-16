#include <csignal>
#include <fstream>
#include <iostream>

#include <boost/stacktrace.hpp>

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/stack_trace.hpp>

static void handler(int sig) {
    std::cout << "\nSIGNAL \"" << std::to_string(sig) << " on rank " << hpxRank() << "\n";
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

struct StackTraceActivator {
    StackTraceActivator() {
        signal(SIGABRT, handler);
        signal(SIGSEGV, handler);
        signal(SIGFPE, handler);
        signal(SIGILL, handler);
        signal(SIGBUS, handler);
    }
    inline void touch() {}
};

static thread_local StackTraceActivator activator;

void stackTraceActivate() {
    activator.touch();
}
