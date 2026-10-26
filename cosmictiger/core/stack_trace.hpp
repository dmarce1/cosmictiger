// Copyright (c) 2026 Octo-Tiger AUTHORS
// Copyright (c) 2026 Dominic C. Marcello
// Distributed under the Boost Software License, Version 1.0.

#pragma once

#include <memory>
#include <string>

// One owner per process, kept alive until the HPX runtime has stopped.
// Install from an HPX startup callback, which runs on worker localities too.
class StackTraceHandler {
public:
    StackTraceHandler();
    ~StackTraceHandler();

    StackTraceHandler(StackTraceHandler const&) = delete;
    StackTraceHandler& operator=(StackTraceHandler const&) = delete;

    void install(unsigned int localityId = 0);

private:
    struct State;
    std::unique_ptr<State> state_;
};

// The same symbolization backend is used for assertions and fatal signals.
std::string stackTrace();
