/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstdio>
#include <memory>

// Deleter for std::unique_ptr<FILE>. A functor avoids GCC's
// -Wignored-attributes warning that decltype(&fclose) triggers, because glibc
// attaches attributes to fclose that a template argument cannot carry.
struct FileCloser {
    void operator()(FILE* file) const {
        if (file != nullptr) {
            std::fclose(file);
        }
    }
};

using UniqueFile = std::unique_ptr<FILE, FileCloser>;
