/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstddef>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>

struct TreeId {
    int proc;
    int index;
    inline bool operator==(TreeId other) const {
        return proc == other.proc && index == other.index;
    }
    inline bool operator<(TreeId other) const {
        if (proc < other.proc) {
            return true;
        } else if (proc > other.proc) {
            return false;
        } else if (index < other.index) {
            return true;
        } else {
            return false;
        }
    }
    inline bool operator!=(TreeId other) const {
        return proc != other.proc || index != other.index;
    }
    template <class A>
    void serialize(A&& a, unsigned) {
        a & proc;
        a & index;
    }
};

struct TreeIdHash {
    inline size_t operator()(TreeId id) const {
        int const lineSize = getOptions().compute.treeCacheLineSize;
        int const i = id.index / lineSize;
        return i * (hpxSize() - 1) + ((id.proc < hpxRank()) ? id.proc : id.proc - 1);
    }
};

struct TreeIdHashLo {
    inline size_t operator()(TreeId id) const {
        TreeIdHash hash;
        return hash(id) % kTreeCacheSize;
    }
};

struct TreeIdHashHi {
    inline size_t operator()(TreeId id) const {
        TreeIdHash hash;
        return hash(id) / kTreeCacheSize;
    }
};
