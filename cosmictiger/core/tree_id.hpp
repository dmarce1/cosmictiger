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
    /// Compare both the owning locality and node index.
    inline bool operator==(TreeId other) const {
        return proc == other.proc && index == other.index;
    }
    /// Order tree identifiers lexicographically by locality and then node index.
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
    /// Test whether either the owning locality or node index differs.
    inline bool operator!=(TreeId other) const {
        return proc != other.proc || index != other.index;
    }
    /// Archive this record in the field order required by distributed transport.
    template <class A>
    void serialize(A&& a, unsigned) {
        a & proc;
        a & index;
    }
};

struct TreeIdHash {
    /// Map a remote node cache line and its locality to a combined cache key.
    inline size_t operator()(TreeId id) const {
        int const lineSize = getOptions().compute.treeCacheLineSize;
        int const i = id.index / lineSize;
        return i * (hpxSize() - 1) + ((id.proc < hpxRank()) ? id.proc : id.proc - 1);
    }
};

struct TreeIdHashLo {
    /// Select the cache bucket from the low part of the tree-node hash.
    inline size_t operator()(TreeId id) const {
        TreeIdHash hash;
        return hash(id) % kTreeCacheSize;
    }
};

struct TreeIdHashHi {
    /// Return the remaining tree-node hash after removing the cache-bucket part.
    inline size_t operator()(TreeId id) const {
        TreeIdHash hash;
        return hash(id) / kTreeCacheSize;
    }
};
