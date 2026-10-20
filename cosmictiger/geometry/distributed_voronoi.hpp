/*
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 * See LICENSE_1_0.txt.
 */

#ifndef COSMICTIGER_GEOMETRY_DISTRIBUTED_VORONOI_HPP_
#define COSMICTIGER_GEOMETRY_DISTRIBUTED_VORONOI_HPP_

#include <functional>
#include <memory>
#include <span>
#include <vector>

#include <cosmictiger/geometry/voronoi.hpp>

namespace voronoi {

struct SitePartition {
    int owner = 0;
    uint64_t count = 0;
    Point lower = {};
    Point upper = {};
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & owner & count;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            archive& lower[dimension] & upper[dimension];
        }
    }
};

struct SiteQuery {
    // Center in the source locality's fundamental unit box. Returned positions
    // are translated by image; the search radius is in code lengths.
    Point center = {};
    double radius = 0;
    std::array<int, 3> image = {};
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & radius;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            archive& center[dimension] & image[dimension];
        }
    }
};

struct SiteQueryBatch {
    int owner = 0;
    std::vector<SiteQuery> queries;
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & owner & queries;
    }
};

// Immutable spatial index: safe for concurrent remote requests throughout a
// snapshot/evolution geometry phase. Only generator geometry crosses ranks.
class SiteIndex {
    struct Impl;
    std::shared_ptr<Impl const> impl_;

public:
    explicit SiteIndex(std::vector<Site> sites);
    SitePartition partition(int owner) const;
    std::vector<Site> query(std::span<SiteQuery const> queries) const;
};

using FetchSites = std::function<std::vector<Site>(std::vector<SiteQueryBatch> const&)>;

// Periodic unit-box tessellation; all owned generators must be supplied. The
// transport callback completes every requested search before returning. It may
// batch asynchronous rank requests; an incomplete transport must throw.
// Output order matches owned. No view/crop selection belongs in this phase.
// A bounded cell is accepted only after its Delaunay circumspheres are covered
// by the searched halo. IDs are caller-owned and can persist across evolution.
std::vector<Cell> buildDistributedCells(std::span<Site const> owned,
    std::span<SitePartition const> partitions, FetchSites const& fetch);

}    // namespace voronoi

#endif
