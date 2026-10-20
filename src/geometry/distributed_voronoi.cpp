/*
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 * See LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <cosmictiger/geometry/distributed_voronoi.hpp>

namespace voronoi {
namespace detail {

    struct Bounds {
        Point lower;
        Point upper;
    };

    bool intersects(Bounds const& bounds, SiteQuery const& query) {
        double distance2 = 0;
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            double const distance = std::max({bounds.lower[dimension] - query.center[dimension],
                query.center[dimension] - bounds.upper[dimension], 0.0});
            distance2 += distance * distance;
        }
        return distance2 <= query.radius * query.radius;
    }

    // Both locality routing and point queries use a BVH. No particle is broadcast
    // globally, and routing does not scan every locality for every generator.
    class BoundsTree {
        struct Node {
            Bounds bounds;
            size_t begin;
            size_t end;
            size_t left = 0;
            size_t right = 0;
        };
        std::vector<Node> nodes_;
        std::vector<size_t> order_;
        std::vector<Bounds> bounds_;

        size_t build(size_t begin, size_t end) {
            Bounds bounds = bounds_[order_[begin]];
            for (size_t i = begin + 1; i < end; ++i) {
                for (size_t dimension = 0; dimension < 3; ++dimension) {
                    bounds.lower[dimension] =
                        std::min(bounds.lower[dimension], bounds_[order_[i]].lower[dimension]);
                    bounds.upper[dimension] =
                        std::max(bounds.upper[dimension], bounds_[order_[i]].upper[dimension]);
                }
            }
            size_t const node = nodes_.size();
            nodes_.push_back({bounds, begin, end});
            if (end - begin > 16) {
                size_t axis = 0;
                for (size_t dimension = 1; dimension < 3; ++dimension) {
                    if (bounds.upper[dimension] - bounds.lower[dimension] >
                        bounds.upper[axis] - bounds.lower[axis]) {
                        axis = dimension;
                    }
                }
                size_t const middle = begin + (end - begin) / 2;
                std::nth_element(order_.begin() + begin, order_.begin() + middle,
                    order_.begin() + end, [&](size_t a, size_t b) {
                        return bounds_[a].lower[axis] + bounds_[a].upper[axis] <
                            bounds_[b].lower[axis] + bounds_[b].upper[axis];
                    });
                size_t const left = build(begin, middle);
                size_t const right = build(middle, end);
                nodes_[node].left = left;
                nodes_[node].right = right;
            }
            return node;
        }

        template <typename Callback>
        void visit(size_t nodeIndex, SiteQuery const& query, Callback const& callback) const {
            Node const& node = nodes_[nodeIndex];
            if (!intersects(node.bounds, query)) {
                return;
            }
            if (node.left) {
                visit(node.left, query, callback);
                visit(node.right, query, callback);
            } else {
                for (size_t i = node.begin; i < node.end; ++i) {
                    if (intersects(bounds_[order_[i]], query)) {
                        callback(order_[i]);
                    }
                }
            }
        }

    public:
        explicit BoundsTree(std::vector<Bounds> bounds)
          : bounds_(std::move(bounds)) {
            order_.resize(bounds_.size());
            std::iota(order_.begin(), order_.end(), size_t(0));
            if (!order_.empty()) {
                build(0, order_.size());
            }
        }
        template <typename Callback>
        void query(SiteQuery const& query, Callback const& callback) const {
            if (!nodes_.empty()) {
                visit(0, query, callback);
            }
        }
        Bounds bounds() const {
            return nodes_.empty() ? Bounds{} : nodes_[0].bounds;
        }
    };

    using Identity = std::tuple<uint64_t, int, int, int>;
    Identity identity(Site const& site) {
        return {site.id, site.image[0], site.image[1], site.image[2]};
    }

    std::vector<Bounds> siteBounds(std::span<Site const> sites) {
        std::vector<Bounds> bounds;
        bounds.reserve(sites.size());
        for (auto const& site : sites) {
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                if (!std::isfinite(site.position[dimension]) || site.position[dimension] < 0 ||
                    site.position[dimension] >= 1 || site.image[dimension] != 0) {
                    throw std::invalid_argument(
                        "Voronoi sites must be canonical periodic positions");
                }
            }
            bounds.push_back({site.position, site.position});
        }
        return bounds;
    }

}    // namespace detail

struct SiteIndex::Impl {
    std::vector<Site> sites;
    detail::BoundsTree tree;
    explicit Impl(std::vector<Site> input)
      : sites(std::move(input))
      , tree(detail::siteBounds(sites)) {}
};

SiteIndex::SiteIndex(std::vector<Site> sites)
  : impl_(std::make_shared<Impl>(std::move(sites))) {}

SitePartition SiteIndex::partition(int owner) const {
    auto const bounds = impl_->tree.bounds();
    return {owner, impl_->sites.size(), bounds.lower, bounds.upper};
}

std::vector<Site> SiteIndex::query(std::span<SiteQuery const> queries) const {
    std::vector<Site> result;
    std::set<detail::Identity> found;
    for (auto const& query : queries) {
        if (!(query.radius >= 0) || !std::isfinite(query.radius)) {
            throw std::invalid_argument("Invalid Voronoi halo radius");
        }
        for (double const coordinate : query.center) {
            if (!std::isfinite(coordinate)) {
                throw std::invalid_argument("Invalid Voronoi halo center");
            }
        }
        impl_->tree.query(query, [&](size_t index) {
            Site site = impl_->sites[index];
            site.image = query.image;
            if (found.insert(detail::identity(site)).second) {
                for (size_t dimension = 0; dimension < 3; ++dimension) {
                    site.position[dimension] += query.image[dimension];
                }
                result.push_back(site);
            }
        });
    }
    return result;
}

std::vector<Cell> buildDistributedCells(std::span<Site const> owned,
    std::span<SitePartition const> partitions, FetchSites const& fetch) {
    if (owned.empty()) {
        return {};
    }
    detail::siteBounds(owned);
    uint64_t total = 0;
    std::vector<detail::Bounds> bounds;
    std::vector<int> owners;
    std::set<int> uniqueOwners;
    for (auto const& partition : partitions) {
        if (!uniqueOwners.insert(partition.owner).second) {
            throw std::invalid_argument("Duplicate Voronoi locality metadata");
        }
        if (partition.count) {
            if (std::numeric_limits<uint64_t>::max() - total < partition.count) {
                throw std::overflow_error("Voronoi global site count overflow");
            }
            total += partition.count;
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                if (!std::isfinite(partition.lower[dimension]) ||
                    !std::isfinite(partition.upper[dimension]) || partition.lower[dimension] < 0 ||
                    partition.upper[dimension] >= 1 ||
                    partition.lower[dimension] > partition.upper[dimension]) {
                    throw std::invalid_argument("Invalid Voronoi locality bounds");
                }
            }
            bounds.push_back({partition.lower, partition.upper});
            owners.push_back(partition.owner);
        }
    }
    if (total < owned.size() || !uniqueOwners.count(owned[0].owner)) {
        throw std::invalid_argument("Missing Voronoi locality metadata");
    }
    detail::BoundsTree const routing(std::move(bounds));
    std::vector<Site> sites(owned.begin(), owned.end());
    std::vector<size_t> selected(owned.size());
    std::iota(selected.begin(), selected.end(), size_t(0));
    std::map<detail::Identity, size_t> identities;
    for (size_t i = 0; i < sites.size(); ++i) {
        if (sites[i].owner != owned[0].owner ||
            !identities.emplace(detail::identity(sites[i]), i).second) {
            throw std::invalid_argument("Invalid owned Voronoi identities");
        }
    }
    // Images of each generator bound its cell by +/- 1/2 along each axis.
    // Thus sqrt(3) covers twice every possible cell-vertex radius. The 27
    // nearest unit boxes suffice: farther images cannot be nearest to a point
    // in the owner's unwrapped cell. The margin includes cospherical ties.
    double const maximumRadius = std::sqrt(3.0) * (1.0 + 1e-10);
    double const initialRadius = std::min(maximumRadius, 2.0 * std::cbrt(1.0 / double(total)));
    std::vector<double> radii(owned.size(), initialRadius);
    std::vector<bool> search(owned.size(), true);
    std::vector<Cell> cells;
    for (unsigned round = 0; round < 64; ++round) {
        std::map<int, std::vector<SiteQuery>> queries;
        for (size_t i = 0; i < owned.size(); ++i) {
            if (!search[i]) {
                continue;
            }
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    for (int z = -1; z <= 1; ++z) {
                        SiteQuery query{owned[i].position, radii[i], {x, y, z}};
                        for (size_t dimension = 0; dimension < 3; ++dimension) {
                            query.center[dimension] -= query.image[dimension];
                        }
                        routing.query(query, [&](size_t index) {
                            int const owner = owners[index];
                            if (owner != owned[i].owner || x != 0 || y != 0 || z != 0) {
                                queries[owner].push_back(query);
                            }
                        });
                    }
                }
            }
        }
        std::vector<SiteQueryBatch> batches;
        batches.reserve(queries.size());
        for (auto& [owner, queriesForOwner] : queries) {
            batches.push_back({owner, std::move(queriesForOwner)});
        }
        for (auto const& site : fetch(batches)) {
            auto const [entry, inserted] = identities.emplace(detail::identity(site), sites.size());
            if (inserted) {
                sites.push_back(site);
            } else if (sites[entry->second].position != site.position ||
                sites[entry->second].owner != site.owner) {
                throw std::runtime_error("Inconsistent Voronoi halo identity");
            }
        }
        cells = buildCells(sites, selected);
        bool finished = true;
        for (size_t i = 0; i < cells.size(); ++i) {
            search[i] = !cells[i].complete || !(cells[i].searchRadius * (1.0 + 1e-12) < radii[i]);
            if (search[i]) {
                if (radii[i] == maximumRadius) {
                    throw std::runtime_error("Voronoi halo did not close at the periodic bound");
                }
                finished = false;
                radii[i] = std::min(
                    maximumRadius, std::max(radii[i] * 2.0, cells[i].searchRadius * (1.0 + 1e-10)));
            }
        }
        if (finished) {
            return cells;
        }
    }
    throw std::runtime_error("Voronoi halo expansion failed to converge");
}

}    // namespace voronoi
