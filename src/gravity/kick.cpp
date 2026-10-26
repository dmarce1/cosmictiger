/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <stack>
#include <utility>

#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>

namespace detail {

static thread_local std::stack<KickLists> localLists;

KickLists kickAcquireLists() {
    if (localLists.empty()) {
        localLists.push(KickLists());
    }
    KickLists lists = std::move(localLists.top());
    localLists.pop();
    return std::move(lists);
}

void kickReleaseLists(KickLists&& lists) {
    lists.cclist.resize(0);
    lists.cplist.resize(0);
    lists.pclist.resize(0);
    lists.nextlist.resize(0);
    lists.leaflist.resize(0);
    localLists.push(std::move(lists));
}

bool kickSourcesAreSingleRank(Vector<TreeId> const& checklist) {
    for (auto const& source : checklist) {
        if (!treeGetNode(source)->isLocal()) {
            return false;
        }
    }
    return true;
}

}    // namespace detail

KickReturn kickWalkDispatch(KickParameters params, ExpansionBuffer l,
    Array<Position, kDimensions> pos, TreeId self, Vector<TreeId> direct, Vector<TreeId> periodic,
    std::shared_ptr<KickWorkspace> workspace, KickContext const& context) {
    return fmmDispatch([&](auto orderTag) {
        return kickWalk<decltype(orderTag)::value>(params, std::move(l), pos, self,
            std::move(direct), std::move(periodic), std::move(workspace), context);
    });
}
