// Local gravity source views and callback-based remote position access.
/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <atomic>

#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/particles/particles.hpp>

static std::atomic<GravityRemoteSourceReader> remoteSourceReader{nullptr};

/// Publish the callback used to gather remote owned-particle source data.
void gravitySetRemoteSourceReader(GravityRemoteSourceReader reader) {
    remoteSourceReader.store(reader);
}

/// Gather a node's positions and optional types from its local store or remote reader.
void gravityReadSourcePositions(TreeNode const& source, Position* x, Position* y, Position* z,
    char* types, ParticleInteger offset) {
    if (!source.isLocalHere()) {
        auto const reader = remoteSourceReader.load();
        ALWAYS_ASSERT(reader != nullptr);
        reader(source, x, y, z, types, offset);
        return;
    }
    auto const view = fmmSourceView();
    auto const range = source.particleRange;
    ALWAYS_ASSERT(range.first >= 0 && range.second <= view.count);
    ParticleInteger const count = range.second - range.first;
    std::copy_n(view.positions[kXi] + range.first, count, x + offset);
    std::copy_n(view.positions[kYi] + range.first, count, y + offset);
    std::copy_n(view.positions[kZi] + range.first, count, z + offset);
    if (types != nullptr) {
        if (view.types != nullptr) {
            std::copy_n(view.types + range.first, count, types + offset);
        } else {
            std::fill_n(types + offset, count, kDarkMatterType);
        }
    }
}

/// Return a non-owning view of the local particle arrays.
FmmSourceView fmmSourceView() {
    auto const& opts = getOptions();
    FmmSourceView view;
    view.positions = particlesX;
    view.types = opts.hydro.enabled ? particlesTy : nullptr;
    view.count = particlesSize();
    return view;
}
