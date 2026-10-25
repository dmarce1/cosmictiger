/* CosmicTiger - shared gravity source access; BSL-1.0. */
#include <algorithm>
#include <atomic>

#include <cosmictiger/gravity/source_access.hpp>
#include <cosmictiger/particles/particles.hpp>

static std::atomic<GravityRemoteSourceReader> remoteSourceReader{nullptr};
void gravitySetRemoteSourceReader(GravityRemoteSourceReader reader) {
    remoteSourceReader.store(reader);
}
void gravityReadSourcePositions(TreeNode const& source, Position* x, Position* y, Position* z,
    char* types, ParticleInteger offset) {
    if (!source.isLocalHere()) {
        ALWAYS_ASSERT(source.sourceStore == FmmParticleStore::Owned);
        auto const reader = remoteSourceReader.load();
        ALWAYS_ASSERT(reader != nullptr);
        reader(source, x, y, z, types, offset);
        return;
    }
    auto const view = treeSourceView(source);
    auto const range = source.particleRange;
    ALWAYS_ASSERT(range.first >= 0 && range.second <= view.count);
    ParticleInteger const count = range.second - range.first;
    std::copy_n(view.positions[XDIM] + range.first, count, x + offset);
    std::copy_n(view.positions[YDIM] + range.first, count, y + offset);
    std::copy_n(view.positions[ZDIM] + range.first, count, z + offset);
    if (types != nullptr) {
        if (view.types != nullptr) {
            std::copy_n(view.types + range.first, count, types + offset);
        } else {
            std::fill_n(types + offset, count, DARK_MATTER_TYPE);
        }
    }
}
