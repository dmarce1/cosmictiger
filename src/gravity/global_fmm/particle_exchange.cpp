/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#include <cosmictiger/gravity/global_fmm/particle_exchange.hpp>
#include <cosmictiger/gravity/global_fmm/source_forest.hpp>
#include <cosmictiger/particles/particles.hpp>

void globalFmmReadSourcePositions(TreeNode const& source, Position* x, Position* y, Position* z,
    char* types, float* masses, ParticleInteger offset) {
    if (treeSourceForestReadParticleRange(source, x, y, z, types, masses, offset)) {
        return;
    }
    if (treeSourceForestImportsEnabled()) {
        treeSourceForestCountFallback();
    }
    particlesGlobalReadPos(source.globalParticleRange(), x, y, z, types, offset);
    if (masses != nullptr) {
        particlesGlobalReadMasses(source.globalParticleRange(), masses, offset);
    }
}
