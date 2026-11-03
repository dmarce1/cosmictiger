/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#include <cosmictiger/gravity/global_fmm/particle_exchange.hpp>
#include <cosmictiger/particles/particles.hpp>

void globalFmmReadSourcePositions(TreeNode const& source, Position* x, Position* y, Position* z,
    char* types, float* masses, ParticleInteger offset) {
    particlesGlobalReadPos(source.globalParticleRange(), x, y, z, types, offset);
    if (masses != nullptr) {
        particlesGlobalReadMasses(source.globalParticleRange(), masses, offset);
    }
}
