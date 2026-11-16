// Read distributed gravity particle positions and species through the shared cache.
/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#include <cosmictiger/gravity/global_fmm/particle_exchange.hpp>
#include <cosmictiger/particles/particles.hpp>

void globalFmmReadSourcePositions(TreeNode const& source, Position* x, Position* y, Position* z,
    char* types, ParticleInteger offset) {
    particlesGlobalReadPos(source.globalParticleRange(), x, y, z, types, offset);
}
