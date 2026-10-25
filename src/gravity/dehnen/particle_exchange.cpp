/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#include <cosmictiger/gravity/dehnen/particle_exchange.hpp>
#include <cosmictiger/particles/particles.hpp>

void dehnenReadSourcePositions(TreeNode const& source, Position* x, Position* y, Position* z,
    char* types, ParticleInteger offset) {
    particlesGlobalReadPos(source.globalParticleRange(), x, y, z, types, offset);
}
