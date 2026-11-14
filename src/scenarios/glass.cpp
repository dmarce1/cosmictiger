#include <cstdio>
#include <ios>
#include <stdexcept>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/scenarios/glass.hpp>

namespace scenarios {

/// Create random single-locality particles for a fixed-background glass relaxation.
void initializeGlass(DriverParameters& state) {
    auto const& options = getOptions();
    if (hpxSize() != 1) {
        throw std::invalid_argument("Glass generation requires one HPX locality");
    }
    if (options.initial.closePack) {
        throw std::invalid_argument("Glass generation does not support initial.closePack; the file "
                                    "stores side^3 particles");
    }
    auto const count = options.initial.numberOfParticles;
    particlesResize(count);
    for (ParticleInteger i = 0; i < count; i++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            particlesPos(dim, i) = rand1();
            particlesVel(dim, i) = 0.0;
        }
        particlesRung(i) = 0;
    }
    state.tauMax = 1.0;
    state.a = 1.0;
    state.adot = 0.0;
}

/// Validate the cubic particle count and write fixed-point glass positions in
/// dimension-major order.
void saveGlass() {
    auto const& opts = getOptions();
    int const side = opts.initial.latticeSide;
    if (uint64_t(particlesSize()) != uint64_t(side) * side * side) {
        throw std::logic_error("Glass particle count does not match its lattice dimensions");
    }
    FILE* file = fopen(outputPath("glass.bin").c_str(), "wb");
    if (file == nullptr) {
        throw std::ios_base::failure("Unable to write glass.bin");
    }
    if (fwrite(&side, sizeof(side), 1, file) != 1) {
        fclose(file);
        throw std::ios_base::failure("Unable to write glass dimensions");
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        for (ParticleInteger i = 0; i < particlesSize(); i++) {
            Fixed32 const position(particlesPos(dim, i));
            if (fwrite(&position, sizeof(position), 1, file) != 1) {
                fclose(file);
                throw std::ios_base::failure("Unable to write glass particle");
            }
        }
    }
    if (fclose(file) != 0) {
        throw std::ios_base::failure("Unable to finish writing glass.bin");
    }
}

}    // namespace scenarios
