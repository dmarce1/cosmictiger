#include <cstdio>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/scenarios/glass.hpp>

namespace scenarios {

void initializeGlass(DriverParameters& state) {
    if (hpxSize() != 1) {
        THROW_ERROR("Glass generation requires one HPX locality\n");
    }
    auto const count = getOptions().initial.numberOfParticles;
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

void saveGlass() {
    int const side = getOptions().initial.particlesPerDimension;
    FILE* file = fopen(outputPath("glass.bin").c_str(), "wb");
    if (file == nullptr) {
        THROW_ERROR("Unable to write glass.bin\n");
    }
    if (fwrite(&side, sizeof(side), 1, file) != 1) {
        fclose(file);
        THROW_ERROR("Unable to write glass dimensions\n");
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        for (ParticleInteger i = 0; i < particlesSize(); i++) {
            auto const position = particlesPos(dim, i);
            if (fwrite(&position, sizeof(position), 1, file) != 1) {
                fclose(file);
                THROW_ERROR("Unable to write glass particle\n");
            }
        }
    }
    if (fclose(file) != 0) {
        THROW_ERROR("Unable to finish writing glass.bin\n");
    }
}

}    // namespace scenarios
