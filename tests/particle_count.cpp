#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/core/particle_count.hpp>

TEST(ParticleCount, CubicRequestsRoundUpAtExactBoundaries) {
    EXPECT_EQ(particle_count::roundLattice(999).count, 1000);
    EXPECT_EQ(particle_count::roundLattice(1000).count, 1000);
    EXPECT_EQ(particle_count::roundLattice(1001).count, 1331);
    EXPECT_EQ(particle_count::roundLattice(1331).side, 11);
    EXPECT_EQ(particle_count::roundLattice(0).count, 0);
}

TEST(ParticleCount, ClosePackedAndLayeredLayoutsKeepWholeCells) {
    EXPECT_EQ(particle_count::roundLattice(999, 2).count, 1024);
    EXPECT_EQ(particle_count::roundLattice(999, 4).count, 1372);
    EXPECT_EQ(particle_count::roundLattice(108, 4).count, 108);
    EXPECT_EQ(particle_count::roundLattice(576, 72).count, 576);
    EXPECT_EQ(particle_count::roundLattice(577, 72).count, 1944);
    EXPECT_EQ(particle_count::roundLattice(4096, 96).count, 6144);
    EXPECT_EQ(particle_count::roundLattice(999, 1, 4).count, 1728);
    EXPECT_EQ(particle_count::roundLattice(1728, 1, 4).count, 1728);
}

TEST(ParticleCount, KelvinHelmholtzUsesCommonCellsWithAnExactDensityRatio) {
    for (size_t sites : {1, 2, 4}) {
        for (size_t requested : {1, 511, 512, 999, 1000, 1001, 262144}) {
            auto const layout = particle_count::roundKelvinHelmholtz(requested, sites);
            EXPECT_GE(layout.count, requested);
            EXPECT_EQ(layout.count, 12 * sites * particle_count::spatialPower(layout.side));
            EXPECT_EQ(layout.highCount, 2 * layout.lowCount);
            EXPECT_EQ(
                particle_count::roundKelvinHelmholtz(layout.count, sites).count, layout.count);
        }
    }
}

TEST(ParticleCount, OverflowIsRejectedBeforeMultiplication) {
    EXPECT_THROW(
        particle_count::roundLattice(std::numeric_limits<size_t>::max()), std::overflow_error);
    EXPECT_THROW(particle_count::roundLattice(1, 0), std::invalid_argument);
    EXPECT_THROW(particle_count::roundLattice(1, 1, 0), std::invalid_argument);
    EXPECT_THROW(particle_count::roundKelvinHelmholtz(std::numeric_limits<size_t>::max(), 1),
        std::overflow_error);
}
