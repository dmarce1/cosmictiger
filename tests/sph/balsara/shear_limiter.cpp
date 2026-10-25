/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <gtest/gtest.h>

#include <cosmictiger/sph/balsara/shear_limiter.hpp>

TEST(Balsara, BalsaraFactor) {
    EXPECT_EQ(balsara::balsaraFactor(0.0, 0.0), 0.0);
    EXPECT_EQ(balsara::balsaraFactor(0.0f, 2.0f), 0.0f);
    EXPECT_EQ(balsara::balsaraFactor(-3.0, 0.0), 1.0);
    EXPECT_NEAR(balsara::balsaraFactor(-3.0, 2.0), 0.6, 1.0e-15);
    EXPECT_NEAR(balsara::balsaraFactor(3.0f, 2.0f), 0.6f, 1.0e-7f);
}
