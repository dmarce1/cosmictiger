/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef RANDOM_NUMBER_GENERATOR_HPP_
#define RANDOM_NUMBER_GENERATOR_HPP_

class RandomNumberGenerator {
    // L’Ecuyer, Blouin and Coutre
    static constexpr unsigned kA1_ = 107374182;
    static constexpr unsigned kA5_ = 104480;
    static constexpr unsigned kM_ = (1U << 31) - 1;
    unsigned x1_;
    unsigned x2_;
    unsigned x3_;
    unsigned x4_;
    unsigned x5_;

public:
    void reseed(int seed) {
        unsigned const a0 = 48271;
        x1_ = seed;
        x2_ = (x1_ * a0) % kM_;
        x3_ = (x2_ * a0) % kM_;
        x4_ = (x3_ * a0) % kM_;
        x5_ = (x4_ * a0) % kM_;
    }
    RandomNumberGenerator(int seed) {
        reseed(seed);
    }
    int randomNumber() {
        long long const x0 = (kA1_ * x1_ + kA5_ * x5_) % kM_;
        x5_ = x4_;
        x4_ = x3_;
        x3_ = x2_;
        x2_ = x1_;
        x1_ = x0;
        return (int) x0;
    }
    int randomMaximum() const {
        return kM_ - 1;
    }
    float randomUnitInterval() {
        return ((float) randomNumber() + 0.5f) / (float) randomMaximum();
    }
};

#endif /* RANDOM_NUMBER_GENERATOR_HPP_ */
