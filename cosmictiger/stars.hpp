/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <cosmictiger/particles.hpp>

struct StarParticle {
    float zform;
    float timeRemaining;
    float totalLife;
    float stellarMass;
    float y;
    float z;
    int type;
    bool remove;
    int dmIndex;
};

struct StarsStats {
    size_t stars;
    size_t popI;
    size_t popII;
    size_t popIII;
    StarsStats() {
        popI = popII = popIII = stars = 0;
    }
    StarsStats& operator+=(StarsStats const& other) {
        stars += other.stars;
        popI += other.popI;
        popII += other.popII;
        popIII += other.popIII;
        return *this;
    }
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & stars;
        arc & popI;
        arc & popII;
        arc & popIII;
    }
};

StarParticle& starsGet(PartInt index);
PartInt starsSize();
double starsFind(float a, float dt, int minrung, int step, float tau);
void starsRemove(float a, float dt, int minrung, int step);
StarsStats starsStatistics(float);
void starsSave(FILE* fp);
void starsLoad(FILE* fp);
void starsTestMass();
