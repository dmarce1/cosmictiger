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

#include <cosmictiger/sph_cuda.hpp>

struct AuxRecord1 {
    Fixed32 x;
    Fixed32 y;
    Fixed32 z;
};

struct AuxRecord2 {
    float vx;
    float vy;
    float vz;
};

struct AuxWorkspace {
    DeviceVector<AuxRecord1> rec1;
    DeviceVector<AuxRecord2> rec2;
    DeviceVector<int> neighbors;
};

__global__ void sphCudaAux(SphRunParameters params, SphRunCudaData data, SphReduction* reduce) {}
