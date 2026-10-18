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

#include <cosmictiger/drift.hpp>
#include <cosmictiger/hpx.hpp>

HPX_PLAIN_ACTION(drift, DriftAction);

void drift(
    double scale, double dt, double tau0, double tau1, double tauMax, int rung, double drag) {
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futures;
    if (rung < 0) {
        for (auto const& child : hpxChildren()) {
            futures.push_back(
                hpx::async<DriftAction>(child, scale, dt, tau0, tau1, tauMax, rung, drag));
        }
    }
    cudaDrift(rung, scale, dt, tau0, tau1, tauMax, drag);
    particlesMemadviseCpu();
    for (auto& future : futures) {
        future.get();
    }
    profilerExit();
}
