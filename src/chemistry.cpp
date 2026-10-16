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
#include <fenv.h>

#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/timer.hpp>

#define NRATES 20
#define NCOOL 15

float const ktoev = float(1. / 11604.45);

HPX_PLAIN_ACTION(chemistryDoStep, ChemistryDoStepAction);

Pair<double> chemistryDoStep(float a, int minrung, float t0, float adot, int dir) {
    profilerEnter(__FUNCTION__);
    Timer tm;
    tm.start();
    Vector<hpx::future<Pair<double>>> futs;
    if (!getOptions().chem) {
        Pair<double> rc;
        rc.first = rc.second = 0.0;
        return rc;
    }
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ChemistryDoStepAction>(c, a, minrung, t0, adot, dir));
    }
    double echange = 0.0;
    int numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads, a, minrung, t0, dir, adot]() {
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            Vector<ChemistryAttributes> chems;
            float const mass = getOptions().sphMass;
            float const stars = getOptions().stars;
            int const n = getOptions().sphNeighborNumber;
            for (ParticleInteger i = b; i < e; i++) {
                int rung1 = sphParticlesRung(i);
                //	int rung2 = sph_particles_oldrung(i);
                if (rung1 >= minrung && !sphParticlesIsstar(i)) {
                    ChemistryAttributes chem;
                    float t = sphParticlesTemperature(i, a);
                    sphParticlesNormalizeFracs(i);
                    float const factor = 1.0f / (1.0f - sphParticlesZ(i));
                    chem.h = sphParticlesH(i) * factor;
                    chem.he = sphParticlesHe0(i) * factor;
                    chem.hp = sphParticlesHp(i) * factor;
                    chem.hn = sphParticlesHn(i) * factor;
                    chem.h2 = sphParticlesH2(i) * factor;
                    chem.hep = sphParticlesHep(i) * factor;
                    chem.hepp = sphParticlesHepp(i) * factor;
#ifdef HOPKINS
                    chem.eint = sphParticlesEintRho(i);
                    chem.rho = sphParticlesRhoRho(i);
#else
                    chem.eint = sphParticlesEint(i);
                    chem.rho = sphParticlesRho(i);
#endif

                    if (chem.he < 0.0 && chem.he > -5e-7) {
                        chem.hep += chem.he;
                        chem.he = 0.0;
                    }
                    if (chem.hepp < 0.0 && chem.hepp > -5e-7) {
                        chem.hep += chem.hepp;
                        chem.hepp = 0.0;
                    }
                    ALWAYS_ASSERT(chem.hepp >= 0.f);
                    ALWAYS_ASSERT(chem.h2 >= 0.f);
                    ALWAYS_ASSERT(chem.hp >= 0.f);
                    ALWAYS_ASSERT(chem.hn >= 0.f);
                    ALWAYS_ASSERT(chem.hep >= 0.f);
                    //					if( stars ) {
                    //						chem.cold_mass = sph_particles_cold_mass(i);
                    //					} else {
                    chem.coldMass = 0.f;
                    //					}
                    if (chem.coldMass > -1e-4 && chem.coldMass < 0.0) {
                        chem.coldMass = 0.0;
                    } else if (chem.coldMass < 0.0) {
                        PRINT("%e\n", chem.coldMass);
                        ALWAYS_ASSERT(chem.coldMass >= 0.0);
                    }
                    double dt = (rungDt[rung1]) * t0;
                    if (stars) {
                    }
                    chem.dt = dt;
                    if (t > 5e8) {
                        PRINT("T-------------> %e\n", t);
                        if (t > TMAX) {
                            int k = sphParticlesParticleIndex(i);
                            float vx = particlesVel(XDIM, k);
                            float vy = particlesVel(YDIM, k);
                            float vz = particlesVel(ZDIM, k);
                            PRINT("CHEMISTRY OUT OF RANGE %e %e %e  %e  %e  %e \n", chem.rho,
                                chem.eint, sphParticlesSmoothingLength(i), vx, vy, vz);
                            PRINT("%e %e %e %e %e %e %e\n", chem.he, chem.hp, chem.hn, chem.h2,
                                chem.hep, chem.hepp, sphParticlesZ(i));
                        }
                    }
                    if (t < 0.f) {
                        PRINT("NEGATIVE T !\n", t);
                        abort();
                    }
                    chems.push_back(chem);
                }
            }
            double flops;
            flops = cudaChemistryStep(chems, a);
            int j = 0;
            double echange = 0.0;
            double const sphMass = getOptions().sphMass;
            for (ParticleInteger i = b; i < e; i++) {
                int rung1 = sphParticlesRung(i);
                if (rung1 >= minrung && !sphParticlesIsstar(i)) {
                    ChemistryAttributes chem = chems[j++];
                    //					double cv = 1.5 + 0.5* chem.H2 / (1. - .75 *
                    //(chem.He+chem.Hep+chem.Hepp) - 0.5 * chem.H2); 					double gamma
                    //= 1. + 1. / cv;
                    double gamma = getOptions().gamma;
                    //		PRINT( "%e\n", gamma);
                    float const factor = 1.0f - sphParticlesZ(i);
                    sphParticlesH(i) = chem.h * factor;
                    sphParticlesHe0(i) = chem.he * factor;
                    sphParticlesHp(i) = chem.hp * factor;
                    sphParticlesHn(i) = chem.hn * factor;
                    sphParticlesH2(i) = chem.h2 * factor;
                    sphParticlesHep(i) = chem.hep * factor;
                    sphParticlesHepp(i) = chem.hepp * factor;
                    sphParticlesNormalizeFracs(i);
                    sphParticlesTcool(i) = chem.tcool > 0.0 ? chem.tcool : 1e38;
#ifdef HOPKINS
                    double const e0 = sphParticlesEintRho(i);
                    double const rho = sphParticlesRhoRho(i);
#ifdef ENTROPY
                    sphParticlesRec2(i).entropy =
                        (gamma - 1.0) * chem.eint * pow(rho, (1.0 - gamma));
#else
                    sphParticlesRec2(i).eint = chem.eint;
#endif
                    double const thisChange = (sphParticlesEintRho(i) - e0) * sphMass / sqr(a);
#else
                    double const e0 = sphParticlesEint(i);
                    double const rho = sphParticlesRho(i);
#ifdef ENTROPY
                    sphParticlesRec2(i).entropy =
                        (gamma - 1.0) * chem.eint * pow(rho, (1.0 - gamma));
#else
                    sphParticlesRec2(i).eint = chem.eint;
#endif
                    double const thisChange = (sphParticlesEint(i) - e0) * sphMass / sqr(a);
#endif
                    //			PRINT( "%e\n", this_change );
                    echange += thisChange;
                    if (stars) {
                        ALWAYS_ASSERT(chem.coldMass >= 0.0);
                        //						sph_particles_cold_mass(i) = chem.cold_mass;
                    }
                }
            }
            Pair<double> rc;
            rc.first = echange;
            rc.second = flops;
            return rc;
        }));
    }
    double flops = 0.0;
    for (auto& f : futs) {
        auto tmp = f.get();
        echange += tmp.first;
        flops += tmp.second;
    }
    tm.stop();
    if (hpxRank() == 0) {
        PRINT("CHEM GFLOPS = %e eheat = %e\n", flops / 1024 / 1024 / 1024 / tm.read(), echange);
    }
    profilerExit();
    Pair<double> rc;
    rc.first = echange;
    rc.second = flops;
    return rc;
}
