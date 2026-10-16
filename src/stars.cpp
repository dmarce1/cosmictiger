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

#include <gsl/gsl_rng.h>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stars.hpp>

Vector<StarParticle> stars;

float starsSampleMass(gsl_rng*);
float starsLuminosity(float mass);
float starsLifetime(float mass);
float starsRemnantMass(float mass);
float starsHeliumProduced(float mass);

HPX_PLAIN_ACTION(starsFind, StarsFindAction);
HPX_PLAIN_ACTION(starsRemove, StarsRemoveAction);

void starsSave(FILE* fp) {
    size_t size = stars.size();
    fwrite(&size, sizeof(size_t), 1, fp);
    fwrite(stars.data(), sizeof(StarParticle), stars.size(), fp);
}

StarParticle& starsGet(ParticleInteger index) {
    return stars[index];
}

ParticleInteger starsSize() {
    return stars.size();
}

void starsLoad(FILE* fp) {
    size_t size;
    FREAD(&size, sizeof(size_t), 1, fp);
    stars.resize(size);
    FREAD(stars.data(), sizeof(StarParticle), size, fp);
}

double starsFind(float a, float dt, int minrung, int step, float t0) {
    profilerEnter("FUNCTION");

    PRINT("------------------------------------>>> Searching for STARS "
          "<<<--------------------------------------------------------\n");
    Vector<hpx::future<double>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<StarsFindAction>(c, a, dt, minrung, step, t0));
    }
    MutexType mutex;
    std::atomic<int> found(0);
    Vector<ParticleInteger> indices;
    int const numberOfThreads = hpxHardwareConcurrency();
    Vector<gsl_rng*> rndGens(numberOfThreads);
    for (int i = 0; i < numberOfThreads; i++) {
        rndGens[i] = gsl_rng_alloc(gsl_rng_taus);
        gsl_rng_set(rndGens[i], step * numberOfThreads + i);
    }
    double eloss = 0.0;

    static double const sphMass = getOptions().sphMass;
    static double const codeToG = getOptions().codeToG;
    static double const codeToCm = getOptions().codeToCm;
    static double const codeToS = getOptions().codeToS;
    static double const g = getOptions().gm;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(
            [proc, t0, numberOfThreads, a, minrung, &found, &mutex, &indices, dt, &rndGens]() {
                double eloss = 0.0;
                double const dmSoft = getOptions().hsoft;
                double const codeToS = getOptions().codeToS;
                double const codeToG = getOptions().codeToG;
                double const codeToCm = getOptions().codeToCm;
                double const rho0 = 3.0 / (8.0 * M_PI * constants::kG) *
                    pow(getOptions().hubble * constants::kH0, 2);
                double const rho0B = rho0 * getOptions().omegaB;
                double const rho0C = rho0 * getOptions().omegaC;
                double const codeToDensity = codeToG / pow(codeToCm, 3);
                double const codeToEnergy = pow(codeToCm / codeToS, 2);
                ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
                ParticleInteger const e =
                    (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
                for (ParticleInteger i = b; i < e; i++) {
                    if (!sphParticlesIsstar(i)) {
                        char rung = sphParticlesRung(i);
                        ParticleInteger const kk = sphParticlesDmIndex(i);
                        double const rhoTot = particlesRho(kk) * codeToDensity;
#ifdef HOPKINS
                        double const rhoB = sphParticlesRhoRho(i) * codeToDensity;
                        double const eint = sphParticlesEintRho(i) * codeToEnergy;
#else
                        double const rhoB = sphParticlesRho(i) * codeToDensity;
                        double const eint = sphParticlesEint(i) * codeToEnergy;
#endif
                        double const rhoC = rhoTot - rhoB;
                        double const tcool = sphParticlesTcool(i);
                        if (rhoB > 0.0f) {
                            double const tdyn =
                                sqrt((3.0 * M_PI * a * a * a) / (32.0 * constants::kG * rhoB));
                            double const pre = (getOptions().gamma - 1.0) * rhoB * eint;
                            double const cs = sqrt(pre / rhoB * getOptions().gamma);
                            double const deltaB = (rhoB - rho0B);
                            double const delta0 = (rhoTot - rho0);
                            double const dt = t0 * 0.5 * codeToS;
                            constexpr double kEta = 20.0;
                            if (deltaB / delta0 > 0.0) {
                                double const a = pow(constants::kG, -1.5);
                                double const b = pow(rhoB, -0.5);
                                double const c = pow(cs, 3);
                                double const d = pow(deltaB / delta0, 1.5);
                                double const mj = a * b * c * d;
                                double const m0 = getOptions().sphMass * codeToG;
                                if (rhoB > kEta * rho0B) {
                                    if (tcool < tdyn) {
                                        if (m0 > mj) {
                                            if (sphParticlesDivv(i) < 0.0) {
                                                double const eps = dt / tdyn;
                                                double const p = 1.0 - exp(-eps);
                                                bool const makeStar =
                                                    (gsl_rng_uniform(rndGens[proc]) < p);
                                                if (makeStar) {
                                                    sphParticlesIsstar(i) = true;
#ifdef HOPKINS
                                                    eloss += sphParticlesEintRho(i) *
                                                        getOptions().sphMass / (a * a);
#else
                                                    eloss += sphParticlesEint(i) *
                                                        getOptions().sphMass / (a * a);
#endif
#ifdef ENTROPY
                                                    sphParticlesEntr(i) = 0.0;
#else
                                                    sphParticlesEint(i) = 0.0;
#endif
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        //				PRINT( "%e\n", rho/rho0);
                    }
                }
                return eloss;
            }));
    }
    for (auto& f : futs) {
        eloss += f.get();
    }
    for (int i = 0; i < numberOfThreads; i++) {
        gsl_rng_free(rndGens[i]);
    }
    profilerExit();
    return eloss;
}

float starsRemnantMass(float mi, float z);

HPX_PLAIN_ACTION(starsStatistics, StarsStatisticsAction);

StarsStats starsStatistics(float a) {
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<StarsStats>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<StarsStatisticsAction>(c, a));
    }
    StarsStats stats;
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads]() {
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            StarsStats stats;
            for (ParticleInteger i = b; i < e; i++) {
                if (sphParticlesIsstar(i)) {
                    stats.stars++;
                }
            }
            return stats;
        }));
    }
    for (auto& f : futs) {
        stats += f.get();
    }
    if (hpxRank() == 0) {
        FILE* fp = fopen("stars.txt", "at");
        fprintf(fp, "%e %li \n", 1.f / a - 1.f, stats.stars);
        fclose(fp);
    }
    profilerExit();
    return stats;
}

#define WIND_RATIO 0.5

void starsRemove(float a, float dt, int minrung, int step) {
    //	profiler_enter(__FUNCTION__);

    return;
    /*
     vector<hpx::future<void>> futs;
     vector<hpx::future<void>> futs2;
     for (auto& c : hpx_children()) {
     futs.push_back(hpx::async<stars_find_action>(c, a, dt, minrung, step));
     }
     const int numberOfThreads = hpx_hardware_concurrency();
     vector<gsl_rng *> rnd_gens(numberOfThreads);
     for (int i = 0; i < numberOfThreads; i++) {
     rnd_gens[i] = gsl_rng_alloc(gsl_rng_taus);
     gsl_rng_set(rnd_gens[i], step * numberOfThreads + i);
     }
     static const double sph_mass = get_options().sph_mass;
     static const double code_to_g = get_options().code_to_g;
     static const double code_to_cm = get_options().code_to_cm;
     static const double code_to_s = get_options().code_to_s;
     vector<part_int> to_gas_indices;
     mutex_type to_gas_mutex;
     std::atomic<int> remnants(0);
     std::atomic<int> clouds(0);
     std::atomic<int> nstars(0);
     for (int proc = 0; proc < numberOfThreads; proc++) {
     futs2.push_back(hpx::async([&rnd_gens,proc, numberOfThreads, a, dt, &to_gas_mutex,
     &to_gas_indices, minrung, &remnants, &clouds,&nstars]() { const part_int b = (size_t) proc *
     stars.size() / numberOfThreads; const part_int e = (size_t) (proc+1) * stars.size() /
     numberOfThreads; for( part_int i = b; i < e; i++) { if( stars[i].type == STAR_TYPE ) { float
     real_dt = a * dt * code_to_s * constants::seconds_to_years; stars[i].time_remaining -= real_dt;
     float remnant_mass_ratio = stars_remnant_mass(stars[i].stellar_mass, stars[i].Z); float
     wind_rate = WIND_RATIO * (1.0 - remnant_mass_ratio); float dt_star = stars[i].total_life;
     dt_star /= constants::seconds_to_years; dt_star /= code_to_s; dt_star /= a; float pwind = 1.0 -
     exp(-wind_rate*dt/dt_star); if( gsl_rng_uniform_pos(rnd_gens[proc]) < pwind) {
     std::lock_guard<mutex_type> lock(to_gas_mutex);
     //		PRINT( "Stellar wind!\n");
     to_gas_indices.push_back(i);
     stars[i].remove = true;
     } else {
     if( stars[i].time_remaining < 0.0 && particles_rung(stars[i].dm_index) >= minrung ) {
     if( gsl_rng_uniform_pos(rnd_gens[proc]) > remnant_mass_ratio ) {
     std::lock_guard<mutex_type> lock(to_gas_mutex);
     to_gas_indices.push_back(i);
     stars[i].remove = true;
     } else {
     stars[i].type = REMNANT_TYPE;
     stars[i].zform = 1.0f / a - 1.f;
     stars[i].stellar_mass *= remnant_mass_ratio;
     }
     }
     }
     }
     if( stars[i].type == REMNANT_TYPE ) {
     remnants++;
     } else if( stars[i].type == CLOUD_TYPE) {
     clouds++;
     } else if( stars[i].type == STAR_TYPE) {
     nstars++;
     }
     }
     }));
     }
     hpx::wait_all(futs2.begin(), futs2.end());
     static const auto h0 = 1.0 / get_options().parts_dim;
     PRINT("Restoring gas\n");
     for (auto& i : to_gas_indices) {
     star_particle star = stars[i];
     const int j = star.dm_index;
     const int k = sph_particles_size();
     sph_particles_resize(k + 1, false);
     sph_particles_dm_index(k) = j;
     particles_type(j) = SPH_TYPE;
     particles_cat_index(j) = k;
     const double T = 10000.0;
     const double N = sph_mass * code_to_g * ((1. - star.Y) * 2.f + star.Y * .25f * 3.f + 0.5f *
     star.Z) * constants::avo; const double Cv = 1.5 * constants::kb; double E = Cv * N * T;
     constexpr float kFSn = 6e-4;
     //		const double fSN = 0.0e-5;
     E /= sqr(code_to_cm) * code_to_g / sqr(code_to_s);
     float wind_energy = 0.0;
     if (star.stellar_mass > 7.5) {
     //		PRINT("Supernova!\n");
     //		E += 0.5 * kFSn * sph_mass / star.stellar_mass * a * a;
     //	wind_energy += 0.5 * kFSn * sph_mass / star.stellar_mass * a * a;
     double dZ = 0.02;
     double dHe = 0.20 * (1.0 - star.Y - star.Z);
     star.Y += dHe;
     star.Z += dZ;
     } else if (star.time_remaining > 0.0) {
     //		wind_energy = 0.5 * sqr(1e-3) * a * a;
     }
     if (wind_energy > 0.0) {
     const int n = sph_particles_dm_index(k);
     auto& vx = particles_vel(XDIM, n);
     auto& vy = particles_vel(YDIM, n);
     auto& vz = particles_vel(ZDIM, n);
     float nx = gsl_rng_uniform_pos(rnd_gens[0]);
     float ny = gsl_rng_uniform_pos(rnd_gens[0]);
     float nz = gsl_rng_uniform_pos(rnd_gens[0]);
     float ninv = 1.0 / sqrt(sqr(nx, ny, nz));
     nx *= ninv;
     ny *= ninv;
     nz *= ninv;
     float vdotn = vx * nx + vy * ny + vz * nz;
     float dv = sqrt(sqr(vdotn) + 2.0 * wind_energy / sph_mass) - vdotn;
     float dvx = nx * dv;
     float dvy = ny * dv;
     float dvz = nz * dv;
     particles_vel(XDIM, n) += dvx;
     particles_vel(YDIM, n) += dvy;
     particles_vel(ZDIM, n) += dvz;
     }
     sph_particles_H2(k) = 0.f;
     sph_particles_Hp(k) = 1.f - star.Y - star.Z;
     sph_particles_He0(k) = 0.f;
     sph_particles_Hep(k) = 0.f;
     sph_particles_Hn(k) = 0.f;
     sph_particles_Hepp(k) = star.Y;
     sph_particles_Z(k) = star.Z;
     sph_particles_dent_con(k) = 0.f;
     sph_particles_dent_pred(k) = 0.f;
     sph_particles_smooth_len(k) = h0;
     sph_particles_tdyn(k) = 1e38;
     E *= a * a;
     sph_particles_ent(k) = -E;
     for (int dim = 0; dim < NDIM; dim++) {
     sph_particles_dvel_pred(dim, k) = 0.f;
     sph_particles_dvel_con(dim, k) = 0.f;
     }
     }
     for (auto& i : to_gas_indices) {
     while (i < stars.size() && stars[i].remove) {
     const int j = stars.size() - 1;
     stars[i] = stars.back();
     if (!stars[i].remove) {
     particles_cat_index(stars[i].dm_index) = i;
     }
     stars.pop_back();
     }
     }
     hpx::wait_all(futs.begin(), futs.end());
     PRINT("%i stars, %i remnants, %i clouds\n", (int ) nstars, (int ) remnants, (int ) clouds);
     for (int i = 0; i < numberOfThreads; i++) {
     gsl_rng_free(rnd_gens[i]);
     }
     //	profiler_exit();*/
}

float starsSampleMass(gsl_rng* rndgen) {
    float y = gsl_rng_uniform_pos(rndgen);
    constexpr float kM1 = 0.08;
    constexpr float kM2 = 0.5;
    constexpr float kAlpha1 = -.7f;
    constexpr float kAlpha2 = .3f;
    constexpr float kAlpha3 = 1.3f;
    float mass;
    if (y < 0.0360115f) {
        float const y0 = 0.f;
        float const y1 = powf(kM1, 1.f - kAlpha1);
        y = y0 + (y1 - y0) * gsl_rng_uniform_pos(rndgen);
        mass = powf(y, 1.0f / (1.f - kAlpha1));
    } else if (y < 0.0360115f + 0.227977f) {
        float const y0 = powf(kM1, 1.f - kAlpha2);
        float const y1 = powf(kM2, 1.f - kAlpha2);
        y = y0 + (y1 - y0) * gsl_rng_uniform_pos(rndgen);
        mass = powf(y, 1.0f / (1.f - kAlpha2));
    } else {
        float const y0 = 0.f;
        float const y1 = powf(kM2, 1.f - kAlpha3);
        y = y0 + (y1 - y0) * gsl_rng_uniform_pos(rndgen);
        mass = powf(y, 1.0f / (1.f - kAlpha3));
    }
    if (mass > 265.f) {
        return starsSampleMass(rndgen);
    }
    return mass;
}

float starsLuminosity(float mass) {
    constexpr float kAlpha1 = .238175;
    constexpr float kAlpha2 = 1.f;
    constexpr float kAlpha3 = sqrtf(2);
    constexpr float kAlpha4 = 31726.5f;
    constexpr float kBeta1 = 2.3f;
    constexpr float kBeta2 = 4.f;
    constexpr float kBeta3 = 3.5;
    constexpr float kBeta4 = 1.f;
    if (mass < .43) {
        return kAlpha1 * powf(mass, kBeta1);
    } else if (mass < 2) {
        return kAlpha2 * powf(mass, kBeta2);
    } else if (mass < 55) {
        return kAlpha3 * powf(mass, kBeta3);
    } else {
        return kAlpha4 * powf(mass, kBeta4);
    }
}

float starsLifetime(float mass) {
    float l = mass / starsLuminosity(mass);
    l *= 12e9;
    // PRINT( "Stellar lifetime = %e B yr \n", l/1e9);
    return l;
}

float starsHeliumProduced(float mass) {
    double const l0 = 3.839e33;
    double const m0Value = 1.989e33;
    double const lValue = starsLuminosity(mass) * l0;
    //	PRINT( "%e\n", L);
    double const t = starsLifetime(mass) / constants::kSecondsToYears;
    //	PRINT( "%e\n", t);
    double const eValue = lValue * t;
    //	PRINT( "%e\n", E);
    double const mHe = eValue / sqr(constants::kC) / m0Value / 0.007;
    //	PRINT( "%e\n", MHe);
    return mHe / mass;
}

/*
 float stars_remnant_mass(float m) {
 float mf;
 //Cummings et al 2018
 if (m < 0.85) {
 mf = std::min(m, 0.080 * m + 0.489);
 } else if (m < 3.60) {
 mf = 0.187 * m + .184;
 } else if (m < 7.5) {
 mf = 0.107 * x + 0.471 * m;
 }
 //Fryer et al 2012
 else if( m < )
 }
 */

float mremFracBig(float mValue, float z) {
    // Spera et al 2015
    auto const mcoCoeffs = [](float& b1, float& k1, float& k2, float& delta1, float& delta2,
                               float z) {
        if (z > 4e-3) {
            b1 = 59.63 - 2.969e3 * z + 4.988e4 * sqr(z);
            k1 = 45.04 - 2.176e3 * z + 3.806e4 * sqr(z);
            k2 = 1.389e2 - 4.664e3 * z + 5.106e4 * sqr(z);
            delta1 = 2.790e-2 - 1.780e-2 * z + 77.05 * sqr(z);
            delta2 = 6.730e-3 + 2.690 * z - 52.39 * sqr(z);
        } else if (z > 1e-3) {
            b1 = 40.98 + 3.415e4 * z - 8.064e6 * sqr(z);
            k1 = 35.17 + 1.548e4 * z - 3.759e6 * sqr(z);
            k2 = 20.36 + 1.162e5 * z - 2.276e7 * sqr(z);
            delta1 = 2.500e-2 - 4.346 * z + 1.340e3 * sqr(z);
            delta2 = 1.750e-2 + 11.39 * z - 2.902 * sqr(z);
        } else {
            b1 = 67.07;
            k1 = 46.89;
            k2 = 1.138e2;
            delta1 = 2.199e-2;
            delta2 = 2.602e-2;
        }
    };

    auto const gmxy = [](float mValue, float x, float y) {
        return 0.5 / (1.0 + pow(10, (x - mValue) * y));
    };

    auto const mcoValue = [mcoCoeffs, gmxy](float mValue, float z) {
        float b1, k1, k2, delta1, delta2;
        mcoCoeffs(b1, k1, k2, delta1, delta2, z);
        return -2.0 + (b1 + 2.0) * (gmxy(mValue, k1, delta1) + gmxy(mValue, k2, delta2));
    };

    float mco = mcoValue(mValue, z);
    float mrem;
    if (z < 5e-4) {
        float mZ = -6.476e2 * z + 1.911;
        float qZ = 2.300e3 * z + 11.67;
        float fmcoz = mZ * mco + qZ;
        float pmco = -2.333 + 0.1559 * mco + 0.2700 * sqr(mco);
        if (mco < 5) {
            mrem = std::max(pmco, 1.27f);
        } else if (mco < 10.0) {
            mrem = pmco;
        } else {
            mrem = std::min(pmco, fmcoz);
        }
    } else {
        float a1, a2, lValue, eta, m, q;
        if (z > 1e-3) {
            a1 = 1.340 - 29.46 / (1.0 + pow(z / 1.110e-3, 2.361));
            a2 = 80.22 - 74.73 * pow(z, 0.965) / (2.72e-3 + pow(z, 0.965));
            lValue = 5.683 + 3.533 / (1.0 + pow(z / 7.430e-3, 1.993));
            eta = 1.066 - 1.121 / (1.0 + pow(z / 2.558e-2, 0.609));
        } else {
            a1 = 1.150e5 * z - 1.258e2;
            a2 = 91.56 - 1.957e4 * z - 1.558e7 * sqr(z);
            lValue = 1.134e4 * z - 2.143;
            eta = 3.090e-2 - 22.30 * z + 7.363e4 * sqr(z);
        }
        if (z > 2e-3) {
            m = 1.127;
            q = 1.061;
        } else if (z > 1e-3) {
            m = -43.82 * z + 1.304;
            q = -1.296e4 * z + 26.98;
        } else {
            m = -6.476e2 * z + 1.911;
            q = 2.300e3 * z + 11.67;
        }
        float hmcoz = a1 + (a2 - a1) / (1.0 + pow(10, (lValue - mco) * eta));
        float fmcoz = m * mco + q;
        if (mco < 5) {
            mrem = std::max(hmcoz, 1.27f);
        } else if (mco < 10) {
            mrem = hmcoz;
        } else {
            mrem = std::max(hmcoz, fmcoz);
        }
    }
    return std::min(std::max(mrem / mValue, 0.0f), 1.0f);
}
float starsRemnantMass(float mi, float z) {
    float mf;
    if (mi < 2.85) {    // Cummings 2018
        mf = std::min(0.080f * mi + 0.489f, mi);
    } else if (mi < 3.60) {    // Cummings 2018
        mf = 0.187 * mi + 0.184;
    } else if (mi < 8.0) {    // Cummings 2018
        mf = 0.107 * mi + 0.471;
    } else {
        mf = mi * mremFracBig(mi, z);
    }
    return mf / mi;
}
void starsTestMass() {
    constexpr int kN = 200000;
    Vector<float> m;
    float dm = 0.01;
    Vector<float> cnt(10000, 0.0);
    for (int i = 0; i < kN; i++) {
        static auto* rndgen = gsl_rng_alloc(gsl_rng_taus);
        m.push_back(starsSampleMass(rndgen));
        //	PRINT( "%e\n", m[i]);
        int const j = m[i] / dm;
        if (j < 10000 && j >= 0) {
            cnt[j] += 1.0;
        }
    }
    FILE* fp = fopen("mass.txt", "wt");
    for (int i = 0; i < 10000; i++) {
        float const m = (i + 0.5) * dm;
        fprintf(fp, "%e %e %e %e %e %e \n", m, cnt[i], starsLifetime(m), starsRemnantMass(m, 0.0),
            starsRemnantMass(m, 0.002), starsRemnantMass(m, 0.02));
        //		printf( "%e %e\n", m, cnt[i]);
    }
    fclose(fp);
}
