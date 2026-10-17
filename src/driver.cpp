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

#define SMOOTHLEN_BUFFER 0.21
#define SCALE_DT 0.01

#include <array>
#include <cstdint>
#include <string>

#include <dirent.h>
#include <sys/types.h>

#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/driver.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/group_tree.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/groups_find.hpp>
#include <cosmictiger/hydro.hpp>
#include <cosmictiger/initialize.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/output.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>
#include <cosmictiger/profiler.hpp>
#include <cosmictiger/simulation_units.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/stars.hpp>
#include <cosmictiger/time.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/view.hpp>

HPX_PLAIN_ACTION(writeCheckpoint, WriteCheckpointAction);
HPX_PLAIN_ACTION(readCheckpoint, ReadCheckpointAction);

double domainTime = 0.0;
double sortTime = 0.0;
double kickTime = 0.0;
double driftTime = 0.0;
double flopsPerNode = 1e6;
double flopsPerParticle = 1e5;
bool usedGpu;

struct Timing {
    double numberOfParticles;
    double time;
    Timing() {
        numberOfParticles = time = 0.0;
    }
};

#define MIN_BUCKET 64
#define MAX_BUCKET 192

Vector<Timing> timings;

double nfwDensity(double r) {
    double rmax = 10.0;
    return 1.0 / r / sqr(1.0 + r) / (4.0 * M_PI * (-1.0 + 1.0 / (1.0 + rmax) + log(1.0 + rmax)));
}

double sampleDensityDistribution(std::function<double(double)> const& f) {
    double target = rand1();
    double menc = 0.0;
    double r = 0.0;
    double dr = 0.001;
    double iValue, i0;
    iValue = 0.0;
    while (menc < target) {
        menc += 0.5 * iValue * dr;
        i0 = 0.5 * iValue;
        r += dr;
        iValue = f(r) * 4.0 * M_PI * sqr(r);
        i0 += 0.5 * iValue;
        menc += 0.5 * iValue * dr;
    }
    double dr2 = (menc - target) / i0;
    r -= dr2;
    return r;
}
/*
 void distribution_init(const std::function<float(float, float, float)>& pdf) {
 const int N = get_options().Nfour;
 range<int64_t> box;
 for (int dim = 0; dim < NDIM; dim++) {
 box.begin[dim] = 0;
 box.end[dim] = N;
 }
 size_t sz = box.volume();
 vector<float> rho_x(sz);
 array<int64_t, NDIM> I;
 array<float, NDIM> x;
 for (I[0] = 0; I[0] < N; I[0]++) {
 for (I[1] = 0; I[1] < N; I[1]++) {
 for (I[2] = 0; I[2] < N; I[2]++) {
 for (int dim = 0; dim < NDIM; dim++) {
 x[dim] = (float) I[dim] / N;
 }
 rho_x[box.index(I)] = pdf(x[XDIM], x[YDIM], x[ZDIM]);
 }
 }
 }
 float rho0 = 0.0;
 for (I[0] = 0; I[0] < N; I[0]++) {
 for (I[1] = 0; I[1] < N; I[1]++) {
 for (I[2] = 0; I[2] < N; I[2]++) {
 rho0 += rho_x[box.index(I)] / sz;
 }
 }
 }
 for (I[0] = 0; I[0] < N; I[0]++) {
 for (I[1] = 0; I[1] < N; I[1]++) {
 for (I[2] = 0; I[2] < N; I[2]++) {
 rho_x[box.index(I)] = (rho_x[box.index(I)] - rho0) / rho0;
 }
 }
 }
 fft3d_init(N);
 fft3d_accumulate_real(box, rho_x);
 fft3d_execute();
 array<float, NDIM> k;
 const auto fbox = fft3d_complex_range();
 auto rho_k = fft3d_read_complex(fbox);
 particles_resize(N * N * N);
 for (int i = 0; i < particles_size(); i++) {
 particles_rung(i) = 0;
 }
 for (int dim1 = 0; dim1 < NDIM; dim1++) {
 auto dphi_dx_k = rho_k;
 for (I[0] = fbox.begin[XDIM]; I[0] < fbox.end[XDIM]; I[0]++) {
 for (I[1] = fbox.begin[YDIM]; I[1] < fbox.end[YDIM]; I[1]++) {
 for (I[2] = fbox.begin[ZDIM]; I[2] < fbox.end[ZDIM]; I[2]++) {
 float k2 = 0.0f;
 for (int dim2 = 0; dim2 < NDIM; dim2++) {
 k[dim2] = 2.0 * M_PI * (I[dim2] < N / 2 ? I[dim2] : I[dim2] - N) / N;
 k2 += sqr(k[dim2]);
 }
 const auto iii = fbox.index(I);
 if (k2 > 0.0) {
 dphi_dx_k[iii] = dphi_dx_k[iii] * cmplx(0.0, 1.0) * (k[dim1] / k2);
 } else {
 dphi_dx_k[iii] = 0.0f;
 }
 }
 }
 }
 fft3d_destroy();

 fft3d_init(N);
 fft3d_accumulate_complex(fbox, dphi_dx_k);
 fft3d_inv_execute();

 auto dphi_dx_x = fft3d_read_real(box);
 for (I[0] = box.begin[XDIM]; I[0] < box.end[XDIM]; I[0]++) {
 for (I[1] = box.begin[YDIM]; I[1] < box.end[YDIM]; I[1]++) {
 for (I[2] = box.begin[ZDIM]; I[2] < box.end[ZDIM]; I[2]++) {
 const int iii = box.index(I);
 const float dx = dphi_dx_x[iii];
 particles_pos(dim1, iii) = ((float) I[dim1] / N + dx / N);
 particles_vel(dim1, iii) = 0.0;

 }
 }
 }

 }
 fft3d_destroy();

 }
 */
void plummerInit(double r0) {
    auto const numberOfParticles = getOptions().numberOfParticles;
    particlesResize(numberOfParticles);
    auto const pdf = [](double x) {
        if (x > 0.0) {
            return sqr(x) * pow(1.0 + sqr(x), -2.5);
        } else {
            return pow(1.0 + sqr(1.0 / x), -2.5) / sqr(sqr(x));
        }
    };

    /*	distribution_init([](float x, float y, float z) {
     float r = sqrt(sqr(x - 0.51, y - 0.51, z - 0.51));
     r *= 4.0;
     return 1.0 / r / sqr(1+r);
     return 1000*pow(1.0+sqr(r),-2.5);
     });
     return;*/

    double x0 = 0.5;
    double y0 = 0.5;
    double z0 = 0.5;

    for (ParticleInteger i = 0; i < numberOfParticles; i += 2) {
        double r, p;
        //		r = sample_density_distribution(nfw_density);
        r = sampleDensityDistribution(
            [](double r) { return (3.0 / (4.0 * M_PI)) * pow(1.0 + sqr(r), -2.5); });
        r *= getOptions().plummerR;
        double nx, ny, nz, n2;
        do {
            nx = 2.0 * rand1() - 1.0;
            ny = 2.0 * rand1() - 1.0;
            nz = 2.0 * rand1() - 1.0;
            n2 = sqr(nx, ny, nz);
        } while (n2 == 0.0 || n2 > 1.0);
        double ninv = 1.0 / sqrt(n2);
        nx *= ninv;
        ny *= ninv;
        nz *= ninv;
        double x = x0 + r * nx;
        double y = y0 + r * ny;
        double z = z0 + r * nz;
        // double e = abs();
        //	double v = sqrt(-2.0 * log(rand1())) * abs(cos(2.0 * M_PI * rand1()));
        double v = sqrt(
            getOptions().gm * numberOfParticles / sqrt(sqr(getOptions().plummerR) + sqr(r)) / 2);
        double v0 = sqrt(getOptions().gm * numberOfParticles * 0.25);
        do {
            nx = 2.0 * rand1() - 1.0;
            ny = 2.0 * rand1() - 1.0;
            nz = 2.0 * rand1() - 1.0;
            n2 = sqr(nx, ny, nz);
        } while (n2 == 0.0 || n2 > 1.0);
        ninv = 1.0 / sqrt(n2);
        nx *= ninv;
        ny *= ninv;
        nz *= ninv;
        double vx = v * nx;
        double vy = v * ny;
        double vz = v * nz;
        particlesPos(XDIM, i) = x;
        particlesPos(YDIM, i) = y;
        particlesPos(ZDIM, i) = z;
        particlesVel(XDIM, i) = vx;
        particlesVel(YDIM, i) = vy;
        particlesVel(ZDIM, i) = vz;
        particlesPos(XDIM, i + 1) = 2.0 * x0 - x;
        particlesPos(YDIM, i + 1) = 2.0 * y0 - y;
        particlesPos(ZDIM, i + 1) = 2.0 * z0 - z;
        particlesVel(XDIM, i + 1) = -vx;
        particlesVel(YDIM, i + 1) = -vy;
        particlesVel(ZDIM, i + 1) = -vz;
        particlesRung(i + 1) = 0;
        particlesRung(i) = 0;
        /*if (i % 4 == 0) {
            particles_pos(XDIM, i) += 0.025;
            particles_vel(XDIM, i) -= 50.0 * v0;
            particles_pos(XDIM, i + 1) += 0.025;
            particles_vel(XDIM, i + 1) -= 50.0 * v0;
        } else {
            particles_pos(XDIM, i) -= 0.025;
            particles_vel(XDIM, i) += 50.0 * v0;
            particles_pos(XDIM, i + 1) -= 0.025;
            particles_vel(XDIM, i + 1) += 50.0 * v0;
        }*/
    }
}

Vector<double> readCheckpointList() {
    Vector<double> chkpts;
    FILE* fp = fopen("checkpoint.txt", "rt");
    if (fp != nullptr) {
        float z;
        PRINT("Checkpoints will be written at z = \n");
        while (fscanf(fp, "%f\n", &z) == 1) {
            PRINT("%f\n", z);
            chkpts.push_back(z);
        }

        fclose(fp);
    }
    return chkpts;
}

static void flushTimings(bool reset) {
    auto const theta = getOptions().theta;
    char* fname;
    asprintf(&fname, "buckets.%.2f.txt", theta);
    FILE* fp = fopen(fname, "wt");
    free(fname);
    for (int i = 0; i < timings.size(); i++) {
        if (timings[i].numberOfParticles > 0.0) {
            fprintf(fp, "%i %e\n", i, timings[i].numberOfParticles / timings[i].time);
        }
    }
    if (reset) {
        timings = decltype(timings)();
    }
    fclose(fp);
}

void doGroups(int number, double scale) {
    profilerEnter(__FUNCTION__);

    Timer total;
    total.start();
    PRINT("Doing groups\n");
    Timer tm;
    tm.start();
    groupTreeCreate();
    tm.stop();
    PRINT("group_tree_create = %e\n", tm.read());
    tm.reset();
    tm.start();
    particlesGroupsInit();
    tm.stop();
    PRINT("particles_group_init = %e\n", tm.read());
    tm.reset();
    size_t active;
    int iter = 1;
    do {
        TreeId rootId;
        rootId.proc = 0;
        rootId.index = 0;
        Vector<TreeId> checklist;
        checklist.push_back(rootId);
        tm.start();
        active = groupsFind(rootId, std::move(checklist), getOptions().linkingLength).get();
        tm.stop();
        PRINT("%i groups_find = %e active = %li\n", iter, tm.read(), active);
        tm.reset();
        particlesIncGroupCacheEpoch();
        groupTreeIncCacheEpoch();
        iter++;
    } while (active > 0);
    tm.start();
    groupTreeDestroy();
    tm.stop();
    PRINT("tree cleanup = %e\n", tm.read());
    Timer reduce;
    reduce.start();
    for (int wave = 0; wave < GROUP_WAVES; wave++) {
        tm.start();
        groupsAddParticles(wave, scale);
        tm.stop();
        PRINT("%i groups_add_particles %e\n", wave, tm.read());
        tm.reset();
        tm.start();
        groupsReduce(scale);
        tm.stop();
        PRINT("groups_reduce %e\n", tm.read());
        tm.reset();
    }
    reduce.stop();
    tm.start();
    groupsCull();
    tm.stop();
    PRINT("groups_cull %e\n", tm.read());
    tm.reset();
    tm.start();
    auto ngroups = groupsSave(
        number, scale, cosmosTime(1.0e-6 * scale, scale) * getOptions().codeToS / constants::kSpyr);
    tm.stop();
    PRINT("groups_save %e\n", tm.read());
    total.stop();
    PRINT("Reduction time = %e\n", reduce.read());
    PRINT("Total time = %e\n", total.read());
    PRINT("Group count = %li of %li candidates\n", ngroups.first, ngroups.second);
    particlesGroupsDestroy();
    profilerExit();
}

SphTreeCreateReturn sphStep1(int minrung, double scale, double tau, double t0, int phase,
    double adot, int maxRung, int iter, double dt, double* eheat, bool verbose) {
    bool const stars = getOptions().stars;
    bool const diff = getOptions().diffusion;
    bool const chem = getOptions().chem;
    verbose = true;
    *eheat = 0.0;
    if (verbose) {
        PRINT("Doing SPH step with minrung = %i\n", minrung);
    }
    SphTreeCreateParameters tparams;

    Timer tm;
    Timer totalTm;
    totalTm.start();
    tparams.minRung = minrung;
    tparams.hWt = (1.0 + SMOOTHLEN_BUFFER);
    TreeId rootId;
    rootId.proc = 0;
    rootId.index = 0;
    SphTreeCreateReturn sr;
    Vector<TreeId> checklist;
    checklist.push_back(rootId);
    SphTreeNeighborParameters tnparams;

    tnparams.hWt = (1.0 + SMOOTHLEN_BUFFER);
    tnparams.minRung = minrung;

    tm.start();
    if (verbose) {
        PRINT("starting sph_tree_create = %e\n", tm.read());
    }
    profilerEnter("sph_tree_create");
    sr = sphTreeCreate(tparams);
    profilerExit();
    tm.stop();
    if (verbose) {
        PRINT("sph_tree_create time = %e %i\n", tm.read(), sr.numberOfActiveParticles);
    }
    tm.reset();
    return sr;
}

SphRunReturn sphStep2(int minrung, double scale, double tau, double t0, int phase, double adot,
    int maxRung, int iter, double dt, EnergiesT* energies, bool verbose) {
    bool const stars = getOptions().stars;
    bool const diff = getOptions().diffusion;
    bool const chem = getOptions().chem;
    bool const conduction = getOptions().conduction;
    bool const vsoft = getOptions().vsoft;
    float dtinvCfl;
    float dtinvVisc;
    float dtinvDiff;
    float dtinvCond;
    float dtinvAcc;
    float dtinvDivv;
    float dtinvOmega;
    verbose = true;
    double flops;
    if (verbose) {
        PRINT("Doing SPH step with minrung = %i\n", minrung);
    }
    sphParticlesApplyUpdates(minrung, 0, t0, tau);

    SphRunParameters sparams;
    if (adot != 0.0) {
        sparams.maxDt = SCALE_DT * scale / fabs(adot);
    }
    sparams.adot = adot;
    sparams.tau = tau;
    sparams.tzero = tau == 0.0;
    sparams.maxRung = maxRung;
    sparams.a = scale;
    sparams.t0 = t0;
    sparams.minRung = minrung;
    bool cont;
    SphRunReturn kr;
    sparams.set = SPH_SET_ACTIVE;
    sparams.phase = phase;
    bool const glass = getOptions().glass;
    SphTreeNeighborParameters tnparams;
    tnparams.hWt = (1.0 + SMOOTHLEN_BUFFER);
    tnparams.minRung = minrung;
    TreeId rootId;
    rootId.proc = 0;
    rootId.index = 0;
    Vector<TreeId> checklist;
    checklist.push_back(rootId);

    profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
    tnparams.seti = SPH_INTERACTIONS_I;
    tnparams.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
    sphTreeNeighbor(tnparams, rootId, checklist).get();
    profilerExit();

    sparams.phase = 0;
    int densityIterations = 0;
    if (!vsoft) {
        sphParticlesResetConverged();
        do {
            if (++densityIterations > 256) {
                THROW_ERROR("SPH smoothing lengths did not converge after 256 tree rebuilds\n");
            }
            sparams.set = SPH_SET_ACTIVE;
            sparams.runType = SPH_RUN_PREHYDRO1;
            Timer tm;
            tm.start();
            kr = sphRun(sparams, true);
            tm.stop();
            if (verbose) {
                PRINT("sph_run(SPH_RUN_PREHYDRO1): tm = %e min_h = %e max_h = %e\n", tm.read(),
                    kr.hmin, kr.hmax);
            }
            tm.reset();
            cont = kr.rc;
            tnparams.hWt = (1.0 + SMOOTHLEN_BUFFER);
            tnparams.runType = SPH_TREE_NEIGHBOR_BOXES;
            tnparams.seto = cont ? SPH_SET_ACTIVE : SPH_SET_ALL;
            tnparams.seti = cont ? SPH_SET_ALL : SPH_SET_ALL;
            //			tnparams.set = SPH_SET_ACTIVE;
            tm.start();
            profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
            sphTreeNeighbor(tnparams, rootId, Vector<TreeId>()).get();
            profilerExit();
            tm.stop();
            PRINT("%e\n", tm.read());
            tm.reset();
            tm.start();
            tnparams.seti = cont ? SPH_INTERACTIONS_I : SPH_INTERACTIONS_IJ;
            tnparams.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
            profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
            sphTreeNeighbor(tnparams, rootId, checklist).get();
            profilerExit();
            tm.stop();
            PRINT("%e\n", tm.read());
            tm.reset();

            kr = SphRunReturn();
        } while (cont);
    } else {
        sphParticlesSoftlens2smoothlens(minrung);

        Timer tm;
        tnparams.hWt = 1.01;
        tnparams.runType = SPH_TREE_NEIGHBOR_BOXES;
        tnparams.seto = SPH_SET_ALL;
        tnparams.seti = SPH_SET_ALL;
        //			tnparams.set = SPH_SET_ACTIVE;
        tm.start();
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
        sphTreeNeighbor(tnparams, rootId, Vector<TreeId>()).get();
        profilerExit();
        tm.stop();
        PRINT("%e\n", tm.read());
        tm.reset();
        tm.start();
        tnparams.seti = SPH_INTERACTIONS_IJ;
        tnparams.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
        sphTreeNeighbor(tnparams, rootId, checklist).get();
        profilerExit();
        tm.stop();
        PRINT("%e\n", tm.read());
        tm.reset();
    }
    sphParticlesResetConverged();
    densityIterations = 0;
    do {
        if (++densityIterations > 256) {
            THROW_ERROR("SPH neighbor densities did not converge after 256 tree rebuilds\n");
        }
        sparams.set = SPH_SET_ACTIVE;
        sparams.runType = SPH_RUN_PREHYDRO2;
        Timer tm;
        tm.start();
        kr = sphRun(sparams, true);
        tm.stop();
        if (verbose) {
            PRINT("sph_run(SPH_RUN_PREHYDRO2): tm = %e min_h = %e max_h = %e\n", tm.read(), kr.hmin,
                kr.hmax);
        }
        tm.reset();
        if (vsoft) {
            break;
        }
        cont = kr.rc;
        tnparams.hWt = cont ? (1.0 + SMOOTHLEN_BUFFER) : 1.01;
        tnparams.runType = SPH_TREE_NEIGHBOR_BOXES;
        tnparams.seto = SPH_SET_ALL;
        tnparams.seti = SPH_SET_ALL;
        //			tnparams.set = SPH_SET_ACTIVE;
        tm.start();
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
        sphTreeNeighbor(tnparams, rootId, Vector<TreeId>()).get();
        profilerExit();
        tm.stop();
        PRINT("%e\n", tm.read());
        tm.reset();
        tm.start();
        tnparams.seti = SPH_INTERACTIONS_IJ;
        tnparams.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
        sphTreeNeighbor(tnparams, rootId, checklist).get();
        profilerExit();
        tm.stop();
        PRINT("%e\n", tm.read());
        tm.reset();
        kr = SphRunReturn();
    } while (cont);
    Timer tm;

    sparams.phase = 0;
    sparams.runType = SPH_RUN_HYDRO;
    tm.reset();
    tm.start();
    kr = sphRun(sparams, true);
    tm.stop();
    maxRung = 0;
    if (verbose) {
        PRINT("sph_run(SPH_RUN_HYDRO): tm = %e\n", tm.read());
    }
    tm.reset();

    sphParticlesApplyUpdates(minrung, 1, t0, tau);

    if (chem && tau > 0.0) {
        PRINT("Doing chemistry step\n");
        Timer tm;
        tm.start();
        energies->heating -= chemistryDoStep(scale, minrung, t0, cosmosDadt(scale), -1).first;
        tm.stop();
        PRINT("Took %e s\n", tm.read());
    }

    auto eValue = particlesSumEnergies();
    energies->kin = eValue.kin / sqr(scale);
    energies->xmom = eValue.xmom / scale;
    energies->ymom = eValue.ymom / scale;
    energies->zmom = eValue.zmom / scale;
    energies->nmom = eValue.nmom / scale;
    if (minrung == 0) {
        energies->pot = eValue.pot / scale;
    }
    energies->therm = eValue.therm / sqr(scale);

#ifdef IMPLICIT_CONDUCTION
    if (conduction) {
        tnparams.hWt = 1.01;
        tnparams.runType = SPH_TREE_NEIGHBOR_BOXES;
        tnparams.seti = SPH_SET_ALL;
        tnparams.seto = SPH_SET_ALL;
        tm.start();
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
        sphTreeNeighbor(tnparams, rootId, Vector<TreeId>()).get();
        profilerExit();
        tm.stop();
        tm.reset();
        tm.start();
        tnparams.seti = SPH_INTERACTIONS_IJ;
        tnparams.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
        profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
        sphTreeNeighbor(tnparams, rootId, checklist).get();
        profilerExit();
        tm.stop();
        tm.reset();

        Timer dtm;
        dtm.start();

        sphParticlesResetConverged();

        sparams.runType = SPH_RUN_COND_INIT;
        tm.reset();
        tm.start();
        sphRun(sparams, true);
        tm.stop();
        if (verbose) {
            PRINT("sph_run(SPH_RUN_COND_INIT): tm = %e \n", tm.read());
        }
        tm.reset();
        CondUpdateReturn err;
        do {
            sparams.runType = SPH_RUN_CONDUCTION;
            tm.reset();
            tm.start();
            sphRun(sparams, true);
            tm.stop();
            err = sphApplyConductionUpdate(minrung);
            if (verbose) {
                PRINT("sph_run(SPH_RUN_CONDUCTION): tm = %e err_max = %e err_rms = %e\n", tm.read(),
                    err.errMax, err.errRms);
            }
            tm.reset();
        } while (err.errMax > SPH_DIFFUSION_TOLER1 || err.errRms > SPH_DIFFUSION_TOLER2);
        dtm.stop();
        PRINT("Conduction took %e seconds total\n", dtm.read());
    }
#endif
    bool foundStars = false;
    if (stars && minrung <= 1) {
        //	sph_particles_entropy_to_energy();
        double eloss = 0.0;
        if (eloss = starsFind(scale, dt, minrung, iter, t0)) {
            energies->heating += eloss;
        }
        PRINT("%e----------------------------------------------------------------------------------"
              "-------------------------------------\n",
            eloss);
        starsStatistics(scale);
        foundStars = true;
    }

#ifdef HOPKINS
    bool rerun2 = true;
#else
    bool rerun2 = foundStars;
#endif
    if (rerun2) {
        sparams.phase = 0;
        int doneiters = 0;
        sphParticlesResetConverged();
        do {
            sparams.set = SPH_SET_ACTIVE;
            sparams.runType = SPH_RUN_PREHYDRO2;
            Timer tm;
            tm.start();
            kr = sphRun(sparams, true);
            tm.stop();
            if (verbose) {
                PRINT("sph_run(SPH_RUN_PREHYDRO2): tm = %e min_h = %e max_h = %e\n", tm.read(),
                    kr.hmin, kr.hmax);
            }
            tm.reset();
            cont = kr.rc;
            tnparams.hWt = cont ? (1.0 + SMOOTHLEN_BUFFER) : 1.01;
            tnparams.runType = SPH_TREE_NEIGHBOR_BOXES;
            tnparams.seto = SPH_SET_ALL;
            tnparams.seti = SPH_SET_ALL;
            //			tnparams.set = SPH_SET_ACTIVE;
            tm.start();
            profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
            sphTreeNeighbor(tnparams, rootId, Vector<TreeId>()).get();
            profilerExit();
            tm.stop();
            tm.reset();

            tm.start();
            tnparams.seti = SPH_INTERACTIONS_IJ;
            tnparams.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
            profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
            sphTreeNeighbor(tnparams, rootId, checklist).get();
            profilerExit();
            tm.stop();
            tm.reset();
            kr = SphRunReturn();
        } while (cont);
        if (stars && minrung <= 1) {
            //		sph_particles_energy_to_entropy();
        }
    } else if (stars && minrung <= 1) {
        //		sph_particles_energy_to_entropy();
    }

    sparams.phase = 1;
    sparams.runType = SPH_RUN_HYDRO;
    tm.reset();
    tm.start();
    kr = sphRun(sparams, true);
    dtinvCfl = kr.dtinvCfl;
    dtinvDivv = kr.dtinvDivv;
    dtinvVisc = kr.dtinvVisc;
    dtinvDiff = kr.dtinvDiff;
    dtinvCond = kr.dtinvCond;
    dtinvAcc = kr.dtinvAcc;
    dtinvOmega = kr.dtinvOmega;
    tm.stop();
    maxRung = std::max(kr.maxRung, maxRung);
    if (verbose) {
        PRINT("sph_run(SPH_RUN_HYDRO): tm = %e max_vsig = %e max_rung = %i, %i\n", tm.read(),
            kr.maxVsig, kr.maxRungHydro, kr.maxRungGrav);
    }
    tm.reset();

    tnparams.hWt = 1.01;
    tnparams.runType = SPH_TREE_NEIGHBOR_BOXES;
    tnparams.seti = SPH_SET_ALL;
    tnparams.seto = SPH_SET_ACTIVE;
    tm.start();
    profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_NEIGHBORS");
    sphTreeNeighbor(tnparams, rootId, Vector<TreeId>()).get();
    profilerExit();
    tm.stop();
    tm.reset();
    tm.start();
    tnparams.seti = SPH_INTERACTIONS_IJ;
    tnparams.runType = SPH_TREE_NEIGHBOR_NEIGHBORS;
    profilerEnter("sph_tree_neighbor:SPH_TREE_NEIGHBOR_BOXES");
    sphTreeNeighbor(tnparams, rootId, checklist).get();
    profilerExit();
    tm.stop();
    tm.reset();

    bool rc = true;
    while (rc) {
        sparams.runType = SPH_RUN_RUNGS;
        tm.start();
        rc = sphRun(sparams, true).rc;
        //	max_rung = kr.max_rung;
        tm.stop();
        if (verbose) {
            PRINT("sph_run(SPH_RUN_RUNGS): tm = %e \n", tm.read());
        }
        tm.reset();
    }
    sphParticlesApplyUpdates(minrung, 2, t0, tau);
    if (verbose) {
        PRINT("Completing SPH step with max_rungs = %i, %i\n", kr.maxRungHydro, kr.maxRungGrav);
    }

#ifdef ENTROPY
    sphParticlesCacheFreeEntr();
#else
    sphParticlesCacheFreeEint();
#endif

    sphParticlesCacheFree1();

    sphTreeDestroy(true);
    sphParticlesCacheFree2();
    sphParticlesResetSemiactive();
    //	PRINT( "%i\n", max_rung);

    float dtinvMax = std::max(dtinvCfl, dtinvVisc);
    dtinvMax = std::max(dtinvMax, dtinvDiff);
    dtinvMax = std::max(dtinvMax, dtinvCond);
    dtinvMax = std::max(dtinvMax, dtinvAcc);
    dtinvMax = std::max(dtinvMax, dtinvDivv);
    dtinvMax = std::max(dtinvMax, dtinvOmega);
    FILE* fp = fopen("timestep.txt", "at");
    fprintf(fp, "%e %i %e %e %e %e %e %e %e ", tau, maxRung, dtinvCfl, dtinvVisc, dtinvDiff,
        dtinvCond, dtinvAcc, dtinvDivv, dtinvOmega);
    if (dtinvMax == dtinvCfl) {
        PRINT("CFL");
        fprintf(fp, "%s\n", "CFL");
    } else if (dtinvMax == dtinvVisc) {
        PRINT("VISCOSITY");
        fprintf(fp, "%s\n", "VISCOSITY");
    } else if (dtinvMax == dtinvDiff) {
        PRINT("DIFFUSION");
        fprintf(fp, "%s\n", "DIFFUSION");
    } else if (dtinvMax == dtinvCond) {
        PRINT("CONDUCTION");
        fprintf(fp, "%s\n", "CONDUCTION");
    } else if (dtinvMax == dtinvAcc) {
        PRINT("ACCELERATION");
        fprintf(fp, "%s\n", "ACCELERATION");
    } else if (dtinvMax == dtinvDivv) {
        PRINT("VELOCITY DIVERGENCE");
        fprintf(fp, "%s\n", "DIVV");
    } else if (dtinvMax == dtinvOmega) {
        PRINT("OMEGA");
        fprintf(fp, "%s\n", "OMEGA");
    } else {
        ALWAYS_ASSERT(false);
    }
    PRINT(" LIMITED TIMESTEP\n");
    fclose(fp);
    kr.maxRung = maxRung;
    return kr;
}

std::pair<KickReturn, TreeCreateReturn> kickStep(int minrung, double scale, double dadt, double t0,
    double theta, bool firstCall, bool fullEval) {
    Timer tm;
    tm.start();
    PRINT("domains_begin\n");
    domainsBegin(minrung);
    PRINT("domains_end \n");
    domainsEnd();
    tm.stop();
    domainTime += tm.read();
    tm.reset();
    tm.start();
    bool const sph = getOptions().sph;
    // ALWAYS_ASSERT(sph_particles_max_smooth_len() != INFINITY);
    TreeCreateParameters tparams(minrung, theta, 0.f);
    PRINT("Create tree %i %e\n", minrung, theta);
    profilerEnter("tree_create");
    auto sr = treeCreate(tparams);
    profilerExit();
    PRINT("Done with tree\n");
    bool const vsoft = getOptions().vsoft;

    tm.stop();
    sortTime += tm.read();
    tm.reset();
    tm.start();
    //	PRINT("numberOfActiveParticles = %li\n", sr.numberOfActiveParticles);
    KickParameters kparams;
    if (dadt != 0.0) {
        kparams.maxDt = SCALE_DT * scale / fabs(dadt);
    }
    kparams.htime = false;
    kparams.ascending = kparams.descending = kparams.top = true;
    kparams.doPhi = fullEval;
    kparams.sign = 1.0;
    kparams.nodeLoad = flopsPerNode / flopsPerParticle;
    kparams.gpu = getOptions().cuda;
    usedGpu = kparams.gpu;
    kparams.minLevel = tparams.minLevel;
    kparams.saveForce = getOptions().saveForce;
    kparams.gm = getOptions().gravity ? getOptions().gm : 0.0;
    kparams.eta = getOptions().eta;
    kparams.h = getOptions().hsoft;
    kparams.a = scale;
    kparams.firstCall = firstCall;
    kparams.minRung = minrung;
    kparams.t0 = t0;
    kparams.theta = theta;
    Expansion<float> l;
    for (int i = 0; i < EXPANSION_SIZE; i++) {
        l[i] = 0.0f;
    }
    Array<Fixed32, NDIM> pos;
    for (int dim = 0; dim < NDIM; dim++) {
        pos[dim] = 0.f;
    }
    TreeId rootId;
    rootId.proc = 0;
    rootId.index = 0;
    Vector<TreeId> checklist;
    checklist.push_back(rootId);
    PRINT("Do kick\n");
    profilerEnter("kick");
    KickReturn kr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
    profilerExit();
    tm.stop();
    kickTime += tm.read();

    treeDestroy();
    particlesCacheFree();
    PRINT("kick done\n");
    return std::make_pair(kr, sr);
}

std::pair<KickReturn, TreeCreateReturn> kickStepHierarchical(int& minrung, int maxRung,
    double scale, double adot, double tau, double t0, double theta, EnergiesT* energies,
    int minrung0, bool doPhi, hpx::future<void>* lcFut, bool nocrop, double drag) {
    profilerEnter(__FUNCTION__);
    Timer tm;
    KickReturn kr;
    TreeCreateReturn sr;
    //	minrung = std::max(minrung, 1);
    //	max_rung = std::max(max_rung,minrung);
    // PRINT( "%i %i %i\n", minrung, max_rung, minrung0);
    Vector<int> levels(std::max(maxRung - minrung, 0) + 1);
    int k = 0;
    for (int i = maxRung; i > minrung; i--) {
        levels[k++] = i;
    }
    levels[k++] = minrung;
    bool ascending = true;
    bool top;
    bool clipTop = false;
    //	PRINT("climbing kick ladder\n");
    Timer totalTime;
    totalTime.start();
    double particlesProcessed = 0.0;
    for (int li = 0; li < levels.size(); li++) {
        if (levels[li] == minrung) {
            ascending = false;
            top = true;
        } else {
            top = false;
        }
        if (ascending) {
            //			PRINT("ASCENDING  rung %i\n", levels[li]);
        } else if (top) {
            //			PRINT("AT TOP     rung %i\n", levels[li]);
        } else if (!ascending) {
            //			PRINT("DESCENDING rung %i\n", levels[li]);
        }

        if (!ascending && !top) {
            particlesPushRungs();
        }

        if (ascending || top) {
            tm.reset();
            tm.start();
            domainsBegin(levels[li]);
            domainsEnd();
            tm.stop();
            //			PRINT("Domains took %e\n", tm.read());
        }

        if (!ascending) {
            tm.reset();
            tm.start();
            particlesSortByRung(levels[li]);
            tm.stop();
            //			PRINT("Rung sort took %e\n", tm.read());
        }
        auto rng = particlesCurrentRange();
        particlesProcessed += rng.second - rng.first;
        if (top && minrung == minrung0) {
            auto counts = particlesRungCounts();
            if (counts.size() > minrung0 + 1) {
                auto const total = getOptions().numberOfParticles;
                //				PRINT("Rungs\n");
                for (int i = 0; i < counts.size(); i++) {
                    //					PRINT("%i %li %f %%\n", i, counts[i], 100.0 * counts[i] /
                    // total);
                }
                size_t fast = 0;
                size_t slow = counts[minrung0];
                for (int i = minrung0 + 1; i < counts.size(); i++) {
                    fast += counts[i];
                }
                if (3 * fast > slow) {
                    clipTop = true && !nocrop;
                    //					PRINT("------------------------------------\n");
                    //					PRINT("Setting minimum level to %i\n", minrung + 1);
                    //					PRINT("------------------------------------\n");
                }
            }
        }

        /*		if (ascending || top) {
         tm.reset();
         tm.start();
         const float dt = 0.5 * rung_dt[levels[li]] * t0;
         drift(scale, dt, tau, tau + dt, t0, levels[li]);
         tm.stop();
         PRINT("drift = %e\n", tm.read());
         }*/
        TreeCreateParameters tparams(levels[li], theta, 0.f);
        tm.reset();
        tm.start();
        auto thisSr = treeCreate(tparams);
        tm.stop();
        //		PRINT("Tree create took %e\n", tm.read());
        if (top) {
            sr = thisSr;
        }
        KickParameters kparams;
        if (clipTop && levels[li] == minrung0 + 1) {
            kparams.top = true;
        } else {
            kparams.top = top;
        }
        kparams.sign = getOptions().createGlass ? -1.0 : 1.0;
        kparams.ascending = ascending || top;
        if (clipTop && top) {
            kparams.descending = false;
        } else {
            kparams.descending = !ascending || top;
        }
        kparams.velInit = getOptions().plummer;
        kparams.nodeLoad = flopsPerNode / flopsPerParticle;
        kparams.gpu = true;
        kparams.minLevel = tparams.minLevel;
        kparams.saveForce = getOptions().saveForce;
        kparams.gm = getOptions().gm;
        kparams.eta = getOptions().eta;
        kparams.h = getOptions().hsoft;
        kparams.a = scale;
        kparams.firstCall = tau == 0.0;
        kparams.minRung = levels[li];
        kparams.doPhi = doPhi && top;
        kparams.t0 = t0;
        kparams.maxDt = SCALE_DT / adot;
        kparams.theta = theta;
        Expansion<float> l;
        for (int i = 0; i < EXPANSION_SIZE; i++) {
            l[i] = 0.0f;
        }
        Array<Fixed32, NDIM> pos;
        for (int dim = 0; dim < NDIM; dim++) {
            pos[dim] = 0.f;
        }
        TreeId rootId;
        rootId.proc = 0;
        rootId.index = 0;
        Vector<TreeId> checklist;
        checklist.push_back(rootId);
        tm.reset();
        tm.start();
        if (top) {
            resetGravityCounters();
            setGravityCounterUse(true);
        }
        KickReturn thisKr = kick(kparams, l, pos, rootId, checklist, checklist, nullptr);
        if (top) {
            setGravityCounterUse(false);
        }
        energies->tckin += thisKr.dkin;
        tm.stop();
        //		PRINT("Kick took %e\n", tm.read());
        if (clipTop && top) {
            particlesSetMinrung(minrung0 + 1);
        }
        if (top) {
            kr = thisKr;
            energies->pot = kr.pot / scale;
            energies->kin = kr.kin / sqr(scale);
            energies->xmom = kr.xmom / scale;
            energies->ymom = kr.ymom / scale;
            energies->zmom = kr.zmom / scale;
            energies->nmom = kr.nmom / scale;
        }
        if (clipTop && top) {
            levels.push_back(minrung0 + 1);
        } else if (!ascending || top) {
            maxRung = thisKr.maxRung;
            kr.maxRung = maxRung;
            if (maxRung > levels[li]) {
                levels.push_back(levels[li] + 1);
            }
        }
        tm.stop();
        if ((!ascending || top) && !(clipTop && top)) {
            tm.reset();
            tm.start();
            double const dt = rungDt[levels[li]] * t0;
            if (lcFut) {
                lcFut->get();
                *lcFut = hpx::make_ready_future();
            }
            double a = 1.0 / cosmosAinv(adot, scale, dt);
            if (getOptions().plummer) {
                a = 1.0;
            }
            //			PRINT( "%e %e %e\n", a, a0, a1);
            drift(a, dt, tau, tau + dt, getOptions().numberOfSteps * t0, levels[li], drag);
            tm.stop();
            // PRINT("Drift took %e\n", tm.read());
        }
        treeReset();
        if (ascending && !top) {
            particlesPopRungs();
        }
    }
    //	PRINT("done climbing kick ladder\n");
    if (clipTop) {
        minrung++;
    }
    totalTime.stop();
    int bucketSize = getOptions().bucketSize;
    if (timings.size() <= bucketSize) {
        timings.resize(bucketSize + 1);
    }
    timings[bucketSize].numberOfParticles += particlesProcessed;
    timings[bucketSize].time += totalTime.read();
    profilerExit();
    return std::make_pair(kr, sr);
}

double doPowerSpectrum(int num, double a) {
    auto const softFilter = [](double kh) {
        if (kh > 0.01) {
            return -15.0 * (3.0 * kh * cos(kh) + ((sqr(kh) - 3.0) * sin(kh))) * pow(kh, -5.0);
        } else {
            return 1.0 - sqr(kh) / 14.0 + sqr(sqr(kh)) / 504.0;
        }
    };
    profilerEnter(__FUNCTION__);
    int const m = 8;
    PRINT("Computing power spectrum\n");
    double const h = getOptions().hubble;
    auto const glass = getOptions().createGlass;
    double const omegaM = getOptions().omegaM;
    double const boxSize =
        (units::scalesFromOptions(getOptions()).lengthScale() / units::kMegaparsec).value();
    double const n3 = pow((double) getOptions().nfour, 3);
    double const d1 = cosmosGrowthFactor(omegaM, a) / cosmosGrowthFactor(omegaM, 1.0);
    double factor = pow(boxSize, 3) / n3 / (glass ? 1.0 : (n3 * sqr(d1)));
    if (getOptions().closePack) {
        factor *= 0.25;
    }
    double maxpow = 0.0;
    std::string filename = "power." + std::to_string(num) + ".txt";
    FILE* fp = fopen(filename.c_str(), "wt");
    if (fp == NULL) {
        THROW_ERROR("Unable to open %s for writing\n", filename.c_str());
    }
    double kmin = 0.0;
    for (int mfactor = 1; mfactor <= 1; mfactor *= m) {
        auto power0 = powerSpectrumCompute(mfactor);
        double kmax = power0.k.back() * 1.000001;
        //	if (Mfactor != M * M * M) {
        //		kmax = sqrt(power0.k.front() * kmax * M);
        //	}
        double const c1 = h * h * h * factor;
        double const kshot = 2.0 * M_PI * getOptions().particlesPerDimension / boxSize;
        for (int i = 0; i < power0.p.size(); i++) {
            double const k = power0.k[i];
            if (k >= kmin && k <= kmax) {
                double const shotNoise =
                    power0.k[std::min(i + 1, (int) power0.p.size() - 1)] > kshot ?
                    1.0 * factor * n3 :
                    0.0;
                fprintf(fp, "%e %e %e %e %e\n", k / h, (power0.p[i] - power0.perr[i]) * c1,
                    power0.p[i] * c1, (power0.p[i] + power0.perr[i]) * c1, h * h * h * shotNoise);
            }
        }
        kmin = kmax;
    }
    fclose(fp);
    profilerExit();
    return maxpow;
}

void outputTimeFile() {
    profilerEnter(__FUNCTION__);
    double const a0 = 1.0 / (1.0 + getOptions().z0);
    double const tauMax = cosmosConformalTime(a0, 1.0);
    double a = a0;
    int m = getOptions().numberOfSteps;
    int n = 64;
    double const dtau = tauMax / (m * n);
    double tau = cosmosConformalTime(a0 * 1.0e-6, a0);
    double t = cosmosTime(a0 * 1.0e-6, a0);
    double z;
    FILE* fp = fopen("time.txt", "wt");
    if (fp == NULL) {
        THROW_ERROR("Unable to open time.txt for reading\n");
    }
    fprintf(
        fp, "%4s %13s %13s %13s %13s\n", "step", "real yrs", "conformal yrs", "scale", "redshift");
    for (int i = 0; i <= m; i++) {
        z = 1.0 / a - 1.0;
        double const c0 = getOptions().codeToS / constants::kSpyr;
        fprintf(fp, "%4i %13.4e %13.4e %13.4e %13.4e\n", i, t * c0, tau * c0, a, z);
        for (int j = 0; j < n; j++) {
            double const dadt1 = cosmosDadtau(a);
            double const dadt2 = cosmosDadtau(a + dadt1 * dtau);
            t += 0.5 * (2.0 * a + dadt1 * dtau) * dtau;
            a += 0.5 * (dadt1 + dadt2) * dtau;
            tau += dtau;
        }
    }
    fclose(fp);
    profilerExit();
}

void saveGlass() {
    int const n = getOptions().particlesPerDimension;
    FILE* fp = fopen("glass.bin", "wb");
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(&particlesPos(XDIM, 0), sizeof(Fixed32), particlesSize(), fp);
    fwrite(&particlesPos(YDIM, 0), sizeof(Fixed32), particlesSize(), fp);
    fwrite(&particlesPos(ZDIM, 0), sizeof(Fixed32), particlesSize(), fp);
    fclose(fp);
}

void driver() {
    Timer totalTime;
    totalTime.start();
    Timer tmr;
    tmr.start();
    DriverParameters params{};
    Timer buckets50;
    Timer buckets20;
    Timer buckets2;
    hpx::future<void> lcFut = hpx::make_ready_future();
    int buckets[11] = {80, 112, 112, 128, 128, 160, 176, 184, 184, 184, 192};
    double a0 = 1.0 / (1.0 + getOptions().z0);
    if (getOptions().readCheck != -1) {
        params = readCheckpoint();
    } else {
        outputTimeFile();
        if (getOptions().createGlass) {
            if (hpxSize() != 1) {
                THROW_ERROR("Create glass can only be done with a single locality\n");
            }
            size_t const numberOfParticles = getOptions().numberOfParticles;
            size_t const b = hpxRank() * numberOfParticles / hpxSize();
            size_t const e = (hpxRank() + 1) * numberOfParticles / hpxSize();
            particlesResize(e - b);
            for (ParticleInteger i = 0; i < e - b; i++) {
                double const x = rand1();
                double const y = rand1();
                double const z = rand1();
                particlesPos(XDIM, i) = x;
                particlesPos(YDIM, i) = y;
                particlesPos(ZDIM, i) = z;
                particlesVel(XDIM, i) = 0.0;
                particlesVel(YDIM, i) = 0.0;
                particlesVel(ZDIM, i) = 0.0;
                particlesRung(0) = 0;
            }
            params.tauMax = 1.0;
            params.a = 1.0;
            params.adot = 0.0;
        } else {
            if (getOptions().plummer) {
                params.tauMax = 1.0e-4;
                params.a = 1.0;
                params.adot = 0.0;
                plummerInit(1e-3);
            } else {
                params.a = a0;
                params.adot = cosmosDadt(a0);
                params.tauMax = cosmosConformalTime(a0, 1.0);
                initialize(getOptions().z0);
            }
        }
        if (getOptions().doTracers) {
            particlesSetTracers();
        }
        params.step = 0;
        params.flops = 0;
        params.bucketSize = 80;
        PRINT("TAU_MAX = %e\n", params.tauMax);
        params.tau = 0.0;
        params.maxRung = 0;
        params.itime = 0;
        params.iter = 0;
        params.minrung0 = 0;
        params.runtime = 0.0;
        params.totalProcessed = 0;
        params.years = cosmosTime(1e-6 * a0, a0) * getOptions().codeToS / constants::kSpyr;
        auto tmp1 = particlesSumEnergies();
        params.energies.tckin = tmp1.kin;
        //		write_checkpoint(params);
    }
    auto const gadgetFile = getOptions().gadget4Restart;
    if (gadgetFile != "") {
        auto header = particlesReadGadget4(gadgetFile);
        auto const numberOfParticles = header.npartTotal[1];
        auto opts = getOptions();
        opts.codeToG = header.mass[1] * 1e10 / opts.hubble;
        opts.codeToG *= constants::kM0;
        opts.codeToCm = pow(opts.codeToG * (8.0 * M_PI) * numberOfParticles * constants::kG /
                (3.0 * opts.omegaM * sqr(constants::kH0 * opts.hubble)),
            1.0 / 3.0);
        opts.codeToS = opts.codeToCm / constants::kC;
        double hValue = constants::kH0 * opts.codeToS;
        opts.gm = opts.omegaM * 3.0 * sqr(hValue * opts.hubble) / (8.0 * M_PI) / numberOfParticles;
        setOptions(opts);
        params.a = 1.0 / (header.redshift + 1.0);
        params.adot = cosmosDadt(a0);
        params.tauMax = cosmosConformalTime(params.a, 1.0);
        params.years = cosmosTime(1e-6 * a0, params.a) * getOptions().codeToS / constants::kSpyr;
        PRINT("box_size = %e Mpc\n", opts.codeToCm / constants::kMpcToCm);
        PRINT("code_to_g = %e\n", opts.codeToG);
        PRINT("code_to_cm = %e\n", opts.codeToCm);
        PRINT("code_to_s = %e\n", opts.codeToS);
    }
    PRINT("tau_max = %e\n", params.tauMax);
    auto& years = params.years;
    int& maxRung = params.maxRung;
    auto& a = params.a;
    auto& adot = params.adot;
    auto& tau = params.tau;
    auto& tauMax = params.tauMax;
    auto& energies = params.energies;
    auto& energy0 = params.energy0;
    auto& bucketSize = params.bucketSize;
    auto& itime = params.itime;
    auto& minrung0 = params.minrung0;
    minrung0 = std::max(minrung0, getOptions().minrung);
    auto& iter = params.iter;
    int& step = params.step;
    auto& totalProcessed = params.totalProcessed;
    auto& runtime = params.runtime;
    int numberOfSteps = getOptions().numberOfSteps;
    double z0 = getOptions().z0;
    double dloga = log(z0 + 1.0) / numberOfSteps;
    double pot;
    int thisIter = 0;
    double lastTheta = -1.0;
    Timer reset;
    reset.start();
    if (getOptions().doLc) {
        lcInit(tau, tauMax);
    }
    double dt = 0.0;
    int jiter = 0;
    if (tau == 0.0) {
        particlesSetMinrung(minrung0);
    }
    auto const checkLc = [&tau, &dt, &tauMax, &a, &iter](bool force) {
        //	profiler_enter("light cone");
        if (force || lcTimeToFlush(tau, tauMax)) {
            Timer tm;
            PRINT("Flushing light cone\n");
            tm.start();
            lcBuffer2homes();
            tm.stop();
            PRINT("lc_buffer2homes %e\n", tm.read());
            tm.reset();
            tm.start();
            lcParticleBoundaries1();
            tm.stop();
            PRINT("lc_particles_boundaries1 %e\n", tm.read());
            tm.reset();
            tm.start();
            double const linkingLength =
                getOptions().lcB * pow(getOptions().numberOfParticles, -1.0 / NDIM);
            lcFormTrees(tau, linkingLength);
            tm.stop();
            PRINT("lc_form_trees %e\n", tm.read());
            tm.reset();
            tm.start();
            size_t cnt;
            lcFindNeighbors();
            PRINT("neighbors found\n");
            tm.stop();
            PRINT("lc_neighbors_found %e\n", tm.read());
            Timer tm2;
            tm2.start();
            do {
                Timer tm;
                tm.start();
                lcParticleBoundaries2();
                tm.stop();
                PRINT("%e ", tm.read());
                tm.reset();
                tm.start();
                cnt = lcFindGroups();
                tm.stop();
                PRINT("%e %li\n", cnt, tm.read());
            } while (cnt > 0);
            tm2.stop();
            PRINT("loop time = %e\n", tm2.read());
            tm.start();
            lcGroups2homes();
            tm.stop();
            PRINT("lc_groups2homes %e\n", tm.read());
            tm.reset();
            tm.start();
            lcParts2groups(a, linkingLength, iter);
            tm.stop();
            PRINT("lc_parts2groups %e\n", tm.read());
            tm.reset();
            tm.start();
            lcInit(tau, tauMax);
            tm.stop();
            PRINT("lc_init %e\n", tm.read());
            tm.reset();
        }
        //	profiler_exit();
    };
    auto checkpointlist = readCheckpointList();
    double const hsoft0 = getOptions().hsoft;
    bool doCheck = false;
    double checkz = 1.0 / a - 1.0;
    buckets50.start();
    EnergiesT energies0;
    double glassError = 1.0;
    for (;; step++) {
        //		PRINT("STEP  = %i\n", step);
        double t0 = tauMax / getOptions().numberOfSteps;
        do {
            Timer stepTm;
            stepTm.start();
            profilerEnter(__FUNCTION__);
            tmr.stop();
            if ((params.iter > 0 && tmr.read() > getOptions().checkFreq) || doCheck) {
                totalTime.stop();
                runtime += totalTime.read();
                totalTime.reset();
                totalTime.start();
                PRINT("WRITING CHECKPOINT FOR Z = %e\n", doCheck ? checkz : 1.0 / a - 1.0);
                writeCheckpoint(params);
                tmr.reset();
                if (doCheck) {
                    char* cmd;
                    asprintf(&cmd, "mv checkpoint.%i checkpoint.z.%.1f\n", params.iter, checkz);
                    system(cmd);
                    free(cmd);
                }
                doCheck = false;
            }
            tmr.start();
            int minrung = minRung(itime);
            bool fullEval = minrung == 0;
            if (fullEval) {
                int const number = step;
                if (getOptions().doTracers) {
                    outputTracers(number);
                }
                if (getOptions().doSlice) {
                    outputSlice(number, years);
                }
                if (fullEval) {
                    viewOutputViews((tau + 1e-6 * t0) / t0, a);
                }
                if (getOptions().doViews) {
                    Timer tm;
                    tm.start();
                    outputView(number, years);
                    tm.stop();
                    //					PRINT("View %i took %e \n", number, tm.read());
                }
            }
            bool rebound = false;
            if (tau == 0.0) {
                rebound = true;
            } else if (minrung == 0) {
                //				PRINT("Checking imbalance\n");
                double imbalance = domainsGetLoadImbalance();
                rebound = (imbalance > MAX_LOAD_IMBALANCE);
            }
            if (rebound) {
                PRINT("Doing rebound\n");
                domainsRebound();
                domainsBegin(0);
                domainsEnd();
                PRINT("Rebound done\n");
            }
            double theta;
            double const z = 1.0 / a - 1.0;
            auto opts = getOptions();
            opts.hsoft = hsoft0;    // / a;
            if (getOptions().createGlass) {
                theta = 0.4;
            } else {
                /*				if (z > 20.0) {
                 theta = 0.41;
                 } else if (z > 2.0) {
                 theta = 0.51;
                 } else {
                 theta = 0.61;
                 }*/
                if (z > 20.0) {
                    theta = 0.41;
                } else if (z > 2.0) {
                    theta = 0.51;
                } else {
                    theta = 0.61;
                }
            }
            auto const ts = 100 * tau / t0 / getOptions().numberOfSteps;
            if (ts <= 10.0) {
                bucketSize = 112;
            } else if (ts < 55.0) {
                bucketSize = 112 + (ts - 10.0) / 45.0 * (184 - 112);
            } else {
                bucketSize = 184;
            }
            // bucket_size = 128;
            opts.bucketSize = bucketSize;
            opts.theta = theta;
            setOptions(opts);
            lastTheta = theta;
            std::pair<KickReturn, TreeCreateReturn> tmp;
            bool nocrop = false;
            if (int(tau / t0) % 8 == 0 && fullEval) {
                nocrop = true;
                minrung0 = std::max(minrung0 - 1, getOptions().minrung);
            }
            int thisMinrung = std::max(minrung, minrung0);
            int om = thisMinrung;
            //				PRINT("MINRUNG0 = %i\n", minrung0);
            //			PRINT( "Doing kick\n");
            resetFlops();
            if (fullEval) {
                //				kick_workspace::clear_buffers();
                //				tree_destroy(true);
                if (getOptions().doPower) {
                    doPowerSpectrum(step, a);
                }
#ifndef CHECK_MUTUAL_SORT
                if (getOptions().doGroups) {
                    doGroups(step, a);
                }
#endif
            }
            if (getOptions().useGlass && minrung == minrung0) {
                //			particles_displace(rand1(), rand1(), rand1());
            }
            if (getOptions().sph) {
                double heating = 0.0;
                tmp = kickStep(minrung, a, a * adot, t0, theta, tau == 0.0, fullEval);
                // Migration must finish before the SPH tree stores gas indices.
                sphStep1(minrung, a, tau, t0, 1, a * adot, maxRung, iter, dt, &heating);
            } else {
                tmp = kickStepHierarchical(om, maxRung, a, adot, tau, t0, theta, &energies,
                    minrung0, fullEval, &lcFut, nocrop, sqrt(glassError));
            }
            if (tau == 0.0) {
                energies0 = energies;
            }
            double const flops = flopsPerSecond();
            resetFlops();
            if (om != thisMinrung) {
                minrung0++;
            }
            KickReturn kr = tmp.first;
            int maxRung0 = maxRung;
            maxRung = kr.maxRung;
            if (getOptions().sph) {
                maxRung = std::max(maxRung,
                    sphStep2(minrung, a, tau, t0, 1, a * adot, maxRung, iter, dt, &energies)
                        .maxRung);
                if (fullEval) {
                    energies.pot = kr.pot / a;
                }
                // Gas is synchronized by SPH above; undo the diagnostic contribution
                // of the next DM half-kick without changing stored particle momenta.
                energies.kin += kr.darkMatterKineticCorrection / sqr(a);
                energies.xmom += kr.darkMatterXMomentumCorrection / a;
                energies.ymom += kr.darkMatterYMomentumCorrection / a;
                energies.zmom += kr.darkMatterZMomentumCorrection / a;
                energies.nmom += kr.darkMatterMomentumNormCorrection / a;
                energies.tckin = energies.kin * sqr(a);
            }

            //			PRINT("GRAVITY max_rung = %i\n", kr.max_rung);
            if (minrung <= 0) {
                double energy = a * (energies.kin + energies.pot + energies.therm) +
                    energies.heating + energies.cosmic;
                if (tau == 0) {
                    energy0 = 0.0;
                    energies.cosmic = -energy;
                    energy = 0.0;
                }
                double const norm = getOptions().sph ?
                    energies.kin + fabs(energies.pot) + energies.therm + fabs(energies.heating) +
                        fabs(energies.cosmic) :
                    energies.kin + fabs(energies.pot) + energies.cosmic;
                double const err = (energy - energy0) / norm;
                if (tau > 0.0) {
                    glassError = fabs(energies0.pot - a * energies.pot) / (a * energies.pot);
                }
                FILE* fp = fopen("energy.txt", "at");
                double const momentumNorm = energies.nmom > 0.0 ? energies.nmom : 1.0;
                fprintf(fp, "%e %e %e %e %e %e %e %e %e %e\n", tau / t0, a,
                    energies.xmom / momentumNorm, energies.ymom / momentumNorm,
                    energies.zmom / momentumNorm, a * energies.pot, a * energies.kin,
                    energies.cosmic, err, glassError);
                fclose(fp);
                energies0 = energies;
                energies0.pot *= a;
            }
            dt = t0 / (1 << maxRung);
            TreeCreateReturn sr = tmp.second;
            //			PRINT("Done kicking\n");
            double adotdot;
            double a0 = a;
            if (!getOptions().createGlass && !getOptions().plummer) {
                cosmosUpdate(adotdot, adot, a, dt);
            }
            energies.cosmic += (a - a0) * (energies.tckin / (a0 * a0) + energies.therm);
            double const dyears = 0.5 * (a0 + a) * dt * getOptions().codeToS / constants::kSpyr;
            auto const z0 = 1.0 / a0 - 1.0;
            auto const z1 = 1.0 / a - 1.0;
            for (auto& z : checkpointlist) {
                if ((z - z0) * (z - z1) < 0.0) {
                    doCheck = true;
                    checkz = z;
                }
            }

            //			PRINT("%e %e\n", a1, a);
            Timer dtm;
            dtm.start();
            if (getOptions().sph) {
                drift(2.0 / (1.0 / a0 + 1.0 / a), dt, tau, tau + dt, tauMax);
            }
            dtm.stop();
            driftTime += dtm.read();
            FILE* textfp = fopen("progress.txt", "at");
            if (textfp == nullptr) {
                THROW_ERROR("unable to open progress.txt for writing\n");
            }
            iter++;
            Timer remainingTime;
            totalTime.stop();
            remainingTime.start();
            runtime += totalTime.read();
            double pps = totalProcessed / runtime;
            //	PRINT( "%e %e %e %e\n", kr.node_flops, kr.part_flops, sr.flops, dr.flops);
            double const numberOfParticles = getOptions().numberOfParticles;
            double close, direct;
            getGravityCounters(close, direct);
            direct += close;
            if (fullEval) {
                PRINT_BOTH(textfp,
                    "\n%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
                    "runtime", "i", "z", "a", "adot", "timestep", "years", "mnr", "mxr", "bs",
                    "Tflops", "time", "close", "direct");
            }
            stepTm.stop();
            PRINT_BOTH(textfp,
                "%10.3e %10i %10.3e %10.3e %10.3e %10.3e %10.3e %10i %10i %10i %10.3e %10.3e "
                "%10.3e %10.3e \n",
                runtime, iter - 1, z, a, adot, tau / t0, years, minrung, maxRung, bucketSize,
                flops * 1e-12, stepTm.read(), close, direct);
            fclose(textfp);
            totalTime.reset();
            remainingTime.stop();
            runtime += remainingTime.read();
            totalTime.start();
            itime = inc(itime, maxRung);
            domainTime = 0.0;
            sortTime = 0.0;
            kickTime = 0.0;
            driftTime = 0.0;
            tau += dt;
            thisIter++;
            years += dyears;
            if (1.0 / a < getOptions().z1 + 1.0) {
                PRINT("Reached minimum redshift, exiting...\n");
                break;
            } else if (thisIter > getOptions().maxIter) {
                PRINT("Reached maximum iteration, exiting...\n");
                break;
            }
            profilerExit();
            profilerOutput();
            profilerEnter(__FUNCTION__);
            jiter++;
            if (jiter > 100) {
                //			abort();
            }
            if (getOptions().doLc && minRung(itime) <= minrung0) {
                checkLc(false);
            }
            lcFut = hpx::make_ready_future();

        } while (itime != 0);
        if (thisIter > getOptions().maxIter || 1.0 / a < getOptions().z1 + 1.0) {
            break;
        } else if (getOptions().createGlass && std::abs(glassError) < 1.0e-5) {
            saveGlass();
            break;
        }
    }
    buckets2.stop();
    FILE* fp = fopen("buckets.txt", "at");
    fprintf(fp, "%i %e %e %e\n", getOptions().bucketSize, buckets50.read(), buckets20.read(),
        buckets2.read());
    fclose(fp);
    if (getOptions().doLc) {
        checkLc(true);
        particlesFree();
    }
    KickWorkspace::clearBuffers();
}

bool dirExists(char const* path) {
    DIR* dir = opendir(path);
    if (dir) {
        closedir(dir);
        return true;
    } else {
        return false;
    }
}

namespace detail {

constexpr uint64_t kCheckpointMagic = 0x4354535048303031ULL;
constexpr uint32_t kCheckpointVersion = 1;
constexpr uint32_t kSphCheckpointVersion = 2;
constexpr uint32_t kMonaghanSph = 1;

struct CheckpointHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    double darkMatterMass;
    double gasMass;
    double gamma;
    double softening;
    double gravitationalMass;
    double omegaMatter;
    double omegaRadiation;
    double omegaLambda;
    double unitLength;
    double unitTime;
    double unitMass;
    uint64_t particlesPerSpecies;
    int32_t nominalSteps;
    uint32_t reserved;
};

// The pre-SPH main branch wrote this native layout directly.
struct LegacyEnergies {
    double pot;
    double kin;
    double tckin;
    double xmom;
    double ymom;
    double cosmic;
    double zmom;
    double nmom;
};

struct LegacyDriver {
    double a;
    double adot;
    double dummy;
    double tau;
    double tauMax;
    double energy0;
    LegacyEnergies energies;
    int maxRung;
    int iter;
    int minRung0;
    int step;
    int bucketSize;
    size_t totalProcessed;
    double flops;
    double runtime;
    double years;
    TimeType itime;
};

uint32_t checkpointFlags(Options const& opts) {
    constexpr uint32_t kSph = 1U << 0;
    constexpr uint32_t kChemistry = 1U << 1;
    constexpr uint32_t kStars = 1U << 2;
    constexpr uint32_t kConduction = 1U << 3;
    constexpr uint32_t kDiffusion = 1U << 4;
    constexpr uint32_t kVariableSoftening = 1U << 5;
    constexpr uint32_t kHierarchical = 1U << 6;
    constexpr uint32_t kLightcone = 1U << 7;
    constexpr uint32_t kGroups = 1U << 8;
    constexpr uint32_t kTracers = 1U << 9;
    // Older headers leave this bit clear and retain the fixed-softening behavior.
    constexpr uint32_t kSphSmoothing = 1U << 10;
    return (opts.sph ? kSph : 0U) | (opts.chem ? kChemistry : 0U) | (opts.stars ? kStars : 0U) |
        (opts.conduction ? kConduction : 0U) | (opts.diffusion ? kDiffusion : 0U) |
        (opts.vsoft ? kVariableSoftening : 0U) | (opts.htime ? kHierarchical : 0U) |
        (opts.doLc ? kLightcone : 0U) | (opts.doGroups ? kGroups : 0U) |
        (opts.doTracers ? kTracers : 0U) | (opts.sphSmoothing ? kSphSmoothing : 0U);
}

uint32_t checkpointKernelCode(std::string const& family, int degree) {
    uint32_t familyCode = 0;
    if (family == "spline") {
        familyCode = 1;
    } else if (family == "wendland") {
        familyCode = 2;
    } else if (family == "ferrers") {
        familyCode = 3;
    } else {
        THROW_ERROR("Unknown checkpoint kernel family: %s\n", family.c_str());
    }
    if (degree < 0 || degree > 15) {
        THROW_ERROR("Checkpoint kernel degree does not fit its selection field\n");
    }
    return (familyCode << 4) | static_cast<uint32_t>(degree);
}

uint32_t checkpointPhysicsCode(Options const& opts) {
    // Preserve the existing low-byte formulation and native header layout.
    // Historical default kernels encode zero in both new selection bytes.
    uint32_t const gravityCode =
        checkpointKernelCode(opts.gravityKernelFamily, opts.gravityKernelDegree) ^
        checkpointKernelCode("ferrers", 8);
    uint32_t const hydroCode =
        checkpointKernelCode(opts.hydroKernelFamily, opts.hydroKernelDegree) ^
        checkpointKernelCode("spline", 3);
    return (opts.sph ? kMonaghanSph : 0U) | (gravityCode << 8) | (hydroCode << 16);
}

bool legacyCheckpointCompatible(Options const& opts) {
    return !opts.sph && opts.htime && opts.gravityKernelFamily == "ferrers" &&
        opts.gravityKernelDegree == 8;
}

CheckpointHeader checkpointHeader(Options const& opts) {
    return {kCheckpointMagic, opts.sph ? kSphCheckpointVersion : kCheckpointVersion,
        checkpointFlags(opts), opts.darkMatterMass, opts.sphMass, opts.gamma, opts.hsoft, opts.gm,
        opts.omegaM, opts.omegaR, opts.omegaLam, opts.codeToCm, opts.codeToS, opts.codeToG,
        opts.numberOfParticles, opts.numberOfSteps, checkpointPhysicsCode(opts)};
}

bool checkpointCompatible(CheckpointHeader const& header, Options const& opts) {
    return header.version == (opts.sph ? kSphCheckpointVersion : kCheckpointVersion) &&
        header.flags == checkpointFlags(opts) && header.reserved == checkpointPhysicsCode(opts) &&
        header.darkMatterMass == opts.darkMatterMass && header.gasMass == opts.sphMass &&
        header.gamma == opts.gamma && header.softening == opts.hsoft &&
        header.gravitationalMass == opts.gm && header.omegaMatter == opts.omegaM &&
        header.omegaRadiation == opts.omegaR && header.omegaLambda == opts.omegaLam &&
        header.unitLength == opts.codeToCm && header.unitTime == opts.codeToS &&
        header.unitMass == opts.codeToG && header.particlesPerSpecies == opts.numberOfParticles &&
        header.nominalSteps == opts.numberOfSteps;
}

// Stored only for Monaghan SPH checkpoints; the dark-matter layout is unchanged.
std::array<double, 13> sphCheckpointParameters(Options const& opts) {
    return {opts.alpha0, opts.beta, opts.viscosityEpsilon, opts.sphNeighborNumber, opts.hmin,
        opts.hmax, opts.cfl, opts.eta, opts.yAcceleration, opts.centralAcceleration, opts.hcentral,
        opts.damping, opts.gravity ? 1.0 : 0.0};
}

void writeDriverState(FILE* fp, DriverParameters const& params) {
    auto const header = checkpointHeader(getOptions());
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        THROW_ERROR("Unable to write checkpoint header\n");
    }
    if (getOptions().sph) {
        auto const sphParameters = sphCheckpointParameters(getOptions());
        if (fwrite(sphParameters.data(), sizeof(double), sphParameters.size(), fp) !=
            sphParameters.size()) {
            THROW_ERROR("Unable to write SPH checkpoint parameters\n");
        }
    }
    if (fwrite(&params, sizeof(params), 1, fp) != 1) {
        THROW_ERROR("Unable to write checkpoint driver state\n");
    }
}

DriverParameters readDriverState(FILE* fp) {
    DriverParameters params{};
    CheckpointHeader header{};
    uint64_t magic = 0;
    FREAD(&magic, sizeof(magic), 1, fp);
    if (fseek(fp, 0, SEEK_SET) != 0) {
        THROW_ERROR("Unable to rewind checkpoint\n");
    }
    if (magic == kCheckpointMagic) {
        FREAD(&header, sizeof(header), 1, fp);
        if (!checkpointCompatible(header, getOptions())) {
            THROW_ERROR("Checkpoint physics, particle layout or integrator differs from the "
                        "requested run\n");
        }
        if (getOptions().sph) {
            auto sphParameters = sphCheckpointParameters(getOptions());
            FREAD(sphParameters.data(), sizeof(double), sphParameters.size(), fp);
            if (sphParameters != sphCheckpointParameters(getOptions())) {
                THROW_ERROR("SPH checkpoint parameters differ from the requested run\n");
            }
        }
        FREAD(&params, sizeof(params), 1, fp);
    } else {
        if (!legacyCheckpointCompatible(getOptions())) {
            THROW_ERROR("Legacy checkpoints support only the original dark-matter hierarchical "
                        "mode with the degree-eight Ferrers gravity kernel\n");
        }
        LegacyDriver old{};
        FREAD(&old, sizeof(old), 1, fp);
        params.a = old.a;
        params.adot = old.adot;
        params.dummy = old.dummy;
        params.tau = old.tau;
        params.tauMax = old.tauMax;
        params.energy0 = old.energy0;
        params.energies.pot = old.energies.pot;
        params.energies.kin = old.energies.kin;
        params.energies.tckin = old.energies.tckin;
        params.energies.xmom = old.energies.xmom;
        params.energies.ymom = old.energies.ymom;
        params.energies.zmom = old.energies.zmom;
        params.energies.nmom = old.energies.nmom;
        params.energies.cosmic = old.energies.cosmic;
        params.maxRung = old.maxRung;
        params.iter = old.iter;
        params.minrung0 = old.minRung0;
        params.step = old.step;
        params.bucketSize = old.bucketSize;
        params.totalProcessed = old.totalProcessed;
        params.flops = old.flops;
        params.runtime = old.runtime;
        params.years = old.years;
        params.itime = old.itime;
    }
    return params;
}

}    // namespace detail

void checkpointStateTest() {
    constexpr double kScale = 0.5;
    constexpr double kClock = 0.125;
    constexpr double kThermal = 3.0;
    constexpr double kHeating = 0.25;
    constexpr int kRung = 4;
    FILE* fp = tmpfile();
    if (fp == nullptr) {
        THROW_ERROR("Cannot create checkpoint test stream\n");
    }
    DriverParameters input{};
    input.a = kScale;
    input.tau = kClock;
    input.tauMax = 2.0;
    input.itime = 128;
    input.maxRung = kRung;
    input.energies.therm = kThermal;
    input.energies.heating = kHeating;
    detail::writeDriverState(fp, input);
    rewind(fp);
    auto const restored = detail::readDriverState(fp);
    ALWAYS_ASSERT(
        restored.a == input.a && restored.tau == input.tau && restored.tauMax == input.tauMax);
    ALWAYS_ASSERT(restored.itime == input.itime && restored.maxRung == input.maxRung);
    ALWAYS_ASSERT(restored.energies.therm == kThermal && restored.energies.heating == kHeating);
    auto changed = getOptions();
    auto const header = detail::checkpointHeader(changed);
    changed.outputUnits = "both";
    changed.outputFrame = "lab";
    ALWAYS_ASSERT(detail::checkpointCompatible(header, changed));
    changed = getOptions();
    if (changed.sph) {
        auto oldFormulation = header;
        oldFormulation.reserved = 0;
        oldFormulation.version = 1;
        ALWAYS_ASSERT(!detail::checkpointCompatible(oldFormulation, changed));
    }
    changed.sph = !changed.sph;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    changed.htime = !changed.htime;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    changed.chem = !changed.chem;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    changed.sphSmoothing = !changed.sphSmoothing;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    auto const differentKernelDegree = [](std::string const& family, int degree) {
        if (family == "spline") {
            return degree == 3 ? 5 : 3;
        }
        if (family == "wendland") {
            return degree == 2 ? 5 : 2;
        }
        return degree == 8 ? 6 : 8;
    };
    changed = getOptions();
    changed.gravityKernelFamily = changed.gravityKernelFamily == "spline" ? "ferrers" : "spline";
    changed.gravityKernelDegree = 8;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    auto const changedGravityHeader = detail::checkpointHeader(changed);
    ALWAYS_ASSERT(detail::checkpointCompatible(changedGravityHeader, changed));
    ALWAYS_ASSERT(!detail::checkpointCompatible(changedGravityHeader, getOptions()));
    changed = getOptions();
    changed.gravityKernelDegree =
        differentKernelDegree(changed.gravityKernelFamily, changed.gravityKernelDegree);
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    changed.hydroKernelFamily = changed.hydroKernelFamily == "spline" ? "ferrers" : "spline";
    changed.hydroKernelDegree = 8;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    auto const changedHydroHeader = detail::checkpointHeader(changed);
    ALWAYS_ASSERT(detail::checkpointCompatible(changedHydroHeader, changed));
    ALWAYS_ASSERT(!detail::checkpointCompatible(changedHydroHeader, getOptions()));
    changed = getOptions();
    changed.hydroKernelDegree =
        differentKernelDegree(changed.hydroKernelFamily, changed.hydroKernelDegree);
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    auto defaultKernelOptions = getOptions();
    defaultKernelOptions.gravityKernelFamily = "ferrers";
    defaultKernelOptions.gravityKernelDegree = 8;
    defaultKernelOptions.hydroKernelFamily = "spline";
    defaultKernelOptions.hydroKernelDegree = 3;
    ALWAYS_ASSERT(detail::checkpointHeader(defaultKernelOptions).reserved ==
        (defaultKernelOptions.sph ? detail::kMonaghanSph : 0U));
    auto legacyKernelOptions = defaultKernelOptions;
    legacyKernelOptions.sph = false;
    legacyKernelOptions.htime = true;
    ALWAYS_ASSERT(detail::legacyCheckpointCompatible(legacyKernelOptions));
    legacyKernelOptions.gravityKernelFamily = "spline";
    ALWAYS_ASSERT(!detail::legacyCheckpointCompatible(legacyKernelOptions));
    legacyKernelOptions.gravityKernelFamily = "ferrers";
    legacyKernelOptions.gravityKernelDegree = 6;
    ALWAYS_ASSERT(!detail::legacyCheckpointCompatible(legacyKernelOptions));
    auto legacySphOptions = getOptions();
    legacySphOptions.sph = true;
    legacySphOptions.sphSmoothing = false;
    auto const legacySphHeader = detail::checkpointHeader(legacySphOptions);
    ALWAYS_ASSERT(legacySphHeader.version == 2);
    ALWAYS_ASSERT((legacySphHeader.flags & (1U << 10)) == 0U);
    ALWAYS_ASSERT(detail::checkpointCompatible(legacySphHeader, legacySphOptions));
    legacySphOptions.sphSmoothing = true;
    ALWAYS_ASSERT(!detail::checkpointCompatible(legacySphHeader, legacySphOptions));
    auto const smoothingHeader = detail::checkpointHeader(legacySphOptions);
    ALWAYS_ASSERT(detail::checkpointCompatible(smoothingHeader, legacySphOptions));
    legacySphOptions.sphSmoothing = false;
    ALWAYS_ASSERT(!detail::checkpointCompatible(smoothingHeader, legacySphOptions));
    auto const sphParameters = detail::sphCheckpointParameters(getOptions());
    changed = getOptions();
    changed.alpha0 += 0.25;
    ALWAYS_ASSERT(sphParameters != detail::sphCheckpointParameters(changed));
    changed = getOptions();
    changed.sphNeighborNumber += 1.0;
    ALWAYS_ASSERT(sphParameters != detail::sphCheckpointParameters(changed));
    changed = getOptions();
    changed.hmax *= 2.0;
    ALWAYS_ASSERT(sphParameters != detail::sphCheckpointParameters(changed));
    changed = getOptions();
    changed.numberOfSteps++;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    changed.numberOfParticles++;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    if (detail::legacyCheckpointCompatible(getOptions())) {
        detail::LegacyDriver legacy{};
        legacy.a = kScale;
        legacy.tau = kClock;
        legacy.maxRung = kRung;
        legacy.energies.pot = -1.0;
        rewind(fp);
        ALWAYS_ASSERT(fwrite(&legacy, sizeof(legacy), 1, fp) == 1);
        rewind(fp);
        auto const previous = detail::readDriverState(fp);
        ALWAYS_ASSERT(previous.a == kScale && previous.tau == kClock && previous.maxRung == kRung);
        ALWAYS_ASSERT(previous.energies.pot == -1.0 && previous.energies.therm == 0.0);
    }
    fclose(fp);
    PRINT("Checkpoint state and mode validation passed\n");
}

void writeCheckpoint(DriverParameters params) {
    profilerEnter(__FUNCTION__);
    //	params.step--;
    if (hpxRank() == 0) {
        PRINT("Writing checkpoint\n");
        std::string command;
        command = std::string("mkdir -p checkpoint.") + std::to_string(params.iter);
        if (system(command.c_str()) != 0) {
            THROW_ERROR("Unable to execute : %s\n", command.c_str());
        }
    }
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<WriteCheckpointAction>(c, params));
    }
    std::string const fname = std::string("checkpoint.") + std::to_string(params.iter) +
        std::string("/checkpoint.") + std::to_string(hpxRank()) + std::string(".dat");
    FILE* fp = fopen(fname.c_str(), "wb");
    if (fp == nullptr) {
        THROW_ERROR("Unable to open %s for writing\n", fname.c_str());
    }
    detail::writeDriverState(fp, params);
    particlesSave(fp);
    domainsSave(fp);
    if (getOptions().doLc) {
        lcSave(fp);
    }
    fclose(fp);
    hpx::wait_all(futs.begin(), futs.end());
    if (hpxRank() == 0) {
        PRINT("Done writing checkpoint\n");
    }
    profilerExit();
}

DriverParameters readCheckpoint() {
    DriverParameters params{};
    if (hpxRank() == 0) {
        PRINT("Reading checkpoint\n");
    }
    Vector<hpx::future<DriverParameters>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ReadCheckpointAction>(c));
    }
    int const iter = getOptions().readCheck;
    std::string const fname = std::string("checkpoint.") + std::to_string(iter) +
        std::string("/checkpoint.") + std::to_string(hpxRank()) + std::string(".dat");
    FILE* fp = fopen(fname.c_str(), "rb");
    if (fp == nullptr) {
        THROW_ERROR("Unable to open %s for reading.\n", fname.c_str());
    }
    params = detail::readDriverState(fp);
    particlesLoad(fp);
    domainsLoad(fp);
    if (getOptions().doLc) {
        lcLoad(fp);
    }
    fclose(fp);
    hpx::wait_all(futs.begin(), futs.end());
    if (hpxRank() == 0) {
        PRINT("Done reading checkpoint\n");
    }
    return params;
}
