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

#include <atomic>

#include <chealpix.h>
#include <hpx/threading_base/annotated_function.hpp>
#include <silo.h>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stars.hpp>
#include <cosmictiger/view.hpp>

HPX_PLAIN_ACTION(outputView, OutputViewAction);

static Vector<Range<double>> viewBoxes;

struct DmPartInfo {
    Fixed32 x;
    Fixed32 y;
    Fixed32 z;
    float vx;
    float vy;
    float vz;
    int rung;
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & x;
        arc & y;
        arc & z;
        arc & vx;
        arc & vy;
        arc & vz;
        arc & rung;
    }
};

struct SphPartInfo : public DmPartInfo {
    float eint;
    float rho;
    float h;
    float hValue2;
    float hp;
    float h2;
    float hn;
    float hep;
    float hepp;
    float he;
    float zValue2;
    float alpha;
    float coldFrac;
    template <typename A>
    void serialize(A&& arc, unsigned ver) {
        DmPartInfo::serialize(arc, ver);
        arc & eint;
        arc & rho;
        arc & h;
        arc & hValue2;
        arc & hp;
        arc & h2;
        arc & hn;
        arc & hep;
        arc & hepp;
        arc & coldFrac;
        arc & he;
        arc & zValue2;
        arc & alpha;
    }
};

struct StarPartInfo : public DmPartInfo {
    float yValue2;
    float zValue2;
    float m;
    float zform;
    template <typename A>
    void serialize(A&& arc, unsigned ver) {
        DmPartInfo::serialize(arc, ver);
        arc & yValue2;
        arc & zValue2;
        arc & m;
        arc & zform;
    }
};

struct ViewReturn {
    Vector<Vector<SphPartInfo>> hydro;
    Vector<Vector<DmPartInfo>> dm;
    Vector<Vector<StarPartInfo>> star;
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & hydro;
        arc & dm;
        arc & star;
    }
};

ViewReturn viewGetParticles(Vector<Range<double>> boxes);

HPX_PLAIN_ACTION(viewGetParticles, ViewGetParticlesAction);

ViewReturn viewGetParticles(Vector<Range<double>> boxes = Vector<Range<double>>()) {
    if (hpxRank() == 0) {
        boxes = viewBoxes;
    }
    ViewReturn rc;
    rc.hydro.resize(boxes.size());
    rc.dm.resize(boxes.size());
    rc.star.resize(boxes.size());
    Vector<hpx::future<ViewReturn>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ViewGetParticlesAction>(c, boxes));
    }
    bool const chem = getOptions().chem;
    bool const stars = getOptions().stars;
    int const nthreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < nthreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, nthreads, boxes, chem, stars]() {
                ViewReturn rc;
                rc.hydro.resize(boxes.size());
                rc.dm.resize(boxes.size());
                rc.star.resize(boxes.size());
                PartInt const b = (size_t) proc * particlesSize() / nthreads;
                PartInt const e = (size_t) (proc + 1) * particlesSize() / nthreads;
                for (PartInt i = b; i < e; i++) {
                    Array<double, NDIM> x;
                    x[XDIM] = particlesPos(XDIM, i).toDouble();
                    x[YDIM] = particlesPos(YDIM, i).toDouble();
                    x[ZDIM] = particlesPos(ZDIM, i).toDouble();
                    for (int j = 0; j < boxes.size(); j++) {
                        if (boxes[j].contains(x)) {
                            PartInt const k = particlesType(i);
                            switch (k) {
                            case DARK_MATTER_TYPE: {
                                DmPartInfo info{};
                                info.x = particlesPos(XDIM, i);
                                info.y = particlesPos(YDIM, i);
                                info.z = particlesPos(ZDIM, i);
                                info.vx = particlesVel(XDIM, i);
                                info.vy = particlesVel(YDIM, i);
                                info.vz = particlesVel(ZDIM, i);
                                info.rung = particlesRung(i);
                                rc.dm[j].push_back(info);
                                break;
                            }
                            case SPH_TYPE: {
                                SphPartInfo info{};
                                int const l = particlesCatIndex(i);
                                if (!sphParticlesIsstar(l)) {
                                    info.x = particlesPos(XDIM, i);
                                    info.y = particlesPos(YDIM, i);
                                    info.z = particlesPos(ZDIM, i);
                                    info.vx = particlesVel(XDIM, i);
                                    info.vy = particlesVel(YDIM, i);
                                    info.vz = particlesVel(ZDIM, i);
#ifdef HOPKINS
                                    info.eint = sphParticlesEintPre(l);
                                    info.rho = sphParticlesRhoRho(l);
#else
                                    info.eint = sphParticlesEint(l);
                                    info.rho = sphParticlesRho(l);
#endif
                                    info.h = sphParticlesSmoothLen(l);
                                    info.rung = particlesRung(i);
                                    info.alpha = sphParticlesAlpha(l);
                                    if (chem) {
                                        info.hValue2 = sphParticlesH(l);
                                        info.hp = sphParticlesHp(l);
                                        info.hn = sphParticlesHn(l);
                                        info.h2 = sphParticlesH2(l);
                                        info.hep = sphParticlesHep(l);
                                        info.hepp = sphParticlesHepp(l);
                                        info.zValue2 = sphParticlesZ(l);
                                        info.he = sphParticlesHe0(l);
                                    }
                                    if (stars) {
                                        info.coldFrac = sphParticlesColdMass(l);
                                    } else {
                                        info.coldFrac = 0.0;
                                    }
                                    rc.hydro[j].push_back(info);
                                } else {
                                    int const k = particlesCatIndex(i);
                                    StarPartInfo info{};
                                    info.x = particlesPos(XDIM, i);
                                    info.y = particlesPos(YDIM, i);
                                    info.z = particlesPos(ZDIM, i);
                                    info.vx = particlesVel(XDIM, i);
                                    info.vy = particlesVel(YDIM, i);
                                    info.vz = particlesVel(ZDIM, i);
                                    info.rung = particlesRung(i);
                                    rc.star[j].push_back(info);
                                }
                                break;
                            }
                            }
                        }
                    }
                }
                return rc;
            },
            "collect_view_particles")));
    }

    for (auto& f : futs) {
        auto tmp = f.get();
        for (int i = 0; i < boxes.size(); i++) {
            rc.hydro[i].insert(rc.hydro[i].begin(), tmp.hydro[i].begin(), tmp.hydro[i].end());
            rc.dm[i].insert(rc.dm[i].begin(), tmp.dm[i].begin(), tmp.dm[i].end());
            rc.star[i].insert(rc.star[i].begin(), tmp.star[i].begin(), tmp.star[i].end());
        }
    }
    return rc;
}

void viewOutputViews(int cycle, double a) {
    profilerEnter("view_output_views");
    bool const chem = getOptions().chem;
    bool const stars = getOptions().stars;
    if (!viewBoxes.size()) {
        return;
    }
    double const codeToCm = getOptions().codeToCm;
    double const codeToS = getOptions().codeToS;
    double const codeToG = getOptions().codeToG;
    //	const double code_to_velocity = code_to_cm / code_to_s / a;
    //	const double code_to_energy = sqr(code_to_cm / code_to_s / a);
    //	const double code_to_density = code_to_g / (code_to_cm * sqr(code_to_cm));
    double const codeToVelocity = 1.;
    double const codeToEnergy = 1.;
    double const codeToDensity = 1.;
    for (int bi = 0; bi < viewBoxes.size(); bi++) {
        bool stars = getOptions().stars;
        PRINT("Outputing view for box (%e %e) (%e %e) (%e %e)\n", viewBoxes[bi].begin[XDIM],
            viewBoxes[bi].end[XDIM], viewBoxes[bi].begin[YDIM], viewBoxes[bi].end[YDIM],
            viewBoxes[bi].begin[ZDIM], viewBoxes[bi].end[ZDIM]);
        std::string filename = "view." + std::to_string(bi) + "." + std::to_string(cycle) + ".silo";
        DBfile* db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_PDB);
        ViewReturn parts;
        parts = viewGetParticles();
        Vector<float> x, y, z;
        auto opts = DBMakeOptlist(1);
        float z0 = 1.0 / a - 1.0;
        float tm = cosmosTime(1e-6, a) * getOptions().codeToS * constants::kSecondsToYears / 1e9;
        DBAddOption(opts, DBOPT_TIME, &tm);
        if (parts.dm[bi].size()) {
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.dm[bi].size(); i++) {
                x.push_back(parts.dm[bi][i].x.toFloat());
                y.push_back(parts.dm[bi][i].y.toFloat());
                z.push_back(parts.dm[bi][i].z.toFloat());
            }
            float* coords1[NDIM] = {x.data(), y.data(), z.data()};
            DBPutPointmesh(db, "dark_matter", NDIM, coords1, x.size(), DB_FLOAT, opts);
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.dm[bi].size(); i++) {
                x.push_back(parts.dm[bi][i].vx * codeToVelocity);
                y.push_back(parts.dm[bi][i].vy * codeToVelocity);
                z.push_back(parts.dm[bi][i].vz * codeToVelocity);
            }
            DBPutPointvar1(db, "vx", "dark_matter", x.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "vy", "dark_matter", y.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "vz", "dark_matter", z.data(), x.size(), DB_FLOAT, opts);
            x.resize(0);
            for (int i = 0; i < parts.dm[bi].size(); i++) {
                x.push_back(parts.dm[bi][i].rung);
            }
            DBPutPointvar1(db, "rung", "dark_matter", x.data(), x.size(), DB_FLOAT, opts);
        }
        if (parts.hydro[bi].size()) {
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.hydro[bi].size(); i++) {
                x.push_back(parts.hydro[bi][i].x.toFloat());
                y.push_back(parts.hydro[bi][i].y.toFloat());
                z.push_back(parts.hydro[bi][i].z.toFloat());
            }
            float* coords2[NDIM] = {x.data(), y.data(), z.data()};
            //			PRINT( "gas points\n");
            DBPutPointmesh(db, "gas", NDIM, coords2, x.size(), DB_FLOAT, opts);
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.hydro[bi].size(); i++) {
                x.push_back(parts.hydro[bi][i].vx * codeToVelocity);
                y.push_back(parts.hydro[bi][i].vy * codeToVelocity);
                z.push_back(parts.hydro[bi][i].vz * codeToVelocity);
            }
            // PRINT( "Vels\n");
            DBPutPointvar1(db, "hydro_vx", "gas", x.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "hydro_vy", "gas", y.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "hydro_vz", "gas", z.data(), x.size(), DB_FLOAT, opts);
            x.resize(0);
            for (int i = 0; i < parts.hydro[bi].size(); i++) {
                x.push_back(parts.hydro[bi][i].rung);
            }
            // PRINT( "rungs\n");
            DBPutPointvar1(db, "hydro_rung", "gas", x.data(), x.size(), DB_FLOAT, opts);
            if (stars) {
                x.resize(0);
                for (int i = 0; i < parts.hydro[bi].size(); i++) {
                    x.push_back(parts.hydro[bi][i].coldFrac);
                }
                DBPutPointvar1(db, "cold_frac", "gas", x.data(), x.size(), DB_FLOAT, opts);
            }
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.hydro[bi].size(); i++) {
                double const h = parts.hydro[bi][i].h;
                x.push_back(parts.hydro[bi][i].rho * codeToDensity);
                y.push_back(parts.hydro[bi][i].eint * codeToEnergy);
                z.push_back(parts.hydro[bi][i].alpha);
            }
            //			PRINT( "h and ent\n");
            DBPutPointvar1(db, "rho", "gas", x.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "eint", "gas", y.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "alpha", "gas", z.data(), z.size(), DB_FLOAT, opts);
            x.resize(0);
            y.resize(0);
            z.resize(0);
            if (chem) {
                double const codeToEnergyDensity = getOptions().codeToG /
                    (getOptions().codeToCm * sqr(getOptions().codeToS));    // 7
                double const codeToDensity =
                    pow(getOptions().codeToCm, -3) * getOptions().codeToG;    // 10
                Vector<float> p, q, t, u, w, s;
                for (int i = 0; i < parts.hydro[bi].size(); i++) {
                    x.push_back(parts.hydro[bi][i].hp);
                    y.push_back(parts.hydro[bi][i].hn);
                    z.push_back(parts.hydro[bi][i].h2);
                    p.push_back(parts.hydro[bi][i].hep);
                    q.push_back(parts.hydro[bi][i].hepp);
                    u.push_back(parts.hydro[bi][i].he);
                    s.push_back(parts.hydro[bi][i].hValue2);
                    w.push_back(parts.hydro[bi][i].zValue2);
                    double const h = parts.hydro[bi][i].h;
                    double rho = parts.hydro[bi][i].rho;
                    double const hp = parts.hydro[bi][i].hp;
                    double const hn = parts.hydro[bi][i].hn;
                    double const h2 = parts.hydro[bi][i].h2;
                    double const he = parts.hydro[bi][i].he;
                    double const zValue = parts.hydro[bi][i].zValue2;
                    double const hep = parts.hydro[bi][i].hep;
                    double const hepp = parts.hydro[bi][i].hepp;
                    double const yValue = he + hep + hepp;
                    double const hValue = parts.hydro[bi][i].hValue2;
                    double n = hValue + 2.f * hp + .5f * h2 + .25f * he + .5f * hep + .75f * hepp +
                        zValue / 10.0;
                    rho *= codeToDensity * pow(a, -3);
                    n *= constants::kAvo * rho;                                     // 8
                    double cv = 1.5 + 0.5 * h2 / (1. - .75 * yValue - 0.5 * h2);    // 4
                    double gamma = 1. + 1. / cv;                                    // 5
                    cv *= double(constants::kB);                                    // 1
                    double eint = parts.hydro[bi][i].eint;                          // 11
                    eint /= sqr(a);
                    eint *= sqr(getOptions().codeToCm / getOptions().codeToS);
                    // PRINT( "%e\n",  (code_to_energy_density * pow(code_to_density, -gamma)));
                    double energy = eint * rho;
                    double tValue = energy / (n * cv);    // 5
                    if (stars) {
                        tValue /= 1.f - parts.hydro[bi][i].coldFrac;
                    }
                    //	PRINT("%e %e %e %e %e %e \n", energy, n, cv, rho, K, T);
                    t.push_back(tValue);
                }
                DBPutPointvar1(db, "H", "gas", s.data(), x.size(), DB_FLOAT, opts);
                DBPutPointvar1(db, "He", "gas", u.data(), x.size(), DB_FLOAT, opts);
                DBPutPointvar1(db, "Hp", "gas", x.data(), x.size(), DB_FLOAT, opts);
                DBPutPointvar1(db, "Hn", "gas", y.data(), x.size(), DB_FLOAT, opts);
                DBPutPointvar1(db, "H2", "gas", z.data(), x.size(), DB_FLOAT, opts);
                DBPutPointvar1(db, "Hep", "gas", p.data(), x.size(), DB_FLOAT, opts);
                DBPutPointvar1(db, "Hepp", "gas", q.data(), x.size(), DB_FLOAT, opts);
                DBPutPointvar1(db, "Z", "gas", w.data(), x.size(), DB_FLOAT, opts);
                DBPutPointvar1(db, "T", "gas", t.data(), x.size(), DB_FLOAT, opts);
            }
        }
        if (parts.star[bi].size()) {
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.star[bi].size(); i++) {
                x.push_back(parts.star[bi][i].x.toFloat());
                y.push_back(parts.star[bi][i].y.toFloat());
                z.push_back(parts.star[bi][i].z.toFloat());
            }
            float* coords3[NDIM] = {x.data(), y.data(), z.data()};
            DBPutPointmesh(db, "stars", NDIM, coords3, x.size(), DB_FLOAT, opts);
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.star[bi].size(); i++) {
                x.push_back(parts.star[bi][i].vx * codeToVelocity);
                y.push_back(parts.star[bi][i].vy * codeToVelocity);
                z.push_back(parts.star[bi][i].vz * codeToVelocity);
            }
            DBPutPointvar1(db, "star_vx", "stars", x.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "star_vy", "stars", y.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "star_vz", "stars", z.data(), x.size(), DB_FLOAT, opts);
            x.resize(0);
            for (int i = 0; i < parts.star[bi].size(); i++) {
                x.push_back(parts.star[bi][i].rung);
            }
            DBPutPointvar1(db, "star_rung", "stars", x.data(), x.size(), DB_FLOAT, opts);

            Vector<float> p;
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.star[bi].size(); i++) {
                x.push_back(parts.star[bi][i].yValue2);
                y.push_back(parts.star[bi][i].zValue2);
                z.push_back(parts.star[bi][i].m);
                p.push_back(parts.star[bi][i].zform);
            }
            DBPutPointvar1(db, "star_Y", "stars", x.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "star_Z", "stars", y.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "star_M", "stars", z.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "star_zform", "stars", p.data(), x.size(), DB_FLOAT, opts);
        }
        DBFreeOptlist(opts);
        DBClose(db);
    }
    profilerExit();
}

void viewReadViewFile() {
    FILE* fp = fopen("view.txt", "rt");
    if (fp == nullptr) {
        PRINT("No view file found.\n");
        return;
    }
    Range<float> viewBox;
    while (fscanf(fp, "%f %f %f %f %f %f\n", &viewBox.begin[XDIM], &viewBox.end[XDIM],
               &viewBox.begin[YDIM], &viewBox.end[YDIM], &viewBox.begin[ZDIM],
               &viewBox.end[ZDIM]) == 2 * NDIM) {
        PRINT("Reading box for view at x range (%f,%f) yrange (%f,%f) zrange (%f,%f)\n",
            viewBox.begin[XDIM], viewBox.end[XDIM], viewBox.begin[YDIM], viewBox.end[YDIM],
            viewBox.begin[ZDIM], viewBox.end[ZDIM]);
        Range<double> viewBoxD;
        for (int dim = 0; dim < NDIM; dim++) {
            viewBoxD.begin[dim] = viewBox.begin[dim];
            viewBoxD.end[dim] = viewBox.end[dim];
        }
        viewBoxes.push_back(viewBoxD);
    }
    fclose(fp);
}

Vector<float> outputView(int number, double time) {
    profilerEnter("output_view");
    Vector<hpx::future<void>> futs;
    Vector<hpx::future<Vector<float>>> valFuts;
    for (auto const& c : hpxChildren()) {
        valFuts.push_back(hpx::async<OutputViewAction>(c, number, time));
    }
    int const nthreads = hpxHardwareConcurrency();
    int const nside = getOptions().viewSize;
    int const npix = 12 * sqr(nside);
    Vector<std::atomic<float>> values(npix);
    for (auto& v : values) {
        v = 0.0f;
    }
    for (int proc = 0; proc < nthreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [nthreads, proc, nside, &values]() {
                PartInt const begin = (size_t) proc * particlesSize() / nthreads;
                PartInt const end = (size_t) (proc + 1) * particlesSize() / nthreads;
                for (int i = begin; i < end; i++) {
                    for (int xi = -1; xi <= 0; xi++) {
                        for (int yi = -1; yi <= 0; yi++) {
                            for (int zi = -1; zi <= 0; zi++) {
                                double vec[NDIM];
                                long int ipix;
                                vec[XDIM] = particlesPos(XDIM, i).toFloat() + xi;
                                vec[YDIM] = particlesPos(YDIM, i).toFloat() + yi;
                                vec[ZDIM] = particlesPos(ZDIM, i).toFloat() + zi;
                                float const r2 = sqr(vec[XDIM], vec[YDIM], vec[ZDIM]);
                                if (r2 < 1.0 && r2 > 0.0) {
                                    vec2pix_ring(nside, vec, &ipix);
                                    atomicAdd(values[ipix], particlesMass(i) / r2);
                                }
                            }
                        }
                    }
                }
            },
            "project_view_mass")));
    }
    hpx::wait_all(futs.begin(), futs.end());
    Vector<float> results(values.size());
    for (int i = 0; i < results.size(); i++) {
        results[i] = values[i];
    }
    for (auto& valFut : valFuts) {
        auto const vals = valFut.get();
        for (int i = 0; i < vals.size(); i++) {
            results[i] += vals[i];
        }
    }
    if (hpxRank() == 0) {
        std::string filename = "view." + std::to_string(number) + ".dat";
        FILE* fp = fopen(filename.c_str(), "wb");
        if (fp == NULL) {
            THROW_ERROR("unable to open %s for writing\n", filename.c_str());
        }
        fwrite(&nside, sizeof(int), 1, fp);
        fwrite(&npix, sizeof(int), 1, fp);
        fwrite(results.data(), sizeof(float), npix, fp);
        fwrite(&number, sizeof(int), 1, fp);
        fwrite(&time, sizeof(double), 1, fp);
        fclose(fp);
    }
    profilerExit();
    return results;
}
