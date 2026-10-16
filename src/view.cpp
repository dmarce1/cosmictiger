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
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>

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
#include <cosmictiger/units.hpp>
#include <cosmictiger/view.hpp>

HPX_PLAIN_ACTION(outputView, OutputViewAction);

static Vector<Range<double>> viewBoxes;

struct DarkMatterParticleInfo {
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

struct SphParticleInfo : public DarkMatterParticleInfo {
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
        DarkMatterParticleInfo::serialize(arc, ver);
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

struct StarParticleInfo : public DarkMatterParticleInfo {
    float yValue2;
    float zValue2;
    float m;
    float zform;
    template <typename A>
    void serialize(A&& arc, unsigned ver) {
        DarkMatterParticleInfo::serialize(arc, ver);
        arc & yValue2;
        arc & zValue2;
        arc & m;
        arc & zform;
    }
};

struct ViewReturn {
    Vector<Vector<SphParticleInfo>> hydro;
    Vector<Vector<DarkMatterParticleInfo>> dm;
    Vector<Vector<StarParticleInfo>> star;
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & hydro;
        arc & dm;
        arc & star;
    }
};

namespace detail {

void checkViewWrite(int result, char const* name) {
    if (result < 0) {
        THROW_ERROR("Cannot write converted particle Silo object %s\n", name);
    }
}

void writeViewText(DBfile* db, char const* name, char const* value) {
    int const size = std::strlen(value) + 1;
    checkViewWrite(DBWrite(db, name, value, &size, 1, DB_CHAR), name);
}

void writeViewScalar(DBfile* db, char const* name, double value) {
    int const size = 1;
    checkViewWrite(DBWrite(db, name, &value, &size, 1, DB_DOUBLE), name);
}

class ViewUnitConversion {
    units::UnitScales scales_;
    units::CosmologicalContext context_;
    bool cosmological_;
    bool lab_;

    template <typename Quantity>
    double displayValue(Quantity const& quantity) const {
        return cosmological_ ? units::toCosmologicalValue(quantity, context_) :
                               quantity.quantity().value();
    }

public:
    ViewUnitConversion(double a, bool cosmological, bool lab)
      : scales_(getOptions().codeToCm, getOptions().codeToG, getOptions().codeToS)
      , context_(a, getOptions().hubble)
      , cosmological_(cosmological)
      , lab_(lab) {}

    double position(Fixed32 const& value) const {
        // Keep every fixed-point bit until the explicit physical-output boundary.
        units::ComovingPosition<Fixed32> const position(value);
        return lab_ ? displayValue(units::toProperLength(position, scales_, context_)) :
                      displayValue(units::toComovingLength(position, scales_));
    }

    double length(double value) const {
        units::StoredLength const length(value);
        return lab_ ? displayValue(units::toProperLength(length, scales_, context_)) :
                      displayValue(units::toComovingLength(length, scales_));
    }

    double velocity(double value) const {
        return displayValue(
            units::toPeculiarVelocity(units::StoredVelocity(value), scales_, context_));
    }

    double density(double value) const {
        units::StoredDensity const density(value);
        return lab_ ? displayValue(units::toProperDensity(density, scales_, context_)) :
                      displayValue(units::toComovingDensity(density, scales_));
    }

    double specificEnergy(double value) const {
        return displayValue(
            units::toPhysicalSpecificEnergy(units::StoredSpecificEnergy(value), scales_, context_));
    }

    double mass(double value) const {
        auto const mass = units::toMass(units::StoredMass(value), scales_);
        return cosmological_ ? units::toCosmologicalValue(mass, context_) : mass.value();
    }
};

template <typename Particle, typename Value>
void writeViewField(DBfile* db, char const* name, char const* mesh,
    Vector<Particle> const& particles, char const* unitName, Value value) {
    Vector<double> values;
    values.reserve(particles.size());
    for (auto const& particle : particles) {
        values.push_back(value(particle));
    }
    auto* options = DBMakeOptlist(1);
    DBAddOption(options, DBOPT_UNITS, const_cast<char*>(unitName));
    checkViewWrite(
        DBPutPointvar1(db, name, mesh, values.data(), values.size(), DB_DOUBLE, options), name);
    DBFreeOptlist(options);
}

template <typename Particle>
void writeViewSpecies(DBfile* db, char const* mesh, char const* prefix,
    Vector<Particle> const& particles, ViewUnitConversion const& conversion, char const* lengthUnit,
    char const* velocityUnit, char const* massUnit, double mass, double time) {
    Vector<double> x, y, z;
    x.reserve(particles.size());
    y.reserve(particles.size());
    z.reserve(particles.size());
    for (auto const& particle : particles) {
        x.push_back(conversion.position(particle.x));
        y.push_back(conversion.position(particle.y));
        z.push_back(conversion.position(particle.z));
    }
    double* coordinates[NDIM] = {x.data(), y.data(), z.data()};
    auto* options = DBMakeOptlist(4);
    DBAddOption(options, DBOPT_XUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_YUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_ZUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_DTIME, &time);
    checkViewWrite(DBPutPointmesh(db, mesh, NDIM, coordinates, x.size(), DB_DOUBLE, options), mesh);
    DBFreeOptlist(options);
    std::string const fieldPrefix(prefix);
    writeViewField(db, (fieldPrefix + "vx").c_str(), mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vx); });
    writeViewField(db, (fieldPrefix + "vy").c_str(), mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vy); });
    writeViewField(db, (fieldPrefix + "vz").c_str(), mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vz); });
    writeViewField(db, (fieldPrefix + "rung").c_str(), mesh, particles, "1",
        [](Particle const& particle) { return double(particle.rung); });
    writeViewField(db, (fieldPrefix + "mass").c_str(), mesh, particles, massUnit,
        [&conversion, mass](Particle const&) { return conversion.mass(mass); });
}

void writeConvertedView(DBfile* db, ViewReturn const& particles, int box, double a, double time,
    bool cosmological, bool lab) {
    char const* lengthUnit = cosmological ? "Mpc/h" : "cm";
    char const* velocityUnit = cosmological ? "km/s" : "cm/s";
    char const* massUnit = cosmological ? "Msun/h" : "g";
    char const* densityUnit = cosmological ? "(Msun/h)/(Mpc/h)^3" : "g/cm^3";
    char const* energyUnit = cosmological ? "(km/s)^2" : "erg/g";
    ViewUnitConversion const conversion(a, cosmological, lab);
    ViewUnitConversion const cgsLab(a, false, true);
    writeViewText(db, "unit_system", cosmological ? "cosmological" : "cgs");
    writeViewText(db, "frame", lab ? "lab" : "comoving");
    writeViewText(db, "position_convention", lab ? "proper" : "comoving");
    writeViewText(db, "density_convention", lab ? "proper" : "comoving");
    writeViewText(db, "velocity_convention", "peculiar");
    writeViewText(db, "hubble_flow", "excluded");
    writeViewText(db, "specific_energy_convention", "physical");
    writeViewText(db, "integration_state", "stored_staggered");
    writeViewText(db, "temporal_convention",
        "Positions are at the snapshot epoch; velocities and internal energy retain the "
        "integrator's stored kick stage; density retains the last SPH density evaluation. "
        "Unit conversion does not synchronize these quantities in time.");
    writeViewText(db, "length_unit", lengthUnit);
    writeViewText(db, "velocity_unit", velocityUnit);
    writeViewText(db, "mass_unit", massUnit);
    writeViewText(db, "density_unit", densityUnit);
    writeViewText(db, "specific_energy_unit", energyUnit);
    writeViewText(db, "time_unit", "Gyr");
    writeViewText(db, "time_convention", "cosmological_age");
    writeViewScalar(db, "time", time);
    writeViewScalar(db, "scale_factor", a);
    writeViewScalar(db, "hubble_parameter", getOptions().hubble);
    writeViewScalar(db, "code_length_cm", getOptions().codeToCm);
    writeViewScalar(db, "code_mass_g", getOptions().codeToG);
    writeViewScalar(db, "code_time_s", getOptions().codeToS);

    if (!particles.dm[box].empty()) {
        double const mass = getOptions().sph ? getOptions().dmMass : 1.0;
        writeViewSpecies(db, "dark_matter", "", particles.dm[box], conversion, lengthUnit,
            velocityUnit, massUnit, mass, time);
    }
    auto const& gas = particles.hydro[box];
    if (!gas.empty()) {
        writeViewSpecies(db, "gas", "hydro_", gas, conversion, lengthUnit, velocityUnit, massUnit,
            getOptions().sphMass, time);
        writeViewField(
            db, "rho", "gas", gas, densityUnit, [&conversion](SphParticleInfo const& particle) {
                return conversion.density(particle.rho);
            });
        writeViewField(
            db, "eint", "gas", gas, energyUnit, [&conversion](SphParticleInfo const& particle) {
                return conversion.specificEnergy(particle.eint);
            });
        writeViewField(db, "kernel_support_radius", "gas", gas, lengthUnit,
            [&conversion](
                SphParticleInfo const& particle) { return conversion.length(particle.h); });
        writeViewField(db, "alpha", "gas", gas, "1",
            [](SphParticleInfo const& particle) { return double(particle.alpha); });
        if (getOptions().stars) {
            writeViewField(db, "cold_frac", "gas", gas, "1",
                [](SphParticleInfo const& particle) { return double(particle.coldFrac); });
        }
        if (getOptions().chem) {
            struct ChemistryField {
                char const* name;
                float SphParticleInfo::* member;
            };
            ChemistryField const fields[] = {{"H", &SphParticleInfo::hValue2},
                {"He", &SphParticleInfo::he}, {"Hp", &SphParticleInfo::hp},
                {"Hn", &SphParticleInfo::hn}, {"H2", &SphParticleInfo::h2},
                {"Hep", &SphParticleInfo::hep}, {"Hepp", &SphParticleInfo::hepp},
                {"Z", &SphParticleInfo::zValue2}};
            for (auto const& field : fields) {
                writeViewField(
                    db, field.name, "gas", gas, "1", [&field](SphParticleInfo const& particle) {
                        return double(particle.*(field.member));
                    });
            }
            // Preserve the existing chemistry model; only centralize its CGS conversion.
            writeViewField(db, "T", "gas", gas, "K", [&cgsLab](SphParticleInfo const& particle) {
                double const helium = particle.he + particle.hep + particle.hepp;
                double const count = particle.hValue2 + 2.f * particle.hp + .5f * particle.h2 +
                    .25f * particle.he + .5f * particle.hep + .75f * particle.hepp +
                    particle.zValue2 / 10.0;
                double const rho = cgsLab.density(particle.rho);
                double const numberDensity = count * constants::kAvo * rho;
                double const heatCapacity =
                    (1.5 + 0.5 * particle.h2 / (1. - .75 * helium - 0.5 * particle.h2)) *
                    double(constants::kB);
                double temperature =
                    cgsLab.specificEnergy(particle.eint) * rho / (numberDensity * heatCapacity);
                if (getOptions().stars) {
                    temperature /= 1.f - particle.coldFrac;
                }
                return temperature;
            });
        }
    }
    if (!particles.star[box].empty()) {
        writeViewSpecies(db, "stars", "star_", particles.star[box], conversion, lengthUnit,
            velocityUnit, massUnit, getOptions().sphMass, time);
        writeViewField(db, "star_Y", "stars", particles.star[box], "1",
            [](StarParticleInfo const& particle) { return double(particle.yValue2); });
        writeViewField(db, "star_Z", "stars", particles.star[box], "1",
            [](StarParticleInfo const& particle) { return double(particle.zValue2); });
        writeViewField(db, "star_M", "stars", particles.star[box], massUnit,
            [&conversion](
                StarParticleInfo const& particle) { return conversion.mass(particle.m); });
        writeViewField(db, "star_zform", "stars", particles.star[box], "1",
            [](StarParticleInfo const& particle) { return double(particle.zform); });
    }
}

void writeConvertedViews(DBfile* db, ViewReturn const& particles, int box, double a) {
    double const time =
        cosmosTime(1e-6, a) * getOptions().codeToS * constants::kSecondsToYears / 1e9;
    for (bool cosmological : {false, true}) {
        char const* systemName = cosmological ? "cosmological" : "cgs";
        if (getOptions().outputUnits != "both" && getOptions().outputUnits != systemName) {
            continue;
        }
        checkViewWrite(DBMkDir(db, systemName), systemName);
        checkViewWrite(DBSetDir(db, systemName), systemName);
        for (bool lab : {false, true}) {
            char const* frameName = lab ? "lab" : "comoving";
            if (getOptions().outputFrame != "both" && getOptions().outputFrame != frameName) {
                continue;
            }
            checkViewWrite(DBMkDir(db, frameName), frameName);
            checkViewWrite(DBSetDir(db, frameName), frameName);
            writeConvertedView(db, particles, box, a, time, cosmological, lab);
            checkViewWrite(DBSetDir(db, ".."), "..");
        }
        checkViewWrite(DBSetDir(db, ".."), "..");
    }
}

}    // namespace detail

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
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, boxes, chem, stars]() {
                ViewReturn rc;
                rc.hydro.resize(boxes.size());
                rc.dm.resize(boxes.size());
                rc.star.resize(boxes.size());
                ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                for (ParticleInteger i = b; i < e; i++) {
                    Array<double, NDIM> x;
                    x[XDIM] = particlesPos(XDIM, i).toDouble();
                    x[YDIM] = particlesPos(YDIM, i).toDouble();
                    x[ZDIM] = particlesPos(ZDIM, i).toDouble();
                    for (int j = 0; j < boxes.size(); j++) {
                        if (boxes[j].contains(x)) {
                            ParticleInteger const k = particlesType(i);
                            switch (k) {
                            case DARK_MATTER_TYPE: {
                                DarkMatterParticleInfo info{};
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
                                SphParticleInfo info{};
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
                                    info.h = sphParticlesSmoothingLength(l);
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
                                    StarParticleInfo info{};
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

void viewOutputViews(int cycle, double a, bool force) {
    if (!getOptions().outputSilo || viewBoxes.empty() ||
        (!force && cycle % getOptions().outputSiloInterval != 0)) {
        return;
    }
    profilerEnter("view_output_views");
    bool const chem = getOptions().chem;
    bool const stars = getOptions().stars;
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
        auto const filename = std::filesystem::path(getOptions().outputDirectory) /
            ("view." + std::to_string(bi) + "." + std::to_string(cycle) + ".silo");
        DBfile* db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_PDB);
        if (db == nullptr) {
            THROW_ERROR("Cannot create particle Silo file %s\n", filename.c_str());
        }
        ViewReturn parts;
        parts = viewGetParticles();
        if (getOptions().outputUnits != "code") {
            detail::writeConvertedViews(db, parts, bi, a);
            detail::checkViewWrite(DBClose(db), filename.c_str());
            continue;
        }
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

void viewConfigure() {
    viewBoxes.clear();
    auto const& opts = getOptions();
    if (!opts.outputSilo) {
        PRINT("Silo particle output: disabled (output.silo=false)\n");
        return;
    }
    std::error_code directoryError;
    std::filesystem::create_directories(opts.outputDirectory, directoryError);
    if (directoryError) {
        THROW_ERROR("Cannot create output.directory %s: %s\n", opts.outputDirectory.c_str(),
            directoryError.message().c_str());
    }
    Range<double> viewBox;
    viewBox.begin[XDIM] = opts.outputXMin;
    viewBox.end[XDIM] = opts.outputXMax;
    viewBox.begin[YDIM] = opts.outputYMin;
    viewBox.end[YDIM] = opts.outputYMax;
    viewBox.begin[ZDIM] = opts.outputZMin;
    viewBox.end[ZDIM] = opts.outputZMax;
    viewBoxes.push_back(viewBox);
    PRINT("Silo particle output: x [%g, %g], y [%g, %g], z [%g, %g], every %d synchronized coarse "
          "steps\n",
        opts.outputXMin, opts.outputXMax, opts.outputYMin, opts.outputYMax, opts.outputZMin,
        opts.outputZMax, opts.outputSiloInterval);
    PRINT("Silo particle output directory: %s\n", opts.outputDirectory.c_str());
}

Vector<float> outputView(int number, double time) {
    profilerEnter("output_view");
    Vector<hpx::future<void>> futs;
    Vector<hpx::future<Vector<float>>> valFuts;
    for (auto const& c : hpxChildren()) {
        valFuts.push_back(hpx::async<OutputViewAction>(c, number, time));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    int const nside = getOptions().viewSize;
    int const npix = 12 * sqr(nside);
    Vector<std::atomic<float>> values(npix);
    for (auto& v : values) {
        v = 0.0f;
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [numberOfThreads, proc, nside, &values]() {
                ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
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
