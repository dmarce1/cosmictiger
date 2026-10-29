#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cosmictiger/analysis/rockstar.hpp>
#include <cosmictiger/analysis/rockstar_properties.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/analysis/bh.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/simulation_units.hpp>

// ROCKSTAR section/equation references: Behroozi, Wechsler & Wu (2013),
// https://arxiv.org/html/1110.4372v3

Float16 rockstarEncodeHalf(double value, char const* fieldName) {
    Float16 encoded;
    try {
        encoded = double2float16(value);
    } catch (std::overflow_error const&) {
        throw std::overflow_error("ROCKSTAR field " + std::string(fieldName) + " = " +
            std::to_string(value) + " exceeds IEEE binary16 finite range [-65504, 65504]");
    }
    if ((encoded.bits & 0x7c00u) == 0x7c00u) {
        throw std::domain_error("Nonfinite ROCKSTAR field " + std::string(fieldName));
    }
    return encoded;
}

double rockstarBoundedScaleRadius(double scaleRadius, double outerRadius) {
    // Production builds use -ffast-math; test finiteness by bits, not comparisons.
    if (!rockstar::detail::isFinite(scaleRadius) || !rockstar::detail::isFinite(outerRadius) ||
        !(scaleRadius > 0.0) || !(scaleRadius <= outerRadius)) {
        return 0.0;
    }
    return scaleRadius;
}

void rockstarWriteCatalogue(
    std::string const& filename, RockstarRecord const* records, size_t count) {
    static_assert(
        std::endian::native == std::endian::little || std::endian::native == std::endian::big);
    auto const& cosmology = getOptions().cosmology;
    auto const convention =
        rockstar::haloDensityConvention(cosmology.omegaLambda, cosmology.omegaK);
    // Preserve the raw record offsets, but never leave an old encoding marker
    // attached to a partial rewrite. Readers require this sidecar for new files.
    std::string const formatPath = filename + ".format";
    std::filesystem::remove(formatPath);
    FILE* output = std::fopen(filename.c_str(), "wb");
    if (output == nullptr) {
        throw std::runtime_error("Cannot write ROCKSTAR catalogue: " + filename);
    }
    size_t const written =
        count == 0 ? 0 : std::fwrite(records, sizeof(RockstarRecord), count, output);
    int const closeResult = std::fclose(output);
    if (written != count || closeResult != 0) {
        throw std::runtime_error("Incomplete ROCKSTAR catalogue: " + filename);
    }
    std::string const temporaryPath = formatPath + ".tmp";
    std::ofstream format(temporaryPath);
    format << "format_version 2\n"
           << "float16 ieee754_binary16\n"
           << "record_size " << sizeof(RockstarRecord) << '\n'
           << "record_count " << count << '\n'
           << "byte_order " << (std::endian::native == std::endian::little ? "little" : "big")
           << '\n'
           << "virial_density " << rockstar::virialDensityLabel(convention) << '\n'
           << "so_density_reference " << rockstar::soDensityReferenceLabel(convention) << '\n';
    format.close();
    if (!format) {
        throw std::runtime_error("Cannot write ROCKSTAR format metadata: " + temporaryPath);
    }
    std::filesystem::rename(temporaryPath, formatPath);
}

namespace detail {

constexpr double kFourPiOverThree = 4.0 * std::numbers::pi / 3.0;

rockstar::HaloDensityThresholds haloOverdensity(double scale) {
    auto const& cosmology = getOptions().cosmology;
    return rockstar::haloDensityThresholds(
        scale, cosmology.omegaM, cosmology.omegaR, cosmology.omegaLambda, cosmology.omegaK);
}

double radiusSquared(RockstarParticle const& particle, Subgroup const& group) {
    double const dx = double(particle.x) - group.x;
    double const dy = double(particle.y) - group.y;
    double const dz = double(particle.z) - group.z;
    return dx * dx + dy * dy + dz * dz;
}

std::vector<double> sortedRadii(DeviceVector<RockstarParticle> const& particles,
    Subgroup const& group, double positionFactor = 1.0) {
    std::vector<double> radii;
    radii.reserve(particles.size());
    for (auto const& particle : particles) {
        radii.push_back(std::sqrt(radiusSquared(particle, group)) / positionFactor);
    }
    std::sort(radii.begin(), radii.end());
    return radii;
}

Array<double, kDimensions> bulkVelocity(DeviceVector<RockstarParticle> const& particles) {
    Array<double, kDimensions> result{};
    if (!particles.empty()) {
        for (auto const& particle : particles) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                result[dim] += particle.xValue[dim + kDimensions] / double(particles.size());
            }
        }
    }
    return result;
}

// Sec. III.5.1: the core-velocity aperture is 0.1 Rvir.
void coreVelocity(Subgroup& group, DeviceVector<RockstarParticle> const& particles, double radius) {
    Array<double, kDimensions> velocity{};
    size_t count = 0;
    double nearest = std::numeric_limits<double>::max();
    size_t nearestIndex = 0;
    for (size_t i = 0; i < particles.size(); ++i) {
        double const r2 = radiusSquared(particles[i], group);
        if (r2 < nearest) {
            nearest = r2;
            nearestIndex = i;
        }
        if (r2 <= 0.01 * radius * radius) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                velocity[dim] += particles[i].xValue[dim + kDimensions];
            }
            ++count;
        }
    }
    // A discrete halo can have no particle inside 0.1 R. Use its nearest
    // particle rather than silently changing the aperture to ten percent by count.
    if (count == 0 && !particles.empty()) {
        count = 1;
        for (int dim = 0; dim < kDimensions; ++dim) {
            velocity[dim] = particles[nearestIndex].xValue[dim + kDimensions];
        }
    }
    group.vxc = count ? velocity[0] / count : 0.0;
    group.vyc = count ? velocity[1] / count : 0.0;
    group.vzc = count ? velocity[2] / count : 0.0;
}

DeviceVector<float> potentials(DeviceVector<RockstarParticle> const& particles) {
    DeviceVector<Array<float, kDimensions>> positions;
    for (auto const& particle : particles) {
        Array<float, kDimensions> position;
        for (int dim = 0; dim < kDimensions; ++dim) {
            position[dim] = particle.xValue[dim];
        }
        positions.pushBack(position);
    }
    return bhEvaluatePotential(positions, getOptions().compute.cuda && particles.size() > 4096);
}

void energies(Subgroup& group, DeviceVector<RockstarParticle> const& particles, double scale) {
    auto const velocity = bulkVelocity(particles);
    auto const phi = potentials(particles);
    double kinetic = 0.0;
    double potential = 0.0;
    for (size_t i = 0; i < particles.size(); ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const dv = particles[i].xValue[dim + kDimensions] - velocity[dim];
            kinetic += 0.5 * dv * dv;
        }
        // Our units: BH returns comoving specific potential; phi/a is physical.
        // The factor 1/2 counts each pair once in the total binding energy.
        potential += 0.5 * phi[i] / scale;
    }
    group.t = kinetic;
    group.w = potential;
}

void rebuildChildren(Vector<Subgroup>& groups) {
    for (auto& group : groups) {
        group.children.clear();
    }
    for (size_t i = 0; i < groups.size(); ++i) {
        if (groups[i].internalParent >= 0) {
            groups[groups[i].internalParent].children.push_back(i);
        }
    }
}

void compactGroups(Vector<Subgroup>& groups, std::vector<bool> const& keep) {
    std::vector<int> map(groups.size(), -1);
    int count = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (keep[i]) {
            map[i] = count++;
        }
    }
    for (size_t i = 0; i < groups.size(); ++i) {
        int parent = groups[i].internalParent;
        while (parent >= 0 && !keep[parent]) {
            parent = groups[parent].internalParent;
        }
        groups[i].parent = parent < 0 ? -1 : map[parent];
    }
    Vector<Subgroup> retained;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (keep[i]) {
            groups[i].internalParent = groups[i].parent;
            retained.push_back(std::move(groups[i]));
        }
    }
    groups = std::move(retained);
    rebuildChildren(groups);
}

// Sec. III.4: associate cores by their largest previous-halo contribution.
std::pair<GroupInteger, GroupInteger> previousCore(Subgroup const& group) {
    std::unordered_map<GroupInteger, size_t> counts;
    std::unordered_map<GroupInteger, GroupInteger> hosts;
    double const aperture = 0.01 * group.rVir * group.rVir;
    for (auto const& particle : group.parts) {
        if (particle.previousHalo != kNoParticleGroup &&
            radiusSquared(particle, group) <= aperture) {
            ++counts[particle.previousHalo];
            hosts[particle.previousHalo] = particle.previousHost;
        }
    }
    if (counts.empty()) {
        double closest = std::numeric_limits<double>::max();
        RockstarParticle const* nearest = nullptr;
        for (auto const& particle : group.parts) {
            double const distance = radiusSquared(particle, group);
            if (particle.previousHalo != kNoParticleGroup && distance < closest) {
                closest = distance;
                nearest = &particle;
            }
        }
        if (nearest) {
            return {nearest->previousHalo, nearest->previousHost};
        }
    }
    GroupInteger winner = kNoParticleGroup;
    size_t most = 0;
    for (auto const& entry : counts) {
        if (entry.second > most || (entry.second == most && entry.first < winner)) {
            winner = entry.first;
            most = entry.second;
        }
    }
    return {winner, winner == kNoParticleGroup ? kNoParticleGroup : hosts[winner]};
}

// Sec. III.4 uses Eq. (3) to choose internal parents before SO masses exist.
void internalHierarchy(Vector<Subgroup>& groups) {
    for (size_t i = 0; i < groups.size(); ++i) {
        double closest = std::numeric_limits<double>::max();
        groups[i].internalParent = -1;
        for (size_t j = 0; j < groups.size(); ++j) {
            if (groups[j].parts.size() < groups[i].parts.size() ||
                (groups[j].parts.size() == groups[i].parts.size() && j >= i)) {
                continue;
            }
            double const dx = double(groups[i].x) - groups[j].x;
            double const dy = double(groups[i].y) - groups[j].y;
            double const dz = double(groups[i].z) - groups[j].z;
            double const dvx = groups[i].vxc - groups[j].vxc;
            double const dvy = groups[i].vyc - groups[j].vyc;
            double const dvz = groups[i].vzc - groups[j].vzc;
            double const distance = (dx * dx + dy * dy + dz * dz) /
                    std::max(1.e-30, double(groups[j].rDyn) * groups[j].rDyn) +
                (dvx * dvx + dvy * dvy + dvz * dvz) / std::max(1.e-30, double(groups[j].sigma2V));
            if (distance < closest) {
                closest = distance;
                groups[i].internalParent = j;
            }
        }
    }
    std::vector<std::pair<GroupInteger, GroupInteger>> previous;
    for (auto const& group : groups) {
        previous.push_back(previousCore(group));
    }
    // Sec. III.4: preserve a previous host's identity when major-merger cores
    // exchange which one has the larger current exclusive particle count.
    for (size_t i = 0; i < groups.size(); ++i) {
        int const parent = groups[i].internalParent;
        if (parent >= 0 && previous[i].first != kNoParticleGroup &&
            previous[parent].second == previous[i].first &&
            previous[parent].first != previous[i].first) {
            groups[i].internalParent = groups[parent].internalParent;
            groups[parent].internalParent = i;
        }
    }
    rebuildChildren(groups);
}

void measureMasses(Vector<Subgroup>& groups, double scale) {
    auto const density = haloOverdensity(scale);
    double const mean = getOptions().initial.numberOfParticles;
    for (size_t i = 0; i < groups.size(); ++i) {
        auto particles = rockstarGatherParticles(groups, i);
        auto const radii = sortedRadii(particles, groups[i]);
        auto const virial = rockstar::sphericalOverdensity(radii, mean * density.virialMeanMatter);
        // Sec. III.5.2 permits critical- or background-density SO thresholds.
        // The catalogue metadata records whether these thresholds reference
        // critical density or the mean-matter convention for other cosmologies.
        auto const mass200 =
            rockstar::sphericalOverdensity(radii, mean * 200.0 * density.soReferenceMeanMatter);
        auto const mass500 =
            rockstar::sphericalOverdensity(radii, mean * 500.0 * density.soReferenceMeanMatter);
        auto const mass2500 =
            rockstar::sphericalOverdensity(radii, mean * 2500.0 * density.soReferenceMeanMatter);
        auto& group = groups[i];
        group.massVir = virial.count;
        group.rVir = virial.radius;
        group.mass200 = mass200.count;
        group.mass500 = mass500.count;
        group.mass2500 = mass2500.count;
        group.radius500 = mass500.radius;
        coreVelocity(group, group.parts, group.rVir);
    }
}

DeviceVector<RockstarParticle> withinRadius(
    DeviceVector<RockstarParticle> const& particles, Subgroup const& group, double radius) {
    DeviceVector<RockstarParticle> selected;
    for (auto const& particle : particles) {
        if (std::sqrt(radiusSquared(particle, group)) <= radius) {
            selected.pushBack(particle);
        }
    }
    return selected;
}

struct Shape {
    double bOverA = 0.0;
    double cOverA = 0.0;
    std::array<double, 3> major{};
};

// Sec. III.5.4, Eq. (11): mass-distribution tensor; callers exclude substructure.
Shape shape(DeviceVector<RockstarParticle> const& particles, Subgroup const& group) {
    Shape result;
    if (particles.empty()) {
        return result;
    }
    std::array<std::array<double, 3>, 3> moment{};
    for (auto const& particle : particles) {
        std::array<double, 3> dx{double(particle.x) - group.x, double(particle.y) - group.y,
            double(particle.z) - group.z};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                moment[row][column] += dx[row] * dx[column] / particles.size();
            }
        }
    }
    auto const eigen = rockstar::symmetricEigenpairs(moment);
    if (eigen.values[0] > 0.0) {
        result.bOverA = std::sqrt(std::max(0.0, eigen.values[1] / eigen.values[0]));
        result.cOverA = std::sqrt(std::max(0.0, eigen.values[2] / eigen.values[0]));
        for (int dim = 0; dim < 3; ++dim) {
            result.major[dim] = eigen.vectors[0][dim] * std::sqrt(eigen.values[0]);
        }
    }
    return result;
}

}    // namespace detail

// Secs. III.4 and III.5.2: host mass includes descendants, not vice versa.
DeviceVector<RockstarParticle> rockstarGatherParticles(Vector<Subgroup> const& groups, int index) {
    auto result = groups[index].parts;
    for (int child : groups[index].children) {
        auto const particles = rockstarGatherParticles(groups, child);
        for (auto const& particle : particles) {
            result.pushBack(particle);
        }
    }
    return result;
}

void rockstarUpdateStatistics(
    Subgroup& group, float positionFactor, float velocityFactor, float scale) {
    if (group.parts.empty()) {
        return;
    }
    auto const velocity = detail::bulkVelocity(group.parts);
    Array<double, kDimensions> center{};
    for (auto const& particle : group.parts) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            center[dim] += particle.xValue[dim] / double(group.parts.size());
        }
    }
    double positionVariance = 0.0;
    for (auto const& particle : group.parts) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const dx = particle.xValue[dim] - center[dim];
            positionVariance += dx * dx / group.parts.size();
        }
    }
    // Sec. III.5.1: retain the center minimizing sigma_x/sqrt(N).
    double const error = std::sqrt(positionVariance / group.parts.size());
    if (error <= group.sigmaXMin) {
        group.sigmaXMin = error;
        group.x = center[0];
        group.y = center[1];
        group.z = center[2];
    }
    group.vxb = velocity[0];
    group.vyb = velocity[1];
    group.vzb = velocity[2];
    auto const density = detail::haloOverdensity(scale);
    auto const radii = detail::sortedRadii(group.parts, group, positionFactor);
    double const numberDensity = getOptions().initial.numberOfParticles * density.virialMeanMatter;
    auto const virial = rockstar::sphericalOverdensity(radii, numberDensity);
    double const radius = virial.count ?
        virial.radius :
        std::cbrt(group.parts.size() / (detail::kFourPiOverThree * numberDensity));
    group.rVir = radius * positionFactor;
    detail::coreVelocity(group, group.parts, group.rVir);
    double velocityVariance = 0.0;
    positionVariance = 0.0;
    for (auto const& particle : group.parts) {
        double const dvx = particle.vx - group.vxc;
        double const dvy = particle.vy - group.vyc;
        double const dvz = particle.vz - group.vzc;
        velocityVariance += (dvx * dvx + dvy * dvy + dvz * dvz) / group.parts.size();
        positionVariance += detail::radiusSquared(particle, group) / group.parts.size();
    }
    // A cold seed has a well-defined center; a finite floor keeps assignment
    // distances defined without changing the velocities themselves.
    group.sigma2V = std::max(velocityVariance, 1.e-30);
    group.sigma2X = std::max(positionVariance, 1.e-30);
    double circular = 0.0;
    for (size_t i = 0; i < radii.size(); ++i) {
        if (radii[i] > 0.0) {
            circular = std::max(
                circular, std::sqrt(getOptions().gravity.gm * (i + 1) / (scale * radii[i])));
        }
    }
    group.vcircMax = circular * velocityFactor;
    // Sec. III.3, Eq. (4), expressed in our comoving coordinate units.
    group.rDyn = std::max(1.e-15,
                     circular /
                         std::sqrt(detail::kFourPiOverThree * getOptions().gravity.gm *
                             numberDensity / scale)) *
        positionFactor;
}

// Sec. III.5.3: single-pass unbinding and the default 50% bound-fraction cut.
bool rockstarHaloBound(Subgroup& halo, Vector<RockstarParticle>& unboundParticles, float scale) {
    auto const phi = detail::potentials(halo.parts);
    size_t const originalCount = halo.parts.size();
    DeviceVector<RockstarParticle> retained;
    for (size_t i = 0; i < halo.parts.size(); ++i) {
        auto const& particle = halo.parts[i];
        double const dx = particle.vx - halo.vxb;
        double const dy = particle.vy - halo.vyb;
        double const dz = particle.vz - halo.vzb;
        if (0.5 * (dx * dx + dy * dy + dz * dz) + phi[i] / scale > 0.0) {
            unboundParticles.push_back(particle);
        } else {
            retained.pushBack(particle);
        }
    }
    bool const accepted = retained.size() >= kRockstarMinGroup &&
        retained.size() >= kRockstarMinBound * originalCount;
    if (!accepted) {
        for (auto const& particle : retained) {
            unboundParticles.push_back(particle);
        }
        retained.resize(0);
    }
    halo.parts = std::move(retained);
    detail::energies(halo, halo.parts, scale);
    return accepted;
}

void rockstarFinalizeSubgroups(Vector<Subgroup>& groups, float scale) {
    if (groups.empty()) {
        return;
    }
    int ordinal = 0;
    for (auto& group : groups) {
        for (auto& particle : group.parts) {
            particle.ordinal = ordinal++;
        }
    }
    detail::internalHierarchy(groups);
    struct BindingSystem {
        std::unordered_map<int, double> potentials;
        Array<double, kDimensions> velocity{};
    };
    std::vector<BindingSystem> systems(groups.size());
    std::vector<int> bindingRoot(groups.size());
    // Sec. III.5.3: freeze potentials for one pass; >=1:3 mergers use the
    // combined potential. Sec. III.5.1 supplies the substructure-inclusive bulk
    // velocity. Particle memberships remain separate within the binding system.
    for (size_t i = 0; i < groups.size(); ++i) {
        int root = i;
        while (groups[root].internalParent >= 0) {
            int const parent = groups[root].internalParent;
            size_t const childCount = groups[root].parts.size();
            size_t const parentCount = groups[parent].parts.size();
            // Temporal host preservation can make the parent the smaller
            // current halo; the major-merger boundary is symmetric in mass.
            if (3 * std::min(childCount, parentCount) < std::max(childCount, parentCount)) {
                break;
            }
            root = parent;
        }
        bindingRoot[i] = root;
        if (systems[root].potentials.empty()) {
            auto const particles = rockstarGatherParticles(groups, root);
            auto const phi = detail::potentials(particles);
            systems[root].velocity = detail::bulkVelocity(particles);
            for (size_t j = 0; j < particles.size(); ++j) {
                systems[root].potentials[particles[j].ordinal] = phi[j] / scale;
            }
        }
    }
    std::vector<bool> keep(groups.size(), false);
    std::function<Vector<RockstarParticle>(int)> unbind = [&](int index) {
        auto& group = groups[index];
        for (int child : group.children) {
            auto const unbound = unbind(child);
            for (auto const& particle : unbound) {
                group.parts.pushBack(particle);
            }
        }
        auto const& system = systems[bindingRoot[index]];
        size_t const originalCount = group.parts.size();
        DeviceVector<RockstarParticle> retained;
        Vector<RockstarParticle> removed;
        for (auto const& particle : group.parts) {
            double kinetic = 0.0;
            for (int dim = 0; dim < kDimensions; ++dim) {
                double const dv = particle.xValue[dim + kDimensions] - system.velocity[dim];
                kinetic += 0.5 * dv * dv;
            }
            if (kinetic + system.potentials.at(particle.ordinal) <= 0.0) {
                retained.pushBack(particle);
            } else {
                removed.push_back(particle);
            }
        }
        keep[index] = retained.size() >= kRockstarMinGroup &&
            retained.size() >= kRockstarMinBound * originalCount;
        if (!keep[index]) {
            for (auto const& particle : retained) {
                removed.push_back(particle);
            }
            retained.resize(0);
        }
        group.parts = std::move(retained);
        return removed;
    };
    for (size_t i = 0; i < groups.size(); ++i) {
        if (groups[i].internalParent == -1) {
            unbind(i);
        }
    }
    detail::compactGroups(groups, keep);
    rockstarPruneCatalogueGroups(groups, scale);
}

void rockstarPruneCatalogueGroups(Vector<Subgroup>& groups, float scale) {
    auto const refresh = [&groups, scale]() {
        for (auto& group : groups) {
            rockstarUpdateStatistics(group, 1.0, 1.0, scale);
            group.catalogueId = kNoParticleGroup;
            for (auto const& particle : group.parts) {
                group.catalogueId = std::min(group.catalogueId, particle.identity);
            }
        }
        detail::measureMasses(groups, scale);
    };
    refresh();
    while (!groups.empty()) {
        std::vector<bool> keep;
        bool removed = false;
        for (auto const& group : groups) {
            keep.push_back(group.massVir >= kRockstarMinGroup);
            removed = removed || !keep.back();
        }
        if (!removed) {
            break;
        }
        // Rejecting an unresolved subhalo removes its catalogue entry, not
        // its bound particles from an enclosing retained host. Donate only
        // exclusive particles; surviving descendants retain their ownership.
        for (size_t i = 0; i < groups.size(); ++i) {
            if (keep[i]) {
                continue;
            }
            int parent = groups[i].internalParent;
            while (parent >= 0 && !keep[parent]) {
                parent = groups[parent].internalParent;
            }
            if (parent >= 0) {
                for (auto const& particle : groups[i].parts) {
                    groups[parent].parts.pushBack(particle);
                }
            }
        }
        detail::compactGroups(groups, keep);
        // Donation can improve a center estimate and change the SO boundary.
        // Recheck until stable; every pass removes at least one candidate.
        refresh();
    }
    for (auto& group : groups) {
        detail::energies(group, group.parts, scale);
    }
    // Sec. III.4: output host relations use SO containment; internal children remain intact
    // for inclusive mass measurements and are deliberately separate.
    for (size_t i = 0; i < groups.size(); ++i) {
        groups[i].parent = -1;
        double closest = std::numeric_limits<double>::max();
        for (size_t j = 0; j < groups.size(); ++j) {
            if (groups[j].massVir <= groups[i].massVir) {
                continue;
            }
            double const dx = double(groups[i].x) - groups[j].x;
            double const dy = double(groups[i].y) - groups[j].y;
            double const dz = double(groups[i].z) - groups[j].z;
            double const r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < groups[j].rVir * groups[j].rVir && r2 < closest) {
                closest = r2;
                groups[i].parent = j;
            }
        }
    }
}

RockstarRecord rockstarMakeRecord(Vector<Subgroup> const& groups, int index, float scale,
    Array<double, kDimensions> const& origin, bool inclusive) {
    Subgroup const& group = groups[index];
    auto candidates = inclusive ? rockstarGatherParticles(groups, index) : group.parts;
    auto const density = detail::haloOverdensity(scale);
    double const mean = getOptions().initial.numberOfParticles;
    auto const allRadii = detail::sortedRadii(candidates, group);
    auto const virial = rockstar::sphericalOverdensity(allRadii, mean * density.virialMeanMatter);
    auto const so200 =
        rockstar::sphericalOverdensity(allRadii, mean * 200.0 * density.soReferenceMeanMatter);
    auto const so500 =
        rockstar::sphericalOverdensity(allRadii, mean * 500.0 * density.soReferenceMeanMatter);
    auto const so2500 =
        rockstar::sphericalOverdensity(allRadii, mean * 2500.0 * density.soReferenceMeanMatter);
    auto const particles = detail::withinRadius(candidates, group, virial.radius);
    auto const radii = detail::sortedRadii(particles, group);
    Subgroup measured;
    measured.x = group.x;
    measured.y = group.y;
    measured.z = group.z;
    detail::coreVelocity(measured, group.parts, virial.radius);
    detail::energies(measured, particles, scale);
    auto const bulk = detail::bulkVelocity(particles);
    double sigmaX = 0.0;
    double sigmaCore = 0.0;
    double sigmaBulk = 0.0;
    Array<double, kDimensions> center{};
    Array<double, kDimensions> angular{};
    for (auto const& particle : particles) {
        double const dx = double(particle.x) - group.x;
        double const dy = double(particle.y) - group.y;
        double const dz = double(particle.z) - group.z;
        double const dvx = particle.vx - bulk[0];
        double const dvy = particle.vy - bulk[1];
        double const dvz = particle.vz - bulk[2];
        angular[0] += scale * (dy * dvz - dz * dvy);
        angular[1] += scale * (dz * dvx - dx * dvz);
        angular[2] += scale * (dx * dvy - dy * dvx);
        sigmaX += (dx * dx + dy * dy + dz * dz) / particles.size();
        sigmaBulk += (dvx * dvx + dvy * dvy + dvz * dvz) / particles.size();
        double const cx = particle.vx - measured.vxc;
        double const cy = particle.vy - measured.vyc;
        double const cz = particle.vz - measured.vzc;
        sigmaCore += (cx * cx + cy * cy + cz * cz) / particles.size();
        for (int dim = 0; dim < kDimensions; ++dim) {
            center[dim] += particle.xValue[dim] / double(particles.size());
        }
    }
    // Sec. III.5.4: Vmax from the enclosed radial mass profile.
    double circular = 0.0;
    double circularRadius = 0.0;
    for (size_t i = 0; i < radii.size();) {
        size_t end = i + 1;
        while (end < radii.size() && radii[end] == radii[i]) {
            ++end;
        }
        if (radii[i] > 0.0) {
            double const speed = std::sqrt(getOptions().gravity.gm * end / (scale * radii[i]));
            if (speed > circular) {
                circular = speed;
                circularRadius = radii[i];
            }
        }
        i = end;
    }
    double const outerRadius = radii.empty() ? 0.0 : radii.back();
    double rs = 0.0;
    double rho = 0.0;
    if (radii.size() >= kRockstarMinGroup && outerRadius > 0.0) {
        Vector<double> fitRadii(radii.begin(), radii.end());
        cosmosNfwFit(fitRadii, rs, rho);
    }
    // Reference ROCKSTAR (nfw.c, calc_scale_from_bins) keeps Rs inside the
    // binned profile; this project stores a scale radius outside the fitted
    // profile as the unresolved value zero. The fit already guarantees
    // rs <= rmax; the guard makes it structural, so encoding rs can only fail
    // when encoding rmax already has.
    rs = rockstarBoundedScaleRadius(rs, outerRadius);
    double const klypin = circular > 0.0 && virial.count && virial.radius > 0.0 ?
        cosmosKlypinFit(circular, scale * virial.radius, virial.count) / scale :
        0.0;
    auto const haloShape =
        detail::shape(detail::withinRadius(group.parts, group, virial.radius), group);
    auto const shape500 =
        detail::shape(detail::withinRadius(group.parts, group, so500.radius), group);
    auto const units = units::scalesFromOptions(getOptions());
    double const massConversion =
        (units.massScale() * getOptions().cosmology.hubble / units::kSolarMass).value();
    double const positionConversion =
        (units.lengthScale() * getOptions().cosmology.hubble / units::kMegaparsec).value();
    double const lengthConversion = 1000.0 * positionConversion;
    double const velocityConversion =
        (units.velocityScale() / units::Velocity::from_value(1.e5)).value();
    double const angularConversion = massConversion * positionConversion * velocityConversion;
    double const energy = measured.t + measured.w;
    double const angularMagnitude =
        std::sqrt(angular[0] * angular[0] + angular[1] * angular[1] + angular[2] * angular[2]);
    double const mass = virial.count;
    double const gm = getOptions().gravity.gm;
    RockstarRecord record{};
    ALWAYS_ASSERT(index >= 0 && index < 32768);
    record.id.haloId = index;
    record.id.sys = inclusive && !group.children.empty();
    record.pid = group.parent;
    record.x = (origin[0] + group.x) * positionConversion;
    record.y = (origin[1] + group.y) * positionConversion;
    record.z = (origin[2] + group.z) * positionConversion;
    record.mvir = mass * massConversion;
    record.m200 = so200.count * massConversion;
    record.m500 = so500.count * massConversion;
    record.m2500 = so2500.count * massConversion;
    record.e = energy * massConversion * velocityConversion * velocityConversion;
    record.jx = angular[0] * angularConversion;
    record.jy = angular[1] * angularConversion;
    record.jz = angular[2] * angularConversion;
    record.vxc = rockstarEncodeHalf(measured.vxc * velocityConversion, "vxc");
    record.vyc = rockstarEncodeHalf(measured.vyc * velocityConversion, "vyc");
    record.vzc = rockstarEncodeHalf(measured.vzc * velocityConversion, "vzc");
    record.vxb = rockstarEncodeHalf(bulk[0] * velocityConversion, "vxb");
    record.vyb = rockstarEncodeHalf(bulk[1] * velocityConversion, "vyb");
    record.vzb = rockstarEncodeHalf(bulk[2] * velocityConversion, "vzb");
    record.sigV = rockstarEncodeHalf(std::sqrt(sigmaCore) * velocityConversion, "sigV");
    record.sigVb = rockstarEncodeHalf(std::sqrt(sigmaBulk) * velocityConversion, "sigVb");
    record.sigX = rockstarEncodeHalf(std::sqrt(sigmaX) * lengthConversion, "sigX");
    record.rvir = rockstarEncodeHalf(virial.radius * lengthConversion, "rvir");
    record.rmax = rockstarEncodeHalf(outerRadius * lengthConversion, "rmax");
    record.rvmax = rockstarEncodeHalf(circularRadius * lengthConversion, "rvmax");
    record.rs = rockstarEncodeHalf(rs * lengthConversion, "rs");
    record.rkly = rockstarEncodeHalf(klypin * lengthConversion, "rkly");
    record.vcircMax = rockstarEncodeHalf(circular * velocityConversion, "vcircMax");
    // Sec. III.5.4: peak/COM and core/bulk offsets, followed by T/|U|.
    record.deltaX = rockstarEncodeHalf(
        std::sqrt(std::pow(center[0] - group.x, 2) + std::pow(center[1] - group.y, 2) +
            std::pow(center[2] - group.z, 2)) *
            lengthConversion,
        "deltaX");
    record.deltaV = rockstarEncodeHalf(
        std::sqrt(std::pow(bulk[0] - measured.vxc, 2) + std::pow(bulk[1] - measured.vyc, 2) +
            std::pow(bulk[2] - measured.vzc, 2)) *
            velocityConversion,
        "deltaV");
    record.toW = rockstarEncodeHalf(measured.w < 0.0 ? measured.t / -measured.w : 0.0, "toW");
    // Sec. III.5.4, Eq. (9): Peebles spin; our sums use unit particle masses.
    record.lambda = rockstarEncodeHalf(mass > 0.0 ?
            angularMagnitude * std::sqrt(std::abs(energy)) / (gm * std::pow(mass, 2.5)) :
            0.0,
        "lambda");
    // Eq. (10): Bullock spin, with physical Rvir = scale * comoving Rvir.
    record.lambdaB = rockstarEncodeHalf(mass > 0.0 && virial.radius > 0.0 ?
            angularMagnitude / std::sqrt(2.0 * gm * mass * mass * mass * scale * virial.radius) :
            0.0,
        "lambdaB");
    record.boa = rockstarEncodeHalf(haloShape.bOverA, "boa");
    record.coa = rockstarEncodeHalf(haloShape.cOverA, "coa");
    record.ax = rockstarEncodeHalf(haloShape.major[0] * lengthConversion, "ax");
    record.ay = rockstarEncodeHalf(haloShape.major[1] * lengthConversion, "ay");
    record.az = rockstarEncodeHalf(haloShape.major[2] * lengthConversion, "az");
    record.boa500 = rockstarEncodeHalf(shape500.bOverA, "boa500");
    record.coa500 = rockstarEncodeHalf(shape500.cOverA, "coa500");
    record.ax500 = rockstarEncodeHalf(shape500.major[0] * lengthConversion, "ax500");
    record.ay500 = rockstarEncodeHalf(shape500.major[1] * lengthConversion, "ay500");
    record.az500 = rockstarEncodeHalf(shape500.major[2] * lengthConversion, "az500");
    record.zValue = rockstarEncodeHalf(1.0 / scale - 1.0, "zValue");
    return record;
}

GroupEntry rockstarMakeGroupEntry(Vector<Subgroup> const& groups, int index, float scale,
    Array<double, kDimensions> const& origin) {
    auto const& group = groups[index];
    auto const allParticles = rockstarGatherParticles(groups, index);
    auto const candidates = detail::sortedRadii(allParticles, group);
    auto const density = detail::haloOverdensity(scale);
    auto const virial = rockstar::sphericalOverdensity(
        candidates, getOptions().initial.numberOfParticles * density.virialMeanMatter);
    auto const particles = detail::withinRadius(allParticles, group, virial.radius);
    auto const radii = detail::sortedRadii(particles, group);
    auto const bulk = detail::bulkVelocity(particles);
    GroupEntry entry{};
    entry.id = group.catalogueId;
    entry.mass = particles.size() * getOptions().units.massG / constants::kM0;
    entry.incomplete = false;
    Subgroup measured;
    detail::energies(measured, particles, scale);
    double const count = std::max(1, particles.size());
    Array<double, kDimensions> center{};
    for (auto const& particle : particles) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            center[dim] += particle.xValue[dim] / count;
        }
    }
    entry.ekin = measured.t / count;
    entry.epot = measured.w / count;
    for (int dim = 0; dim < kDimensions; ++dim) {
        double const position = origin[dim] + center[dim];
        entry.com[dim] = position - std::floor(position);
        entry.vel[dim] = bulk[dim];
    }
    for (auto const& particle : particles) {
        double const dx = double(particle.x) - group.x;
        double const dy = double(particle.y) - group.y;
        double const dz = double(particle.z) - group.z;
        double const vx = particle.vx - bulk[0];
        double const vy = particle.vy - bulk[1];
        double const vz = particle.vz - bulk[2];
        entry.lang[0] += scale * (dy * vz - dz * vy) / count;
        entry.lang[1] += scale * (dz * vx - dx * vz) / count;
        entry.lang[2] += scale * (dx * vy - dy * vx) / count;
        // Preserve the legacy GroupEntry center-of-mass inertia convention.
        double const cx = particle.x - center[0];
        double const cy = particle.y - center[1];
        double const cz = particle.z - center[2];
        entry.ixx += cy * cy + cz * cz;
        entry.iyy += cx * cx + cz * cz;
        entry.izz += cx * cx + cy * cy;
        entry.ixy += cx * cy;
        entry.ixz += cx * cz;
        entry.iyz += cy * cz;
        entry.vxdisp += vx * vx / count;
        entry.vydisp += vy * vy / count;
        entry.vzdisp += vz * vz / count;
        entry.ravg += std::sqrt(dx * dx + dy * dy + dz * dz) / count;
    }
    entry.vxdisp = std::sqrt(entry.vxdisp);
    entry.vydisp = std::sqrt(entry.vydisp);
    entry.vzdisp = std::sqrt(entry.vzdisp);
    auto const percentile = [&radii](double fraction) {
        if (radii.empty()) {
            return 0.0;
        }
        double const coordinate = fraction * (radii.size() - 1);
        size_t const lo = coordinate;
        size_t const hi = std::min(lo + 1, radii.size() - 1);
        return radii[lo] + (coordinate - lo) * (radii[hi] - radii[lo]);
    };
    entry.r25 = percentile(0.25);
    entry.r50 = percentile(0.5);
    entry.r75 = percentile(0.75);
    entry.r90 = percentile(0.9);
    entry.rmax = radii.empty() ? 0.0 : radii.back();
    // Sec. III.5.4: exclusive overlaps supply the particle-based merger tree links.
    for (auto const& particle : group.parts) {
        if (particle.previousHalo != kNoParticleGroup) {
            ++entry.parents[particle.previousHalo];
        }
    }
    return entry;
}
