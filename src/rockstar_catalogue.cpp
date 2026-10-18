#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <cosmictiger/bh.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/rockstar_properties.hpp>
#include <cosmictiger/simulation_units.hpp>

namespace detail {

constexpr double kFourPiOverThree = 4.0 * M_PI / 3.0;

rockstar::VirialOverdensity haloOverdensity(double scale) {
    auto const& cosmology = getOptions().cosmology;
    double const matter = cosmology.omegaM / (scale * scale * scale);
    double const expansion = matter + cosmology.omegaR / std::pow(scale, 4) +
        cosmology.omegaK / (scale * scale) + cosmology.omegaLambda;
    return rockstar::virialOverdensityFromMatterFraction(
        matter / expansion, cosmology.omegaLambda == 0.0 && cosmology.omegaK > 0.0);
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

Array<double, NDIM> bulkVelocity(DeviceVector<RockstarParticle> const& particles) {
    Array<double, NDIM> result{};
    if (!particles.empty()) {
        for (auto const& particle : particles) {
            for (int dim = 0; dim < NDIM; ++dim) {
                result[dim] += particle.xValue[dim + NDIM] / double(particles.size());
            }
        }
    }
    return result;
}

void coreVelocity(Subgroup& group, DeviceVector<RockstarParticle> const& particles, double radius) {
    Array<double, NDIM> velocity{};
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
            for (int dim = 0; dim < NDIM; ++dim) {
                velocity[dim] += particles[i].xValue[dim + NDIM];
            }
            ++count;
        }
    }
    // A discrete halo can have no particle inside 0.1 R. Use its nearest
    // particle rather than silently changing the aperture to ten percent by count.
    if (count == 0 && !particles.empty()) {
        count = 1;
        for (int dim = 0; dim < NDIM; ++dim) {
            velocity[dim] = particles[nearestIndex].xValue[dim + NDIM];
        }
    }
    group.vxc = count ? velocity[0] / count : 0.0;
    group.vyc = count ? velocity[1] / count : 0.0;
    group.vzc = count ? velocity[2] / count : 0.0;
}

DeviceVector<float> potentials(DeviceVector<RockstarParticle> const& particles) {
    DeviceVector<Array<float, NDIM>> positions;
    for (auto const& particle : particles) {
        Array<float, NDIM> position;
        for (int dim = 0; dim < NDIM; ++dim) {
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
        for (int dim = 0; dim < NDIM; ++dim) {
            double const dv = particles[i].xValue[dim + NDIM] - velocity[dim];
            kinetic += 0.5 * dv * dv;
        }
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

std::pair<GroupInteger, GroupInteger> previousCore(Subgroup const& group) {
    std::unordered_map<GroupInteger, size_t> counts;
    std::unordered_map<GroupInteger, GroupInteger> hosts;
    double const aperture = 0.01 * group.rVir * group.rVir;
    for (auto const& particle : group.parts) {
        if (particle.previousHalo != NO_GROUP && radiusSquared(particle, group) <= aperture) {
            ++counts[particle.previousHalo];
            hosts[particle.previousHalo] = particle.previousHost;
        }
    }
    if (counts.empty()) {
        double closest = std::numeric_limits<double>::max();
        RockstarParticle const* nearest = nullptr;
        for (auto const& particle : group.parts) {
            double const distance = radiusSquared(particle, group);
            if (particle.previousHalo != NO_GROUP && distance < closest) {
                closest = distance;
                nearest = &particle;
            }
        }
        if (nearest) {
            return {nearest->previousHalo, nearest->previousHost};
        }
    }
    GroupInteger winner = NO_GROUP;
    size_t most = 0;
    for (auto const& entry : counts) {
        if (entry.second > most || (entry.second == most && entry.first < winner)) {
            winner = entry.first;
            most = entry.second;
        }
    }
    return {winner, winner == NO_GROUP ? NO_GROUP : hosts[winner]};
}

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
    // Preserve a previous host's identity when overlapping major-merger cores
    // exchange which one has the larger current exclusive particle count.
    for (size_t i = 0; i < groups.size(); ++i) {
        int const parent = groups[i].internalParent;
        if (parent >= 0 && previous[i].first != NO_GROUP &&
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
        auto const virial = rockstar::sphericalOverdensity(radii, mean * density.meanMatter);
        // Public m200/m500/m2500 fields consistently mean critical-density SO masses.
        auto const mass200 =
            rockstar::sphericalOverdensity(radii, mean * 200.0 / density.matterFraction);
        auto const mass500 =
            rockstar::sphericalOverdensity(radii, mean * 500.0 / density.matterFraction);
        auto const mass2500 =
            rockstar::sphericalOverdensity(radii, mean * 2500.0 / density.matterFraction);
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
    Array<double, NDIM> center{};
    for (auto const& particle : group.parts) {
        for (int dim = 0; dim < NDIM; ++dim) {
            center[dim] += particle.xValue[dim] / double(group.parts.size());
        }
    }
    double positionVariance = 0.0;
    for (auto const& particle : group.parts) {
        for (int dim = 0; dim < NDIM; ++dim) {
            double const dx = particle.xValue[dim] - center[dim];
            positionVariance += dx * dx / group.parts.size();
        }
    }
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
    double const numberDensity = getOptions().initial.numberOfParticles * density.meanMatter;
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
    group.rDyn = std::max(1.e-15,
                     circular /
                         std::sqrt(detail::kFourPiOverThree * getOptions().gravity.gm *
                             numberDensity / scale)) *
        positionFactor;
}

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
    bool const accepted = retained.size() >= ROCKSTAR_MIN_GROUP &&
        retained.size() >= ROCKSTAR_MIN_BOUND * originalCount;
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
        Array<double, NDIM> velocity{};
    };
    std::vector<BindingSystem> systems(groups.size());
    std::vector<int> bindingRoot(groups.size());
    // Freeze pre-unbinding potentials. A major merger is tested in the combined
    // system, including its bulk velocity, while membership remains separate.
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
            for (int dim = 0; dim < NDIM; ++dim) {
                double const dv = particle.xValue[dim + NDIM] - system.velocity[dim];
                kinetic += 0.5 * dv * dv;
            }
            if (kinetic + system.potentials.at(particle.ordinal) <= 0.0) {
                retained.pushBack(particle);
            } else {
                removed.push_back(particle);
            }
        }
        keep[index] = retained.size() >= ROCKSTAR_MIN_GROUP &&
            retained.size() >= ROCKSTAR_MIN_BOUND * originalCount;
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
            group.catalogueId = NO_GROUP;
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
            keep.push_back(group.massVir >= ROCKSTAR_MIN_GROUP);
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
    // Output host relations use SO containment; internal children remain intact
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
    Array<double, NDIM> const& origin, bool inclusive) {
    Subgroup const& group = groups[index];
    auto candidates = inclusive ? rockstarGatherParticles(groups, index) : group.parts;
    auto const density = detail::haloOverdensity(scale);
    double const mean = getOptions().initial.numberOfParticles;
    auto const allRadii = detail::sortedRadii(candidates, group);
    auto const virial = rockstar::sphericalOverdensity(allRadii, mean * density.meanMatter);
    auto const so200 =
        rockstar::sphericalOverdensity(allRadii, mean * 200.0 / density.matterFraction);
    auto const so500 =
        rockstar::sphericalOverdensity(allRadii, mean * 500.0 / density.matterFraction);
    auto const so2500 =
        rockstar::sphericalOverdensity(allRadii, mean * 2500.0 / density.matterFraction);
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
    Array<double, NDIM> center{};
    Array<double, NDIM> angular{};
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
        for (int dim = 0; dim < NDIM; ++dim) {
            center[dim] += particle.xValue[dim] / double(particles.size());
        }
    }
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
    double rs = 0.0;
    double rho = 0.0;
    if (radii.size() >= ROCKSTAR_MIN_GROUP && radii.back() > 0.0) {
        Vector<double> fitRadii(radii.begin(), radii.end());
        cosmosNfwFit(fitRadii, rs, rho);
    }
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
    record.vxc = double2float16(measured.vxc * velocityConversion);
    record.vyc = double2float16(measured.vyc * velocityConversion);
    record.vzc = double2float16(measured.vzc * velocityConversion);
    record.vxb = double2float16(bulk[0] * velocityConversion);
    record.vyb = double2float16(bulk[1] * velocityConversion);
    record.vzb = double2float16(bulk[2] * velocityConversion);
    record.sigV = double2float16(std::sqrt(sigmaCore) * velocityConversion);
    record.sigVb = double2float16(std::sqrt(sigmaBulk) * velocityConversion);
    record.sigX = double2float16(std::sqrt(sigmaX) * lengthConversion);
    record.rvir = double2float16(virial.radius * lengthConversion);
    record.rmax = double2float16((radii.empty() ? 0.0 : radii.back()) * lengthConversion);
    record.rvmax = double2float16(circularRadius * lengthConversion);
    record.rs = double2float16(rs * lengthConversion);
    record.rkly = double2float16(klypin * lengthConversion);
    record.vcircMax = double2float16(circular * velocityConversion);
    record.deltaX =
        double2float16(std::sqrt(std::pow(center[0] - group.x, 2) +
                           std::pow(center[1] - group.y, 2) + std::pow(center[2] - group.z, 2)) *
            lengthConversion);
    record.deltaV = double2float16(
        std::sqrt(std::pow(bulk[0] - measured.vxc, 2) + std::pow(bulk[1] - measured.vyc, 2) +
            std::pow(bulk[2] - measured.vzc, 2)) *
        velocityConversion);
    record.toW = double2float16(measured.w < 0.0 ? measured.t / -measured.w : 0.0);
    record.lambda = double2float16(mass > 0.0 ?
            angularMagnitude * std::sqrt(std::abs(energy)) / (gm * std::pow(mass, 2.5)) :
            0.0);
    record.lambdaB = double2float16(mass > 0.0 && virial.radius > 0.0 ?
            angularMagnitude / std::sqrt(2.0 * gm * mass * mass * mass * scale * virial.radius) :
            0.0);
    record.boa = double2float16(haloShape.bOverA);
    record.coa = double2float16(haloShape.cOverA);
    record.ax = double2float16(haloShape.major[0] * lengthConversion);
    record.ay = double2float16(haloShape.major[1] * lengthConversion);
    record.az = double2float16(haloShape.major[2] * lengthConversion);
    record.boa500 = double2float16(shape500.bOverA);
    record.coa500 = double2float16(shape500.cOverA);
    record.ax500 = double2float16(shape500.major[0] * lengthConversion);
    record.ay500 = double2float16(shape500.major[1] * lengthConversion);
    record.az500 = double2float16(shape500.major[2] * lengthConversion);
    record.zValue = double2float16(1.0 / scale - 1.0);
    return record;
}

GroupEntry rockstarMakeGroupEntry(
    Vector<Subgroup> const& groups, int index, float scale, Array<double, NDIM> const& origin) {
    auto const& group = groups[index];
    auto const allParticles = rockstarGatherParticles(groups, index);
    auto const candidates = detail::sortedRadii(allParticles, group);
    auto const density = detail::haloOverdensity(scale);
    auto const virial = rockstar::sphericalOverdensity(
        candidates, getOptions().initial.numberOfParticles * density.meanMatter);
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
    Array<double, NDIM> center{};
    for (auto const& particle : particles) {
        for (int dim = 0; dim < NDIM; ++dim) {
            center[dim] += particle.xValue[dim] / count;
        }
    }
    entry.ekin = measured.t / count;
    entry.epot = measured.w / count;
    for (int dim = 0; dim < NDIM; ++dim) {
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
    // Exclusive particle overlaps provide the particle-based merger tree links.
    for (auto const& particle : group.parts) {
        if (particle.previousHalo != NO_GROUP) {
            ++entry.parents[particle.previousHalo];
        }
    }
    return entry;
}
