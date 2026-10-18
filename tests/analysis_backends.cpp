/* CosmicTiger - CPU analysis integration regression; GPL-2.0-or-later. */
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include <healpix/healpix_base.h>

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>

namespace detail {

void requireAnalysis(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool analysisFinite(float value) {
    return (std::bit_cast<uint32_t>(value) & 0x7f800000U) != 0x7f800000U;
}

int analysisPixelOwner(int pixel, int pixels) {
    int rank = 0;
    while (pixel >= (rank + 1) * pixels / hpxSize()) {
        ++rank;
    }
    return rank;
}

LightconeParticle analysisLightconeParticle(
    int identity, double radius, double angle, double height = 0.0) {
    LightconeParticle particle{};
    particle.pos[XDIM] = radius * std::cos(angle);
    particle.pos[YDIM] = radius * std::sin(angle);
    particle.pos[ZDIM] = height;
    // Velocities carry exact identities through tree permutation and routing.
    particle.vel[XDIM] = identity;
    particle.group = LC_NO_GROUP;
    return particle;
}

Vector<LightconeParticle> analysisLightconeFixture(int nside) {
    double angle = 0.25;
    if (hpxSize() > 1) {
        requireAnalysis(nside > 0, "Distributed lightcone fixture needs multiple pixels");
        T_Healpix_Base<int> pixels(nside, NEST, SET_NSIDE);
        int const pixelCount = 12 * nside * nside;
        auto const owner = [&](double azimuth) {
            vec3 direction;
            direction.x = std::cos(azimuth);
            direction.y = std::sin(azimuth);
            direction.z = 0.0;
            return analysisPixelOwner(pixels.vec2pix(direction), pixelCount);
        };
        constexpr int kSamples = 4096;
        double const increment = 2.0 * M_PI / kSamples;
        bool foundBoundary = false;
        for (int i = 1; i < kSamples; ++i) {
            if (owner((i - 1) * increment) != owner(i * increment)) {
                angle = (i - 0.5) * increment;
                foundBoundary = true;
                break;
            }
        }
        requireAnalysis(foundBoundary, "No cross-rank lightcone pixel boundary found");
        requireAnalysis(owner(angle - 0.003) != owner(angle + 0.003),
            "Linked lightcone fixture did not straddle a rank boundary");
    }
    Vector<LightconeParticle> particles;
    for (int i = 0; i < 6; ++i) {
        double const offset = i < 3 ? -0.003 : 0.003;
        particles.push_back(analysisLightconeParticle(
            i, 0.55 + 0.0001 * (i % 3), angle + offset, 0.0001 * (i % 3)));
    }
    for (int i = 6; i < 10; ++i) {
        particles.push_back(
            analysisLightconeParticle(i, 0.55, angle + M_PI, 0.08 + 0.0002 * (i - 6)));
    }
    for (int i = 10; i < 12; ++i) {
        auto particle = analysisLightconeParticle(i, 0.002 * (i - 10), 0.0, 0.8);
        particles.push_back(particle);
    }
    particles.push_back(analysisLightconeParticle(12, 0.0, 0.0, -0.4));
    return particles;
}

}    // namespace detail

void analysisPowerSetup(bool perturb) {
    int const side = getOptions().initial.particlesPerDimension;
    int const count = side * side * side;
    int const begin = hpxRank() * count / hpxSize();
    int const end = (hpxRank() + 1) * count / hpxSize();
    particlesResize(end - begin);
    for (int global = begin; global < end; ++global) {
        int const local = global - begin;
        double const x = (global / (side * side) + 0.5) / side;
        particlesPos(XDIM, local) = x + (perturb ? 0.03 * std::sin(2.0 * M_PI * x) : 0.0);
        particlesPos(YDIM, local) = ((global / side) % side + 0.5) / side;
        particlesPos(ZDIM, local) = (global % side + 0.5) / side;
        for (int dim = 0; dim < NDIM; ++dim) {
            particlesVel(dim, local) = 0.0f;
        }
    }
}
HPX_PLAIN_ACTION(analysisPowerSetup, AnalysisPowerSetupAction);

double analysisDensityMass(int fold) {
    int const side = getOptions().initial.fourierSize;
    Range<int64_t> box;
    for (int dim = 0; dim < NDIM; ++dim) {
        box.begin[dim] = CLOUD_MIN;
        box.end[dim] = side + CLOUD_MAX;
    }
    auto const density = accumulateDensityCpu(fold, side, box);
    double mass = 0.0;
    for (auto const value : density) {
        mass += value;
    }
    return mass;
}
HPX_PLAIN_ACTION(analysisDensityMass, AnalysisDensityMassAction);

void analysisAddLightconeParticles(Vector<LightconeParticle> particles) {
    std::filesystem::create_directories(getOptions().lightcone.directory);
    lcAddParticles(std::move(particles));
}
HPX_PLAIN_ACTION(analysisAddLightconeParticles, AnalysisAddLightconeParticlesAction);

Vector<LightconeParticle> analysisReadLightconeBuffer() {
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::tmpfile(), &std::fclose);
    detail::requireAnalysis(file != nullptr, "Cannot open temporary lightcone checkpoint");
    lcSave(file.get());
    std::rewind(file.get());
    size_t count = 0;
    detail::requireAnalysis(std::fread(&count, sizeof(count), 1, file.get()) == 1,
        "Cannot read lightcone checkpoint count");
    detail::requireAnalysis(count <= 32, "Unexpected lightcone fixture particle count");
    Vector<LightconeParticle> particles(count);
    detail::requireAnalysis(
        std::fread(particles.data(), sizeof(LightconeParticle), count, file.get()) == count,
        "Cannot read lightcone checkpoint particles");
    return particles;
}
HPX_PLAIN_ACTION(analysisReadLightconeBuffer, AnalysisReadLightconeBufferAction);

namespace detail {

Vector<LightconeParticle> analysisGatherLightcone() {
    Vector<LightconeParticle> particles;
    for (auto const& locality : hpxLocalities()) {
        auto local = hpx::async<AnalysisReadLightconeBufferAction>(locality).get();
        particles.insert(particles.end(), local.begin(), local.end());
    }
    std::sort(particles.begin(), particles.end(),
        [](auto const& left, auto const& right) { return left.vel[XDIM] < right.vel[XDIM]; });
    return particles;
}

void analysisDistributeLightcone(Vector<LightconeParticle> const& particles) {
    for (int rank = 0; rank < hpxSize(); ++rank) {
        Vector<LightconeParticle> local;
        for (size_t i = rank; i < particles.size(); i += hpxSize()) {
            local.push_back(particles[i]);
        }
        hpx::async<AnalysisAddLightconeParticlesAction>(hpxLocalities()[rank], std::move(local))
            .get();
    }
}

void analysisFindLightconeGroups(double tau) {
    double const linkingLength = getOptions().lightcone.linkingLength *
        std::pow(getOptions().initial.numberOfParticles, -1.0 / NDIM);
    lcBuffer2homes();
    lcParticleBoundaries1();
    lcFormTrees(tau, linkingLength);
    lcFindNeighbors();
    bool converged = false;
    for (int iteration = 0; iteration < 64; ++iteration) {
        lcParticleBoundaries2();
        if (lcFindGroups() == 0) {
            converged = true;
            break;
        }
    }
    requireAnalysis(converged, "CPU lightcone grouping did not converge within 64 passes");
    lcGroups2homes();
}

void analysisCheckComponent(Vector<LightconeParticle> const& particles, int begin, int end,
    int firstIdentity, bool edge = false) {
    auto const group = particles[begin].group;
    requireAnalysis(group != LC_NO_GROUP && (group == LC_EDGE_GROUP) == edge,
        "Lightcone component has an incorrect group classification");
    for (int i = begin; i < end; ++i) {
        requireAnalysis(particles[i].vel[XDIM] == firstIdentity + i - begin,
            "Lightcone grouping lost, duplicated or reordered particle identities");
        requireAnalysis(particles[i].group == group,
            "Lightcone linked component did not converge to a common group");
    }
}

void analysisWriteLightconeCatalogue(int epoch, double tau) {
    double const linkingLength = getOptions().lightcone.linkingLength *
        std::pow(getOptions().initial.numberOfParticles, -1.0 / NDIM);
    lcParts2groups(1.0, linkingLength, epoch);
    for (int rank = 0; rank < hpxSize(); ++rank) {
        auto const path = std::filesystem::path(getOptions().lightcone.directory) /
            ("lc.groups." + std::to_string(epoch) + "." + std::to_string(rank));
        requireAnalysis(std::filesystem::exists(path) && std::filesystem::file_size(path) == 0,
            "Lightcone small-group filtering did not write the expected empty catalogue");
    }
    lcInit(tau, 1.0);
}

}    // namespace detail

void analysisBackendsTest() {
    using detail::requireAnalysis;
    requireAnalysis(!cudaEnabled() && !getOptions().hydro.enabled,
        "analysisBackends requires compute.cuda=false and hydro.enabled=false");
    requireAnalysis(getOptions().initial.particlesPerDimension == 8 &&
            !getOptions().initial.closePack && hpxSize() <= 2,
        "analysisBackends requires an 8-cubed lattice and at most two localities");
    for (auto const& locality : hpxLocalities()) {
        hpx::async<AnalysisPowerSetupAction>(locality, false).get();
    }
    for (int fold : {1, 2, 3}) {
        double mass = 0.0;
        for (auto const& locality : hpxLocalities()) {
            mass += hpx::async<AnalysisDensityMassAction>(locality, fold).get();
        }
        requireAnalysis(std::abs(mass - 512.0) < 0.01,
            "CPU density assignment did not conserve deposited particle mass");
    }
    auto const uniformPower = powerSpectrumCompute(1);
    requireAnalysis(!uniformPower.p.empty(), "CPU power spectrum produced no bins");
    for (auto const power : uniformPower.p) {
        requireAnalysis(detail::analysisFinite(power) && std::abs(power) < 1.0e-5,
            "A uniform lattice has nonzero resolved CPU power");
    }
    for (auto const& locality : hpxLocalities()) {
        hpx::async<AnalysisPowerSetupAction>(locality, true).get();
    }
    auto const perturbedPower = powerSpectrumCompute(1);
    auto const foldedPower = powerSpectrumCompute(2);
    double totalPower = 0.0;
    requireAnalysis(
        foldedPower.p.size() == perturbedPower.p.size(), "Folded CPU power changed its bin count");
    for (size_t bin = 0; bin < perturbedPower.p.size(); ++bin) {
        requireAnalysis(detail::analysisFinite(perturbedPower.p[bin]) &&
                detail::analysisFinite(perturbedPower.perr[bin]) &&
                detail::analysisFinite(foldedPower.p[bin]) &&
                detail::analysisFinite(foldedPower.perr[bin]),
            "CPU power spectrum contains nonfinite results");
        requireAnalysis(foldedPower.k[bin] == 2.0f * perturbedPower.k[bin],
            "Folded CPU power has incorrect wavenumbers");
        totalPower += perturbedPower.p[bin];
    }
    requireAnalysis(totalPower > 1.0e-3, "CPU power did not detect a perturbed density field");

    lcInit(0.2, 1.0);
    detail::analysisDistributeLightcone(detail::analysisLightconeFixture(lcNside()));
    detail::analysisFindLightconeGroups(0.2);
    auto first = detail::analysisGatherLightcone();
    requireAnalysis(first.size() == 12,
        "Lightcone finder did not retain two linked components and the edge pair");
    detail::analysisCheckComponent(first, 0, 6, 0);
    detail::analysisCheckComponent(first, 6, 10, 6);
    detail::analysisCheckComponent(first, 10, 12, 10, true);
    requireAnalysis(
        first[0].group != first[6].group, "Lightcone finder merged disconnected components");
    detail::analysisWriteLightconeCatalogue(0, 0.99);
    auto retained = detail::analysisGatherLightcone();
    requireAnalysis(retained.size() == 2, "Lightcone catalogue did not retain only edge particles");
    detail::analysisCheckComponent(retained, 0, 2, 10, true);
    requireAnalysis(lcNside() == 0, "Lightcone fixture did not exercise a shrinking pixel map");

    Vector<LightconeParticle> additional;
    additional.push_back(detail::analysisLightconeParticle(13, 0.4, 0.0));
    additional.push_back(detail::analysisLightconeParticle(14, 0.401, 0.0));
    detail::analysisDistributeLightcone(additional);
    detail::analysisFindLightconeGroups(0.99);
    auto second = detail::analysisGatherLightcone();
    requireAnalysis(second.size() == 4, "Repeated lightcone flush contains stale or missing data");
    detail::analysisCheckComponent(second, 0, 2, 10);
    detail::analysisCheckComponent(second, 2, 4, 13);
    requireAnalysis(second[0].group != second[2].group,
        "Repeated lightcone flush merged disconnected components");
    detail::analysisWriteLightconeCatalogue(1, 1.0);
    detail::analysisFindLightconeGroups(1.0);
    requireAnalysis(detail::analysisGatherLightcone().empty(),
        "Empty lightcone flush retained stale particles");
    detail::analysisWriteLightconeCatalogue(2, 1.0);
    PRINT("CPU analysis backend integration regression passed (%i localities)\n", hpxSize());
}
