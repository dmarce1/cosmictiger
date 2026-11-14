/* CosmicTiger - CPU analysis integration regression; BSL-1.0. */
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>

#include <healpix_base.h>

#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/analysis/power.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/fft.hpp>
#include <cosmictiger/cuda/cuda_unordered_map.hpp>
#include <cosmictiger/particles/particles.hpp>

// Exercise the production deconvolution/binning stages using an analytic
// Fourier fixture, independently of the particle deposition kernel.
PowerSpectrumT powerSpectrumFromTransform();

void analysisSetIndependentFourierModes() {
    auto const& opts = getOptions();
    auto const box = fft3dComplexRange();
    auto& values = fft3dComplexVector();
    int const n = opts.initial.fourierSize;
    double const filter = cloudFilter(2.0 * std::numbers::pi / n);
    Array<int64_t, kDimensions> index;
    for (index[0] = box.begin[0]; index[0] < box.end[0]; ++index[0]) {
        for (index[1] = box.begin[1]; index[1] < box.end[1]; ++index[1]) {
            for (index[2] = box.begin[2]; index[2] < box.end[2]; ++index[2]) {
                double amplitude = 0.0;
                if ((index[0] == 1 || index[0] == n - 1) && index[1] == 0 && index[2] == 0) {
                    amplitude = 1.0;
                } else if (index[0] == 0 && (index[1] == 1 || index[1] == n - 1) && index[2] == 0) {
                    amplitude = 2.0;
                } else if (index[0] == 0 && index[1] == 0 && index[2] == 1) {
                    amplitude = 3.0;
                }
                values[box.index(index)] = ComplexFloat(amplitude / filter, 0.0f);
            }
        }
    }
}
HPX_PLAIN_ACTION(analysisSetIndependentFourierModes, AnalysisSetIndependentFourierModesAction);

namespace detail {

void requireAnalysis(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void analysisCheckHostContainers() {
    // This fixture runs under the no-CUDA-runtime preload guard. Rehashing and
    // destroying nested owners must release host allocations before any CUDA
    // pool or managed global has been initialized.
    CudaUnorderedMap<DeviceVector<int>> values;
    for (int key = 0; key < 40; ++key) {
        values[key].pushBack(key + 1);
    }
    requireAnalysis(values.size() == 40, "Host map growth lost an entry");
    for (int key = 0; key < 40; ++key) {
        requireAnalysis(values[key].size() == 1 && values[key][0] == key + 1,
            "Host map rehash corrupted nested vector storage");
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
    particle.pos[kXi] = radius * std::cos(angle);
    particle.pos[kYi] = radius * std::sin(angle);
    particle.pos[kZi] = height;
    // Velocities carry exact identities through tree permutation and routing.
    particle.vel[kXi] = identity;
    particle.group = kLightconeNoGroup;
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
        double const increment = 2.0 * std::numbers::pi / kSamples;
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
            analysisLightconeParticle(i, 0.55, angle + std::numbers::pi, 0.08 + 0.0002 * (i - 6)));
    }
    for (int i = 10; i < 12; ++i) {
        auto particle = analysisLightconeParticle(i, 0.002 * (i - 10), 0.0, 0.8);
        particles.push_back(particle);
    }
    particles.push_back(analysisLightconeParticle(12, 0.0, 0.0, -0.4));
    return particles;
}

}    // namespace detail

// With hydro the fixture mirrors the cosmological lattice: one gas particle per
// dark-matter site, offset by half a spacing, carrying the complementary mass.
void analysisPowerSetup(bool perturb) {
    auto const& opts = getOptions();
    bool const hydro = opts.hydro.enabled;
    int const side = opts.initial.latticeSide;
    int const count = side * side * side;
    int const begin = hpxRank() * count / hpxSize();
    int const end = (hpxRank() + 1) * count / hpxSize();
    int const species = hydro ? 2 : 1;
    particlesResize(species * (end - begin));
    for (int global = begin; global < end; ++global) {
        for (int kind = 0; kind < species; ++kind) {
            int const local = species * (global - begin) + kind;
            double const offset = kind == 0 ? 0.0 : 0.5 / side;
            double const x = (global / (side * side) + 0.5) / side + offset;
            particlesPos(kXi, local) =
                x + (perturb ? 0.03 * std::sin(2.0 * std::numbers::pi * x) : 0.0);
            particlesPos(kYi, local) = ((global / side) % side + 0.5) / side + offset;
            particlesPos(kZi, local) = (global % side + 0.5) / side + offset;
            for (int dim = 0; dim < kDimensions; ++dim) {
                particlesVel(dim, local) = 0.0f;
            }
            if (hydro) {
                particlesType(local) = kind == 0 ? kDarkMatterType : kSphType;
            }
        }
    }
}
HPX_PLAIN_ACTION(analysisPowerSetup, AnalysisPowerSetupAction);

double analysisParticleMass() {
    double mass = 0.0;
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        mass += particlesMass(i);
    }
    return mass;
}
HPX_PLAIN_ACTION(analysisParticleMass, AnalysisParticleMassAction);

double analysisDensityMass(int fold) {
    auto const& opts = getOptions();
    int const side = opts.initial.fourierSize;
    Range<int64_t> box;
    for (int dim = 0; dim < kDimensions; ++dim) {
        box.begin[dim] = kCloudMin;
        box.end[dim] = side + kCloudMax;
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
    auto const& opts = getOptions();
    std::filesystem::create_directories(opts.lightcone.directory);
    lcAddParticles(std::move(particles));
}
HPX_PLAIN_ACTION(analysisAddLightconeParticles, AnalysisAddLightconeParticlesAction);

Vector<LightconeParticle> analysisReadLightconeBuffer() {
    auto const closeFile = [](FILE* stream) { std::fclose(stream); };
    std::unique_ptr<FILE, decltype(closeFile)> file(std::tmpfile(), closeFile);
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
        [](auto const& left, auto const& right) { return left.vel[kXi] < right.vel[kXi]; });
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
    auto const& opts = getOptions();
    double const linkingLength =
        opts.lightcone.linkingLength * std::pow(opts.initial.numberOfParticles, -1.0 / kDimensions);
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
    requireAnalysis(group != kLightconeNoGroup && (group == kLightconeEdgeGroup) == edge,
        "Lightcone component has an incorrect group classification");
    for (int i = begin; i < end; ++i) {
        requireAnalysis(particles[i].vel[kXi] == firstIdentity + i - begin,
            "Lightcone grouping lost, duplicated or reordered particle identities");
        requireAnalysis(particles[i].group == group,
            "Lightcone linked component did not converge to a common group");
    }
}

void analysisWriteLightconeCatalogue(int epoch, double tau) {
    auto const& opts = getOptions();
    double const linkingLength =
        opts.lightcone.linkingLength * std::pow(opts.initial.numberOfParticles, -1.0 / kDimensions);
    lcParts2groups(1.0, linkingLength, epoch);
    for (int rank = 0; rank < hpxSize(); ++rank) {
        auto const path = std::filesystem::path(opts.lightcone.directory) /
            ("lc.groups." + std::to_string(epoch) + "." + std::to_string(rank));
        requireAnalysis(std::filesystem::exists(path) && std::filesystem::file_size(path) == 0,
            "Lightcone small-group filtering did not write the expected empty catalogue");
    }
    lcInit(tau, 1.0);
}

double analysisMapTotal(int epoch) {
    auto const& opts = getOptions();
    double total = 0.0;
    for (int rank = 0; rank < hpxSize(); ++rank) {
        auto const path = std::filesystem::path(opts.lightcone.directory) /
            ("lc.healpix." + std::to_string(epoch) + "." + std::to_string(rank));
        auto const closeFile = [](FILE* stream) { std::fclose(stream); };
        std::unique_ptr<FILE, decltype(closeFile)> file(std::fopen(path.c_str(), "rb"), closeFile);
        requireAnalysis(file != nullptr, "Lightcone epoch did not write its HEALPix map");
        int header[3];
        requireAnalysis(std::fread(header, sizeof(int), 3, file.get()) == 3,
            "Cannot read lightcone map header");
        for (int pixel = header[1]; pixel <= header[2]; ++pixel) {
            float value;
            requireAnalysis(
                std::fread(&value, sizeof(value), 1, file.get()) == 1, "Truncated lightcone map");
            total += value;
        }
    }
    return total;
}

// Only dark matter reaches the lightcone halo-finder buffer, while every
// species reaches the HEALPix map weighted by its mass.
void analysisCheckSpeciesLightcone(int epoch) {
    auto const& opts = getOptions();
    T_Healpix_Base<int> pixels(opts.lightcone.mapSize, NEST, SET_NSIDE);
    std::vector<std::vector<LightconeEntry>> entries(1);
    double expectedWeight = 0.0;
    constexpr int kDarkMatterCount = 3;
    for (int i = 0; i < kDarkMatterCount + 2; ++i) {
        bool const darkMatter = i < kDarkMatterCount;
        double const radius = 0.5 + 0.02 * i;
        double const angle = 0.3 * i;
        LightconeEntry entry{};
        entry.x = radius * std::cos(angle);
        entry.y = radius * std::sin(angle);
        entry.z = 0.0;
        entry.vx = 20.0f + i;
        vec3 direction;
        direction.x = entry.x.toDouble();
        direction.y = entry.y.toDouble();
        direction.z = 0.0;
        entry.pix = pixels.vec2pix(direction);
        entry.type = darkMatter ? kDarkMatterType : kSphType;
        double const mass = darkMatter ? opts.initial.darkMatterMass : opts.initial.gasMass;
        expectedWeight += mass / (radius * radius);
        entries[0].push_back(entry);
    }
    lcAddParticles(entries, 1.0, 0.0);
    requireAnalysis(analysisGatherLightcone().size() == kDarkMatterCount,
        "Gas entries reached the lightcone halo-finder buffer");
    analysisFindLightconeGroups(1.0);
    analysisWriteLightconeCatalogue(epoch, 1.0);
    double const total = analysisMapTotal(epoch);
    requireAnalysis(std::abs(total - expectedWeight) <= 1.0e-5 * expectedWeight,
        "Lightcone HEALPix map did not weight every species by its mass");
}

}    // namespace detail

void analysisBackendsTest() {
    auto const& opts = getOptions();
    using detail::requireAnalysis;
    requireAnalysis(!cudaEnabled(), "analysisBackends requires every module CUDA option false");
    detail::analysisCheckHostContainers();
    requireAnalysis(opts.initial.latticeSide == 8 && !opts.initial.closePack && hpxSize() <= 2,
        "analysisBackends requires an 8-cubed lattice and at most two localities");
    for (auto const& locality : hpxLocalities()) {
        hpx::async<AnalysisPowerSetupAction>(locality, false).get();
    }
    double expectedMass = 0.0;
    for (auto const& locality : hpxLocalities()) {
        expectedMass += hpx::async<AnalysisParticleMassAction>(locality).get();
    }
    // Species masses are complementary, so the total matches the DMO lattice.
    requireAnalysis(std::abs(expectedMass - 512.0) < 1.0e-3,
        "Fixture species masses do not sum to one per lattice site");
    for (int fold : {1, 2, 3}) {
        double mass = 0.0;
        for (auto const& locality : hpxLocalities()) {
            mass += hpx::async<AnalysisDensityMassAction>(locality, fold).get();
        }
        requireAnalysis(std::abs(mass - expectedMass) < 0.01,
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

    fft3dInit(opts.initial.fourierSize);
    for (auto const& locality : hpxLocalities()) {
        hpx::async<AnalysisSetIndependentFourierModesAction>(locality).get();
    }
    auto const knownPower = powerSpectrumFromTransform();
    requireAnalysis(std::abs(knownPower.p[0] - 14.0 / 3.0) < 2.e-6,
        "Power mean did not count each conjugate pair once");
    requireAnalysis(std::abs(knownPower.perr[0] - 7.0 / 3.0) < 2.e-6,
        "Power error deconvolved twice or counted duplicate modes");
    fft3dDestroy();

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
    if (opts.hydro.enabled) {
        detail::analysisCheckSpeciesLightcone(3);
    }
    std::cout << std::format(
                     "CPU analysis backend integration regression passed ({:d} localities)\n",
                     hpxSize())
              << std::flush;
}
