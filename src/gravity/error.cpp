/* CosmicTiger - sampled gravity errors during evolution; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/error.hpp>
#include <cosmictiger/gravity/error_cuda.hpp>
#include <cosmictiger/gravity/reference_table.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/particle_selection.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {

struct GravityErrorSample {
    size_t slot;
    Array<Position, kDimensions> position;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & slot & position;
    }
};

struct GravityErrorCapture {
    size_t slot;
    Array<float, kDimensions> acceleration;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & slot & acceleration;
    }
};

using GravityErrorAcceleration = Array<double, kDimensions>;
Vector<GravityErrorAcceleration> gravityErrorReference;

size_t gravityErrorLocalCount() {
    return particlesSize();
}

Vector<GravityErrorSample> gravityErrorSelect(
    size_t globalOffset, size_t totalCount, size_t sampleCount) {
    particlesGravitySamplesBegin(sampleCount);
    Vector<GravityErrorSample> samples;
    for (size_t slot = 0; slot < sampleCount; ++slot) {
        size_t const globalIndex = evenlySpacedParticleIndex(slot, totalCount, sampleCount);
        if (globalIndex >= globalOffset && globalIndex - globalOffset < particlesSize()) {
            ParticleInteger const index = globalIndex - globalOffset;
            particlesGravitySample(index) = slot;
            GravityErrorSample sample{};
            sample.slot = slot;
            for (int dim = 0; dim < kDimensions; ++dim) {
                sample.position[dim] = particlesPos(dim, index);
            }
            samples.push_back(sample);
        }
    }
    return samples;
}

Vector<GravityErrorAcceleration> gravityErrorDirect(Vector<GravityErrorSample> const& samples) {
    Vector<GravityErrorAcceleration> result(samples.size());
    if (samples.empty() || particlesSize() == 0) {
        return result;
    }
    auto const& correctionTable = gravityReferenceCorrectionTable();
    double const supportRadius = getOptions().gravity.supportRadius;
    double const gm = getOptions().gravity.gm;
#ifdef USE_CUDA
    if (cudaEnabled()) {
        std::vector<GravityErrorPosition> positions(samples.size());
        for (size_t slot = 0; slot < samples.size(); ++slot) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                positions[slot][dim] = samples[slot].position[dim];
            }
        }
        auto const accelerations = gravityErrorDirectCuda(
            positions, particlesSize(),
            [](size_t begin, std::span<GravityErrorSource> sources) {
                for (size_t offset = 0; offset < sources.size(); ++offset) {
                    ParticleInteger const source = begin + offset;
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        sources[offset].position[dim] = particlesPos(dim, source);
                    }
                    sources[offset].mass = particlesMass(source);
                }
            },
            selectedGravityKernel(), supportRadius, correctionTable);
        for (size_t slot = 0; slot < samples.size(); ++slot) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                result[slot][dim] = gm * accelerations[slot][dim];
            }
        }
        return result;
    }
#endif
    int const threads = std::max(1, hpxHardwareConcurrency());
    Vector<hpx::future<void>> work;
    int const tasks = std::min(size_t(threads), samples.size());
    for (int task = 0; task < tasks; ++task) {
        work.push_back(hpx::async(hpx::annotated_function(
            [&, task] {
                size_t const begin = size_t(task) * samples.size() / tasks;
                size_t const end = size_t(task + 1) * samples.size() / tasks;
                for (size_t slot = begin; slot < end; ++slot) {
                    std::array<GravityReferenceSum, kDimensions> sum;
                    for (ParticleInteger source = 0; source < particlesSize(); ++source) {
                        std::array<double, kDimensions> separation;
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            separation[dim] = doubleDistance(
                                samples[slot].position[dim], particlesPos(dim, source));
                        }
                        auto const force = tabulatedGravityReference(separation,
                            selectedGravityKernel(), supportRadius, correctionTable.data());
                        double const mass = particlesMass(source);
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            sum[dim].add(mass * force[dim]);
                        }
                    }
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        result[slot][dim] = gm * sum[dim].value();
                    }
                }
            },
            "gravity.error.direct")));
    }
    hpx::wait_all(work.begin(), work.end());
    for (auto& future : work) {
        future.get();
    }
    return result;
}

Vector<GravityErrorCapture> gravityErrorCollect() {
    Vector<GravityErrorCapture> result;
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        ParticleInteger const slot = particlesGravitySample(index);
        if (slot >= 0) {
            result.push_back({size_t(slot), particlesGravitySampleAccelerations[slot]});
        }
    }
    particlesGravitySamplesBegin(0);
    return result;
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::gravityErrorLocalCount, GravityErrorLocalCountAction);
HPX_PLAIN_ACTION(detail::gravityErrorSelect, GravityErrorSelectAction);
HPX_PLAIN_ACTION(detail::gravityErrorDirect, GravityErrorDirectAction);
HPX_PLAIN_ACTION(detail::gravityErrorCollect, GravityErrorCollectAction);

void gravityErrorBegin() {
    if (getOptions().gravity.errorInterval <= 0) {
        return;
    }
    if (!detail::gravityErrorReference.empty()) {
        throw std::runtime_error("A gravity error reference is already pending");
    }
    auto const& localities = hpxLocalities();
    Vector<hpx::future<size_t>> counts;
    for (auto const& locality : localities) {
        counts.push_back(hpx::async<GravityErrorLocalCountAction>(locality));
    }
    Vector<size_t> offsets(localities.size() + 1, 0);
    for (size_t rank = 0; rank < localities.size(); ++rank) {
        offsets[rank + 1] = offsets[rank] + counts[rank].get();
    }
    size_t const sampleCount =
        std::min(offsets.back(), size_t(getOptions().gravity.errorSampleSize));
    if (sampleCount == 0) {
        return;
    }
    Vector<hpx::future<Vector<detail::GravityErrorSample>>> selections;
    for (size_t rank = 0; rank < localities.size(); ++rank) {
        selections.push_back(hpx::async<GravityErrorSelectAction>(
            localities[rank], offsets[rank], offsets.back(), sampleCount));
    }
    Vector<detail::GravityErrorSample> samples(sampleCount);
    for (auto& selection : selections) {
        for (auto const& sample : selection.get()) {
            samples[sample.slot] = sample;
        }
    }
    PRINT("Computing direct gravity reference for %zu of %zu particles using %s Ewald table\n",
        sampleCount, offsets.back(), cudaEnabled() ? "CUDA" : "CPU");
    Vector<hpx::future<Vector<detail::GravityErrorAcceleration>>> references;
    for (auto const& locality : localities) {
        references.push_back(hpx::async<GravityErrorDirectAction>(locality, samples));
    }
    std::vector<std::array<detail::GravityReferenceSum, kDimensions>> sums(sampleCount);
    for (auto& reference : references) {
        auto const partial = reference.get();
        for (size_t slot = 0; slot < sampleCount; ++slot) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                sums[slot][dim].add(partial[slot][dim]);
            }
        }
    }
    detail::gravityErrorReference.resize(sampleCount);
    for (size_t slot = 0; slot < sampleCount; ++slot) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            detail::gravityErrorReference[slot][dim] = sums[slot][dim].value();
        }
    }
}

void gravityErrorFinish(int coarseStep, double time, double scaleFactor) {
    size_t const sampleCount = detail::gravityErrorReference.size();
    if (sampleCount == 0) {
        return;
    }
    Vector<hpx::future<Vector<detail::GravityErrorCapture>>> captures;
    for (auto const& locality : hpxLocalities()) {
        captures.push_back(hpx::async<GravityErrorCollectAction>(locality));
    }
    std::vector<bool> seen(sampleCount, false);
    detail::GravityReferenceSum errorSquared;
    detail::GravityReferenceSum referenceSquared;
    double maxAbsolute = 0;
    double maxRelative = 0;
    size_t zeroReferenceCount = 0;
    for (auto& capture : captures) {
        for (auto const& sample : capture.get()) {
            if (sample.slot >= sampleCount || seen[sample.slot]) {
                throw std::runtime_error("Gravity error sample ownership changed incorrectly");
            }
            seen[sample.slot] = true;
            double errorNormSquared = 0;
            double referenceNormSquared = 0;
            for (int dim = 0; dim < kDimensions; ++dim) {
                double const actual = sample.acceleration[dim];
                double const expected = detail::gravityErrorReference[sample.slot][dim];
                if (!std::isfinite(actual) || !std::isfinite(expected)) {
                    throw std::runtime_error(
                        "Gravity error sample was not captured or is nonfinite");
                }
                double const difference = actual - expected;
                errorNormSquared += difference * difference;
                referenceNormSquared += expected * expected;
            }
            errorSquared.add(errorNormSquared);
            referenceSquared.add(referenceNormSquared);
            double const absolute = std::sqrt(errorNormSquared);
            double const reference = std::sqrt(referenceNormSquared);
            zeroReferenceCount += reference == 0;
            // A nonzero error against an exactly zero force is undefined in
            // relative units; report infinity while keeping absolute errors.
            double const relative = reference > 0 ?
                absolute / reference :
                (absolute == 0 ? 0 : std::numeric_limits<double>::infinity());
            maxAbsolute = std::max(maxAbsolute, absolute);
            maxRelative = std::max(maxRelative, relative);
        }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        throw std::runtime_error("Gravity error sample is missing after the kick");
    }
    double const rmsAbsolute = std::sqrt(errorSquared.value() / sampleCount);
    double const rmsReference = std::sqrt(referenceSquared.value() / sampleCount);
    double const rmsRelative = rmsReference > 0 ?
        rmsAbsolute / rmsReference :
        (rmsAbsolute == 0 ? 0 : std::numeric_limits<double>::infinity());
    std::string const path = outputPath("gravity_error.csv");
    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error("Cannot open gravity error output: " + path);
    }
    if (output.tellp() == 0) {
        output << "coarse_step,time,scale_factor,sample_count,rms_absolute,rms_relative,"
                  "max_absolute,max_relative,rms_reference,zero_reference_count\n";
    }
    output << std::setprecision(17) << coarseStep << ',' << time << ',' << scaleFactor << ','
           << sampleCount << ',' << rmsAbsolute << ',' << rmsRelative << ',' << maxAbsolute << ','
           << maxRelative << ',' << rmsReference << ',' << zeroReferenceCount << '\n';
    output.close();
    if (!output) {
        throw std::runtime_error("Cannot write gravity error output: " + path);
    }
    PRINT("Gravity error at coarse step %i (%zu particles): RMS absolute %.6e, RMS relative "
          "%.6e, max relative %.6e\n",
        coarseStep, sampleCount, rmsAbsolute, rmsRelative, maxRelative);
    detail::gravityErrorReference.clear();
}
