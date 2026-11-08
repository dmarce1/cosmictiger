/* CosmicTiger - streamed mesh-field kick consumer; BSL-1.0. */

#include <algorithm>
#include <cmath>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/stack_trace.hpp>
#include <cosmictiger/gravity/pm_kick.hpp>
#include <cosmictiger/gravity/softening.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#endif

namespace {

struct LongKick {
    bool active = false;
    float closeHalf = 0.0f;
    float openHalf = 0.0f;
    Vector3<float> acceleration{};
};

LongKick readLongKick(
    ParticleInteger index, KickParameters const& parameters, FlopCounter<>& flops) {
    LongKick result;
    if (parameters.longCloseTimestep <= 0.0f && parameters.longOpenTimestep <= 0.0f) {
        return result;
    }
    if (!particlesHasLongAcceleration(getOptions())) {
        THROW_ERROR("A long half-kick was requested without the particle-mesh acceleration\n");
    }
    result.active = true;
    result.closeHalf = 0.5f * parameters.longCloseTimestep;
    result.openHalf = 0.5f * parameters.longOpenTimestep;
    for (int dim = 0; dim < kDimensions; ++dim) {
        result.acceleration[dim] = particlesLongAcceleration(dim, index);
    }
    flops += 2;
    return result;
}

}    // namespace

KickReturn particleMeshApplyField(ParticleInteger index, Vector3<float> const& acceleration,
    float potential, KickParameters const& parameters, FlopCount* flopCount,
    FlopCounts* precisionCounts) {
    KickReturn result;
    FlopCounter<> flops;
    auto const& options = getOptions();
    auto& rung = particlesRung(index);
    if (rung < parameters.minRung) {
        return result;
    }
    LongKick const longKick = readLongKick(index, parameters, flops);
    Vector3<float> total = acceleration;
    if (longKick.active) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            total[dim] += longKick.acceleration[dim];
        }
        flops += kDimensions;
    }
    particlesCaptureGravitySample(index, total[kXi], total[kYi], total[kZi]);
    bool const saveAcceleration =
        parameters.saveForce || saveParticleAcceleration(options.analysis);
    bool const savePotential = parameters.saveForce || saveParticlePotential(options.analysis);
    if (saveAcceleration) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesGravityAcceleration(dim, index) = total[dim];
        }
    }
    if (savePotential && parameters.doPhi) {
        particlesPot(index) = potential;
    }
    auto& vx = particlesVel(kXi, index);
    auto& vy = particlesVel(kYi, index);
    auto& vz = particlesVel(kZi, index);
    float const gx = acceleration[kXi];
    float const gy = acceleration[kYi];
    float const gz = acceleration[kZi];
    float const effectiveRadius =
        gravitySofteningEffectiveRadius(parameters.gravitySmoothingLength);
    // A new outer interval closes the previous substep with the old base timestep.
    float const closingBase =
        parameters.closingBaseTimestep > 0.0f ? parameters.closingBaseTimestep : parameters.t0;
    flops += 1;
    auto submitFlops = [&]() {
        FlopCounts counts;
        if constexpr (kCountFlops) {
            // The radius expression promotes its float numerator, divides in
            // double precision, and narrows to float. All remaining solver
            // arithmetic in this kick uses float, including its FMA arguments.
            counts.singlePrecision = FlopCount(flops) - 1;
            counts.doublePrecision = 3;
        }
        if (flopCount != nullptr) {
            *flopCount += counts.total();
        }
        if (precisionCounts != nullptr) {
            *precisionCounts += counts;
        }
        if (flopCount == nullptr && precisionCounts == nullptr) {
            addCpuFlops(counts.singlePrecision, FlopPrecision::Single);
            addCpuFlops(counts.doublePrecision, FlopPrecision::Double);
        }
    };
    auto const applyLong = [&](float sign, float half) {
        if (longKick.active && half > 0.0f) {
            vx = std::fma(sign * longKick.acceleration[kXi], half, vx);
            vy = std::fma(sign * longKick.acceleration[kYi], half, vy);
            vz = std::fma(sign * longKick.acceleration[kZi], half, vz);
            flops += 3 * kDimensions;
        }
    };
    float const mass = particlesMass(index);
    float const oldKinetic = 0.5f * mass * sqr(vx, vy, vz);
#if COSMICTIGER_WITH_SPH
    bool const gas = options.hydro.enabled && particlesType(index) == kSphType;
    if (!parameters.htime && gas) {
        // The hydro integrator owns the gas momentum and timestep. As in the
        // ordinary FMM kick, publish the active short gas force without a
        // gravity-only rung update. The long half-kicks of an outer boundary
        // are applied directly; their diagnostic corrections report through the
        // dark-matter channels so the synchronized energies stay consistent.
        THROW_ERROR("Mesh gravity with midpoint SPH is deferred\n");
        if (!parameters.firstCall) {
            applyLong(parameters.sign, longKick.closeHalf);
        }
        float const closedVx = vx;
        float const closedVy = vy;
        float const closedVz = vz;
        // Hydro sums the gas energies and momenta in its kick.
        result.pot = parameters.doPhi ? 0.5f * mass * potential : 0.0f;
        result.maxRung = rung;
        applyLong(parameters.sign, longKick.openHalf);
        if (longKick.active) {
            result.darkMatterKineticCorrection =
                0.5f * mass * (sqr(closedVx, closedVy, closedVz) - sqr(vx, vy, vz));
            result.darkMatterXMomentumCorrection = mass * (closedVx - vx);
            result.darkMatterYMomentumCorrection = mass * (closedVy - vy);
            result.darkMatterZMomentumCorrection = mass * (closedVz - vz);
            result.darkMatterMomentumNormCorrection =
                mass * (std::sqrt(sqr(closedVx, closedVy, closedVz)) - std::sqrt(sqr(vx, vy, vz)));
        }
        result.dkin = 0.5f * mass * sqr(vx, vy, vz) - oldKinetic;
        submitFlops();
        return result;
    }
#endif
    int proposedRung = 0;
    if (!parameters.htime || parameters.descending) {
        // The inner rung follows the short force at the softening scale; the
        // outer PM interval already bounds the long-force variation.
        float const accelerationSquared = sqr(gx, gy, gz) + 1.0e-35f;
        float const requiredTimestep = std::min(parameters.maxDt,
            std::min(parameters.eta * std::sqrt(parameters.a) *
                    std::sqrt(effectiveRadius / std::sqrt(accelerationSquared)),
                parameters.t0));
        if (!(requiredTimestep > 0.0f) || !std::isfinite(requiredTimestep)) {
            THROW_ERROR("Mesh kick timestep is not finite and positive\n");
        }
        float const requestedRung = std::ceil(std::log2(parameters.t0 / requiredTimestep));
        if (!std::isfinite(requestedRung) || requestedRung >= kMaxRung) {
            THROW_ERROR("Mesh kick timestep exceeds the available rung range\n");
        }
        proposedRung = int(requestedRung);
        // Squared acceleration, timestep bound, and division/log2/ceil rung proposal.
        flops += 15;
    }
    if (parameters.htime) {
        float const sign = parameters.sign * (parameters.top ? 1.0f : -1.0f);
        float const halfStep = 0.5f * rungDt[parameters.minRung] * closingBase;
        flops += 3;
        auto const applyKick = [&](float step) {
            vx = std::fma(sign * gx, step, vx);
            vy = std::fma(sign * gy, step, vy);
            vz = std::fma(sign * gz, step, vz);
            flops += 3 * kDimensions;
        };
        if (parameters.ascending && !parameters.firstCall) {
            applyKick(halfStep);
            if (parameters.top) {
                applyLong(parameters.sign, longKick.closeHalf);
            }
        }
        result.kin = 0.5f * mass * sqr(vx, vy, vz);
        result.xmom = mass * vx;
        result.ymom = mass * vy;
        result.zmom = mass * vz;
        result.nmom = mass * std::sqrt(sqr(vx, vy, vz));
        if (parameters.descending) {
            int const nextRung = std::max(
                parameters.minRung + int(proposedRung > parameters.minRung), int(rung) - 1);
            result.maxRung = nextRung;
            ALWAYS_ASSERT(nextRung >= 0 && nextRung < kMaxRung);
            rung = nextRung;
            // The opening half-kick uses the current (next) outer interval;
            // only a closing outer boundary needs a second step.
            float openStep = halfStep;
            if (closingBase != parameters.t0) {
                openStep = 0.5f * rungDt[parameters.minRung] * parameters.t0;
                flops += 2;
            }
            applyKick(openStep);
            if (parameters.top) {
                applyLong(parameters.sign, longKick.openHalf);
            }
        }
    } else {
        float const sign = parameters.sign;
        if (parameters.frozenNearClose) {
            // PMMM outer interval, old membership at a new boundary (docs/pmmm.md):
            // Δv = g_old (δ_old/2 - H_old/2), the last inner closing half-kick on
            // the old near field minus the near part of the old long closing
            // half-kick, (H_old/2)(T_new - S_old). The new-membership pass adds
            // (H_old/2) S_new + (H_old/2) L_new to finish the closing. A rung-zero
            // particle has δ_old = H_old and receives nothing here.
            // Symbol directory: g_old → near acceleration on the old membership
            // (gx, gy, gz); δ_old → rungDt[rung]*closingBase; H_old → closingBase.
            float const step = 0.5f * closingBase * (rungDt[rung] - 1.0f);
            vx = std::fma(sign * gx, step, vx);
            vy = std::fma(sign * gy, step, vy);
            vz = std::fma(sign * gz, step, vz);
            flops += 3 + 3 * kDimensions;
            result.maxRung = rung;
            result.dkin = 0.5f * mass * sqr(vx, vy, vz) - oldKinetic;
            submitFlops();
            return result;
        }
        if (!parameters.firstCall) {
            // The frozen-membership boundary pass closes the short force over the
            // whole old interval; its old-membership pass already applied the
            // rung-dependent remainder.
            float const closingFraction = parameters.closeShortOverInterval ? 1.0f : rungDt[rung];
            float const halfStep = 0.5f * closingFraction * closingBase;
            vx = std::fma(sign * gx, halfStep, vx);
            vy = std::fma(sign * gy, halfStep, vy);
            vz = std::fma(sign * gz, halfStep, vz);
            flops += 2 + 3 * kDimensions;
            applyLong(sign, longKick.closeHalf);
        }
        result.kin = 0.5f * mass * sqr(vx, vy, vz);
        result.xmom = mass * vx;
        result.ymom = mass * vy;
        result.zmom = mass * vz;
        result.nmom = mass * std::sqrt(sqr(vx, vy, vz));
        float const synchronizedVx = vx;
        float const synchronizedVy = vy;
        float const synchronizedVz = vz;
        rung = std::max({proposedRung, int(rung) - 1, parameters.minRung, 1});
        result.maxRung = rung;
        if (parameters.fixedRung != 0) {
            rung = parameters.fixedRung;
        }
        ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
        float const halfStep = 0.5f * rungDt[rung] * parameters.t0;
        vx = std::fma(sign * gx, halfStep, vx);
        vy = std::fma(sign * gy, halfStep, vy);
        vz = std::fma(sign * gz, halfStep, vz);
        flops += 2 + 3 * kDimensions;
        applyLong(sign, longKick.openHalf);
        result.darkMatterKineticCorrection =
            0.5f * mass * (sqr(synchronizedVx, synchronizedVy, synchronizedVz) - sqr(vx, vy, vz));
        result.darkMatterXMomentumCorrection = mass * (synchronizedVx - vx);
        result.darkMatterYMomentumCorrection = mass * (synchronizedVy - vy);
        result.darkMatterZMomentumCorrection = mass * (synchronizedVz - vz);
        result.darkMatterMomentumNormCorrection = mass *
            (std::sqrt(sqr(synchronizedVx, synchronizedVy, synchronizedVz)) -
                std::sqrt(sqr(vx, vy, vz)));
    }
    result.pot = parameters.doPhi ? 0.5f * mass * potential : 0.0;
    result.dkin = 0.5f * mass * sqr(vx, vy, vz) - oldKinetic;
    // Energy/momentum diagnostics and their reductions are deliberately excluded.
    submitFlops();
    return result;
}
