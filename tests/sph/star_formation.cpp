/* CosmicTiger - full-step GADGET-4 source transaction regression; BSL-1.0. */
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/chemistry/sph.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/thermodynamics.hpp>
#include <cosmictiger/sph/tree.hpp>
#include <cosmictiger/stars/star_formation.hpp>

namespace detail {
constexpr double kSfScale = 0.25;
struct SfRecord {
    uint64_t id{}, eventCounter{}, formationTime{}, specificRate{}, elapsedTime{};
    uint32_t energy{}, entropy{}, electrons{}, mass{}, bulkViscosity{};
    int proposedRung{};
    std::array<uint64_t, kDimensions> position{};
    std::array<uint32_t, kDimensions> velocity{};
    uint64_t rung{}, type{};
    uint32_t density{};
    bool operator==(SfRecord const&) const = default;
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & id & eventCounter & formationTime & specificRate & elapsedTime & energy &
            entropy & electrons & mass & bulkViscosity & proposedRung & position & velocity & rung &
            type & density;
    }
};
using SfSnapshot = std::vector<std::vector<SfRecord>>;
void requireSf(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("GADGET-4 SF regression: ") + message);
    }
}
void requireSfClose(double actual, double expected, char const* message, double tolerance = 2.e-6) {
    requireSf(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <=
                tolerance * std::max({std::abs(actual), std::abs(expected), 1.e-100}),
        message);
}
float sfStoredDensity(double density) {
    auto const& options = getOptions();
    return float(units::toStoredDensity(units::ProperDensity(units::Density::from_value(density)),
        units::scalesFromOptions(options),
        units::CosmologicalContext(kSfScale, options.cosmology.hubble))
            .value());
}
float sfStoredEnergy(double energy) {
    auto const& options = getOptions();
    return float(units::toStoredSpecificEnergy(
        units::PhysicalSpecificEnergy(units::SpecificEnergy::from_value(energy)),
        units::scalesFromOptions(options),
        units::CosmologicalContext(kSfScale, options.cosmology.hubble))
            .value());
}
double sfStep() {
    auto const& opts = getOptions();
    return 1.e11 / (kSfScale * opts.units.timeSeconds);
}
void initializeSfFixture() {
    particlesResize(3);
    sphParticlesResize(3, false);
    sphParticlesEntropyInitialized = true;
    sphParticlesThermalScaleFactor = kSfScale;
    auto const model = starFormationModel();
    for (ParticleInteger index = 0; index < 3; ++index) {
        particlesType(index) = kSphType;
        particlesCatIndex(index) = index;
        particlesRung(index) = index == 0 ? 1 : 3;
        particlesStarFormationStates[index] = {};
        particlesStarFormationStates[index].id = uint64_t(hpxRank()) * 3 + index + 1;
        sphParticlesParticleIndex(index) = index;
        sphParticlesDensity(index) =
            sfStoredDensity((index == 2 ? 0.5 : 10.0) * model.thresholdDensity);
        sphParticlesSetInternalEnergy(index, sfStoredEnergy(model.parameters.coldEnergy));
        sphParticlesSmoothingLength(index) = 0.125f;
        sphParticlesElectronFractions[index] = 1.0f;
        sphParticlesSetProposedRung(index, char(4 + index));
        if (sphParticlesBulkViscosityCoefficients != nullptr) {
            sphParticlesBulkViscosityCoefficients[index] = 0.25f + 0.125f * index;
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, index) = 0.25 + index * 0.125;
            particlesVel(dim, index) = float(dim - 1) / 4.0f;
        }
    }
}
std::vector<SfRecord> sfSnapshot() {
    std::vector<SfRecord> result;
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        auto const& state = particlesStarFormationStates[index];
        SfRecord record;
        record.id = state.id;
        record.eventCounter = state.eventCounter;
        record.formationTime = std::bit_cast<uint64_t>(state.formationTime);
        record.specificRate = std::bit_cast<uint64_t>(state.specificStarFormationRate);
        record.elapsedTime = std::bit_cast<uint64_t>(state.elapsedPhysicalTime);
        bool const gas = particlesType(index) == kSphType;
        auto const gasIndex = particlesCatIndex(index);
        record.energy = std::bit_cast<uint32_t>(gas ? sphParticlesInternalEnergy(gasIndex) : 0.0f);
        record.entropy = std::bit_cast<uint32_t>(gas ? sphParticlesEntropies[gasIndex] : 0.0f);
        record.mass = std::bit_cast<uint32_t>(particlesMass(index));
        if (gas) {
            record.proposedRung = sphParticlesProposedRung(gasIndex);
            if (sphParticlesBulkViscosityCoefficients != nullptr) {
                record.bulkViscosity =
                    uint32_t(sphParticlesBulkViscosityCoefficients[gasIndex].bits);
            }
        }
        record.electrons =
            std::bit_cast<uint32_t>(gas ? sphParticlesElectronFractions[gasIndex] : 0.0f);
        for (int dim = 0; dim < kDimensions; ++dim) {
            record.velocity[dim] = std::bit_cast<uint32_t>(particlesVel(dim, index));
            record.position[dim] = uint64_t(particlesPos(dim, index).raw());
        }
        record.rung = uint64_t(particlesRung(index));
        record.type = uint64_t(particlesType(index));
        record.density = std::bit_cast<uint32_t>(gas ? sphParticlesDensity(gasIndex) : 0.0f);
        result.push_back(record);
    }
    return result;
}

void alterSfFixture(int operation) {
    if (operation == 0 && hpxRank() == hpxSize() - 1) {
        particlesStarFormationStates[2].id = 0;
    } else if (operation == 1) {
        particlesStarFormationStates[2].id = uint64_t(hpxRank()) * 3 + 3;
    } else if (operation == 2) {
        particlesStarFormationStates[1].specificStarFormationRate = 1.0;
    } else if (operation == 4 || operation == 5) {
        for (ParticleInteger particle = 0; particle < particlesSize(); ++particle) {
            if (particlesType(particle) == kSphType) {
                bool const convert = operation == 5 || hpxRank() == hpxSize() - 1 || particle == 2;
                particlesStarFormationStates[particle].specificStarFormationRate =
                    convert ? 1.0 : 0.0;
            }
        }
    } else if (operation == 6) {
        starFormationCommitLocal();
    } else if (operation == 7) {
        auto const before = sfSnapshot();
        bool rejected = false;
        try {
            starFormationPrepareLocal({1, 1}, {sfStep(), sfStep()}, kSfScale);
        } catch (std::invalid_argument const&) {
            rejected = true;
        }
        starFormationCommitLocal();
        requireSf(rejected && sfSnapshot() == before,
            "duplicate source slots were committed or retained a pending transaction");
    } else if (operation == 3) {
        auto const before = sfSnapshot();
        FILE* file = std::tmpfile();
        requireSf(file != nullptr, "cannot open checkpoint fixture");
        particlesSave(file);
        for (ParticleInteger index = 0; index < particlesSize(); ++index) {
            particlesStarFormationStates[index] = {};
        }
        std::rewind(file);
        particlesLoad(file);
        std::fclose(file);
        requireSf(
            sfSnapshot() == before, "checkpoint changed saved rate/electron guess/particle state");
    }
}
std::vector<SfRecord> sfSourceReference() {
    auto const& opts = getOptions();
    auto parameters = starFormationParameters(kSfScale);
    parameters.explicitTimestep = sfStep();
    std::vector<SfRecord> result;
    for (ParticleInteger gas = 1; gas < 3; ++gas) {
        star_formation_backend::Input input;
        input.state = particlesStarFormationStates[sphParticlesParticleIndex(gas)];
        input.electronFraction = sphParticlesElectronFractions[gas];
        input.density = sphParticlesDensity(gas);
        input.energy = sphParticlesInternalEnergy(gas);
        input.rung = sphParticlesRung(gas);
        auto const output = star_formation_backend::advance(input, parameters);
        requireSf(output.status == 0, "scalar source reference failed");
        SfRecord record;
        auto const prepared = sph_entropy::prepareStoredEnergy(
            output.energy, input.density, opts.hydro.heatCapacityRatio, kSfScale);
        record.energy = std::bit_cast<uint32_t>(prepared.storedEnergy);
        record.entropy = std::bit_cast<uint32_t>(prepared.entropy);
        record.electrons = std::bit_cast<uint32_t>(output.electronFraction);
        record.specificRate = std::bit_cast<uint64_t>(output.state.specificStarFormationRate);
        result.push_back(record);
    }
    return result;
}
}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeSfFixture, InitializeSfFixtureAction);
HPX_PLAIN_ACTION(detail::sfSnapshot, SfSnapshotAction);
HPX_PLAIN_ACTION(detail::alterSfFixture, AlterSfFixtureAction);
HPX_PLAIN_ACTION(detail::sfSourceReference, SfSourceReferenceAction);

namespace detail {
template <class Action, class... Arguments>
void sfAll(Arguments const&... arguments) {
    std::vector<hpx::future<void>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<Action>(locality, arguments...));
    }
    for (auto& future : futures) {
        future.get();
    }
}
template <class Action>
SfSnapshot sfCollect() {
    std::vector<hpx::future<std::vector<SfRecord>>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<Action>(locality));
    }
    SfSnapshot result;
    for (auto& future : futures) {
        result.push_back(future.get());
    }
    return result;
}

void sfLedger(SfSnapshot const& before, SfSnapshot const& after, double heating) {
    auto const& opts = getOptions();
    double expected = 0.0;
    for (size_t rank = 0; rank < before.size(); ++rank) {
        for (size_t particle = 0; particle < 3; ++particle) {
            expected -= opts.initial.gasMass *
                (double(std::bit_cast<float>(after[rank][particle].energy)) -
                    double(std::bit_cast<float>(before[rank][particle].energy))) /
                kSfScale;
        }
    }
    requireSfClose(heating, expected, "full-source energy ledger is incorrect", 2.e-12);
}
}    // namespace detail

void starFormationTest() {
    auto const& opts = getOptions();
    detail::requireSf(
        opts.starFormation.enabled && opts.chemistry.enabled, "fixture requires cooling and SF");
    detail::sfAll<InitializeSfFixtureAction>();
    particlesInitializeStarFormationIds();
    auto const initial = detail::sfCollect<SfSnapshotAction>();
    auto const reference = detail::sfCollect<SfSourceReferenceAction>();
    auto first = starFormationClose(3, detail::kSfScale, 0.0, detail::sfStep(), detail::sfStep());
    auto after = detail::sfCollect<SfSnapshotAction>();
    detail::requireSf(first.converted == 0, "initial zero saved SFR should not form stars");
    detail::sfLedger(initial, after, first.heating);
    for (size_t rank = 0; rank < after.size(); ++rank) {
        detail::requireSf(initial[rank][0] == after[rank][0], "inactive particle changed");
        for (size_t gas = 1; gas < 3; ++gas) {
            auto const& ref = reference[rank][gas - 1];
            auto const& actual = after[rank][gas];
            detail::requireSf(actual.proposedRung == initial[rank][gas].proposedRung &&
                    actual.bulkViscosity == initial[rank][gas].bulkViscosity,
                "source changed a viscosity coefficient or a hydro proposal");
            detail::requireSf(actual.entropy == ref.entropy,
                "survivor did not commit the prepared canonical entropy");
            detail::requireSfClose(std::bit_cast<float>(actual.energy),
                std::bit_cast<float>(ref.energy), "source disagrees with scalar energy");
            detail::requireSfClose(std::bit_cast<float>(actual.electrons),
                std::bit_cast<float>(ref.electrons), "source disagrees with scalar Ne");
            detail::requireSfClose(std::bit_cast<double>(actual.specificRate),
                std::bit_cast<double>(ref.specificRate), "source disagrees with saved SFR");
        }
    }
    detail::sfAll<AlterSfFixtureAction>(3);
    detail::sfAll<AlterSfFixtureAction>(0);
    auto const invalid = detail::sfCollect<SfSnapshotAction>();
    bool failed = false;
    try {
        starFormationClose(3, detail::kSfScale, 0.0, detail::sfStep(), detail::sfStep());
    } catch (std::exception const&) {
        failed = true;
    }
    detail::requireSf(failed && detail::sfCollect<SfSnapshotAction>() == invalid,
        "failed rank source batch partially committed");
    detail::sfAll<AlterSfFixtureAction>(6);
    detail::requireSf(detail::sfCollect<SfSnapshotAction>() == invalid,
        "failed collective source retained a commit-able successful sibling");
    detail::sfAll<AlterSfFixtureAction>(1);
    detail::sfAll<AlterSfFixtureAction>(7);
    detail::sfAll<AlterSfFixtureAction>(2);
    auto const beforeConversion = detail::sfCollect<SfSnapshotAction>();
    domainsRebound();
    SphTreeCreateParameters treeParameters{};
    treeParameters.minimumRung = 0;
    sphTreeCreate(treeParameters);
    detail::requireSf(sphTreeAllValid(), "conversion fixture did not retain a valid octree");
    auto const conversion =
        starFormationClose(3, detail::kSfScale, 0.0, detail::sfStep(), detail::sfStep());
    after = detail::sfCollect<SfSnapshotAction>();
    detail::requireSf(!sphTreeAllValid(), "conversion left stale gas leaf ranges valid");
    detail::requireSf(
        conversion.converted == uint64_t(hpxSize()), "saved rate did not convert one gas per rank");
    detail::sfLedger(beforeConversion, after, conversion.heating);
    for (size_t rank = 0; rank < after.size(); ++rank) {
        detail::requireSf(
            after[rank][1].type == kStarType, "conversion did not change particle type");
        auto const& star = after[rank][1];
        auto const& gas = beforeConversion[rank][1];
        detail::requireSf(star.velocity == gas.velocity && star.mass == gas.mass &&
                star.position == gas.position && star.rung == gas.rung && star.id == gas.id,
            "conversion changed mass, momentum, position, rung or identity");
    }
    // Two pending conversions on the last rank compact the same gas array. Other
    // ranks retain one gas record, so the following collective includes an empty owner.
    detail::sfAll<AlterSfFixtureAction>(4);
    auto const beforeCompaction = detail::sfCollect<SfSnapshotAction>();
    auto const compacted =
        starFormationClose(0, detail::kSfScale, 0.0, detail::sfStep(), detail::sfStep());
    after = detail::sfCollect<SfSnapshotAction>();
    detail::requireSf(compacted.converted == uint64_t(hpxSize()) + 1,
        "multiple pending conversions lost a compacted gas record");
    detail::sfLedger(beforeCompaction, after, compacted.heating);
    for (size_t rank = 0; rank < after.size(); ++rank) {
        for (size_t particle = 0; particle < 3; ++particle) {
            auto const& original = beforeCompaction[rank][particle];
            auto const& current = after[rank][particle];
            detail::requireSf(current.id == original.id && current.position == original.position &&
                    current.velocity == original.velocity && current.rung == original.rung &&
                    current.mass == original.mass,
                "compaction detached main-particle state");
        }
        detail::requireSf(after[rank][2].type == kStarType &&
                after[rank][0].type == (rank + 1 == after.size() ? kStarType : kSphType),
            "conversion emptied the wrong gas owner");
    }
    detail::sfAll<AlterSfFixtureAction>(3);
    sphTreeCreate(treeParameters);
    detail::requireSf(sphTreeAllValid(), "octree could not rebuild with an empty gas owner");
    detail::sfAll<AlterSfFixtureAction>(5);
    auto const exhausted =
        starFormationClose(0, detail::kSfScale, 0.0, detail::sfStep(), detail::sfStep());
    detail::requireSf(exhausted.converted == uint64_t(hpxSize() - 1),
        "empty gas owner did not participate safely in source transaction");
    detail::sfAll<AlterSfFixtureAction>(3);
    auto const empty =
        starFormationClose(0, detail::kSfScale, 0.0, detail::sfStep(), detail::sfStep());
    detail::requireSf(empty.converted == 0 && empty.heating == 0.0,
        "exhausted gas population produced another source event");
    std::cout << "GADGET-4 star formation passed: full-step event-before-cooling, saved SFR, "
                 "inactive gas, electron state, energy ledger, failed-batch atomicity, "
                 "canonical entropy, checkpoint, compaction, octree invalidation, empty ranks, "
                 "and whole-particle conversion\n"
              << std::flush;
}
