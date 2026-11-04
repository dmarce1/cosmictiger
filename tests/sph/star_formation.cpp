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
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/stars/star_formation.hpp>

namespace detail {
constexpr double kSfScale = 0.25;
struct SfRecord {
    uint64_t id{}, eventCounter{}, formationTime{}, specificRate{}, elapsedTime{};
    uint32_t energy{}, electrons{};
    std::array<uint32_t, kDimensions> velocity{};
    uint64_t rung{}, type{};
    uint32_t density{};
    bool operator==(SfRecord const&) const = default;
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & id & eventCounter & formationTime & specificRate & elapsedTime & energy &
            electrons & velocity & rung & type & density;
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
    return 1.e11 / (kSfScale * getOptions().units.timeSeconds);
}
void initializeSfFixture() {
    particlesResize(3);
    sphParticlesResize(3, false);
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
        sphParticlesSupportRadius(index) = 0.125f;
        sphParticlesElectronFractions[index] = 1.0f;
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
        record.electrons =
            std::bit_cast<uint32_t>(gas ? sphParticlesElectronFractions[gasIndex] : 0.0f);
        for (int dim = 0; dim < kDimensions; ++dim) {
            record.velocity[dim] = std::bit_cast<uint32_t>(particlesVel(dim, index));
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
        record.energy = std::bit_cast<uint32_t>(output.energy);
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
    double expected = 0.0;
    for (size_t rank = 0; rank < before.size(); ++rank) {
        for (size_t particle = 0; particle < 3; ++particle) {
            expected -= getOptions().initial.gasMass *
                (std::bit_cast<float>(after[rank][particle].energy) -
                    std::bit_cast<float>(before[rank][particle].energy)) /
                kSfScale;
        }
    }
    requireSfClose(heating, expected, "full-source energy ledger is incorrect", 2.e-12);
}
}    // namespace detail

void starFormationTest() {
    detail::requireSf(getOptions().starFormation.enabled && getOptions().chemistry.enabled,
        "fixture requires cooling and SF");
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
    detail::sfAll<AlterSfFixtureAction>(1);
    detail::sfAll<AlterSfFixtureAction>(2);
    auto const beforeConversion = detail::sfCollect<SfSnapshotAction>();
    auto const conversion =
        starFormationClose(3, detail::kSfScale, 0.0, detail::sfStep(), detail::sfStep());
    after = detail::sfCollect<SfSnapshotAction>();
    detail::requireSf(
        conversion.converted == uint64_t(hpxSize()), "saved rate did not convert one gas per rank");
    detail::sfLedger(beforeConversion, after, conversion.heating);
    for (size_t rank = 0; rank < after.size(); ++rank) {
        detail::requireSf(
            after[rank][1].type == kStarType, "conversion did not change particle type");
        detail::requireSf(after[rank][1].velocity == beforeConversion[rank][1].velocity,
            "conversion changed momentum");
    }
    starFormationOpenNewStars(detail::sfStep(), detail::kSfScale, detail::sfStep(), 3, false);
    std::cout << "GADGET-4 star formation passed: full-step event-before-cooling, saved SFR, "
                 "inactive gas, electron state, energy ledger, failed-batch atomicity, "
                 "checkpoint, and whole-particle conversion\n"
              << std::flush;
}
