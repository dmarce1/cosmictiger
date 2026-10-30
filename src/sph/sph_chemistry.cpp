/* CosmicTiger - primordial chemistry/SPH boundary; BSL-1.0. */
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hpx/runtime_local/runtime_local_fwd.hpp>

#include <cosmictiger/chemistry/primordial_chemistry.hpp>
#include <cosmictiger/chemistry/primordial_eos.hpp>
#include <cosmictiger/chemistry/primordial_float_environment.hpp>
#include <cosmictiger/chemistry/primordial_units.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {
using RadiationMember = float primordial::Radiation::*;
constexpr RadiationMember kRadiationMembers[] = {&primordial::Radiation::hydrogenIonization,
    &primordial::Radiation::heliumIonization, &primordial::Radiation::heliumPlusIonization,
    &primordial::Radiation::hydrogenMinusDetachment,
    &primordial::Radiation::molecularHydrogenDissociation,
    &primordial::Radiation::molecularHydrogenIonization,
    &primordial::Radiation::molecularHydrogenPlusDissociation,
    &primordial::Radiation::molecularHydrogenPlusIonization,
    &primordial::Radiation::heatingPerHydrogen, &primordial::Radiation::heatingPerHelium,
    &primordial::Radiation::heatingPerHeliumPlus, &primordial::Radiation::heatingPerHydrogenMinus,
    &primordial::Radiation::heatingPerMolecularHydrogen,
    &primordial::Radiation::heatingPerMolecularHydrogenPlus};
struct RadiationRow {
    double redshift;
    primordial::Radiation radiation;
};
struct RadiationTable {
    std::string path;
    uint64_t checksum{};
    std::vector<RadiationRow> rows;
};
RadiationTable const& radiationTable() {
    // Immutable once loaded: restarting checks the checksum of the supplied file.
    static RadiationTable const table = [] {
        RadiationTable result;
        result.path = getOptions().chemistry.radiationFile;
        if (!result.path.empty()) {
            std::ifstream file(result.path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Cannot open chemistry radiation table: " + result.path);
            }
            std::string const text{
                std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
            result.checksum = 14695981039346656037ULL;
            for (unsigned char const byte : text) {
                result.checksum = (result.checksum ^ byte) * 1099511628211ULL;
            }
            std::istringstream lines(text);
            std::string line;
            if (!std::getline(lines, line) || line != "COSMICTIGER_PRIMORDIAL_RADIATION 1") {
                throw std::runtime_error("Expected COSMICTIGER_PRIMORDIAL_RADIATION 1 header");
            }
            while (std::getline(lines, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                std::istringstream fields(line);
                RadiationRow row{};
                if (!(fields >> row.redshift) || !std::isfinite(row.redshift) ||
                    row.redshift <= -1.0 ||
                    (!result.rows.empty() && row.redshift <= result.rows.back().redshift)) {
                    throw std::runtime_error("Radiation redshifts must be finite, strictly "
                                             "increasing and greater than -1");
                }
                for (auto const member : kRadiationMembers) {
                    double value;
                    if (!(fields >> value) || !std::isfinite(value) || value < 0.0) {
                        throw std::runtime_error(
                            "Radiation table requires 14 finite nonnegative rate/heating columns");
                    }
                    row.radiation.*member =
                        primordial::checkedScalar(value, "radiation table entry");
                }
                std::string extra;
                if (fields >> extra) {
                    throw std::runtime_error("Unexpected radiation table column");
                }
                result.rows.push_back(row);
            }
            if (result.rows.size() < 2) {
                throw std::runtime_error("Radiation table requires at least two redshifts");
            }
        }
        return result;
    }();
    return table;
}
primordial::Environment environmentAt(
    double scaleFactor, float storedDensity = 0.0f, bool radiation = true) {
    auto const& options = getOptions();
    primordial::Environment environment;
    environment.network = options.chemistry.network == "atomic6" ? primordial::Network::Atomic6 :
                                                                   primordial::Network::Molecular9;
    double const hydrogen = 1.0 - options.cosmology.heliumFraction;
    environment.heliumToHydrogenNumberRatio = primordial::checkedScalar(
        options.cosmology.heliumFraction / (4.0 * hydrogen), "He/H number ratio");
    environment.cmbTemperature = primordial::checkedScalar(
        2.73 * options.cosmology.thetaCmb / scaleFactor, "CMB temperature");
    auto const scales = units::scalesFromOptions(options);
    units::CosmologicalContext const context(scaleFactor, options.cosmology.hubble);
    environment.nHydrogen = primordial::hydrogenNumberDensity(
        units::toProperDensity(units::StoredDensity(storedDensity), scales, context), hydrogen);
    if (radiation && !options.chemistry.radiationFile.empty()) {
        auto const& rows = radiationTable().rows;
        double const redshift = 1.0 / scaleFactor - 1.0;
        if (redshift < rows.front().redshift || redshift > rows.back().redshift) {
            throw std::runtime_error("Current redshift is outside the chemistry radiation table");
        }
        auto upper = std::upper_bound(rows.begin(), rows.end(), redshift,
            [](double value, RadiationRow const& row) { return value < row.redshift; });
        if (upper == rows.end()) {
            environment.radiation = rows.back().radiation;
        } else {
            auto const& lower = *(upper - 1);
            float const weight = static_cast<float>(
                (redshift - lower.redshift) / (upper->redshift - lower.redshift));
            for (auto const member : kRadiationMembers) {
                environment.radiation.*member =
                    std::lerp(lower.radiation.*member, upper->radiation.*member, weight);
            }
        }
    }
    return environment;
}
primordial::State stateAt(primordial::State state, float storedEnergy, double scaleFactor) {
    auto const& options = getOptions();
    state.thermalEnergy = primordial::energyPerHydrogen(
        units::toPhysicalSpecificEnergy(units::StoredSpecificEnergy(storedEnergy),
            units::scalesFromOptions(options),
            units::CosmologicalContext(scaleFactor, options.cosmology.hubble)),
        1.0 - options.cosmology.heliumFraction);
    return state;
}
float storedEnergyAt(float energy, double scaleFactor) {
    auto const& options = getOptions();
    return primordial::checkedScalar(
        units::toStoredSpecificEnergy(
            primordial::physicalEnergy(energy, 1.0 - options.cosmology.heliumFraction),
            units::scalesFromOptions(options),
            units::CosmologicalContext(scaleFactor, options.cosmology.hubble))
            .value(),
        "stored a^2 u");
}

std::vector<primordial::StepResult> integrateChemistryBatch(
    std::vector<primordial::BatchInput> const& inputs) {
    auto const& options = getOptions();
    primordial::SolverOptions solverOptions;
    solverOptions.relativeTolerance = options.chemistry.relativeTolerance;
    for (int i = 0; i < primordial::kAbundanceCount; ++i) {
        solverOptions.abundanceAbsoluteTolerance[i] = options.chemistry.abundanceAbsoluteTolerance;
    }
    solverOptions.energyAbsoluteTolerance = options.chemistry.energyAbsoluteTolerance;
    solverOptions.maxSubsteps = options.chemistry.maxSubsteps;
    std::vector<primordial::StepResult> results(inputs.size());
    std::string error;
    bool const dispatched = inputs.empty() ||
        primordial::integrateBatch(inputs.data(), results.data(), inputs.size(), solverOptions,
            options.compute.cuda ? primordial::Backend::Cuda : primordial::Backend::Cpu,
            static_cast<unsigned>(hpx::get_num_worker_threads()), &error);
    if (!dispatched) {
        throw std::runtime_error("Chemistry batch dispatch failed: " + error);
    }
    for (size_t index = 0; index < inputs.size(); ++index) {
        auto const& result = results[index];
        if (result.status != primordial::SolverStatus::Success ||
            result.acceptedTime != inputs[index].interval) {
            throw std::runtime_error("Chemistry failed on rank " + std::to_string(hpxRank()) +
                " batch index " + std::to_string(index) + " status " +
                std::to_string(int(result.status)) + "; source update was not committed");
        }
    }
    return results;
}

std::vector<ParticleInteger> pendingIndices;
std::vector<int> pendingRungs;
std::vector<primordial::StepResult> pendingResults;
std::vector<float> pendingEnergies;
std::vector<SphChemistryStep> pendingSteps;
}    // namespace detail

primordial::State sphChemistrySynchronizeEnergy(
    primordial::State const& state, float storedEnergy, double scaleFactor) {
    primordial::StrictFloatEnvironment const floatingEnvironment;
    if (!floatingEnvironment.valid()) {
        throw std::runtime_error("Chemistry requires IEEE nearest rounding and gradual underflow");
    }
    return detail::stateAt(state, storedEnergy, scaleFactor);
}

std::vector<SphChemistryBatchResult> sphChemistryAdvanceBatch(
    std::vector<SphChemistryBatchInput> const& inputs, double scaleFactor) {
    primordial::StrictFloatEnvironment const floatingEnvironment;
    if (!floatingEnvironment.valid()) {
        throw std::runtime_error("Chemistry requires IEEE nearest rounding and gradual underflow");
    }
    if (!(scaleFactor > 0.0) || !std::isfinite(scaleFactor)) {
        throw std::invalid_argument("Chemistry batch requires a finite positive scale factor");
    }
    std::vector<primordial::BatchInput> physicalInputs;
    physicalInputs.reserve(inputs.size());
    for (auto const& input : inputs) {
        if (input.physicalInterval < 0.0 || !std::isfinite(input.physicalInterval)) {
            throw std::invalid_argument("Chemistry batch requires finite nonnegative intervals");
        }
        physicalInputs.push_back({detail::stateAt(input.state, input.storedEnergy, scaleFactor),
            detail::environmentAt(scaleFactor, input.storedDensity),
            primordial::checkedScalar(input.physicalInterval, "physical interval in seconds")});
    }
    auto const integrated = detail::integrateChemistryBatch(physicalInputs);
    std::vector<SphChemistryBatchResult> results;
    results.reserve(inputs.size());
    for (size_t index = 0; index < inputs.size(); ++index) {
        float const storedEnergy = inputs[index].physicalInterval == 0.0 ?
            inputs[index].storedEnergy :
            detail::storedEnergyAt(integrated[index].state.thermalEnergy, scaleFactor);
        results.push_back({integrated[index].state, storedEnergy});
    }
    return results;
}

primordial::Eos sphChemistryEos(
    primordial::State const& state, float storedEnergy, double scaleFactor) {
    primordial::StrictFloatEnvironment const floatingEnvironment;
    if (!floatingEnvironment.valid()) {
        throw std::runtime_error("Chemistry requires IEEE nearest rounding and gradual underflow");
    }
    primordial::Eos eos;
    auto const status = primordial::evaluateEos(detail::stateAt(state, storedEnergy, scaleFactor),
        detail::environmentAt(scaleFactor, 0.0f, false), eos);
    if (status != primordial::Status::Success) {
        throw std::runtime_error(
            "Chemistry EOS rejected gas state (status " + std::to_string(int(status)) + ")");
    }
    return eos;
}
void sphChemistryInitializeParticle(long long index, double scaleFactor) {
    auto const& options = getOptions();
    primordial::StrictFloatEnvironment const floatingEnvironment;
    if (!floatingEnvironment.valid()) {
        throw std::runtime_error("Chemistry requires IEEE nearest rounding and gradual underflow");
    }
    auto& state = sphParticlesChemistryStates[index];
    double const fractions[] = {options.chemistry.hydrogenIonFraction,
        options.chemistry.heliumIonFraction, options.chemistry.heliumDoubleIonFraction,
        options.chemistry.hydrogenMinusFraction, options.chemistry.molecularFraction,
        options.chemistry.molecularIonFraction};
    for (int i = 0; i < primordial::kAbundanceCount; ++i) {
        bool const upper = fractions[i] > 0.5;
        state.abundances[i] = primordial::SignedAbundance::fromMinority(
            static_cast<float>(upper ? 1.0 - fractions[i] : fractions[i]), upper);
    }
    auto const status =
        primordial::thermalEnergyAtTemperature(state, detail::environmentAt(scaleFactor),
            static_cast<float>(options.chemistry.initialTemperature), state.thermalEnergy);
    if (status != primordial::Status::Success) {
        throw std::runtime_error(
            "Initial chemistry composition/temperature is outside the selected model");
    }
    sphParticlesInternalEnergy(index) = detail::storedEnergyAt(state.thermalEnergy, scaleFactor);
    sphParticlesChemistrySteps[index] = {};
}

double sphChemistryPrepare(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, bool opening, double explicitTimestep, int uniformRung);
void sphChemistryCommit();
HPX_PLAIN_ACTION(sphChemistryPrepare, SphChemistryPrepareAction);
HPX_PLAIN_ACTION(sphChemistryCommit, SphChemistryCommitAction);

double sphChemistryPrepare(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, bool opening, double explicitTimestep, int uniformRung) {
    std::vector<hpx::future<double>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphChemistryPrepareAction>(child, minimumRung, scaleFactor,
            scaleDerivative, baseTimestep, opening, explicitTimestep, uniformRung));
    }
    double heating = 0.0;
    std::exception_ptr failure;
    try {
        auto const& options = getOptions();
        auto const scales = units::scalesFromOptions(options);
        std::vector<primordial::BatchInput> inputs;
        detail::pendingIndices.clear();
        detail::pendingRungs.clear();
        detail::pendingSteps.clear();
        auto append = [&](ParticleInteger i, primordial::State const& state, float energy,
                          double halfInterval, int rung) {
            if (!(halfInterval > 0.0) || !std::isfinite(halfInterval)) {
                throw std::runtime_error("Chemistry requires a positive pending SPH half-interval");
            }
            double const midpointScale =
                scaleFactor + (opening ? 0.5 : -0.5) * scaleDerivative * halfInterval;
            primordial::BatchInput input;
            input.state = detail::stateAt(state, energy, scaleFactor);
            input.environment = detail::environmentAt(scaleFactor, sphParticlesDensity(i));
            input.interval = primordial::physicalSeconds(
                units::toPhysicalInterval(units::ConformalTime(halfInterval * scales.timeScale()),
                    units::CosmologicalContext(midpointScale, options.cosmology.hubble)));
            inputs.push_back(input);
            detail::pendingIndices.push_back(i);
            detail::pendingRungs.push_back(rung);
            detail::pendingSteps.push_back({opening ? halfInterval : 0.0, 0.0});
        };
        if (!std::isfinite(explicitTimestep)) {
            throw std::runtime_error("Explicit chemistry/SPH timestep is not finite");
        }
        for (ParticleInteger i = 0; i < sphParticlesSize(); ++i) {
            int const currentRung = sphParticlesRung(i);
            if (currentRung < minimumRung) {
                continue;
            }
            int const rung = opening && uniformRung >= 0 ? uniformRung : currentRung;
            if (rung < 0 || rung >= kMaxRung) {
                throw std::runtime_error("Chemistry/SPH requires an unsupported timestep rung");
            }
            double const selectedHalf = explicitTimestep >= 0.0 ?
                0.5 * explicitTimestep :
                std::ldexp(baseTimestep, -1 - rung);
            auto const& step = sphParticlesChemistrySteps[i];
            if (opening) {
                if (step.halfInterval != 0.0 || step.pendingHeating != 0.0) {
                    throw std::runtime_error(
                        "Chemistry opening attempted before closing the old interval");
                }
            } else {
                if (!std::isfinite(step.pendingHeating)) {
                    throw std::runtime_error("Pending chemistry heating is not finite");
                }
                if (step.halfInterval != selectedHalf) {
                    throw std::runtime_error(
                        "Pending chemistry interval differs from the old SPH timestep");
                }
            }
            append(i, sphParticlesChemistryStates[i], sphParticlesInternalEnergy(i),
                opening ? selectedHalf : step.halfInterval, rung);
        }
        detail::pendingResults = detail::integrateChemistryBatch(inputs);
        detail::pendingEnergies.resize(inputs.size());
        for (size_t j = 0; j < inputs.size(); ++j) {
            ParticleInteger const i = detail::pendingIndices[j];
            auto const& result = detail::pendingResults[j];
            detail::pendingEnergies[j] =
                detail::storedEnergyAt(result.state.thermalEnergy, scaleFactor);
            float const beforeEnergy = sphParticlesInternalEnergy(i);
            double const change = -options.initial.gasMass *
                double(detail::pendingEnergies[j] - beforeEnergy) / scaleFactor;
            if (opening) {
                detail::pendingSteps[j].pendingHeating = change;
            } else {
                heating += change + sphParticlesChemistrySteps[i].pendingHeating;
            }
        }
    } catch (...) {
        failure = std::current_exception();
    }
    // Drain every prepare before reporting failure. A caller may retry, and no
    // previous rank's work may still be writing the next transaction's buffers.
    for (auto& future : remote) {
        try {
            heating += future.get();
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    return heating;
}
void sphChemistryCommit() {
    std::vector<hpx::future<void>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphChemistryCommitAction>(child));
    }
    for (size_t j = 0; j < detail::pendingResults.size(); ++j) {
        ParticleInteger const i = detail::pendingIndices[j];
        sphParticlesChemistryStates[i] = detail::pendingResults[j].state;
        sphParticlesInternalEnergy(i) = detail::pendingEnergies[j];
        sphParticlesChemistrySteps[i] = detail::pendingSteps[j];
        sphParticlesRung(i) = detail::pendingRungs[j];
    }
    detail::pendingIndices.clear();
    detail::pendingRungs.clear();
    detail::pendingResults.clear();
    detail::pendingEnergies.clear();
    detail::pendingSteps.clear();
    for (auto& future : remote) {
        future.get();
    }
}
double sphChemistryClose(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep) {
    // CKDKC: the old hydro corrector has finished before this closing C.
    // Prepare every rank before committing any source update.
    double const heating = sphChemistryPrepare(
        minimumRung, scaleFactor, scaleDerivative, baseTimestep, false, explicitTimestep, -1);
    sphChemistryCommit();
    sphParticlesClearHydroCaches();
    return heating;
}
void sphChemistryOpen(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep, int uniformRung) {
    // CKDKC: use the selected SPH interval, with only the local source solve subcycled.
    sphChemistryPrepare(minimumRung, scaleFactor, scaleDerivative, baseTimestep, true,
        explicitTimestep, uniformRung);
    sphChemistryCommit();
    sphParticlesClearHydroCaches();
}
std::array<double, 17> sphChemistryCheckpointParameters() {
    auto const& c = getOptions().chemistry;
    return {double(primordial::kPhysicsVersion), c.network == "atomic6" ? 6.0 : 9.0, 0.0,
        c.initialTemperature, c.hydrogenIonFraction, c.heliumIonFraction, c.heliumDoubleIonFraction,
        c.hydrogenMinusFraction, c.molecularFraction, c.molecularIonFraction, c.relativeTolerance,
        c.abundanceAbsoluteTolerance, c.energyAbsoluteTolerance, double(c.maxSubsteps),
        getOptions().cosmology.heliumFraction, getOptions().cosmology.thetaCmb, 2.0};
}
uint64_t sphChemistryRadiationChecksum() {
    return detail::radiationTable().checksum;
}

std::array<float, 12> sphChemistryOutput(
    primordial::State const& state, float storedEnergy, double scaleFactor) {
    primordial::StrictFloatEnvironment const floatingEnvironment;
    if (!floatingEnvironment.valid()) {
        throw std::runtime_error("Chemistry requires IEEE nearest rounding and gradual underflow");
    }
    auto const c = primordial::composition(state, detail::environmentAt(scaleFactor, 0.0f, false));
    auto const eos = sphChemistryEos(state, storedEnergy, scaleFactor);
    return {c.hydrogen.toFloat(), c.hydrogenPlus.toFloat(), c.helium.toFloat(),
        c.heliumPlus.toFloat(), c.heliumDoublePlus.toFloat(), c.hydrogenMinus.toFloat(),
        c.molecularHydrogen.toFloat(), c.molecularHydrogenPlus.toFloat(), c.electrons.toFloat(),
        eos.temperature, eos.meanMolecularWeight, eos.adiabaticIndex};
}

HPX_PLAIN_ACTION(sphChemistryPressureIntegral, SphChemistryPressureIntegralAction);
double sphChemistryPressureIntegral(double scaleFactor) {
    std::vector<hpx::future<double>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphChemistryPressureIntegralAction>(child, scaleFactor));
    }
    double sum = 0.0;
    for (ParticleInteger i = 0; i < sphParticlesSize(); ++i) {
        auto const eos = sphChemistryEos(
            sphParticlesChemistryStates[i], sphParticlesInternalEnergy(i), scaleFactor);
        sum += getOptions().initial.gasMass * double(eos.pressureEnergyRatio) *
            sphParticlesInternalEnergy(i) / (scaleFactor * scaleFactor);
    }
    for (auto& future : remote) {
        sum += future.get();
    }
    return sum;
}
