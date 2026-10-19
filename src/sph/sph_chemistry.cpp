/* CosmicTiger - primordial chemistry/SPH boundary; GPL-2.0-or-later. */
#include <algorithm>
#include <array>
#include <cmath>
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

namespace {
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
std::vector<primordial::StepResult> pendingResults;
std::vector<float> pendingEnergies;
}    // namespace

primordial::Eos sphChemistryEos(
    primordial::State const& state, float storedEnergy, double scaleFactor) {
    primordial::StrictFloatEnvironment const floatingEnvironment;
    if (!floatingEnvironment.valid()) {
        throw std::runtime_error("Chemistry requires IEEE nearest rounding and gradual underflow");
    }
    primordial::Eos eos;
    auto const status = primordial::evaluateEos(
        stateAt(state, storedEnergy, scaleFactor), environmentAt(scaleFactor, 0.0f, false), eos);
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
    auto const status = primordial::thermalEnergyAtTemperature(state, environmentAt(scaleFactor),
        static_cast<float>(options.chemistry.initialTemperature), state.thermalEnergy);
    if (status != primordial::Status::Success) {
        throw std::runtime_error(
            "Initial chemistry composition/temperature is outside the selected model");
    }
    sphParticlesInternalEnergy(index) = storedEnergyAt(state.thermalEnergy, scaleFactor);
    sphParticlesRung(index) = options.chemistry.synchronizedRung;
}

double sphChemistryPrepare(
    double scaleFactor, double intervalScaleFactor, double conformalInterval);
void sphChemistryCommit();
HPX_PLAIN_ACTION(sphChemistryPrepare, SphChemistryPrepareAction);
HPX_PLAIN_ACTION(sphChemistryCommit, SphChemistryCommitAction);
HPX_PLAIN_ACTION(sphChemistrySetRungs, SphChemistrySetRungsAction);

double sphChemistryPrepare(
    double scaleFactor, double intervalScaleFactor, double conformalInterval) {
    std::vector<hpx::future<double>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphChemistryPrepareAction>(
            child, scaleFactor, intervalScaleFactor, conformalInterval));
    }
    auto const& options = getOptions();
    auto const scales = units::scalesFromOptions(options);
    float const seconds = primordial::physicalSeconds(
        units::toPhysicalInterval(units::ConformalTime(conformalInterval * scales.timeScale()),
            units::CosmologicalContext(intervalScaleFactor, options.cosmology.hubble)));
    std::vector<primordial::BatchInput> inputs(sphParticlesSize());
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i].state =
            stateAt(sphParticlesChemistryStates[i], sphParticlesInternalEnergy(i), scaleFactor);
        inputs[i].environment = environmentAt(scaleFactor, sphParticlesDensity(i));
        inputs[i].interval = seconds;
    }
    primordial::SolverOptions solverOptions;
    solverOptions.relativeTolerance = options.chemistry.relativeTolerance;
    for (int i = 0; i < primordial::kAbundanceCount; ++i) {
        solverOptions.abundanceAbsoluteTolerance[i] = options.chemistry.abundanceAbsoluteTolerance;
    }
    solverOptions.energyAbsoluteTolerance = options.chemistry.energyAbsoluteTolerance;
    solverOptions.maxSubsteps = options.chemistry.maxSubsteps;
    pendingResults.resize(inputs.size());
    pendingEnergies.resize(inputs.size());
    std::string error;
    bool const dispatched = options.compute.cuda ?
        primordial::integrateBatchCuda(
            inputs.data(), pendingResults.data(), inputs.size(), solverOptions, &error) :
        primordial::integrateBatchCpu(inputs.data(), pendingResults.data(), inputs.size(),
            solverOptions, static_cast<unsigned>(hpx::get_num_worker_threads()));
    if (!dispatched) {
        throw std::runtime_error("Chemistry batch dispatch failed: " + error);
    }
    double heating = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto const& result = pendingResults[i];
        if (result.status != primordial::SolverStatus::Success || result.acceptedTime != seconds) {
            throw std::runtime_error("Chemistry failed on rank " + std::to_string(hpxRank()) +
                " gas index " + std::to_string(i) + " status " +
                std::to_string(int(result.status)) + "; source update was not committed");
        }
        pendingEnergies[i] = storedEnergyAt(result.state.thermalEnergy, scaleFactor);
        heating -= options.initial.gasMass *
            double(pendingEnergies[i] - sphParticlesInternalEnergy(i)) / scaleFactor;
    }
    for (auto& future : remote) {
        heating += future.get();
    }
    return heating;
}
void sphChemistryCommit() {
    std::vector<hpx::future<void>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphChemistryCommitAction>(child));
    }
    if (pendingResults.size() != size_t(sphParticlesSize())) {
        throw std::runtime_error("Chemistry transaction size changed");
    }
    for (size_t i = 0; i < pendingResults.size(); ++i) {
        sphParticlesChemistryStates[i] = pendingResults[i].state;
        sphParticlesInternalEnergy(i) = pendingEnergies[i];
    }
    pendingResults.clear();
    pendingEnergies.clear();
    for (auto& future : remote) {
        future.get();
    }
}
double sphChemistryAdvance(
    double scaleFactor, double intervalScaleFactor, double conformalInterval) {
    // Strang's symmetric source splitting: SIAM J. Numer. Anal. 5 (1968),
    // 506-517, doi:10.1137/0705041. Prepare every rank before committing any.
    double const heating = sphChemistryPrepare(scaleFactor, intervalScaleFactor, conformalInterval);
    sphChemistryCommit();
    sphParticlesClearHydroCaches();
    return heating;
}
void sphChemistrySetRungs() {
    std::vector<hpx::future<void>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphChemistrySetRungsAction>(child));
    }
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        particlesRung(i) = getOptions().chemistry.synchronizedRung;
    }
    for (auto& future : remote) {
        future.get();
    }
}
std::array<double, 17> sphChemistryCheckpointParameters() {
    auto const& c = getOptions().chemistry;
    return {double(primordial::kPhysicsVersion), c.network == "atomic6" ? 6.0 : 9.0,
        double(c.synchronizedRung), c.initialTemperature, c.hydrogenIonFraction,
        c.heliumIonFraction, c.heliumDoubleIonFraction, c.hydrogenMinusFraction,
        c.molecularFraction, c.molecularIonFraction, c.relativeTolerance,
        c.abundanceAbsoluteTolerance, c.energyAbsoluteTolerance, double(c.maxSubsteps),
        getOptions().cosmology.heliumFraction, getOptions().cosmology.thetaCmb, 1.0};
}
uint64_t sphChemistryRadiationChecksum() {
    return radiationTable().checksum;
}

std::array<float, 12> sphChemistryOutput(
    primordial::State const& state, float storedEnergy, double scaleFactor) {
    primordial::StrictFloatEnvironment const floatingEnvironment;
    if (!floatingEnvironment.valid()) {
        throw std::runtime_error("Chemistry requires IEEE nearest rounding and gradual underflow");
    }
    auto const c = primordial::composition(state, environmentAt(scaleFactor, 0.0f, false));
    auto const eos = sphChemistryEos(state, storedEnergy, scaleFactor);
    return {c.hydrogen.toFloat(), c.hydrogenPlus.toFloat(), c.helium.toFloat(),
        c.heliumPlus.toFloat(), c.heliumDoublePlus.toFloat(), c.hydrogenMinus.toFloat(),
        c.molecularHydrogen.toFloat(), c.molecularHydrogenPlus.toFloat(), c.electrons.toFloat(),
        eos.temperature, eos.meanMolecularWeight, eos.adiabaticIndex};
}

double& sphChemistryPendingHeating() {
    static double heating = 0.0;
    return heating;
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
