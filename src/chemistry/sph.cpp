/* CosmicTiger - GADGET-4 equilibrium cooling/SPH boundary; BSL-1.0. */
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <hpx/runtime_local/runtime_local_fwd.hpp>

#include <cosmictiger/chemistry/gadget4_cooling.hpp>
#include <cosmictiger/chemistry/sph.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/sph/particles.hpp>

namespace detail {
double coolingPhysicalEnergy(float energy, double scaleFactor) {
    auto const& options = getOptions();
    return units::toPhysicalSpecificEnergy(units::StoredSpecificEnergy(energy),
        units::scalesFromOptions(options),
        units::CosmologicalContext(scaleFactor, options.cosmology.hubble))
        .quantity()
        .value();
}
double coolingPhysicalDensity(float density, double scaleFactor) {
    auto const& options = getOptions();
    return units::toProperDensity(units::StoredDensity(density), units::scalesFromOptions(options),
        units::CosmologicalContext(scaleFactor, options.cosmology.hubble))
        .quantity()
        .value();
}
float coolingStoredEnergy(double energy, double scaleFactor) {
    auto const& options = getOptions();
    double const result = units::toStoredSpecificEnergy(
        units::PhysicalSpecificEnergy(units::SpecificEnergy::from_value(energy)),
        units::scalesFromOptions(options),
        units::CosmologicalContext(scaleFactor, options.cosmology.hubble))
                              .value();
    if (!gadget4::detail::finite(result) || result <= 0.0 ||
        result > std::numeric_limits<float>::max() || result < std::numeric_limits<float>::min()) {
        throw std::range_error("Cooling returned unrepresentable stored specific energy");
    }
    return static_cast<float>(result);
}
std::vector<gadget4::Result> integrateCoolingBatch(std::vector<gadget4::Input> const& inputs,
    gadget4::Parameters const& parameters,
    std::vector<SphChemistryBatchInput> const& sourceInputs) {
    std::vector<gadget4::Result> results(inputs.size());
    std::string error;
    bool success = true;
    if (getOptions().chemistry.cuda) {
#ifdef USE_CUDA
        cudaSetDevice();
#endif
        success = gadget4::integrateBatch(inputs.data(), results.data(), inputs.size(), parameters,
            gadget4::Backend::Cuda, 1, &error);
    } else if (!inputs.empty()) {
        size_t const workers = std::min<size_t>(hpx::get_num_worker_threads(), inputs.size());
        std::vector<hpx::future<bool>> futures;
        for (size_t worker = 0; worker < workers; ++worker) {
            size_t const begin = inputs.size() * worker / workers;
            size_t const end = inputs.size() * (worker + 1) / workers;
            futures.push_back(hpx::async([&, begin, end]() {
                return gadget4::integrateBatchCpu(
                    inputs.data() + begin, results.data() + begin, end - begin, parameters, 1);
            }));
        }
        hpx::wait_all(futures);
        for (auto& future : futures) {
            success = future.get() && success;
        }
    }
    if (!success) {
        throw std::runtime_error("GADGET-4 cooling dispatch failed: " + error);
    }
    for (size_t index = 0; index < inputs.size(); ++index) {
        if (results[index].status != gadget4::Status::Success) {
            auto const& input = inputs[index];
            std::ostringstream message;
            message << std::scientific << std::setprecision(17)
                    << "GADGET-4 cooling failed on rank " << hpxRank() << " gas particle "
                    << sourceInputs[index].particleIndex << " batch index " << index << ": "
                    << gadget4::statusName(results[index].status) << " rho=" << input.density
                    << " u=" << input.specificEnergy << " Ne=" << input.electronFraction
                    << " seconds=" << input.interval << "; no source was committed";
            throw std::runtime_error(message.str());
        }
    }
    return results;
}
struct PendingCoolingUpdate {
    ParticleInteger gas;
    float entropy;
    float electronFraction;
};
std::vector<PendingCoolingUpdate> pendingCooling;
}    // namespace detail

gadget4::Parameters sphCoolingParameters(double scaleFactor) {
    auto const& options = getOptions();
    return gadget4::makeParameters(
        scaleFactor, options.run.scenario == "cosmology", options.chemistry.minimumSpecificEnergy);
}
std::vector<SphChemistryBatchResult> sphChemistryAdvanceBatch(
    std::vector<SphChemistryBatchInput> const& inputs, double scaleFactor) {
    if (!(scaleFactor > 0.0) || !gadget4::detail::finite(scaleFactor)) {
        throw std::invalid_argument("Cooling requires a finite positive scale factor");
    }
    std::vector<gadget4::Input> physicalInputs;
    physicalInputs.reserve(inputs.size());
    for (auto const& input : inputs) {
        gadget4::Input physicalInput;
        physicalInput.electronFraction = input.electronFraction;
        physicalInput.specificEnergy =
            detail::coolingPhysicalEnergy(input.storedEnergy, scaleFactor);
        physicalInput.density = detail::coolingPhysicalDensity(input.storedDensity, scaleFactor);
        physicalInput.interval = input.physicalInterval;
        physicalInputs.push_back(physicalInput);
    }
    auto const integrated =
        detail::integrateCoolingBatch(physicalInputs, sphCoolingParameters(scaleFactor), inputs);
    std::vector<SphChemistryBatchResult> results;
    results.reserve(inputs.size());
    for (auto const& result : integrated) {
        SphChemistryBatchResult stored;
        stored.electronFraction = result.electronFraction;
        stored.storedEnergy = detail::coolingStoredEnergy(result.specificEnergy, scaleFactor);
        results.push_back(stored);
    }
    return results;
}
void sphChemistryInitializeParticle(long long index, double scaleFactor) {
    auto const& options = getOptions();
    sphParticlesElectronFractions[index] = 1.0f;
    double energy = detail::coolingPhysicalEnergy(sphParticlesInternalEnergy(index), scaleFactor);
    // GADGET-4 InitGasTemp initializes only missing (zero) IC energy.
    if (energy == 0.0 && options.chemistry.initializationTemperature > 0.0) {
        double const temperature = options.chemistry.initializationTemperature;
        double const mu = temperature > 1.0e4 ?
            4.0 / (8.0 - 5.0 * (1.0 - gadget4::kHydrogenMassFraction)) :
            4.0 / (1.0 + 3.0 * gadget4::kHydrogenMassFraction);
        energy = gadget4::kBoltzmann * temperature /
            (gadget4::kGammaMinusOne * gadget4::kProtonMass * mu);
    }
    sphParticlesSetInternalEnergy(index,
        detail::coolingStoredEnergy(
            std::max(energy, options.chemistry.minimumSpecificEnergy), scaleFactor));
}

double sphChemistryPhysicalInterval(double interval, double scaleFactor) {
    if (!(interval > 0.0) || !gadget4::detail::finite(interval)) {
        throw std::invalid_argument("Cooling requires a positive completed interval");
    }
    auto const& options = getOptions();
    auto const scales = units::scalesFromOptions(options);
    units::CosmologicalContext const context(scaleFactor, options.cosmology.hubble);
    return units::toPhysicalInterval(units::ConformalTime(interval * scales.timeScale()), context)
        .quantity()
        .value();
}

double sphCoolingPrepare(
    int minimumRung, double scaleFactor, double baseTimestep, double explicitTimestep);
void sphCoolingCommit();
void sphCoolingDiscard();
HPX_PLAIN_ACTION(sphCoolingPrepare, SphCoolingPrepareAction);
HPX_PLAIN_ACTION(sphCoolingCommit, SphCoolingCommitAction);
HPX_PLAIN_ACTION(sphCoolingDiscard, SphCoolingDiscardAction);

double sphCoolingPrepare(
    int minimumRung, double scaleFactor, double baseTimestep, double explicitTimestep) {
    std::vector<hpx::future<double>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphCoolingPrepareAction>(
            child, minimumRung, scaleFactor, baseTimestep, explicitTimestep));
    }
    double heating = 0.0;
    std::exception_ptr failure;
    try {
        std::vector<ParticleInteger> indices;
        std::vector<double> intervals;
        for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
            int const rung = sphParticlesRung(index);
            if (rung < minimumRung) {
                continue;
            }
            if (rung < 0 || rung >= kMaxRung) {
                throw std::runtime_error("Cooling received an unsupported SPH rung");
            }
            indices.push_back(index);
            intervals.push_back(
                explicitTimestep >= 0.0 ? explicitTimestep : std::ldexp(baseTimestep, -rung));
        }
        heating = sphCoolingPrepareLocal(indices, intervals, scaleFactor);
    } catch (...) {
        failure = std::current_exception();
    }
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
        sphCoolingDiscard();
        std::rethrow_exception(failure);
    }
    return heating;
}
void sphCoolingCommit() {
    std::vector<hpx::future<void>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphCoolingCommitAction>(child));
    }
    sphCoolingCommitLocal();
    for (auto& future : remote) {
        future.get();
    }
}
void sphCoolingDiscard() {
    std::vector<hpx::future<void>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphCoolingDiscardAction>(child));
    }
    detail::pendingCooling.clear();
    for (auto& future : remote) {
        future.get();
    }
}
double sphCoolingPrepareLocal(std::vector<ParticleInteger> const& indices,
    std::vector<double> const& intervals, double scaleFactor) {
    detail::pendingCooling.clear();
    if (indices.size() != intervals.size()) {
        throw std::invalid_argument("Cooling slots and intervals differ in count");
    }
    if (!sphParticlesEntropyInitialized || sphParticlesThermalScaleFactor != scaleFactor) {
        throw std::logic_error("Cooling requires initialized entropy at the source scale factor");
    }
    auto const& options = getOptions();
    auto const scales = units::scalesFromOptions(options);
    units::CosmologicalContext const context(scaleFactor, options.cosmology.hubble);
    std::vector<SphChemistryBatchInput> inputs;
    inputs.reserve(indices.size());
    std::unordered_set<ParticleInteger> selected;
    selected.reserve(indices.size());
    for (size_t selectedIndex = 0; selectedIndex < indices.size(); ++selectedIndex) {
        ParticleInteger const index = indices[selectedIndex];
        if (index < 0 || index >= sphParticlesSize() || !selected.insert(index).second) {
            throw std::invalid_argument("Cooling requires distinct owned gas slots");
        }
        double const interval = intervals[selectedIndex];
        if (!(interval > 0.0) || !gadget4::detail::finite(interval)) {
            throw std::runtime_error("Cooling requires a positive completed SPH interval");
        }
        double const seconds =
            units::toPhysicalInterval(units::ConformalTime(interval * scales.timeScale()), context)
                .quantity()
                .value();
        SphChemistryBatchInput input;
        input.electronFraction = sphParticlesElectronFractions[index];
        input.storedEnergy = sphParticlesInternalEnergy(index);
        input.storedDensity = sphParticlesDensity(index);
        input.physicalInterval = seconds;
        input.particleIndex = index;
        inputs.push_back(input);
    }
    auto const results = sphChemistryAdvanceBatch(inputs, scaleFactor);
    std::vector<detail::PendingCoolingUpdate> prepared;
    prepared.reserve(results.size());
    double heating = 0.0;
    for (size_t selectedIndex = 0; selectedIndex < inputs.size(); ++selectedIndex) {
        auto const thermal = sph_entropy::prepareStoredEnergy(results[selectedIndex].storedEnergy,
            inputs[selectedIndex].storedDensity, options.hydro.heatCapacityRatio, scaleFactor);
        prepared.push_back(
            {indices[selectedIndex], thermal.entropy, results[selectedIndex].electronFraction});
        heating -= options.initial.gasMass *
            (double(thermal.storedEnergy) - inputs[selectedIndex].storedEnergy) / scaleFactor;
    }
    detail::pendingCooling = std::move(prepared);
    return heating;
}
void sphCoolingCommitLocal() {
    for (auto const& update : detail::pendingCooling) {
        sphParticlesEntropies[update.gas] = update.entropy;
        sphParticlesElectronFractions[update.gas] = update.electronFraction;
    }
    detail::pendingCooling.clear();
}
double sphChemistryClose(
    int minimumRung, double scaleFactor, double, double baseTimestep, double explicitTimestep) {
    double const heating =
        sphCoolingPrepare(minimumRung, scaleFactor, baseTimestep, explicitTimestep);
    sphCoolingCommit();
    return heating;
}
std::array<double, 6> sphChemistryCheckpointParameters() {
    auto const& options = getOptions();
    return {6.0, gadget4::kHydrogenMassFraction, gadget4::kGamma,
        options.chemistry.initializationTemperature, options.chemistry.minimumSpecificEnergy,
        options.run.scenario == "cosmology" ? 1.0 : 0.0};
}
std::array<float, 9> sphChemistryOutput(
    float electronFraction, float storedEnergy, float storedDensity, double scaleFactor) {
    auto const parameters = sphCoolingParameters(scaleFactor);
    double ne = electronFraction, temperature;
    double const density = detail::coolingPhysicalDensity(storedDensity, scaleFactor);
    auto status =
        gadget4::temperatureFromEnergy(detail::coolingPhysicalEnergy(storedEnergy, scaleFactor),
            density, parameters, ne, temperature);
    gadget4::GasState gas;
    if (status == gadget4::Status::Success) {
        status = gadget4::equilibriumAtTemperature(temperature, density, parameters, ne, gas);
    }
    if (status != gadget4::Status::Success) {
        throw std::runtime_error(
            std::string("Equilibrium cooling diagnostic failed: ") + gadget4::statusName(status));
    }
    return {float(gas.hydrogenNeutral), float(gas.hydrogenIonized), float(gas.heliumNeutral),
        float(gas.heliumIonized), float(gas.heliumDoubleIonized), float(gas.electronFraction),
        float(temperature), float(gas.meanMolecularWeight), float(gadget4::kGamma)};
}
HPX_PLAIN_ACTION(sphChemistryPressureIntegral, SphChemistryPressureIntegralAction);
double sphChemistryPressureIntegral(double scaleFactor) {
    std::vector<hpx::future<double>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<SphChemistryPressureIntegralAction>(child, scaleFactor));
    }
    double sum = 0.0;
    for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
        sum += getOptions().initial.gasMass * gadget4::kGammaMinusOne *
            sphParticlesInternalEnergy(index) / (scaleFactor * scaleFactor);
    }
    for (auto& future : remote) {
        sum += future.get();
    }
    return sum;
}
