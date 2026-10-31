/* CosmicTiger - GADGET-4 equilibrium atomic cooling; BSL-1.0. */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <cosmictiger/cuda/cuda_export.hpp>

namespace gadget4 {

// Physical constants and numerical choices of official GADGET-4, commit
// 6fb393b5e25907f2ff06211f67d341c5e40e90d1, src/data/constants.h and cooling_sfr/.
inline constexpr double kHydrogenMassFraction = 0.76;
inline constexpr double kHeliumNumberRatio =
    (1.0 - kHydrogenMassFraction) / (4.0 * kHydrogenMassFraction);
inline constexpr double kBoltzmann = 1.38065e-16;        // erg K^-1.
inline constexpr double kProtonMass = 1.67262178e-24;    // g.
inline constexpr double kGamma = 5.0 / 3.0;
inline constexpr double kGammaMinusOne = kGamma - 1.0;
inline constexpr double kMinimumLogTemperature = 1.0;
inline constexpr double kMaximumLogTemperature = 9.0;
inline constexpr size_t kRateTableIntervals = 2000;
inline constexpr size_t kRateTableSize = kRateTableIntervals + 1;
inline constexpr double kLogTemperatureSpacing =
    (kMaximumLogTemperature - kMinimumLogTemperature) / double(kRateTableIntervals);
inline constexpr int kMaximumIterations = 10000;

enum class Status { Success, InvalidInput, NonConvergence, BackendUnavailable, BackendFailure };
enum class Backend { Cpu, Cuda };

// GADGET-4 stores only the equilibrium electron-fraction guess per gas particle.
struct State {
    float electronFraction{1.0f};

    bool operator==(State const&) const = default;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & electronFraction;
    }
};

struct RateRow {
    double hydrogenExcitation{}, heliumIonExcitation{}, freeFree{};
    double hydrogenRecombination{}, heliumRecombination{}, heliumDoubleRecombination{};
    double heliumDielectronicRecombination{};
    double hydrogenIonization{}, heliumIonization{}, heliumIonIonization{};
};
using RateTable = std::array<RateRow, kRateTableSize>;

struct Parameters {
    RateRow const* rates{};    // hostRateTable().data(), or deviceRateTable() on CUDA.
    bool comoving{};
    double redshift{};
    double minimumSpecificEnergy{};    // erg/g; application floor, not a rate-table bound.
};

// Transient equilibrium diagnostics, never a separately evolved phase inventory.
struct GasState {
    double hydrogenNeutral{1.0}, hydrogenIonized{};
    double heliumNeutral{kHeliumNumberRatio}, heliumIonized{}, heliumDoubleIonized{};
    double electronFraction{}, temperature{}, meanMolecularWeight{};
    RateRow coefficients{};
};

struct Input {
    State state{};
    double specificEnergy{};    // erg/g.
    double density{};           // proper g/cm^3.
    double interval{};          // physical seconds.
};
struct Result {
    State state{};
    double specificEnergy{};
    Status status{Status::InvalidInput};
};

RateTable buildRateTable();
RateTable const& hostRateTable();
Parameters makeParameters(double scaleFactor, bool comoving, double minimumSpecificEnergy);
char const* statusName(Status status);

// All scalar physical arguments use CGS. Failed calls preserve output arguments.
CUDA_EXPORT inline Status equilibriumAtTemperature(double temperature, double density,
    Parameters const& parameters, double& electronFraction, GasState& gas);
CUDA_EXPORT inline Status temperatureFromEnergy(double specificEnergy, double density,
    Parameters const& parameters, double& electronFraction, double& temperature);
CUDA_EXPORT inline Status coolingRate(double specificEnergy, double density,
    Parameters const& parameters, double& electronFraction, double& netRatePerHydrogenSquared);
CUDA_EXPORT inline Status coolingTime(double specificEnergy, double density,
    Parameters const& parameters, double& electronFraction, double& seconds);
// Raw GADGET-4 DoCooling: the caller, rather than this function, applies MinEgySpec.
CUDA_EXPORT inline Status doCooling(double specificEnergy, double density, double interval,
    Parameters const& parameters, double& electronFraction, double& newSpecificEnergy);
// Ordinary gas application, including GADGET-4's MinEgySpec floor.
CUDA_EXPORT inline Result advance(Input const& input, Parameters const& parameters);

bool integrateBatchCpu(Input const* inputs, Result* results, size_t count,
    Parameters const& parameters, unsigned workers = 1);
bool integrateBatchCuda(Input const* inputs, Result* results, size_t count,
    Parameters const& parameters, std::string* error = nullptr);
bool integrateBatch(Input const* inputs, Result* results, size_t count,
    Parameters const& parameters, Backend backend, unsigned workers = 1,
    std::string* error = nullptr);
// Immutable official table cached for the current CUDA device. Shared by SF and cooling.
// A null return reports unavailable/failed CUDA setup; it never selects a CPU fallback.
RateRow const* deviceRateTable(std::string* error = nullptr, Status* status = nullptr);

}    // namespace gadget4

#include <cosmictiger/chemistry/gadget4_cooling_impl.hpp>
