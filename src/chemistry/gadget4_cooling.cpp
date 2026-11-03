/* CosmicTiger - GADGET-4 cooling tables and CPU batching; BSL-1.0. */
#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cosmictiger/chemistry/gadget4_cooling.hpp>

namespace gadget4 {
namespace detail {
    void failedBatch(Input const* inputs, Result* results, size_t count, Status status) {
        if (inputs != nullptr && results != nullptr) {
            for (size_t index = 0; index < count; ++index) {
                results[index].electronFraction = inputs[index].electronFraction;
                results[index].specificEnergy = inputs[index].specificEnergy;
                results[index].status = status;
            }
        }
    }
}    // namespace detail

RateTable buildRateTable() {
    RateTable table{};
    // Cen (1992) fits and guards as selected by GADGET-4 MakeRateTable.
    // The 1.43e-27 tabulated free-free prefactor differs intentionally from
    // the 1.42e-27 fully ionized high-temperature continuation.
    for (size_t index = 0; index < table.size(); ++index) {
        RateRow& row = table[index];
        double const temperature =
            std::pow(10.0, kMinimumLogTemperature + kLogTemperatureSpacing * double(index));
        double const suppression = 1.0 / (1.0 + std::sqrt(temperature / 1.0e5));
        if (118348.0 / temperature < 70.0) {
            row.hydrogenExcitation = 7.5e-19 * std::exp(-118348.0 / temperature) * suppression;
        }
        if (473638.0 / temperature < 70.0) {
            row.heliumIonExcitation = 5.54e-17 * std::pow(temperature, -0.397) *
                std::exp(-473638.0 / temperature) * suppression;
        }
        double const logTemperature = std::log10(temperature);
        row.freeFree = 1.43e-27 * std::sqrt(temperature) *
            (1.1 + 0.34 * std::exp(-(5.5 - logTemperature) * (5.5 - logTemperature) / 3.0));
        row.hydrogenRecombination = 8.4e-11 * std::pow(temperature / 1000.0, -0.2) /
            (1.0 + std::pow(temperature / 1.0e6, 0.7)) / std::sqrt(temperature);
        row.heliumRecombination = 1.5e-10 * std::pow(temperature, -0.6353);
        row.heliumDoubleRecombination = 4.0 * row.hydrogenRecombination;
        if (470000.0 / temperature < 70.0) {
            row.heliumDielectronicRecombination = 1.9e-3 * std::pow(temperature, -1.5) *
                std::exp(-470000.0 / temperature) * (1.0 + 0.3 * std::exp(-94000.0 / temperature));
        }
        if (157809.1 / temperature < 70.0) {
            row.hydrogenIonization =
                5.85e-11 * std::sqrt(temperature) * std::exp(-157809.1 / temperature) * suppression;
        }
        if (285335.4 / temperature < 70.0) {
            row.heliumIonization =
                2.38e-11 * std::sqrt(temperature) * std::exp(-285335.4 / temperature) * suppression;
        }
        if (631515.0 / temperature < 70.0) {
            row.heliumIonIonization =
                5.68e-12 * std::sqrt(temperature) * std::exp(-631515.0 / temperature) * suppression;
        }
    }
    return table;
}

RateTable const& hostRateTable() {
    static RateTable const table = buildRateTable();
    return table;
}

Parameters makeParameters(double scaleFactor, bool comoving, double minimumSpecificEnergy) {
    if (!detail::finite(minimumSpecificEnergy) || minimumSpecificEnergy < 0.0 ||
        !detail::finite(scaleFactor) || scaleFactor <= 0.0) {
        throw std::invalid_argument("Invalid GADGET-4 cooling parameters");
    }
    Parameters parameters;
    parameters.rates = hostRateTable().data();
    parameters.comoving = comoving;
    parameters.redshift = comoving ? 1.0 / scaleFactor - 1.0 : 0.0;
    parameters.minimumSpecificEnergy = minimumSpecificEnergy;
    return parameters;
}

char const* statusName(Status status) {
    switch (status) {
    case Status::Success:
        return "success";
    case Status::InvalidInput:
        return "invalid input";
    case Status::NonConvergence:
        return "nonconvergence";
    case Status::BackendUnavailable:
        return "backend unavailable";
    case Status::BackendFailure:
        return "backend failure";
    }
    return "unknown status";
}

bool integrateBatchCpu(Input const* inputs, Result* results, size_t count,
    Parameters const& parameters, unsigned workers) {
    if (count == 0) {
        return true;
    }
    if (inputs == nullptr || results == nullptr) {
        return false;
    }
    workers = unsigned(std::min<size_t>(std::max(1U, workers), count));
    auto run = [&](unsigned worker) {
        for (size_t index = worker; index < count;) {
            results[index] = advance(inputs[index], parameters);
            if (count - index <= workers) {
                break;
            }
            index += workers;
        }
    };
    std::vector<std::thread> threads;
    try {
        threads.reserve(workers - 1);
        for (unsigned worker = 1; worker < workers; ++worker) {
            threads.emplace_back(run, worker);
        }
        run(0);
    } catch (std::exception const&) {
        for (auto& thread : threads) {
            thread.join();
        }
        detail::failedBatch(inputs, results, count, Status::BackendFailure);
        return false;
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return true;
}

#ifndef USE_CUDA
RateRow const* deviceRateTable(std::string* error, Status* status) {
    if (status != nullptr) {
        *status = Status::BackendUnavailable;
    }
    if (error != nullptr) {
        *error = "GADGET-4 cooling was built without CUDA support";
    }
    return nullptr;
}
bool integrateBatchCuda(
    Input const* inputs, Result* results, size_t count, Parameters const&, std::string* error) {
    if (error != nullptr) {
        error->clear();
    }
    if (count == 0) {
        return true;
    }
    detail::failedBatch(inputs, results, count, Status::BackendUnavailable);
    if (error != nullptr) {
        *error = "GADGET-4 cooling was built without CUDA support";
    }
    return false;
}
#endif

bool integrateBatch(Input const* inputs, Result* results, size_t count,
    Parameters const& parameters, Backend backend, unsigned workers, std::string* error) {
    if (error != nullptr) {
        error->clear();
    }
    if (backend == Backend::Cuda) {
        return integrateBatchCuda(inputs, results, count, parameters, error);
    }
    if (backend != Backend::Cpu) {
        detail::failedBatch(inputs, results, count, Status::BackendUnavailable);
        if (error != nullptr) {
            *error = "Unknown GADGET-4 cooling backend";
        }
        return false;
    }
    bool const success = integrateBatchCpu(inputs, results, count, parameters, workers);
    if (!success && error != nullptr) {
        *error = "GADGET-4 CPU batch dispatch failed";
    }
    return success;
}

}    // namespace gadget4
