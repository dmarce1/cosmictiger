/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Licensed under the GNU General Public License, version 2 or later.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cosmictiger/hpx.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/tree.hpp>

namespace detail {

using LinearVector = std::array<double, NDIM>;
using LinearWave = std::array<int, NDIM>;
using LinearField = std::vector<LinearVector>;
constexpr double kLinearFourPi = 4.0 * std::numbers::pi;

struct LinearSample {
    LinearField acceleration;
    LinearField displacement;
    std::array<double, 2> meanSupport{};
    double minimumSupport = std::numeric_limits<double>::max();
    double maximumSupport = 0.0;
};

struct LinearProjection {
    std::array<LinearVector, 2> mean{};
    std::array<LinearVector, 2> sine{};
    std::array<LinearVector, 2> cosine{};
    std::array<LinearVector, 2> residual{};
};

std::vector<std::string> linearSplit(std::string const& text, char delimiter) {
    std::vector<std::string> result;
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, delimiter)) {
        result.push_back(item);
    }
    return result;
}

std::string linearEnvironment(char const* name, char const* fallback) {
    auto const* value = std::getenv(name);
    return value == nullptr ? fallback : value;
}

struct LinearExperiment {
    int side;
    int species;
    size_t perSpecies;
    LinearField reference;
    std::string loadFile;

    explicit LinearExperiment(Options const& options)
      : side(options.initial.particlesPerDimension)
      , species(options.hydro.enabled ? 2 : 1)
      , perSpecies(size_t(side) * side * side)
      , reference(perSpecies * species)
      , loadFile(linearEnvironment("COSMICTIGER_LINEAR_GROWTH_LOAD_FILE", "")) {
        if (!loadFile.empty()) {
            std::ifstream input(loadFile);
            if (!input) {
                throw std::runtime_error("Cannot read linear-growth particle load: " + loadFile);
            }
            std::string line;
            size_t count = 0;
            while (std::getline(input, line)) {
                auto const first = line.find_first_not_of(" \t\r");
                if (first == std::string::npos || line[first] == '#') {
                    continue;
                }
                if (count == reference.size()) {
                    throw std::runtime_error("Linear-growth particle load has too many rows");
                }
                std::istringstream row(line);
                for (int dim = 0; dim < NDIM; ++dim) {
                    double& coordinate = reference[count][dim];
                    if (!(row >> coordinate) || !std::isfinite(coordinate) ||
                        !(coordinate >= 0.0 && coordinate < 1.0)) {
                        throw std::runtime_error(
                            "Linear-growth particle load requires finite xyz coordinates in [0,1)");
                    }
                }
                std::string extra;
                if (row >> extra) {
                    throw std::runtime_error("Linear-growth particle load requires exactly xyz");
                }
                ++count;
            }
            if (count != reference.size()) {
                throw std::runtime_error("Linear-growth particle load has the wrong row count");
            }
            return;
        }
        for (int kind = 0; kind < species; ++kind) {
            double const offset = 0.5 * kind;
            for (int ix = 0; ix < side; ++ix) {
                for (int iy = 0; iy < side; ++iy) {
                    for (int iz = 0; iz < side; ++iz) {
                        size_t const index = kind * perSpecies + (ix * side + iy) * side + iz;
                        reference[index] = {
                            (ix + offset) / side, (iy + offset) / side, (iz + offset) / side};
                    }
                }
            }
        }
    }

    double phase(LinearWave const& wave, size_t index) const {
        double result = 0.0;
        for (int dim = 0; dim < NDIM; ++dim) {
            result += wave[dim] * reference[index][dim];
        }
        return 2.0 * std::numbers::pi * result;
    }

    LinearField mode(LinearWave const& wave, std::array<LinearVector, 2> const& amplitudes) const {
        LinearField result(reference.size());
        for (size_t index = 0; index < result.size(); ++index) {
            int const kind = index / perSpecies;
            double const sine = std::sin(phase(wave, index));
            for (int dim = 0; dim < NDIM; ++dim) {
                result[index][dim] = amplitudes[kind][dim] * sine / side;
            }
        }
        return result;
    }

    LinearProjection project(LinearField const& field, LinearWave const& wave) const {
        LinearProjection result;
        using ProjectionMatrix = std::array<std::array<double, 3 + NDIM>, 3>;
        std::array<ProjectionMatrix, 2> equations{};
        for (size_t index = 0; index < reference.size(); ++index) {
            int const kind = index / perSpecies;
            std::array<double, 3> const basis{
                1.0, std::sin(phase(wave, index)), std::cos(phase(wave, index))};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    equations[kind][row][column] += basis[row] * basis[column] / perSpecies;
                }
                for (int dim = 0; dim < NDIM; ++dim) {
                    equations[kind][row][3 + dim] += basis[row] * field[index][dim] / perSpecies;
                }
            }
        }
        for (int kind = 0; kind < species; ++kind) {
            auto& matrix = equations[kind];
            for (int column = 0; column < 3; ++column) {
                int pivot = column;
                for (int row = column + 1; row < 3; ++row) {
                    if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                        pivot = row;
                    }
                }
                if (std::abs(matrix[pivot][column]) < 1.0e-10) {
                    throw std::runtime_error("Degenerate sinusoidal mode on particle load");
                }
                std::swap(matrix[pivot], matrix[column]);
                double const diagonal = matrix[column][column];
                for (int entry = column; entry < 3 + NDIM; ++entry) {
                    matrix[column][entry] /= diagonal;
                }
                for (int row = 0; row < 3; ++row) {
                    if (row != column) {
                        double const factor = matrix[row][column];
                        for (int entry = column; entry < 3 + NDIM; ++entry) {
                            matrix[row][entry] -= factor * matrix[column][entry];
                        }
                    }
                }
            }
            for (int dim = 0; dim < NDIM; ++dim) {
                result.mean[kind][dim] = matrix[0][3 + dim];
                result.sine[kind][dim] = matrix[1][3 + dim];
                result.cosine[kind][dim] = matrix[2][3 + dim];
            }
        }
        for (size_t index = 0; index < reference.size(); ++index) {
            int const kind = index / perSpecies;
            double const sine = std::sin(phase(wave, index));
            double const cosine = std::cos(phase(wave, index));
            for (int dim = 0; dim < NDIM; ++dim) {
                double const residual = field[index][dim] - result.mean[kind][dim] -
                    result.sine[kind][dim] * sine - result.cosine[kind][dim] * cosine;
                result.residual[kind][dim] += residual * residual / perSpecies;
            }
        }
        for (int kind = 0; kind < species; ++kind) {
            for (int dim = 0; dim < NDIM; ++dim) {
                result.residual[kind][dim] = std::sqrt(result.residual[kind][dim]);
            }
        }
        return result;
    }

    LinearSample force(LinearField const& displacement) const {
        auto const& options = getOptions();
        particlesResize(reference.size());
        if (species == 2) {
            sphParticlesResize(perSpecies, false);
        }
        using PositionKey = std::array<uint32_t, NDIM + 1>;
        std::map<PositionKey, size_t> labels;
        for (size_t index = 0; index < reference.size(); ++index) {
            int const kind = index / perSpecies;
            particlesRung(index) = 1;
            particlesPot(index) = 0.0f;
            if (species == 2) {
                particlesType(index) = kind == 0 ? DARK_MATTER_TYPE : SPH_TYPE;
                particlesCatIndex(index) = kind == 0 ? NO_INDEX : index - perSpecies;
                if (kind == 1) {
                    sphParticlesParticleIndex(index - perSpecies) = index;
                    sphParticlesSupportRadius(index - perSpecies) = 1.0 / side;
                }
            }
            PositionKey key{};
            key[NDIM] = kind;
            for (int dim = 0; dim < NDIM; ++dim) {
                if (!std::isfinite(displacement[index][dim])) {
                    throw std::runtime_error("Nonfinite linear-growth displacement");
                }
                if (loadFile.empty() && std::abs(displacement[index][dim]) * side >= 0.25) {
                    throw std::runtime_error("Linear-growth displacement exceeds label-safe range");
                }
                double position = reference[index][dim] + displacement[index][dim];
                position -= std::floor(position);
                particlesPos(dim, index) = position;
                key[dim] = particlesPos(dim, index).raw();
                particlesVel(dim, index) = 0.0f;
                particlesGravityAcceleration(dim, index) = 0.0f;
            }
            if (!labels.emplace(key, index).second) {
                throw std::runtime_error(
                    "Linear-growth fixed32 positions are not unique per species");
            }
        }
        particlesSortByRung(0);
        auto const built = treeCreate(
            TreeCreateParameters(0, options.gravity.openingAngle, options.gravity.supportRadius));
        if (built.maxGravitySupportRadius != float(options.gravity.supportRadius)) {
            throw std::runtime_error(
                "Linear-growth tree changed the configured fixed gravity support");
        }
        KickParameters parameters;
        parameters.htime = false;
        parameters.sign = 1.0;
        parameters.minLevel = 0;
        parameters.minRung = 0;
        parameters.a = 1.0;
        parameters.t0 = 0.001;
        parameters.theta = options.gravity.openingAngle;
        parameters.gravitySupportRadius = options.gravity.supportRadius;
        parameters.eta = options.gravity.timestepEta;
        parameters.gm = options.gravity.gm;
        parameters.velInit = false;
        parameters.saveForce = true;
        parameters.firstCall = true;
        parameters.gpu = options.compute.cuda;
        parameters.nodeLoad = 1.0;
        parameters.ascending = true;
        parameters.descending = true;
        parameters.top = true;
        parameters.doPhi = true;
        ExpansionBuffer local;
        local.zero();
        Array<Fixed32, NDIM> origin{};
        kick(parameters, local, origin, built.id, Vector<TreeId>{built.id},
            Vector<TreeId>{built.id}, nullptr);
        LinearSample result;
        result.acceleration.resize(reference.size());
        result.displacement.resize(reference.size());
        std::vector<bool> seen(reference.size(), false);
        for (ParticleInteger particle = 0; particle < particlesSize(); ++particle) {
            int const kind = species == 2 && particlesIsSph(particle) ? 1 : 0;
            PositionKey key{};
            key[NDIM] = kind;
            for (int dim = 0; dim < NDIM; ++dim) {
                key[dim] = particlesPos(dim, particle).raw();
            }
            auto const found = labels.find(key);
            if (found == labels.end()) {
                throw std::runtime_error(
                    "Linear-growth force changed a particle position or species");
            }
            size_t const index = found->second;
            if (seen[index]) {
                throw std::runtime_error("Linear-growth particle labels are not unique");
            }
            seen[index] = true;
            double const support = built.maxGravitySupportRadius;
            result.meanSupport[kind] += support / perSpecies;
            result.minimumSupport = std::min(result.minimumSupport, support);
            result.maximumSupport = std::max(result.maximumSupport, support);
            for (int dim = 0; dim < NDIM; ++dim) {
                result.acceleration[index][dim] = particlesGravityAcceleration(dim, particle);
                double difference = particlesPos(dim, particle).toDouble() - reference[index][dim];
                difference -= std::round(difference);
                result.displacement[index][dim] = difference;
                if (!std::isfinite(result.acceleration[index][dim])) {
                    throw std::runtime_error("Nonfinite linear-growth acceleration");
                }
            }
        }
        treeDestroy(true);
        return result;
    }
};

void validateLinearWave(LinearWave const& wave, int side) {
    int norm = 0;
    for (int dim = 0; dim < NDIM; ++dim) {
        norm += wave[dim] * wave[dim];
        if (std::abs(wave[dim]) >= side / 2) {
            throw std::runtime_error("Linear-growth mode must be below component Nyquist");
        }
    }
    if (norm == 0) {
        throw std::runtime_error("Linear-growth mode must be nonzero");
    }
}

std::vector<LinearWave> linearWaves(int side) {
    std::vector<LinearWave> result;
    auto const text =
        linearEnvironment("COSMICTIGER_LINEAR_GROWTH_MODES", "1,0,0;2,0,0;3,0,0;1,1,0;1,1,1");
    for (auto const& item : linearSplit(text, ';')) {
        auto const components = linearSplit(item, ',');
        if (components.size() != NDIM) {
            throw std::runtime_error("Linear-growth modes require three comma-separated integers");
        }
        LinearWave wave;
        for (int dim = 0; dim < NDIM; ++dim) {
            wave[dim] = std::stoi(components[dim]);
        }
        validateLinearWave(wave, side);
        result.push_back(wave);
    }
    return result;
}

void writeLinearMetadata(LinearExperiment const& experiment, LinearSample const& baseline) {
    auto const& options = getOptions();
    double baselineSquared = 0.0;
    double baselineMaximum = 0.0;
    std::array<double, 2> baselineSpeciesSquared{};
    for (size_t index = 0; index < baseline.acceleration.size(); ++index) {
        auto const& acceleration = baseline.acceleration[index];
        double squared = 0.0;
        for (double component : acceleration) {
            squared += component * component;
        }
        baselineSquared += squared / baseline.acceleration.size();
        baselineSpeciesSquared[index / experiment.perSpecies] += squared / experiment.perSpecies;
        baselineMaximum = std::max(baselineMaximum, std::sqrt(squared));
    }
    std::ofstream output("linear_growth_metadata.json");
    output << std::setprecision(17) << "{\n"
           << "  \"side\": " << experiment.side << ",\n"
           << "  \"species\": " << experiment.species << ",\n"
           << "  \"load_kind\": \"" << (experiment.loadFile.empty() ? "lattice" : "file") << "\",\n"
           << "  \"load_file\": " << std::quoted(experiment.loadFile) << ",\n"
           << "  \"projection\": \"least squares mean + sine + cosine\",\n"
           << "  \"paired_evolution\": "
           << (linearEnvironment("COSMICTIGER_LINEAR_GROWTH_PAIRED", "0") == "1" ? "true" : "false")
           << ",\n"
           << "  \"mass_dm\": " << (options.hydro.enabled ? options.initial.darkMatterMass : 1.0)
           << ",\n"
           << "  \"mass_gas\": " << (options.hydro.enabled ? options.initial.gasMass : 0.0) << ",\n"
           << "  \"gm\": " << options.gravity.gm << ",\n"
           << "  \"kernel_family\": \"" << options.gravity.kernelFamily << "\",\n"
           << "  \"kernel_degree\": " << options.gravity.kernelDegree << ",\n"
           << "  \"softening_mode\": \"fixed\",\n"
           << "  \"configured_support_radius_box\": " << options.gravity.supportRadius
           << ",\n"
           // Retain the historical metadata key; fixed gravity has no neighbor target.
           << "  \"neighbor_number\": 0,\n"
           << "  \"theta\": " << options.gravity.openingAngle << ",\n"
           << "  \"backend\": \"" << (options.compute.cuda ? "cuda" : "cpu") << "\",\n"
           << "  \"omega_m\": " << options.cosmology.omegaM << ",\n"
           << "  \"omega_r\": " << options.cosmology.omegaR << ",\n"
           << "  \"omega_k\": " << options.cosmology.omegaK << ",\n"
           << "  \"omega_lambda\": " << options.cosmology.omegaLambda << ",\n"
           << "  \"baseline_force_rms\": " << std::sqrt(baselineSquared) << ",\n"
           << "  \"baseline_force_max\": " << baselineMaximum << ",\n"
           << "  \"baseline_force_rms_dm\": " << std::sqrt(baselineSpeciesSquared[0]) << ",\n"
           << "  \"baseline_force_rms_gas\": " << std::sqrt(baselineSpeciesSquared[1]) << ",\n"
           << "  \"baseline_support_dm\": " << baseline.meanSupport[0] << ",\n"
           << "  \"baseline_support_gas\": " << baseline.meanSupport[1] << ",\n"
           << "  \"baseline_min_support\": " << baseline.minimumSupport << ",\n"
           << "  \"baseline_max_support\": " << baseline.maximumSupport << ",\n"
           << "  \"pressure_enabled\": false,\n"
           << "  \"response_normalization\": \"acceleration / (4*pi*G*rho_comoving)\"\n}\n";
    if (!output) {
        throw std::runtime_error("Cannot write linear-growth metadata");
    }
}

void writeLinearInitialParticles(LinearExperiment const& experiment, LinearSample const& sample) {
    std::ofstream output("linear_growth_particles_initial.csv");
    output
        << std::setprecision(17)
        << "particle,species,reference_x,reference_y,reference_z,x,y,z,"
           "realized_u_x,realized_u_y,realized_u_z,acceleration_x,acceleration_y,acceleration_z\n";
    for (size_t index = 0; index < experiment.reference.size(); ++index) {
        output << index << ',' << index / experiment.perSpecies;
        for (double coordinate : experiment.reference[index]) {
            output << ',' << coordinate;
        }
        for (int dim = 0; dim < NDIM; ++dim) {
            double position = experiment.reference[index][dim] + sample.displacement[index][dim];
            position -= std::floor(position);
            output << ',' << position;
        }
        for (double displacement : sample.displacement[index]) {
            output << ',' << displacement;
        }
        for (double acceleration : sample.acceleration[index]) {
            output << ',' << acceleration;
        }
        output << '\n';
    }
    if (!output) {
        throw std::runtime_error("Cannot write initial linear-growth particle sample");
    }
}

void measureLinearResponse(LinearExperiment const& experiment, LinearSample const& baseline) {
    auto const waves = linearWaves(experiment.side);
    std::vector<double> amplitudes;
    for (auto const& item :
        linearSplit(linearEnvironment("COSMICTIGER_LINEAR_GROWTH_AMPLITUDES", "0.01,0.005"), ',')) {
        double const amplitude = std::stod(item);
        if (!(amplitude > 0.0 && amplitude < 0.1)) {
            throw std::runtime_error("Linear-growth amplitudes must be between zero and 0.1 cells");
        }
        amplitudes.push_back(amplitude);
    }
    std::ofstream output("linear_growth_response.csv");
    output << std::setprecision(17)
           << "mode_x,mode_y,mode_z,amplitude_spacing,input_species,input_axis,output_species,"
              "output_axis,response_sin,response_cos,baseline_sin,baseline_cos,mean_support_dm,"
              "mean_support_gas,min_support,max_support,realized_amplitude,response_residual_rms\n";
    bool const particleOutputEnabled =
        linearEnvironment("COSMICTIGER_LINEAR_GROWTH_PARTICLE_OUTPUT", "0") == "1";
    std::ofstream particleOutput;
    if (particleOutputEnabled) {
        particleOutput.open("linear_growth_response_particles.csv");
        particleOutput
            << std::setprecision(17)
            << "mode_x,mode_y,mode_z,amplitude_spacing,input_species,input_axis,particle,"
               "output_species,output_axis,acceleration_positive,acceleration_negative,"
               "displacement_positive,displacement_negative\n";
    }
    for (auto const& wave : waves) {
        auto const background = experiment.project(baseline.acceleration, wave);
        for (double amplitude : amplitudes) {
            PRINT("Linear response mode=(%i,%i,%i) amplitude=%g cells\n", wave[0], wave[1], wave[2],
                amplitude);
            for (int inputKind = 0; inputKind < experiment.species; ++inputKind) {
                for (int inputAxis = 0; inputAxis < NDIM; ++inputAxis) {
                    std::array<LinearVector, 2> coefficients{};
                    coefficients[inputKind][inputAxis] = amplitude;
                    auto const positive = experiment.force(experiment.mode(wave, coefficients));
                    coefficients[inputKind][inputAxis] = -amplitude;
                    auto const negative = experiment.force(experiment.mode(wave, coefficients));
                    if (particleOutputEnabled) {
                        for (size_t index = 0; index < experiment.reference.size(); ++index) {
                            for (int dim = 0; dim < NDIM; ++dim) {
                                particleOutput << wave[0] << ',' << wave[1] << ',' << wave[2] << ','
                                               << amplitude << ',' << inputKind << ',' << inputAxis
                                               << ',' << index << ','
                                               << index / experiment.perSpecies << ',' << dim << ','
                                               << positive.acceleration[index][dim] << ','
                                               << negative.acceleration[index][dim] << ','
                                               << positive.displacement[index][dim] << ','
                                               << negative.displacement[index][dim] << '\n';
                            }
                        }
                    }
                    auto accelerationDifference = positive.acceleration;
                    for (size_t index = 0; index < accelerationDifference.size(); ++index) {
                        for (int dim = 0; dim < NDIM; ++dim) {
                            accelerationDifference[index][dim] -= negative.acceleration[index][dim];
                        }
                    }
                    auto const response = experiment.project(accelerationDifference, wave);
                    auto const positiveDisplacement =
                        experiment.project(positive.displacement, wave);
                    auto const negativeDisplacement =
                        experiment.project(negative.displacement, wave);
                    double const realized = positiveDisplacement.sine[inputKind][inputAxis] -
                        negativeDisplacement.sine[inputKind][inputAxis];
                    for (int outputKind = 0; outputKind < experiment.species; ++outputKind) {
                        for (int outputAxis = 0; outputAxis < NDIM; ++outputAxis) {
                            double const sine =
                                response.sine[outputKind][outputAxis] / (kLinearFourPi * realized);
                            double const cosine = response.cosine[outputKind][outputAxis] /
                                (kLinearFourPi * realized);
                            output
                                << wave[0] << ',' << wave[1] << ',' << wave[2] << ',' << amplitude
                                << ',' << inputKind << ',' << inputAxis << ',' << outputKind << ','
                                << outputAxis << ',' << sine << ',' << cosine << ','
                                << background.sine[outputKind][outputAxis] << ','
                                << background.cosine[outputKind][outputAxis] << ','
                                << 0.5 * (positive.meanSupport[0] + negative.meanSupport[0]) << ','
                                << 0.5 * (positive.meanSupport[1] + negative.meanSupport[1]) << ','
                                << std::min(positive.minimumSupport, negative.minimumSupport) << ','
                                << std::max(positive.maximumSupport, negative.maximumSupport) << ','
                                << 0.5 * realized << ','
                                << response.residual[outputKind][outputAxis] /
                                    (kLinearFourPi * std::abs(realized))
                                << '\n';
                        }
                    }
                }
            }
            output.flush();
            if (particleOutputEnabled) {
                particleOutput.flush();
            }
        }
    }
    if (!output || (particleOutputEnabled && !particleOutput)) {
        throw std::runtime_error("Cannot write linear-growth response");
    }
}

struct LinearState {
    LinearField displacement;
    LinearField velocity;
};

void writeLinearParticleStates(std::ostream& output, LinearExperiment const& experiment,
    std::vector<LinearState> const& states, bool paired, std::string const& label, int run,
    int step, double scale) {
    // Positions and displacements are in box units. Velocity is d(displacement)/dln(a),
    // not the stored particle momentum. The state displacement is unwrapped; the
    // realized displacement uses the same fixed32 periodic representation as force().
    std::array<char const*, 3> const trajectoryNames{"positive", "negative", "zero"};
    for (size_t trajectory = 0; trajectory < states.size(); ++trajectory) {
        auto const& state = states[trajectory];
        for (size_t index = 0; index < experiment.reference.size(); ++index) {
            output << label << ',' << run << ',' << step << ',' << scale << ','
                   << (paired ? trajectoryNames[trajectory] : "single") << ',' << index << ','
                   << index / experiment.perSpecies;
            LinearVector realizedDisplacement{};
            for (int dim = 0; dim < NDIM; ++dim) {
                double position = experiment.reference[index][dim] + state.displacement[index][dim];
                position -= std::floor(position);
                Fixed32 const realizedPosition(position);
                output << ',' << realizedPosition.toDouble();
                double difference = realizedPosition.toDouble() - experiment.reference[index][dim];
                realizedDisplacement[dim] = difference - std::round(difference);
            }
            for (double displacement : state.displacement[index]) {
                output << ',' << displacement;
            }
            for (double displacement : realizedDisplacement) {
                output << ',' << displacement;
            }
            for (double velocity : state.velocity[index]) {
                output << ',' << velocity;
            }
            output << '\n';
        }
    }
    output.flush();
}

LinearState linearAdd(LinearState const& state, LinearState const& derivative, double step) {
    auto result = state;
    for (size_t index = 0; index < state.displacement.size(); ++index) {
        for (int dim = 0; dim < NDIM; ++dim) {
            result.displacement[index][dim] += step * derivative.displacement[index][dim];
            result.velocity[index][dim] += step * derivative.velocity[index][dim];
        }
    }
    return result;
}

LinearState linearDerivative(
    LinearExperiment const& experiment, LinearState const& state, double logScale) {
    auto const& options = getOptions();
    double const scale = std::exp(logScale);
    double const matter = options.cosmology.omegaM / std::pow(scale, 3);
    double const radiation = options.cosmology.omegaR / std::pow(scale, 4);
    double const curvature = options.cosmology.omegaK / (scale * scale);
    double const hubbleSquared = matter + radiation + curvature + options.cosmology.omegaLambda;
    double const drag =
        2.0 - (3.0 * matter + 4.0 * radiation + 2.0 * curvature) / (2.0 * hubbleSquared);
    double const gravity = 1.5 * matter / hubbleSquared / kLinearFourPi;
    auto const sample = experiment.force(state.displacement);
    LinearState result{state.velocity, LinearField(state.velocity.size())};
    for (size_t index = 0; index < state.displacement.size(); ++index) {
        for (int dim = 0; dim < NDIM; ++dim) {
            result.velocity[index][dim] =
                gravity * sample.acceleration[index][dim] - drag * state.velocity[index][dim];
        }
    }
    return result;
}

void advanceLinearState(
    LinearExperiment const& experiment, LinearState& state, double logScale, double stepSize) {
    auto const first = linearDerivative(experiment, state, logScale);
    auto const second = linearDerivative(
        experiment, linearAdd(state, first, 0.5 * stepSize), logScale + 0.5 * stepSize);
    auto const third = linearDerivative(
        experiment, linearAdd(state, second, 0.5 * stepSize), logScale + 0.5 * stepSize);
    auto const fourth =
        linearDerivative(experiment, linearAdd(state, third, stepSize), logScale + stepSize);
    for (size_t index = 0; index < state.displacement.size(); ++index) {
        for (int dim = 0; dim < NDIM; ++dim) {
            state.displacement[index][dim] += stepSize / 6.0 *
                (first.displacement[index][dim] + 2.0 * second.displacement[index][dim] +
                    2.0 * third.displacement[index][dim] + fourth.displacement[index][dim]);
            state.velocity[index][dim] += stepSize / 6.0 *
                (first.velocity[index][dim] + 2.0 * second.velocity[index][dim] +
                    2.0 * third.velocity[index][dim] + fourth.velocity[index][dim]);
        }
    }
}

std::array<LinearVector, 2> linearFieldRms(
    LinearExperiment const& experiment, LinearField const& field) {
    std::array<LinearVector, 2> result{};
    for (size_t index = 0; index < field.size(); ++index) {
        int const kind = index / experiment.perSpecies;
        for (int dim = 0; dim < NDIM; ++dim) {
            result[kind][dim] += field[index][dim] * field[index][dim] / experiment.perSpecies;
        }
    }
    for (int kind = 0; kind < experiment.species; ++kind) {
        for (int dim = 0; dim < NDIM; ++dim) {
            result[kind][dim] = std::sqrt(result[kind][dim]);
        }
    }
    return result;
}

std::array<std::complex<double>, 2> linearDensity(
    LinearExperiment const& experiment, LinearState const& state, LinearWave const& wave) {
    std::array<std::complex<double>, 2> result{};
    for (size_t index = 0; index < experiment.reference.size(); ++index) {
        double phase = 0.0;
        for (int dim = 0; dim < NDIM; ++dim) {
            double position = experiment.reference[index][dim] + state.displacement[index][dim];
            position -= std::floor(position);
            Fixed32 const realizedPosition(position);
            phase += wave[dim] * realizedPosition.toDouble();
        }
        result[index / experiment.perSpecies] +=
            std::polar(1.0, -2.0 * std::numbers::pi * phase) / double(experiment.perSpecies);
    }
    return result;
}

void evolveLinearModes(LinearExperiment const& experiment, std::string const& filename) {
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("Cannot read linear-growth evolution requests: " + filename);
    }
    std::ofstream output("linear_growth_evolution.csv");
    output << std::setprecision(17)
           << "label,run_index,step,a,species,axis,u_sin,u_cos,v_sin,v_cos,residual_rms\n";
    bool const paired = linearEnvironment("COSMICTIGER_LINEAR_GROWTH_PAIRED", "0") == "1";
    bool const particleOutputEnabled =
        linearEnvironment("COSMICTIGER_LINEAR_GROWTH_PARTICLE_OUTPUT", "0") == "1";
    std::ofstream particleOutput;
    if (particleOutputEnabled) {
        particleOutput.open("linear_growth_particles.csv");
        particleOutput << std::setprecision(17)
                       << "label,run_index,step,a,trajectory,particle,species,x,y,z,"
                          "u_x,u_y,u_z,realized_u_x,realized_u_y,realized_u_z,v_x,v_y,v_z\n";
    }
    std::ofstream baselineOutput;
    std::ofstream densityOutput;
    if (paired) {
        baselineOutput.open("linear_growth_baseline_evolution.csv");
        baselineOutput
            << std::setprecision(17)
            << "label,run_index,step,a,species,axis,u_sin,u_cos,v_sin,v_cos,residual_rms,"
               "u_rms,v_rms,even_u_rms,even_v_rms\n";
        densityOutput.open("linear_growth_density.csv");
        densityOutput << std::setprecision(17)
                      << "label,run_index,step,a,species,density_diff_real,density_diff_imag,"
                         "baseline_real,baseline_imag,even_real,even_imag\n";
    }
    std::string line;
    int run = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#' || line.starts_with("label,")) {
            continue;
        }
        auto const values = linearSplit(line, ',');
        if (values.size() != 19) {
            throw std::runtime_error(
                "Evolution request needs label, mode, scales, steps and 12 coefficients");
        }
        std::string const label = values[0];
        LinearWave wave;
        for (int dim = 0; dim < NDIM; ++dim) {
            wave[dim] = std::stoi(values[1 + dim]);
        }
        validateLinearWave(wave, experiment.side);
        double const initial = std::stod(values[4]);
        double const final = std::stod(values[5]);
        int const steps = std::stoi(values[6]);
        if (!(initial > 0.0 && final > initial && steps > 0 && steps <= 10000)) {
            throw std::runtime_error("Invalid linear-growth evolution interval");
        }
        std::array<LinearVector, 2> displacement{}, velocity{};
        for (int kind = 0; kind < experiment.species; ++kind) {
            for (int dim = 0; dim < NDIM; ++dim) {
                displacement[kind][dim] = std::stod(values[7 + 6 * kind + dim]);
                velocity[kind][dim] = std::stod(values[10 + 6 * kind + dim]);
            }
        }
        std::vector<LinearState> states;
        states.push_back({experiment.mode(wave, displacement), experiment.mode(wave, velocity)});
        if (paired) {
            auto negative = states.front();
            for (size_t index = 0; index < experiment.reference.size(); ++index) {
                for (int dim = 0; dim < NDIM; ++dim) {
                    negative.displacement[index][dim] *= -1.0;
                    negative.velocity[index][dim] *= -1.0;
                }
            }
            states.push_back(std::move(negative));
            states.push_back({LinearField(experiment.reference.size()),
                LinearField(experiment.reference.size())});
        }
        double const stepSize = std::log(final / initial) / steps;
        PRINT("Linear growth evolution %s mode=(%i,%i,%i), %i steps\n", label.c_str(), wave[0],
            wave[1], wave[2], steps);
        for (int step = 0; step <= steps; ++step) {
            double const logScale = std::log(initial) + step * stepSize;
            if (particleOutputEnabled) {
                writeLinearParticleStates(particleOutput, experiment, states, paired, label, run,
                    step, std::exp(logScale));
            }
            auto observed = states.front();
            if (paired) {
                auto even = states.front();
                for (size_t index = 0; index < experiment.reference.size(); ++index) {
                    for (int dim = 0; dim < NDIM; ++dim) {
                        observed.displacement[index][dim] = 0.5 *
                            (states[0].displacement[index][dim] -
                                states[1].displacement[index][dim]);
                        observed.velocity[index][dim] =
                            0.5 * (states[0].velocity[index][dim] - states[1].velocity[index][dim]);
                        even.displacement[index][dim] = 0.5 *
                                (states[0].displacement[index][dim] +
                                    states[1].displacement[index][dim]) -
                            states[2].displacement[index][dim];
                        even.velocity[index][dim] = 0.5 *
                                (states[0].velocity[index][dim] + states[1].velocity[index][dim]) -
                            states[2].velocity[index][dim];
                    }
                }
                auto const baselineU = experiment.project(states[2].displacement, wave);
                auto const baselineV = experiment.project(states[2].velocity, wave);
                auto const baselineUrms = linearFieldRms(experiment, states[2].displacement);
                auto const baselineVrms = linearFieldRms(experiment, states[2].velocity);
                auto const evenUrms = linearFieldRms(experiment, even.displacement);
                auto const evenVrms = linearFieldRms(experiment, even.velocity);
                auto const positiveDensity = linearDensity(experiment, states[0], wave);
                auto const negativeDensity = linearDensity(experiment, states[1], wave);
                auto const baselineDensity = linearDensity(experiment, states[2], wave);
                for (int kind = 0; kind < experiment.species; ++kind) {
                    auto const differential = 0.5 * (positiveDensity[kind] - negativeDensity[kind]);
                    auto const evenDensity = 0.5 * (positiveDensity[kind] + negativeDensity[kind]) -
                        baselineDensity[kind];
                    densityOutput << label << ',' << run << ',' << step << ',' << std::exp(logScale)
                                  << ',' << kind << ',' << differential.real() << ','
                                  << differential.imag() << ',' << baselineDensity[kind].real()
                                  << ',' << baselineDensity[kind].imag() << ','
                                  << evenDensity.real() << ',' << evenDensity.imag() << '\n';
                    for (int dim = 0; dim < NDIM; ++dim) {
                        baselineOutput << label << ',' << run << ',' << step << ','
                                       << std::exp(logScale) << ',' << kind << ',' << dim << ','
                                       << experiment.side * baselineU.sine[kind][dim] << ','
                                       << experiment.side * baselineU.cosine[kind][dim] << ','
                                       << experiment.side * baselineV.sine[kind][dim] << ','
                                       << experiment.side * baselineV.cosine[kind][dim] << ','
                                       << experiment.side * baselineU.residual[kind][dim] << ','
                                       << experiment.side * baselineUrms[kind][dim] << ','
                                       << experiment.side * baselineVrms[kind][dim] << ','
                                       << experiment.side * evenUrms[kind][dim] << ','
                                       << experiment.side * evenVrms[kind][dim] << '\n';
                    }
                }
                baselineOutput.flush();
                densityOutput.flush();
            }
            auto const u = experiment.project(observed.displacement, wave);
            auto const v = experiment.project(observed.velocity, wave);
            for (int kind = 0; kind < experiment.species; ++kind) {
                for (int dim = 0; dim < NDIM; ++dim) {
                    output << label << ',' << run << ',' << step << ',' << std::exp(logScale) << ','
                           << kind << ',' << dim << ',' << experiment.side * u.sine[kind][dim]
                           << ',' << experiment.side * u.cosine[kind][dim] << ','
                           << experiment.side * v.sine[kind][dim] << ','
                           << experiment.side * v.cosine[kind][dim] << ','
                           << experiment.side * u.residual[kind][dim] << '\n';
                }
            }
            output.flush();
            if (step == steps) {
                break;
            }
            for (auto& state : states) {
                advanceLinearState(experiment, state, logScale, stepSize);
            }
        }
        ++run;
    }
    if (!output || (paired && (!baselineOutput || !densityOutput)) ||
        (particleOutputEnabled && !particleOutput)) {
        throw std::runtime_error("Cannot write linear-growth evolution trace");
    }
}

}    // namespace detail

void linearGrowthTest() {
    auto options = getOptions();
    if (hpxSize() != 1 || options.initial.particlesPerDimension < 4 ||
        options.initial.particlesPerDimension > 32 || options.initial.closePack ||
        options.initial.useGlass) {
        throw std::runtime_error(
            "linearGrowth requires one locality, 4 <= N <= 32 and no standard glass/close-pack "
            "loader; "
            "use COSMICTIGER_LINEAR_GROWTH_LOAD_FILE for a custom particle load");
    }
    options.gravity.enabled = true;
    options.run.hierarchicalGravity = false;

    options.analysis.saveForce = true;

    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.initial.numberOfParticles = size_t(options.initial.particlesPerDimension) *
        options.initial.particlesPerDimension * options.initial.particlesPerDimension;
    // G rho_comoving = 1 makes the measured response independent of code units.
    double const massSum = options.hydro.enabled ?
        double(float(options.initial.darkMatterMass)) + double(float(options.initial.gasMass)) :
        1.0;
    options.gravity.gm = 1.0 / (options.initial.numberOfParticles * massSum);
    setOptions(options);
    detail::LinearExperiment const experiment(options);
    auto const baseline = experiment.force(detail::LinearField(experiment.reference.size()));
    detail::writeLinearMetadata(experiment, baseline);
    if (detail::linearEnvironment("COSMICTIGER_LINEAR_GROWTH_PARTICLE_OUTPUT", "0") == "1") {
        detail::writeLinearInitialParticles(experiment, baseline);
    }
    if (detail::linearEnvironment("COSMICTIGER_LINEAR_GROWTH_SKIP_RESPONSE", "0") != "1") {
        detail::measureLinearResponse(experiment, baseline);
    }
    auto const evolveFile = detail::linearEnvironment("COSMICTIGER_LINEAR_GROWTH_EVOLVE_FILE", "");
    if (!evolveFile.empty()) {
        detail::evolveLinearModes(experiment, evolveFile);
    }
    PRINT("Linear growth diagnostic complete (%s, %i species, N=%i)\n",
        options.compute.cuda ? "CUDA" : "CPU", experiment.species, experiment.side);
}
