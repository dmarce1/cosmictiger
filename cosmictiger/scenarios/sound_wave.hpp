#pragma once

#include <cmath>

#include <cosmictiger/scenarios/scenario.hpp>

namespace scenarios {

constexpr double kSoundWaveAmplitude = 1.0e-6;
constexpr double kSoundWaveGamma = 5.0 / 3.0;
constexpr double kSoundWavePressure = 3.0 / 5.0;

struct SoundWaveState {
    double density;
    double velocity;
    double pressure;
};

// Linear right-going acoustic eigenmode with rho_0=c_s=lambda=1.
// Hopkins (2015), section 4.2.1: https://doi.org/10.1093/mnras/stv195.
inline SoundWaveState soundWaveState(double position, double time = 0.0) {
    double const perturbation =
        kSoundWaveAmplitude * std::sin(2.0 * std::acos(-1.0) * (position - time));
    return {1.0 + perturbation, perturbation, kSoundWavePressure + perturbation};
}

// Invert integral_0^x rho(s) ds to place equal masses in the perturbed density.
inline double soundWavePosition(double massCoordinate) {
    double const waveNumber = 2.0 * std::acos(-1.0);
    double position = massCoordinate;
    for (int iteration = 0; iteration < 4; iteration++) {
        double const enclosedMass =
            position + kSoundWaveAmplitude * (1.0 - std::cos(waveNumber * position)) / waveNumber;
        position -= (enclosedMass - massCoordinate) / soundWaveState(position).density;
    }
    return position;
}

ScenarioPlan initializeSoundWave();

}    // namespace scenarios
