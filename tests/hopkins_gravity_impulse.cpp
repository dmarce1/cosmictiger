/* CosmicTiger - conservative HOLD/MFM impulse regression; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <cosmictiger/hydro/hopkins/gravity_impulse.hpp>

namespace detail {

using Vector = std::array<double, 3>;

struct State {
    double mass = 1.0;
    Vector momentum{};
    double energy = 0.0;
};

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("Hopkins gravity impulse: ") + message);
    }
}

bool near(long double first, long double second, long double scale = 1.0L) {
    long double const magnitude = std::max({scale, std::abs(first), std::abs(second)});
    return std::abs(first - second) <= 64.0L * std::numeric_limits<double>::epsilon() * magnitude;
}

long double kinetic(State const& state) {
    long double result = 0.0L;
    for (double component : state.momentum) {
        result += static_cast<long double>(component) * component / (2.0L * state.mass);
    }
    return result;
}

long double thermal(State const& state) {
    return static_cast<long double>(state.energy) - kinetic(state);
}

State makeState(double mass, Vector const& velocity, double internalEnergy) {
    State state;
    state.mass = mass;
    for (int dimension = 0; dimension < 3; ++dimension) {
        state.momentum[dimension] = mass * velocity[dimension];
    }
    state.energy = double(kinetic(state) + mass * internalEnergy);
    return state;
}

void kick(State& state, Vector const& impulse) {
    hopkins::applyGravityImpulse(state.mass, state.momentum, state.energy, impulse);
}

void signedKicksPreserveThermalEnergy() {
    for (double mass : {0.125, 1.0, 17.0}) {
        State const initial = makeState(mass, {1.25, -3.75, 2.5}, 4.0);
        State split = initial;
        Vector net{};
        // A positive coarse opening and signed fine corrections are the HOLD
        // pattern; absent hydro exchanges their result must be one net boost.
        for (Vector const& impulse : {Vector{0.25, -0.5, 0.125}, Vector{-0.0625, 0.125, -0.25},
                 Vector{0.5, -0.25, 0.0625}}) {
            kick(split, impulse);
            require(near(thermal(split), thermal(initial), initial.energy),
                "a signed gravity kick changed internal energy");
            for (int dimension = 0; dimension < 3; ++dimension) {
                net[dimension] += impulse[dimension];
            }
        }
        State combined = initial;
        kick(combined, net);
        for (int dimension = 0; dimension < 3; ++dimension) {
            require(near(split.momentum[dimension], combined.momentum[dimension]),
                "a kick ladder disagrees with its net momentum impulse");
        }
        require(near(split.energy, combined.energy), "a kick ladder double-counted gravity work");
        for (double& component : net) {
            component = -component;
        }
        kick(combined, net);
        require(near(combined.energy, initial.energy), "an inverse kick failed to undo its work");
    }
}

void pairTransferPrecedesGravity() {
    State left = makeState(2.0, {1.0, -2.0, 0.5}, 20.0);
    State right = makeState(3.0, {-0.25, 0.5, -1.0}, 20.0);
    double const initialEnergy = left.energy + right.energy;
    Vector const initialMomentum{left.momentum[0] + right.momentum[0],
        left.momentum[1] + right.momentum[1], left.momentum[2] + right.momentum[2]};
    Vector const transfer{0.75, -0.5, 0.25};
    for (int dimension = 0; dimension < 3; ++dimension) {
        left.momentum[dimension] -= transfer[dimension];
        right.momentum[dimension] += transfer[dimension];
    }
    left.energy -= 1.125;
    right.energy += 1.125;
    require(
        left.energy + right.energy == initialEnergy, "pair energy transfer was not conservative");
    auto const beforeLeft = left;
    auto const beforeRight = right;
    Vector const impulseLeft{0.5, -0.25, 0.75};
    Vector const impulseRight{-0.125, 0.375, -0.5};
    kick(left, impulseLeft);
    kick(right, impulseRight);
    for (int dimension = 0; dimension < 3; ++dimension) {
        long double const expected = initialMomentum[dimension] +
            static_cast<long double>(left.mass) * impulseLeft[dimension] +
            static_cast<long double>(right.mass) * impulseRight[dimension];
        require(near(left.momentum[dimension] + right.momentum[dimension], expected),
            "gravity corrupted equal-and-opposite pair momentum accounting");
    }
    long double const expectedEnergy =
        initialEnergy + kinetic(left) - kinetic(beforeLeft) + kinetic(right) - kinetic(beforeRight);
    require(near(left.energy + right.energy, expectedEnergy),
        "gravity work omitted the momentum received from an elapsed pair interval");
    require(near(thermal(left), thermal(beforeLeft), left.energy) &&
            near(thermal(right), thermal(beforeRight), right.energy),
        "gravity changed post-exchange thermal energy");
}

void galileanCovariance() {
    Vector const velocity{0.25, -0.75, 1.5};
    Vector const boost{7.0, -3.0, 11.0};
    Vector const impulse{-0.5, 0.125, 0.75};
    auto boostedVelocity = velocity;
    for (int dimension = 0; dimension < 3; ++dimension) {
        boostedVelocity[dimension] += boost[dimension];
    }
    State ordinary = makeState(3.0, velocity, 2.0);
    State boosted = makeState(3.0, boostedVelocity, 2.0);
    kick(ordinary, impulse);
    kick(boosted, impulse);
    long double expectedEnergy = ordinary.energy;
    for (int dimension = 0; dimension < 3; ++dimension) {
        expectedEnergy +=
            static_cast<long double>(boost[dimension]) * ordinary.momentum[dimension] +
            0.5L * ordinary.mass * boost[dimension] * boost[dimension];
    }
    require(near(boosted.energy, expectedEnergy), "gravity work changed under a uniform boost");
    require(near(thermal(ordinary), thermal(boosted), boosted.energy),
        "a uniform boost changed post-kick internal energy");
}

void unrepresentableImpulseDoesNotHeat() {
    State state = makeState(1.0, {1.0, -2.0, 4.0}, 3.0);
    State const initial = state;
    Vector const impulse{1.0e-30, -1.0e-30, 1.0e-30};
    for (int repetition = 0; repetition < 1000; ++repetition) {
        kick(state, impulse);
    }
    require(state.momentum == initial.momentum && state.energy == initial.energy,
        "an unrepresentable momentum kick injected energy");
}

}    // namespace detail

int main() {
    try {
        detail::signedKicksPreserveThermalEnergy();
        detail::pairTransferPrecedesGravity();
        detail::galileanCovariance();
        detail::unrepresentableImpulseDoesNotHeat();
        std::cout << "Hopkins conservative gravity impulse: all tests passed\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
