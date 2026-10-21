#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/chemistry/primordial_chemistry.hpp>
#include <cosmictiger/chemistry/primordial_float_environment.hpp>

namespace {
void require(bool condition, char const* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        throw std::runtime_error(message);
    }
}
struct CoupledReaction {
    float forward{8.0f}, reverse{2.0f}, heat{0.4f}, relaxation{1.0f}, equilibriumEnergy{2.0f};
    primordial::Status operator()(primordial::WorkingState const& state,
        primordial::Environment const& environment, primordial::Derivative& derivative) const {
        auto const status = primordial::validateComposition(state, environment);
        if (status != primordial::Status::Success) {
            return status;
        }
        if (!state.thermalEnergy.isFinite() || state.thermalEnergy <= FloatFloat(0.0f)) {
            return primordial::Status::InvalidInput;
        }
        derivative = {};
        derivative.abundances[0] = FloatFloat(forward) * (FloatFloat(1.0f) - state.abundances[0]) -
            FloatFloat(reverse) * state.abundances[0];
        derivative.thermalEnergy = FloatFloat(heat) * state.abundances[0] -
            FloatFloat(relaxation) * (state.thermalEnergy - FloatFloat(equilibriumEnergy));
        return primordial::Status::Success;
    }
};
struct QuadraticTraceLoss {
    float coefficient{};
    bool upper{};
    primordial::Status operator()(primordial::WorkingState const& state,
        primordial::Environment const& environment, primordial::Derivative& derivative) const {
        auto const status = primordial::validateComposition(state, environment);
        if (status != primordial::Status::Success) {
            return status;
        }
        FloatFloat const minority =
            upper ? FloatFloat(1.0f) - state.abundances[0] : state.abundances[0];
        FloatFloat const loss = FloatFloat(coefficient) * minority * minority;
        derivative = {};
        derivative.abundances[0] = upper ? loss : -loss;
        return primordial::Status::Success;
    }
};
void checkColdEvolution() {
    using namespace primordial;
    for (int scenario = 0; scenario != 3; ++scenario) {
        Environment environment;
        environment.network = Network::Molecular9;
        environment.heliumToHydrogenNumberRatio = 0.0f;
        environment.nHydrogen = scenario == 2 ? 1.0e4f : 0.0f;
        environment.cmbTemperature = scenario == 0 ? 80.0f : (scenario == 1 ? 20.0f : 0.0f);
        State initial;
        initial.abundances[scenario == 2 ? 4 : 0] = SignedAbundance::fromFraction(1.0f);
        float const initialTemperature = scenario == 0 ? 10.0f : 80.0f;
        require(thermalEnergyAtTemperature(initial, environment, initialTemperature,
                    initial.thermalEnergy) == Status::Success,
            "cold evolution initial state");
        // Fully ionized zero-density gas has constant composition and cv=3kB.
        // Compton exchange then gives the exact exponential T(t). The fully
        // molecular case uses an independent binary64 RK4 integration of the
        // CDS Table 2 cooling, LTE bridge, and J<=201 heat capacity; 1000/2000
        // RK4 steps agree within 2e-13 K at the binary32 interval below.
        double const comptonRate = 6.35264354e-26 *
            std::pow(static_cast<double>(environment.cmbTemperature), 4) /
            (3.0 * 8.617333262145e-5);
        float const interval = scenario == 2 ? 1.0e11f : static_cast<float>(2.0 / comptonRate);
        double const referenceTemperature = scenario == 2 ? 77.56686062469178 :
                                                            environment.cmbTemperature +
                (initialTemperature - environment.cmbTemperature) *
                    std::exp(-comptonRate * interval);
        double previousError = 1.0e99;
        unsigned previousSteps = 0;
        for (float tolerance : {1.0e-3f, 1.0e-4f, 1.0e-5f}) {
            SolverOptions options;
            options.relativeTolerance = tolerance;
            // At 10 K the default 1e-6 eV absolute budget dominates the
            // relative budget. Tighten it to measure timestep convergence.
            options.energyAbsoluteTolerance = 1.0e-10f;
            auto const result = advance(initial, environment, interval, options);
            Eos eos;
            require(result.status == Status::Success && result.acceptedTime == interval &&
                    evaluateEos(result.state, environment, eos) == Status::Success,
                "cold physical evolution completes");
            require(eos.temperature >= 10.0f && eos.temperature < 100.0f,
                "cold evolution stays in the supported domain");
            require(scenario == 0 ? eos.temperature > initialTemperature :
                                    eos.temperature < initialTemperature,
                "cold source has the correct heating or cooling sign");
            double const error = std::abs(eos.temperature - referenceTemperature);
            require(error < previousError * 0.45 && result.acceptedSubsteps > previousSteps,
                "cold physical solution converges with timestep tolerance");
            previousError = error;
            previousSteps = result.acceptedSubsteps;
            std::printf("cold scenario%d tolerance%.3g T%.9g reference%.12g steps%u\n", scenario,
                tolerance, eos.temperature, referenceTemperature, result.acceptedSubsteps);
        }
        require(previousError / referenceTemperature < 0.0015,
            "refined cold solution agrees with independent reference");
    }
    Environment environment;
    environment.network = Network::Molecular9;
    environment.cmbTemperature = 136.275f;
    auto const model = [](WorkingState const& state, Environment const& ambient,
                           Derivative& derivative) {
        return evaluateRhs(state, ambient, derivative);
    };
    for (float temperature : {10.0f, 30.0f, 35.0f, 36.0f, 40.0f, 49.0f, 80.0f}) {
        for (float density : {1.0f, 1.0e4f}) {
            environment.nHydrogen = density;
            for (float molecularIonFraction : {0.0f, 0.1f}) {
                WorkingState state;
                state.abundances[0] = FloatFloat(0.2f);
                state.abundances[4] = FloatFloat(0.1f);
                state.abundances[5] = FloatFloat(molecularIonFraction);
                float capacity;
                require(thermalEnergyAtTemperature(state, environment, temperature,
                            state.thermalEnergy, capacity) == Status::Success,
                    "cold Jacobian initial state");
                Derivative derivative;
                FloatFloat jacobian[primordial::detail::kSolverVariables]
                                   [primordial::detail::kSolverVariables];
                SolverOptions options;
                require(model(state, environment, derivative) == Status::Success &&
                        primordial::detail::solverJacobian(
                            state, environment, derivative, options, model, jacobian),
                    "cold endpoint and H2+ subnormal Jacobian probes remain finite");
            }
        }
    }
}
}    // namespace

TEST(PrimordialSolver, ColdEvolution) {
    checkColdEvolution();
}

TEST(PrimordialSolver, CoupledConvergence) {
    using namespace primordial;
    Environment environment;
    environment.heliumToHydrogenNumberRatio = 0.0f;
    State initial;
    initial.abundances[0] = SignedAbundance::fromFraction(0.1f);
    initial.thermalEnergy = 1.0f;
    CoupledReaction model;
    long double const time = 2.0L;
    long double const rate = static_cast<long double>(model.forward) + model.reverse;
    long double const xEquilibrium = model.forward / rate;
    long double const initialX = initial.abundances[0].value().toDouble();
    long double const equilibriumEnergy =
        model.equilibriumEnergy + model.heat * xEquilibrium / model.relaxation;
    long double const expectedX = xEquilibrium + (initialX - xEquilibrium) * std::exp(-rate * time);
    long double const expectedEnergy = equilibriumEnergy +
        (initial.thermalEnergy - equilibriumEnergy) * std::exp(-model.relaxation * time) +
        model.heat * (initialX - xEquilibrium) *
            (std::exp(-rate * time) - std::exp(-model.relaxation * time)) /
            (model.relaxation - rate);
    long double previousError = 1.0L;
    for (float tolerance : {1.e-3f, 1.e-4f, 1.e-5f}) {
        SolverOptions options;
        options.relativeTolerance = tolerance;
        StepResult const result =
            advanceWithModel(initial, environment, static_cast<float>(time), options, model);
        long double const error = std::fabs(result.state.thermalEnergy - expectedEnergy);
        std::printf("tol %.8g status%d x%.12g e%.12g error%.9Lg accepted%u rejected%u newton%u\n",
            tolerance, static_cast<int>(result.status),
            result.state.abundances[0].value().toDouble(), result.state.thermalEnergy, error,
            result.acceptedSubsteps, result.rejectedSubsteps, result.newtonIterations);
        require(result.status == Status::Success, "coupled integration completes");
        require(std::fabs(result.state.abundances[0].value().toDouble() - expectedX) < 2.e-5L,
            "stiff equilibrium reference");
        require(
            error < previousError * 0.5L, "first order refinement improves global thermal error");
        previousError = error;
    }
}

TEST(PrimordialSolver, QuadraticTraceLoss) {
    using namespace primordial;
    Environment environment;
    environment.heliumToHydrogenNumberRatio = 0.0f;
    State initial;
    initial.abundances[0] = SignedAbundance::fromFraction(0.1f);
    initial.thermalEnergy = 1.0f;
    // An oversized finite-difference probe used to report Success with the
    // initial abundance unchanged, although x'=-k*x^2 depleted it by 1000x.
    // Test both signed branches and trace scales against x=x0/(1+k*x0*t).
    for (float minority : {1.e-13f, 1.e-20f}) {
        for (bool upper : {false, true}) {
            State quadraticInitial = initial;
            quadraticInitial.abundances[0] = SignedAbundance::fromMinority(minority, upper);
            QuadraticTraceLoss const quadratic{minority > 1.e-15f ? 1.e6f : 1.e13f, upper};
            SolverOptions quadraticOptions;
            quadraticOptions.abundanceAbsoluteTolerance[0] = minority * 1.e-11f;
            auto quadraticResult = advanceWithModel(
                quadraticInitial, environment, 1.e10f, quadraticOptions, quadratic);
            long double const exact = static_cast<long double>(minority) /
                (1.0L + static_cast<long double>(quadratic.coefficient) * minority * 1.e10L);
            long double const actual = (upper ? quadraticResult.state.abundances[0].complement() :
                                                quadraticResult.state.abundances[0].value())
                                           .toDouble();
            std::printf("quadratic minority%.8g upper%d status%d relativeerror%.9Lg steps%u\n",
                minority, upper, static_cast<int>(quadraticResult.status),
                std::fabs(actual / exact - 1.0L), quadraticResult.acceptedSubsteps);
            require(
                quadraticResult.status == Status::Success && quadraticResult.acceptedTime == 1.e10f,
                "quadratic trace loss completes");
            require(std::fabs(actual / exact - 1.0L) < 0.015L && actual < minority * 0.01L,
                "trace nonlinear evolution cannot falsely converge unchanged");
        }
    }
}

TEST(PrimordialSolver, TraceEquilibriumAndDomains) {
    using namespace primordial;
    Environment environment;
    environment.heliumToHydrogenNumberRatio = 0.0f;
    State initial;
    initial.abundances[0] = SignedAbundance::fromFraction(0.1f);
    initial.thermalEnergy = 1.0f;
    CoupledReaction model;
    State trace = initial;
    trace.abundances[0] = SignedAbundance::fromMinority(1.e-12f, true);
    CoupledReaction traceModel;
    traceModel.forward = 1.0f;
    traceModel.reverse = 1.e-12f;
    traceModel.heat = 0.0f;
    traceModel.relaxation = 0.0f;
    SolverOptions options;
    StepResult const traced = advanceWithModel(trace, environment, 1.e10f, options, traceModel);
    std::printf("trace status%d minority%.15g raw%.15g accepted%u rejected%u\n",
        static_cast<int>(traced.status), traced.state.abundances[0].complement().toDouble(),
        traced.state.abundances[0].raw(), traced.acceptedSubsteps, traced.rejectedSubsteps);
    require(traced.status == Status::Success, "trace equilibrium completes");
    require(std::fabs(traced.state.abundances[0].complement().toDouble() / 1.e-12 - 1.0) < 1.e-6,
        "trace neutral retained");
    require(SignedAbundance::fromFraction(1.0f).upper() &&
            SignedAbundance::fromFraction(1.0f).complementFraction() == 0.0f,
        "fully ionized endpoint uses negative zero");
    require(!SignedAbundance::fromFraction(0.0f).upper(), "neutral endpoint uses positive zero");
    auto identity = advanceWithModel(trace, environment, 0.0f, options, traceModel);
    require(identity.status == Status::Success && identity.acceptedTime == 0.0f &&
            identity.state.abundances[0].raw() == trace.abundances[0].raw() &&
            identity.state.thermalEnergy == trace.thermalEnergy && identity.newtonIterations == 0,
        "zero interval identity");
    auto negative = advanceWithModel(trace, environment, -1.0f, options, traceModel);
    require(negative.status == Status::InvalidInput && negative.acceptedTime == 0.0f,
        "negative interval rejected before work");
    auto nonfinite = advanceWithModel(
        trace, environment, std::numeric_limits<float>::infinity(), options, traceModel);
    require(nonfinite.status == Status::InvalidInput, "infinite interval rejected");
    State bad = trace;
    bad.abundances[0] = SignedAbundance::fromRaw(std::numeric_limits<float>::quiet_NaN());
    require(advanceWithModel(bad, environment, 1.0f, options, traceModel).status ==
            Status::InvalidInput,
        "nonfinite abundance rejected");
    SolverOptions limited;
    limited.initialStepFraction = 1.e-4f;
    limited.maxSubsteps = 1;
    auto partial = advanceWithModel(initial, environment, 2.0f, limited, model);
    require(partial.status == Status::StepLimit && partial.acceptedTime > 0.0f &&
            partial.acceptedTime < 2.0f,
        "bounded work reports only accepted time");
    require(partial.state.abundances[0].value() > initial.abundances[0].value(),
        "bounded work preserves last accepted state");
    options.relativeTolerance = 1.e-8f;
    require(advanceWithModel(trace, environment, 1.0f, options, traceModel).status ==
            Status::InvalidInput,
        "unvalidated precision request rejected");
}

TEST(PrimordialSolver, PhotoionizationAndHeating) {
    using namespace primordial;
    Environment environment;
    environment.heliumToHydrogenNumberRatio = 0.0f;
    State initial;
    initial.abundances[0] = SignedAbundance::fromFraction(0.1f);
    initial.thermalEnergy = 1.0f;
    SolverOptions options;
    // Independent full-physics reference: zero density and TCMB remove collisions
    // and Compton exchange. Photoionization gives dx/dt=Gamma(1-x), while
    // de/dt=Q(1-x), so e-e0=(Q/Gamma)(x-x0) exactly.
    Environment photoEnvironment;
    photoEnvironment.nHydrogen = 0.0f;
    photoEnvironment.cmbTemperature = 0.0f;
    photoEnvironment.heliumToHydrogenNumberRatio = 0.0f;
    photoEnvironment.radiation.hydrogenIonization = 2.e-7f;
    photoEnvironment.radiation.heatingPerHydrogen = 6.e-7f;
    State photoInitial = initial;
    require(thermalEnergyAtTemperature(photoInitial, photoEnvironment, 1.e4f,
                photoInitial.thermalEnergy) == Status::Success,
        "photo reference initial EOS");
    options = SolverOptions{};
    float const photoInterval = 1.e7f;
    auto photo = advance(photoInitial, photoEnvironment, photoInterval, options);
    require(photo.status == Status::Success && photo.acceptedTime == photoInterval,
        "physical photo solve completes");
    long double const photoX0 = photoInitial.abundances[0].value().toDouble();
    long double const photoExact = 1.0L -
        (1.0L - photoX0) *
            std::exp(-static_cast<long double>(photoEnvironment.radiation.hydrogenIonization) *
                photoInterval);
    long double const photoX = photo.state.abundances[0].value().toDouble();
    long double const deposited =
        static_cast<long double>(photoEnvironment.radiation.heatingPerHydrogen) /
        photoEnvironment.radiation.hydrogenIonization * (photoX - photoX0);
    require(
        std::fabs(photoX - photoExact) < 1.5e-3L, "physical photo independent analytic reference");
    require(std::fabs(photo.state.thermalEnergy - photoInitial.thermalEnergy - deposited) < 2.e-6L,
        "species and photo heating remain coupled");
    std::printf("photo x%.12Lg expected%.12Lg thermal%.12g steps%u\n", photoX, photoExact,
        photo.state.thermalEnergy, photo.acceptedSubsteps);
}

TEST(PrimordialSolver, MolecularReference) {
    using namespace primordial;
    SolverOptions options;
    // Low-density molecular fixture independently integrated by SciPy Radau
    // through primordial_reference.py, using the production RHS (no copied
    // network). It exercises fast H-/H2+ relaxation from exact zero.
    Environment molecularEnvironment;
    molecularEnvironment.network = Network::Molecular9;
    molecularEnvironment.nHydrogen = 1.0f;
    molecularEnvironment.cmbTemperature = 30.0f;
    State molecularInitial;
    molecularInitial.abundances[0] = SignedAbundance::fromFraction(1.e-4f);
    molecularInitial.abundances[4] = SignedAbundance::fromFraction(2.e-6f);
    require(thermalEnergyAtTemperature(molecularInitial, molecularEnvironment, 1000.0f,
                molecularInitial.thermalEnergy) == Status::Success,
        "molecular initial EOS");
    auto molecular = advance(molecularInitial, molecularEnvironment, 1.e12f, options);
    require(molecular.status == Status::Success, "stiff molecular production solve completes");
    double const molecularReference[kAbundanceCount] = {9.997928678789275e-5, 0.0, 0.0,
        6.314741092653406e-11, 2.165038068128412e-6, 1.529134142100652e-12};
    for (int i = 0; i < kAbundanceCount; ++i) {
        double const value = molecular.state.abundances[i].value().toDouble();
        require(std::fabs(value - molecularReference[i]) <= 3.e-5 * molecularReference[i] + 1.e-20,
            "stiff molecular independent reference");
    }
    require(std::fabs(molecular.state.thermalEnergy - 0.1394771634018972) < 1.e-7,
        "molecular thermal reference");
    require(validateComposition(molecular.state, molecularEnvironment) == Status::Success,
        "molecular accepted state satisfies elemental and charge bounds");
}

TEST(PrimordialSolver, ThermalEndpoints) {
    using namespace primordial;
    SolverOptions options;
    // At exact thermal endpoints the Newton Jacobian must find feasible
    // derivative probes while the physical trajectory points into the domain.
    for (int endpoint = 0; endpoint != 2; ++endpoint) {
        Environment boundaryEnvironment;
        State boundaryInitial;
        float interval;
        if (endpoint == 0) {
            boundaryEnvironment.nHydrogen = 1.0f;
            boundaryEnvironment.cmbTemperature = 30.0f;
            boundaryInitial.abundances[0] = SignedAbundance::fromFraction(1.0f);
            boundaryInitial.abundances[2] = SignedAbundance::fromFraction(1.0f);
            interval = 1.e12f;
        } else {
            boundaryEnvironment.network = Network::Molecular9;
            boundaryEnvironment.nHydrogen = 0.0f;
            boundaryEnvironment.cmbTemperature = 0.0f;
            boundaryEnvironment.radiation.heatingPerMolecularHydrogen = 1.e-10f;
            boundaryInitial.abundances[4] = SignedAbundance::fromFraction(1.0f);
            interval = 1.e8f;
        }
        float const temperature = endpoint == 0 ? maximumTemperature(boundaryEnvironment.network) :
                                                  minimumTemperature(boundaryEnvironment.network);
        require(thermalEnergyAtTemperature(boundaryInitial, boundaryEnvironment, temperature,
                    boundaryInitial.thermalEnergy) == Status::Success,
            "thermal endpoint initial state");
        auto boundary = advance(boundaryInitial, boundaryEnvironment, interval, options);
        Eos boundaryEos;
        require(boundary.status == Status::Success && boundary.acceptedTime == interval &&
                boundary.newtonIterations > 3 &&
                evaluateEos(boundary.state, boundaryEnvironment, boundaryEos) == Status::Success,
            "thermal endpoint Jacobian remains feasible");
        require(endpoint == 0 ? boundaryEos.temperature < temperature :
                                boundaryEos.temperature > temperature,
            "endpoint trajectory moves into supported thermal domain");
        if (endpoint == 1) {
            double const exactEnergy = boundaryInitial.thermalEnergy +
                0.5 *
                    static_cast<double>(boundaryEnvironment.radiation.heatingPerMolecularHydrogen) *
                    interval;
            require(std::fabs(boundary.state.thermalEnergy - exactEnergy) < 2.e-9,
                "endpoint feasibility does not alter physical heating");
        }
    }
}

TEST(PrimordialSolver, CpuBatchAndFloatingEnvironment) {
    // Restore the caller on failure as well as on the successful path below.
    primordial::StrictFloatEnvironment const savedEnvironment;
    ASSERT_TRUE(savedEnvironment.valid());
    using namespace primordial;
    Environment environment;
    environment.heliumToHydrogenNumberRatio = 0.0f;
    State initial;
    initial.abundances[0] = SignedAbundance::fromFraction(0.1f);
    initial.thermalEnergy = 1.0f;
    SolverOptions options;
    Environment photoEnvironment;
    photoEnvironment.nHydrogen = 0.0f;
    photoEnvironment.cmbTemperature = 0.0f;
    photoEnvironment.heliumToHydrogenNumberRatio = 0.0f;
    photoEnvironment.radiation.hydrogenIonization = 2.e-7f;
    photoEnvironment.radiation.heatingPerHydrogen = 6.e-7f;
    State photoInitial = initial;
    require(thermalEnergyAtTemperature(photoInitial, photoEnvironment, 1.e4f,
                photoInitial.thermalEnergy) == Status::Success,
        "photo reference initial EOS");
    options = SolverOptions{};
    float const photoInterval = 1.e7f;
    std::vector<BatchInput> batch(7);
    for (auto& input : batch) {
        input.state = photoInitial;
        input.environment = photoEnvironment;
        input.interval = photoInterval;
    }
    batch[1].interval = 0.0f;
    batch[2].interval = -1.0f;
    batch[3].environment.nHydrogen = 1.e5f;
    batch[4].state.thermalEnergy = std::numeric_limits<float>::quiet_NaN();
    batch[5].state.abundances[3] = SignedAbundance::fromFraction(0.01f);
    std::vector<StepResult> one(batch.size()), parallel(batch.size());
    require(integrateBatchCpu(nullptr, nullptr, 0), "empty CPU batch");
    require(!integrateBatchCpu(nullptr, one.data(), 1), "null CPU input rejected");
    require(integrateBatchCpu(batch.data(), one.data(), batch.size(), options, 1),
        "CPU batch dispatch");
    require(integrateBatchCpu(batch.data(), parallel.data(), batch.size(), options, 2),
        "parallel CPU batch dispatch");
    for (size_t i = 0; i < batch.size(); ++i) {
        require(
            one[i].status == parallel[i].status && one[i].acceptedTime == parallel[i].acceptedTime,
            "CPU worker count preserves lane status");
        if (one[i].status == Status::Success) {
            require(one[i].state.thermalEnergy == parallel[i].state.thermalEnergy &&
                    one[i].state.abundances[0].raw() == parallel[i].state.abundances[0].raw(),
                "CPU worker count preserves states exactly");
        }
    }
    // A caller can arrive with FTZ/DAZ and non-nearest rounding. CPU workers
    // establish their own strict environment and restore the caller on return.
    int const originalRounding = std::fegetround();
#if defined(__SSE__)
    unsigned const originalControl = _mm_getcsr();
#endif
    require(std::fesetround(FE_DOWNWARD) == 0, "set caller rounding for regression");
#if defined(__SSE__)
    _mm_setcsr(_mm_getcsr() | (1U << 15) | (1U << 6));
    unsigned const callerControl = _mm_getcsr();
#endif
    require(integrateBatchCpu(batch.data(), parallel.data(), batch.size(), options, 2),
        "CPU batch establishes strict floating environment");
    require(std::fegetround() == FE_DOWNWARD, "CPU batch restores caller rounding");
#if defined(__SSE__)
    require(_mm_getcsr() == callerControl, "CPU batch restores caller FTZ and DAZ");
#endif
    std::fesetround(originalRounding);
#if defined(__SSE__)
    _mm_setcsr(originalControl);
#endif
    require(one[0].state.thermalEnergy == parallel[0].state.thermalEnergy &&
            one[0].state.abundances[0].raw() == parallel[0].state.abundances[0].raw(),
        "CPU results independent of caller FTZ/DAZ and rounding");
    require(one[0].status == Status::Success && one[1].status == Status::Success &&
            one[2].status == Status::InvalidInput && one[3].status == Status::UnsupportedState &&
            one[4].status == Status::InvalidInput && one[5].status == Status::InvalidInput,
        "batch carries independent success and failure statuses");
#ifndef USE_CUDA
    std::string backendError;
    require(!integrateBatchCuda(batch.data(), one.data(), batch.size(), options, &backendError) &&
            !backendError.empty() && one[0].status == Status::BackendUnavailable &&
            one[0].state.thermalEnergy == batch[0].state.thermalEnergy,
        "unavailable CUDA fails explicitly without changing state");
#endif
}
