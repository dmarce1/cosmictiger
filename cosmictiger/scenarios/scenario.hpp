#pragma once

#include <string>
#include <vector>

namespace scenarios {

struct EvolutionStage {
    double duration;
    int numberOfSteps;
    double damping{0.0};
    double viscosityAlpha{-1.0};
};

struct ScenarioPlan {
    std::vector<EvolutionStage> stages;
    void (*finish)(double time){nullptr};
};

struct ScenarioDefinition {
    char const* name;
    bool gravity;
    bool hydro;
    bool fixedTime;
    int numberOfSteps;
    ScenarioPlan (*initialize)();
};

ScenarioDefinition const& definition(std::string const& name);

}    // namespace scenarios
