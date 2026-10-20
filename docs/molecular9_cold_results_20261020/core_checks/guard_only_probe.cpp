#include <cmath>
#include <cstdio>
#include <initializer_list>

#include <cosmictiger/chemistry/primordial_solver.hpp>

int main() {
    using namespace primordial;
    std::puts("T,rate_status,eos_status,rhs_status,thermal_status,H2plus_lte,H2plus_low,H2plus_bridge,thermal_e,cv,jacobian_ok");
    for (float const temperature : {10.0f,20.0f,30.0f,32.0f,35.0f,40.0f,50.0f,80.0f,99.0f,100.0f}) {
        Environment environment;
        environment.network=Network::Molecular9;environment.nHydrogen=1.0f;environment.cmbTemperature=136.275f;
        WorkingState state;state.abundances[0]=FloatFloat(0.0002f);state.abundances[4]=FloatFloat(0.01f);
        float capacity=0.0f;
        thermalEnergyAtTemperature(state,environment,temperature,state.thermalEnergy,capacity);
        Eos eos;auto const eosStatus=evaluateEos(state,environment,eos);
        RateCoefficients rates;auto const rateStatus=evaluateRateCoefficients(temperature,environment.network,rates);
        auto const c=composition(state,environment);evaluateDensityDependentRates(c,environment,temperature,rates);
        FloatFloat flux[kReactionCount];reactionFluxes(c,environment,rates,flux);ThermalRates thermal;
        auto const thermalStatus=evaluateThermalRates(c,environment,temperature,rates,flux,thermal);
        Derivative derivative;auto const rhsStatus=evaluateRhs(state,environment,derivative);
        auto const ionLte=(2.0e-19f/kElectronVolt)*std::pow(temperature,.1f)*std::exp(-3125.0f/temperature);
        auto const ionAtom=(1.36e-22f/kElectronVolt)*std::exp(-3152.0f/temperature);
        auto const ionElectron=(1.1e-19f/kElectronVolt)*std::pow(temperature,-.34f)*std::exp(-3025.0f/temperature);
        auto const ionLow=FloatFloat(environment.nHydrogen)*(c.hydrogen*FloatFloat(ionAtom)+c.electrons*FloatFloat(ionElectron));
        SolverOptions options;FloatFloat jacobian[7][7];
        auto const model=[](WorkingState const& x,Environment const& e,Derivative& d){return evaluateRhs(x,e,d);};
        bool const jacobianOk=rhsStatus==Status::Success&&primordial::detail::solverJacobian(state,environment,derivative,options,model,jacobian);
        std::printf("%.9g,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",temperature,int(rateStatus),int(eosStatus),int(rhsStatus),int(thermalStatus),ionLte,ionLow.toDouble(),thermal.molecularIonLineCooling.toDouble(),state.thermalEnergy.toDouble(),capacity,int(jacobianOk));
    }
}
