from pathlib import Path
root=Path(__file__).resolve().parents[2]
Path('/tmp/eb_integrator_audit').mkdir(exist_ok=True)
source=(root/'src/cosmology/boltzmann.cpp').read_text()
body=source[source.index('void einsteinBoltzmannInit('):source.index('void einsteinBoltzmannInterpolationFunction(')]
zero=(root/'src/cosmology/zero_order.cpp').read_text()
zero=zero[zero.index('void ZeroOrderUniverse::computeMatterFractions'):]
prefix='''#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cosmictiger/cosmology/boltzmann.hpp>
#include <cosmictiger/cosmology/recombination_thermal.hpp>
#include <cosmictiger/cosmology/recfast.hpp>
#define THROW_ERROR(...) throw std::runtime_error("domain")
#define PRINT(...) do {} while (false)
'''
main='''
int main(int argc, char** argv) {
    using namespace constants;
    auto const history=recfast::loadIonizationHistory(argv[1]);
    CosmicParameters p{};
    p.omegaB=.049389; p.omegaC=.26503; p.neff=3.046; p.y=.2454006;
    p.theta=2.7255/kTcmb; p.hubble=.6732;
    double const rhoc=3*pow(p.hubble*kH0,2)/(8*M_PI*kG);
    p.omegaGam=4*kSigma/kC*pow(2.7255,4)/(rhoc*kC*kC);
    p.omegaNu=p.omegaGam*p.neff*(7./8)*pow(4./11,4./3);
    ZeroOrderUniverse uni;
    createZeroOrderUniverse(&uni,history,1.1,p);
    for (double k: {.01,.1,1.}) {
        CosState u{};
        einsteinBoltzmannInit(&u,&uni,k,1.,uni.amin,.966);
        einsteinBoltzmann(&u,&uni,k,uni.amin,1.);
        printf("cap=CAP k=%.8g dc=%.12g db=%.12g dg=%.12g vb=%.12g vg=%.12g F2=%.12g\\n",k,u[deltaci],u[deltabi],u[deltagami],u[thetabi],u[thetagami],u[F2i]);
    }
}
'''
for cap in ['5e-2','2.5e-2','1.25e-2','6.25e-3','3.125e-3']:
    Path('/tmp/eb_integrator_audit/background_'+cap+'.cpp').write_text(prefix+zero+body.replace('(double) 5e-2','(double) '+cap)+main.replace('CAP',cap))
