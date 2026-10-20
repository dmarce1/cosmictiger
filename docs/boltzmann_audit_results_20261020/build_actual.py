from pathlib import Path
root=Path(__file__).resolve().parents[2]
Path('/tmp/eb_integrator_audit').mkdir(exist_ok=True)
source=(root/'src/cosmology/boltzmann.cpp').read_text()
body=source[source.index('void einsteinBoltzmannInit('):source.index('void einsteinBoltzmannInterpolationFunction(')]
macros=(root/'cosmictiger/cosmology/boltzmann.hpp').read_text()
macros=macros[macros.index('#define LMAX'):macros.index('#include <array>')]
zero=(root/'src/cosmology/zero_order.cpp').read_text()
fractions=zero[zero.index('void ZeroOrderUniverse::computeMatterFractions'):zero.index('double ZeroOrderUniverse::conformalTimeToScaleFactor')]
prefix='''#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>
#define THROW_ERROR(...) throw std::runtime_error("domain")
struct CosmicParameters { double omegaB, omegaC, omegaGam, omegaNu; double neff, y, theta, hubble; };
struct ZeroOrderUniverse {
    double amin=1e-10, amax=1.1;
    CosmicParameters params{.05,.25,.00005,.000034,3.04,.24,1.,.7};
    std::function<double(double)> cs2=[](double) {return 0.;};
    std::function<double(double)> sigmaT=[](double) {return 0.;};
    void computeMatterFractions(double&,double&,double) const;
    void computeRadiationFractions(double&,double&,double) const;
};
inline double hubbleFunction(double a, double littleh, double omegaM, double omegaR) {
    return littleh*(1e7/2.99792458e10)*sqrt(omegaR/pow(a,4)+omegaM/pow(a,3)+(1-omegaM-omegaR));
}
'''
prefix+=macros+'\nusing CosState=std::array<double,NFIELD>;\nvoid einsteinBoltzmann(CosState*,ZeroOrderUniverse const*,double,double,double);\n'+fractions
# Unmodified source for terminal fixture and instrumented sigma times.
main='''
int main() {
    ZeroOrderUniverse uni;
    double const amin=1e-7;
    double const hubble=hubbleFunction(amin,uni.params.hubble,.3,.000084);
    for (double h: {.05,.025,.0125,.001,.000001}) {
        CosState u{};
        u[taui]=1./(amin*hubble);
        u[FLi+LMAX-1]=1.;
        std::vector<double> times;
        uni.sigmaT=[&](double a){ times.push_back(log(a/amin)/h); return 0.;};
        einsteinBoltzmann(&u,&uni,1e-9,amin,amin*exp(h));
        printf("terminal h=%.8g result=%.12g times=",h,u[FLi+LMAX-1]);
        for (double t:times) { printf(" %.8g",t); }
        printf("\\n");
    }
}
'''
Path('/tmp/eb_integrator_audit/actual_terminal.cpp').write_text(prefix+body+main)
# Source extracted evolution with optional timestep cap, constant electron fraction,
# sound speed zero (explicit controlled fixture, no RECFAST claim).
main2='''
int main() {
    ZeroOrderUniverse uni;
    double const amin=3.355e-9;
    uni.sigmaT=[&](double a){
        double const hcgs=uni.params.hubble*(1e7/3.086e24)*sqrt(.000084/pow(a,4)+.3/pow(a,3)+.699916);
        double const rhoc=3*pow(uni.params.hubble*(1e7/3.086e24),2)/(8*M_PI*6.67430e-8);
        double const ne=.05*rhoc*(1-.24)/1.6735575e-24/pow(a,3);
        return 2.99792458e10*6.6524587321e-25*ne/hcgs;
    };
    for (double k: {.01,.1,1.}) {
        CosState u{};
        einsteinBoltzmannInit(&u,&uni,k,1.,amin,.966);
        einsteinBoltzmann(&u,&uni,k,amin,.001);
        printf("cap=CAP k=%.8g dc=%.12g db=%.12g dg=%.12g vb=%.12g vg=%.12g F2=%.12g\\n",k,u[deltaci],u[deltabi],u[deltagami],u[thetabi],u[thetagami],u[F2i]);
    }
}
'''
for cap in ['5e-2','2.5e-2','1.25e-2','6.25e-3']:
    Path('/tmp/eb_integrator_audit/actual_'+cap+'.cpp').write_text(prefix+body.replace('(double) 5e-2','(double) '+cap)+main2.replace('CAP',cap))
