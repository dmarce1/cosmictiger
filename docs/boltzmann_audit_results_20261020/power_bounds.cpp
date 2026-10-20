#include <cmath>
#include <cstdio>
#include <vector>
template<class T> using Vector = std::vector<T>;
#define PRINT(...) std::printf(__VA_ARGS__)
struct PowerSpectrumFunction {
    Vector<float> pValue2;
    float logkmin;
    float logkmax;
    float dlogk;
    float operator()(float k) const {
        float logk = std::log(k);
        float const k0 = (logk - logkmin) / dlogk;
        int i0 = int(k0) - 1;
        if (i0 < 0) {
            PRINT("power.init does not have sufficient range min %e max %e tried %e\n",
                exp(logkmin), exp(logkmax), k);
        }
        if (i0 > pValue2.size() - 4) {
            PRINT("power.init does not have sufficient range min %e max %e tried %e\n",
                exp(logkmin), exp(logkmax), k);
        }
        int i1 = i0 + 1;
        int i2 = i0 + 2;
        int i3 = i0 + 3;
        float x = k0 - i1;
        float y0 = std::log(pValue2[i0]);
        float y1 = std::log(pValue2[i1]);
        float y2 = std::log(pValue2[i2]);
        float y3 = std::log(pValue2[i3]);
        float k1 = (y2 - y0) * 0.5;
        float k2 = (y3 - y1) * 0.5;
        float a = y1;
        float b = k1;
        float c = -2 * k1 - k2 - 3 * y1 + 3 * y2;
        float d = k1 + k2 + 2 * y1 - 2 * y2;
        return std::exp(a + b * x + c * x * x + d * x * x * x);
    }
};

int main() {
    PowerSpectrumFunction f;
    f.logkmin = std::log(1e-5 * .6732);
    f.logkmax = std::log(25.271 * .6732);
    f.dlogk = (f.logkmax-f.logkmin)/1024;
    f.pValue2.resize(1024, 1.f);
    std::printf("power=%g\n", f(50.f));
}
