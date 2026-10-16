#include <cosmictiger/direct.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/random_number_generator.hpp>

struct TestParticle {
    std::array<Fixed32, NDIM> x;
    gforce f;
    template <class A>
    void serialize(A&& arc, unsigned) {
        arc & x;
        arc & f;
    }
};

#ifdef TEST_FORCE

std::vector<TestParticle> directGetTestParticles(int cnt) {
    particle_server pserv;
    auto const& parts = pserv.get_particle_set();
    std::vector<TestParticle> testparts(cnt);
    std::set<int> used;
    for (int i = 0; i < cnt; i++) {
        int index;
        do {
            index = rand() % parts.size();
        } while (used.find(index) != used.end());
        used.insert(index);
        for (int dim = 0; dim < NDIM; dim++) {
            testparts[i].x[dim] = parts.pos(dim, index);
            testparts[i].f.f[dim] = parts.force(dim, index);
        }
        testparts[i].f.phi = parts.pot(index);
    }
    return testparts;
}

HPX_PLAIN_ACTION(directGetTestParticles, DirectGetTestParticlesAction);
HPX_PLAIN_ACTION(cuda_direct, CudaDirectAction);

void directForceTest() {
    particle_server pserv;
    std::vector<int> counts(hpxSize());
    for (int i = 0; i < N_TEST_PARTS; i++) {
        Array<Fixed32, NDIM> x;
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = randFixed32();
        }
        counts[pserv.get_domain_bounds().find_proc(x)]++;
    }
    std::vector<hpx::future<std::vector<TestParticle>>> futs1;
    for (int i = 0; i < hpxSize(); i++) {
        futs1.push_back(hpx::async<DirectGetTestParticlesAction>(hpxLocalities()[i], counts[i]));
    }
    std::vector<TestParticle> testParts;
    for (auto& f : futs1) {
        auto tmp = f.get();
        testParts.insert(testParts.end(), tmp.begin(), tmp.end());
    }
    std::vector<std::array<Fixed32, NDIM>> pts;
    for (int i = 0; i < testParts.size(); i++) {
        pts.push_back(testParts[i].x);
    }
    std::vector<hpx::future<std::vector<gforce>>> futs2;
    for (int i = 0; i < hpxSize(); i++) {
        futs2.push_back(hpx::async<CudaDirectAction>(hpxLocalities()[i], pts));
    }

    std::vector<gforce> total(pts.size());
    for (int i = 0; i < pts.size(); i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            total[i].f[dim] = 0.0;
        }
        total[i].phi = 0.0;
    }
    for (auto& f : futs2) {
        auto tmp = f.get();
        for (int i = 0; i < pts.size(); i++) {
            for (int dim = 0; dim < NDIM; dim++) {
                total[i].f[dim] += tmp[i].f[dim];
            }
            total[i].phi += tmp[i].phi;
        }
    }
    double gerr = 0.0, phierr = 0.0;
    double gnorm = 0.0, phinorm = 0.0;
    for (int i = 0; i < pts.size(); i++) {
        double gnum = 0.0, gdir = 0.0;
        double phinum, phidir;
        for (int dim = 0; dim < NDIM; dim++) {
            gdir += sqr(total[i].f[dim]);
            gnum += sqr(testParts[i].f.f[dim]);
        }
        gdir = std::sqrt(gdir);
        gnum = std::sqrt(gnum);
        phinum = testParts[i].f.phi;
        phidir = total[i].phi - PHI0 * global().opts.G * global().opts.M;
        gnorm += gdir;
        phinorm += std::abs(phidir);
        gerr += std::abs(gdir - gnum);
        phierr += std::abs(phidir - phinum);
        //	printf( "%e %e %e %e\n", gdir, gnum, phidir, phinum);
    }
    gerr /= gnorm;
    phierr /= phinorm;
    printf("gerr = %e\n", gerr);
    printf("phierr = %e\n", phierr);
}

#endif
