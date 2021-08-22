#include <cosmictiger/options.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/view.hpp>

#include <atomic>

#include <chealpix.h>

vector<float> output_view(int number);

HPX_PLAIN_ACTION (output_view);

vector<float> output_view(int number) {
	vector<hpx::future<void>> futs;
	vector<hpx::future<vector<float>>>val_futs;
	for (const auto& c : hpx_children()) {
		val_futs.push_back(hpx::async < output_view_action > (c, number));
	}
	const int nthreads = hpx::thread::hardware_concurrency();
	const int Nside = get_options().view_size;
	const int Npix = 12 * sqr(Nside);
	vector<std::atomic<float>> values(Npix);
	for (auto& v : values) {
		v = 0.0f;
	}
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads,proc,Nside,&values]() {
			const part_int begin = (size_t) proc * particles_size() / nthreads;
			const part_int end = (size_t) (proc + 1) * particles_size() / nthreads;
			for( int i = begin; i < end; i++) {
				for( int xi = -1; xi <= 0; xi++) {
					for( int yi = -1; yi <= 0; yi++) {
						for( int zi = -1; zi <= 0; zi++) {
							double vec[NDIM];
							long int ipix;
							vec[XDIM] = particles_pos(XDIM,i).to_float() + xi;
							vec[YDIM] = particles_pos(YDIM,i).to_float() + yi;
							vec[ZDIM] = particles_pos(ZDIM,i).to_float() + zi;
							const float r2 = sqr(vec[XDIM],vec[YDIM],vec[ZDIM]);
							if( r2 < 1.0 ) {
								vec2pix_ring(Nside, vec, &ipix);
								atomic_add(values[ipix], 1.0 / r2);
							}
						}
					}
				}
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	vector<float> results(values.size());
	for (int i = 0; i < results.size(); i++) {
		results[i] = values[i];
	}
	for (auto& val_fut : val_futs) {
		const auto vals = val_fut.get();
		for (int i = 0; i < vals.size(); i++) {
			results[i] += vals[i];
		}
	}
	if (hpx_rank() == 0) {
		std::string filename = "view." + std::to_string(number) + ".dat";
		FILE* fp = fopen(filename.c_str(), "wb");
		if (fp == NULL) {
			THROW_ERROR("unable to open %s for writing\n", filename.c_str());
		}
		fwrite(&Nside, sizeof(int), 1, fp);
		fwrite(&Npix, sizeof(int), 1, fp);
		fwrite(results.data(), sizeof(float), Npix, fp);
		fclose(fp);
	}
	return results;
}
