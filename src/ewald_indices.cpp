#include <tigerfmm/ewald_indices.hpp>
#include <tigerfmm/hpx.hpp>

HPX_PLAIN_ACTION(ewald_const::init, ewald_const_init_action);

void ewald_const::init() {
	vector<hpx::future<void>> futs;
	auto children = hpx_children();
	for( const auto& c : children ) {
		futs.push_back(hpx::async<ewald_const_init_action>(c));
	}

	ewald_const::init_gpu();

	hpx::wait_all(futs.begin(), futs.end());

}
