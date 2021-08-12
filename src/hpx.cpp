#include <cosmictiger/hpx.hpp>
#include <cosmictiger/cuda.hpp>

static int rank;
static int nranks;
static vector<hpx::id_type> localities;
static vector<hpx::id_type> children;

HPX_PLAIN_ACTION (hpx_init);

void hpx_yield() {
	hpx::this_thread::yield();
}


int hpx_hardware_concurrency() {
	return hpx::thread::hardware_concurrency();
}

void hpx_init() {
	rank = hpx::get_locality_id();
	auto tmp = hpx::find_all_localities();
	localities.insert(localities.end(), tmp.begin(), tmp.end());
	nranks = localities.size();
	int base = (rank + 1) << 1;
	const int index1 = base - 1;
	const int index2 = base;
	if (index1 < nranks) {
		children.push_back(localities[index1]);
	}
	if (index2 < nranks) {
		children.push_back(localities[index2]);
	}
#ifdef USE_CUDA
	cuda_init();
#endif

	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async < hpx_init_action > (c));
	}
	hpx::wait_all(futs.begin(), futs.end());

}

int hpx_rank() {
	return rank;
}

int hpx_size() {
	return nranks;
}

const vector<hpx::id_type>& hpx_localities() {
	return localities;
}

const vector<hpx::id_type>& hpx_children() {
	return children;
}
