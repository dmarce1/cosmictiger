/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cosmictiger/domain.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/tree.hpp>

namespace detail {
constexpr float gravityScale = 1.75f;
void requireClose(char const* label, double actual, double expected, double tolerance = 3.0e-5) {
	if (!(std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)))) {
		throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
	}
}
void initializePair(double separation) {
	particles_resize(2);
	sph_particles_resize(1, false);
	for (part_int i = 0; i < 2; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			particles_pos(dim, i) = dim == XDIM ? 0.375 - i * separation : 0.375;
			particles_vel(dim, i) = i == 1 ? 0.125 * (dim + 1) : 0.0;
		}
		particles_rung(i) = 1;
		particles_type(i) = i == 0 ? DARK_MATTER_TYPE : SPH_TYPE;
		particles_cat_index(i) = i == 0 ? NO_INDEX : 0;
	}
	sph_particles_dm_index(0) = 1;
	sph_particles_smooth_len(0) = 0.02;
#ifdef ENTROPY
	sph_particles_r2[0].A = 4.0;
#else
	sph_particles_r2[0].eint = 4.0;
#endif
	sph_particles_r1[0].frac[0] = 0.75;
	sph_particles_r5[0].dfrac[0] = 0.125;
	sph_particles_gforce(XDIM, 0) = 0.0;
}
tree_id findLeaf(tree_id node, part_int particle) {
	auto const* ptr = tree_get_node(node);
	if (ptr->leaf) {
		return node;
	}
	for (int child = 0; child < NCHILD; child++) {
		auto const next = ptr->children[child];
		auto const range = tree_get_node(next)->part_range;
		if (particle >= range.first && particle < range.second) {
			return findLeaf(next, particle);
		}
	}
	throw std::runtime_error("Particle missing from gravity tree");
}
void verifyMappings() {
	for (part_int k = 0; k < sph_particles_size(); k++) {
		auto const i = sph_particles_dm_index(k);
		if (i < 0 || i >= particles_size() || !particles_is_sph(i) || particles_cat_index(i) != k) {
			throw std::runtime_error("Gas category mapping was not preserved");
		}
	}
}
kick_params makeKickParams() {
	kick_params p;
	p.htime = false;
	p.sign = 1.0;
	p.min_level = 0;
	p.min_rung = 0;
	p.a = 1.0;
	p.t0 = 0.001;
	p.theta = 0.5;
	p.h = get_options().hsoft;
	p.eta = get_options().eta;
	p.GM = gravityScale;
	p.vel_init = false;
	p.save_force = true;
	p.first_call = true;
	p.gpu = false;
	p.node_load = 1.0;
	p.ascending = true;
	p.descending = true;
	p.top = true;
	p.do_phi = true;
	return p;
}
void resetKickState() {
	for (part_int i = 0; i < particles_size(); i++) {
		particles_rung(i) = 1;
		for (int dim = 0; dim < NDIM; dim++) {
			particles_vel(dim, i) = particles_is_sph(i) ? 0.125 * (dim + 1) : 0.0;
			particles_gforce(dim, i) = 0.0;
		}
		particles_pot(i) = 0.0;
	}
}
void verifyKick(double inverseCube, double inverseRadius, double separation) {
	for (part_int i = 0; i < particles_size(); i++) {
		bool const gas = particles_is_sph(i);
		double const sourceMass = gas ? get_options().dm_mass : get_options().sph_mass;
		double const acceleration = gravityScale * (gas ? 1.0 : -1.0) * sourceMass * separation * inverseCube;
		requireClose("kick acceleration", particles_gforce(XDIM, i), acceleration);
		requireClose("kick potential with self mass removed", particles_pot(i), -gravityScale * sourceMass * inverseRadius, 2e-4);
		requireClose("transverse force", particles_gforce(YDIM, i), 0.0);
		if (gas) {
			requireClose("gas acceleration handoff", sph_particles_gforce(XDIM, particles_cat_index(i)), acceleration);
			for (int dim = 0; dim < NDIM; dim++) {
				requireClose("gas velocity must not be gravity-kicked twice", particles_vel(dim, i), 0.125 * (dim + 1));
			}
		} else if (particles_vel(XDIM, i) == 0.0) {
			throw std::runtime_error("Ordinary DM kick did not update velocity");
		}
	}
}
void verifySynchronizedDiagnostics(kick_return const& result) {
	double kinetic = 0.0, xMomentum = 0.0, yMomentum = 0.0, zMomentum = 0.0, momentumNorm = 0.0;
	for (part_int i = 0; i < particles_size(); i++) {
		double const mass = particles_mass(i);
		double const vx = particles_vel(XDIM, i), vy = particles_vel(YDIM, i), vz = particles_vel(ZDIM, i);
		kinetic += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
		xMomentum += mass * vx;
		yMomentum += mass * vy;
		zMomentum += mass * vz;
		momentumNorm += mass * std::sqrt(vx * vx + vy * vy + vz * vz);
	}
	double const gasMass = get_options().sph_mass;
	double const initialSpeed2 = 0.125 * 0.125 * (1.0 + 4.0 + 9.0);
	requireClose("synchronized kinetic diagnostic", kinetic + result.dmKineticCorrection, 0.5 * gasMass * initialSpeed2);
	requireClose("synchronized x momentum", xMomentum + result.dmXMomentumCorrection, gasMass * 0.125);
	requireClose("synchronized y momentum", yMomentum + result.dmYMomentumCorrection, gasMass * 0.25);
	requireClose("synchronized z momentum", zMomentum + result.dmZMomentumCorrection, gasMass * 0.375);
	requireClose("synchronized momentum norm", momentumNorm + result.dmMomentumNormCorrection, gasMass * std::sqrt(initialSpeed2));
}

void runPairCase(bool useCuda, bool softened) {
	double const nominalSeparation = softened ? 0.0025 : 0.25;
	initializePair(nominalSeparation);
	auto const built = tree_create(tree_create_params(0, 0.5, get_options().hsoft));
	verifyMappings();
	requireClose("root mixed mass", tree_get_node(built.id)->mpos->multi[0], 1.0);
	part_int const gas = sph_particles_dm_index(0);
	part_int const dm = 1 - gas;
	double const separation = particles_pos(XDIM, dm).to_double() - particles_pos(XDIM, gas).to_double();
	double inverseCube = 1.0 / (separation * separation * separation);
	double inverseRadius = 1.0 / separation;
	if (softened) {
		double const h = get_options().hsoft;
		double const q2 = separation * separation / (h * h);
		inverseCube = (1155.0 / 128.0 + q2 * (-693.0 / 32.0 + q2 * (1485.0 / 64.0 + q2 * (-385.0 / 32.0 + q2 * 315.0 / 128.0)))) / (h * h * h);
		inverseRadius = (693.0 / 256.0 + q2 * (-1155.0 / 256.0 + q2 * (693.0 / 128.0 + q2 * (-495.0 / 128.0 + q2 * (385.0 / 256.0 - q2 * 63.0 / 256.0))))) / h;
	}
	for (part_int sink = 0; sink < 2; sink++) {
		auto const self = findLeaf(built.id, sink);
		auto const source = findLeaf(built.id, 1 - sink);
		vector<tree_id> sources{source};
		double const sourceMass = particles_mass(1 - sink);
		double const expected = (particles_is_sph(sink) ? 1.0 : -1.0) * sourceMass * separation * inverseCube;
		force_vectors direct(1);
		cpu_gravity_pp(direct, true, self, sources, get_options().hsoft);
		requireClose("CPU particle-particle mass", direct.gx[0], expected);
		requireClose("CPU particle-particle potential", direct.phi[0], -sourceMass * inverseRadius);
		if (!softened) {
			requireClose("leaf monopole mass", tree_get_node(source)->mpos->multi[0], sourceMass);
			force_vectors far(1);
			cpu_gravity_pc(far, true, self, sources);
			requireClose("CPU particle-cell mass", far.gx[0], expected);
			for (int path = 0; path < 2; path++) {
				expansion<float> local;
				local = 0.0f;
				if (path == 0) {
					cpu_gravity_cp(local, sources, self, true);
				} else {
					cpu_gravity_cc(GRAVITY_DIRECT, local, sources, self, true);
				}
				array<float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = distance(particles_pos(dim, sink), tree_get_node(self)->pos[dim]);
				}
				auto const value = L2P(local, dx, true);
				requireClose(path == 0 ? "CPU cell-particle mass" : "CPU cell-cell mass", -SCALE_FACTOR2 * value(1, 0, 0), expected);
			}
		}
	}
	expansion<float> local;
	local = 0.0f;
	array<fixed32, NDIM> origin;
	for (int dim = 0; dim < NDIM; dim++) {
		origin[dim] = 0.0;
	}
	auto params = makeKickParams();
	resetKickState();
	auto const cpuResult = kick(params, local, origin, built.id, vector<tree_id>{built.id}, vector<tree_id>{}, nullptr);
	requireClose("CPU kick total potential", cpuResult.pot, -gravityScale * get_options().dm_mass * get_options().sph_mass * inverseRadius, 2e-4);
	verifyKick(inverseCube, inverseRadius, separation);
	verifySynchronizedDiagnostics(cpuResult);
#ifdef USE_CUDA
	if (useCuda) {
		resetKickState();
		kick_workitem item;
		item.L = local;
		item.pos = origin;
		item.self = built.id;
		item.dchecklist.push_back(built.id);
		auto const cudaResult = cuda_execute_kicks(params, particles_x[XDIM], particles_x[YDIM], particles_x[ZDIM], tree_data(), vector<kick_workitem>{item});
		requireClose("CUDA kick total potential", cudaResult.pot, cpuResult.pot, 2e-4);
		verifyKick(inverseCube, inverseRadius, separation);
		verifySynchronizedDiagnostics(cudaResult);
	}
#else
	if (useCuda) {
		throw std::runtime_error("CUDA regression requested from a non-CUDA build");
	}
#endif
	tree_destroy(true);
}
void checkpointPair() {
	FILE* file = std::tmpfile();
	if (file == nullptr) {
		throw std::runtime_error("Cannot create temporary checkpoint");
	}
	part_int const gas = sph_particles_dm_index(0);
	auto const position = particles_pos(XDIM, gas);
	particles_save(file);
	particles_pos(XDIM, gas) = 0.9;
	sph_particles_smooth_len(0) = 0.5;
	std::rewind(file);
	particles_load(file);
	std::fclose(file);
	verifyMappings();
	requireClose("checkpoint particle count", particles_size(), 2);
	requireClose("checkpoint gas count", sph_particles_size(), 1);
	requireClose("checkpoint position", particles_pos(XDIM, sph_particles_dm_index(0)).to_double(), position.to_double());
	requireClose("checkpoint hydro length", sph_particles_smooth_len(0), 0.02);
	requireClose("checkpoint composition", sph_particles_r1[0].frac[0], 0.75);
	requireClose("checkpoint chemistry stage", sph_particles_r5[0].dfrac[0], 0.125);
}
} // namespace detail

void initializeSphDomainRegression() {
	using namespace detail;
	initializePair(0.25);
	int const target = (hpx_rank() + 1) % hpx_size();
	for (part_int i = 0; i < 2; i++) {
		particles_pos(XDIM, i) = (target + (i == 0 ? 0.25 : 0.75)) / hpx_size();
	}
	sph_particles_r1[0].alpha = 10.0 + target;
	sph_particles_r5[0].dfrac[0] = 20.0 + target;
	sph_particles_r6[0].dvel[XDIM] = 30.0 + target;
	sph_particles_dv2[YDIM][0] = 40.0 + target;
	sph_particles_g[ZDIM][0] = 50.0 + target;
}
HPX_PLAIN_ACTION(initializeSphDomainRegression);
void validateSphDomainRegression() {
	using namespace detail;
	verifyMappings();
	requireClose("migrated gas count", sph_particles_size(), 1);
	auto const gas = sph_particles_dm_index(0);
	int const target = int(particles_pos(XDIM, gas).to_double() * hpx_size());
	requireClose("migration alpha", sph_particles_r1[0].alpha, 10.0 + target);
	requireClose("migration chemistry stage", sph_particles_r5[0].dfrac[0], 20.0 + target);
	requireClose("migration velocity stage", sph_particles_r6[0].dvel[XDIM], 30.0 + target);
	requireClose("migration predicted velocity", sph_particles_dv2[YDIM][0], 40.0 + target);
	requireClose("migration gravity", sph_particles_g[ZDIM][0], 50.0 + target);
	array<double, NDIM> position;
	for (part_int i = 0; i < particles_size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			position[dim] = particles_pos(dim, i).to_double();
		}
		if (!domains_find_my_box().contains(position)) {
			throw std::runtime_error("Migrated particle lies outside its domain");
		}
	}
}
HPX_PLAIN_ACTION(validateSphDomainRegression);

void validateSphRemoteTreeCache() {
	tree_id remote;
	remote.proc = (hpx_rank() + 1) % hpx_size();
	remote.index = 0;
	auto const* node = tree_get_node(remote);
	if (!(node->mpos->multi[0] > 0.0f && node->mpos->multi[0] <= float(hpx_size()))) {
		throw std::runtime_error("Remote tree cache lost the populated root multipole");
	}
}
HPX_PLAIN_ACTION(validateSphRemoteTreeCache);

void sphGravityTest() {
	using namespace detail;
	reset_gravity_counters();
	double closeCount, directCount;
	get_gravity_counters(closeCount, directCount);
	requireClose("empty close-interaction diagnostic", closeCount, 0.0);
	requireClose("empty direct-interaction diagnostic", directCount, 0.0);
	auto opts = get_options();
	bool const useCuda = opts.cuda;
	opts.sph = true;
	opts.vsoft = false;
	opts.gravity = true;
	opts.save_force = true;
	opts.stars = false;
	opts.do_groups = false;
	opts.do_tracers = false;
	opts.dm_mass = 0.75;
	opts.sph_mass = 0.25;
	opts.bucket_size = 1;
	opts.nparts = 2 * hpx_size();
	opts.hsoft = 0.01;
	set_options(opts);
	if (hpx_size() == 1) {
		runPairCase(useCuda, false);
		runPairCase(useCuda, true);
		checkpointPair();
		sph_particles_resize(2);
		auto const secondGas = sph_particles_dm_index(1);
		for (int dim = 0; dim < NDIM; dim++) {
			particles_pos(dim, secondGas) = 0.75;
			particles_vel(dim, secondGas) = 0.0;
		}
		particles_rung(secondGas) = 1;
		sphCheckpointTest();
	}
	vector<hpx::future<void>> pending;
	for (auto const& locality : hpx_localities()) {
		pending.push_back(hpx::async<initializeSphDomainRegression_action>(locality));
	}
	for (auto& future : pending) {
		future.get();
	}
	domains_rebound();
	domains_begin(0);
	domains_end();
	pending.clear();
	for (auto const& locality : hpx_localities()) {
		pending.push_back(hpx::async<validateSphDomainRegression_action>(locality));
	}
	for (auto& future : pending) {
		future.get();
	}
	if (hpx_size() > 1) {
		tree_create(tree_create_params(0, 0.5, get_options().hsoft));
		pending.clear();
		for (auto const& locality : hpx_localities()) {
			pending.push_back(hpx::async<validateSphRemoteTreeCache_action>(locality));
		}
		for (auto& future : pending) {
			future.get();
		}
		tree_destroy(true);
	}
	PRINT("SPH gravity regression passed (%s, %i localities)\n", useCuda ? "CPU and CUDA" : "CPU", hpx_size());
}
