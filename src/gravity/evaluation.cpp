/* CosmicTiger - Nitadori PMMM evaluation; BSL-1.0. */

#include <cosmictiger/gravity/fmm/tree_local.hpp>
#include <cosmictiger/gravity/mesh_size.hpp>
#include <cosmictiger/math/vector3.hpp>
#include "actions.hpp"

namespace pmmm::detail {

/// Scale and validate the combined particle field, then apply its kick or store diagnostic
/// field values.
void consumeField(ParticleInteger index, std::array<double, 4> const& field, KickReturn& result,
    FlopCount& doubleFlops, FlopCounts& kickFlops) {
    Vector3<float> acceleration;
    double const gm = getOptions().gravity.gm;
    for (int component = 0; component < 4; ++component) {
        if (!std::isfinite(field[component]) ||
            std::abs(gm * field[component]) > std::numeric_limits<float>::max()) {
            throw std::runtime_error("PMMM combined particle field is not representable");
        }
        if (component != 0) {
            acceleration[component - 1] = gm * field[component];
        }
    }
    float const potential = state.request.doPotential ? gm * field[0] : 0.0f;
    // Each retained component is scaled in double and narrowed to float.
    doubleFlops += 2 * (kDimensions + (state.request.doPotential ? 1 : 0));
    if (state.request.applyKick) {
        result += particleMeshApplyField(
            index, acceleration, potential, state.request.parameters, nullptr, &kickFlops);
    } else {
        state.acceleration[index] = acceleration;
        if (!state.potential.empty()) {
            state.potential[index] = potential;
        }
    }
}

// Boundary far solve of the PMMM outer interval (docs/pmmm.md).
//
//   L_new,i = gm ∇ evaluated from the local expansion of particle i's cell,
//   H_acc   = min_i η sqrt(a h / |L_new,i|),   h = 1/N,
//   W_far   = Σ_i ½ m_i gm φ_far,i.
//
// Symbol directory: L_new → far acceleration stored in particlesLongAcceleration;
// η → gravity.timestepEta; a → the scale factor; h → cell width;
// N → meshSize, the number of cells per side.
// Code names: L_new → acceleration, H_acc → result.minimumTimestep,
// W_far → result.pot, N → n.
void evaluateFar() {
    int const n = cache.meshSize;
    double const h = 1.0 / n;
    double const localTrace = (4 * std::numbers::pi / 3) * h * h * state.background[0];
    double const gm = getOptions().gravity.gm;
    double const scale = state.request.scale;
    double const eta = state.request.eta;
    if (!particlesHasLongAcceleration(getOptions())) {
        throw std::runtime_error("The PMMM boundary far solve needs the long-acceleration field");
    }
    addCpuFlops(5, FlopPrecision::Double);
    parallelFor(state.cells.size(), "gravity.pmmm.evaluateFar", [&](size_t cellIndex) {
        auto const& cell = state.cells[cellIndex];
        auto const center = coordinates(cell.key);
        auto const coefficients = localExpansion(cell.key);
        KickReturn result;
        FlopCount flops = 0;
        fmmDispatch([&](auto tag) {
            constexpr int kOrder = decltype(tag)::value;
            namespace cartesian = cosmictiger::gravity::cartesian;
            namespace local = cosmictiger::gravity::tree_local;
            auto const root = state.sourceCells[cellIndex]->ready.get();
            auto const rootPosition = treeGetNode(root.id)->pos;
            Vector3<Position> meshCenter;
            Vector3<double> displacement;
            for (int axis = 0; axis < kDimensions; ++axis) {
                double const coordinate = (center[axis] + .5) * h;
                meshCenter[axis] = coordinate;
                displacement[axis] = double(kScaleFactor) *
                    (unwrappedDoubleDistance(rootPosition[axis], meshCenter[axis]) +
                        (meshCenter[axis].toDouble() - coordinate));
            }
            // Match the mesh-center construction and native fixed-position
            // displacement accounting used by the ordinary far descent.
            flops += kDimensions * (kPositionBits == 64 ? 13 : 8) + 12 + 6;
            cartesian::Expansion<double, kOrder> meshLocal{}, rootLocal{};
            flops += local::fromMesh(meshLocal, coefficients.data(), localTrace, cache.meshSize);
            double const curvature = local::backgroundCurvature(localTrace, cache.meshSize);
            flops += local::translate(rootLocal, meshLocal, displacement, curvature, true);
            auto descend = [&](auto const& self, TreeId id,
                               cartesian::Expansion<double, kOrder> const& expansion) -> void {
                auto const* node = treeGetNode(id);
                if (!node->leaf) {
                    for (auto const child : node->children) {
                        cartesian::Expansion<double, kOrder> translated{};
                        flops +=
                            local::translate(translated, expansion,
                                local::displacement<double>(treeGetNode(child)->pos, node->pos),
                                curvature, true) +
                            9;
                        self(self, child, translated);
                    }
                    return;
                }
                for (ParticleInteger i = node->particleRange.first; i < node->particleRange.second;
                    ++i) {
                    Vector3<Position> position;
                    for (int axis = 0; axis < kDimensions; ++axis) {
                        position[axis] = particlesPos(axis, i);
                    }
                    cosmictiger::gravity::Field<double> field{};
                    flops +=
                        local::evaluate(field, expansion,
                            local::displacement<double>(position, node->pos), curvature, true) +
                        9;
                    double accelerationSquared = 0;
                    for (int component = 0; component < 4; ++component) {
                        if (!std::isfinite(field[component]) ||
                            std::abs(gm * field[component]) > std::numeric_limits<float>::max()) {
                            throw std::runtime_error(
                                "PMMM far particle field is not representable");
                        }
                    }
                    for (int axis = 0; axis < kDimensions; ++axis) {
                        double const acceleration = gm * field[axis + 1];
                        particlesLongAcceleration(axis, i) = float(acceleration);
                        accelerationSquared += acceleration * acceleration;
                    }
                    double const timestep = eta *
                        std::sqrt(scale * h / std::max(std::sqrt(accelerationSquared), 1.0e-30));
                    if (!(timestep > 0) || !std::isfinite(timestep)) {
                        throw std::runtime_error(
                            "PMMM outer interval bound is not finite and positive");
                    }
                    result.minimumTimestep = std::min(result.minimumTimestep, timestep);
                    result.pot += .5 * double(particlesMass(i)) * gm * field[0];
                    flops += 3 * kDimensions + 8;
                }
            };
            descend(descend, root.id, rootLocal);
        });
        addCpuFlops(flops, FlopPrecision::Double);
        std::lock_guard<MutexType> lock(state.queueMutex);
        state.kickResult += result;
        state.diagnostics.evaluatedParticles += cell.end - cell.begin;
    });
    state.boundaryTreeReady = true;
    state.boundaryMeshSize = cache.meshSize;
    state.boundaryOrder = cache.order;
}

}    // namespace pmmm::detail

HPX_REGISTER_ACTION(PmmmEvaluateFarAction, PmmmEvaluateFarAction);
