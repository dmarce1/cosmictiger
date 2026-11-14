/* CosmicTiger - owned-sink source-forest traversal; BSL-1.0. */

#include <exception>
#include <stdexcept>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/execution.hpp>
#include <cosmictiger/gravity/global_fmm/measurement.hpp>
#include <cosmictiger/gravity/global_fmm/source_forest.hpp>

void validateSourceForestLocal(
    KickParameters params, Vector<SourceRootDescriptor> roots, bool reference) {
    treeSourceForestValidate(
        roots, params.theta, params.gravitySmoothingLength, params.gpu, !reference);
}
HPX_PLAIN_ACTION(validateSourceForestLocal, ValidateSourceForestLocalAction);

KickReturn kickSourceForestLocal(
    KickParameters params, Vector<SourceRootDescriptor> roots, bool reference) {
    validateSourceForestLocal(params, roots, reference);
    FmmMeasurementScope walkTimer(FmmMeasurementPhase::Traversal);
    treeSourceForestUseImports(!reference);
    try {
        Vector<TreeId> sources;
        for (auto const& root : roots) {
            if (root.sourceCount != 0) {
                sources.push_back(root.root);
            }
        }
        ExpansionBuffer local;
        local.zero();
        Vector3<Position> center{};
        KickReturn result =
            kick(params, local, center, roots[hpxRank()].root, {}, sources, nullptr);
        treeSourceForestUseImports(true);
        return result;
    } catch (...) {
        treeSourceForestUseImports(true);
        throw;
    }
}
HPX_PLAIN_ACTION(kickSourceForestLocal, KickSourceForestLocalAction);

KickReturn kickSourceForest(KickParameters params, bool reference) {
    if (hpxRank() != 0) {
        throw std::logic_error("Source-forest coordination must start on locality zero");
    }
    auto const roots = treeGeometricRoots();
    if (roots.size() != size_t(hpxSize())) {
        throw std::runtime_error("Geometric source forest is not built");
    }
    // Every locality must still own the frozen generation, including in the
    // complete-source reference mode. Finish this read-only phase everywhere
    // before any locality can modify velocities, forces, or deferred kicks.
    auto validation = detail::geometricLaunchAndDrain<void>(roots.size(), [&](size_t rank) {
        return hpx::async<ValidateSourceForestLocalAction>(
            hpxLocalities()[rank], params, roots, reference);
    });
    for (auto& future : validation) {
        future.get();
    }
    KickReturn result;
    std::exception_ptr failure;
    try {
        auto futures = detail::geometricLaunchAndDrain<KickReturn>(roots.size(), [&](size_t rank) {
            return hpx::async<KickSourceForestLocalAction>(
                hpxLocalities()[rank], params, roots, reference);
        });
        for (auto& future : futures) {
            result += future.get();
        }
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        treeRetireGeometric();
        std::rethrow_exception(failure);
    }
    return result;
}
