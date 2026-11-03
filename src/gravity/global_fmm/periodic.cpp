/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/global_fmm/periodic.hpp>
#include <cosmictiger/math/vector3.hpp>

FlopCounts globalFmmBackground(int order, KickParameters const& params, ExpansionBuffer& expansion,
    Vector3<Position>& position, TreeId self, Vector<TreeId> const& checks) {
    return fmmDispatch(order, [&](auto tag) {
        return globalFmmBackgroundImpl<decltype(tag)::value>(
            params, expansion, position, self, checks);
    });
}
FlopCounts globalFmmPeriodic(int order, KickParameters const& params, ExpansionBuffer& expansion,
    TreeId self, Vector<TreeId>& checks, KickInteractionCounts& counts) {
    return fmmDispatch(order, [&](auto tag) {
        return globalFmmPeriodicImpl<decltype(tag)::value>(params, expansion, self, checks, counts);
    });
}
