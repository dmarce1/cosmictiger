/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#include <cosmictiger/gravity/dehnen/periodic.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/math/vector3.hpp>

FlopCounts dehnenBackground(int order, KickParameters const& params, ExpansionBuffer& expansion,
    Vector3<Position>& position, TreeId self, Vector<TreeId> const& checks) {
    return fmmDispatch(order, [&](auto tag) {
        return dehnenBackgroundImpl<decltype(tag)::value>(
            params, expansion, position, self, checks);
    });
}
FlopCounts dehnenPeriodic(int order, KickParameters const& params, ExpansionBuffer& expansion,
    TreeId self, Vector<TreeId>& checks, KickInteractionCounts& counts) {
    return fmmDispatch(order, [&](auto tag) {
        return dehnenPeriodicImpl<decltype(tag)::value>(params, expansion, self, checks, counts);
    });
}
