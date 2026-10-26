/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#include <cosmictiger/gravity/dehnen/periodic.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>

FlopCount dehnenBackground(int order, KickParameters const& params, ExpansionBuffer& expansion,
    Array<Position, kDimensions>& position, TreeId self, Vector<TreeId> const& checks) {
    return fmmDispatch(order, [&](auto tag) {
        return dehnenBackgroundImpl<decltype(tag)::value>(
            params, expansion, position, self, checks);
    });
}
FlopCount dehnenPeriodic(int order, KickParameters const& params, ExpansionBuffer& expansion,
    TreeId self, Vector<TreeId>& checks, KickInteractionCounts& counts) {
    return fmmDispatch(order, [&](auto tag) {
        return dehnenPeriodicImpl<decltype(tag)::value>(params, expansion, self, checks, counts);
    });
}
