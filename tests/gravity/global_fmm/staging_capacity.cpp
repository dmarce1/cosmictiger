/* CosmicTiger - GPU source staging arena regression; BSL-1.0. */

// Test GPU staging capacity and owned-arena preservation without launching CUDA.

#include <exception>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/tree_build.hpp>

namespace detail {

void requireStagingCapacity(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}    // namespace detail

// Exercise the host staging allocator without launching CUDA. The owned arena is
// filled through its last slot, and the final allocator block extends beyond it.
void geometricStagingCapacityTest() {
    auto const savedOptions = getOptions();
    auto options = savedOptions;
    options.gravity.cuda = false;
    options.compute.treeAllocLineSize = kMinTrees + 1;
    setOptions(options);
    std::exception_ptr failure;
    try {
        treeAllocateNodesLocal(kMinTrees);
        TreeNode node{};
        node.children[kLeftChild] = node.children[kRightChild] = TreeId{-1, -1};
        node.particleRange = {0, 1};
        node.processRange = {hpxRank(), hpxRank() + 1};
        node.pos = {.25, .5, .75};
        node.leaf = true;
        MultipoleBuffer multipole;
        multipole.zero();
        multipole[0] = 17.0f;
        int const first = treeAllocateNode();
        treeStoreNode(first, node, multipole);
        int last = first;
        for (size_t index = 1; index < size_t(treeNodesSize()); ++index) {
            last = treeAllocateNode();
        }
        treeStoreNode(last, node, multipole);
        TreeNode* const ownedData = treeData();
        auto const ownedCapacity = treeNodesSize();
        auto const reservedEnd = treeNodesNextIndex();
        auto const* const ownedMultipole = ownedData[last].mpos;
        detail::requireStagingCapacity(reservedEnd > ownedCapacity,
            "Staging regression did not include allocator block padding");
        constexpr int kArenaReusePasses = 2;
        for (int repeat = 0; repeat < kArenaReusePasses; ++repeat) {
            TreeStagingArena staged(2);
            detail::requireStagingCapacity(staged.remoteBegin() == size_t(ownedCapacity),
                "Staging did not preserve the full occupied local index range");
            detail::requireStagingCapacity(staged.data() != ownedData &&
                    staged.data()[first].mpos == ownedData[first].mpos &&
                    staged.data()[last].mpos == ownedMultipole,
                "Staging did not retain the owned positioned multipoles");
            TreeNode remote = ownedData[last];
            remote.processRange = {hpxRank() + 1, hpxRank() + 2};
            remote.particleRange = {3, 5};
            size_t const firstRemote = staged.appendRemote(remote);
            size_t const secondRemote = staged.appendRemote(remote);
            detail::requireStagingCapacity(firstRemote == size_t(ownedCapacity) &&
                    secondRemote == firstRemote + 1 && staged.size() == secondRemote + 1,
                "Remote staging could not append beyond a full owned arena");
            staged.data()[last].particleRange = {7, 8};
            staged.data()[firstRemote].particleRange = {9, 11};
            detail::requireStagingCapacity(treeData() == ownedData &&
                    treeNodesSize() == ownedCapacity && treeNodesNextIndex() == reservedEnd &&
                    ownedData[last].particleRange == node.particleRange &&
                    ownedData[last].mpos == ownedMultipole,
                "Staging moved or mutated the live owned tree");
            MultipoleBuffer copied;
            treeReadMultipole(staged.data()[last].mpos, copied);
            detail::requireStagingCapacity(
                copied[0] == multipole[0], "Staging invalidated a live owned multipole");
            bool rejected = false;
            try {
                staged.appendRemote(remote);
            } catch (std::length_error const&) {
                rejected = true;
            }
            detail::requireStagingCapacity(rejected && staged.size() == secondRemote + 1,
                "Staging failed to reject an append beyond its reserved capacity");
        }
        TreeStagingArena localOnly(0);
        detail::requireStagingCapacity(localOnly.data() == ownedData,
            "Local-only staging unnecessarily replaced the owned array");
        bool rejected = false;
        try {
            TreeStagingArena oversized(std::numeric_limits<size_t>::max());
        } catch (std::overflow_error const&) {
            rejected = true;
        }
        detail::requireStagingCapacity(
            rejected, "Staging accepted an overflowing remote node count");
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        setOptions(savedOptions);
        treeAllocateNodesLocal(0);
        std::rethrow_exception(failure);
    }
    setOptions(savedOptions);
    treeAllocateNodesLocal(0);
}
