/* CosmicTiger - geometric FMM source-forest exchange; BSL-1.0. */

// Source-tree descriptors, export payloads and APIs for exchanging imported source forests.

#pragma once

#include <array>
#include <cstdint>
#include <exception>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/geometric_tree_geometry.hpp>
#include <cosmictiger/gravity/tree.hpp>

inline constexpr int kGeometricSourceForestVersion = 2;

struct SourceRootDescriptor {
    TreeId root{};    // Owner locality and arena index of its independent tree root.
    SourceMultipoleBuffer multi{};    // Root multipole coefficients for the selected sources.
    Range<double> cover{};      // Outward bounds of occupied Ewald-depth cells; unit box if empty.
    uint64_t generation = 0;    // Shared build epoch used to reject stale source exports.
    int geometryVersion = kGeometricSourceForestVersion;    // Geometry/exchange contract version.
    ParticleInteger sourceCount = 0;    // Owned particles included in this source tree.
    size_t nodeCount = 0;               // Total nodes in the owned tree, including empty siblings.
    size_t leafCount = 0;               // Terminal nodes in the owned tree, including empty leaves.
    int ewaldDepth = 0;                 // Minimum nonempty-leaf depth required by periodic FMM.
    float theta = 0;                    // Opening angle used to build and certify this generation.
    float smoothingLength = 0;          // Gravity smoothing support in unit-box coordinates.
    bool gpu = false;                   // Whether this locality will traverse sources on the GPU.

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & root;
        archive & multi;
        archive & cover;
        archive & generation;
        archive & geometryVersion;
        archive & sourceCount;
        archive & nodeCount;
        archive & leafCount;
        archive & ewaldDepth;
        archive & theta;
        archive & smoothingLength;
        archive & gpu;
    }
};

struct SourceExportNode {
    TreeId id{};                 // Original owner locality and node index, preserved across export.
    GeometricTreeCell cell{};    // Exact dyadic cell identity, also for collapsed subtrees.
    TreeNode node{};    // Destination-specific node metadata and serialized multipole contents.

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & id;
        archive & cell;
        archive & node;
    }
};

// HPX vectors otherwise transfer this trivially copyable wrapper as bytes,
// bypassing TreeNode::load and leaving its multipole pointer on the sender.
namespace hpx::traits {
template <>
struct is_bitwise_serializable<SourceExportNode> : std::false_type {};
template <>
struct is_not_bitwise_serializable<SourceExportNode> : std::true_type {};
}    // namespace hpx::traits

struct SourceExportParticle {
    std::array<Position, kDimensions> position{};    // Fixed-point coordinates in the unit box.

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        for (auto& coordinate : position) {
            archive & coordinate;
        }
    }
};
static_assert(sizeof(SourceExportParticle) == kDimensions * sizeof(Position));

struct SourceExportView {
    int owner = -1;             // Locality that owns the exported sources.
    int destination = -1;       // Locality for which this export was certified.
    uint64_t generation = 0;    // Frozen source-tree epoch shared by all participating owners.
    int geometryVersion = kGeometricSourceForestVersion;    // Required geometry/exchange version.
    int positionBits = kPositionBits;             // Fixed-point precision required of the receiver.
    int maximumDepth = kGeometricTreeMaxDepth;    // Supported depth checked for compatibility.
    int order = kFmmOrder;            // Multipole expansion order checked for compatibility.
    int destinationEwaldDepth = 0;    // Receiver's minimum nonempty-leaf depth for certification.
    float theta = 0;                // Receiver's opening angle used to certify collapsed subtrees.
    float smoothingLength = 0;      // Receiver's gravity smoothing support in unit-box coordinates.
    bool destinationGpu = false;    // Include particle data for coarse terminals on GPU walks.
    Range<double> destinationCover{};    // Receiver's occupied Ewald-depth cell bounds.
    Vector<SourceExportNode> nodes;      // Exported tree in parent-before-children traversal order.
    // Payload order follows particle-bearing leaves/terminals in nodes. Their
    // existing particleRange supplies each source index without per-particle IDs.
    Vector<SourceExportParticle> particles;    // Position payload for particle-bearing terminals.
    char uniformParticleType = kDarkMatterType;    // Shared mass class when particleTypes is empty.
    Vector<char> particleTypes;    // Per-particle mass classes for mixed payloads; stars use gas.

    // Imported sources need only the gravity mass class. Stars share the gas
    // mass; this normalization never changes owned particle species identities.
    void appendParticle(SourceExportParticle particle, char type) {
        type = type == kDarkMatterType ? kDarkMatterType : kSphType;
        if (particles.empty()) {
            uniformParticleType = type;
        } else if (particleTypes.empty() && type != uniformParticleType) {
            particleTypes.assign(particles.size(), uniformParticleType);
        }
        if (!particleTypes.empty()) {
            particleTypes.push_back(type);
        }
        particles.push_back(particle);
    }

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & owner;
        archive & destination;
        archive & generation;
        archive & geometryVersion;
        archive & positionBits;
        archive & maximumDepth;
        archive & order;
        archive & destinationEwaldDepth;
        archive & theta;
        archive & smoothingLength;
        archive & destinationGpu;
        archive & destinationCover;
        archive & nodes;
        archive & particles;
        archive & uniformParticleType;
        archive & particleTypes;
    }
};

void treeSourceForestReset();
void treeSourceForestRegisterOwner(SourceRootDescriptor const& owner);
void treeSourceForestValidate(Vector<SourceRootDescriptor> const& roots, float theta,
    float smoothingLength, bool gpu, bool requireImports = true);
void treeSourceForestUseImports(bool enabled);
bool treeSourceForestImportsEnabled();
TreeNode const* treeSourceForestImportedNode(TreeId id);
void treeSourceForestMaterialize(TreeId id);
bool treeSourceForestReadParticleRange(TreeNode const& source, Position* x, Position* y,
    Position* z, char* types, float* masses, ParticleInteger offset);
void treeSourceForestCountFallback();
uint64_t treeSourceForestFallbackCount();
void treeExchangeSourceForest(Vector<SourceRootDescriptor> const& roots);

#ifndef __CUDACC__
namespace detail {
// Allocate every slot before dispatch. If constructing a later payload or
// launching its action throws, earlier actions must finish before callers can
// retire the generation or repartition its particles. Stored task errors
// remain in the returned futures for callers to consume with get().
template <class Result, class Launch>
Vector<hpx::future<Result>> geometricLaunchAndDrain(size_t count, Launch&& launch) {
    Vector<hpx::future<Result>> futures(count);
    std::exception_ptr failure;
    try {
        for (size_t index = 0; index < count; ++index) {
            futures[index] = launch(index);
        }
        hpx::wait_all_nothrow(futures.begin(), futures.end());
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        // HPX waits can migrate the lightweight thread. End the native catch
        // block before yielding, then rethrow the explicitly retained exception.
        for (auto& future : futures) {
            if (future.valid()) {
                future.wait();
            }
        }
        std::rethrow_exception(failure);
    }
    return futures;
}
}    // namespace detail
#endif
