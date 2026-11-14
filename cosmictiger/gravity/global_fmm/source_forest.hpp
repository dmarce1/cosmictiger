/* CosmicTiger - geometric FMM source-forest exchange; BSL-1.0. */
#pragma once

#include <array>
#include <cstdint>
#include <exception>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/geometric_tree_geometry.hpp>
#include <cosmictiger/gravity/tree.hpp>

inline constexpr int kGeometricSourceForestVersion = 2;

struct SourceRootDescriptor {
    TreeId root{};
    SourceMultipoleBuffer multi{};
    Range<double> cover{};
    uint64_t generation = 0;
    int geometryVersion = kGeometricSourceForestVersion;
    ParticleInteger sourceCount = 0;
    size_t nodeCount = 0;
    size_t leafCount = 0;
    int ewaldDepth = 0;
    float theta = 0;
    float smoothingLength = 0;
    bool gpu = false;

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
    TreeId id{};
    GeometricTreeCell cell{};
    TreeNode node{};

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
    std::array<Position, 3> position{};

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        for (auto& coordinate : position) {
            archive & coordinate;
        }
    }
};
static_assert(sizeof(SourceExportParticle) == 3 * sizeof(Position));

struct SourceExportView {
    int owner = -1;
    int destination = -1;
    uint64_t generation = 0;
    int geometryVersion = kGeometricSourceForestVersion;
    int positionBits = kPositionBits;
    int maximumDepth = kGeometricTreeMaxDepth;
    int order = kFmmOrder;
    int destinationEwaldDepth = 0;
    float theta = 0;
    float smoothingLength = 0;
    bool destinationGpu = false;
    Range<double> destinationCover{};
    Vector<SourceExportNode> nodes;
    // Payload order follows particle-bearing leaves/terminals in nodes. Their
    // existing particleRange supplies each source index without per-particle IDs.
    Vector<SourceExportParticle> particles;
    char uniformParticleType = kDarkMatterType;
    Vector<char> particleTypes;

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
