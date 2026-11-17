/* CosmicTiger - precision, dimensionality, support, and persistence tests for MFM; BSL-1.0. */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <cosmictiger/core/file_handle.hpp>
#include <cosmictiger/mfm/particles.hpp>
#include <tests/mfm/prehydro_cases.hpp>

// Standalone host allocation and assertion services; production kernels are linked unchanged.
void* allocateManagedMemory(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    auto* result = std::malloc(bytes);
    if (result == nullptr) {
        throw std::bad_alloc();
    }
    return result;
}

void freeManagedMemory(void* pointer) noexcept {
    std::free(pointer);
}

Options const& getOptions() {
    static Options const options{};
    return options;
}

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

namespace detail {

template <typename Real, int dimensions>
struct MfmCase {
    using Value = Real;
    static constexpr int kDimensions = dimensions;
};

template <typename Case>
class MfmCpuTest : public testing::Test {};

using MfmCases = testing::Types<MfmCase<float, 1>, MfmCase<float, 2>, MfmCase<float, 3>,
    MfmCase<double, 1>, MfmCase<double, 2>, MfmCase<double, 3>>;
TYPED_TEST_SUITE(MfmCpuTest, MfmCases);

TYPED_TEST(MfmCpuTest, DensityAndInverseMatchIndependentReference) {
    using Real = typename TypeParam::Value;
    constexpr int kDimensions = TypeParam::kDimensions;
    MfmFixture<Real, kDimensions> fixture;
    auto const result = fixture.densityCpu();
    EXPECT_FALSE(result.needsAnotherPass);
    EXPECT_EQ(result.geometryFailures, 0);
    checkMfmReference(fixture);
    EXPECT_EQ(fixture.data_.sinkGeometryValid[0], 0);
    EXPECT_EQ(fixture.data_.sinkDensity[0], Real(0));
}

TYPED_TEST(MfmCpuTest, PeriodicSeamPreservesGeometry) {
    using Real = typename TypeParam::Value;
    constexpr int kDimensions = TypeParam::kDimensions;
    MfmFixture<Real, kDimensions> fixture;
    for (int i = 0; i < fixture.count_; ++i) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            auto& position = fixture.positions_[axis][fixture.source(i)];
            double const shifted = position.toDouble() + 0.48;
            position = shifted - std::floor(shifted);
        }
    }
    EXPECT_EQ(fixture.densityCpu().geometryFailures, 0);
    checkMfmReference(fixture);
}

TYPED_TEST(MfmCpuTest, ActiveNeighborRefreshesSemiactiveSinks) {
    using Real = typename TypeParam::Value;
    MfmFixture<Real, TypeParam::kDimensions> fixture;
    for (int i = 1; i < fixture.count_; ++i) {
        fixture.sourceRungs_[fixture.source(i)] = 0;
        fixture.data_.sinkParticleRungs[fixture.particle(i)] = 0;
    }
    EXPECT_EQ(fixture.densityCpu().geometryFailures, 0);
    for (int i = 0; i < fixture.count_; ++i) {
        EXPECT_TRUE(fixture.data_.sinkFlags[fixture.sink(i)].semiactive());
    }
    checkMfmReference(fixture);
}

TYPED_TEST(MfmCpuTest, NoActiveSourcesLeaveDensityUntouched) {
    using Real = typename TypeParam::Value;
    MfmFixture<Real, TypeParam::kDimensions> fixture;
    for (int i = 0; i < fixture.count_; ++i) {
        fixture.sourceRungs_[fixture.source(i)] = 0;
        fixture.data_.sinkParticleRungs[fixture.particle(i)] = 0;
        fixture.data_.sinkDensity[fixture.sink(i)] = Real(19);
    }
    EXPECT_EQ(fixture.densityCpu().geometryFailures, 0);
    for (int i = 0; i < fixture.count_; ++i) {
        EXPECT_FALSE(fixture.data_.sinkFlags[fixture.sink(i)].semiactive());
        EXPECT_EQ(fixture.data_.sinkDensity[fixture.sink(i)], Real(19));
    }
}

TYPED_TEST(MfmCpuTest, SelfDensitySurvivesSingularGeometry) {
    using Real = typename TypeParam::Value;
    constexpr int kDimensions = TypeParam::kDimensions;
    MfmFixture<Real, kDimensions> fixture(1);
    int const sink = fixture.sink(0);
    fixture.data_.sinkGeometryValid[sink] = 1;
    fixture.data_.sinkInverseMoment[sink](0, 0) = Real(17);
    auto const result = fixture.densityCpu();
    EXPECT_EQ(result.geometryFailures, 1);
    EXPECT_FALSE(result.needsAnotherPass);
    EXPECT_EQ(fixture.data_.sinkGeometryValid[sink], 0);
    EXPECT_EQ(fixture.data_.sinkInverseMoment[sink](0, 0), Real(0));
    requireMfmClose(fixture.data_.sinkDensity[sink],
        referenceMfmWeight<kDimensions>(0) /
            std::pow(double(fixture.data_.sinkSmoothingLength[sink]), kDimensions),
        2e-6);
}

TYPED_TEST(MfmCpuTest, BoundedSupportPublishesActualDensity) {
    using Real = typename TypeParam::Value;
    MfmFixture<Real, TypeParam::kDimensions> fixture;
    fixture.parameters_.minimumSmoothingLength = Real(0.125);
    fixture.parameters_.maximumSmoothingLength = Real(0.125);
    fixture.parameters_.initializeDensity = true;
    for (int i = 0; i < fixture.count_; ++i) {
        fixture.data_.sinkFlags[fixture.sink(i)].setConverged(false);
        fixture.data_.sinkGeometryValid[fixture.sink(i)] = 1;
    }
    EXPECT_FALSE(fixture.smoothingCpu().needsAnotherPass);
    std::vector<Real> initialDensity;
    for (int i = 0; i < fixture.count_; ++i) {
        int const sink = fixture.sink(i);
        EXPECT_EQ(fixture.data_.sinkSmoothingLength[sink], Real(0.125));
        EXPECT_TRUE(fixture.data_.sinkFlags[sink].converged());
        EXPECT_EQ(fixture.data_.sinkGeometryValid[sink], 0);
        EXPECT_GT(fixture.data_.sinkDensity[sink], Real(0));
        initialDensity.push_back(fixture.data_.sinkDensity[sink]);
    }
    EXPECT_EQ(fixture.densityCpu().geometryFailures, 0);
    for (int i = 0; i < fixture.count_; ++i) {
        requireMfmClose(fixture.data_.sinkDensity[fixture.sink(i)], initialDensity[i], 2e-6);
    }
    checkMfmReference(fixture);
}

TYPED_TEST(MfmCpuTest, FreeSupportSolveMeetsNeighborConstraint) {
    using Real = typename TypeParam::Value;
    constexpr int kDimensions = TypeParam::kDimensions;
    MfmFixture<Real, kDimensions> fixture;
    fixture.parameters_.minimumSmoothingLength = Real(0.001);
    fixture.parameters_.smoothingLengthTolerance = Real(1e-4);
    fixture.parameters_.initializeDensity = true;
    for (int i = 0; i < fixture.count_; ++i) {
        fixture.data_.sinkFlags[fixture.sink(i)].setConverged(false);
    }
    EXPECT_FALSE(fixture.smoothingCpu().needsAnotherPass);
    for (int i = 0; i < fixture.count_; ++i) {
        int const sink = fixture.sink(i);
        double const support = fixture.data_.sinkSmoothingLength[sink];
        EXPECT_TRUE(fixture.data_.sinkFlags[sink].converged());
        requireMfmClose(fixture.data_.sinkDensity[sink] * std::pow(support, kDimensions) /
                fixture.data_.gasMass,
            mfm::targetKernelSum<Real, kDimensions>(fixture.data_.targetNeighborCount), 1.1e-4);
    }
}

TYPED_TEST(MfmCpuTest, StagedSupportRetryInvalidatesOldGeometry) {
    using Real = typename TypeParam::Value;
    MfmFixture<Real, TypeParam::kDimensions> fixture;
    fixture.parameters_.minimumSmoothingLength = Real(0.3);
    fixture.parameters_.maximumSmoothingLength = Real(0.3);
    for (int node = 0; node < fixture.data_.numberOfTargetNodes; ++node) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            fixture.nodes_[node].outerBox.begin[axis] = 0.4;
            fixture.nodes_[node].outerBox.end[axis] = 0.6;
        }
    }
    for (int i = 0; i < fixture.count_; ++i) {
        int const sink = fixture.sink(i);
        fixture.data_.sinkFlags[sink].setConverged(false);
        fixture.data_.sinkGeometryValid[sink] = 1;
        fixture.data_.sinkInverseMoment[sink](0, 0) = Real(17);
        fixture.data_.sinkDensity[sink] = Real(23);
    }
    EXPECT_TRUE(fixture.smoothingCpu().needsAnotherPass);
    for (int i = 0; i < fixture.count_; ++i) {
        int const sink = fixture.sink(i);
        EXPECT_FALSE(fixture.data_.sinkFlags[sink].converged());
        EXPECT_EQ(fixture.data_.sinkGeometryValid[sink], 0);
        EXPECT_EQ(fixture.data_.sinkInverseMoment[sink](0, 0), Real(0));
        EXPECT_EQ(fixture.data_.sinkDensity[sink], Real(23));
    }
    for (int node = 0; node < fixture.data_.numberOfTargetNodes; ++node) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            fixture.nodes_[node].outerBox.begin[axis] = -1.0;
            fixture.nodes_[node].outerBox.end[axis] = 2.0;
        }
    }
    EXPECT_FALSE(fixture.smoothingCpu().needsAnotherPass);
    EXPECT_EQ(fixture.densityCpu().geometryFailures, 0);
    checkMfmReference(fixture);
}

TYPED_TEST(MfmCpuTest, InversionRejectsBadGeometryAndScalesCorrectly) {
    using Real = typename TypeParam::Value;
    constexpr int kDimensions = TypeParam::kDimensions;
    math::SymmetricMatrix<Real, kDimensions> moment;
    math::SymmetricMatrix<Real, kDimensions> inverse;
    for (int axis = 0; axis < kDimensions; ++axis) {
        moment(axis, axis) = Real(1);
    }
    EXPECT_TRUE((mfm::invertMoment(moment, Real(1), Real(0.25), Real(1e4), inverse)));
    EXPECT_EQ(inverse(0, 0), Real(16));
    EXPECT_TRUE((mfm::invertMoment(moment, Real(1), Real(0.125), Real(1e4), inverse)));
    EXPECT_EQ(inverse(0, 0), Real(64));
    moment(0, 0) = Real(0);
    EXPECT_FALSE((mfm::invertMoment(moment, Real(1), Real(0.25), Real(1e4), inverse)));
    EXPECT_EQ(inverse(0, 0), Real(0));
    moment(0, 0) = std::numeric_limits<Real>::quiet_NaN();
    EXPECT_FALSE((mfm::invertMoment(moment, Real(1), Real(0.25), Real(1e4), inverse)));
    if constexpr (kDimensions > 1) {
        moment(0, 0) = Real(1e-6);
        EXPECT_FALSE((mfm::invertMoment(moment, Real(1), Real(0.25), Real(1e4), inverse)));
    }
}

TYPED_TEST(MfmCpuTest, PackedStorageSurvivesGrowthSwapAndRestart) {
    using Real = typename TypeParam::Value;
    constexpr int kDimensions = TypeParam::kDimensions;
    using Storage = mfm::ParticleStorage<Real, kDimensions>;
    static_assert(sizeof(typename Storage::InverseMoment) ==
        sizeof(Real) * kDimensions * (kDimensions + 1) / 2);
    Storage storage;
    storage.resize(3, 5);
    storage.inverseMoments()[1](0, 0) = Real(17.25);
    storage.geometryValid()[1] = 1;
    storage.resize(6);
    storage.swap(0, 1);
    EXPECT_EQ(storage.inverseMoments()[0](0, 0), Real(17.25));
    EXPECT_EQ(storage.geometryValid()[0], 1);
    storage.resize(1);
    storage.resize(4);
    EXPECT_EQ(storage.geometryValid()[1], 0);
    EXPECT_EQ(storage.inverseMoments()[1](0, 0), Real(0));
    UniqueFile checkpoint(std::tmpfile());
    ASSERT_NE(checkpoint.get(), nullptr);
    storage.save(checkpoint.get());
    storage.inverseMoments()[0](0, 0) = Real(99);
    std::rewind(checkpoint.get());
    storage.load(checkpoint.get(), 4);
    EXPECT_EQ(storage.inverseMoments()[0](0, 0), Real(17.25));
    EXPECT_EQ(storage.geometryValid()[0], 1);
    std::rewind(checkpoint.get());
    EXPECT_THROW(storage.load(checkpoint.get(), 3), std::runtime_error);
    std::rewind(checkpoint.get());
    using OtherReal = std::conditional_t<std::is_same_v<Real, float>, double, float>;
    mfm::ParticleStorage<OtherReal, kDimensions> otherPrecision;
    EXPECT_THROW(otherPrecision.load(checkpoint.get(), 4), std::runtime_error);
    storage.geometryValid()[0] = 2;
    EXPECT_THROW(storage.save(checkpoint.get()), std::runtime_error);
    storage.geometryValid()[0] = 1;
    storage.inverseMoments()[0](0, 0) = std::numeric_limits<Real>::infinity();
    EXPECT_THROW(storage.save(checkpoint.get()), std::runtime_error);
}

}    // namespace detail
