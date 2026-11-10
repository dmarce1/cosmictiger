/* CosmicTiger - transient owned-source tree moments; BSL-1.0. */
#pragma once

#include <variant>

#include <cosmictiger/gravity/fmm_types.hpp>

// Double arithmetic and transport during the owned upward build only. Persistent
// tree/ghost records and all walk inputs continue to use MultipoleBuffer.
struct SourceMultipoleBuffer {
    Array<double, kFmmMaxMultipoleCoefficients> coefficients{};
    /// Access a coefficient in the double-precision source buffer.
    double& operator[](int i) {
        return coefficients[i];
    }
    /// Read a coefficient in the double-precision source buffer.
    double const& operator[](int i) const {
        return coefficients[i];
    }
    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        fmmDispatch([&](auto tag) {
            for (int i = 0; i < FmmOrder<decltype(tag)::value>::kMultipoleCoefficients_; ++i) {
                archive& coefficients[i];
            }
        });
    }
};

// The active alternative determines serialization precision. Ghost builds and
// the compatibility root cache retain float payloads; an owned build retains
// double through its futures, without also retaining a float upward copy.
class TreeBuildMoments {
    std::variant<MultipoleBuffer, SourceMultipoleBuffer> moments_{MultipoleBuffer{}};

public:
    /// Test whether source moments currently use the double-precision alternative.
    bool isDouble() const {
        return std::holds_alternative<SourceMultipoleBuffer>(moments_);
    }
    /// Replace the current variant with newly initialized double-precision source
    /// storage.
    void useDouble() {
        moments_.emplace<SourceMultipoleBuffer>();
    }
    /// Access the single-precision alternative; that alternative must currently be
    /// active.
    MultipoleBuffer& single() {
        return std::get<MultipoleBuffer>(moments_);
    }
    /// Access the single-precision alternative; that alternative must currently be
    /// active.
    MultipoleBuffer const& single() const {
        return std::get<MultipoleBuffer>(moments_);
    }
    /// Access the double-precision alternative; that alternative must currently be
    /// active.
    SourceMultipoleBuffer& precise() {
        return std::get<SourceMultipoleBuffer>(moments_);
    }
    /// Access the double-precision alternative; that alternative must currently be
    /// active.
    SourceMultipoleBuffer const& precise() const {
        return std::get<SourceMultipoleBuffer>(moments_);
    }
    /// Read the indexed source coefficient, promoting single-precision storage when
    /// needed.
    double operator[](int i) const {
        return isDouble() ? precise()[i] : single()[i];
    }
    /// Return a single-precision copy of the active source coefficients at the
    /// configured FMM order.
    MultipoleBuffer narrow() const {
        if (!isDouble()) {
            return single();
        }
        MultipoleBuffer result{};
        fmmDispatch([&](auto tag) {
            for (int i = 0; i < FmmOrder<decltype(tag)::value>::kMultipoleCoefficients_; ++i) {
                result[i] = precise()[i];
            }
        });
        return result;
    }
    /// Replace retained double-precision moments with their single-precision copy.
    void discardDouble() {
        moments_ = narrow();
    }
    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        bool wide = isDouble();
        archive & wide;
        if (wide) {
            if (!isDouble()) {
                useDouble();
            }
            archive& precise();
        } else {
            if (isDouble()) {
                moments_.emplace<MultipoleBuffer>();
            }
            archive& single();
        }
    }
};
