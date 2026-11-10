/* CosmicTiger - transient owned-source tree moments; BSL-1.0. */
#pragma once

#include <variant>

#include <cosmictiger/gravity/fmm_types.hpp>

// Double arithmetic and transport during the owned upward build only. Persistent
// tree/ghost records and all walk inputs continue to use MultipoleBuffer.
struct SourceMultipoleBuffer {
    Array<double, kFmmMaxMultipoleCoefficients> coefficients{};
    double& operator[](int i) {
        return coefficients[i];
    }
    double const& operator[](int i) const {
        return coefficients[i];
    }
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
    bool isDouble() const {
        return std::holds_alternative<SourceMultipoleBuffer>(moments_);
    }
    void useDouble() {
        moments_.emplace<SourceMultipoleBuffer>();
    }
    MultipoleBuffer& single() {
        return std::get<MultipoleBuffer>(moments_);
    }
    MultipoleBuffer const& single() const {
        return std::get<MultipoleBuffer>(moments_);
    }
    SourceMultipoleBuffer& precise() {
        return std::get<SourceMultipoleBuffer>(moments_);
    }
    SourceMultipoleBuffer const& precise() const {
        return std::get<SourceMultipoleBuffer>(moments_);
    }
    double operator[](int i) const {
        return isDouble() ? precise()[i] : single()[i];
    }
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
    void discardDouble() {
        moments_ = narrow();
    }
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
