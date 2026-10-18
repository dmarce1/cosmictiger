#pragma once

#include <nvtx3/nvToolsExt.h>

namespace detail {

// Global NVTX ranges remain valid when an HPX task migrates between OS threads.
class HostProfileRange {
    nvtxRangeId_t range_;
    bool active_;

public:
    explicit HostProfileRange(char const* label, bool enabled = true)
        : range_(enabled ? nvtxRangeStartA(label) : 0), active_(enabled) {}

    HostProfileRange(char const* label, bool enabled, unsigned long long workCount)
        : range_(0), active_(enabled) {
        if (enabled) {
            nvtxEventAttributes_t event{};
            event.version = NVTX_VERSION;
            event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            event.messageType = NVTX_MESSAGE_TYPE_ASCII;
            event.message.ascii = label;
            event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            event.payload.ullValue = workCount;
            range_ = nvtxRangeStartEx(&event);
        }
    }

    HostProfileRange(HostProfileRange const&) = delete;
    HostProfileRange& operator=(HostProfileRange const&) = delete;

    ~HostProfileRange() {
        finish();
    }

    void finish() {
        if (active_) {
            nvtxRangeEnd(range_);
            active_ = false;
        }
    }
};

}    // namespace detail
