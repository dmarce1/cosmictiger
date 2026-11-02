/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#pragma once

#include <string>

// Manual phase profiler. COSMICTIGER_PROFILE (a CMake option, on by default)
// compiles the instrumentation; without it every call is an empty inline.
//
// profileBegin/profileEnd delimit phases on the coordinator's
// logical control flow (driver, scenario driver, initialization). Phases nest:
// entering a child pauses its parent, so each name accumulates exclusive wall
// time and the rows of the report sum to the instrumented total. Do not
// instrument kernels, per-particle loops, or work that runs concurrently on
// other HPX tasks; the single phase stack is not meant for parallel scopes.
namespace profile {

#if COSMICTIGER_PROFILE
void profileBegin(char const* name);
void profileEnd();
// Table of exclusive wall time per phase, sorted by time; empty with no phases.
std::string profileReport();
// Writes profileReport() to filename (overwrite). Safe while phases are open.
void profileOutput(std::string const& filename);
#else
inline void profileBegin(char const*) noexcept {}
inline void profileEnd() noexcept {}
inline std::string profileReport() {
    return {};
}
inline void profileOutput(std::string const&) noexcept {}
#endif

// Balance a coordinator phase on every exit path, including exceptions. Use
// enabled only where the caller can identify a serialized coordinator entry;
// being on rank zero alone does not make a parallel task safe to instrument.
class ScopedPhase {
public:
    explicit ScopedPhase(char const* name, bool enabled = true)
      : enabled_(enabled) {
        if (enabled_) {
            profileBegin(name);
        }
    }

    ~ScopedPhase() {
        if (enabled_) {
            profileEnd();
        }
    }

    ScopedPhase(ScopedPhase const&) = delete;
    ScopedPhase& operator=(ScopedPhase const&) = delete;
    ScopedPhase(ScopedPhase&&) = delete;
    ScopedPhase& operator=(ScopedPhase&&) = delete;

private:
    bool enabled_;
};

}    // namespace profile
