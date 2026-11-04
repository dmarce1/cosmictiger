"""Independent checks of chronological midpoint event logs and checkpoint state."""

import math
import re
import sys

from checkpoint_format import require


MIDPOINT = re.compile(r"SPH midpoint: time=(\S+) rung=(-?\d+) dt=(\S+)")


def check_midpoint_state(state):
    require(70000 <= state.gas_version < 71000,
            "midpoint checkpoint retained obsolete per-particle force or entropy-rate arrays")
    require(len(state.proposed_rungs) == state.gas_count,
            "checkpoint omitted the persistent hydro timestep bounds")
    require(all(0 <= rung < 16 for rung in state.proposed_rungs),
            "checkpoint has an invalid hydro timestep bound")
    require(all(value > 0 and math.isfinite(value) for value in state.entropies),
            "midpoint entropy is not finite and positive")
    require(all(math.isfinite(value) for value in state.velocities),
            "midpoint velocity is not finite")
    require(all(value > 0 and math.isfinite(value) for value in state.densities),
            "midpoint density is not finite and positive")


def validate_midpoints(output, label, base_timestep, endpoint=None, uniform_rung=None,
                       global_intervals=None):
    """Verify actual positive H events, independently of signed gravity events.

    Derived dyadic midpoint lattice: t_H=(j+1/2) h_r, h_r=t_base/2^r.
    Adaptive changes may alter which rungs are occupied, but every executed
    exact-rung event must remain on this lattice and occur only once. Checkpoints
    contain positions at endpoint with any boundary-time H still pending.
    Global integration uses arbitrary durations and logs rung=-1 instead.
    """
    events = [(float(time), int(rung), float(duration))
              for time, rung, duration in MIDPOINT.findall(output)]
    require(events, f"{label}: no positive midpoint hydro events")
    previous = -math.inf
    seen = set()
    for time, rung, duration in events:
        require(math.isfinite(time) and math.isfinite(duration) and duration > 0 and time > 0,
                f"{label}: hydro received a nonpositive or nonfinite interval")
        require(time >= previous, f"{label}: hydro events ran backwards in time")
        require((time, rung) not in seen, f"{label}: hydro repeated at a gravity correction")
        previous = time
        seen.add((time, rung))
        if rung >= 0:
            expected = math.ldexp(base_timestep, -rung)
            require(math.isclose(duration, expected, rel_tol=2e-12),
                    f"{label}: midpoint duration differs from its target rung")
            phase = time / duration - 0.5
            require(abs(phase - round(phase)) <= 128 * sys.float_info.epsilon * max(1, abs(phase)),
                    f"{label}: hydro ran at a boundary instead of its midpoint")
        if endpoint is not None:
            require(time < endpoint, f"{label}: checkpoint advanced hydro at a pending boundary")
    if uniform_rung is not None:
        require(endpoint is not None and all(rung == uniform_rung for _, rung, _ in events),
                f"{label}: uniform case updated a different hydro rung")
        duration = math.ldexp(base_timestep, -uniform_rung)
        expected = math.ceil(endpoint / duration - 0.5)
        require(len(events) == expected,
                f"{label}: missed or duplicated a uniform-rung midpoint ({len(events)} != {expected})")
    if global_intervals is not None:
        require(len(events) == len(global_intervals),
                f"{label}: global integration missed or repeated a hydro midpoint")
        for (time, rung, duration), (start, end) in zip(events, global_intervals):
            tolerance = 128 * sys.float_info.epsilon * max(abs(end), abs(duration))
            require(rung == -1 and abs(duration - (end - start)) <= tolerance and
                    abs(time - (start + 0.5 * (end - start))) <= tolerance,
                    f"{label}: global hydro did not run once at its interval center")
    return {"count": len(events), "rungs": sorted({rung for _, rung, _ in events}),
            "first_time": events[0][0], "last_time": events[-1][0]}
