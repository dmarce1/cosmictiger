#!/usr/bin/env python3
"""Apply a diagnostic staging variant to cosmictiger/gravity_cuda.hpp (from the committed text)."""
import subprocess, sys, pathlib
W = pathlib.Path("/home/dmarce1/workspace/cosmictiger/.claude/worktrees/nifty-lederberg-7f7b6e")
F = W / "cosmictiger/gravity_cuda.hpp"
variant = sys.argv[1]
src = subprocess.check_output(["git", "-C", str(W), "show", "HEAD:cosmictiger/gravity_cuda.hpp"], text=True)

def must_replace(s, old, new, count=1):
    assert s.count(old) == count, (old, s.count(old))
    return s.replace(old, new)

copy_block = """        cuda::memcpy_async(group, control.sourceX + particleIndex, data.x + imin,
            sizeof(Fixed32) * sz, control.barrier);
        cuda::memcpy_async(group, control.sourceY + particleIndex, data.y + imin,
            sizeof(Fixed32) * sz, control.barrier);
        cuda::memcpy_async(group, control.sourceZ + particleIndex, data.z + imin,
            sizeof(Fixed32) * sz, control.barrier);
"""
launder = """    // Diagnostic variant: hide the barrier address from the compiler.
    auto* {name} = &control.barrier;
    asm volatile("" : "+l"({name}));
"""
head = "    int particleIndex = 0;\n    while (particleIndex < capacity && i < partsz) {\n"
wait = "    control.barrier.arrive_and_wait();\n    __syncwarp();\n    return particleIndex;\n"

if variant == "base":
    out = src
elif variant == "launder_copy":
    out = must_replace(src, head, launder.format(name="copyBarrier") + head)
    out = must_replace(out, copy_block, copy_block.replace("control.barrier", "*copyBarrier"))
elif variant == "launder_wait":
    out = must_replace(src, head, launder.format(name="waitBarrier") + head)
    out = must_replace(out, wait, wait.replace("control.barrier.arrive_and_wait()", "waitBarrier->arrive_and_wait()"))
elif variant == "launder_both":
    out = must_replace(src, head, launder.format(name="stagingBarrier") + head)
    out = must_replace(out, copy_block, copy_block.replace("control.barrier", "*stagingBarrier"))
    out = must_replace(out, wait, wait.replace("control.barrier.arrive_and_wait()", "stagingBarrier->arrive_and_wait()"))
elif variant == "block_sync":
    out = must_replace(src, wait, wait.replace("    __syncwarp();\n", "    group.sync();\n    __syncwarp();\n"))
elif variant == "sync_copy":
    sync_block = """        for (int j = tid; j < sz; j += WARP_SIZE) {
            control.sourceX[particleIndex + j] = data.x[imin + j];
            control.sourceY[particleIndex + j] = data.y[imin + j];
            control.sourceZ[particleIndex + j] = data.z[imin + j];
        }
"""
    out = must_replace(src, copy_block, sync_block)
    out = must_replace(out, wait, "    __syncwarp();\n    return particleIndex;\n")
else:
    sys.exit("unknown variant " + variant)
F.write_text(out)
print("applied", variant, "changed" if out != src else "unchanged")
