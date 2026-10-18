#!/usr/bin/env python3
"""Prepare NVTX host-range comparison; compile/link only with --build.

All source changes are confined to benchmark copies. Reuses the exact original
before/after CUDA objects and synchronized adaptive harness. No binary is run.
"""

import argparse
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Source anchor is not unique: " + repr(old))
    return text.replace(old, new, 1)


def instrument_kick(text):
    text = replace_once(text, "#include <cosmictiger/timer.hpp>\n",
                        '#include <cosmictiger/timer.hpp>\n#include "profile_ranges.hpp"\n')
    text = replace_once(text,
        "    std::shared_ptr<KickWorkspace> cudaWorkspace) {\n    FlopCounter<int> flops = 0;",
        "    std::shared_ptr<KickWorkspace> cudaWorkspace) {\n"
        '    detail::HostProfileRange rootProfile("kick.root", self.proc == 0 && self.index == 0);\n'
        "    FlopCounter<int> flops = 0;")
    for checklist, label, kind in [("echecklist", "cpu.walk.ewald", "Ewald"),
                                   ("dchecklist", "cpu.walk.direct", "Direct")]:
        text = replace_once(text, f"        auto& checklist = {checklist};\n        do {{",
            f"        auto& checklist = {checklist};\n"
            f'        detail::HostProfileRange checklistProfile("{label}", true, checklist.size());\n        do {{')
        text = replace_once(text,
            "        } while (checklist.size() && selfPtr->leaf);\n"
            f"        cpuGravityCc(kGravity{kind}, l, cclist, self, params.doPhi);",
            "        } while (checklist.size() && selfPtr->leaf);\n"
            "        checklistProfile.finish();\n"
            f"        cpuGravityCc(kGravity{kind}, l, cclist, self, params.doPhi);")
    calls = [
        ("cpuGravityCc(kGravityEwald, l, cclist, self, params.doPhi);",
         '"cpu.gravity.cc.ewald"', "cclist"),
        ("cpuGravityCc(kGravityDirect, l, cclist, self, params.doPhi);",
         '"cpu.gravity.cc.direct"', "cclist"),
        ("cpuGravityCp(l, cplist, self, params.doPhi);",
         'selfPtr->leaf ? "cpu.gravity.cp.leaf" : "cpu.gravity.cp.internal"', "cplist"),
        ("cpuGravityPc(forces, params.doPhi, self, pclist);",
         '"cpu.gravity.pc"', "pclist"),
        ("cpuGravityPp(forces, params.doPhi, self, leaflist, params.gravitySupportRadius);",
         '"cpu.gravity.pp"', "leaflist"),
    ]
    for call, label, collection in calls:
        expression = re.compile(r"^( +)" + re.escape(call) + "$", re.MULTILINE)
        matches = list(expression.finditer(text))
        if len(matches) != 1:
            raise RuntimeError("Gravity call anchor is not unique: " + call)
        indent = matches[0].group(1)
        replacement = (indent + "{\n" + indent + "    detail::HostProfileRange gravityProfile(" +
                       label + ", " + collection + ".size() != 0, " + collection + ".size());\n" +
                       indent + "    " + call + "\n" + indent + "}")
        text = expression.sub(lambda unused: replacement, text, count=1)
    return text


def instrument_workspace(text):
    text = replace_once(text, "#include <cosmictiger/timer.hpp>\n",
                        '#include <cosmictiger/timer.hpp>\n#include "profile_ranges.hpp"\n')
    text = replace_once(text, "void KickWorkspace::toGpu() {\n",
        'void KickWorkspace::toGpu() {\n    detail::HostProfileRange preparationProfile("gpu.prepare", true, workitems_.size());\n')
    text = replace_once(text, "    tree2Gpu();\n    particlesMemadviseGpu();\n",
        '    preparationProfile.finish();\n'
        '    detail::HostProfileRange adviceProfile("gpu.advice_to_gpu");\n'
        '    tree2Gpu();\n    particlesMemadviseGpu();\n'
        '    adviceProfile.finish();\n'
        '    detail::HostProfileRange executeProfile("gpu.execute", true, workitems_.size());\n')
    text = replace_once(text,
        "        &particlesPos(ZDIM, 0), treeNodes, std::move(workitems_));\n    tree2Cpu();",
        "        &particlesPos(ZDIM, 0), treeNodes, std::move(workitems_));\n"
        "    executeProfile.finish();\n"
        '    detail::HostProfileRange restoreProfile("gpu.restore");\n    tree2Cpu();')
    text = replace_once(text, "        std::lock_guard<MutexType> lock(mutex_);\n"
        "        for (auto const& id : item.dchecklist) {",
        "        std::lock_guard<MutexType> lock(mutex_);\n"
        '        detail::HostProfileRange insertionProfile("gpu.add_work.set_insert", true,\n'
        '            item.dchecklist.size() + item.echecklist.size());\n'
        "        for (auto const& id : item.dchecklist) {")
    text = replace_once(text,
        "        // Publish the work before another producer can observe completion.\n",
        "        insertionProfile.finish();\n"
        "        // Publish the work before another producer can observe completion.\n")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    work = Path(__file__).resolve().parent
    comparison = work.parent.parent
    repo = comparison.parent.parent
    adaptive = comparison / "adaptive_suite"
    parent_path = adaptive / "build_manifest.json"
    parent = json.loads(parent_path.read_text())
    if not parent["completed"]:
        raise RuntimeError("Adaptive comparison manifest is incomplete")
    original_path = Path(parent["parent_manifest"])
    if digest(original_path) != parent["parent_manifest_sha256"]:
        raise RuntimeError("Original manifest was modified")
    original = json.loads(original_path.read_text())
    build = Path(parent["build_directory"])
    baseline = comparison / "baseline_source"
    checked_inputs = {str(parent_path): digest(parent_path),
                      str(original_path): digest(original_path)}
    for path, expected in parent["checked_input_sha256"].items():
        if digest(path) != expected:
            raise RuntimeError("Parent comparison input changed: " + path)
        checked_inputs[path] = expected
    harness = adaptive / "stf_walk.cpp.o"
    if digest(harness) != parent["harness_object_sha256"]:
        raise RuntimeError("Common adaptive harness object changed")
    checked_inputs[str(harness)] = digest(harness)
    workspace_source = repo / "src/kick_workspace.cpp"
    if workspace_source.read_bytes() != (baseline / "src/kick_workspace.cpp").read_bytes():
        raise RuntimeError("Workspace source differs from the baseline")
    checked_inputs[str(workspace_source)] = digest(workspace_source)
    checked_inputs[str(baseline / "src/kick_workspace.cpp")] = digest(baseline / "src/kick_workspace.cpp")
    helper = work / "profile_ranges.hpp"
    checked_inputs[str(helper)] = digest(helper)
    nvtx = Path("/usr/local/cuda/include/nvtx3/nvToolsExt.h")
    checked_inputs[str(nvtx)] = digest(nvtx)

    specs = [("kick_before.cpp", baseline / "src/kick.cpp", instrument_kick),
             ("kick_after.cpp", repo / "src/kick.cpp", instrument_kick),
             ("kick_workspace.cpp", workspace_source, instrument_workspace)]
    patch = []
    for name, source, transform in specs:
        old = source.read_text()
        new = transform(old)
        target = work / name
        if target.exists() and target.read_text() != new:
            raise RuntimeError("Prepared source was edited; preserve it first: " + str(target))
        if not target.exists():
            target.write_text(new)
        checked_inputs[str(target)] = digest(target)
        patch.extend(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                     fromfile=str(source), tofile="host_ranges/" + name))
    (work / "source_changes.patch").write_text("".join(patch))

    database_path = build / "compile_commands.json"
    checked_inputs[str(database_path)] = digest(database_path)
    entries = json.loads(database_path.read_text())
    commands = []
    for name, source, unused in specs:
        if name == "kick_before.cpp":
            template = next(command for command in original["baseline_compile_commands"]
                            if command[-1] == str(source))
        else:
            entry = next(row for row in entries if row["file"] == str(source))
            template = shlex.split(entry["command"])
        command = [str(work / name) if part == str(source) else part for part in template]
        target = work / (name + ".o")
        for flag in ["-o", "-MF", "-MT"]:
            if flag in command:
                command[command.index(flag) + 1] = str(target) + (".d" if flag == "-MF" else "")
        commands.append({"name": "compile_" + name.replace(".", "_"),
                         "arguments": command, "cwd": str(build)})
    binaries = {}
    reused_objects = {}
    for variant in ["before", "after"]:
        kick = (comparison / "baseline_objects/src/kick.cpp.o" if variant == "before"
                else Path("CMakeFiles/cosmictiger.dir/src/kick.cpp.o"))
        mapping = {str(kick): str(work / ("kick_" + variant + ".cpp.o")),
                   "CMakeFiles/cosmictiger.dir/src/kick_workspace.cpp.o": str(work / "kick_workspace.cpp.o"),
                   str(adaptive / (variant + "_device_link.o")): str(work / (variant + "_device_link.o"))}
        binary = work / ("cosmictiger_" + variant)
        binaries[variant] = {"path": str(binary)}
        for kind in ["device_link", "link"]:
            row = next(row for row in parent["commands"] if row["name"] == variant + "_" + kind)
            command = [mapping.get(part, part) for part in row["arguments"]]
            if kind == "link":
                command[command.index("-o") + 1] = str(binary)
            for part in row["arguments"]:
                if not part.endswith(".o") or part in mapping:
                    continue
                path = Path(part)
                if not path.is_absolute():
                    path = Path(row["cwd"]) / path
                expected = checked_inputs.get(str(path))
                if expected is None:
                    raise RuntimeError("Unverified link object: " + str(path))
                reused_objects[str(path)] = expected
            commands.append({"name": variant + "_" + kind,
                             "arguments": command, "cwd": row["cwd"]})
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parent_manifest": str(parent_path), "parent_manifest_sha256": digest(parent_path),
        "baseline_commit": parent["baseline_commit"],
        "build_directory": str(build),
        "purpose": "NVTX global asynchronous host ranges to separate CPU traversal/gravity, GPU work preparation, transfer advice, GPU execute, restore, and checklist set insertion.",
        "build_method": "Compile instrumented before/after kick.cpp and one common kick_workspace.cpp; reuse original version-specific CUDA objects and common adaptive harness; regenerate device links and executables.",
        "range_semantics": "NVTX rangeStartA or rangeStartEx plus rangeEnd supports HPX OS-thread migration; CPU walk/gravity ranges may overlap across workers and must not be summed as elapsed wall time. Root range includes its wait. GPU execute includes CUDA host setup, kernel, copies, and synchronization.",
        "payload_semantics": "Unsigned 64-bit payload: cpu.walk.* = initial checklist node count, cpu.gravity.* = accepted list node count, gpu.prepare/gpu.execute = workitem count, gpu.add_work.set_insert = input direct plus Ewald checklist node count. Other labels have no payload.",
        "checked_input_sha256": checked_inputs,
        "reused_object_sha256": reused_objects,
        "commands": commands, "binaries": binaries,
        "completed": False, "executed": [],
    }
    save(work / "build_plan.json", manifest)
    if not args.build:
        (work / "build_plan.txt").write_text("\n".join(
            row["name"] + ": " + shlex.join(row["arguments"]) for row in commands) + "\n")
        print("Prepared verified host NVTX source copies; no compilation or runtime performed.")
        return
    manifest_path = work / "build_manifest.json"
    if manifest_path.exists() or any(Path(row["path"]).exists() for row in binaries.values()):
        raise RuntimeError("Build outputs already exist; preserve them before rebuilding")
    save(manifest_path, manifest)
    for row in commands:
        log_path = work / (row["name"] + ".build.log")
        with log_path.open("w") as log:
            result = subprocess.run(row["arguments"], cwd=row["cwd"], stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        manifest["executed"].append({"name": row["name"], "exit_code": result.returncode})
        save(manifest_path, manifest)
        if result.returncode:
            raise RuntimeError("Build failed; see " + str(log_path))
        print("Completed " + row["name"], flush=True)
    for path, expected in checked_inputs.items():
        if digest(path) != expected:
            raise RuntimeError("Build input changed: " + path)
    for variant, row in binaries.items():
        row["sha256"] = digest(row["path"])
        row["cuda_object_sha256"] = parent["replacement_objects"]["CMakeFiles/cosmictiger.dir/src/kick.cu.o"][variant]
    manifest["built_object_sha256"] = {str(path): digest(path) for path in sorted(work.glob("*.o"))}
    manifest["completed"] = True
    save(manifest_path, manifest)
    print("Host NVTX comparison built; no runtime execution performed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
