#!/usr/bin/env python3
"""Compare existing Nsight Compute exports without running CUDA workloads.

Source-page CSV repeats SASS addresses for inline source contexts. Deduplicate
addresses before attributing instructions or sampled stalls to device functions.
The resulting instruction total is checked against the raw SASS counter.
"""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_COLUMNS = {
    "warp_instructions": 7,
    "thread_instructions": 8,
    "predicated_on_thread_instructions": 9,
    "warp_samples": 4,
    "not_issued_warp_samples": 5,
    "theoretical_local_sectors": 24,
    "long_scoreboard_samples": 38,
    "short_scoreboard_samples": 46,
    "wait_samples": 49,
    "no_instruction_samples": 43,
    "barrier_samples": 32,
}


def number(value):
    return 0.0 if value in ("", "-") else float(value.replace(",", ""))


def open_export(name):
    path = ROOT / name
    if path.exists():
        return path.open()
    return gzip.open(path.with_suffix(path.suffix + ".gz"), "rt")


def load_variant(variant):
    with open_export(f"ncu_{variant}_raw.csv") as stream:
        rows = list(csv.reader(stream))
    raw = dict(zip(rows[0], rows[2]))
    units = dict(zip(rows[0], rows[1]))
    details = {}
    with open_export(f"ncu_{variant}_details.csv") as stream:
        for row in list(csv.reader(stream))[1:]:
            if len(row) > 14 and row[12]:
                details[f"{row[11]} / {row[12]}"] = {
                    "unit": row[13], "value": row[14]
                }

    addresses = {}
    duplicate_count = 0
    source_file = function = source_line = ""
    with open_export(f"ncu_{variant}_source.csv") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0] == "File Path":
                source_file = row[1]
            elif row[0] == "Function Name":
                function = row[1].split("(")[0]
            elif len(row) > max(SOURCE_COLUMNS.values()) and row[0] != "Line No":
                if row[0]:
                    source_line = row[0]
                if not row[2].startswith("0x"):
                    continue
                metrics = {name: number(row[index]) for name, index in SOURCE_COLUMNS.items()}
                context = {"file": source_file, "line": source_line}
                if row[2] in addresses:
                    duplicate_count += 1
                    existing = addresses[row[2]]
                    assert existing["metrics"] == metrics, (variant, row[2])
                    assert existing["function"] == function, (variant, row[2])
                    existing["contexts"].append(context)
                else:
                    addresses[row[2]] = {
                        "address": row[2], "function": function,
                        "assembly": row[3].strip(), "metrics": metrics,
                        "contexts": [context],
                    }

    function_totals = defaultdict(lambda: defaultdict(float))
    for address in addresses.values():
        for name, value in address["metrics"].items():
            function_totals[address["function"]][name] += value
    totals = {
        name: sum(item[name] for item in function_totals.values())
        for name in SOURCE_COLUMNS
    }
    assert totals["warp_instructions"] == number(raw["sass__inst_executed_per_opcode"])
    assert totals["predicated_on_thread_instructions"] == number(raw["thread_inst_executed_true"])
    return {
        "raw": raw, "units": units, "details": details,
        "functions": dict(function_totals), "source_totals": totals,
        "unique_addresses": len(addresses), "duplicate_rows_removed": duplicate_count,
        "highest_stall_addresses": sorted(
            addresses.values(), key=lambda item: item["metrics"]["warp_samples"], reverse=True
        )[:40],
    }


def delta(before, after):
    return {
        "before": before, "after": after, "difference": after - before,
        "percent_change": 100.0 * (after / before - 1.0) if before else None,
    }


def main():
    variants = {name: load_variant(name) for name in ("before", "after")}
    functions = {}
    for function in sorted(set(variants["before"]["functions"]) | set(variants["after"]["functions"])):
        functions[function] = {
            name: delta(
                variants["before"]["functions"].get(function, {}).get(name, 0.0),
                variants["after"]["functions"].get(function, {}).get(name, 0.0),
            ) for name in SOURCE_COLUMNS
        }
    raw_metrics = {}
    for name, before in variants["before"]["raw"].items():
        try:
            comparison = delta(float(before), float(variants["after"]["raw"][name]))
        except ValueError:
            continue
        raw_metrics[name] = {"unit": variants["before"]["units"][name], **comparison}

    result = {
        "scope": "One measured force-only adaptive kernel after one warmup; original comparison binaries",
        "caveats": [
            "Nsight Compute replays kernels and changes execution/cache/migration conditions; its duration is not the original application timing.",
            "PC sample counts are statistical observations, not exact elapsed-time attribution.",
            "SASS software-patched instruction counts and hardware instruction counters come from distinct passes and differ.",
            "Inline source contexts repeat addresses; duplicate addresses were removed and aggregate counts verified against raw counters.",
        ],
        "variants": variants, "function_comparison": functions, "raw_comparison": raw_metrics,
    }
    (ROOT / "ncu_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "ncu_function_comparison.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["function", "metric", "before", "after", "difference", "percent_change"])
        for function, metrics in functions.items():
            for name, values in metrics.items():
                writer.writerow([function, name, *values.values()])
    for function, metrics in sorted(functions.items(), key=lambda item: item[1]["warp_instructions"]["before"], reverse=True):
        instructions = metrics["warp_instructions"]
        samples = metrics["warp_samples"]
        print(f"{function:28} instructions {instructions['before']:12,.0f} -> {instructions['after']:12,.0f}; warp samples {samples['before']:8,.0f} -> {samples['after']:8,.0f}")


if __name__ == "__main__":
    main()
