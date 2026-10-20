#!/usr/bin/env python3
"""Plot saved native recombination histories and matter-transfer validation.

No simulation or reference calculation is run. The raw fractional-redshift
CAMB temperature interpolation artifact remains visible in the lower-right
inset; the main thermal comparison uses CAMB's integer-redshift table nodes.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmictiger_native_recombination_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

COLORS = {"0": "#0072B2", "3_046": "#D55E00"}
SETTINGS = (("default", "-"), ("refined", "--"))


def readRows(path):
    with path.open() as source:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(source)]
    if not rows or any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError(f"Empty or nonfinite comparison data: {path}")
    return rows


def matchedRows(candidate, reference, field, integerOnly=False, maxRedshift=10000):
    index = {f"{row['z']:.10g}": row for row in candidate}
    result = []
    for row in reference:
        redshift = row["z"]
        if redshift > maxRedshift:
            continue
        if integerOnly and not math.isclose(redshift, round(redshift), abs_tol=1e-8, rel_tol=0):
            continue
        match = index.get(f"{redshift:.10g}")
        if match is None or row[field] == 0:
            raise ValueError(f"Missing or zero reference at z={redshift:g}")
        result.append((redshift, 100 * (match[field] / row[field] - 1)))
    return sorted(result)


def configureAxes(axis, logarithmicY=False):
    axis.set_xscale("log")
    if logarithmicY:
        axis.set_yscale("log")
    axis.grid(alpha=0.18, linewidth=0.7)
    axis.set_xlim(1, 10001)
    axis.tick_params(which="both", direction="out")


def saveFigure(figure, directory, name):
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        figure.savefig(path, dpi=200, facecolor="white")
        print(path)
    plt.close(figure)


def plotHistories(results):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.1), sharex="col",
                                gridspec_kw={"height_ratios": [1.1, 1]})
    artifact = axes[1, 1].inset_axes([0.40, 0.16, 0.56, 0.43])
    maxima = {"x_e": {"CAMB": 0.0, "RECFAST": 0.0},
              "T_m": {"CAMB": 0.0, "RECFAST": 0.0}}
    for name, color in COLORS.items():
        camb = readRows(results / "camb" / f"neff_{name}.csv")
        recfast = readRows(results / "recfast" / f"neff_{name}" / "reference.csv")
        for setting, style in SETTINGS:
            native = readRows(results / "final_history" / f"neff_{name}_{setting}" / "history.csv")
            for column, field in enumerate(("x_e", "T_m")):
                selected = [row for row in native if row["z"] <= 10000]
                axes[0, column].plot(
                    [1 + row["z"] for row in selected], [row[field] for row in selected],
                    color=color, linestyle=style, linewidth=1.7 if setting == "default" else 0.9,
                    zorder=3)
                difference = matchedRows(native, camb, field, integerOnly=(field == "T_m"))
                axes[1, column].plot(
                    [1 + row[0] for row in difference], [row[1] for row in difference],
                    color=color, linestyle=style, linewidth=1.5, zorder=3)
                maxima[field]["CAMB"] = max(maxima[field]["CAMB"],
                                             max(abs(row[1]) for row in difference))
                if setting == "default":
                    recfastDifference = matchedRows(native, recfast, field)
                    axes[1, column].plot(
                        [1 + row[0] for row in recfastDifference],
                        [row[1] for row in recfastDifference], color=color, linestyle=":",
                        linewidth=1.1, alpha=0.85)
                    maxima[field]["RECFAST"] = max(maxima[field]["RECFAST"],
                                                   max(abs(row[1]) for row in recfastDifference))
                    for reference, marker, stride in ((camb, "x", 55), (recfast, "o", 30)):
                        sample = sorted((row for row in reference if row["z"] <= 10000),
                                        key=lambda row: row["z"])[::stride]
                        axes[0, column].plot(
                            [1 + row["z"] for row in sample], [row[field] for row in sample],
                            linestyle="none", marker=marker, markersize=3.6, color=color,
                            markerfacecolor="none", markeredgewidth=0.8, alpha=0.75, zorder=4)
            rawTemperature = matchedRows(native, camb, "T_m", maxRedshift=2)
            artifact.plot([row[0] for row in rawTemperature], [row[1] for row in rawTemperature],
                          color=color, linestyle=style, linewidth=1.15)

    for row in range(2):
        for column in range(2):
            configureAxes(axes[row, column], logarithmicY=(row == 0))
    for axis in axes[1]:
        axis.axhline(0, color="0.35", linewidth=0.65)
        axis.set_xlabel(r"$1+z$")
    axes[0, 0].set_title("Free electrons", loc="left", weight="bold")
    axes[0, 1].set_title("Gas temperature", loc="left", weight="bold")
    axes[0, 0].set_ylabel(r"$x_e=n_e/n_{\rm H}$")
    axes[0, 1].set_ylabel(r"$T_{\rm gas}$ [K]")
    axes[1, 0].set_ylabel(r"$x_e$ difference [%]")
    axes[1, 1].set_ylabel(r"$T_{\rm gas}$ difference [%]")
    axes[1, 0].set_title("Matched history samples", loc="left", fontsize=10)
    axes[1, 1].set_title("CAMB integer-z nodes; RECFAST output nodes", loc="left", fontsize=10)
    axes[1, 1].set_ylim(-0.014, 0.007)
    axes[1, 0].text(0.03, 0.96,
        f"Max |difference|\nCAMB: {maxima['x_e']['CAMB']:.3f}%\n"
        f"RECFAST: {maxima['x_e']['RECFAST']:.3f}%",
        transform=axes[1, 0].transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    axes[1, 1].text(0.03, 0.96,
        f"Max |difference|\nCAMB: {maxima['T_m']['CAMB']:.4f}%\n"
        f"RECFAST: {maxima['T_m']['RECFAST']:.4f}%",
        transform=axes[1, 1].transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    artifact.set_title("Raw CAMB interpolation at low z", fontsize=8.5, pad=4)
    artifact.set_xlabel("z", fontsize=8, labelpad=0)
    artifact.set_ylabel("Difference [%]", fontsize=8, labelpad=1)
    artifact.set_xlim(0, 2)
    artifact.set_ylim(-6, 1)
    artifact.set_xticks([0, 1, 2])
    artifact.set_yticks([-5, 0])
    artifact.tick_params(labelsize=8)
    artifact.grid(alpha=0.18)
    artifact.axhline(0, color="0.5", linewidth=0.5)
    artifact.set_facecolor("#fbfbfb")

    colorLegend = [Line2D([], [], color=color, linewidth=2, label=rf"$N_{{\rm eff}}={name.replace('_', '.')}$")
                   for name, color in COLORS.items()]
    axes[0, 0].legend(handles=colorLegend, loc="upper left", fontsize=10, frameon=False)
    axes[0, 1].legend(handles=[
        Line2D([], [], color="0.2", linewidth=1.7, label="Native default"),
        Line2D([], [], color="0.2", linewidth=1.0, linestyle="--", label="Native refined"),
        Line2D([], [], color="0.2", linestyle="none", marker="x", markersize=4, label="CAMB 1.6.5"),
        Line2D([], [], color="0.2", linestyle="none", marker="o", markerfacecolor="none", markersize=4,
               label="RECFAST 1.5.2"),
    ], loc="upper left", fontsize=9, frameon=False)
    axes[1, 0].legend(handles=[
        Line2D([], [], color="0.25", linestyle="-", label="Default / CAMB − 1"),
        Line2D([], [], color="0.25", linestyle="--", label="Refined / CAMB − 1"),
        Line2D([], [], color="0.25", linestyle=":", label="Default / RECFAST − 1"),
    ], loc="lower left", fontsize=8.5, frameon=False)
    figure.suptitle("Native primordial recombination against independent histories", fontsize=15)
    figure.text(0.5, 0.025,
        r"Hswitch=1, Heswitch=6; no reionization. Native default: rtol $10^{-7}$, "
        r"max $\Delta\ln a=0.01$; refined: $10^{-8}$, 0.005."
        "\nTemperature inset retains CAMB's fractional-z spline artifact; it is not a native integration error.",
        ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0, 0.085, 1, 0.95), h_pad=1.7, w_pad=2.2)
    saveFigure(figure, results, "native_history_comparison")


def plotTransfers(results):
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), sharex=True)
    largest = [0.0, 0.0]
    for column, name in enumerate(COLORS):
        for setting, style in SETTINGS:
            folder = results / "final_transfers" / f"neff_{name}_{setting}"
            metadata = json.loads((folder / "run_metadata.json").read_text())
            if not metadata.get("complete"):
                raise ValueError(f"Transfer campaign not complete: {folder}")
            rows = readRows(folder / "camb_comparison" / "differences.csv")
            for redshift, color in ((49.0, "#D55E00"), (0.0, "#0072B2")):
                selected = sorted((row for row in rows if row["z"] == redshift),
                                  key=lambda row: row["k_mpc"])
                if len(selected) != 31:
                    raise ValueError(f"Expected 31 k modes: {folder}, z={redshift:g}")
                for row, field in enumerate(("delta_cb", "p_cb_mpc3")):
                    differences = [100 * entry[f"{field}_relative_error"] for entry in selected]
                    axes[row, column].semilogx(
                        [entry["k_mpc"] for entry in selected], differences,
                        color=color, linestyle=style, linewidth=1.7,
                        marker="o" if setting == "default" else None, markersize=2.4,
                        label=f"z = {redshift:g}, {setting}")
                    largest[row] = max(largest[row], max(abs(value) for value in differences))
        axes[0, column].set_title(rf"$N_{{\rm eff}}={name.replace('_', '.')}$", fontsize=13)
        axes[1, column].set_xlabel(r"$k$ [Mpc$^{-1}$]")
        for row in range(2):
            axes[row, column].axhline(0, color="0.35", linewidth=0.65)
            axes[row, column].grid(alpha=0.18)
            axes[row, column].set_xlim(1e-4, 10)
            axes[row, column].yaxis.set_major_formatter(ScalarFormatter())
    axes[0, 0].set_ylabel(r"$\delta_{cb}$ difference [%]")
    axes[1, 0].set_ylabel(r"$P_{cb}$ difference [%]")
    axes[0, 1].legend(fontsize=9, loc="best", frameon=False)
    for row in range(2):
        lower = min(axis.get_ylim()[0] for axis in axes[row])
        upper = max(axis.get_ylim()[1] for axis in axes[row])
        for axis in axes[row]:
            axis.set_ylim(lower, upper)
    figure.suptitle("Native recombination: matter transfer and power vs CAMB 1.6.5", fontsize=15)
    figure.text(0.5, 0.025,
        r"Difference $=100\,(\mathrm{native}/\mathrm{CAMB}-1)$. Matched primordial amplitude; no fit."
        "\n"
        r"Boltzmann default: $L=64$, rtol $10^{-6}$; refined: $L=128$, rtol $10^{-7}$. "
        "Native recombination fixed at its default settings.", ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0, 0.085, 1, 0.95), h_pad=1.5, w_pad=2)
    saveFigure(figure, results, "native_matter_comparison")
    print(f"Max matter-transfer difference: {largest[0]:.6g}%; power: {largest[1]:.6g}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--history-only", action="store_true")
    arguments = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    plotHistories(arguments.results)
    if not arguments.history_only:
        plotTransfers(arguments.results)


if __name__ == "__main__":
    main()
