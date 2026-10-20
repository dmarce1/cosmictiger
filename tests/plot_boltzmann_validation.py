#!/usr/bin/env python3
"""Plot the saved, independently normalized CAMB comparison."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmictiger_boltzmann_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, (name, label) in enumerate((("0", "0"), ("3_046", "3.046"))):
        for setting, style in (("default", "-"), ("refined", "--")):
            path = (args.results / f"final_neff_{name}_{setting}" /
                    "camb_comparison" / "differences.csv")
            with path.open() as source:
                rows = list(csv.DictReader(source))
            for redshift, color in ((49.0, "#bd5b19"), (0.0, "#1764a0")):
                selected = sorted((row for row in rows if float(row["z"]) == redshift),
                                  key=lambda row: float(row["k_mpc"]))
                ks = [float(row["k_mpc"]) for row in selected]
                for index, field in enumerate(("delta_cb", "p_cb_mpc3")):
                    values = [100 * float(row[f"{field}_relative_error"]) for row in selected]
                    axes[index, column].semilogx(
                        ks, values, style, color=color, linewidth=1.7,
                        label=f"z = {redshift:g}, {setting}")
        axes[0, column].set_title(rf"$N_{{\rm eff}}={label}$")
        axes[1, column].set_xlabel(r"$k$ [Mpc$^{-1}$]")
        for row in range(2):
            axes[row, column].axhline(0, color="0.3", linewidth=0.65)
            axes[row, column].grid(alpha=0.18)
    axes[0, 0].set_ylabel(r"Combined density difference [%]")
    axes[1, 0].set_ylabel(r"Combined power difference [%]")
    axes[0, 1].legend(fontsize=9, loc="best")
    figure.suptitle("CosmicTiger vs CAMB 1.6.5", fontsize=16)
    figure.text(
        0.5, 0.015,
        r"Default: $L=64$, rtol $=10^{-6}$; refined: $L=128$, rtol $=10^{-7}$."
        "\nFlat, massless-neutrino models; matched primordial amplitude; no fit.",
        ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.065, 1, 0.95))
    figure.savefig(args.results / "camb_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
