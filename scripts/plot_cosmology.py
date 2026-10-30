# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Render cosmology.txt; also embedded in the simulation executable by CMake."""

import argparse
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[variable] = "1"

    # Compute nodes need neither a display nor a writable home/cache directory.
    with tempfile.TemporaryDirectory(prefix="cosmictiger_plot_") as config_directory:
        os.environ["MPLCONFIGDIR"] = config_directory
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        data = np.loadtxt(args.samples, ndmin=2)
        if (data.shape[1] != 3 or data.shape[0] < 2 or not np.isfinite(data).all()
                or (data[:, 1] <= 0).any() or (np.diff(data[:, 0]) < 0).any()):
            raise ValueError("Expected finite, ordered tau/a/adot background samples")
        tau, scale, rate = data.T
        with plt.rc_context({"font.size": 12, "axes.spines.top": False,
                             "axes.spines.right": False}):
            figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7.5),
                                       layout="constrained")
            figure.suptitle("Cosmological background")
            axes[0].plot(tau, scale, color="#2166ac", linewidth=2)
            axes[0].set_ylabel(r"Scale factor $a$")
            axes[1].axhline(0, color="#555555", linewidth=0.8)
            axes[1].plot(tau, rate, color="#b35806", linewidth=2)
            axes[1].set_ylabel(r"$\dot{a} = da/dt$ [code time$^{-1}$]")
            axes[1].set_xlabel(r"Elapsed conformal time $\tau$ [code time]")
            for axis in axes:
                axis.grid(alpha=0.2)
                axis.ticklabel_format(style="sci", axis="both", scilimits=(-3, 4),
                                      useMathText=True, useOffset=False)
            if tau[-1] > tau[0]:
                axes[1].set_xlim(tau[0], tau[-1])
            # Publish only a complete image; an interrupted renderer leaves no PNG.
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".png",
                                                 delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                figure.savefig(temporary_path, format="png", dpi=120)
                os.replace(temporary_path, args.output)
            finally:
                plt.close(figure)
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
