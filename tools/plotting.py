# tools\visualization_utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_pixel_error_per_level(levels, mean_err, std_err):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        levels,
        mean_err,
        yerr=std_err,
        fmt="o",
        capsize=6,
        elinewidth=2,
        markeredgewidth=2
    )

    ax.set_xlabel("Lumbar Level")
    ax.set_ylabel("Pixel Error (px)")
    ax.set_title("Mean ± Std Pixel Error per Lumbar Level")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    return fig
