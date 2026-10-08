"""Qualitative overlays: image | GT mask | manipulation heatmap | Otsu prediction.

Colour roles are fixed across panels: ground truth is orange, the prediction blue, and the
heatmap a single-hue blue ramp (light = low score, dark = high score).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.ndimage import binary_dilation, binary_erosion  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
PRED_COLOR = "#2a78d6"
GT_COLOR = "#eb6834"
HEATMAP = LinearSegmentedColormap.from_list(
    "blue_sequential",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def _two_tone(mask: np.ndarray, color: str) -> np.ndarray:
    return np.where(mask[..., None], to_rgb(color), to_rgb(SURFACE))


def boundary_rgba(mask: np.ndarray, color: str) -> np.ndarray:
    width = max(1, round(max(mask.shape) / 300))
    edge = mask & ~binary_erosion(mask, iterations=width)
    edge = binary_dilation(edge, iterations=width)
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[edge] = (*to_rgb(color), 1.0)
    return rgba


def save_overlay(
    path: Path,
    image: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
    gt: Optional[np.ndarray],
    display_wh: tuple[int, int],
    title: str,
    score_label: str,
    pred_label: str,
) -> None:
    """All arrays share one (H, W) grid; `display_wh` restores the original aspect ratio."""
    w, h = display_wh
    extent = (0, w, h, 0)
    panel_w = 4.0
    fig, axes = plt.subplots(
        1, 4, figsize=(4 * panel_w, panel_w * h / w + 1.0), facecolor=SURFACE,
        constrained_layout=True,
    )
    for ax in axes:
        ax.set_axis_off()

    axes[0].imshow(image, extent=extent)
    axes[0].set_title("Image", color=INK, fontsize=10)

    if gt is not None:
        axes[1].imshow(_two_tone(gt, GT_COLOR), extent=extent)
        axes[1].set_title(f"GT mask ({gt.mean():.1%} inpainted)", color=INK, fontsize=10)
    else:
        axes[1].imshow(np.ones((h, w, 3)) * to_rgb(SURFACE), extent=extent)
        axes[1].text(0.5, 0.5, "GT mask not available\n(TGIF masks/ folder needed)",
                     ha="center", va="center", color=INK_MUTED, fontsize=9,
                     transform=axes[1].transAxes)
        axes[1].set_title("GT mask", color=INK, fontsize=10)

    im = axes[2].imshow(score, cmap=HEATMAP, vmin=0, vmax=1, extent=extent)
    if gt is not None:
        axes[2].imshow(boundary_rgba(gt, GT_COLOR), extent=extent)
    axes[2].set_title(score_label, color=INK, fontsize=10)
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    axes[3].imshow(_two_tone(pred, PRED_COLOR), extent=extent)
    handles = [Patch(color=PRED_COLOR, label="predicted (Otsu)")]
    if gt is not None:
        axes[3].imshow(boundary_rgba(gt, GT_COLOR), extent=extent)
        handles.append(Line2D([], [], color=GT_COLOR, lw=2, label="GT boundary"))
    axes[3].legend(handles=handles, loc="lower right", fontsize=7, frameon=True,
                   facecolor=SURFACE, edgecolor=SURFACE, labelcolor=INK_MUTED)
    axes[3].set_title(pred_label, color=INK, fontsize=10)

    fig.suptitle(title, color=INK_MUTED, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
