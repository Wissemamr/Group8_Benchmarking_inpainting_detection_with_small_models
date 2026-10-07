"""Per-pixel LPIPS map of a single VGG layer.

Adapted from AEROBLADE's `_PatchedLPIPS` (aeroblade/src/aeroblade/distances.py): same network
(lpips 0.1.4, VGG16, v0.1 linear heads), same input handling as its `normalize=True` call ([0, 1]
-> [-1, 1] -> LPIPS scaling layer), and the same per-layer map `lins[k]((f0 - f1) ** 2)` on
channel-normalised features, returned before any upsampling.

AEROBLADE's `lpips_vgg_2` distance is the spatial mean of this map for layer 2 (relu2_2, at
H/2 x W/2), negated; here the map itself is kept. Only the VGG slices up to the requested layer
are run: the output is identical and the VGG cost for layer 2 drops by about 3x.

With `pad > 0`, both images are reflect-padded by `pad` pixels and the map is cropped back.
VGG's zero padding otherwise pulls the error at the outermost ~5 pixels down to ~15% of the
interior level, which reads as "reconstructs well" at every image border. `pad = 0` is
AEROBLADE's exact computation.
"""

from __future__ import annotations

import warnings

import lpips
import numpy as np
import torch
import torch.nn.functional as F


class LPIPSLayerMap(torch.nn.Module):
    def __init__(self, layer: int = 2, pad: int = 0) -> None:
        super().__init__()
        if not 1 <= layer <= 5:
            raise ValueError("VGG has LPIPS layers 1..5")
        stride = 2 ** (layer - 1)
        if pad % stride:
            raise ValueError(f"pad must be a multiple of {stride} for layer {layer}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lpips = lpips.LPIPS(net="vgg", spatial=True, verbose=False)
        self.lpips.eval()
        self.layer = layer
        self.pad = pad
        self.crop = pad // stride
        self.slices = [getattr(self.lpips.net, f"slice{i}") for i in range(1, layer + 1)]

    @torch.inference_mode()
    def forward(self, img0: torch.Tensor, img1: torch.Tensor, pad: int | None = None) -> torch.Tensor:
        """(N, 1, h, w) map for (N, 3, H, W) images in [0, 1]; (h, w) = (H, W) / 2**(layer-1).
        `pad` overrides the constructor's reflect padding for this call (0 = AEROBLADE's exact map)."""
        pad = self.pad if pad is None else pad
        stride = 2 ** (self.layer - 1)
        if pad % stride:
            raise ValueError(f"pad must be a multiple of {stride} for layer {self.layer}")
        crop = pad // stride
        if pad:
            img0 = F.pad(img0, (pad,) * 4, mode="reflect")
            img1 = F.pad(img1, (pad,) * 4, mode="reflect")
        x0 = self.lpips.scaling_layer(2 * img0 - 1)
        x1 = self.lpips.scaling_layer(2 * img1 - 1)
        for block in self.slices:
            x0, x1 = block(x0), block(x1)
        diff = (lpips.normalize_tensor(x0) - lpips.normalize_tensor(x1)) ** 2
        out = self.lpips.lins[self.layer - 1](diff)
        if crop:
            out = out[..., crop:-crop, crop:-crop]
        return out


def upsample(layer_map: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear upsampling to image resolution, as `lpips.upsample` does."""
    t = torch.from_numpy(np.ascontiguousarray(layer_map, dtype=np.float32))[None, None]
    return F.interpolate(t, size=size_hw, mode="bilinear", align_corners=False)[0, 0].numpy()
